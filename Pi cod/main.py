import argparse
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gc
import os


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        return F.leaky_relu(self.conv(x) + self.shortcut(x), 0.2)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.enc1 = ResidualBlock(in_channels, 32)
        self.enc2 = ResidualBlock(32, 64)
        self.enc3 = ResidualBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ResidualBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ResidualBlock(64, 32)
        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        t2 = self.up2(s3)
        t2 = torch.cat([t2, s2], dim=1)
        t2 = self.dec2(t2)
        t1 = self.up1(t2)
        t1 = torch.cat([t1, s1], dim=1)
        t1 = self.dec1(t1)
        return self.final(t1)


def preprocess(data, p_low, p_high):
    data = np.clip(data, p_low, p_high)
    data = (data - p_low) / (p_high - p_low + 1e-8)
    return data.astype(np.float32)


def postprocess(data, p_low, p_high):
    data = np.clip(data, 0, 1)
    data = data * (p_high - p_low) + p_low
    return data.astype(np.float32)


class CalciumDataset(Dataset):
    def __init__(self, filepath, p_low, p_high, indices, mode='train', window_size=5, patch_size=128):
        self.filepath = filepath
        self.indices = indices
        self.p_low = p_low
        self.p_high = p_high
        self.mode = mode
        self.window_size = window_size
        self.offset = window_size // 2
        self.patch_size = patch_size
        self._tif = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self._tif is None:
            self._tif = tifffile.TiffFile(self.filepath)

        start_idx = self.indices[idx]
        pages = self._tif.pages[start_idx: start_idx + self.window_size]
        window_np = np.stack([p.asarray() for p in pages]).astype(np.float32)
        window = torch.from_numpy(preprocess(window_np, self.p_low, self.p_high))

        if self.mode == 'train':
            c, h, w = window.shape
            y = np.random.randint(0, h - self.patch_size)
            x = np.random.randint(0, w - self.patch_size)
            window = window[:, y:y + self.patch_size, x:x + self.patch_size]

            if np.random.rand() > 0.5: window = torch.flip(window, [1])
            if np.random.rand() > 0.5: window = torch.flip(window, [2])

            target = window[self.offset].clone().unsqueeze(0)
            input_stack = window.clone()
            mask = torch.zeros_like(target)

            prob_mask = torch.rand(target.shape)
            mask_indices = prob_mask < 0.015

            if mask_indices.any():
                mask[mask_indices] = 1.0
                shift = np.random.choice([-1, 1])
                input_stack[self.offset][mask_indices.squeeze(0)] = torch.roll(target, shifts=shift, dims=2)[
                    mask_indices]

            return input_stack, target, mask
        else:
            return window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='denoised.tif')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--load_weights', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with tifffile.TiffFile(args.input) as tif:
        num_frames = len(tif.pages)
        sample_indices = np.linspace(0, num_frames - 1, 10, dtype=int)
        samples = np.stack([tif.pages[int(i)].asarray() for i in sample_indices])
        p_low, p_high = np.percentile(samples, (1, 99.9))
        del samples
        gc.collect()

    all_indices = np.arange(num_frames - args.window + 1)
    train_indices = all_indices[::2]

    model = UNet(in_channels=args.window, out_channels=1).to(device)

    if args.load_weights and os.path.exists(args.load_weights):
        model.load_state_dict(torch.load(args.load_weights, map_location=device))
    else:
        train_ds = CalciumDataset(args.input, p_low, p_high, train_indices, mode='train', window_size=args.window)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for b_in, b_tar, b_mask in train_loader:
                b_in, b_tar, b_mask = b_in.to(device), b_tar.to(device), b_mask.to(device)
                optimizer.zero_grad()
                out = model(b_in)
                loss = F.huber_loss(out * b_mask, b_tar * b_mask, delta=0.1, reduction='sum') / (b_mask.sum() + 1e-8)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                del b_in, b_tar, b_mask, out
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.6f}")
            torch.cuda.empty_cache()
            gc.collect()
        torch.save(model.state_dict(), "denoise_model.pth")

    model.eval()
    test_ds = CalciumDataset(args.input, p_low, p_high, all_indices, mode='test', window_size=args.window)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad(), tifffile.TiffWriter(args.output, bigtiff=True) as tif_out:
        for window in test_loader:
            window = window.to(device)
            _, _, h, w = window.shape
            ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
            if ph > 0 or pw > 0:
                window = F.pad(window, (0, pw, 0, ph), mode='reflect')

            output = model(window)
            if ph > 0 or pw > 0:
                output = output[:, :, :h, :w]

            res = postprocess(output.squeeze().cpu().numpy(), p_low, p_high)
            tif_out.write(res, contiguous=True)
            del window, output
            gc.collect()


if __name__ == "__main__":
    main()