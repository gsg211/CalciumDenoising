import argparse
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gc


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x):
        return F.leaky_relu(self.conv(x) + self.shortcut(x), 0.1)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.enc1 = ResidualBlock(in_channels, 32)
        self.enc2 = ResidualBlock(32, 64)
        self.enc3 = ResidualBlock(64, 128)
        self.enc4 = ResidualBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ResidualBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ResidualBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ResidualBlock(64, 32)
        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        t3 = self.up3(s4)
        t3 = torch.cat([t3, s3], dim=1)
        t3 = self.dec3(t3)
        t2 = self.up2(t3)
        t2 = torch.cat([t2, s2], dim=1)
        t2 = self.dec2(t2)
        t1 = self.up1(t2)
        t1 = torch.cat([t1, s1], dim=1)
        t1 = self.dec1(t1)
        return self.final(t1)

class DataSet(Dataset):
    def __init__(self, tiff_path, indices, p_low, p_high, mode='train', window=4, patch_size=96):
        self.tiff_path = tiff_path
        self.indices = indices
        self.p_low = np.float32(p_low)
        self.p_high = np.float32(p_high)
        self.mode = mode
        self.window = window
        self.patch_size = patch_size
        self.data = tifffile.memmap(tiff_path)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        half = self.window // 2

        ctx_indices = []
        for i in range(-half, half + 1):
            if i == 0: continue
            idx_to_add = max(0, min(self.data.shape[0] - 1, t + i))
            ctx_indices.append(idx_to_add)

        context = self.data[ctx_indices].astype(np.float32)
        target = self.data[t:t + 1].astype(np.float32)

        context = np.clip((context - self.p_low) / (self.p_high - self.p_low + 1e-8), 0, 1)
        target = np.clip((target - self.p_low) / (self.p_high - self.p_low + 1e-8), 0, 1)

        if self.mode == 'train':
            h, w = context.shape[1:]
            y = np.random.randint(0, h - self.patch_size)
            x = np.random.randint(0, w - self.patch_size)
            context = context[:, y:y + self.patch_size, x:x + self.patch_size]
            target = target[:, y:y + self.patch_size, x:x + self.patch_size]

            if np.random.rand() > 0.5:
                context, target = context[:, ::-1, :].copy(), target[:, ::-1, :].copy()
            if np.random.rand() > 0.5:
                context, target = context[:, :, ::-1].copy(), target[:, :, ::-1].copy()

        return torch.from_numpy(context).float(), torch.from_numpy(target).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='denoised_output.tif')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--window', type=int, default=4, help="Number of context frames (must be even)")
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()


    if args.window % 2 != 0:
        args.window += 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    with tifffile.TiffFile(args.input) as tif:
        num_frames = len(tif.pages)
        sample = tif.pages[num_frames // 2].asarray()
        p_low, p_high = np.percentile(sample, (0.5, 99.5))
        del sample


    model = UNet(in_channels=args.window, out_channels=1).to(device).float()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    criterion = nn.L1Loss()
    scaler = torch.amp.GradScaler('cuda')

    train_indices = np.arange(args.window, num_frames - args.window)[::2]
    train_ds = DataSet(args.input, train_indices, p_low, p_high, mode='train', window=args.window)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)

    print(f"Starting training")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for b_in, b_tar in train_loader:
            b_in, b_tar = b_in.to(device), b_tar.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(b_in)
                loss = criterion(out, b_tar)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss / len(train_loader):.6f}")
        gc.collect()


    model.eval()
    test_indices = np.arange(num_frames)
    test_ds = DataSet(args.input, test_indices, p_low, p_high, mode='test', window=args.window)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad(), tifffile.TiffWriter(args.output, bigtiff=True) as tif_out:
        for i, (b_in, _) in enumerate(test_loader):
            b_in = b_in.to(device)
            _, _, h, w = b_in.shape

            ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
            if ph > 0 or pw > 0:
                b_in = F.pad(b_in, (0, pw, 0, ph), mode='reflect')

            with torch.amp.autocast('cuda'):
                output = model(b_in)

            res = output[0, 0, :h, :w].cpu().numpy()
            res = (res * (p_high - p_low) + p_low).astype(np.float32)

            tif_out.write(res, contiguous=True)
            if i % 1000 == 0: print(f"Processing Frame {i}/{num_frames}")

    print("Success")


if __name__ == "__main__":
    main()
