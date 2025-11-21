import argparse
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



class DenoiseNet(nn.Module):
    def __init__(self):
        super(DenoiseNet, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2, 2)

        #decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(32, 1, 3, padding=1))  # Output layer

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        d1 = self.dec1(self.up(e2))

        output = self.dec2(d1 + e1)

        return output



class CalciumDataset(Dataset):
    def __init__(self, data_tensor, mode='train', noise_factor=0.1):

        self.data = data_tensor
        self.mode = mode
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clean_frame = self.data[idx].unsqueeze(0)

        if self.mode == 'train':

            noise = torch.randn_like(clean_frame) * self.noise_factor
            noisy_input = clean_frame + noise
            return noisy_input, clean_frame
        else:
            return clean_frame


def load_tiff(filepath):
    print(f"Loading {filepath}...")
    img_stack = tifffile.imread(filepath)

    img_stack = img_stack.astype(np.float32)

    min_val = np.min(img_stack)
    max_val = np.max(img_stack)
    img_stack = (img_stack - min_val) / (max_val - min_val)

    print(f"Data Loaded. Shape: {img_stack.shape}. Normalized to [0, 1].")
    return torch.from_numpy(img_stack), min_val, max_val


def save_tiff(tensor_stack, output_path, min_val, max_val):
    arr = tensor_stack.cpu().numpy()

    arr = arr * (max_val - min_val) + min_val
    arr = arr.astype(np.float32)

    if arr.ndim == 4:
        arr = np.squeeze(arr, axis=1)

    tifffile.imwrite(output_path, arr)
    print(f"Saved denoised file to {output_path}")



def main():
    parser = argparse.ArgumentParser(description='Denoise Calcium Imaging Data')
    parser.add_argument('--input', type=str, required=True, help='Path to input .tif file')
    parser.add_argument('--output', type=str, default='denoised_output.tif', help='Path to save result')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--split', type=float, default=0.8, help='Train/Test split ratio (0.8 = 80% train)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    full_data, min_v, max_v = load_tiff(args.input)

    split_idx = int(len(full_data) * args.split)
    train_data = full_data[:split_idx]
    test_data = full_data[split_idx:]

    train_dataset = CalciumDataset(train_data, mode='train', noise_factor=0.05)
    test_dataset = CalciumDataset(test_data, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = DenoiseNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting Training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss / len(train_loader):.6f}")

    print("Denoising Test Data...")
    model.eval()
    denoised_stack = []

    with torch.no_grad():
        for input_frame in test_loader:
            input_frame = input_frame.to(device)
            output = model(input_frame)
            denoised_stack.append(output)

    denoised_tensor = torch.cat(denoised_stack, dim=0)

    save_tiff(denoised_tensor, args.output, min_v, max_v)


if __name__ == "__main__":
    main()