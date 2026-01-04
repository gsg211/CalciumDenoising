import numpy as np
import tifffile


def estimate_noise_MAD(img):
    flat = img.reshape(-1)
    if flat.size > 1000000:
        flat = flat[::flat.size // 1000000]

    median = np.median(flat)
    mad = np.median(np.abs(flat - median))
    return 1.4826 * mad


def calculate_snr_tiff(path):
    with tifffile.TiffFile(path) as tif:
        img_mmap = tif.asarray(out='memmap')

        if img_mmap.ndim == 3:
            img_avg = np.zeros(img_mmap.shape[1:], dtype=np.float32)
            for i in range(img_mmap.shape[0]):
                img_avg += img_mmap[i].astype(np.float32)
            img_avg /= img_mmap.shape[0]
        else:
            img_avg = img_mmap.astype(np.float32)

        signal_mean = np.mean(img_avg)
        noise_std = estimate_noise_MAD(img_avg)

        if noise_std <= 1e-9:
            return np.inf

        snr = 20 * np.log10(signal_mean / noise_std)
        return snr


if __name__ == "__main__":
    tiff_path = "noisy_6000frames.tif"
    tiff_path = "denoised_output2.tif"
    try:
        snr_value = calculate_snr_tiff(tiff_path)
        print(f"SNR pentru rezultat {tiff_path}: {snr_value:.4f}")
    except Exception as e:
        print(f"Error: {e}")