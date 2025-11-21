import numpy as np
import tifffile


def estimate_noise_MAD(img):
    """
    Estimează zgomotul folosind Median Absolute Deviation (MAD).
    Funcționează bine pentru imagini de microscopie.
    """
    median = np.median(img)
    mad = np.median(np.abs(img - median))
    noise_std = 1.4826 * mad  # MAD → STD
    return noise_std


def calculate_snr_tiff(path):
    """
    Calculează SNR pentru un fișier TIFF (1 frame sau stack).
    Returnează SNR-ul în dB.
    """
    # Load TIFF
    img = tifffile.imread(path).astype(np.float32)

    # Dacă este stack, facem media pe stack
    if img.ndim == 3:
        img_avg = np.mean(img, axis=0)
    else:
        img_avg = img

    # Semnal – media intensității
    signal_mean = np.mean(img_avg)

    # Zgomot – estimare robustă
    noise_std = estimate_noise_MAD(img_avg)

    # Evitare împărțire la zero
    if noise_std == 0:
        return np.inf

    # Calcul SNR (în dB)
    snr = 20 * np.log10(signal_mean / noise_std)

    return snr


# Exemplu de utilizare:
if __name__ == "__main__":
    tiff_path = "noisy_6000frames.tif"
    snr_value = calculate_snr_tiff(tiff_path)
    print(f"SNR pentru {tiff_path}: {snr_value:.2f} dB")

    tiff_path = "denoised_output.tif"
    snr_value = calculate_snr_tiff(tiff_path)
    print(f"SNR pentru rezultat {tiff_path}: {snr_value:.2f} dB")