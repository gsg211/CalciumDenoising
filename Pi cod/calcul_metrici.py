import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def load_tif(path):
    img = cv2.imread(path, -1)
    if img is None:
        raise FileNotFoundError(f"Nu s-a putut incarca: {path}")
    return img.astype(np.float64)


def cnr(img, signal_mask, background_mask):
    signal_pixels = img[signal_mask]
    background_pixels = img[background_mask]
    mu_signal = np.mean(signal_pixels)
    mu_background = np.mean(background_pixels)
    sigma_background = np.std(background_pixels)
    return abs(mu_signal - mu_background) / sigma_background if sigma_background != 0 else float('inf')

def signal_leakage(noisy_img, denoised_img, signal_mask):
    noisy_signal = noisy_img[signal_mask]
    denoised_signal = denoised_img[signal_mask]
    return 1 - (np.mean(denoised_signal) / np.mean(noisy_signal))

def snr_global(img, signal_mask, background_mask):
    signal_pixels = img[signal_mask]
    background_pixels = img[background_mask]
    mu_signal = np.mean(signal_pixels)
    sigma_background = np.std(background_pixels)
    return 10 * np.log10(mu_signal / sigma_background) if sigma_background != 0 else float('inf')

def create_masks(img, signal_percentile=90, background_percentile=10):
    threshold_signal = np.percentile(img, signal_percentile)
    threshold_background = np.percentile(img, background_percentile)
    signal_mask = img >= threshold_signal
    background_mask = img <= threshold_background
    return signal_mask, background_mask



def evaluate_and_plot(noisy_path, denoised_path):
    noisy_img = load_tif(noisy_path)
    denoised_img = load_tif(denoised_path)

    signal_mask, background_mask = create_masks(denoised_img)

    cnr_noisy = cnr(noisy_img, signal_mask, background_mask)
    cnr_denoised = cnr(denoised_img, signal_mask, background_mask)

    snr_noisy = snr_global(noisy_img, signal_mask, background_mask)
    snr_denoised = snr_global(denoised_img, signal_mask, background_mask)

    leakage = signal_leakage(noisy_img, denoised_img, signal_mask)




    metrics = ['CNR', 'SNR global', 'Signal Leakage']
    noisy_values = [cnr_noisy, snr_noisy, 0]
    denoised_values = [cnr_denoised, snr_denoised, leakage]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x - width/2, noisy_values, width, label='Noisy', color='lightcoral')
    ax.bar(x + width/2, denoised_values, width, label='Denoised', color='steelblue')

    ax.set_ylabel('Valori metrici')
    ax.set_title('Comparatie metrici: Noisy vs Denoised')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for i in range(len(metrics)):
        ax.text(i - width/2, noisy_values[i]+0.1, f'{noisy_values[i]:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, denoised_values[i]+0.1, f'{denoised_values[i]:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    noisy_path = "noisy_6000frames.tif"
    denoised_path = "denoised_output.tif"
    evaluate_and_plot(noisy_path, denoised_path)
