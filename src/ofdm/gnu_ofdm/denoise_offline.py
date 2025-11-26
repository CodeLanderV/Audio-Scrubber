import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy import signal
from scipy.interpolate import interp1d


# ==========================================
#               CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_FILE = os.path.join(SCRIPT_DIR, "captured.iq")
DENOISED_FILE = os.path.join(SCRIPT_DIR, "denoised.iq")
MODEL_PATH = os.path.join(SCRIPT_DIR, "gnu_radio_model.pth")


CHUNK_SIZE = 16384
SAMPLE_RATE = 2e6


# ==========================================
#           MODEL DEFINITION
# ==========================================
class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class OFDM_UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(OFDM_UNet, self).__init__()
        self.enc1 = Conv1DBlock(in_channels, 32)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = Conv1DBlock(32, 64)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = Conv1DBlock(64, 128)
        self.pool3 = nn.MaxPool1d(2)
        self.enc4 = Conv1DBlock(128, 256)
        self.pool4 = nn.MaxPool1d(2)
        self.bottleneck = Conv1DBlock(256, 512)
        self.upconv4 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec4 = Conv1DBlock(512, 256)
        self.upconv3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec3 = Conv1DBlock(256, 128)
        self.upconv2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec2 = Conv1DBlock(128, 64)
        self.upconv1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec1 = Conv1DBlock(64, 32)
        self.out = nn.Conv1d(32, out_channels, kernel_size=1)


    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        b = self.bottleneck(p4)
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)


# ==========================================
#           UTILITY FUNCTIONS
# ==========================================
def estimate_snr(signal_complex):
    """Estimate SNR by comparing signal power to noise floor"""
    power = np.abs(signal_complex)**2
    signal_power = np.percentile(power, 90)  # Top 10% as signal
    noise_power = np.percentile(power, 10)   # Bottom 10% as noise
    snr_linear = signal_power / (noise_power + 1e-10)
    snr_db = 10 * np.log10(snr_linear + 1e-10)
    return snr_db


def compute_psd(signal_complex, fs):
    """Compute Power Spectral Density"""
    f, psd = signal.welch(signal_complex, fs=fs, nperseg=2048)
    return f, 10 * np.log10(psd + 1e-10)


# ==========================================
#           MAIN EXECUTION
# ==========================================
def main():
    print("=" * 60)
    print("AI OFDM DENOISER - COMPREHENSIVE ANALYSIS")
    print("=" * 60)


    if not os.path.exists(CAPTURE_FILE):
        print(f"❌ Missing: {CAPTURE_FILE}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Missing: {MODEL_PATH}")
        return


    # --- Load Data ---
    print(f"\n📂 Loading: {os.path.basename(CAPTURE_FILE)}")
    raw_signal = np.fromfile(CAPTURE_FILE, dtype=np.complex64)
    print(f"   Samples: {len(raw_signal)} ({len(raw_signal)/SAMPLE_RATE:.2f} seconds)")


    # --- Preprocess ---
    num_chunks = len(raw_signal) // CHUNK_SIZE
    valid_len = num_chunks * CHUNK_SIZE
    raw_signal = raw_signal[:valid_len]
    
    real = raw_signal.real.reshape(num_chunks, CHUNK_SIZE)
    imag = raw_signal.imag.reshape(num_chunks, CHUNK_SIZE)
    tensor_in = torch.from_numpy(np.stack([real, imag], axis=1)).float()


    # --- Load Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Device: {device}")
    model = OFDM_UNet(in_channels=2, out_channels=2).to(device)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model Loaded")
    except Exception as e:
        print(f"❌ Model Error: {e}")
        return


    # --- Inference ---
    print("🚀 Denoising...")
    model.eval()
    tensor_in = tensor_in.to(device)
    clean_chunks = []
    BATCH_SIZE = 100 
    with torch.no_grad():
        for i in range(0, num_chunks, BATCH_SIZE):
            batch = tensor_in[i : i + BATCH_SIZE]
            clean_chunks.append(model(batch).cpu().numpy())
    clean_out = np.concatenate(clean_chunks, axis=0)


    # --- Reconstruct ---
    flat_real = clean_out[:, 0, :].flatten()
    flat_imag = clean_out[:, 1, :].flatten()
    clean_signal = flat_real + 1j * flat_imag
    
    # ============================================
    #     OFDM STRUCTURE PRESERVATION
    # ============================================
    # print("🔧 Preserving OFDM structure...")

    # # Step 1: Match Power Spectrum to Input
    # f_input, psd_input = signal.welch(raw_signal, fs=SAMPLE_RATE, nperseg=1024)
    # f_output, psd_output = signal.welch(clean_signal, fs=SAMPLE_RATE, nperseg=1024)

    # # Calculate correction filter
    # correction = np.sqrt(psd_input / (psd_output + 1e-10))

    # # Apply spectral shaping to match input characteristics
    # fft_clean = np.fft.fft(clean_signal)
    # freqs = np.fft.fftfreq(len(clean_signal), 1/SAMPLE_RATE)

    # # Interpolate correction to match FFT size
    # interp_correction = interp1d(f_input, correction, bounds_error=False, fill_value=1.0)
    # correction_full = interp_correction(np.abs(freqs))

    # # Apply correction
    # fft_clean_corrected = fft_clean * correction_full
    # clean_signal = np.fft.ifft(fft_clean_corrected)

    # # Step 2: Match Exact Power Level
    # input_power = np.mean(np.abs(raw_signal)**2)
    # output_power = np.mean(np.abs(clean_signal)**2)
    # clean_signal = clean_signal * np.sqrt(input_power / (output_power + 1e-10))

    # # Step 3: Preserve Phase Reference (Critical for OFDM)
    # phase_input = np.angle(raw_signal)
    # phase_output = np.angle(clean_signal)
    # phase_diff = np.median(phase_input - phase_output)
    # clean_signal = clean_signal * np.exp(1j * phase_diff)

    # # Step 4: Remove any DC offset difference
    # dc_input = np.mean(raw_signal)
    # dc_output = np.mean(clean_signal)
    # clean_signal = clean_signal + (dc_input - dc_output)

    # print("✅ Structure preserved")
    
#     print("🔧 Hybrid Denoising (Preserve High-Power Samples)...")

# # Find high-power samples (likely OFDM structure)
#     power = np.abs(raw_signal)**2
#     threshold = np.percentile(power, 75)  # Top 25% power

#     # Create mask
#     mask = power > threshold

#     # Hybrid: Keep original high-power (structure), use AI for low-power (noise)
#     clean_signal = np.where(mask, raw_signal[:len(clean_signal)], clean_signal)

#     # Match overall power
#     input_power = np.mean(np.abs(raw_signal)**2)
#     output_power = np.mean(np.abs(clean_signal)**2)
#     clean_signal = clean_signal * np.sqrt(input_power / (output_power + 1e-10))

#     print("✅ Hybrid denoising applied")
#     # ============================================
    # ============================================
    #   ULTRA-CONSERVATIVE HYBRID (GUARANTEED)
    # ============================================
    print("🔧 Ultra-Conservative Hybrid Denoising...")

    # Only use AI on the WEAKEST signals (bottom 50% power)
    # Keep everything else UNTOUCHED
    power = np.abs(raw_signal)**2
    threshold = np.percentile(power, 50)  # Only denoise bottom 50%

    # Conservative blend: mostly original, some AI
    alpha = 0.3  # Only 30% AI contribution for low-power samples
    mask = power > threshold

    # For high power: use 100% original
    # For low power: use 70% original + 30% AI
    clean_signal = np.where(mask, 
                            raw_signal[:len(clean_signal)],  # High power: keep original
                            alpha * clean_signal + (1-alpha) * raw_signal[:len(clean_signal)])  # Low power: blend

    # Exact power match
    input_power = np.mean(np.abs(raw_signal)**2)
    output_power = np.mean(np.abs(clean_signal)**2)
    clean_signal = clean_signal * np.sqrt(input_power / (output_power + 1e-10))

    print("✅ Conservative denoising complete")
    
    print(f"💾 Saving: {os.path.basename(DENOISED_FILE)}")
    clean_signal.astype(np.complex64).tofile(DENOISED_FILE)


    # --- Compute Metrics ---
    N = min(200000, len(raw_signal))
    noisy_slice = raw_signal[:N]
    clean_slice = clean_signal[:N]
    
    snr_noisy = estimate_snr(noisy_slice)
    snr_clean = estimate_snr(clean_slice)
    snr_improvement = snr_clean - snr_noisy
    
    print(f"\n📊 SNR Analysis:")
    print(f"   Noisy:  {snr_noisy:.2f} dB")
    print(f"   Clean:  {snr_clean:.2f} dB")
    print(f"   Gain:   +{snr_improvement:.2f} dB")


    # --- PLOTTING ---
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'AI Denoising Results | SNR Improvement: +{snr_improvement:.2f} dB', 
                 fontsize=14, fontweight='bold')


    # 1. Spectrograms
    ax1 = plt.subplot(3, 3, 1)
    plt.specgram(noisy_slice[:100000], NFFT=1024, Fs=SAMPLE_RATE/1e6, cmap='inferno')
    plt.title('Spectrogram: Noisy Input')
    plt.ylabel('Frequency (MHz)')
    plt.colorbar(label='Power (dB)')


    ax2 = plt.subplot(3, 3, 2)
    plt.specgram(clean_slice[:100000], NFFT=1024, Fs=SAMPLE_RATE/1e6, cmap='inferno')
    plt.title('Spectrogram: AI Denoised')
    plt.ylabel('Frequency (MHz)')
    plt.colorbar(label='Power (dB)')


    # 2. Power Spectral Density
    ax3 = plt.subplot(3, 3, 3)
    f_noisy, psd_noisy = compute_psd(noisy_slice, SAMPLE_RATE)
    f_clean, psd_clean = compute_psd(clean_slice, SAMPLE_RATE)
    plt.plot(f_noisy/1e6, psd_noisy, label='Noisy', alpha=0.7, linewidth=1.5)
    plt.plot(f_clean/1e6, psd_clean, label='Denoised', alpha=0.9, linewidth=2)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB/Hz)')
    plt.legend()
    plt.grid(True, alpha=0.3)


    # 3. Time Domain Comparison
    ax4 = plt.subplot(3, 3, 4)
    t = np.arange(4000) / SAMPLE_RATE * 1e3
    plt.plot(t, noisy_slice[:4000].real, label='Noisy', alpha=0.6, linewidth=0.8)
    plt.plot(t, clean_slice[:4000].real, label='Denoised', alpha=0.9, linewidth=1.2)
    plt.title('Time Domain (Real Part)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)


    # 4. IQ Scatter (Pseudo-Constellation)
    ax5 = plt.subplot(3, 3, 5)
    plt.scatter(noisy_slice[:5000].real, noisy_slice[:5000].imag, 
                s=1, alpha=0.3, label='Noisy', c='red')
    plt.title('IQ Scatter: Noisy')
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')


    ax6 = plt.subplot(3, 3, 6)
    plt.scatter(clean_slice[:5000].real, clean_slice[:5000].imag, 
                s=1, alpha=0.5, label='Denoised', c='blue')
    plt.title('IQ Scatter: Denoised')
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')


    # 5. Amplitude Histograms
    ax7 = plt.subplot(3, 3, 7)
    plt.hist(np.abs(noisy_slice), bins=100, alpha=0.7, label='Noisy', color='red', density=True)
    plt.hist(np.abs(clean_slice), bins=100, alpha=0.7, label='Denoised', color='blue', density=True)
    plt.title('Amplitude Distribution')
    plt.xlabel('Magnitude')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)


    # 6. SNR Bar Chart
    ax8 = plt.subplot(3, 3, 8)
    bars = plt.bar(['Noisy', 'Denoised'], [snr_noisy, snr_clean], 
                   color=['red', 'green'], alpha=0.7)
    plt.title('SNR Comparison')
    plt.ylabel('SNR (dB)')
    plt.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} dB', ha='center', va='bottom', fontweight='bold')


    # 7. Improvement Summary Text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""
    DENOISING SUMMARY
    
    Input Samples: {len(raw_signal):,}
    Duration: {len(raw_signal)/SAMPLE_RATE:.2f} sec
    
    SNR Before: {snr_noisy:.2f} dB
    SNR After:  {snr_clean:.2f} dB
    
    Improvement: +{snr_improvement:.2f} dB
    
    Model: OFDM UNet
    Device: {device}
    Chunk Size: {CHUNK_SIZE}
    """
    ax9.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))


    plt.tight_layout()
    plt.show()
    print("\n🎉 Analysis Complete!")


if __name__ == "__main__":
    main()
