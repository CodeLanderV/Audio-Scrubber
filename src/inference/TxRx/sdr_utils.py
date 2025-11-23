"""
================================================================================
SDR UTILITIES - Data Conversion, Plotting, Helper Functions
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import io


class SDRUtils:
    """Utility functions for SDR operations."""
    
    @staticmethod
    def load_data(file_path):
        """
        Load any type of data from file.
        
        Args:
            file_path: Path to file (image, text, binary, etc.)
            
        Returns:
            bytes: Raw file data
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = f.read()
        
        print(f"📁 Loaded: {path.name} ({len(data)} bytes)")
        return data
    
    @staticmethod
    def save_data(data, output_path):
        """
        Save data to file.
        
        Args:
            data: bytes to save
            output_path: Output file path
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(data)
        
        print(f"💾 Saved: {path.name} ({len(data)} bytes)")
    
    @staticmethod
    def plot_waveform(waveform, title, filename, sample_rate=2e6):
        """
        Plot comprehensive waveform analysis (time, constellation, histogram, PSD).
        
        Args:
            waveform: Complex IQ samples
            title: Plot title
            filename: Output filename (will be saved in src/inference/plot/)
            sample_rate: Sample rate in Hz
        """
        output_dir = Path('src/inference/plot')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Time domain (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(np.real(waveform[:1000]), label='Real', alpha=0.7, linewidth=0.8)
        ax1.plot(np.imag(waveform[:1000]), label='Imag', alpha=0.7, linewidth=0.8)
        ax1.set_title('Time Domain (First 1000 samples)', fontweight='bold')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Constellation (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(np.real(waveform[:5000]), np.imag(waveform[:5000]), 
                   alpha=0.3, s=1, c='blue')
        ax2.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax2.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        ax2.set_title('Constellation (First 5000 samples)', fontweight='bold')
        ax2.set_xlabel('I (Real)')
        ax2.set_ylabel('Q (Imag)')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Histogram (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(np.real(waveform), bins=100, alpha=0.6, label='Real', density=False)
        ax3.hist(np.imag(waveform), bins=100, alpha=0.6, label='Imag', density=False)
        ax3.set_title('Histogram (Distribution)', fontweight='bold')
        ax3.set_xlabel('Amplitude')
        ax3.set_ylabel('Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Power Spectral Density (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.psd(waveform, NFFT=1024, Fs=sample_rate/1e6, scale_by_freq=False)
        ax4.set_title('Power Spectral Density', fontweight='bold')
        ax4.set_xlabel('Frequency (MHz)')
        ax4.set_ylabel('Power Spectral Density (dB/Hz)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plot saved: {output_path}")
    
    @staticmethod
    def plot_constellation(symbols, title, filename, reference_points=None):
        """
        Plot constellation diagram.
        
        Args:
            symbols: Complex symbols
            title: Plot title
            filename: Output filename
            reference_points: Reference constellation points (optional)
        """
        output_dir = Path('src/inference/plot')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot received symbols
        ax.scatter(np.real(symbols), np.imag(symbols), 
                  alpha=0.3, s=10, c='blue', label='Received')
        
        # Plot reference points if provided
        if reference_points is not None:
            ax.scatter(np.real(reference_points), np.imag(reference_points),
                      c='red', s=200, marker='x', linewidths=3, label='Reference')
        
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('I (Real)', fontweight='bold')
        ax.set_ylabel('Q (Imag)', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.legend()
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Constellation saved: {output_path}")
    
    @staticmethod
    def plot_fm_analysis(audio, demodulated, denoised, modulation, model_name, sample_rate=44100):
        """
        Plot comprehensive FM comparison (waveform + spectrograms at a glance).
        
        Args:
            audio: Original audio waveform (not used, for compatibility)
            demodulated: Demodulated FM signal (noisy)
            denoised: Denoised signal
            modulation: 'FM'
            model_name: Model name for filename
            sample_rate: Audio sample rate (default 44.1kHz)
        """
        output_dir = Path('src/inference/plot')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, height_ratios=[1, 1.5, 1.5])
        
        # Waveform comparison (top, spans both columns)
        ax1 = fig.add_subplot(gs[0, :])
        # Plot last 10 seconds for better visualization
        plot_samples = min(len(demodulated), int(sample_rate * 10))
        time_axis = np.arange(plot_samples) / sample_rate
        ax1.plot(time_axis, demodulated[-plot_samples:], alpha=0.6, linewidth=0.5, 
                label='Original (Noisy)', color='orange')
        ax1.plot(time_axis, denoised[-plot_samples:], alpha=0.8, linewidth=0.5, 
                label='Denoised (Clean)', color='blue')
        ax1.set_title('Waveform Comparison (Last 10s)', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-1.1, 1.1])
        
        # Original spectrogram (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        D_noisy = np.abs(np.fft.rfft(demodulated.reshape(-1, 512), axis=1).T)
        im1 = ax2.imshow(20 * np.log10(D_noisy + 1e-10), aspect='auto', origin='lower',
                        cmap='viridis', interpolation='bilinear', vmin=-80, vmax=0)
        ax2.set_title('Original Spectrogram', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Frequency (Hz)')
        # Set frequency ticks
        freq_ticks = np.linspace(0, D_noisy.shape[0], 5)
        freq_labels = [f'{int(f * sample_rate / 1024)}' for f in freq_ticks]
        ax2.set_yticks(freq_ticks)
        ax2.set_yticklabels(freq_labels)
        plt.colorbar(im1, ax=ax2, label='dB', fraction=0.046, pad=0.04)
        
        # Denoised spectrogram (middle right)
        ax3 = fig.add_subplot(gs[1, 1])
        D_clean = np.abs(np.fft.rfft(denoised.reshape(-1, 512), axis=1).T)
        im2 = ax3.imshow(20 * np.log10(D_clean + 1e-10), aspect='auto', origin='lower',
                        cmap='viridis', interpolation='bilinear', vmin=-80, vmax=0)
        ax3.set_title('Denoised Spectrogram', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_yticks(freq_ticks)
        ax3.set_yticklabels(freq_labels)
        plt.colorbar(im2, ax=ax3, label='dB', fraction=0.046, pad=0.04)
        
        # Estimated noise profile (bottom, spans both columns)
        ax4 = fig.add_subplot(gs[2, :])
        noise_estimate = demodulated - denoised
        ax4.fill_between(np.arange(len(noise_estimate)), noise_estimate, 
                         alpha=0.5, color='red', linewidth=0)
        ax4.plot(noise_estimate, alpha=0.3, linewidth=0.3, color='darkred')
        ax4.set_title('Estimated Noise Profile (Removed by AI)', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Time (samples)')
        ax4.set_ylabel('Amplitude')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([np.min(noise_estimate) * 1.2, np.max(noise_estimate) * 1.2])
        
        plt.suptitle(f'FM Denoising Analysis - Model: {model_name}', 
                    fontsize=16, fontweight='bold', y=0.998)
        
        filename = f'FM_Analysis_{model_name}.png'
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 FM analysis saved: {output_path}")
    
    @staticmethod
    def normalize_power(waveform, target_power=1.0):
        """
        Normalize waveform to target power.
        
        Args:
            waveform: Complex IQ samples
            target_power: Target average power
            
        Returns:
            Normalized waveform
        """
        current_power = np.mean(np.abs(waveform)**2)
        if current_power > 0:
            scale = np.sqrt(target_power / current_power)
            return waveform * scale
        return waveform
    
    @staticmethod
    def scale_for_sdr(waveform, max_amplitude=0.8):
        """
        Scale waveform to prevent clipping in SDR.
        
        Args:
            waveform: Complex IQ samples
            max_amplitude: Maximum amplitude (0-1)
            
        Returns:
            Scaled waveform
        """
        peak = np.max(np.abs(waveform))
        if peak > 0:
            return waveform * (max_amplitude / peak)
        return waveform
    
    @staticmethod
    def calculate_snr(clean, noisy):
        """
        Calculate Signal-to-Noise Ratio.
        
        Args:
            clean: Clean signal
            noisy: Noisy signal
            
        Returns:
            SNR in dB
        """
        noise = noisy - clean
        signal_power = np.mean(np.abs(clean)**2)
        noise_power = np.mean(np.abs(noise)**2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = np.inf
        
        return snr
