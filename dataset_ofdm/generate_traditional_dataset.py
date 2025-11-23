"""
================================================================================
OFDM Traditional Dataset Generator - Saves to .iq Files
================================================================================
Generates clean/noisy OFDM signal pairs and saves to disk.

Usage:
    python dataset_ofdm/generate_traditional_dataset.py
    python dataset_ofdm/generate_traditional_dataset.py --samples 5000000
"""

import numpy as np
import argparse
from pathlib import Path
import sys

# Add src to path
src_dir = Path(__file__).resolve().parent.parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ofdm.lib_archived.config import OFDMConfig
from ofdm.lib_archived.transceiver import OFDMTransmitter


class TraditionalOFDMDatasetGenerator:
    """Generate OFDM datasets and save to .iq files."""
    
    def __init__(self, config=None):
        self.config = config or OFDMConfig()
        self.transmitter = OFDMTransmitter(self.config)
    
    def generate_random_payload(self, num_bytes, data_type='mixed'):
        """Generate random bytes with realistic patterns."""
        if data_type == 'mixed':
            data_type = np.random.choice(['binary', 'text', 'image'])
        
        if data_type == 'binary':
            # Pure random (compressed files, encryption)
            return np.random.randint(0, 256, num_bytes, dtype=np.uint8).tobytes()
        
        elif data_type == 'text':
            # ASCII text patterns
            ascii_bytes = np.random.randint(32, 127, num_bytes, dtype=np.uint8)
            return ascii_bytes.tobytes()
        
        else:  # 'image'
            # Smooth gradients + noise
            smooth = np.linspace(0, 255, num_bytes)
            noise = np.random.randn(num_bytes) * 30
            image_bytes = np.clip(smooth + noise, 0, 255).astype(np.uint8)
            return image_bytes.tobytes()
    
    def apply_channel_impairments(self, clean_signal, snr_db):
        """Apply realistic wireless channel impairments."""
        signal = clean_signal.copy()
        N = len(signal)
        
        # 1. Multipath fading (3-tap channel)
        tap_delays = [0, np.random.randint(1, 5), np.random.randint(5, 12)]
        tap_gains = np.random.randn(3) + 1j * np.random.randn(3)
        tap_gains = tap_gains / np.sqrt(np.sum(np.abs(tap_gains)**2))
        
        h = np.zeros(max(tap_delays) + 1, dtype=np.complex64)
        for delay, gain in zip(tap_delays, tap_gains):
            h[delay] = gain
        
        signal = np.convolve(signal, h, mode='same')
        
        # 2. Frequency offset (±40 kHz CFO)
        freq_offset_hz = np.random.uniform(-40000, 40000)
        t = np.arange(N) / self.config.sample_rate
        signal = signal * np.exp(1j * 2 * np.pi * freq_offset_hz * t)
        
        # 3. Phase noise
        phase_std = np.random.uniform(0.01, 0.05)
        phase_noise = np.cumsum(np.random.randn(N) * phase_std)
        signal = signal * np.exp(1j * phase_noise)
        
        # 4. I/Q imbalance
        amp_imbalance = np.random.uniform(0.95, 1.05)
        phase_imbalance = np.random.uniform(-0.05, 0.05)
        
        I = np.real(signal)
        Q = np.imag(signal)
        I_distorted = I
        Q_distorted = amp_imbalance * (Q * np.cos(phase_imbalance) + I * np.sin(phase_imbalance))
        signal = I_distorted + 1j * Q_distorted
        
        # 5. AWGN
        sig_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = sig_power / snr_linear
        noise = (np.random.randn(N) + 1j * np.random.randn(N)) * np.sqrt(noise_power / 2)
        signal = signal + noise
        
        return signal
    
    def generate_dataset(self, total_samples, snr_range=(-5, 30), data_type='mixed'):
        """Generate complete dataset."""
        print(f"📡 Generating OFDM dataset...")
        print(f"   Config: FFT={self.config.fft_size}, CP={self.config.cp_len}")
        print(f"   Target: {total_samples:,} samples")
        print(f"   SNR range: {snr_range[0]} to {snr_range[1]} dB")
        print(f"   Data type: {data_type}")
        
        clean_buffer = []
        noisy_buffer = []
        
        samples_generated = 0
        symbol_len = self.config.symbol_len
        
        # Generate in batches
        batch_size = 1000  # symbols per batch
        
        while samples_generated < total_samples:
            # Calculate how many symbols needed
            remaining = total_samples - samples_generated
            symbols_needed = min(batch_size, int(np.ceil(remaining / symbol_len)))
            
            # Each symbol carries 8 data carriers × 2 bits = 16 bits = 2 bytes (approx)
            # But with header overhead, use more bytes
            bytes_needed = symbols_needed * 20  # Generous padding for headers
            
            # Generate random payload
            payload = self.generate_random_payload(bytes_needed, data_type)
            
            # Transmit
            clean_waveform, _ = self.transmitter.transmit(payload)
            
            # Random SNR for this batch
            snr_db = np.random.uniform(snr_range[0], snr_range[1])
            
            # Apply channel
            noisy_waveform = self.apply_channel_impairments(clean_waveform, snr_db)
            
            clean_buffer.append(clean_waveform)
            noisy_buffer.append(noisy_waveform)
            
            samples_generated += len(clean_waveform)
            
            # Progress
            if len(clean_buffer) % 100 == 0:
                progress = samples_generated / total_samples * 100
                print(f"   Progress: {progress:.1f}% ({samples_generated:,}/{total_samples:,})")
        
        # Concatenate
        clean_signal = np.concatenate(clean_buffer).astype(np.complex64)
        noisy_signal = np.concatenate(noisy_buffer).astype(np.complex64)
        
        # Trim to exact size
        clean_signal = clean_signal[:total_samples]
        noisy_signal = noisy_signal[:total_samples]
        
        print(f"✅ Generated {len(clean_signal):,} samples")
        return clean_signal, noisy_signal
    
    def save_dataset(self, clean_signal, noisy_signal, output_dir, prefix='train'):
        """Save dataset to .iq files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        clean_path = output_dir / f'{prefix}_clean.iq'
        noisy_path = output_dir / f'{prefix}_noisy.iq'
        
        print(f"\n💾 Saving to {output_dir}/")
        clean_signal.tofile(clean_path)
        noisy_signal.tofile(noisy_path)
        
        clean_size_mb = clean_path.stat().st_size / (1024 * 1024)
        noisy_size_mb = noisy_path.stat().st_size / (1024 * 1024)
        
        print(f"   {clean_path.name}: {clean_size_mb:.2f} MB")
        print(f"   {noisy_path.name}: {noisy_size_mb:.2f} MB")
        print(f"   Total: {clean_size_mb + noisy_size_mb:.2f} MB")
        
        # Metadata
        metadata_path = output_dir / f'{prefix}_metadata.txt'
        with open(metadata_path, 'w') as f:
            f.write(f"OFDM Dataset Metadata\n")
            f.write(f"=====================\n")
            f.write(f"Samples: {len(clean_signal):,}\n")
            f.write(f"FFT Size: {self.config.fft_size}\n")
            f.write(f"CP Length: {self.config.cp_len}\n")
            f.write(f"Sample Rate: {self.config.sample_rate:,} Hz\n")
            f.write(f"Symbol Length: {self.config.symbol_len} samples\n")
            f.write(f"Total Symbols: {len(clean_signal) // self.config.symbol_len:,}\n")
            f.write(f"File Size: {clean_size_mb:.2f} MB each\n")
        
        print(f"   {metadata_path.name}")
        print("✅ Dataset saved!")
        
        return clean_path, noisy_path


def main():
    parser = argparse.ArgumentParser(description='Generate OFDM dataset (traditional method)')
    parser.add_argument('--samples', type=int, default=1_000_000,
                       help='Number of samples (default: 1M = ~8MB)')
    parser.add_argument('--output', type=str, default='dataset/OFDM',
                       help='Output directory')
    parser.add_argument('--prefix', type=str, default='train',
                       help='File prefix: train/val/test')
    parser.add_argument('--snr-range', type=float, nargs=2, default=[-5, 30],
                       help='SNR range in dB')
    parser.add_argument('--data-type', type=str, default='mixed',
                       choices=['binary', 'text', 'image', 'mixed'])
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  OFDM Traditional Dataset Generator")
    print("=" * 70)
    print()
    
    # Generate
    generator = TraditionalOFDMDatasetGenerator()
    clean, noisy = generator.generate_dataset(
        total_samples=args.samples,
        snr_range=tuple(args.snr_range),
        data_type=args.data_type
    )
    
    # Save
    generator.save_dataset(clean, noisy, args.output, args.prefix)
    
    print("\n" + "=" * 70)
    print("🎉 Complete!")
    print("=" * 70)
    print(f"\nFiles saved to: {args.output}/")
    print(f"  - {args.prefix}_clean.iq")
    print(f"  - {args.prefix}_noisy.iq")
    print(f"  - {args.prefix}_metadata.txt")
    print()


if __name__ == '__main__':
    main()
