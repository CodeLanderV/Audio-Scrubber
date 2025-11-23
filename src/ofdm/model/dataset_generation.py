"""
================================================================================
OFDM Dataset Generator - On-the-Fly Training Data
================================================================================
PyTorch Dataset class that generates clean/noisy OFDM pairs during training.

Advantages:
- No storage overhead (no .iq files needed)
- Infinite data diversity (never repeats)
- Real-time SNR/impairment adjustment
- Model generalizes to any data type

Usage:
    from ofdm.model.dataset_generation import OFDMDataset
    
    dataset = OFDMDataset(num_samples=10000, chunk_size=6400)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys

# Add src to path
src_dir = Path(__file__).resolve().parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ofdm.lib_archived.config import OFDMConfig
from ofdm.lib_archived.transceiver import OFDMTransmitter


class OFDMDataset(Dataset):
    """
    On-the-fly OFDM dataset generator for training.
    
    Generates clean/noisy OFDM signal pairs with realistic channel impairments.
    Each sample is generated fresh, ensuring infinite diversity.
    """
    
    def __init__(self, num_samples=10000, chunk_size=6400, snr_range=(-5, 30),
                 config=None, impairment_config=None):
        """
        Args:
            num_samples: Number of samples per epoch (virtual dataset size)
            chunk_size: IQ samples per training sample (e.g., 6400 = 80x80 for UNet)
            snr_range: (min_snr, max_snr) in dB
            config: OFDMConfig instance (None = use defaults)
            impairment_config: Dict of channel impairments (None = all enabled)
        """
        self.num_samples = num_samples
        self.chunk_size = chunk_size
        self.snr_range = snr_range
        self.config = config or OFDMConfig()
        self.transmitter = OFDMTransmitter(self.config)
        
        # Channel impairments
        if impairment_config is None:
            self.impairments = {
                'multipath': True,
                'freq_offset': True,
                'phase_noise': True,
                'iq_imbalance': True,
                'awgn': True
            }
        else:
            self.impairments = impairment_config
        
        # QPSK constellation for random data
        self.qpsk_map = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        
        # Pre-calculate how many symbols needed per chunk
        self.symbols_per_chunk = int(np.ceil(chunk_size / self.config.symbol_len))
        self.bits_per_chunk = self.symbols_per_chunk * self.config.data_subcarriers_count * 2
        self.bytes_per_chunk = self.bits_per_chunk // 8
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Generate one clean/noisy pair.
        
        Returns:
            tuple: (noisy_chunk, clean_chunk) as torch tensors
                   Shape: (2, chunk_size) for [real, imag] channels
        """
        # Generate random payload (mixed data types for generalization)
        payload_bytes = self._generate_random_payload()
        
        # Transmit to get clean OFDM waveform
        clean_waveform, _ = self.transmitter.transmit(payload_bytes)
        
        # Trim/pad to exact chunk size
        clean_waveform = self._fit_to_chunk_size(clean_waveform)
        
        # Random SNR for this sample
        snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])
        
        # Apply channel impairments
        noisy_waveform = self._apply_channel(clean_waveform, snr_db)
        
        # Convert to PyTorch tensors [2, N] format (real, imag)
        clean_tensor = self._to_tensor(clean_waveform)
        noisy_tensor = self._to_tensor(noisy_waveform)
        
        return noisy_tensor, clean_tensor
    
    def _generate_random_payload(self):
        """Generate random bytes with realistic data patterns."""
        # Randomly choose data type for each sample (generalization)
        data_type = np.random.choice(['binary', 'text', 'image'])
        
        if data_type == 'binary':
            # Pure random (compressed data, encrypted files)
            return np.random.randint(0, 256, self.bytes_per_chunk, dtype=np.uint8).tobytes()
        
        elif data_type == 'text':
            # Text-like patterns (ASCII has structure)
            ascii_bytes = np.random.randint(32, 127, self.bytes_per_chunk, dtype=np.uint8)
            return ascii_bytes.tobytes()
        
        else:  # 'image'
            # Smooth gradients + noise (natural images)
            smooth = np.linspace(0, 255, self.bytes_per_chunk)
            noise = np.random.randn(self.bytes_per_chunk) * 30
            image_bytes = np.clip(smooth + noise, 0, 255).astype(np.uint8)
            return image_bytes.tobytes()
    
    def _fit_to_chunk_size(self, waveform):
        """Trim or zero-pad waveform to exact chunk_size."""
        if len(waveform) > self.chunk_size:
            return waveform[:self.chunk_size]
        elif len(waveform) < self.chunk_size:
            pad_len = self.chunk_size - len(waveform)
            return np.concatenate([waveform, np.zeros(pad_len, dtype=np.complex64)])
        return waveform
    
    def _apply_channel(self, clean_signal, snr_db):
        """Apply realistic wireless channel impairments."""
        signal = clean_signal.copy()
        N = len(signal)
        
        # 1. Multipath fading
        if self.impairments.get('multipath', True):
            tap_delays = [0, np.random.randint(1, 5), np.random.randint(5, 12)]
            tap_gains = np.random.randn(3) + 1j * np.random.randn(3)
            tap_gains = tap_gains / np.sqrt(np.sum(np.abs(tap_gains)**2))
            
            h = np.zeros(max(tap_delays) + 1, dtype=np.complex64)
            for delay, gain in zip(tap_delays, tap_gains):
                h[delay] = gain
            
            signal = np.convolve(signal, h, mode='same')
        
        # 2. Frequency offset (CFO)
        if self.impairments.get('freq_offset', True):
            freq_offset_hz = np.random.uniform(-40000, 40000)
            t = np.arange(N) / self.config.sample_rate
            signal = signal * np.exp(1j * 2 * np.pi * freq_offset_hz * t)
        
        # 3. Phase noise
        if self.impairments.get('phase_noise', True):
            phase_std = np.random.uniform(0.01, 0.05)
            phase_noise = np.cumsum(np.random.randn(N) * phase_std)
            signal = signal * np.exp(1j * phase_noise)
        
        # 4. I/Q imbalance
        if self.impairments.get('iq_imbalance', True):
            amp_imbalance = np.random.uniform(0.95, 1.05)
            phase_imbalance = np.random.uniform(-0.05, 0.05)
            
            I = np.real(signal)
            Q = np.imag(signal)
            I_distorted = I
            Q_distorted = amp_imbalance * (Q * np.cos(phase_imbalance) + I * np.sin(phase_imbalance))
            signal = I_distorted + 1j * Q_distorted
        
        # 5. AWGN
        if self.impairments.get('awgn', True):
            sig_power = np.mean(np.abs(signal)**2)
            snr_linear = 10**(snr_db / 10)
            noise_power = sig_power / snr_linear
            noise = (np.random.randn(N) + 1j * np.random.randn(N)) * np.sqrt(noise_power / 2)
            signal = signal + noise
        
        return signal
    
    def _to_tensor(self, complex_array):
        """
        Convert complex numpy array to PyTorch tensor.
        
        Format: [2, N] where channel 0 = real, channel 1 = imag
        """
        real = np.real(complex_array).astype(np.float32)
        imag = np.imag(complex_array).astype(np.float32)
        
        # Stack as [2, N]
        tensor = np.stack([real, imag], axis=0)
        return torch.from_numpy(tensor)
    
    def get_sample_stats(self):
        """Get statistics about dataset configuration."""
        return {
            'num_samples': self.num_samples,
            'chunk_size': self.chunk_size,
            'snr_range': self.snr_range,
            'symbols_per_chunk': self.symbols_per_chunk,
            'bits_per_chunk': self.bits_per_chunk,
            'bytes_per_chunk': self.bytes_per_chunk,
            'fft_size': self.config.fft_size,
            'cp_len': self.config.cp_len,
            'impairments': self.impairments
        }


class OFDMValidationDataset(Dataset):
    """
    Fixed validation dataset for consistent evaluation.
    
    Generates dataset once at initialization, then reuses same samples.
    Ensures reproducible validation metrics across epochs.
    """
    
    def __init__(self, num_samples=1000, chunk_size=6400, snr_range=(0, 25),
                 config=None, seed=42):
        """
        Args:
            num_samples: Number of validation samples (smaller than training)
            chunk_size: IQ samples per sample
            snr_range: SNR range for validation
            config: OFDMConfig instance
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        
        # Create temporary generator
        temp_dataset = OFDMDataset(
            num_samples=num_samples,
            chunk_size=chunk_size,
            snr_range=snr_range,
            config=config
        )
        
        # Pre-generate all validation samples
        print(f"📦 Pre-generating {num_samples} validation samples...")
        self.samples = []
        for i in range(num_samples):
            noisy, clean = temp_dataset[i]
            self.samples.append((noisy, clean))
        
        print(f"✅ Validation dataset ready ({num_samples} samples)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# Example usage and testing
if __name__ == '__main__':
    print("=" * 70)
    print("  OFDM Dataset Generator Test")
    print("=" * 70)
    
    # Create dataset
    print("\n1. Creating training dataset...")
    train_dataset = OFDMDataset(
        num_samples=100,
        chunk_size=6400,  # 80x80 for UNet
        snr_range=(-5, 30)
    )
    
    print(f"   Dataset size: {len(train_dataset)} samples")
    print(f"   Stats: {train_dataset.get_sample_stats()}")
    
    # Test sample generation
    print("\n2. Generating test samples...")
    for i in range(3):
        noisy, clean = train_dataset[i]
        print(f"   Sample {i}: noisy {noisy.shape}, clean {clean.shape}")
        print(f"      Noisy SNR: {10 * np.log10(np.mean(clean.numpy()**2) / np.mean((noisy - clean).numpy()**2)):.2f} dB")
    
    # Create validation dataset
    print("\n3. Creating validation dataset...")
    val_dataset = OFDMValidationDataset(num_samples=20, chunk_size=6400)
    
    print(f"   Validation size: {len(val_dataset)} samples")
    
    # Test with DataLoader
    print("\n4. Testing with DataLoader...")
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    
    for batch_idx, (noisy_batch, clean_batch) in enumerate(train_loader):
        print(f"   Batch {batch_idx}: noisy {noisy_batch.shape}, clean {clean_batch.shape}")
        if batch_idx >= 2:
            break
    
    print("\n✅ All tests passed!")
    print("\nUsage in training:")
    print("   from ofdm.model.dataset_generation import OFDMDataset, OFDMValidationDataset")
    print("   train_ds = OFDMDataset(num_samples=10000, chunk_size=6400)")
    print("   val_ds = OFDMValidationDataset(num_samples=1000, chunk_size=6400)")
    print("   train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)")
