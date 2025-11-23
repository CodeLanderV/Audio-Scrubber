"""
================================================================================
UNIVERSAL OFDM DATASET GENERATOR
================================================================================
Generates clean/noisy OFDM waveform pairs for training AI denoiser.
Handles ANY data type: images, text, binary, audio, etc.

Output:
- dataset/OFDM/train_clean.iq
- dataset/OFDM/train_noisy.iq

Usage:
    python src/ofdm/model/generate_dataset.py --samples 10000000 --snr-min 0 --snr-max 25
================================================================================
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ofdm.core import OFDMTransceiver, add_awgn_noise


class DataGenerator:
    """Generate diverse data samples for robust training."""
    
    @staticmethod
    def generate_texts(num_texts=1000):
        """Generate diverse text samples."""
        texts = []
        
        # Common phrases
        common = [
            "The quick brown fox jumps over the lazy dog",
            "Hello World! This is a test message",
            "Lorem ipsum dolor sit amet",
            "OFDM transmission test",
            "Wireless communication system",
            "Digital signal processing",
            "Software defined radio",
            "Real-time data transmission",
        ]
        
        # Numbers and symbols
        for i in range(200):
            texts.append(f"Packet {i:05d} - Status: OK - CRC: {i*123:08x}")
            texts.append(f"Time: {i}ms | Signal: {i%10}/10 | Error: {i%5}%")
        
        # Random ASCII
        for _ in range(300):
            length = np.random.randint(20, 200)
            random_text = ''.join([chr(np.random.randint(32, 127)) for _ in range(length)])
            texts.append(random_text)
        
        # Special characters
        special = [
            "ERROR! System malfunction!!!",
            "SUCCESS: All tests passed.",
            "WARNING: Signal strength low (SNR < 10dB)",
            "INFO: Processing 1,234,567 samples @ 2.0 MSPS",
            "✓ √ ∑ ∏ ∫ ∂ ∆ Ω α β γ δ",  # Math symbols
            "© ® ™ § ¶ † ‡ • ° ± × ÷",  # Special chars
        ]
        
        # Fill to target
        all_texts = (common * 30) + texts + (special * 20)
        np.random.shuffle(all_texts)
        
        return all_texts[:num_texts]
    
    @staticmethod
    def generate_images(num_images=200, sizes=[(16,16), (32,32), (64,64)]):
        """Generate random images and convert to bytes."""
        image_bytes = []
        
        for _ in range(num_images):
            # Random size
            size = sizes[np.random.randint(len(sizes))]
            
            # Generate random RGB image
            img_array = np.random.randint(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
            img = Image.fromarray(img_array, 'RGB')
            
            # Convert to PNG bytes
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            image_bytes.append(buf.getvalue())
        
        return image_bytes
    
    @staticmethod
    def generate_binary(num_samples=300, size_range=(50, 500)):
        """Generate random binary data."""
        binary_data = []
        
        for _ in range(num_samples):
            size = np.random.randint(*size_range)
            data = np.random.bytes(size)
            binary_data.append(data)
        
        return binary_data
    
    @staticmethod
    def generate_structured(num_samples=200):
        """Generate structured packet-like data."""
        structured = []
        
        for i in range(num_samples):
            # Simulated packet: header + payload + checksum
            header = np.array([0xFF, 0xAA, i & 0xFF, (i >> 8) & 0xFF], dtype=np.uint8)
            payload = np.random.randint(0, 256, np.random.randint(50, 200), dtype=np.uint8)
            checksum = np.array([np.sum(payload) & 0xFF], dtype=np.uint8)
            
            packet = np.concatenate([header, payload, checksum]).tobytes()
            structured.append(packet)
        
        return structured


def generate_dataset(
    target_samples=10_000_000,
    snr_range=(0, 25),
    output_dir='dataset/OFDM',
    chunk_save_interval=1_000_000,
    buffer_align=65536
):
    """
    Generate comprehensive training dataset.
    
    Args:
        target_samples: Number of IQ samples to generate
        snr_range: Tuple of (min_snr, max_snr) in dB
        output_dir: Directory to save output files
        chunk_save_interval: Save to disk every N samples (memory management)
        buffer_align: Align total samples to this value (GNU Radio compatibility)
    """
    print("="*80)
    print(" "*25 + "OFDM DATASET GENERATOR")
    print("="*80)
    print(f"\nTarget Samples: {target_samples:,}")
    print(f"SNR Range: {snr_range[0]} to {snr_range[1]} dB")
    print(f"Output Directory: {output_dir}")
    print(f"Buffer Alignment: {buffer_align}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clean_file = output_dir / 'train_clean.iq'
    noisy_file = output_dir / 'train_noisy.iq'
    
    # Initialize transceiver
    transceiver = OFDMTransceiver()
    
    # Generate diverse data
    print("\n📝 Generating diverse data samples...")
    texts = DataGenerator.generate_texts(1000)
    images = DataGenerator.generate_images(200)
    binary_data = DataGenerator.generate_binary(300)
    structured = DataGenerator.generate_structured(200)
    
    # Combine all data sources
    all_data = (
        [t.encode('utf-8') for t in texts] +
        images +
        binary_data +
        structured
    )
    
    print(f"   Total unique samples: {len(all_data)}")
    print(f"   - Texts: {len(texts)}")
    print(f"   - Images: {len(images)}")
    print(f"   - Binary: {len(binary_data)}")
    print(f"   - Structured: {len(structured)}")
    
    # SNR levels (uniform distribution)
    snr_levels = np.linspace(snr_range[0], snr_range[1], 10)
    print(f"\n🎚️  SNR Levels: {snr_levels}")
    
    # Initialize storage
    clean_chunk = []
    noisy_chunk = []
    total_generated = 0
    
    # Open files for appending
    clean_fp = open(clean_file, 'wb')
    noisy_fp = open(noisy_file, 'wb')
    
    print(f"\n🚀 Starting generation...")
    print(f"   Output: {clean_file.name} and {noisy_file.name}")
    
    with tqdm(total=target_samples, unit='samples', unit_scale=True) as pbar:
        data_idx = 0
        
        while total_generated < target_samples:
            # Get next data sample (cycle through)
            data = all_data[data_idx % len(all_data)]
            data_idx += 1
            
            # Generate clean OFDM waveform
            try:
                clean_waveform, meta = transceiver.transmit(data)
            except Exception as e:
                continue
            
            # Generate multiple noisy versions at different SNRs
            for snr_db in snr_levels:
                noisy_waveform = add_awgn_noise(clean_waveform, snr_db)
                
                # Add to chunks
                clean_chunk.extend(clean_waveform)
                noisy_chunk.extend(noisy_waveform)
                
                total_generated += len(clean_waveform)
                pbar.update(len(clean_waveform))
                
                # Save chunk if interval reached
                if len(clean_chunk) >= chunk_save_interval:
                    clean_array = np.array(clean_chunk, dtype=np.complex64)
                    noisy_array = np.array(noisy_chunk, dtype=np.complex64)
                    
                    clean_array.tofile(clean_fp)
                    noisy_array.tofile(noisy_fp)
                    
                    clean_chunk = []
                    noisy_chunk = []
                    
                    clean_fp.flush()
                    noisy_fp.flush()
                
                if total_generated >= target_samples:
                    break
            
            if total_generated >= target_samples:
                break
    
    # Save remaining
    if len(clean_chunk) > 0:
        clean_array = np.array(clean_chunk, dtype=np.complex64)
        noisy_array = np.array(noisy_chunk, dtype=np.complex64)
        
        clean_array.tofile(clean_fp)
        noisy_array.tofile(noisy_fp)
    
    clean_fp.close()
    noisy_fp.close()
    
    # Align to buffer size if needed
    if buffer_align > 0:
        print(f"\n🔧 Aligning to buffer size {buffer_align}...")
        for filepath in [clean_file, noisy_file]:
            data = np.fromfile(filepath, dtype=np.complex64)
            current_len = len(data)
            
            if current_len % buffer_align != 0:
                pad_len = buffer_align - (current_len % buffer_align)
                padded = np.pad(data, (0, pad_len), mode='constant')
                padded.tofile(filepath)
                print(f"   {filepath.name}: {current_len:,} → {len(padded):,} samples")
            else:
                print(f"   {filepath.name}: {current_len:,} samples (already aligned)")
    
    # Verify
    print(f"\n🔍 Verifying files...")
    clean_data = np.fromfile(clean_file, dtype=np.complex64)
    noisy_data = np.fromfile(noisy_file, dtype=np.complex64)
    
    print(f"   Clean: {len(clean_data):,} samples ({len(clean_data)*8/1024**2:.1f} MB)")
    print(f"   Noisy: {len(noisy_data):,} samples ({len(noisy_data)*8/1024**2:.1f} MB)")
    print(f"   Clean power: {np.mean(np.abs(clean_data)**2):.2f}")
    print(f"   Noisy power: {np.mean(np.abs(noisy_data)**2):.2f}")
    print(f"   Match: {'✅' if len(clean_data) == len(noisy_data) else '❌'}")
    
    print("\n" + "="*80)
    print("✅ DATASET GENERATION COMPLETE!")
    print("="*80)
    print(f"\nNext step:")
    print(f"   python src/ofdm/model/train.py \\")
    print(f"       --clean-data {clean_file} \\")
    print(f"       --noisy-data {noisy_file} \\")
    print(f"       --epochs 100 --batch-size 32")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Generate OFDM training dataset')
    
    parser.add_argument('--samples', type=int, default=10_000_000,
                       help='Number of IQ samples (default: 10M)')
    parser.add_argument('--snr-min', type=float, default=0,
                       help='Minimum SNR in dB (default: 0)')
    parser.add_argument('--snr-max', type=float, default=25,
                       help='Maximum SNR in dB (default: 25)')
    parser.add_argument('--output-dir', type=str, default='dataset/OFDM',
                       help='Output directory (default: dataset/OFDM)')
    parser.add_argument('--chunk-size', type=int, default=1_000_000,
                       help='Save interval in samples (default: 1M)')
    parser.add_argument('--buffer-align', type=int, default=65536,
                       help='Buffer alignment (default: 65536, 0=disable)')
    
    args = parser.parse_args()
    
    generate_dataset(
        target_samples=args.samples,
        snr_range=(args.snr_min, args.snr_max),
        output_dir=args.output_dir,
        chunk_save_interval=args.chunk_size,
        buffer_align=args.buffer_align
    )


if __name__ == "__main__":
    main()
