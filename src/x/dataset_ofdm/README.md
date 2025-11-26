# OFDM Dataset Generation - Traditional Method

## Quick Start

### Generate Training Dataset (1M samples, ~8 MB each)
```bash
python dataset_ofdm/generate_traditional_dataset.py
```

### Generate Larger Dataset (5M samples, ~40 MB each)
```bash
python dataset_ofdm/generate_traditional_dataset.py --samples 5000000
```

### Generate Validation Set
```bash
python dataset_ofdm/generate_traditional_dataset.py --samples 500000 --prefix val
```

### Generate Test Set
```bash
python dataset_ofdm/generate_traditional_dataset.py --samples 200000 --prefix test
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--samples` | 1,000,000 | Number of IQ samples |
| `--output` | dataset/OFDM | Output directory |
| `--prefix` | train | File prefix (train/val/test) |
| `--snr-range` | -5 30 | Min and max SNR in dB |
| `--data-type` | mixed | binary/text/image/mixed |

## File Sizes

| Samples | Size (each) | Total | Use Case |
|---------|-------------|-------|----------|
| 200k | ~1.6 MB | ~3.2 MB | Test set |
| 500k | ~4 MB | ~8 MB | Validation |
| 1M | ~8 MB | ~16 MB | Small training |
| 5M | ~40 MB | ~80 MB | **Recommended** |
| 10M | ~80 MB | ~160 MB | Large training |

## Recommended Setup

```bash
# Training (5M samples)
python dataset_ofdm/generate_traditional_dataset.py --samples 5000000 --prefix train

# Validation (500k samples)
python dataset_ofdm/generate_traditional_dataset.py --samples 500000 --prefix val

# Test (200k samples)  
python dataset_ofdm/generate_traditional_dataset.py --samples 200000 --prefix test
```

**Total**: ~88 MB for complete dataset

## What's Generated

Each run creates:
- `{prefix}_clean.iq` - Clean OFDM signals
- `{prefix}_noisy.iq` - Same signals + realistic channel impairments
- `{prefix}_metadata.txt` - Dataset information

## Channel Impairments

Realistic wireless channel effects:
1. **Multipath fading** (3-tap channel)
2. **Frequency offset** (±40 kHz CFO)
3. **Phase noise** (oscillator drift)
4. **I/Q imbalance** (amplitude/phase mismatch)
5. **AWGN** (random SNR from range)

## Using in Training

```python
import numpy as np
import torch

# Load dataset
clean = np.fromfile('dataset/OFDM/train_clean.iq', dtype=np.complex64)
noisy = np.fromfile('dataset/OFDM/train_noisy.iq', dtype=np.complex64)

# Split into chunks (e.g., 6400 samples = 80x80 for UNet)
chunk_size = 6400
clean_chunks = clean[:len(clean)//chunk_size*chunk_size].reshape(-1, chunk_size)
noisy_chunks = noisy[:len(noisy)//chunk_size*chunk_size].reshape(-1, chunk_size)

# Convert to tensors [N, 2, chunk_size] format
def to_tensor(complex_array):
    real = np.real(complex_array).astype(np.float32)
    imag = np.imag(complex_array).astype(np.float32)
    return torch.from_numpy(np.stack([real, imag], axis=1))

clean_tensor = to_tensor(clean_chunks)
noisy_tensor = to_tensor(noisy_chunks)

# Create dataset
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(noisy_tensor, clean_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## SNR Range Guidelines

| SNR Range | Difficulty | Use Case |
|-----------|-----------|----------|
| 10-30 dB | Easy | Clean signals, light noise |
| 0-20 dB | Medium | Typical wireless conditions |
| **-5 to 30 dB** | **Hard** | **Real-world robust (recommended)** |
| -10 to 35 dB | Very Hard | Extreme conditions training |

## Channel Impairment Details

### Default (All Enabled)
- **Multipath**: 3-tap random channel (mimics reflections)
- **CFO**: ±40 kHz frequency offset (20-50 ppm at 915 MHz)
- **Phase Noise**: Cumulative phase drift (0.01-0.05 rad std)
- **I/Q Imbalance**: Amplitude (0.95-1.05×), Phase (±0.05 rad)
- **AWGN**: SNR from specified range

### Simple Channel (`--simple-channel`)
- **AWGN only**: Pure noise, no multipath/CFO
- Useful for debugging or testing AWGN-only scenarios

## Integration with Training Pipeline

Once generated, use the dataset with your training script:

```python
# In your training script
clean_data = np.fromfile('dataset/OFDM/train_clean.iq', dtype=np.complex64)
noisy_data = np.fromfile('dataset/OFDM/train_noisy.iq', dtype=np.complex64)

# Split into chunks for batching
chunk_size = 80 * 80  # Match UNet input size
clean_chunks = clean_data.reshape(-1, chunk_size)
noisy_chunks = noisy_data.reshape(-1, chunk_size)
```

## Metadata File

Each generation creates a metadata file (e.g., `train_metadata.txt`):
```
OFDM Dataset Metadata
=====================
Samples: 1,000,000
FFT Size: 64
CP Length: 16
Sample Rate: 2,000,000 Hz
Data Carriers: 8
Symbol Length: 80 samples
Total Symbols: 12,500
File Size: 8.00 MB each
```

## Example Workflow

```bash
# 1. Generate training data (5M samples, ~40 MB each)
python dataset_ofdm/generate_ofdm_dataset.py --samples 5000000

# 2. Generate validation data (500k samples, ~4 MB each)
python dataset_ofdm/generate_ofdm_dataset.py --samples 500000 --prefix val

# 3. Train your model
python src/ofdm/core/train.py

# 4. Test with hardware
python src/inference/main_inference.py --mode ofdm
```

## Why This Dataset Works

1. **Hardware-Compatible**: Matches actual OFDM transceiver config
2. **Realistic Impairments**: Models real RTL-SDR/PlutoSDR conditions
3. **Data Agnostic**: Mixed data types ensure model generalizes
4. **Compact Size**: ~88 MB total for complete train/val/test split
5. **Fast Generation**: 5M samples in ~30 seconds
6. **Reproducible**: Random variations ensure diverse training samples

## Troubleshooting

**"ModuleNotFoundError: ofdm.lib_archived"**
→ Run from project root: `python dataset_ofdm/generate_ofdm_dataset.py`

**"Out of memory"**
→ Reduce samples or batch size in script (line 145)

**"Files too large for GitHub"**
→ Use 1-2M samples for repo, generate 5-10M locally for training

## Notes

- Generated files are **not** Git-tracked (add to `.gitignore`)
- Regenerate as needed (fast process)
- SNR varies randomly within range for each batch
- Data type 'mixed' ensures model doesn't overfit to specific patterns
