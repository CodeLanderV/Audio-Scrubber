# OFDM On-the-Fly Dataset Generator

## ✅ **Recommended Approach: On-the-Fly Generation**

No need to pre-generate large .iq files! This dataset class generates clean/noisy pairs **during training**.

## Quick Start

```python
from torch.utils.data import DataLoader
from ofdm.model.dataset_generation import OFDMDataset, OFDMValidationDataset

# Training dataset (infinite diversity)
train_dataset = OFDMDataset(
    num_samples=10000,      # Virtual epoch size
    chunk_size=6400,        # 80x80 for UNet input
    snr_range=(-5, 30)      # Wide SNR for robustness
)

# Validation dataset (fixed samples for reproducibility)
val_dataset = OFDMValidationDataset(
    num_samples=1000,
    chunk_size=6400,
    snr_range=(0, 25),
    seed=42                 # Reproducible validation
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training loop
for epoch in range(epochs):
    for noisy, clean in train_loader:
        # noisy: [batch, 2, 6400] - channel 0=real, 1=imag
        # clean: [batch, 2, 6400]
        output = model(noisy)
        loss = criterion(output, clean)
        # ... backprop
```

## Advantages Over Pre-Generated

| Feature | On-the-Fly | Pre-Generated |
|---------|-----------|---------------|
| Storage | **0 MB** | 100s of MB |
| Diversity | **Infinite** (never repeats) | Limited to file |
| SNR flexibility | **Adjustable per epoch** | Fixed |
| Data types | **Mixed** (binary/text/image) | Single type |
| Setup time | Instant | Minutes + transfer |

## What It Generates

Each sample contains:
- **Clean OFDM**: Transmitted waveform (64-FFT, 16-CP, QPSK)
- **Noisy OFDM**: Same + realistic channel impairments

### Channel Impairments (All Enabled by Default)

1. **Multipath fading**: 3-tap channel (indoor/urban environment)
2. **Frequency offset**: ±40 kHz CFO (RTL-SDR realistic)
3. **Phase noise**: Oscillator drift (Wiener process)
4. **I/Q imbalance**: Amplitude/phase mismatch
5. **AWGN**: Random SNR from specified range

### Data Types (Randomized Each Sample)

- **Binary**: Pure random (compressed files, encryption)
- **Text**: ASCII patterns (structured data)
- **Image**: Smooth gradients + noise (natural images)

→ Model learns to denoise **any data type**

## Configuration Options

```python
# Minimal training (fast, less diverse)
train_ds = OFDMDataset(num_samples=1000, chunk_size=6400, snr_range=(5, 25))

# Production training (robust, general)
train_ds = OFDMDataset(num_samples=10000, chunk_size=6400, snr_range=(-5, 30))

# Extreme conditions (very robust)
train_ds = OFDMDataset(num_samples=20000, chunk_size=6400, snr_range=(-10, 35))

# AWGN only (disable other impairments)
impairments = {'multipath': False, 'freq_offset': False, 'phase_noise': False, 
               'iq_imbalance': False, 'awgn': True}
train_ds = OFDMDataset(num_samples=5000, impairment_config=impairments)
```

## Typical Usage in Training Script

```python
# src/ofdm/core/train.py

from ofdm.model.dataset_generation import OFDMDataset, OFDMValidationDataset
from ofdm.model.neuralnet import OFDM_UNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OFDM_UNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Datasets (NO FILES NEEDED!)
train_ds = OFDMDataset(num_samples=10000, chunk_size=6400, snr_range=(-5, 30))
val_ds = OFDMValidationDataset(num_samples=1000, chunk_size=6400)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# Training
for epoch in range(100):
    model.train()
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        
        output = model(noisy)
        loss = criterion(output, clean)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            val_loss += criterion(output, clean).item()
    
    print(f"Epoch {epoch}: Val Loss = {val_loss/len(val_loader):.6f}")
```

## Dataset Statistics

```python
stats = train_dataset.get_sample_stats()
print(stats)
```

Output:
```python
{
    'num_samples': 10000,
    'chunk_size': 6400,
    'snr_range': (-5, 30),
    'symbols_per_chunk': 80,
    'bits_per_chunk': 1280,
    'bytes_per_chunk': 160,
    'fft_size': 64,
    'cp_len': 16,
    'impairments': {
        'multipath': True,
        'freq_offset': True,
        'phase_noise': True,
        'iq_imbalance': True,
        'awgn': True
    }
}
```

## Memory Usage

- **Training dataset**: ~0 MB (generates on-demand)
- **Validation dataset**: ~50 MB RAM for 1000 samples (pre-generated once)
- **Batch in memory**: ~3 MB for batch_size=32

→ **Total: ~53 MB** vs **~200 MB** for pre-generated files

## Testing the Generator

```bash
# Test the dataset class
python src/ofdm/model/dataset_generation.py
```

Output should show:
- ✅ Dataset creation
- ✅ Sample generation with correct shapes
- ✅ Validation dataset pre-generation
- ✅ DataLoader batching

## Why This Works Better

1. **No file management**: No .iq files to download/store/track
2. **Infinite diversity**: Each epoch sees new samples
3. **Model generalizes**: Mixed data types prevent overfitting
4. **Easy experimentation**: Change SNR/impairments on-the-fly
5. **Fast iteration**: No waiting for dataset generation

## Migration from Pre-Generated

If you have existing .iq files, ignore them! Just use:

```python
# OLD (pre-generated)
clean_data = np.fromfile('dataset/OFDM/train_clean.iq', dtype=np.complex64)
noisy_data = np.fromfile('dataset/OFDM/train_noisy.iq', dtype=np.complex64)

# NEW (on-the-fly) - just delete the above and use:
train_ds = OFDMDataset(num_samples=10000, chunk_size=6400)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
```

No .iq files needed!

## Recommended Settings

| Training Stage | num_samples | SNR Range | Notes |
|----------------|-------------|-----------|-------|
| Quick test | 500 | (5, 25) | Fast iteration |
| Development | 5,000 | (-5, 30) | Good balance |
| **Production** | **10,000** | **(-5, 30)** | **Recommended** |
| Extreme robust | 20,000 | (-10, 35) | Very challenging |

Validation: Always 10% of training size with same SNR range.
