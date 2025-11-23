# OFDM AI Denoiser - Model Directory

Clean, optimized implementation for training OFDM waveform denoising models.

## 📁 Directory Structure

```
src/ofdm/model/
├── neuralnet.py           # 1D U-Net architecture
├── generate_dataset.py    # Universal dataset generator
├── train.py              # CUDA-optimized training script
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Generate Training Data

Supports ANY data type: images, text, binary, structured packets.

```bash
# Generate 10M samples with SNR 0-25 dB
python src/ofdm/model/generate_dataset.py --samples 10000000 --snr-min 0 --snr-max 25

# Quick test with 1M samples
python src/ofdm/model/generate_dataset.py --samples 1000000

# Large dataset (100M samples)
python src/ofdm/model/generate_dataset.py --samples 100000000 --output-dir dataset/OFDM_large
```

**Output:**
- `dataset/OFDM/train_clean.iq` - Clean OFDM waveforms
- `dataset/OFDM/train_noisy.iq` - Noisy OFDM waveforms (AWGN)

### 2. Train Model

CUDA-optimized with mixed precision, early stopping, and automatic checkpointing.

```bash
# Basic training (100 epochs, batch size 32)
python src/ofdm/model/train.py \
    --clean-data dataset/OFDM/train_clean.iq \
    --noisy-data dataset/OFDM/train_noisy.iq \
    --epochs 100 \
    --batch-size 32

# Advanced: Custom settings
python src/ofdm/model/train.py \
    --clean-data dataset/OFDM/train_clean.iq \
    --noisy-data dataset/OFDM/train_noisy.iq \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.0005 \
    --chunk-size 2048 \
    --patience 30 \
    --workers 8

# CPU-only (no mixed precision)
python src/ofdm/model/train.py \
    --clean-data dataset/OFDM/train_clean.iq \
    --noisy-data dataset/OFDM/train_noisy.iq \
    --no-amp
```

**Output:**
- `saved_models/OFDM/ofdm_unet_best.pth` - Best model
- `saved_models/OFDM/checkpoint_epoch_X.pth` - Periodic checkpoints
- `saved_models/OFDM/training_history.png` - Training curves

## 📊 Dataset Details

### Data Sources

The dataset generator creates diverse OFDM waveforms from:

1. **Text samples** (1000): Common phrases, numbers, special characters
2. **Images** (200): Random RGB images (16x16, 32x32, 64x64)
3. **Binary data** (300): Random bytes
4. **Structured packets** (200): Header + payload + checksum

### SNR Range

- Default: 0 to 25 dB (uniform distribution)
- 10 SNR levels tested per data sample
- Ensures model works across channel conditions

### Buffer Alignment

- Default: 65536 samples (GNU Radio compatible)
- Ensures compatibility with SDR workflows

## 🧠 Model Architecture

**1D U-Net** for IQ waveform denoising:

- **Input**: `[Batch, 2, Length]` (I/Q channels)
- **Output**: `[Batch, 2, Length]` (denoised I/Q)
- **Parameters**: ~1.2M trainable parameters
- **Memory**: ~5 MB model size

### Architecture Details

```
Encoder:  32 → 64 → 128 → 256
Bottleneck: 512
Decoder: 256 → 128 → 64 → 32
Output: 2 channels (I/Q)
```

## ⚙️ Training Features

### CUDA Optimization

- ✅ Automatic GPU detection
- ✅ Mixed precision training (FP16/FP32)
- ✅ Pin memory for faster data transfer
- ✅ Multi-worker data loading
- ✅ Non-blocking transfers

### Memory Management

- ✅ Streaming dataset (handles large files)
- ✅ Automatic RAM preload (if < 4GB)
- ✅ Chunk-based processing
- ✅ Efficient gradient computation

### Training Control

- ✅ Early stopping (patience-based)
- ✅ Learning rate scheduling
- ✅ Automatic checkpointing
- ✅ Progress visualization
- ✅ Resume from checkpoint

## 📈 Expected Performance

### Training Time (NVIDIA GPU)

- 10M samples: ~30 minutes (RTX 3060)
- 100M samples: ~5 hours (RTX 3060)

### Convergence

- Typical convergence: 40-60 epochs
- Best results: 80-100 epochs

### Validation Loss

- Good model: < 0.01 MSE
- Excellent model: < 0.005 MSE

## 🔧 Troubleshooting

### Out of Memory (CUDA)

```bash
# Reduce batch size
--batch-size 16

# Reduce chunk size
--chunk-size 512

# Disable mixed precision
--no-amp
```

### Slow Training (CPU)

```bash
# Reduce workers
--workers 0

# Smaller dataset
--samples 1000000
```

### Model Not Learning

- Generate more diverse data (increase `--samples`)
- Adjust learning rate (try `--lr 0.0001` or `--lr 0.005`)
- Increase epochs (`--epochs 200`)
- Check dataset quality (verify clean vs noisy power difference)

## 📝 Example Workflow

```bash
# 1. Generate dataset (10M samples)
python src/ofdm/model/generate_dataset.py \
    --samples 10000000 \
    --snr-min 0 \
    --snr-max 25 \
    --output-dir dataset/OFDM

# 2. Train model (with CUDA)
python src/ofdm/model/train.py \
    --clean-data dataset/OFDM/train_clean.iq \
    --noisy-data dataset/OFDM/train_noisy.iq \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001

# 3. Test model
python src/ofdm/core/test_ai_denoising.py
```

## 🎯 Key Files

| File | Purpose |
|------|---------|
| `neuralnet.py` | 1D U-Net model definition |
| `generate_dataset.py` | Create training data from ANY source |
| `train.py` | CUDA-optimized training loop |

## 💡 Tips

1. **More data = Better model**: Generate 50-100M samples for production
2. **Use GPU**: Training is 20-50x faster on CUDA
3. **Monitor training**: Check `training_history.png` for convergence
4. **Early stopping**: Don't overtrain (model will tell you when to stop)
5. **Test regularly**: Use `test_ai_denoising.py` to verify performance

## 🔗 Related Files

- `src/ofdm/core/ofdm_pipeline.py` - OFDM implementation
- `src/ofdm/core/test_ai_denoising.py` - Model testing
- `src/ofdm/TxRx/sdr_hardware_clean.py` - SDR integration

---

**Last Updated**: 2025-11-23  
**Status**: Production Ready ✅
