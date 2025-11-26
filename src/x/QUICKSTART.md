# AudioScrubber OFDM Modem - Quick Start Guide

## One-Command Start

```powershell
cd "d:\Bunker\OneDrive - Amrita vishwa vidyapeetham\BaseCamp\AudioScrubber"
$env:PYTHONIOENCODING='utf-8'

# Default (QPSK, most robust)
python src/inference/main_inference.py --mode ofdm --data "src/inference/TxRx/content/testfile_small.txt"

# 16-QAM (faster, needs clean link)
python src/inference/main_inference.py --mode ofdm --data "src/inference/TxRx/content/testfile_small.txt" --modulation 16qam

# With warm-up buffer (better sync)
python src/inference/main_inference.py --mode ofdm --data "src/inference/TxRx/content/testfile_small.txt" --buffer-frames 3
```

## Core Features

### 1. Modulation Schemes
- **QPSK** (default): 2 bits/symbol, robust, works in noise
- **16-QAM**: 4 bits/symbol, 2x faster, needs clean link

```powershell
# QPSK (automatic, no flag needed)
python src/inference/main_inference.py --mode ofdm --data file.bin

# 16-QAM explicit
python src/inference/main_inference.py --mode ofdm --data file.bin --modulation 16qam
```

### 2. Buffer Frames (Warm-up Symbols)
Prepends random frames for receiver sync. Fixes startup transient errors in first block.

```powershell
# 3 warm-up frames (recommended)
python src/inference/main_inference.py --mode ofdm --data file.bin --buffer-frames 3

# See effect: 0 vs 3 frames
python scripts/test_buffer_frames.py
```

### 3. Enhanced FEC (Error Correction)
- Hamming(7,4) on header
- Parity per-byte on payload
- Automatic, enabled by default

### 4. AI Denoising (Optional)
Denoise received signal before demodulation.

```powershell
# AI denoising (auto-detects model)
python src/inference/main_inference.py --mode ofdm --data file.bin

# Disable AI (raw OFDM only)
python src/inference/main_inference.py --mode ofdm --data file.bin --passthrough
```

## Setup by Scenario

### Robust Setup (Outdoor, Long Distance)
```powershell
python src/inference/main_inference.py --mode ofdm --data file.bin --modulation qpsk --buffer-frames 3 --tx-gain 20
```

### Fast Setup (Short Range, Clean Link)
```powershell
python src/inference/main_inference.py --mode ofdm --data file.bin --modulation 16qam --buffer-frames 2 --tx-gain 0
```

### Balanced Default
```powershell
python src/inference/main_inference.py --mode ofdm --data file.bin
```

## Testing & Verification

```powershell
# Test modulation (no hardware)
python scripts/verify_modulation.py

# Test buffer frames effect (no hardware)
python scripts/test_buffer_frames.py

# Full AI denoising test (no hardware)
python scripts/test_ofdm_ai.py --data src/inference/TxRx/content/testfile_small.txt
```

## Modulation Decision Tree

```
Is link noisy / far / uncertain?
  ├─ YES → Use QPSK (default)
  └─ NO  → Try 16-QAM

First time?
  └─ Use default (QPSK)

Transmission fails?
  └─ Add --buffer-frames 3
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| High BER | Try: `--buffer-frames 3` or `--modulation qpsk` |
| First block errors | Try: `--buffer-frames 3` |
| Connection drops | Check antenna, reduce distance, use QPSK |
| Slow transmission | Normal; increase speed with `--modulation 16qam` |

## Key Concepts

- **QPSK**: 4 constellation points, 2 bits/symbol → slower but robust
- **16-QAM**: 16 constellation points, 4 bits/symbol → 2x faster but needs SNR > 10 dB
- **Buffer Frames**: Random warm-up data helps receiver settle; payload discarded, no data loss
- **Enhanced FEC**: Error correction for header (Hamming) + payload (parity)
- **AI Denoising**: Optional; improves SNR on clean links, auto-enabled if model available

## File Management

- **Input**: Place files in `src/inference/TxRx/content/`
- **Output**: Check `output/` for results, constellation plots, received files
- **Logs**: Inline during transmission/reception

## Advanced Options

```powershell
# Custom TX gain
--tx-gain 10

# Custom RX duration
--rx-duration 5.0

# Custom frequency (MHz)
--freq 915

# Output file
--output received_data.bin

# Disable FEC
# (not recommended; modify main_inference.py if needed)
```

## One-Liner Examples

```powershell
# Transmit image, QPSK, 3 buffer frames
python src/inference/main_inference.py --mode ofdm --data "myimage.png" --buffer-frames 3

# Fast 16-QAM, small test file
python src/inference/main_inference.py --mode ofdm --data "testfile_small.txt" --modulation 16qam

# Loopback test (TX to RX, close antenna)
python src/inference/main_inference.py --mode ofdm --data "testfile_small.txt" --buffer-frames 2 --tx-gain 0

# High power, long distance attempt
python src/inference/main_inference.py --mode ofdm --data "file.bin" --modulation qpsk --buffer-frames 5 --tx-gain 20
```

## Next Steps

1. Start with default: `python src/inference/main_inference.py --mode ofdm --data testfile_small.txt`
2. If fails, add buffer: `--buffer-frames 3`
3. If slow, try 16-QAM: `--modulation 16qam`
4. If still issues, check antenna/distance/RF setup

---

**See Also:**
- `docs/BUFFER_FRAMES_GUIDE.md` — Detailed buffer frames tuning
- `docs/MODULATION_QUICK_CARD.md` — Modulation scheme comparison
- `scripts/verify_modulation.py` — Test modulation implementation
