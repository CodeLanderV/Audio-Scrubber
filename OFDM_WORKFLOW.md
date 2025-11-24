# OFDM SIGNAL WORKFLOW - ACTUAL vs EXPECTED

## Current Situation (WITH 34 dB PATH LOSS)

### What SHOULD Happen (Ideal):
```
TX:  10 bytes → FEC (146 bits) → Modulate → 800 samples (10 OFDM symbols) @ -6 dB
     ↓ [TX at 0 dB gain]
     ↓ [Free space propagation, ~15 dB loss]
     ↓
RX:  ~800-1000 samples (10-12 OFDM symbols) @ -20 dB → Capture rest as noise
     ↓
Demodulate: 10-12 symbols → ~20-24 bits
     ↓
FEC Decode: 20-24 bits → Recover 10 bytes with ~95% confidence
     ↓
Output: CORRECT 10 bytes (100% accuracy)
```

### What IS Happening (ACTUAL - BROKEN):
```
TX:  10 bytes → FEC (146 bits) → Modulate → 800 samples @ -6 dB
     ↓ [TX at 0 dB gain]
     ↓ [SEVERE path loss 34.6 dB - antennas problem!]
     ↓
RX:  Captures 2,000,000 samples (25,000 symbols) but signal is BURIED in noise @ -40 dB
     ├─ Samples 0-600: Pure noise (-42 dB)
     ├─ Samples 600-1400: Actual TX signal (-34 dB) ← ONLY ~800 real samples!
     └─ Samples 1400-2000000: Pure noise (-42 dB)
     ↓
Squelch Filter: Tries to find signal boundary
     ├─ Problem: Noise floor is ONLY 8 dB below signal
     ├─ Result: Can't cleanly separate signal from noise
     └─ Currently captures: Samples 100-1500000 (1.5M symbols instead of 10!)
     ↓
Demodulate: 1,500,000 symbols → 3,000,000 bits (mostly noise!)
     ├─ Real signal bits: ~1,600 bits (from 20 valid symbols)
     └─ Noise bits: ~2,998,400 bits (garbage)
     ↓
FEC Decode: Tries to recover 10 bytes from 3M bit mess
     ├─ Pre-FEC BER: 0.45 (45% errors, mostly noise)
     ├─ FEC corrects STRUCTURE: "here's a valid packet structure"
     └─ But original data is lost in noise!
     ↓
Output: RANDOM 10 bytes (0% accuracy) that LOOK valid
```

---

## OFDM Workflow - Component Breakdown

### 1. MODULATION (TX Side)
```
Input: data_bytes (10 bytes)
  ↓ [STEP A: Add FEC Header]
    - Length header (4 bytes)
    - Data (10 bytes)
    - Total: 14 bytes = 112 bits
  ↓ [STEP B: Apply Enhanced FEC]
    - Encode 112 bits → 146 bits
    - Add error correction info
  ↓ [STEP C: Symbol Mapping (QPSK)]
    - 146 bits → 73 QPSK symbols (2 bits per symbol)
  ↓ [STEP D: OFDM Modulation]
    - Place 8 symbols on data subcarriers (per OFDM symbol)
    - Add pilots on fixed subcarriers
    - Add cyclic prefix (CP = 16 samples)
    - FFT size = 64
    - Symbol length = 64 + 16 = 80 samples
    - Total symbols: 73 / 8 ≈ 10 OFDM symbols
    - Total samples: 10 × 80 = 800 samples ← THIS IS YOUR TX!
  ↓ [STEP E: Power Normalization]
    - Scale to target_power = 0.25
    - Result: -6.02 dB power
  ↓ [TX Hardware]
Output: 800 complex I/Q samples @ -6 dB
```

### 2. TRANSMISSION (Over the Air)
```
TX Output: 800 samples @ -6 dB
  ↓ [PATH LOSS 34.6 dB] ← THIS IS YOUR PROBLEM!
  ├─ Ideal: 10-15 dB free space loss
  ├─ Actual: 34.6 dB loss
  ├─ Cause: ANTENNA/CABLE ISSUE
  │  ├─ Antenna not connected?
  │  ├─ Antenna mismatch?
  │  ├─ Frequency offset?
  │  └─ Cable damaged?
  ↓
RX Input: 800 samples @ -40 dB (buried in -42 dB noise floor)
```

### 3. RECEPTION (RX Side)
```
Captured: 4,000,000 samples @ 2 MSPS for 2 seconds
  ├─ Signal region: ~800 samples @ -34 dB
  └─ Noise region: ~3,999,200 samples @ -42 dB

Energy Detection (Squelch):
  - Peak power: 2.64 dB (relative to RX noise)
  - Threshold: 2.64 - 10 = -7.36 dB absolute
  - Problem: Noise floor is ALSO detected
  - Result: Can only crop to ~1M samples (not 800!)

Demodulation Phase:
  Input: ~1,000,000 samples (12,500 OFDM symbols)
  ├─ Real TX symbols: ~10
  └─ Noise symbols: ~12,490
  
  Processing:
    - CP removal + FFT on each symbol
    - Channel estimation (assumes "signal" in all symbols)
    - Equalization (corrupts noise into "looks like valid data")
    - QPSK demodulation (noise → random QPSK bits)
    
  Output: ~25,000 bits from noise, ~1,600 bits from real signal
           (mostly garbage with some real data mixed in)

FEC Decode:
  Input: 25,000 bits of mostly-noise
  Processing:
    - Header detection: finds "valid" header in noise
    - Payload extraction: pulls out bits
    - Error correction: fixes patterns it can
  Output: "Decoded" 88,883 bytes (from noise!)
          With 100% FEC payload accuracy (but wrong data!)
```

### 4. AI DENOISING (Optional, Currently Broken)
```
Current Status: BYPASSED due to squelch
Reason: Signal too weak (-26 dB) compared to squelch threshold (-20 dB)

What SHOULD happen (if signal was stronger):
  Input: 1M samples of noisy OFDM waveform
  ↓ [AI U-Net Model]
    - Learns: "good OFDM waveforms look like X"
    - Sees: "mostly noise with tiny signal"
    - Problem: Model trained on STRONG signals!
    - Problem: Model trained on CLEAN OFDM!
    - Output: Tries to "fix" to match training data
    - Result: HALLUCINATION - invents signal from noise
  ↓
  Output: "Cleaned" waveform (but worse if no real signal!)
```

---

## What's BROKEN Right Now

| Component | Expected | Actual | Status |
|-----------|----------|--------|--------|
| TX Power | -6 dB | -6 dB | ✅ OK |
| Path Loss | 10-15 dB | **34.6 dB** | ❌ **CRITICAL** |
| RX Power | -20 dB | -40 dB | ❌ **CRITICAL** |
| SNR (in signal) | > 20 dB | 8 dB | ❌ **MARGINAL** |
| Signal Region | 800 samples | 800 samples | ✅ OK |
| **Crop Boundary** | **~800 samples** | **~1M samples** | ❌ **FAILS** |
| Symbols Decoded | ~10 | ~12,500 | ❌ **FAILS** |
| Pre-FEC BER | 0.001-0.05 | **0.45** | ❌ **FAILS** |
| Data Accuracy | 90-100% | **0%** | ❌ **FAILS** |
| Plots (Noisy) | Show signal | Show mostly noise | ❌ **MISLEADING** |
| Plots (Clean) | Show clean | Show same as noisy | ❌ **AI disabled** |

---

## HOW TO FIX

### IMMEDIATE (Next 30 minutes)
1. **Check antenna connections physically**
   - Is TX antenna screwed onto Pluto TX port?
   - Is RX antenna screwed onto RTL-SDR connector?
   - Try moving antennas 6 inches closer
   - Try aligning antennas (both vertical or both horizontal)

2. **Run diagnostics**
   ```bash
   python diagnose_tx_rx.py
   ```
   Report if path loss improves

3. **If antennas OK, check frequency**
   - Verify Pluto is actually tuning to 915 MHz
   - Verify RTL-SDR is actually tuning to 915 MHz
   - Try 914.5 MHz or 915.5 MHz (frequency offset)

### SHORT TERM (1-2 hours)
1. Once path loss improves to <20 dB:
   - Pre-FEC BER should drop to 0.01-0.10
   - Data accuracy should jump to 30-50%
   - Plots will show actual signal vs noise

2. Then we can:
   - Enable AI denoising properly
   - Tune squelch threshold
   - Optimize power levels

### LONG TERM
- Replace antennas with higher gain
- Add signal booster/amplifier
- Use directional antennas
- Add shielding to reduce interference

---

## CODE CHANGES MADE THIS SESSION

### 1. Added squelch/energy detection (`sdr_utils.py`)
```python
crop_to_signal()  # Aggressive signal boundary detection
```

### 2. Added squelch gate in demodulation (`ofdm_modulation.py`)
```python
if signal_power_db < squelch_threshold_db:
    # Don't send to AI - prevent hallucination
```

### 3. Added cable loss compensation placeholder
```python
# Ready for future: detect and measure cable loss
```

### 4. Fixed AI denoising error handling
```python
# Handle model output size mismatches
# Return original if AI fails
```

### 5. Added diagnostic script (`diagnose_tx_rx.py`)
```python
# Measure path loss, SNR, identify problems
```

---

## NEXT STEPS FOR YOU

**Priority 1:** Check physical antenna connections
- See HARDWARE_DIAGNOSIS.md for detailed checklist

**Priority 2:** Run post-fix test
```bash
python test_post_fix.py
```
(After fixing antennas)

**Priority 3:** If path loss still >20 dB:
- Check RTL-SDR with `rtl_test` tool
- Verify both devices see same 915 MHz signal
- Try different frequencies

