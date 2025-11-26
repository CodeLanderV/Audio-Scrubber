# BER Data Loss Analysis & Fix

## Problem Summary

**Massive BER corruption due to noise-only symbols being decoded:**

1. **TX Output:** 800 samples (10 OFDM symbols) with -6.02 dB power
2. **RX Captured:** 2,000,000 samples (25,000 symbols) for 1 second
3. **Path Loss:** 34.61 dB (SEVERE - likely antenna/connection issue)
4. **SNR Margin:** Only 8.08 dB
5. **RX Processing:** ALL 25,000 symbols decoded, NOT just the ~800 real TX samples
6. **Result:** ~24,000 symbols of pure noise being decoded as garbage

## Current Flow
```
TX: 800 samples  
  ↓  [34.61 dB path loss]  
RX: 2M samples  
  ↓  [Crop to signal - attempts to find valid region]  
    → Still captures most/all of RX window as "signal"  
  ↓ [Demodulate ALL samples]  
    → 25,000 symbols demodulated  
    → Most are noise, decoded as garbage  
  ↓ [FEC decodes]  
    → 100% payload accuracy BUT  
    → Pre-FEC BER ~0.45 (50 errors/112 bits)  
    → Data Accuracy 0% (10/10 bytes wrong)
```

## Root Cause

The receiver is:
1. ✗ Processing 25,000 symbols instead of just ~10 real symbols
2. ✗ Treating noise-filled symbols as valid data
3. ✓ FEC corrects header+payload structure, but doesn't know original 10 bytes
4. ✓ So it outputs "correct" packets from garbage data

## Solutions

### SHORT TERM (Without fixing antenna/path loss)
1. **Only decode region with actual signal** - use aggressive squelch
   - Find TX pulse boundaries precisely
   - Only pass that to FEC decoder
   
2. **Measure BER on just the signal region**, not all 25,000 symbols

### LONG TERM  
1. **Check physical connections:**
   - Are antennas properly connected to TX Pluto?
   - Are antennas properly connected to RTL-SDR?
   - Are they within line-of-sight?
   
2. **Verify frequency tuning:**
   - TX tuned to 915.000 MHz?
   - RX tuned to 915.000 MHz?
   - Both using same reference clock?

3. **Increase signal power:**
   - Already using target_power = 0.25 (close to max)
   - Pluto TX is at 0 dB gain (max)
   - Further gains require antenna/cable improvements

## Recommended Fix Priority

1. **CRITICAL:** Physically check antenna connections
2. **HIGH:** Use squelch to only decode actual TX region
3. **HIGH:** Report BER only for TX region symbols
4. **MEDIUM:** Add frequency offset detection
5. **MEDIUM:** Add cable loss compensation

