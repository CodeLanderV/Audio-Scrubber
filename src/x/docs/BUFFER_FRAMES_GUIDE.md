## Buffer Frames Feature - Quick Reference

**Purpose:** Skip receiver startup transients by prepending random warm-up frames to the payload.

### Problem Solved
- First few transmitted bytes were often corrupted due to receiver sync/AGC settling.
- Buffer frames give the receiver time to lock onto the signal before real data arrives.

### How It Works
1. **Random junk preamble** (~128 bytes per frame) is generated and prepended to your payload.
2. **Receiver** syncs on these frames during initial lock-on.
3. **Real data** arrives after sync is stable, with fewer errors.

### Usage

**PowerShell (Windows):**
```powershell
cd "d:\Bunker\OneDrive - Amrita vishwa vidyapeetham\BaseCamp\AudioScrubber"
$env:PYTHONIOENCODING='utf-8'

# Transmit with 3 warm-up frames
python src/inference/main_inference.py --mode ofdm --data "src/inference/TxRx/content/testfile_small.txt" --buffer-frames 3

# Or use the helper script:
.\scripts\run_with_buffer.ps1
```

**Command-line arguments:**
- `--buffer-frames N` : Number of random warm-up frames (default: 0)
  - Try 1–5 frames initially; more frames = more sync time but longer TX duration
  - Each frame ≈ 128 bytes

### Example Commands

```powershell
# No buffer (baseline)
python src/inference/main_inference.py --mode ofdm --data "src/inference/TxRx/content/testfile_small.txt" --buffer-frames 0

# 2 frames (short warmup)
python src/inference/main_inference.py --mode ofdm --data "src/inference/TxRx/content/testfile_small.txt" --buffer-frames 2

# 5 frames (longer warmup)
python src/inference/main_inference.py --mode ofdm --data "src/inference/TxRx/content/testfile_small.txt" --buffer-frames 5

# With custom TX gain
python src/inference/main_inference.py --mode ofdm --data "src/inference/TxRx/content/testfile_small.txt" --buffer-frames 3 --tx-gain 10
```

### Expected Improvements
- **First block** bit success rate should increase significantly.
- **Per-block success plot** (`bits_success_rate.png`) should show fewer failures at block 0.
- **Overall BER** should improve, especially with 2–3 frames.

### Tuning Guide
1. **Start with buffer-frames=2** on your typical distance/setup.
2. If still seeing errors in early blocks → **increase to 3–4**.
3. If first block is clean but later blocks degrade → buffer isn't the problem; check:
   - TX power (multipath/fading).
   - RX gain saturation.
   - Sample rate mismatch.
4. **Max recommended: 5** frames (diminishing returns beyond this).

### Files Modified
- `src/inference/TxRx/ofdm_modulation.py` — added `buffer_frames` parameter to `modulate()`.
- `src/inference/main_inference.py` — added `--buffer-frames` CLI argument.

### Why This Works (Technical)
- **AGC & Phase Lock:** Receiver takes 1–2 OFDM frames to settle; buffer provides safe zone.
- **Frame Sync:** Preamble detector locks onto the buffer's strong synchronization signals.
- **Channel Estimation:** Receiver refines channel estimate before decoding real data.
- **No Data Loss:** Buffer bytes are discarded; only your real payload is recovered.
