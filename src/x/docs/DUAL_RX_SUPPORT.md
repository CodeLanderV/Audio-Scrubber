## Dual RX Support for OFDM - Quick Start

### Overview

OFDM now supports receiving from **either RTL-SDR or Adalm Pluto**:
- **TX:** Always Adalm Pluto (TX1)
- **RX:** Your choice → RTL-SDR (default) or Adalm Pluto (RX1)

### Commands

#### Setup 1: Pluto TX + RTL-SDR RX (Default)
```powershell
# RTL-SDR as receiver (no flag needed, it's default)
python src/inference/main_inference.py --mode ofdm --data "src/inference/TxRx/content/testfile_small.txt"

# Explicit:
python src/inference/main_inference.py --mode ofdm --data "file.bin" --rx-device rtl
```

#### Setup 2: Pluto TX + Pluto RX (Same Device or Dual Device)
```powershell
# Use Pluto for both TX and RX
python src/inference/main_inference.py --mode ofdm --data "file.bin" --rx-device pluto
```

#### Setup 3: Combined with Other Options
```powershell
# 16-QAM + Pluto RX + Buffer frames
python src/inference/main_inference.py --mode ofdm --data "file.bin" --modulation 16qam --rx-device pluto --buffer-frames 3

# QPSK + RTL + Enhanced FEC (default)
python src/inference/main_inference.py --mode ofdm --data "file.bin" --modulation qpsk --rx-device rtl --tx-gain 10

# Custom frequency
python src/inference/main_inference.py --mode ofdm --data "file.bin" --freq 900 --rx-device pluto
```

### When to Use Which RX

| Scenario | RX Device | Reason |
|----------|-----------|--------|
| Cheap/Budget setup | RTL-SDR | Low cost, easier to setup |
| Single Pluto device | RTL-SDR | Use Pluto only for TX |
| Dual Pluto devices | Pluto RX | Better sensitivity, synchronized TX/RX |
| Lab/Bench testing | Your choice | Test both paths |
| Production | Pluto RX | More reliable, professional |

### Hardware Configurations

#### Configuration A: Single Pluto + RTL-SDR (Most Common)
```
Pluto (TX IP: 192.168.2.1)  →  Antenna  ←  RTL-SDR
                             
USB Connection (Pluto + RTL to computer)
```

#### Configuration B: Dual Pluto (TX + RX)
```
Pluto1 (TX IP: 192.168.2.1)  →  Antenna  ←  Pluto2 (RX IP: 192.168.3.1)

(Both connected via Ethernet or USB hubs)
```

### Implementation Details

**File modified:** `src/inference/main_inference.py`

**New CLI flag:**
```
--rx-device {rtl|pluto}
  • rtl (default): RTL-SDR receiver
  • pluto: Adalm Pluto receiver (requires second device or reconfiguration)
```

**Logic:**
```python
if args.rx_device == 'pluto':
    rx_sdr = PlutoRX()  # Use Adalm Pluto for RX
else:
    rx_sdr = RTLSDR()   # Use RTL-SDR for RX (default)
```

### Example Output

**With RTL-SDR:**
```
📡 STEP 1: Initialize Hardware
────────────────────────────────────────────────────────────────────────────────
🔌 Checking Pluto at ip:192.168.2.1...
✅ Pluto detected
🔌 Checking RTL-SDR...
✅ RTL-SDR detected
📻 Frequency: 915.000 MHz (OFDM mode)
📡 TX Gain: 0 dB
📥 RX Device: RTL-SDR RX
```

**With Pluto RX:**
```
📡 STEP 1: Initialize Hardware
────────────────────────────────────────────────────────────────────────────────
🔌 Checking Pluto at ip:192.168.2.1...
✅ Pluto detected
🔌 Checking RX Pluto at ip:192.168.3.1...
✅ RX Pluto detected
📻 Frequency: 915.000 MHz (OFDM mode)
📡 TX Gain: 0 dB
📥 RX Device: Adalm Pluto RX
```

### Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| "RTL-SDR not available" | RTL-SDR not connected or driver missing | Check USB connection; install `pip install pyrtlsdr` |
| "RX Pluto not found" with `--rx-device pluto` | RX Pluto IP wrong or not connected | Check IP (default: 192.168.3.1); modify `SDR_CONFIG['pluto_rx_ip']` in `sdr_base.py` |
| Weird RX from Pluto with `--rx-device pluto` | Both TX and RX using same Pluto | Need two separate devices OR use `--rx-device rtl` |
| No reception from RTL | RTL tuner offline or USB error | Restart RTL-SDR; check `rtl_fm` command works |

### Advanced: Dual Pluto Setup

If you have two Pluto devices:

1. **Configure IPs:**
   - TX Pluto: `192.168.2.1` (default)
   - RX Pluto: `192.168.3.1` (edit `SDR_CONFIG['pluto_rx_ip']` in `sdr_base.py` if different)

2. **Connect via Ethernet or USB Hub:**
   - Both Pluto boards to same network or USB hub

3. **Run with Pluto RX:**
   ```powershell
   python src/inference/main_inference.py --mode ofdm --data "file.bin" --rx-device pluto
   ```

### Summary

- **Default:** Pluto TX + RTL-SDR RX
- **Alternate:** Pluto TX + Pluto RX (dual devices)
- **Flag:** `--rx-device {rtl|pluto}`
- **Backward Compatible:** Existing commands still work (RTL is default)

Choose `--rx-device rtl` for general use; switch to `--rx-device pluto` if you have dual Pluto devices for lab testing.
