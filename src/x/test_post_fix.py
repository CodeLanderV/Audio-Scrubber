#!/usr/bin/env python3
"""
POST-FIX VERIFICATION TEST

Run this script AFTER fixing the antenna/connection issues to verify
that your TX/RX system is working properly.

Expected results with working antennas:
- Path Loss: 10-25 dB (depending on antenna placement)
- SNR in TX region: > 15 dB
- Pre-FEC BER (in signal region): 0.001-0.05
- Payload Accuracy: 90-100%
"""

import numpy as np
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parent))

from src.inference.TxRx.sdr_base import PlutoSDR, RTLSDR
from src.inference.TxRx.ofdm_modulation import OFDM_Modulation
from src.inference.TxRx.sdr_utils import SDRUtils

print("="*80)
print("POST-FIX VERIFICATION TEST")
print("="*80)

# Initialize
pluto = PlutoSDR()
rtl = RTLSDR()

if not pluto.check_device() or not rtl.check_device():
    print("❌ Hardware not available")
    sys.exit(1)

# Configure
pluto.configure(freq=915, rate=2000000, gain=0, bandwidth=2000000)
rtl.configure(freq=915, rate=2000000, gain=40)  # Lower gain to avoid noise saturation

# Create test data
modulation = OFDM_Modulation()
test_bytes = np.random.bytes(100)  # 100-byte test payload
tx_waveform = modulation.modulate(test_bytes)

print(f"\n📊 TX Signal:")
tx_power_db = 10 * np.log10(np.mean(np.abs(tx_waveform)**2) + 1e-12)
print(f"   Power: {tx_power_db:.2f} dB")

# Pad and transmit
padded = tx_waveform.copy()
if len(padded) < 65536:
    padded = np.pad(padded, (0, 65536 - len(padded)), mode='constant')

print(f"\n📡 Starting TX...")
pluto.transmit_background(padded)
time.sleep(0.5)

print(f"📥 Capturing RX (2 seconds)...")
rx_waveform = rtl.receive(duration=2.0)

if rx_waveform is None:
    print("❌ RX failed")
    sys.exit(1)

print(f"\n📊 RX Signal:")
rx_power_db = 10 * np.log10(np.mean(np.abs(rx_waveform)**2) + 1e-12)
print(f"   Power: {rx_power_db:.2f} dB")

path_loss = tx_power_db - rx_power_db
print(f"\n📊 Path Loss: {path_loss:.2f} dB")

if path_loss > 25:
    print(f"❌ FAILED: Path loss still > 25 dB")
    print(f"   Antenna/cable issue may still exist")
elif path_loss > 15:
    print(f"⚠️  MARGINAL: Path loss {path_loss:.1f} dB (expected 10-15)")
    print(f"   May need better antenna placement or higher gain")
else:
    print(f"✅ EXCELLENT: Path loss {path_loss:.1f} dB")

# Try decoding
print(f"\n🔄 Attempting decode...")
cropped = SDRUtils.crop_to_signal(rx_waveform, threshold_db=-15, padding=500)
result = modulation.demodulate(cropped)

if result['data']:
    print(f"✅ Decode successful: {len(result['data'])} bytes")
    accuracy = 100.0 * np.sum(np.frombuffer(result['data'][:len(test_bytes)], dtype=np.uint8) == 
                              np.frombuffer(test_bytes, dtype=np.uint8)) / len(test_bytes)
    print(f"   Accuracy: {accuracy:.1f}%")
else:
    print(f"❌ Decode failed")

# Cleanup
pluto.stop()
rtl.stop()

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
