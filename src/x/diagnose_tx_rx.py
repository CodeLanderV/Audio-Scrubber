"""
Diagnostic script to measure TX/RX signal strength and identify path loss issues.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.inference.TxRx.sdr_base import PlutoSDR, RTLSDR
from src.inference.TxRx.ofdm_modulation import OFDM_Modulation
import time

print("="*80)
print("TX/RX SIGNAL STRENGTH DIAGNOSTIC")
print("="*80)

# Initialize hardware
pluto_tx = PlutoSDR()
rtl_rx = RTLSDR()

if not pluto_tx.check_device():
    print("❌ TX Pluto not available")
    sys.exit(1)

if not rtl_rx.check_device():
    print("❌ RTL-SDR not available")
    sys.exit(1)

# Configure
pluto_tx.configure(freq=915, rate=2000000, gain=0, bandwidth=2000000)
rtl_rx.configure(freq=915, rate=2000000, gain=49.6)

# Create test signal
print("\n📊 Creating test signal...")
modulation = OFDM_Modulation()

# Simple test: few bits
test_bytes = b'\x00' * 10  # 10 zero bytes
tx_waveform = modulation.modulate(test_bytes)

print(f"\nTX Signal Properties:")
tx_power = np.mean(np.abs(tx_waveform)**2)
tx_power_db = 10 * np.log10(tx_power + 1e-12)
tx_peak = np.max(np.abs(tx_waveform))
tx_peak_db = 20 * np.log10(tx_peak + 1e-12)

print(f"   Length: {len(tx_waveform)} samples")
print(f"   RMS Power: {tx_power:.6f} ({tx_power_db:.2f} dB)")
print(f"   Peak Amplitude: {tx_peak:.6f} ({tx_peak_db:.2f} dB)")
print(f"   Crest Factor: {tx_peak / np.sqrt(tx_power):.2f}")

# Pad for transmission
padded_waveform = tx_waveform.copy()
if len(padded_waveform) < 65536:
    pad_len = 65536 - len(padded_waveform)
    padded_waveform = np.pad(padded_waveform, (0, pad_len), mode='constant')

print(f"\n📡 Starting TX...")
pluto_tx.transmit_background(padded_waveform)

print(f"⏱️  Waiting 1 second for signal to propagate...")
time.sleep(1)

print(f"\n📥 Capturing RX signal (1 second)...")
rx_waveform = rtl_rx.receive(duration=1.0)

if rx_waveform is not None:
    print(f"\nRX Signal Properties:")
    rx_power = np.mean(np.abs(rx_waveform)**2)
    rx_power_db = 10 * np.log10(rx_power + 1e-12)
    rx_peak = np.max(np.abs(rx_waveform))
    rx_peak_db = 20 * np.log10(rx_peak + 1e-12)
    
    print(f"   Length: {len(rx_waveform)} samples")
    print(f"   RMS Power: {rx_power:.6f} ({rx_power_db:.2f} dB)")
    print(f"   Peak Amplitude: {rx_peak:.6f} ({rx_peak_db:.2f} dB)")
    print(f"   Crest Factor: {rx_peak / np.sqrt(rx_power):.2f}")
    
    # Calculate path loss
    path_loss = tx_power_db - rx_power_db
    print(f"\n📊 PATH LOSS ANALYSIS:")
    print(f"   TX Power:  {tx_power_db:.2f} dB")
    print(f"   RX Power:  {rx_power_db:.2f} dB")
    print(f"   Path Loss: {path_loss:.2f} dB")
    
    if path_loss > 30:
        print(f"   ⚠️  SEVERE path loss detected!")
        print(f"   Possible causes:")
        print(f"      1. Antennas not connected or misaligned")
        print(f"      2. TX/RX isolation (too close or in same shielding)")
        print(f"      3. Frequency mismatch (915 MHz not reached by both)")
        print(f"      4. Cable/connector issues")
    elif path_loss > 15:
        print(f"   ⚠️  High path loss (expected 10-15 dB over short distance)")
    else:
        print(f"   ✅ Normal path loss for SDR test setup")
    
    # Check for signal in first 1000 samples (where TX waveform should be)
    print(f"\n🔍 Signal presence check:")
    first_chunk_power = np.mean(np.abs(rx_waveform[:1000])**2)
    mid_chunk_power = np.mean(np.abs(rx_waveform[500000:501000])**2)
    
    first_db = 10 * np.log10(first_chunk_power + 1e-12)
    mid_db = 10 * np.log10(mid_chunk_power + 1e-12)
    
    print(f"   First 1000 samples (TX region): {first_db:.2f} dB")
    print(f"   Mid-stream (noise only): {mid_db:.2f} dB")
    
    if first_db > mid_db + 3:
        print(f"   ✅ TX signal visible in RX (SNR = {first_db - mid_db:.2f} dB)")
    else:
        print(f"   ❌ TX signal lost in noise!")
else:
    print("❌ RX failed")

# Cleanup
pluto_tx.stop()
rtl_rx.stop()

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
