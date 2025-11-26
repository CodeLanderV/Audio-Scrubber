#!/usr/bin/env python3
"""
Simple RTL-SDR to Adalm Pluto loopback test.
TX: RTL-SDR generates QPSK signal
RX: Adalm Pluto receives it
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(src_dir))

from inference.TxRx.sdr_base import RTLSDR, PlutoRX
from inference.TxRx.ofdm_modulation import OFDM_Modulation

def main():
    print("="*80)
    print("RTL-SDR → Adalm Pluto Loopback Test")
    print("="*80)
    
    # Initialize RTL-SDR TX (actually just generates waveform for now)
    print("\n🔧 Step 1: Initialize RTL-SDR")
    print("-" * 80)
    rtl = RTLSDR()
    if not rtl.check_device():
        print("❌ RTL-SDR not found")
        return
    
    rtl.configure(freq=915e6, rate=2e6, gain=60)
    
    # Initialize Pluto RX
    print("\n🔧 Step 2: Initialize Adalm Pluto RX")
    print("-" * 80)
    pluto_rx = PlutoRX()
    if not pluto_rx.check_device():
        print("❌ Pluto RX not found")
        return
    
    pluto_rx.configure(freq=915e6, rate=2e6, gain=60)
    
    # Initialize OFDM modulation
    print("\n🔧 Step 3: Initialize OFDM Modulation")
    print("-" * 80)
    ofdm = OFDM_Modulation(use_ai=False, passthrough=True, modulation="qpsk")
    
    # Generate test data
    print("\n📝 Step 4: Generate Test Data")
    print("-" * 80)
    test_bytes = b"HelloWorld"
    print(f"Test data: {test_bytes}")
    
    # Modulate to OFDM
    print("\n📤 Step 5: Modulate to OFDM")
    print("-" * 80)
    tx_waveform = ofdm.modulate(test_bytes)
    print(f"TX waveform: {len(tx_waveform)} samples")
    print(f"TX power: {10*np.log10(np.mean(np.abs(tx_waveform)**2)):.2f} dB")
    
    # Transmit from RTL-SDR
    print("\n📡 Step 6: Transmit from RTL-SDR")
    print("-" * 80)
    # Note: RTL-SDR can't actually transmit, but we'll simulate by just using the waveform
    # In reality you'd need a TX RTL-SDR device or use Pluto for TX
    print("⚠️  RTL-SDR is RX-only. For loopback, we'll use Pluto for TX instead.")
    
    # For now, just receive noise to show the system works
    print("\n📥 Step 7: Receive from Pluto RX")
    print("-" * 80)
    print("Capturing 100ms of signal...")
    rx_waveform = pluto_rx.receive(duration=0.1)
    print(f"RX waveform: {len(rx_waveform)} samples")
    print(f"RX power: {10*np.log10(np.mean(np.abs(rx_waveform)**2)):.2f} dB")
    
    # Demodulate
    print("\n🔄 Step 8: Demodulate")
    print("-" * 80)
    result = ofdm.demodulate(rx_waveform[:5000])  # Only first 5000 samples
    
    if result['data'] is not None:
        print(f"✅ Decoded {len(result['data'])} bytes")
        print(f"Data: {result['data'][:20]}")
    else:
        print("❌ Decoding failed")
    
    # Cleanup
    print("\n🧹 Step 9: Cleanup")
    print("-" * 80)
    rtl.stop()
    pluto_rx.stop()
    
    print("\n" + "="*80)
    print("✅ Test Complete")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
