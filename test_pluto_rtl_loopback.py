#!/usr/bin/env python3
"""
Simple Pluto TX → RTL-SDR RX loopback test.
TX: Adalm Pluto transmits QPSK signal
RX: RTL-SDR receives it
"""

import numpy as np
import sys
from pathlib import Path
import time

# Add src to path
src_dir = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(src_dir))

from inference.TxRx.sdr_base import PlutoSDR, RTLSDR
from inference.TxRx.ofdm_modulation import OFDM_Modulation

def main():
    print("="*80)
    print("Pluto TX → RTL-SDR RX Loopback Test")
    print("="*80)
    
    # Initialize Pluto TX
    print("\n🔧 Step 1: Initialize Adalm Pluto TX")
    print("-" * 80)
    pluto_tx = PlutoSDR()
    if not pluto_tx.check_device():
        print("❌ Pluto TX not found")
        return
    
    pluto_tx.configure(freq=915e6, rate=2e6, gain=0)  # 0 dB = max power
    
    # Initialize RTL-SDR RX
    print("\n🔧 Step 2: Initialize RTL-SDR RX")
    print("-" * 80)
    rtl_rx = RTLSDR()
    if not rtl_rx.check_device():
        print("❌ RTL-SDR not found")
        return
    
    rtl_rx.configure(freq=915e6, rate=2e6, gain=60)  # Max gain
    
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
    
    # Transmit from Pluto
    print("\n📡 Step 6: Start Pluto TX (Background)")
    print("-" * 80)
    pluto_tx.transmit_background(tx_waveform)
    
    # Wait for TX to stabilize
    print("⏱️  Waiting 2 seconds for TX to stabilize...")
    time.sleep(2)
    
    # Receive from RTL-SDR
    print("\n📥 Step 7: Receive from RTL-SDR RX")
    print("-" * 80)
    print("Capturing 1.0 second of signal...")
    rx_waveform = rtl_rx.receive(duration=1.0)
    print(f"RX waveform: {len(rx_waveform)} samples")
    print(f"RX power: {10*np.log10(np.mean(np.abs(rx_waveform)**2)):.2f} dB")
    
    # Demodulate (only first 5000 samples to avoid noise)
    print("\n🔄 Step 8: Demodulate (first 5000 samples)")
    print("-" * 80)
    result = ofdm.demodulate(rx_waveform[:5000])
    
    if result['data'] is not None and len(result['data']) > 0:
        print(f"✅ Decoded {len(result['data'])} bytes")
        print(f"Data: {result['data'][:20]}")
        
        # Check accuracy
        if len(result['data']) >= len(test_bytes):
            match = np.sum(np.frombuffer(result['data'][:len(test_bytes)], dtype=np.uint8) == 
                          np.frombuffer(test_bytes, dtype=np.uint8))
            accuracy = 100 * match / len(test_bytes)
            print(f"Accuracy: {accuracy:.1f}% ({match}/{len(test_bytes)} bytes correct)")
    else:
        print("❌ Decoding failed")
    
    # Cleanup
    print("\n🧹 Step 9: Cleanup")
    print("-" * 80)
    pluto_tx.stop()
    rtl_rx.stop()
    
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
