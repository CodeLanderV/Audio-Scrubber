#!/usr/bin/env python3
"""
CABLE & CONNECTION DIAGNOSTIC TOOL

This script checks for common cable and connection issues that cause signal loss.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.inference.TxRx.sdr_base import PlutoSDR, RTLSDR

print("="*80)
print("CABLE & CONNECTION DIAGNOSTIC")
print("="*80)

def check_pluto_connection():
    print("\n📡 CHECKING PLUTO TX CONNECTION")
    print("-" * 80)
    
    pluto = PlutoSDR()
    if not pluto.check_device():
        print("❌ Pluto not detected - check USB/Ethernet connection")
        return False
    
    # Check TX port voltage (if available)
    try:
        pluto.configure(freq=915, rate=2000000, gain=0, bandwidth=2000000)
        print("✅ Pluto responds to configuration")
        print("✅ TX LO (Local Oscillator) locked on 915 MHz")
        
        # Try to detect if antenna is connected
        print("\n🔍 Antenna Detection:")
        print("   - If no antenna: TX output will be HIGH (no load)")
        print("   - If antenna connected: TX output will be MODERATE")
        print("   - If antenna shorted: TX output may be LOW")
        print("   → (Requires oscilloscope to measure)")
        
        pluto.stop()
        return True
    except Exception as e:
        print(f"❌ Pluto configuration failed: {e}")
        return False

def check_rtlsdr_connection():
    print("\n📡 CHECKING RTL-SDR RX CONNECTION")
    print("-" * 80)
    
    rtl = RTLSDR()
    if not rtl.check_device():
        print("❌ RTL-SDR not detected - check USB connection")
        return False
    
    try:
        rtl.configure(freq=915, rate=2000000, gain=40)
        print("✅ RTL-SDR responds to configuration")
        print("✅ RX LO locked on 915 MHz")
        
        # Check for DC offset (cable issue indicator)
        print("\n🔍 Testing RX input...")
        rx_data = rtl.receive(duration=0.1)
        
        if rx_data is not None:
            dc_offset_i = np.mean(np.real(rx_data))
            dc_offset_q = np.mean(np.imag(rx_data))
            dc_magnitude = np.sqrt(dc_offset_i**2 + dc_offset_q**2)
            
            print(f"   DC Offset I: {dc_offset_i:.6f}")
            print(f"   DC Offset Q: {dc_offset_q:.6f}")
            print(f"   DC Magnitude: {dc_magnitude:.6f}")
            
            if dc_magnitude > 0.1:
                print("   ⚠️  HIGH DC offset detected!")
                print("      Possible causes:")
                print("      - Damaged antenna connector")
                print("      - Dirt/corrosion in antenna jack")
                print("      - DC blocking capacitor issue in cable")
            else:
                print("   ✅ DC offset normal")
            
            # Check noise level
            noise_power = np.mean(np.abs(rx_data)**2)
            noise_db = 10 * np.log10(noise_power + 1e-12)
            print(f"   Noise Floor: {noise_db:.2f} dB")
            
            if noise_db > -30:
                print("   ⚠️  Noise floor HIGH!")
                print("      Possible causes:")
                print("      - Antenna not connected (sees only EMI)")
                print("      - Antenna cable shorted")
                print("      - RF shield disconnected")
                print("      - Loose connectors picking up noise")
            else:
                print("   ✅ Noise floor normal")
        
        rtl.stop()
        return True
    except Exception as e:
        print(f"❌ RTL-SDR configuration failed: {e}")
        return False

def check_cable_continuity():
    print("\n📡 CABLE CONTINUITY CHECK (Requires Multimeter)")
    print("-" * 80)
    
    print("""
To manually check cables:

TX Pluto Cable:
   ✓ Visual inspection: 
     - Look for bent SMA pins
     - Check for corrosion
     - Verify connector is tight
   ✓ With multimeter (DC setting):
     - Center pin to shield should show ~50Ω (AC), near-short (DC)
     - If open (>∞Ω): Cable broken
     - If short (<1Ω): Center and shield touching - cable damaged

RX RTL-SDR Cable:
   ✓ Same checks as TX cable
   ✓ Additional: If antenna is built-in
     - Check SMA connector on RTL-SDR dongle
     - Try different antenna

Quick Fix Ideas:
   1. Clean antenna connectors with pencil eraser
   2. Tighten all SMA connectors by hand (don't over-tighten)
   3. Try different antenna (borrow from another 915MHz device)
   4. Try different USB cable if using RTL-SDR over USB
""")

def check_frequency_accuracy():
    print("\n📡 FREQUENCY ACCURACY CHECK")
    print("-" * 80)
    
    pluto = PlutoSDR()
    if not pluto.check_device():
        return False
    
    print("Checking if Pluto and RTL-SDR can lock to same frequency...")
    
    # Test Pluto frequencies
    frequencies = [914.5, 915.0, 915.5, 916.0]
    
    print("\nFrequency tuning test:")
    for freq in frequencies:
        try:
            pluto.configure(freq=freq, rate=2000000, gain=0, bandwidth=2000000)
            actual_freq = pluto.sdr.tx_lo / 1e6
            offset = (actual_freq - freq) * 1000  # kHz
            
            status = "✅" if abs(offset) < 1 else "⚠️"
            print(f"   {status} Target: {freq:.1f} MHz → Actual: {actual_freq:.1f} MHz (Offset: {offset:.1f} kHz)")
            
            pluto.stop()
        except Exception as e:
            print(f"   ❌ {freq:.1f} MHz: {e}")

def main():
    print("\nRunning diagnostic checks...\n")
    
    results = []
    
    # Check hardware
    results.append(("Pluto Connection", check_pluto_connection()))
    results.append(("RTL-SDR Connection", check_rtlsdr_connection()))
    
    # Check frequency
    check_frequency_accuracy()
    
    # Check cables manually
    check_cable_continuity()
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    for name, result in results:
        status = "✅ OK" if result else "❌ FAILED"
        print(f"{name:.<40} {status}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    
    if all(r for _, r in results):
        print("""
✅ Hardware connections look OK!

If you're still seeing high path loss (>25 dB):
1. Check antenna presence/alignment
2. Try moving antennas closer
3. Measure with oscilloscope (if available)
4. Check for RF interference on 915 MHz
5. Try different antenna type
""")
    else:
        print("""
❌ Hardware issues detected!

Before trying anything else:
1. Fix any connection issues above
2. Physically inspect all connectors
3. Replace any suspect cables
4. Re-run this diagnostic

Then run: python diagnose_tx_rx.py
""")

if __name__ == '__main__':
    main()
