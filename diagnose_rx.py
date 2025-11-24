"""
Diagnose RX power issues
"""
import sys
import numpy as np
sys.path.append('src')
sys.path.append('.')

from inference.TxRx.sdr_base import SDR_CONFIG, PlutoRX, RTLSDR

print("="*80)
print("RX POWER DIAGNOSIS")
print("="*80)

print(f"\nCurrent RX Gains:")
print(f"  OFDM RX Gain: {SDR_CONFIG['rx_gain']} dB")

print("\n" + "="*80)
print("POTENTIAL RX ISSUES:")
print("="*80)

issues = [
    ("1. RX Gain too low", f"Current: {SDR_CONFIG['rx_gain']} dB", "RTL-SDR max ~50-60 dB", "Increase to 48 dB"),
    ("2. Pluto RX Saturation", "If using Pluto RX:", "May need lower gain if signal is too strong", "Start with 40 dB"),
    ("3. Cable Losses", "Typical coaxial:", "-2 to -5 dB per meter", "Use short cables"),
    ("4. Antenna Impedance", "Common issue:", "90-110 Ohms instead of 50 Ohms", "Check antenna tuning"),
    ("5. Frequency Offset", "If TX/RX on different devices:", "May have slight freq drift", "Ensure both at 915 MHz"),
    ("6. Path Loss", "Free space path loss at 1m, 915 MHz:", "~32 dB attenuation", "Use amplifiers if needed"),
]

for issue, detail1, detail2, fix in issues:
    print(f"\n{issue}")
    print(f"   {detail1}")
    print(f"   {detail2}")
    print(f"   Fix: {fix}")

print("\n" + "="*80)
print("RECOMMENDED RX CONFIGURATION:")
print("="*80)

print("\nFor RTL-SDR (Budget option):")
print("  • Gain: 48 dB (close to max)")
print("  • Antenna: External 915 MHz Yagi or dipole")
print("  • Cable: <1m RG-58 (minimal loss)")
print("  • Distance: 1-2 meters from TX")

print("\nFor Adalm Pluto RX:")
print("  • Gain: 50-60 dB (depending on TX power)")
print("  • Antenna: Same as TX (vertical, parallel)")
print("  • Cable: <1m low-loss coax")
print("  • Distance: 1-2 meters from TX")

print("\n" + "="*80)
print("ACTION ITEMS:")
print("="*80)

print("""
1. ✅ Increase RX gain to 48 dB (RTL-SDR) or 60 dB (Pluto)
   Update src/inference/TxRx/sdr_base.py:
   'rx_gain': 48  (or 60 for Pluto)

2. ✅ Verify antenna setup:
   • Both antennas should be vertical and parallel
   • ~1 meter apart (test distance)
   • Check for obstructions

3. ✅ Use short cables:
   • TX cable: <1m
   • RX cable: <1m
   • Low-loss coax recommended

4. ✅ Check frequency alignment:
   • TX: 915 MHz ISM band
   • RX: Same frequency
   • Use spectrum analyzer to verify

5. ✅ Test with close distance first (0.5m) then move apart
""")
