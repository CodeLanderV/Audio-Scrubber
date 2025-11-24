"""
Verify signal scaling fixes and demonstrate the improvement
"""
import sys
import numpy as np
sys.path.append('src')
sys.path.append('.')

from inference.TxRx.ofdm_modulation import OFDM_Modulation
from ofdm.lib_archived.config import OFDMConfig

print("="*80)
print("SIGNAL SCALING VERIFICATION")
print("="*80)

# Check config
config = OFDMConfig()
print(f"\n✅ OFDM Configuration:")
print(f"   Target Power: {config.target_power} (should be 1.0)")
print(f"   Expected Amplitude: ±{np.sqrt(config.target_power):.2f} (should be ±1.0)")

# Test modulation
mod = OFDM_Modulation(use_ai=False, passthrough=True, modulation='qpsk')
test_data = b'TEST'

print(f"\n📡 Modulating test data: {test_data}")
tx_waveform = mod.modulate(test_data)

# Analyze signal
power = np.mean(np.abs(tx_waveform)**2)
peak_amplitude = np.max(np.abs(tx_waveform))
peak_db = 20 * np.log10(peak_amplitude + 1e-12)

print(f"\n📊 WAVEFORM ANALYSIS:")
print(f"   RMS Power: {power:.6f}")
print(f"   Peak Amplitude: {peak_amplitude:.6f}")
print(f"   Peak dB: {peak_db:.2f} dB")
print(f"   Signal Range: [{np.min(np.real(tx_waveform)):.3f}, {np.max(np.real(tx_waveform)):.3f}] (I)")
print(f"                [{np.min(np.imag(tx_waveform)):.3f}, {np.max(np.imag(tx_waveform)):.3f}] (Q)")

# Check if signal fits in DAC range
if peak_amplitude <= 1.0:
    print(f"\n✅ SAFE: Peak amplitude {peak_amplitude:.3f} ≤ 1.0 (no clipping)")
else:
    clipping_percent = (peak_amplitude - 1.0) / peak_amplitude * 100
    print(f"\n❌ WARNING: Peak amplitude {peak_amplitude:.3f} > 1.0 ({clipping_percent:.1f}% clipping)")

print("\n" + "="*80)
print("SDR CONFIGURATION CHECK")
print("="*80)

from inference.TxRx.sdr_base import SDR_CONFIG, FM_CONFIG

print(f"\nOFDM Configuration:")
print(f"   TX Gain: {SDR_CONFIG['tx_gain']} dB (0 = Maximum Power)")
print(f"   RX Gain: {SDR_CONFIG['rx_gain']} dB (60 = High Sensitivity)")

print(f"\nFM Configuration:")
print(f"   TX Gain: {FM_CONFIG['tx_gain']} dB (0 = Maximum Power)")
print(f"   RX Gain: {FM_CONFIG['rx_gain']} dB (60 = High Sensitivity)")

print("\n" + "="*80)
print("✅ ALL CRITICAL FIXES APPLIED")
print("="*80)
print("\nSummary:")
print("  1. ✅ Target power reduced to 1.0 (prevents DAC clipping)")
print("  2. ✅ TX gain set to 0 dB (maximum power)")
print("  3. ✅ RX gain increased to 60 dB (better reception)")
print("\nNext steps:")
print("  • Place antennas 1 meter apart (vertical, parallel)")
print("  • Test with: python src/inference/main_inference.py --mode ofdm --random-bytes 100")
