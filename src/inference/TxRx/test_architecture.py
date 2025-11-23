"""
Quick test to verify the new architecture works.
Tests modulation classes without hardware.
"""

import numpy as np
from pathlib import Path
import sys

src_dir = Path(__file__).resolve().parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from inference.TxRx.ofdm_modulation import OFDM_Modulation
from inference.TxRx.fm_modulation import FM_Modulation
from inference.TxRx.sdr_utils import SDRUtils


def test_ofdm():
    """Test OFDM modulation/demodulation."""
    print("="*80)
    print("TEST 1: OFDM Modulation (Passthrough Mode)")
    print("="*80)
    
    # Initialize (passthrough = no AI)
    ofdm = OFDM_Modulation(use_ai=False, passthrough=True)
    
    # Test data
    test_data = b"Hello OFDM! This is a test message for the new architecture."
    print(f"\n📝 Test data: {test_data[:50]}...")
    
    # Modulate
    tx_waveform = ofdm.modulate(test_data)
    print(f"✅ Modulation: {len(test_data)} bytes → {len(tx_waveform)} samples")
    
    # Add noise
    snr_db = 10
    signal_power = np.mean(np.abs(tx_waveform)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(tx_waveform)) + 
                                      1j*np.random.randn(len(tx_waveform)))
    rx_waveform = tx_waveform + noise
    
    print(f"📡 Added noise: SNR = {snr_db} dB")
    
    # Demodulate
    result = ofdm.demodulate(rx_waveform)
    
    if result['data']:
        decoded_data = result['data']
        
        # Compare
        match_len = min(len(test_data), len(decoded_data))
        errors = sum(a != b for a, b in zip(test_data[:match_len], decoded_data[:match_len]))
        accuracy = 100 * (1 - errors / match_len)
        
        print(f"\n✅ Demodulation: {len(decoded_data)} bytes recovered")
        print(f"📊 Accuracy: {accuracy:.2f}% ({errors}/{match_len} byte errors)")
        print(f"📝 Decoded: {decoded_data[:50]}...")
        
        if result['stats']:
            stats = result['stats']
            print(f"📈 BER: {stats.get('ber', 0):.6f}")
    else:
        print("❌ Demodulation failed")
    
    return result['data'] is not None


def test_fm():
    """Test FM modulation/demodulation."""
    print("\n" + "="*80)
    print("TEST 2: FM Modulation (Passthrough Mode)")
    print("="*80)
    
    # Initialize (passthrough = no AI)
    fm = FM_Modulation(use_ai=False, passthrough=True)
    
    # Generate test audio (1 kHz sine wave, 0.5 seconds)
    duration = 0.5
    freq = 1000
    t = np.linspace(0, duration, int(fm.audio_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    
    print(f"\n🎵 Test audio: {freq} Hz sine, {duration}s, {len(audio)} samples")
    
    # Modulate
    tx_waveform = fm.modulate(audio)
    print(f"✅ FM Modulation: {len(audio)} audio samples → {len(tx_waveform)} IQ samples")
    
    # Add noise
    snr_db = 15
    signal_power = np.mean(np.abs(tx_waveform)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(tx_waveform)) + 
                                      1j*np.random.randn(len(tx_waveform)))
    rx_waveform = tx_waveform + noise
    
    print(f"📡 Added noise: SNR = {snr_db} dB")
    
    # Demodulate
    result = fm.demodulate(rx_waveform)
    
    if result['audio'] is not None:
        audio_rx = result['audio']
        print(f"✅ FM Demodulation: {len(audio_rx)} audio samples recovered")
        
        # Calculate correlation (as quality metric)
        if len(audio_rx) >= len(audio):
            audio_rx_trimmed = audio_rx[:len(audio)]
            correlation = np.corrcoef(audio, audio_rx_trimmed)[0, 1]
            print(f"📊 Audio Correlation: {correlation:.4f}")
        else:
            print(f"⚠️  Received audio too short ({len(audio_rx)} < {len(audio)})")
            correlation = 0
        
        return correlation > 0.8  # Good correlation
    else:
        print("❌ FM Demodulation failed")
        return False


def test_utils():
    """Test utility functions."""
    print("\n" + "="*80)
    print("TEST 3: SDRUtils")
    print("="*80)
    
    # Test data conversion
    test_waveform = np.random.randn(1000) + 1j*np.random.randn(1000)
    
    # Normalize power
    normalized = SDRUtils.normalize_power(test_waveform, target_power=1.0)
    actual_power = np.mean(np.abs(normalized)**2)
    print(f"✅ Power normalization: {actual_power:.4f} (target: 1.0)")
    
    # Scale for SDR
    scaled = SDRUtils.scale_for_sdr(test_waveform, max_amplitude=0.8)
    peak = np.max(np.abs(scaled))
    print(f"✅ SDR scaling: peak = {peak:.4f} (max: 0.8)")
    
    # SNR calculation
    clean = np.random.randn(1000) + 1j*np.random.randn(1000)
    noise = 0.1 * (np.random.randn(1000) + 1j*np.random.randn(1000))
    noisy = clean + noise
    snr = SDRUtils.calculate_snr(clean, noisy)
    print(f"✅ SNR calculation: {snr:.2f} dB")
    
    return abs(actual_power - 1.0) < 0.01 and peak <= 0.81


def main():
    """Run all tests."""
    print("\n" + "🧪"*40)
    print("NEW ARCHITECTURE VERIFICATION TESTS")
    print("🧪"*40 + "\n")
    
    results = {}
    
    # Test OFDM
    try:
        results['ofdm'] = test_ofdm()
    except Exception as e:
        print(f"❌ OFDM test error: {e}")
        import traceback
        traceback.print_exc()
        results['ofdm'] = False
    
    # Test FM
    try:
        results['fm'] = test_fm()
    except Exception as e:
        print(f"❌ FM test error: {e}")
        import traceback
        traceback.print_exc()
        results['fm'] = False
    
    # Test Utils
    try:
        results['utils'] = test_utils()
    except Exception as e:
        print(f"❌ Utils test error: {e}")
        import traceback
        traceback.print_exc()
        results['utils'] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test.upper():10s}: {status}")
    print("="*80)
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Architecture is ready for hardware testing.")
        print("\n📂 Check plots in: src/inference/plot/")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
