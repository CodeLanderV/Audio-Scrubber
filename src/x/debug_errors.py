"""
Debug OFDM errors to see where the issue is
"""
import sys
import numpy as np
sys.path.append('src')
sys.path.append('.')

from inference.TxRx.ofdm_modulation import OFDM_Modulation

# Create modulation instance
print("="*80)
print("OFDM ERROR DIAGNOSIS")
print("="*80)

mod = OFDM_Modulation(use_ai=False, passthrough=True, modulation='qpsk')

# Test with small data
test_data = b'HELLO'
print(f'\nTest data: {test_data}')
print(f'Original bits: {np.unpackbits(np.frombuffer(test_data, dtype=np.uint8))}')

# Modulate
print("\n--- MODULATION ---")
tx_waveform = mod.modulate(test_data)
print(f'TX Waveform power: {np.mean(np.abs(tx_waveform)**2):.6f}')
print(f'TX Peak: {np.max(np.abs(tx_waveform)):.6f}')

# Perfect channel (no noise)
print("\n--- DEMODULATION (PERFECT CHANNEL) ---")
result_perfect = mod.demodulate(tx_waveform)
print(f'Original: {test_data}')
print(f'Recovered: {result_perfect["data"] if result_perfect["data"] else b"FAILED"}')
if result_perfect['data']:
    match = test_data == result_perfect['data']
    print(f'Match: {match}')
    print(f'Stats: {result_perfect["stats"]}')

# Add slight noise
print("\n--- DEMODULATION (WITH NOISE) ---")
noise = np.random.randn(len(tx_waveform)) * 0.01 + 1j * np.random.randn(len(tx_waveform)) * 0.01
rx_waveform = tx_waveform + noise
snr = 10*np.log10(np.mean(np.abs(tx_waveform)**2) / np.mean(np.abs(noise)**2))
print(f'Added noise, SNR: {snr:.2f} dB')
print(f'RX Waveform power: {np.mean(np.abs(rx_waveform)**2):.6f}')

result_noisy = mod.demodulate(rx_waveform)
print(f'Original: {test_data}')
print(f'Recovered: {result_noisy["data"] if result_noisy["data"] else b"FAILED"}')
if result_noisy['data']:
    match = test_data == result_noisy['data']
    print(f'Match: {match}')
    print(f'Stats: {result_noisy["stats"]}')

# Check what the bits look like
print("\n--- BIT-LEVEL ANALYSIS ---")
print(f'Header (first 32 bits): {np.unpackbits(np.frombuffer(test_data, dtype=np.uint8))[:32]}')

# More stress test
print("\n" + "="*80)
print("STRESS TEST - Different data sizes")
print("="*80)

for size in [5, 10, 100]:
    print(f"\nTesting {size} bytes:")
    data = bytes([i % 256 for i in range(size)])
    tx = mod.modulate(data)
    
    # Perfect channel
    result = mod.demodulate(tx)
    if result['data']:
        match = (data == result['data'])
        print(f"  Perfect channel: {'PASS' if match else 'FAIL'}")
        if not match:
            print(f"    Expected: {data[:20]}...")
            print(f"    Got:      {result['data'][:20]}...")
    else:
        print(f"  Perfect channel: FAIL (no data decoded)")
