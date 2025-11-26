import sys
import importlib

# Remove cached module
if 'src.ofdm.lib_archived.modulation' in sys.modules:
    del sys.modules['src.ofdm.lib_archived.modulation']

from src.ofdm.lib_archived.modulation import QAM16
import numpy as np

qam = QAM16()
test_bits = np.array([0,1,0,0, 0,0,1,0, 1,1,0,1, 0,1,0,1], dtype=np.uint8)

print('=== MODULATION ===')
symbols = qam.modulate(test_bits)
for i in range(4):
    print(f'Symbol {i}: {symbols[i]}')

print('\n=== DEMODULATION ===')
recovered = qam.demodulate(symbols)
for i in range(4):
    expected = test_bits[i*4:(i+1)*4]
    actual = recovered[i*4:(i+1)*4]
    match = np.array_equal(expected, actual)
    status = "OK" if match else "FAIL"
    print(f'Symbol {i}: Expected {tuple(expected)}, Got {tuple(actual)} [{status}]')

print('\n=== FULL ROUND-TRIP ===')
if np.array_equal(test_bits, recovered):
    print('SUCCESS: Full round-trip passed!')
else:
    print('FAILURE: Bits do not match')
    print(f'  Input:  {test_bits}')
    print(f'  Output: {recovered}')
