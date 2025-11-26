import numpy as np
from src.ofdm.lib_archived.modulation import QAM16

qam = QAM16()
test_bits = np.array([0,1,0,0, 0,0,1,0, 1,1,0,1, 0,1,0,1], dtype=np.uint8)

print('=== INPUT ===')
print('Bits:', test_bits)
print('Groups: ', end='')
for i in range(0, len(test_bits), 4):
    print(f'{tuple(test_bits[i:i+4])} ', end='')
print()

print('\n=== MODULATION ===')
symbols = []
for i in range(0, len(test_bits), 4):
    key = (int(test_bits[i]), int(test_bits[i+1]), int(test_bits[i+2]), int(test_bits[i+3]))
    sym = qam.bit_to_symbol[key]
    symbols.append(sym)
    print(f'Bits {key} -> Symbol {sym:.4f}')

symbols = np.array(symbols, dtype=np.complex64)

print('\n=== DEMODULATION ===')
for i, sym in enumerate(symbols):
    mag = np.abs(sym)
    scale_factor = np.sqrt(1.0 / (mag**2 + 1e-10))
    scaled = sym * scale_factor
    distances = np.abs(scaled - qam.constellation)
    nearest_idx = np.argmin(distances)
    b3, b2, b1, b0 = qam.symbol_to_bits[nearest_idx]
    print(f'Symbol {i}: {sym:.4f} (idx={nearest_idx}) -> ({b3},{b2},{b1},{b0})')
    
    expected_bit_group = test_bits[i*4:(i+1)*4]
    actual_bit_group = (b3, b2, b1, b0)
    match = tuple(expected_bit_group) == actual_bit_group
    status = "OK" if match else "FAIL"
    print(f'  Expected: {tuple(expected_bit_group)}, Got: {actual_bit_group} [{status}]')
