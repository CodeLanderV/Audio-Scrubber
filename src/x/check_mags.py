from src.ofdm.lib_archived.modulation import QAM16
import numpy as np

qam = QAM16()

print('Constellation magnitudes:')
magnitudes = []
for idx in range(16):
    bits = qam.symbol_to_bits[idx]
    sym = qam.constellation[idx]
    mag = np.abs(sym)
    magnitudes.append(mag)
    print(f'{idx:2d}: {bits} mag={mag:.6f}')

avg_mag = np.mean(magnitudes)
print(f'\nAverage magnitude: {avg_mag:.6f}')
print(f'Min magnitude: {np.min(magnitudes):.6f}')
print(f'Max magnitude: {np.max(magnitudes):.6f}')
print(f'Should scale to: {avg_mag:.6f}')
