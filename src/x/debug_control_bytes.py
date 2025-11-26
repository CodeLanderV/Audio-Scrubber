"""Debug what control_bytes contains"""
import sys
import numpy as np
sys.path.append('src')
sys.path.append('.')

from inference.TxRx.ofdm_modulation import OFDM_Modulation

mod = OFDM_Modulation(use_ai=False, passthrough=True, modulation='qpsk')

test_data = b'HELLO'
print(f"Transmit data: {test_data.hex()}")

tx_waveform = mod.modulate(test_data)

# Demodulate
result = mod.demodulate(tx_waveform)

print(f"\nResult keys: {result.keys()}")
print(f"Control data: {result['control_data'].hex() if result['control_data'] else 'None'}")
print(f"AI data: {result['data'].hex() if result['data'] else 'None'}")
print(f"Stats: {result['stats']}")

# What's in self.original_bits?
print(f"\nOriginal bits length: {len(mod.original_bits) if mod.original_bits is not None else 'None'}")
if mod.original_bits is not None:
    header_bits = mod.original_bits[:32]
    print(f"Header bits: {header_bits}")
    print(f"Header as int: {int(''.join(map(str, header_bits)), 2)}")

print(f"\nExpected: b'{test_data.decode()}'")
if result['control_data']:
    print(f"Got:      b'{result['control_data'][:5].decode() if len(result['control_data']) >= 5 else 'SHORT'}'")
