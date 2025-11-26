import numpy as np
from src.ofdm.lib_archived.modulation import QPSK, QAM16

print("=" * 80)
print("TEST 1: BIT PACKING CHECK")
print("=" * 80)

original_bytes = b"\x42\x69\xAA"
print(f"\nOriginal bytes: {original_bytes.hex()} = {list(original_bytes)}")

bits_packed = np.unpackbits(np.frombuffer(original_bytes, dtype=np.uint8))
print(f"Bits (MSB first): {bits_packed}")

repacked_bytes = np.packbits(bits_packed).tobytes()
print(f"Repacked bytes: {repacked_bytes.hex()} = {list(repacked_bytes)}")

if original_bytes == repacked_bytes:
    print("✅ PACKING/UNPACKING IS CORRECT")
else:
    print("❌ FAILED")

print("\n" + "=" * 80)
print("TEST 2: QPSK ROUND-TRIP")
print("=" * 80)

qpsk = QPSK()
print(f"\nInput bits: {bits_packed}")

symbols = qpsk.modulate(bits_packed)
print(f"Modulated to {len(symbols)} symbols")

recovered_bits = qpsk.demodulate(symbols)
print(f"Recovered bits: {recovered_bits}")

matches = np.sum(bits_packed == recovered_bits)
print(f"Match: {matches}/{len(bits_packed)} = {100*matches/len(bits_packed):.1f}%")

if np.array_equal(bits_packed, recovered_bits):
    print("✅ QPSK ROUND-TRIP CORRECT")
else:
    print("❌ MISMATCH")

print("\n" + "=" * 80)
print("TEST 3: BYTE RECOVERY")
print("=" * 80)

original = b"Hello"
bits = np.unpackbits(np.frombuffer(original, dtype=np.uint8))
print(f"Original: {original}")

symbols = qpsk.modulate(bits)
recovered_bits = qpsk.demodulate(symbols)
recovered = np.packbits(recovered_bits[:len(bits)]).tobytes()

print(f"Recovered: {recovered}")
if original == recovered:
    print("✅ BYTE RECOVERY CORRECT")
else:
    print("❌ FAILED")
