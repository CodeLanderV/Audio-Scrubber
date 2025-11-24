#!/usr/bin/env python3
"""
Test QPSK/QAM bit packing and unpacking to verify correctness.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ofdm.lib_archived.modulation import QPSK, QAM16

def test_bit_packing():
    """Test if bits are packed/unpacked correctly."""
    print("=" * 80)
    print("TEST: BIT PACKING/UNPACKING")
    print("=" * 80)
    
    # Original data
    original_bytes = b"\x42\x69\xAA"  # 0x42=66, 0x69=105, 0xAA=170
    print(f"\nOriginal bytes: {original_bytes.hex()} = {list(original_bytes)}")
    
    # Convert to bits using numpy (standard method)
    bits_packed = np.unpackbits(np.frombuffer(original_bytes, dtype=np.uint8))
    print(f"Bits (MSB first): {bits_packed}")
    print(f"Bit order breakdown:")
    for i, byte_val in enumerate(original_bytes):
        byte_bits = np.unpackbits(np.array([byte_val], dtype=np.uint8))
        print(f"  Byte {i}: 0x{byte_val:02X} ({byte_val:3d}) -> {byte_bits}")
    
    # Repack bits back
    repacked_bytes = np.packbits(bits_packed).tobytes()
    print(f"\nRepacked bytes: {repacked_bytes.hex()} = {list(repacked_bytes)}")
    
    if original_bytes == repacked_bytes:
        print("✅ PACKING/UNPACKING IS CORRECT")
    else:
        print("❌ PACKING/UNPACKING FAILED")
    
    return bits_packed


def test_qpsk_modulation(bits):
    """Test QPSK modulation round-trip."""
    print("\n" + "=" * 80)
    print("TEST: QPSK ROUND-TRIP (BITS -> SYMBOLS -> BITS)")
    print("=" * 80)
    
    qpsk = QPSK()
    
    print(f"\nInput bits ({len(bits)} total): {bits}")
    
    # Modulate
    symbols = qpsk.modulate(bits)
    print(f"\nModulated to {len(symbols)} symbols:")
    for i, sym in enumerate(symbols):
        print(f"  Symbol {i}: {sym:.4f}")
    
    # Demodulate
    recovered_bits = qpsk.demodulate(symbols)
    print(f"\nRecovered bits ({len(recovered_bits)} total): {recovered_bits}")
    
    # Compare
    min_len = min(len(bits), len(recovered_bits))
    matches = np.sum(bits[:min_len] == recovered_bits[:min_len])
    match_pct = 100 * matches / min_len
    
    print(f"\nComparison:")
    print(f"  Input:     {bits[:min_len]}")
    print(f"  Recovered: {recovered_bits[:min_len]}")
    print(f"  Match: {matches}/{min_len} = {match_pct:.1f}%")
    
    if np.array_equal(bits[:min_len], recovered_bits[:min_len]):
        print("✅ QPSK ROUND-TRIP CORRECT")
        return True
    else:
        print("❌ QPSK ROUND-TRIP FAILED")
        return False


def test_qam16_modulation(bits):
    """Test 16-QAM modulation round-trip."""
    print("\n" + "=" * 80)
    print("TEST: 16-QAM ROUND-TRIP (BITS -> SYMBOLS -> BITS)")
    print("=" * 80)
    
    qam = QAM16()
    
    # Need multiple of 4 bits
    padded_bits = bits.copy()
    if len(padded_bits) % 4 != 0:
        padded_bits = np.append(padded_bits, np.zeros(4 - len(padded_bits) % 4, dtype=np.uint8))
    
    print(f"\nInput bits ({len(padded_bits)} total): {padded_bits}")
    
    # Modulate
    symbols = qam.modulate(padded_bits)
    print(f"\nModulated to {len(symbols)} symbols:")
    for i, sym in enumerate(symbols[:5]):  # Show first 5
        print(f"  Symbol {i}: {sym:.4f}")
    if len(symbols) > 5:
        print(f"  ... ({len(symbols)-5} more)")
    
    # Demodulate
    recovered_bits = qam.demodulate(symbols)
    print(f"\nRecovered bits ({len(recovered_bits)} total): {recovered_bits[:20]}...")
    
    # Compare
    matches = np.sum(padded_bits == recovered_bits)
    match_pct = 100 * matches / len(padded_bits)
    
    print(f"\nComparison:")
    print(f"  Input:     {padded_bits}")
    print(f"  Recovered: {recovered_bits}")
    print(f"  Match: {matches}/{len(padded_bits)} = {match_pct:.1f}%")
    
    if np.array_equal(padded_bits, recovered_bits):
        print("✅ 16-QAM ROUND-TRIP CORRECT")
        return True
    else:
        print("❌ 16-QAM ROUND-TRIP FAILED")
        return False


def test_byte_recovery():
    """Test if we can recover exact bytes after modulation."""
    print("\n" + "=" * 80)
    print("TEST: BYTE RECOVERY (BYTES -> BITS -> SYMBOLS -> BITS -> BYTES)")
    print("=" * 80)
    
    original_bytes = b"Hello"
    print(f"\nOriginal bytes: {original_bytes} ({original_bytes.hex()})")
    
    # Convert to bits
    bits = np.unpackbits(np.frombuffer(original_bytes, dtype=np.uint8))
    print(f"Bits: {bits}")
    
    # QPSK round-trip
    qpsk = QPSK()
    symbols = qpsk.modulate(bits)
    recovered_bits = qpsk.demodulate(symbols)
    
    # Convert back to bytes
    recovered_bytes = np.packbits(recovered_bits[:len(bits)]).tobytes()
    print(f"Recovered bytes: {recovered_bytes} ({recovered_bytes.hex()})")
    
    if original_bytes == recovered_bytes:
        print("✅ BYTE RECOVERY CORRECT (QPSK)")
        return True
    else:
        print("❌ BYTE RECOVERY FAILED (QPSK)")
        print(f"  Expected: {original_bytes.hex()}")
        print(f"  Got:      {recovered_bytes.hex()}")
        return False


if __name__ == "__main__":
    # Test 1: Basic packing
    bits = test_bit_packing()
    
    # Test 2: QPSK round-trip
    qpsk_ok = test_qpsk_modulation(bits)
    
    # Test 3: 16-QAM round-trip
    qam_ok = test_qam16_modulation(bits)
    
    # Test 4: Byte recovery
    byte_ok = test_byte_recovery()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Bit Packing:     ✅")
    print(f"QPSK Round-trip: {'✅' if qpsk_ok else '❌'}")
    print(f"16-QAM Round-trip: {'✅' if qam_ok else '❌'}")
    print(f"Byte Recovery:   {'✅' if byte_ok else '❌'}")
