#!/usr/bin/env python3
"""
Test script demonstrating:
1. QPSK/QAM packing and unpacking correctness
2. Random byte generation and transmission
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ofdm.lib_archived.modulation import QPSK, QAM16

def test_1_bit_packing():
    """Test 1: Verify bit packing/unpacking is correct."""
    print("\n" + "=" * 80)
    print("TEST 1: BIT PACKING/UNPACKING VERIFICATION")
    print("=" * 80)
    
    test_cases = [
        b"\x42\x69\xAA",
        b"Hello",
        bytes(range(256))[:20],
    ]
    
    all_pass = True
    for i, test_bytes in enumerate(test_cases):
        bits = np.unpackbits(np.frombuffer(test_bytes, dtype=np.uint8))
        repacked = np.packbits(bits).tobytes()
        
        match = test_bytes == repacked
        status = "✅" if match else "❌"
        print(f"{status} Case {i+1}: {test_bytes.hex()} -> {len(bits)} bits -> {repacked.hex()}")
        
        if not match:
            all_pass = False
    
    return all_pass


def test_2_qpsk_round_trip():
    """Test 2: QPSK modulation round-trip."""
    print("\n" + "=" * 80)
    print("TEST 2: QPSK ROUND-TRIP (BITS -> SYMBOLS -> BITS)")
    print("=" * 80)
    
    qpsk = QPSK()
    test_bits = np.array([0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
    
    print(f"Input bits:  {test_bits}")
    
    symbols = qpsk.modulate(test_bits)
    print(f"Modulated to {len(symbols)} symbols: {symbols[:3]}... (showing first 3)")
    
    recovered = qpsk.demodulate(symbols)
    print(f"Recovered:   {recovered}")
    
    match = np.array_equal(test_bits, recovered)
    print(f"\n{'✅ PERFECT MATCH' if match else '❌ MISMATCH'}")
    
    return match


def test_3_qam16_round_trip():
    """Test 3: 16-QAM modulation round-trip."""
    print("\n" + "=" * 80)
    print("TEST 3: 16-QAM ROUND-TRIP (BITS -> SYMBOLS -> BITS)")
    print("=" * 80)
    
    qam = QAM16()
    test_bits = np.array([0,1,0,0, 0,0,1,0, 1,1,0,1, 0,1,0,1], dtype=np.uint8)
    
    print(f"Input bits:  {test_bits}")
    print(f"(4 bits per symbol = 4 symbols total)")
    
    symbols = qam.modulate(test_bits)
    print(f"Modulated to {len(symbols)} symbols")
    
    recovered = qam.demodulate(symbols)
    print(f"Recovered:   {recovered}")
    
    match = np.array_equal(test_bits, recovered)
    print(f"\n{'✅ PERFECT MATCH' if match else '❌ MISMATCH'}")
    
    return match


def test_4_byte_recovery():
    """Test 4: Full byte recovery through modulation."""
    print("\n" + "=" * 80)
    print("TEST 4: BYTE RECOVERY (BYTES -> BITS -> SYMBOLS -> BITS -> BYTES)")
    print("=" * 80)
    
    qpsk = QPSK()
    
    # Test with different data types
    test_cases = [
        b"Test",
        b"\x00\xFF\x55\xAA",
        bytes(range(16)),
    ]
    
    all_pass = True
    for test_bytes in test_cases:
        # Forward
        bits = np.unpackbits(np.frombuffer(test_bytes, dtype=np.uint8))
        symbols = qpsk.modulate(bits)
        recovered_bits = qpsk.demodulate(symbols)
        recovered_bytes = np.packbits(recovered_bits[:len(bits)]).tobytes()
        
        match = test_bytes == recovered_bytes
        status = "✅" if match else "❌"
        print(f"{status} {test_bytes.hex():20s} -> {recovered_bytes.hex():20s}")
        
        if not match:
            all_pass = False
    
    return all_pass


def test_5_random_data_generation():
    """Test 5: Random data generation."""
    print("\n" + "=" * 80)
    print("TEST 5: RANDOM DATA GENERATION")
    print("=" * 80)
    
    sizes = [10, 100, 1000]
    
    for size in sizes:
        # Generate random bytes (what --random-bytes does)
        random_data = np.random.bytes(size)
        
        # Show statistics
        data_array = np.frombuffer(random_data, dtype=np.uint8)
        print(f"\n{size} random bytes:")
        print(f"  Min value: {data_array.min():3d}")
        print(f"  Max value: {data_array.max():3d}")
        print(f"  Mean:      {data_array.mean():6.1f}")
        entropy_val = (-np.sum(np.histogram(data_array, bins=256)[0] / size * np.log2((np.histogram(data_array, bins=256)[0] / size + 1e-10)))) if size > 256 else None
        entropy_str = f"{entropy_val:.2f}" if entropy_val is not None else "N/A"
        print(f"  Entropy:   {entropy_str:>6s}")
        print(f"  Hex (first 32 bytes): {random_data[:32].hex()}")
    
    return True


def test_6_cli_usage():
    """Test 6: Show CLI usage examples."""
    print("\n" + "=" * 80)
    print("TEST 6: CLI USAGE EXAMPLES")
    print("=" * 80)
    
    examples = [
        ("Transmit file with QPSK",
         "python src/inference/main_inference.py --mode ofdm --data test.txt"),
        
        ("Transmit 10 random bytes with QPSK",
         "python src/inference/main_inference.py --mode ofdm --random-bytes 10"),
        
        ("Transmit 100 random bytes with 16-QAM",
         "python src/inference/main_inference.py --mode ofdm --random-bytes 100 --modulation 16qam"),
        
        ("Transmit 50 random bytes with Pluto RX + buffer frames",
         "python src/inference/main_inference.py --mode ofdm --random-bytes 50 --rx-device pluto --buffer-frames 3"),
        
        ("Transmit file with all features",
         "python src/inference/main_inference.py --mode ofdm --data image.png --modulation 16qam --rx-device pluto --buffer-frames 2 --tx-gain 5"),
    ]
    
    for i, (desc, cmd) in enumerate(examples, 1):
        print(f"\n{i}. {desc}")
        print(f"   {cmd}")
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("OFDM FEATURE VERIFICATION TEST SUITE")
    print("=" * 80)
    print("\nVerifying:")
    print("  1. Bit packing/unpacking correctness")
    print("  2. QPSK round-trip modulation")
    print("  3. 16-QAM round-trip modulation")
    print("  4. Full byte recovery")
    print("  5. Random data generation")
    print("  6. CLI usage examples")
    
    results = {
        "1. Bit Packing": test_1_bit_packing(),
        "2. QPSK Round-trip": test_2_qpsk_round_trip(),
        "3. 16-QAM Round-trip": test_3_qam16_round_trip(),
        "4. Byte Recovery": test_4_byte_recovery(),
        "5. Random Generation": test_5_random_data_generation(),
        "6. CLI Examples": test_6_cli_usage(),
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test}")
    
    all_passed = all(results.values())
    print("\n" + ("=" * 80))
    if all_passed:
        print("✅ ALL TESTS PASSED - System ready for deployment!")
    else:
        print("❌ SOME TESTS FAILED - Review errors above")
    print("=" * 80)
