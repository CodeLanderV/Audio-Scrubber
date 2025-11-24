"""
================================================================================
TEST BER IMPROVEMENTS - Comprehensive Validation
================================================================================

Tests the improvements made:
1. Soft-decision QPSK demodulation
2. Improved bit extraction
3. Better receiver error handling
4. Full payload transmission
"""

import sys
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ofdm.lib_archived.transceiver import OFDMTransmitter, OFDMReceiver
from src.ofdm.lib_archived.config import OFDMConfig
from src.ofdm.lib_archived.modulation import QPSK


def add_awgn(signal, snr_db):
    """Add AWGN to signal at specified SNR."""
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise


def test_transmission(payload_size, snr_db, test_name):
    """Test transmission at given conditions."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    print(f"Payload Size: {payload_size} bytes")
    print(f"SNR: {snr_db} dB")
    
    # Generate test data
    payload_bytes = np.random.bytes(payload_size)
    
    # Transmit
    config = OFDMConfig()
    tx = OFDMTransmitter(config, use_header_fec=True)
    
    print("\nTRANSMIT:")
    waveform, meta = tx.transmit(payload_bytes)
    print(f"   Waveform: {len(waveform)} samples")
    print(f"   Symbols: {meta['num_ofdm_symbols']}")
    print(f"   Total bits: {meta['total_bits']}")
    if 'enhanced_fec' in meta:
        print(f"   Enhanced FEC: {meta['enhanced_fec']}")
    print(f"   TX Power: {10*np.log10(np.mean(np.abs(waveform)**2) + 1e-12):.2f} dB")
    
    # Add noise
    noisy_waveform = add_awgn(waveform, snr_db)
    noisy_power = np.mean(np.abs(noisy_waveform)**2)
    
    # Receive
    rx = OFDMReceiver(config, use_header_fec=True)
    
    print("\nRECEIVE:")
    received_bytes, rx_meta = rx.receive(noisy_waveform)
    
    print(f"   Status: {rx_meta.get('status', 'unknown')}")
    print(f"   Received: {len(received_bytes)} bytes")
    print(f"   Expected: {payload_size} bytes")
    print(f"   Errors corrected: {rx_meta.get('errors_corrected', 0)}")
    
    # Analyze
    print("\nANALYSIS:")
    
    if len(received_bytes) == 0:
        print("   [FAIL] No data received!")
        return {
            'success': False,
            'payload_size': payload_size,
            'snr_db': snr_db,
            'received_bytes': 0,
            'byte_errors': payload_size,
            'ber': 1.0,
            'reason': 'No data received'
        }
    
    # Compare bytes
    match_len = min(len(payload_bytes), len(received_bytes))
    byte_errors = np.sum(
        np.frombuffer(payload_bytes[:match_len], dtype=np.uint8) != 
        np.frombuffer(received_bytes[:match_len], dtype=np.uint8)
    )
    
    # Calculate BER
    bits_sent = match_len * 8
    bits_errors = 0
    for i in range(match_len):
        sent_byte = payload_bytes[i]
        recv_byte = received_bytes[i]
        xor = sent_byte ^ recv_byte
        bits_errors += bin(xor).count('1')
    
    ber = bits_errors / bits_sent if bits_sent > 0 else 0
    
    print(f"   Bytes compared: {match_len}/{payload_size}")
    print(f"   Byte errors: {byte_errors}/{match_len}")
    print(f"   Byte error rate: {100*byte_errors/match_len:.2f}%")
    print(f"   Bit errors: {bits_errors}/{bits_sent}")
    print(f"   BER: {ber:.6f} ({100*ber:.4f}%)")
    
    success = (len(received_bytes) == payload_size and byte_errors == 0)
    
    if success:
        print("\n   [PASS] PERFECT TRANSMISSION!")
    elif byte_errors == 0:
        print(f"\n   [WARN] Partial transmission but no errors in received portion")
    else:
        print(f"\n   [FAIL] Transmission with errors")
    
    return {
        'success': success,
        'payload_size': payload_size,
        'snr_db': snr_db,
        'received_bytes': len(received_bytes),
        'byte_errors': byte_errors,
        'ber': ber,
        'match_len': match_len,
        'expected': payload_size
    }


def main():
    print("="*70)
    print("  BER IMPROVEMENT VALIDATION TESTS")
    print("="*70)
    
    results = []
    
    # Test 1: Small payload at various SNRs
    print("\n\nTEST SERIES 1: Small Payload (10 bytes)")
    print("-" * 70)
    for snr in [10, 5, 0, -5]:
        result = test_transmission(10, snr, f"Small Payload, SNR={snr}dB")
        results.append(result)
    
    # Test 2: Medium payload at various SNRs
    print("\n\nTEST SERIES 2: Medium Payload (100 bytes)")
    print("-" * 70)
    for snr in [10, 5, 0]:
        result = test_transmission(100, snr, f"Medium Payload, SNR={snr}dB")
        results.append(result)
    
    # Test 3: Larger payload
    print("\n\nTEST SERIES 3: Large Payload (256 bytes)")
    print("-" * 70)
    for snr in [15, 10, 5]:
        result = test_transmission(256, snr, f"Large Payload, SNR={snr}dB")
        results.append(result)
    
    # Summary
    print("\n\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    print(f"\n[OK] Successful transmissions: {success_count}/{total_count}")
    
    print("\nDetailed Results:")
    print("-" * 70)
    print(f"{'Payload':<12} {'SNR':<8} {'Status':<10} {'RX Bytes':<12} {'BER':<12}")
    print("-" * 70)
    
    for r in results:
        status = "OK" if r['success'] else "ERR"
        print(f"{r['payload_size']:<12} {r['snr_db']:<8} {status:<10} "
              f"{r['received_bytes']}/{r['expected']:<8} {r['ber']:.6f}")
    
    print("\n" + "="*70)
    
    if success_count == total_count:
        print("[PASS] ALL TESTS PASSED!")
    elif success_count > 0:
        print(f"[PARTIAL] {success_count}/{total_count} tests passed")
    else:
        print("[FAIL] All tests failed - check configuration")


if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility
    main()
