"""
Forward Error Correction (FEC) for OFDM Headers
Uses Hamming codes to protect the 4-byte payload length header
"""

import numpy as np
from typing import Tuple

class HeaderFEC:
    """
    FEC encoder/decoder for OFDM packet headers.
    
    Converts 4-byte payload length into protected bits using Hamming codes.
    Provides error correction capability.
    """
    
    @staticmethod
    def hamming_7_4_encode(data_bits: np.ndarray) -> np.ndarray:
        """
        Hamming(7,4) encode: Takes 4 data bits, produces 7 bits with parity.
        Can correct 1 bit error and detect 2 bit errors.
        
        Args:
            data_bits: Array of 4 bits
            
        Returns:
            Array of 7 encoded bits
        """
        if len(data_bits) != 4:
            raise ValueError("Hamming(7,4) requires exactly 4 data bits")
        
        d1, d2, d3, d4 = data_bits
        
        # Parity bits
        p1 = (d1 + d2 + d4) % 2      # p1 covers positions 1,3,5,7
        p2 = (d1 + d3 + d4) % 2      # p2 covers positions 2,3,6,7
        p3 = (d2 + d3 + d4) % 2      # p3 covers positions 4,5,6,7
        
        # Encoded bits: p1, p2, d1, p3, d2, d3, d4
        encoded = np.array([p1, p2, d1, p3, d2, d3, d4], dtype=np.uint8)
        return encoded
    
    @staticmethod
    def hamming_7_4_decode(encoded_bits: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Hamming(7,4) decode: Takes 7 bits, recovers 4 data bits and corrects 1 error.
        
        Args:
            encoded_bits: Array of 7 bits (possibly with 1 error)
            
        Returns:
            Tuple of (4 recovered data bits, error_position or 0 if no error)
        """
        if len(encoded_bits) != 7:
            raise ValueError("Hamming(7,4) decode requires exactly 7 bits")
        
        # Received bits
        r = encoded_bits.copy()
        
        # Calculate syndrome bits (error detection)
        s1 = (r[0] + r[2] + r[4] + r[6]) % 2      # Check p1
        s2 = (r[1] + r[2] + r[5] + r[6]) % 2      # Check p2
        s3 = (r[3] + r[4] + r[5] + r[6]) % 2      # Check p3
        
        # Syndrome tells us error position (0 = no error, 1-7 = error at position)
        error_pos = (s3 << 2) | (s2 << 1) | s1
        
        # Correct the error if found
        if error_pos > 0:
            r[error_pos - 1] = 1 - r[error_pos - 1]  # Flip the erroneous bit
        
        # Extract data bits (positions 2, 4, 5, 6 in 0-indexed)
        d1 = r[2]
        d2 = r[4]
        d3 = r[5]
        d4 = r[6]
        
        data_bits = np.array([d1, d2, d3, d4], dtype=np.uint8)
        return data_bits, error_pos
    
    @staticmethod
    def encode_header(payload_len: int) -> np.ndarray:
        """
        Encode 4-byte payload length header with Hamming FEC.
        
        Converts the 32-bit payload length into 56 bits (8 Hamming(7,4) blocks).
        
        Args:
            payload_len: Integer payload length (0-4294967295)
            
        Returns:
            Array of 56 encoded bits
        """
        # Convert to 4 bytes (big-endian)
        payload_bytes = payload_len.to_bytes(4, 'big')
        
        # Convert bytes to bits
        header_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
        
        # Encode each 4-bit chunk with Hamming(7,4)
        encoded_bits = []
        for i in range(0, 32, 4):
            chunk = header_bits[i:i+4]
            encoded_chunk = HeaderFEC.hamming_7_4_encode(chunk)
            encoded_bits.extend(encoded_chunk)
        
        return np.array(encoded_bits, dtype=np.uint8)
    
    @staticmethod
    def decode_header(encoded_bits: np.ndarray) -> Tuple[int, int]:
        """
        Decode FEC-protected header and recover payload length with error correction.
        
        Decodes 56 bits back to 32-bit payload length.
        Can correct up to 8 single-bit errors (one per Hamming block).
        
        Args:
            encoded_bits: Array of 56 encoded bits (may contain errors)
            
        Returns:
            Tuple of (recovered_payload_len, total_errors_corrected)
        """
        if len(encoded_bits) != 56:
            raise ValueError(f"Expected 56 encoded bits, got {len(encoded_bits)}")
        
        # Decode each 7-bit chunk with Hamming(7,4)
        decoded_bits = []
        total_errors = 0
        
        for i in range(0, 56, 7):
            chunk = encoded_bits[i:i+7]
            decoded_chunk, error_pos = HeaderFEC.hamming_7_4_decode(chunk)
            decoded_bits.extend(decoded_chunk)
            if error_pos > 0:
                total_errors += 1
        
        # Convert bits back to bytes
        decoded_bits = np.array(decoded_bits, dtype=np.uint8)
        decoded_bytes = np.packbits(decoded_bits).tobytes()
        
        # Convert bytes back to integer (big-endian)
        payload_len = int.from_bytes(decoded_bytes, 'big')
        
        return payload_len, total_errors


class HeaderFECWrapper:
    """Wrapper to make header FEC optional in transceiver."""
    
    def __init__(self, use_fec: bool = True):
        """
        Args:
            use_fec: Whether to use FEC protection on headers
        """
        self.use_fec = use_fec
    
    def encode_header(self, payload_len: int) -> np.ndarray:
        """Encode header, optionally with FEC."""
        if self.use_fec:
            return HeaderFEC.encode_header(payload_len)
        else:
            # Raw header: 4 bytes to bits
            payload_bytes = payload_len.to_bytes(4, 'big')
            return np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
    
    def decode_header(self, bits: np.ndarray) -> Tuple[int, int]:
        """
        Decode header, optionally with FEC.
        
        Returns:
            Tuple of (payload_len, errors_corrected)
        """
        if self.use_fec:
            # Expect 56 bits (FEC-encoded)
            if len(bits) < 56:
                # Not enough bits, pad with zeros
                bits = np.pad(bits, (0, 56 - len(bits)), 'constant')
            elif len(bits) > 56:
                bits = bits[:56]
            
            return HeaderFEC.decode_header(bits)
        else:
            # Raw header: 32 bits to bytes
            if len(bits) < 32:
                bits = np.pad(bits, (0, 32 - len(bits)), 'constant')
            elif len(bits) > 32:
                bits = bits[:32]
            
            payload_bytes = np.packbits(bits).tobytes()
            payload_len = int.from_bytes(payload_bytes, 'big')
            return payload_len, 0


# Example usage and testing
if __name__ == '__main__':
    print("="*70)
    print("  HEADER FEC TEST")
    print("="*70)
    
    # Test payload length
    test_payload_len = 27556
    
    print(f"\n1. Original payload length: {test_payload_len}")
    
    # Encode
    encoded = HeaderFEC.encode_header(test_payload_len)
    print(f"2. Encoded to {len(encoded)} bits (Hamming protected)")
    print(f"   Original: 32 bits → Protected: 56 bits (14 bits overhead)")
    
    # Introduce errors
    corrupted = encoded.copy()
    error_positions = [5, 15, 32, 50]  # Simulate 4 bit errors
    for pos in error_positions:
        corrupted[pos] = 1 - corrupted[pos]
    
    print(f"\n3. Introduced {len(error_positions)} bit errors at positions: {error_positions}")
    
    # Decode
    recovered_len, errors_corrected = HeaderFEC.decode_header(corrupted)
    
    print(f"4. Decoded recovered length: {recovered_len}")
    print(f"5. Errors corrected: {errors_corrected}/{len(error_positions)}")
    print(f"\n{'✓ SUCCESS' if recovered_len == test_payload_len else '✗ FAILED'}: "
          f"{'Exact match!' if recovered_len == test_payload_len else 'Mismatch'}")
    
    # Test with wrapper
    print("\n" + "="*70)
    print("  WRAPPER TEST")
    print("="*70)
    
    wrapper_fec = HeaderFECWrapper(use_fec=True)
    wrapper_raw = HeaderFECWrapper(use_fec=False)
    
    print(f"\nWith FEC: {len(wrapper_fec.encode_header(test_payload_len))} bits")
    print(f"Without FEC: {len(wrapper_raw.encode_header(test_payload_len))} bits")
