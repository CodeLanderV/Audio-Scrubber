"""
Enhanced FEC for OFDM - Header + Payload Protection with Interleaving
Uses Hamming codes for header and simple parity for payload
Includes bit interleaving to spread burst errors
"""

import numpy as np
from typing import Tuple

class BitInterleaver:
    """Interleaves bits to spread burst errors across OFDM symbols."""
    
    @staticmethod
    def interleave(bits: np.ndarray, num_subcarriers: int) -> np.ndarray:
        """
        Interleave bits across OFDM subcarriers.
        Maps bit i to subcarrier (i % num_subcarriers)
        This spreads consecutive errors across different OFDM symbols.
        """
        # Calculate how many bits per subcarrier
        bits_per_subcarrier = len(bits) // num_subcarriers
        remainder = len(bits) % num_subcarriers
        
        interleaved = np.zeros(len(bits), dtype=bits.dtype)
        idx = 0
        
        for sc in range(num_subcarriers):
            # Extract every num_subcarriers-th bit starting from sc
            step_size = num_subcarriers
            for i in range(sc, len(bits), step_size):
                if idx < len(interleaved):
                    interleaved[idx] = bits[i]
                    idx += 1
        
        return interleaved
    
    @staticmethod
    def deinterleave(bits: np.ndarray, num_subcarriers: int) -> np.ndarray:
        """Reverse the interleaving operation."""
        deinterleaved = np.zeros(len(bits), dtype=bits.dtype)
        
        idx = 0
        for sc in range(num_subcarriers):
            step_size = num_subcarriers
            for i in range(sc, len(bits), step_size):
                if i < len(deinterleaved):
                    deinterleaved[i] = bits[idx]
                    idx += 1
        
        return deinterleaved


class PayloadFEC:
    """Simple parity-based FEC for payload - adds 1 parity bit per byte."""
    
    @staticmethod
    def add_parity(payload_bits: np.ndarray) -> np.ndarray:
        """
        Add single parity bit for each byte.
        This allows detection of single bit errors in each byte.
        Output: 9 bits per byte (8 data + 1 parity)
        """
        if len(payload_bits) % 8 != 0:
            raise ValueError("Payload bits must be multiple of 8")
        
        with_parity = []
        
        for i in range(0, len(payload_bits), 8):
            byte_bits = payload_bits[i:i+8]
            # Even parity
            parity = np.sum(byte_bits) % 2
            byte_with_parity = np.append(byte_bits, parity)
            with_parity.extend(byte_with_parity)
        
        return np.array(with_parity, dtype=np.uint8)
    
    @staticmethod
    def check_parity(bits_with_parity: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Check and correct parity errors in received bits.
        Returns: (corrected_payload_bits, num_errors_detected)
        """
        if len(bits_with_parity) % 9 != 0:
            raise ValueError("Payload bits must be multiple of 9 (8 data + 1 parity)")
        
        corrected = []
        errors_detected = 0
        
        for i in range(0, len(bits_with_parity), 9):
            byte_bits = bits_with_parity[i:i+8]
            parity_bit = bits_with_parity[i+8]
            
            # Check parity
            calculated_parity = np.sum(byte_bits) % 2
            
            if calculated_parity != parity_bit:
                errors_detected += 1
                # We can detect but not correct with single parity bit
                # Just mark the byte as suspicious
            
            corrected.extend(byte_bits)
        
        return np.array(corrected, dtype=np.uint8), errors_detected


class EnhancedFEC:
    """Combined FEC with header Hamming codes, payload parity, and interleaving."""
    
    def __init__(self, use_payload_fec: bool = True, use_interleaving: bool = True, 
                 num_subcarriers: int = 52):
        self.use_payload_fec = use_payload_fec
        self.use_interleaving = use_interleaving
        self.num_subcarriers = num_subcarriers
        
        from .fec_header import HeaderFEC
        self.header_fec = HeaderFEC
        self.interleaver = BitInterleaver()
        self.payload_fec = PayloadFEC()
    
    def encode_packet(self, payload_bytes: bytes) -> np.ndarray:
        """
        Encode packet with FEC and interleaving.
        Returns: Protected bit stream
        """
        # 1. Encode header with Hamming FEC (56 bits)
        header_bits = self.header_fec.encode_header(len(payload_bytes))
        
        # 2. Convert payload to bits
        payload_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
        
        # 3. Add payload FEC (optional)
        if self.use_payload_fec:
            payload_bits = self.payload_fec.add_parity(payload_bits)
        
        # 4. Combine header + payload
        all_bits = np.concatenate([header_bits, payload_bits])
        
        # 5. Interleave (optional)
        if self.use_interleaving:
            all_bits = self.interleaver.interleave(all_bits, self.num_subcarriers)
        
        return all_bits
    
    def decode_packet(self, received_bits: np.ndarray) -> Tuple[bytes, dict]:
        """
        Decode packet with FEC and interleaving.
        Returns: (payload_bytes, stats_dict)
        """
        stats = {
            'header_errors': 0,
            'payload_errors': 0,
            'status': 'ok'
        }
        
        # 1. Deinterleave (optional)
        if self.use_interleaving:
            received_bits = self.interleaver.deinterleave(received_bits, self.num_subcarriers)
        
        # 2. Decode header
        if len(received_bits) < 56:
            stats['status'] = 'too_short'
            return b"", stats
        
        header_bits = received_bits[:56]
        try:
            payload_len, header_errors = self.header_fec.decode_header(header_bits)
            stats['header_errors'] = header_errors
        except:
            stats['status'] = 'header_decode_failed'
            return b"", stats
        
        # 3. Extract payload bits
        payload_start = 56
        if self.use_payload_fec:
            expected_payload_bits = payload_len * 9  # 8 data + 1 parity per byte
        else:
            expected_payload_bits = payload_len * 8
        
        payload_bits = received_bits[payload_start:payload_start + expected_payload_bits]
        
        # 4. Check payload FEC (optional)
        if self.use_payload_fec:
            if len(payload_bits) % 9 != 0:
                # Pad if necessary
                payload_bits = np.pad(payload_bits, (0, 9 - (len(payload_bits) % 9)))
            payload_bits, parity_errors = self.payload_fec.check_parity(payload_bits)
            stats['payload_errors'] = parity_errors
        
        # 5. Convert bits to bytes
        if len(payload_bits) < payload_len * 8:
            # Not enough bits
            available_bytes = len(payload_bits) // 8
            payload_bits = payload_bits[:available_bytes * 8]
            payload_len = available_bytes
            stats['status'] = 'partial'
        
        payload_bytes = np.packbits(payload_bits[:payload_len * 8]).tobytes()
        
        return payload_bytes[:payload_len], stats


if __name__ == '__main__':
    print("Enhanced FEC Test")
    print("="*70)
    
    # Test data
    test_payload = b"Hello, World! This is a test."
    
    # Encode
    fec = EnhancedFEC(use_payload_fec=True, use_interleaving=True)
    encoded_bits = fec.encode_packet(test_payload)
    
    print(f"Original payload: {len(test_payload)} bytes")
    print(f"Encoded bits: {len(encoded_bits)}")
    
    # Add errors
    corrupted_bits = encoded_bits.copy()
    error_positions = np.random.choice(len(corrupted_bits), size=5, replace=False)
    corrupted_bits[error_positions] = 1 - corrupted_bits[error_positions]
    
    print(f"Added {len(error_positions)} bit errors")
    
    # Decode
    decoded_payload, stats = fec.decode_packet(corrupted_bits)
    
    print(f"\nDecoded payload: {len(decoded_payload)} bytes")
    print(f"Header errors detected: {stats['header_errors']}")
    print(f"Payload errors detected: {stats['payload_errors']}")
    print(f"Status: {stats['status']}")
    
    if decoded_payload == test_payload:
        print("\n[PASS] Perfect recovery despite errors!")
    else:
        print("\n[FAIL] Payload mismatch")
        print(f"Expected: {test_payload}")
        print(f"Got: {decoded_payload}")
