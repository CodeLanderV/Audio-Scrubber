
import numpy as np
from abc import ABC, abstractmethod

class Modulator(ABC):
    """Abstract Base Class for Modulation Schemes."""
    
    @abstractmethod
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Convert bits to complex symbols."""
        pass
        
    @abstractmethod
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """Convert complex symbols to bits."""
        pass
    
    @property
    @abstractmethod
    def bits_per_symbol(self) -> int:
        pass

class QPSK(Modulator):
    """
    QPSK Modulation and Demodulation.
    Maps bits to complex symbols and vice versa.
    """
    
    # Mapping table: (bit1, bit0) -> complex symbol
    # Matches standard QPSK: 00->1+j, 01->-1+j, 11->-1-j, 10->1-j
    MAP = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 1): -1 - 1j,
        (1, 0): 1 - 1j,
    }
    
    @property
    def bits_per_symbol(self) -> int:
        return 2
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Convert bits (0/1) to QPSK symbols."""
        # Ensure even number of bits
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)
            
        symbols = []
        for i in range(0, len(bits), 2):
            key = (int(bits[i]), int(bits[i+1]))
            symbols.append(self.MAP[key])
            
        return np.array(symbols, dtype=np.complex64)
    
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        Aggressive soft-decision demodulation: find nearest constellation point with no clipping.
        """
        bits = []
        constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j])
        constellation_norm = constellation / np.sqrt(2)
        
        for sym in symbols:
            mag = np.abs(sym)
            if mag < 1e-6:
                bits.extend([0, 0])
                continue
            
            # Aggressive scaling: boost ALL signals to target power
            target_power = 2.0
            scale_factor = np.sqrt(target_power / (mag**2 + 1e-10))
            # NO CLIPPING - allow full gain
            
            scaled_sym = sym * scale_factor
            norm_sym = scaled_sym / (np.abs(scaled_sym) + 1e-10)
            
            # Find nearest constellation point
            distances = np.abs(norm_sym - constellation_norm)
            nearest_idx = np.argmin(distances)
            
            bit_map = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
            b0, b1 = bit_map[nearest_idx]
            bits.extend([b0, b1])
        
        return np.array(bits, dtype=np.uint8)


class QAM16(Modulator):
    """
    16-QAM Modulation and Demodulation.
    Maps 4 bits to 16 complex symbols arranged in a 4x4 grid.
    
    Bit-to-symbol mapping: (b3,b2,b1,b0) where b3,b2 map to I-channel, b1,b0 to Q-channel
    Uses Gray-coded constellation for minimum bit errors.
    """
    
    def __init__(self):
        """Initialize 16-QAM constellation with Gray coding."""
        # Gray-coded 4x4 grid for 16-QAM
        # I-channel: -3, -1, +1, +3 (normalized)
        # Q-channel: -3, -1, +1, +3 (normalized)
        i_vals = np.array([-3, -1, 1, 3], dtype=np.float32)
        q_vals = np.array([-3, -1, 1, 3], dtype=np.float32)
        
        # Normalize to unit average power
        # Each point has power: (i^2 + q^2) / 2, average across 16 points
        norm_factor = np.sqrt(10.0)  # (9+1+1+9) * 2 / 16 = 10
        
        # Gray-coded mapping for 16-QAM (in constellation index order)
        gray_codes = [
            (0,0,0,0), (0,0,0,1), (0,0,1,1), (0,0,1,0),
            (0,1,1,0), (0,1,1,1), (0,1,0,1), (0,1,0,0),
            (1,1,0,0), (1,1,0,1), (1,1,1,1), (1,1,1,0),
            (1,0,1,0), (1,0,1,1), (1,0,0,1), (1,0,0,0),
        ]
        
        # Build constellation in Gray-code order
        self.constellation = []
        self.symbol_to_bits = {}
        self.bit_to_symbol = {}
        
        for idx, bits in enumerate(gray_codes):
            # Bits (b3, b2, b1, b0) map to I and Q
            # I-channel uses bits (b3, b2): 00->-3, 01->-1, 11->+1, 10->+3 (Gray code order for 2 bits)
            # Q-channel uses bits (b1, b0): 00->-3, 01->-1, 11->+1, 10->+3 (Gray code order for 2 bits)
            i_bit_pair = (bits[0], bits[1])  # (b3, b2)
            q_bit_pair = (bits[2], bits[3])  # (b1, b0)
            
            # Map Gray code to index: (0,0)->0, (0,1)->1, (1,1)->2, (1,0)->3
            i_idx = {(0,0): 0, (0,1): 1, (1,1): 2, (1,0): 3}[i_bit_pair]
            q_idx = {(0,0): 0, (0,1): 1, (1,1): 2, (1,0): 3}[q_bit_pair]
            
            i_val = i_vals[i_idx]
            q_val = q_vals[q_idx]
            
            sym = (i_val + 1j * q_val) / norm_factor
            self.constellation.append(sym)
            
            # Create mappings
            self.symbol_to_bits[idx] = bits
            self.bit_to_symbol[bits] = sym
        
        self.constellation = np.array(self.constellation, dtype=np.complex64)
    
    @property
    def bits_per_symbol(self) -> int:
        return 4
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Convert bits (0/1) to 16-QAM symbols."""
        # Ensure multiple of 4
        remainder = len(bits) % 4
        if remainder != 0:
            bits = np.append(bits, np.zeros(4 - remainder, dtype=np.uint8))
        
        symbols = []
        for i in range(0, len(bits), 4):
            key = (int(bits[i]), int(bits[i+1]), int(bits[i+2]), int(bits[i+3]))
            symbols.append(self.bit_to_symbol[key])
        
        return np.array(symbols, dtype=np.complex64)
    
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        Soft-decision demodulation: find nearest 16-QAM constellation point.
        Note: 16-QAM does NOT have uniform magnitude, so we don't apply uniform scaling.
        For channels with magnitude distortion, nearest-neighbor detection is most robust.
        """
        bits = []
        
        for sym in symbols:
            # For 16-QAM, don't scale - constellation has variable magnitudes
            # Just find the nearest constellation point directly
            if np.abs(sym) < 1e-6:
                bits.extend([0, 0, 0, 0])
                continue
            
            # Find nearest constellation point (no scaling for 16-QAM)
            distances = np.abs(sym - self.constellation)
            nearest_idx = np.argmin(distances)
            
            b3, b2, b1, b0 = self.symbol_to_bits[nearest_idx]
            bits.extend([b3, b2, b1, b0])
        
        return np.array(bits, dtype=np.uint8)

