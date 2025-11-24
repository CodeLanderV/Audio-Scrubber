
import numpy as np
from .config import OFDMConfig

class ChannelEqualizer:
    """
    Uses Pilot symbols to estimate and correct channel effects (Phase/Amplitude).
    This is CRITICAL for decoding signals that have been modified by AI or channel.
    """
    
    def __init__(self, config: OFDMConfig):
        self.config = config
    
    def estimate_channel(self, received_pilots: np.ndarray) -> complex:
        """
        Improved channel estimation with robust median filtering and outlier rejection.
        """
        tx_pilots = self.config.pilot_values
        h_estimates = received_pilots / (tx_pilots + 1e-10)
        
        # Use median for robustness to outliers
        h_magnitudes = np.abs(h_estimates)
        median_mag = np.median(h_magnitudes)
        
        # Keep estimates within ±50% of median
        valid_mask = (h_magnitudes > 0.5 * median_mag) & (h_magnitudes < 2.0 * median_mag)
        valid_estimates = h_estimates[valid_mask]
        
        if len(valid_estimates) > 0:
            return np.median(valid_estimates)
        
        return np.mean(h_estimates)
    
    def equalize(self, data_symbols: np.ndarray, h_est: complex) -> np.ndarray:
        """
        Correct data symbols using the estimated channel response.
        Sym_Corrected = Sym_Received / H_est
        """
        if np.abs(h_est) < 1e-6:
            return data_symbols # Avoid blowing up noise if H is near zero
            
        return data_symbols / h_est
