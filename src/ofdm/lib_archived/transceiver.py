
import numpy as np
from typing import Tuple, Dict, Any
from .config import OFDMConfig
from .modulation import QPSK, QAM16
from .core import OFDMEngine
from .receiver import ChannelEqualizer
from .fec_header import HeaderFECWrapper
from .fec_enhanced import EnhancedFEC

class OFDMTransmitter:
    def __init__(self, config: OFDMConfig = None, use_header_fec: bool = True, use_enhanced_fec: bool = True):
        self.config = config if config else OFDMConfig()
        
        # Select modulation scheme
        if self.config.modulation_scheme.lower() == "16qam":
            self.modulator = QAM16()
            print(f"🔧 Using 16-QAM modulation (4 bits/symbol)")
        else:
            self.modulator = QPSK()
            print(f"🔧 Using QPSK modulation (2 bits/symbol)")
        
        self.engine = OFDMEngine(self.config)
        self.header_fec = HeaderFECWrapper(use_fec=use_header_fec)
        
        # Use enhanced FEC (with payload protection and interleaving)
        self.use_enhanced_fec = use_enhanced_fec
        if self.use_enhanced_fec:
            # Use enhanced FEC WITHOUT interleaving for now (it causes misalignment)
            self.fec = EnhancedFEC(use_payload_fec=True, use_interleaving=False, 
                                  num_subcarriers=self.config.data_subcarriers_count)
        
    def transmit(self, payload_bytes: bytes) -> Tuple[np.ndarray, Dict]:
        """
        Full Transmission Chain: Bytes -> Bits -> QPSK -> OFDM Waveform
        With optional FEC protection on headers and payload
        """
        # 1. Prepare Packet (FEC-Protected Header + Data)
        if self.use_enhanced_fec:
            bits = self.fec.encode_packet(payload_bytes)
            print(f"TX: Enhanced FEC - Payload {len(payload_bytes)} bytes -> {len(bits)} bits")
        else:
            header_bits = self.header_fec.encode_header(len(payload_bytes))
            payload_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
            bits = np.concatenate([header_bits, payload_bits])
            print(f"TX: Header bits={len(header_bits)}, Payload bits={len(payload_bits)}, Total={len(bits)}")
        
        # 2. Bits to Symbols
        symbols = self.modulator.modulate(bits)
        
        # 3. Pad to fill OFDM symbols
        capacity = self.config.data_subcarriers_count
        num_ofdm_symbols = int(np.ceil(len(symbols) / capacity))
        total_capacity = num_ofdm_symbols * capacity
        
        if len(symbols) < total_capacity:
            padding = np.zeros(total_capacity - len(symbols), dtype=np.complex64)
            symbols = np.concatenate([symbols, padding])
            
        # 4. Generate Waveform
        waveform = []
        for i in range(num_ofdm_symbols):
            chunk = symbols[i*capacity : (i+1)*capacity]
            symbol = self.engine.generate_symbol(chunk)
            waveform.extend(symbol)
            
        waveform = np.array(waveform, dtype=np.complex64)
        
        # 5. Power Normalization (Match Training Data)
        current_pwr = np.mean(np.abs(waveform)**2)
        
        scale = np.sqrt(self.config.target_power / current_pwr) if current_pwr > 0 else 1.0
        waveform *= scale
        
        meta = {
            "payload_len": len(payload_bytes),
            "num_ofdm_symbols": num_ofdm_symbols,
            "raw_power": current_pwr,
            "scale_factor": scale,
            "total_bits": len(bits),
            "enhanced_fec": self.use_enhanced_fec
        }
        return waveform, meta

class OFDMReceiver:
    def __init__(self, config: OFDMConfig = None, use_header_fec: bool = True, use_enhanced_fec: bool = True):
        self.config = config if config else OFDMConfig()
        
        # Select modulation scheme
        if self.config.modulation_scheme.lower() == "16qam":
            self.modulator = QAM16()
            print(f"🔧 Using 16-QAM modulation (4 bits/symbol)")
        else:
            self.modulator = QPSK()
            print(f"🔧 Using QPSK modulation (2 bits/symbol)")
        
        self.engine = OFDMEngine(self.config)
        self.equalizer = ChannelEqualizer(self.config)
        self.header_fec = HeaderFECWrapper(use_fec=use_header_fec)
        
        # Use enhanced FEC (with payload protection and interleaving)
        self.use_enhanced_fec = use_enhanced_fec
        if self.use_enhanced_fec:
            # Use enhanced FEC WITHOUT interleaving for now (it causes misalignment)
            self.fec = EnhancedFEC(use_payload_fec=True, use_interleaving=False,
                                  num_subcarriers=self.config.data_subcarriers_count)
        
    def receive(self, waveform: np.ndarray) -> Tuple[bytes, Dict]:
        """
        Full Receiver Chain: Waveform -> FFT -> Equalize -> QPSK -> Bits -> Bytes
        With FEC header error correction and optional payload FEC
        """
        # 0. SYNCHRONIZATION: Detect packet start (Energy detection)
        synced_waveform = self._synchronize_packet(waveform)
        if synced_waveform is None or len(synced_waveform) == 0:
            print(f"⚠️  No packet detected via synchronization. Using raw waveform.")
            synced_waveform = waveform
        
        # 1. Split into symbols
        sym_len = self.config.symbol_len
        num_symbols = len(synced_waveform) // sym_len
        
        print(f"RX: Processing {num_symbols} symbols (sym_len={sym_len})")
        
        all_data_symbols = []
        
        for i in range(num_symbols):
            # Extract time domain symbol
            time_sym = synced_waveform[i*sym_len : (i+1)*sym_len]
            
            # Core Processing (CP Removal + FFT)
            raw_data, raw_pilots = self.engine.process_received_symbol(time_sym)
            
            # Channel Estimation & Equalization
            h_est = self.equalizer.estimate_channel(raw_pilots)
            corrected_data = self.equalizer.equalize(raw_data, h_est)
            
            all_data_symbols.extend(corrected_data)
            
        all_data_symbols = np.array(all_data_symbols)
        
        print(f"   Total symbols collected: {len(all_data_symbols)}")
        
        # 2. Demodulate - FULL constellation demodulation (not just slicing)
        bits = self.modulator.demodulate(all_data_symbols)
        
        print(f"   Demodulated to {len(bits)} bits")
        
        # 3. Decode using Enhanced FEC or Legacy path
        try:
            if self.use_enhanced_fec:
                print("   Using Enhanced FEC (payload protected + interleaved)")
                payload_bytes, fec_stats = self.fec.decode_packet(bits)
                
                print(f"   Header errors corrected: {fec_stats['header_errors']}")
                print(f"   Payload errors detected: {fec_stats['payload_errors']}")
                print(f"   Status: {fec_stats['status']}")
                
                return payload_bytes, {
                    "status": fec_stats['status'],
                    "len": len(payload_bytes),
                    "header_errors": fec_stats['header_errors'],
                    "payload_errors": fec_stats['payload_errors']
                }
            else:
                # Legacy path
                print("   Using Legacy FEC (header only)")
                return self._receive_legacy(bits)
            
        except Exception as e:
            print(f"❌ Decoding exception: {e}")
            import traceback
            traceback.print_exc()
            return b"", {"error": str(e)}
    
    def _synchronize_packet(self, waveform: np.ndarray) -> np.ndarray:
        """
        SYNCHRONIZATION: Detect packet start using energy detection.
        This is CRITICAL for finding where the OFDM packet actually starts.
        
        Args:
            waveform: Raw received IQ samples
            
        Returns:
            Synchronized waveform starting from packet boundary
        """
        try:
            # Compute signal power
            power = np.abs(waveform) ** 2
            
            if len(power) < self.config.symbol_len:
                return waveform
            
            # Compute threshold: 5x the mean of first 1000 samples (noise floor)
            noise_floor = np.mean(power[:min(1000, len(power))])
            threshold = 5 * noise_floor
            
            # Find first sample above threshold
            above_threshold = np.where(power > threshold)[0]
            
            if len(above_threshold) == 0:
                return waveform
            
            # Align to nearest symbol boundary
            start_idx = above_threshold[0]
            symbol_len = self.config.symbol_len
            aligned_start = (start_idx // symbol_len) * symbol_len
            
            if aligned_start >= len(waveform):
                return waveform
            
            synced_waveform = waveform[aligned_start:]
            print(f"📡 SYNC: Packet detected at sample {aligned_start}, Duration: {len(synced_waveform)} samples")
            
            return synced_waveform
            
        except Exception as e:
            print(f"⚠️  Synchronization failed: {e}. Using raw waveform.")
            return waveform
    
    def _receive_legacy(self, bits: np.ndarray) -> Tuple[bytes, Dict]:
        """Legacy reception path for backward compatibility."""
        # Extract header bits
        header_bits_len = 56 if self.header_fec.use_fec else 32
        if len(bits) < header_bits_len:
            print(f"❌ Insufficient bits for header: {len(bits)} < {header_bits_len}")
            
            # Try to extract whatever payload we can
            available_bytes = len(bits) // 8
            if available_bytes > 0:
                payload = np.packbits(bits[:available_bytes*8]).tobytes()
                return payload[:available_bytes], {
                    "status": "header_too_short", 
                    "bits_received": len(bits),
                    "len": available_bytes
                }
            return b"", {"error": "Packet too short", "bits_received": len(bits)}
        
        header_bits = bits[:header_bits_len]
        payload_bits = bits[header_bits_len:]
        
        print(f"   Header bits: {header_bits_len}, Payload bits: {len(payload_bits)}")
        
        # Decode header with FEC
        payload_len, errors_corrected = self.header_fec.decode_header(header_bits)
        
        print(f"   Header decoded: payload_len={payload_len} bytes, errors_corrected={errors_corrected}")
        
        # Sanity check
        MAX_PAYLOAD = 100000
        if payload_len <= 0 or payload_len > MAX_PAYLOAD:
            print(f"⚠️  Invalid payload length in header: {payload_len}")
            
            # Recovery: use available payload
            available_bytes = len(payload_bits) // 8
            if available_bytes > 0:
                payload = np.packbits(payload_bits[:available_bytes*8]).tobytes()
                print(f"   Recovered {available_bytes} bytes")
                return payload[:available_bytes], {
                    "status": "header_corrupted_recovery", 
                    "len": available_bytes,
                    "invalid_header": payload_len
                }
            else:
                return b"", {"error": "Invalid header and no recoverable data"}
        
        # Extract payload
        available_payload_bits = len(payload_bits)
        available_payload_bytes = available_payload_bits // 8
        
        print(f"   Available payload: {available_payload_bytes} bytes, Expected: {payload_len} bytes")
        
        # Extract whatever we can
        extract_bytes = min(payload_len, available_payload_bytes)
        if extract_bytes > 0:
            payload = np.packbits(payload_bits[:extract_bytes*8]).tobytes()
        else:
            payload = b""
        
        status = "ok" if extract_bytes == payload_len else "partial"
        return payload[:extract_bytes], {
            "status": status, 
            "len": extract_bytes, 
            "expected": payload_len,
            "errors_corrected": errors_corrected
        }
