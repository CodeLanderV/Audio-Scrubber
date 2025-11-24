"""
================================================================================
OFDM MODULATION CLASS - With AI Denoising Support
================================================================================
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress matplotlib font warnings
from pathlib import Path
src_dir = Path(__file__).resolve().parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ofdm.lib_archived.transceiver import OFDMTransmitter, OFDMReceiver
from ofdm.lib_archived.config import OFDMConfig
from ofdm.model.neuralnet import OFDM_UNet
from inference.TxRx.sdr_utils import SDRUtils


class OFDM_Modulation:
    """
    OFDM modulation/demodulation with optional AI denoising and selectable modulation schemes.
    """
    
    def __init__(self, use_ai=True, model_path=None, passthrough=False, use_enhanced_fec=True, modulation="qpsk"):
        """
        Initialize OFDM modulation.
        
        Args:
            use_ai: Enable AI denoising
            model_path: Path to trained model (if None, searches saved_models/OFDM/final_models)
            passthrough: Skip AI denoising (raw OFDM only)
            use_enhanced_fec: Enable enhanced FEC (payload protection + error detection)
            modulation: Modulation scheme - "qpsk" (2 bits/symbol) or "16qam" (4 bits/symbol)
        """
        self.use_ai = use_ai and not passthrough
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_enhanced_fec = use_enhanced_fec
        
        # Initialize OFDM config with selected modulation scheme
        self.config = OFDMConfig()
        self.config.modulation_scheme = modulation.lower()
        
        self.transmitter = OFDMTransmitter(self.config, use_header_fec=True, use_enhanced_fec=use_enhanced_fec)
        self.receiver = OFDMReceiver(self.config, use_header_fec=True, use_enhanced_fec=use_enhanced_fec)
        
        # Store original transmitted bits for BER calculation
        self.original_bits = None
        
        # Squelch parameters (energy detection gate)
        self.squelch_threshold_db = -20.0  # Only process if signal > -20 dB
        self.squelch_enabled = self.config.squelch_enabled
        
        # Load AI model if enabled
        if self.use_ai:
            self._load_model()
        
        print(f"OFDM Modulation initialized:")
        print(f"   Modulation: {self.config.modulation_scheme.upper()}")
        print(f"   AI Denoising: {self.use_ai}")
        print(f"   Enhanced FEC: {self.use_enhanced_fec}")
        print(f"   FFT Size: {self.config.fft_size}")
        print(f"   Cyclic Prefix: {self.config.cp_len}")
        print(f"   Data Carriers: {self.config.data_subcarriers_count}")
        if self.use_ai:
            print(f"   Model: {Path(self.model_path).name if self.model_path else 'None'}")
    
    def _load_model(self):
        """Load trained OFDM model."""
        # If no model path provided, search for best model
        if self.model_path is None:
            search_paths = [
                'saved_models/OFDM/final_models',
                'saved_models/OFDM',
                'x',  # Check x directory too
            ]
            
            for search_dir in search_paths:
                model_dir = Path(search_dir)
                if model_dir.exists():
                    # Look for 1D U-Net model first, then other candidates
                    candidates = list(model_dir.glob('*1dunet*.pth')) + \
                                list(model_dir.glob('unet1d*.pth')) + \
                                list(model_dir.glob('*best*.pth')) + \
                                list(model_dir.glob('ofdm_final*.pth'))
                    if candidates:
                        self.model_path = str(candidates[0])
                        break
        
        if self.model_path is None:
            print("⚠️  No model found. AI denoising disabled.")
            self.use_ai = False
            return
        
        try:
            self.model = OFDM_UNet(in_channels=2, out_channels=2)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle checkpoint dict vs direct state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded: {Path(self.model_path).name}")
        except Exception as e:
            print(f"⚠️  Model loading failed: {e}")
            print(f"   AI denoising disabled.")
            self.use_ai = False
            self.model = None
    
    def modulate(self, data_bytes, image_path=None, buffer_frames=0):
        """
        Modulate data to OFDM waveform.
        
        Args:
            data_bytes: Raw data bytes
            image_path: Optional path to original image for visualization
            buffer_frames: Number of random OFDM frames to prepend (warm-up for receiver sync)
            
        Returns:
            Complex IQ waveform
        """
        print(f"🔄 Modulating {len(data_bytes)} bytes...")
        
        # Store original image for later comparison
        self.original_image = None
        if image_path and Path(image_path).suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
            try:
                from PIL import Image
                self.original_image = Image.open(image_path)
                print(f"📷 Original image: {self.original_image.size}")
            except Exception as e:
                print(f"⚠️  Could not load image for visualization: {e}")
        
        # Prepend random buffer frames for receiver warm-up (helps with startup transients)
        if buffer_frames > 0:
            # Estimate bytes per frame (~80 per OFDM symbol with our config)
            bytes_per_frame = max(10, self.config.fft_size // 8)
            buffer_bytes = np.random.bytes(buffer_frames * bytes_per_frame)
            print(f"📦 Adding {buffer_frames} warm-up frames ({len(buffer_bytes)} bytes) before payload")
            data_bytes_with_buffer = buffer_bytes + data_bytes
        else:
            data_bytes_with_buffer = data_bytes
        
        # Transmit using OFDM transmitter
        waveform, info = self.transmitter.transmit(data_bytes_with_buffer)
        
        # Store original transmitted bits for BER calculation (only the real payload, not buffer)
        # Replicate the bit generation from transmitter
        length_bytes = len(data_bytes).to_bytes(4, 'big')
        length_header = np.frombuffer(length_bytes, dtype=np.uint8)
        if isinstance(data_bytes, np.ndarray):
            full_packet = np.concatenate([length_header, data_bytes.astype(np.uint8)])
        else:
            full_packet = np.concatenate([length_header, np.frombuffer(data_bytes, dtype=np.uint8)])
        self.original_bits = np.unpackbits(full_packet)
        
        # Plot before TX
        model_name = Path(self.model_path).stem if self.model_path else "NoModel"
        SDRUtils.plot_waveform(
            waveform,
            "OFDM Modulated (Before TX)",
            f"BeforeTX_OFDM_{model_name}_waveform.png"
        )
        
        # Calculate actual power
        actual_power = np.mean(np.abs(waveform)**2)
        
        print(f"✅ Modulated to {len(waveform)} samples")
        print(f"   Symbols: {info.get('num_ofdm_symbols', 0)}")
        print(f"   TX Power: {actual_power:.3f} ({10*np.log10(actual_power + 1e-12):.2f} dB)")
        return waveform
    
    def _apply_filter_denoise(self, waveform, filter_type='median', window_size=5):
        """
        Apply classical digital filter for denoising.
        
        Args:
            waveform: Noisy complex IQ samples
            filter_type: 'median' or 'savitzky_golay'
            window_size: Filter window size
            
        Returns:
            Filtered waveform
        """
        from scipy import signal as scipy_signal
        
        try:
            if filter_type == 'median':
                # Median filter - good for impulse noise
                filtered_i = scipy_signal.medfilt(np.real(waveform), kernel_size=window_size)
                filtered_q = scipy_signal.medfilt(np.imag(waveform), kernel_size=window_size)
                filtered = filtered_i + 1j * filtered_q
                print(f"🔧 Applied median filter (window={window_size})")
                
            elif filter_type == 'savitzky_golay':
                # Savitzky-Golay filter - smooths while preserving edges
                polyorder = 2
                filtered_i = scipy_signal.savgol_filter(np.real(waveform), window_size, polyorder)
                filtered_q = scipy_signal.savgol_filter(np.imag(waveform), window_size, polyorder)
                filtered = filtered_i + 1j * filtered_q
                print(f"🔧 Applied Savitzky-Golay filter (window={window_size}, polyorder={polyorder})")
                
            else:
                # Fallback: simple moving average
                filtered_i = np.convolve(np.real(waveform), np.ones(window_size)/window_size, mode='same')
                filtered_q = np.convolve(np.imag(waveform), np.ones(window_size)/window_size, mode='same')
                filtered = filtered_i + 1j * filtered_q
                print(f"🔧 Applied moving average filter (window={window_size})")
            
            return filtered
            
        except Exception as e:
            print(f"⚠️  Filter denoising failed: {e}")
            return waveform
    
    def demodulate(self, waveform):
        """
        Demodulate OFDM waveform with optional AI denoising.
        
        Args:
            waveform: Received complex IQ samples
            
        Returns:
            dict: {
                'data': decoded bytes (or None if failed),
                'control_data': decoded bytes without AI (for comparison),
                'stats': statistics dictionary,
                'waveform_noisy': noisy waveform,
                'waveform_denoised': denoised waveform (if AI used),
                'symbols_noisy': raw received QPSK symbols,
                'symbols_ai': AI denoised QPSK symbols,
                'symbols_filtered': filter denoised QPSK symbols
            }
        """
        print(f"🔄 Demodulating {len(waveform)} samples...")
        
        result = {
            'data': None,
            'control_data': None,
            'stats': {},
            'waveform_noisy': waveform,
            'waveform_denoised': None,
            'waveform_filtered': None,  # Classical filter denoised
            'symbols_noisy': None,
            'symbols_ai': None,
            'symbols_filtered': None
        }
        
        # Control path (no AI)
        print("\n--- Control Path (No AI) ---")
        control_bytes, control_stats_raw = self.receiver.receive(waveform)
        
        # Compute BER for control path (PRE-FEC measurements)
        control_stats = {'ber': 0.0, 'bit_errors': 0, 'total_bits': 0, 'payload_errors': 0, 'payload_accuracy': 100.0}
        if self.original_bits is not None:
            control_bits = self._extract_received_bits(waveform)
            if control_bits is not None:
                control_stats = self._compute_ber_stats(control_bits, self.original_bits)
        
        if control_bytes is not None and len(control_bytes) > 0:
            result['control_data'] = control_bytes
            
            # Compute PAYLOAD error rate (what actually matters!)
            if self.original_bits is not None:
                # Extract original payload (skip 4-byte header)
                header_size = 4
                orig_payload_bits = self.original_bits[header_size*8:]
                orig_payload_bytes = np.packbits(orig_payload_bits).tobytes()
                
                # Compare payload
                payload_errors = 0
                for i, (orig, recv) in enumerate(zip(orig_payload_bytes, control_bytes[header_size:])):
                    if orig != recv:
                        payload_errors += 1
                
                payload_accuracy = 100.0 * (1 - payload_errors / len(control_bytes[header_size:])) if len(control_bytes) > header_size else 100.0
                control_stats['payload_errors'] = payload_errors
                control_stats['payload_accuracy'] = payload_accuracy
                
                print(f"✅ Control: Decoded {len(control_bytes)} bytes (Payload Accuracy: {payload_accuracy:.1f}%)")
            else:
                print(f"✅ Control: Decoded {len(control_bytes)} bytes")
        else:
            print(f"❌ Control: Decoding failed")
        
        # AI denoising path
        if self.use_ai and self.model is not None:
            print("\n--- AI Denoising Path ---")
            
            # SQUELCH: Energy detection gate - only denoise if signal is strong enough
            signal_power = np.mean(np.abs(waveform)**2)
            signal_power_db = 10 * np.log10(signal_power + 1e-12)
            
            if self.squelch_enabled and signal_power_db < self.squelch_threshold_db:
                print(f"🔇 SQUELCH: Signal too weak ({signal_power_db:.2f} dB < {self.squelch_threshold_db} dB threshold)")
                print(f"   ⚠️  NOT sending to AI denoising to prevent hallucination of noise")
                denoised_waveform = waveform  # Don't denoise, use raw waveform
            else:
                print(f"✅ Signal strength: {signal_power_db:.2f} dB (above threshold: {self.squelch_threshold_db} dB)")
                # Denoise waveform
                denoised_waveform = self._denoise_waveform(waveform)
            
            result['waveform_denoised'] = denoised_waveform
            
            # Demodulate denoised waveform
            ai_bytes, ai_stats_raw = self.receiver.receive(denoised_waveform)
            
            # Compute BER for AI path (PRE-FEC measurements)
            ai_stats = {'ber': 0.0, 'bit_errors': 0, 'total_bits': 0, 'payload_errors': 0, 'payload_accuracy': 100.0}
            if self.original_bits is not None:
                ai_bits = self._extract_received_bits(denoised_waveform)
                if ai_bits is not None:
                    ai_stats = self._compute_ber_stats(ai_bits, self.original_bits)
            
            if ai_bytes is not None and len(ai_bytes) > 0:
                # Compute PAYLOAD error rate
                if self.original_bits is not None:
                    # Extract original payload (skip 4-byte header)
                    header_size = 4
                    orig_payload_bits = self.original_bits[header_size*8:]
                    orig_payload_bytes = np.packbits(orig_payload_bits).tobytes()
                    
                    # Compare payload
                    payload_errors = 0
                    for i, (orig, recv) in enumerate(zip(orig_payload_bytes, ai_bytes[header_size:])):
                        if orig != recv:
                            payload_errors += 1
                    
                    payload_accuracy = 100.0 * (1 - payload_errors / len(ai_bytes[header_size:])) if len(ai_bytes) > header_size else 100.0
                    ai_stats['payload_errors'] = payload_errors
                    ai_stats['payload_accuracy'] = payload_accuracy
                    
                    print(f"✅ AI Path: Decoded {len(ai_bytes)} bytes (Payload Accuracy: {payload_accuracy:.1f}%)")
                else:
                    print(f"✅ AI Path: Decoded {len(ai_bytes)} bytes")
                
                result['data'] = ai_bytes
                result['stats'] = ai_stats
            else:
                print(f"❌ AI Path: Decoding failed")
                result['data'] = result['control_data']  # Fallback to control
                result['stats'] = control_stats
            
            # Plot comparison with images if available
            self._plot_denoising_results(
                waveform, denoised_waveform, control_stats, ai_stats,
                control_bytes, ai_bytes
            )
        else:
            # No AI, use control path result
            result['data'] = result['control_data']
            result['stats'] = control_stats
            # Define empty AI stats for plotting
            ai_stats = {'ber': 0.0, 'bit_errors': 0, 'total_bits': 0, 'payload_errors': 0, 'payload_accuracy': 0.0}
        
        # ===== FILTER-BASED DENOISING PATH =====
        print("\n--- Filter-Based Denoising Path (Savitzky-Golay) ---")
        filtered_waveform = self._apply_filter_denoise(waveform, filter_type='savitzky_golay', window_size=5)
        result['waveform_filtered'] = filtered_waveform
        
        # Demodulate filtered waveform
        filtered_bytes, filtered_stats_raw = self.receiver.receive(filtered_waveform)
        
        # Compute BER for filtered path
        filtered_stats = {'ber': 0.0, 'bit_errors': 0, 'total_bits': 0, 'payload_errors': 0, 'payload_accuracy': 100.0}
        if self.original_bits is not None:
            filtered_bits = self._extract_received_bits(filtered_waveform)
            if filtered_bits is not None:
                filtered_stats = self._compute_ber_stats(filtered_bits, self.original_bits)
        
        if filtered_bytes is not None and len(filtered_bytes) > 0:
            if self.original_bits is not None:
                header_size = 4
                orig_payload_bits = self.original_bits[header_size*8:]
                orig_payload_bytes = np.packbits(orig_payload_bits).tobytes()
                
                payload_errors = 0
                for i, (orig, recv) in enumerate(zip(orig_payload_bytes, filtered_bytes[header_size:])):
                    if orig != recv:
                        payload_errors += 1
                
                payload_accuracy = 100.0 * (1 - payload_errors / len(filtered_bytes[header_size:])) if len(filtered_bytes) > header_size else 100.0
                filtered_stats['payload_errors'] = payload_errors
                filtered_stats['payload_accuracy'] = payload_accuracy
                
                print(f"✅ Filter Path: Decoded {len(filtered_bytes)} bytes (Payload Accuracy: {payload_accuracy:.1f}%)")
            else:
                print(f"✅ Filter Path: Decoded {len(filtered_bytes)} bytes")
        else:
            print(f"❌ Filter Path: Decoding failed")
        
        # ===== EXTRACT QPSK SYMBOLS FOR ALL 3 PATHS =====
        print("\n--- Extracting QPSK Symbols for Comparison ---")
        result['symbols_noisy'] = self._extract_symbols(waveform)
        result['symbols_ai'] = self._extract_symbols(result['waveform_denoised'] if result['waveform_denoised'] is not None else waveform)
        result['symbols_filtered'] = self._extract_symbols(filtered_waveform)
        
        print(f"   Noisy symbols: {len(result['symbols_noisy']) if result['symbols_noisy'] is not None else 0}")
        print(f"   AI denoised symbols: {len(result['symbols_ai']) if result['symbols_ai'] is not None else 0}")
        print(f"   Filter denoised symbols: {len(result['symbols_filtered']) if result['symbols_filtered'] is not None else 0}")
        
        # ===== PLOT 3-WAY CONSTELLATION COMPARISON =====
        self._plot_3way_constellation(
            result['symbols_noisy'], 
            result['symbols_ai'], 
            result['symbols_filtered'],
            control_stats, ai_stats, filtered_stats
        )
        
        return result
    
    def _denoise_waveform(self, waveform):
        """
        Apply AI denoising to waveform.
        
        Args:
            waveform: Noisy complex IQ samples
            
        Returns:
            Denoised complex IQ samples
        """
        try:
            # Store original length for restoration
            original_len = len(waveform)
            
            # Pad to even length if necessary (some models require power-of-2 or even sizes)
            # Also ensure minimum length to avoid model issues
            min_length = 256
            if original_len < min_length:
                waveform_padded = np.pad(waveform, (0, min_length - original_len), mode='constant', constant_values=0)
            elif original_len % 2 != 0:
                waveform_padded = np.pad(waveform, (0, 1), mode='constant', constant_values=0)
            else:
                waveform_padded = waveform
            
            # Prepare input (I/Q channels)
            waveform_2ch = np.stack([np.real(waveform_padded), np.imag(waveform_padded)], axis=0)
            waveform_tensor = torch.from_numpy(waveform_2ch).float().unsqueeze(0).to(self.device)
            
            print(f"🧠 AI Input shape: {waveform_tensor.shape}")
            
            # Inference
            with torch.no_grad():
                denoised_tensor = self.model(waveform_tensor)
            
            print(f"🧠 AI Output shape: {denoised_tensor.shape}")
            
            # Convert back to complex
            denoised_2ch = denoised_tensor.cpu().numpy()[0]
            denoised_waveform = denoised_2ch[0] + 1j * denoised_2ch[1]
            
            # Restore original length (trim back to input size)
            denoised_waveform = denoised_waveform[:original_len]
            
            # Final sanity check
            if len(denoised_waveform) != original_len:
                print(f"⚠️  AI output size final check: {len(denoised_waveform)} vs {original_len}")
                if len(denoised_waveform) > original_len:
                    denoised_waveform = denoised_waveform[:original_len]
                else:
                    denoised_waveform = np.pad(denoised_waveform, (0, original_len - len(denoised_waveform)), mode='constant')
            
            print(f"🧠 AI Denoised: {len(denoised_waveform)} samples (padded from {original_len})")
            return denoised_waveform
        except RuntimeError as e:
            if "shape" in str(e).lower() or "size" in str(e).lower():
                print(f"❌ AI model tensor size error: {e}")
                print(f"   Input was {original_len} samples, padded to {len(waveform_padded)}")
                print(f"   This might be a model architecture mismatch")
                print(f"   Returning original waveform (no denoising)")
            else:
                print(f"❌ AI denoising runtime error: {e}")
            return waveform
        except Exception as e:
            print(f"❌ AI denoising failed: {e}")
            print(f"   Returning original waveform (no denoising)")
            return waveform
    
    def crop_to_signal(self, waveform, threshold_db=-20.0, min_duration_ms=1.0):
        """
        SQUELCH: Crop received waveform to only keep portions with signal energy above threshold.
        This prevents AI from hallucinating QPSK symbols from pure noise.
        
        Args:
            waveform: Complex IQ samples
            threshold_db: Energy threshold in dB (default -20 dB)
            min_duration_ms: Minimum signal duration in milliseconds to keep
            
        Returns:
            Cropped waveform with only signal regions (or original if all noise)
        """
        # Calculate signal energy per sample
        energy = np.abs(waveform) ** 2
        energy_db = 10 * np.log10(energy + 1e-12)
        
        # Convert threshold to linear scale
        threshold_linear = 10 ** (threshold_db / 10.0)
        
        # Find regions where energy exceeds threshold
        above_threshold = energy > threshold_linear
        
        # Calculate minimum samples to consider "signal"
        min_samples = int(min_duration_ms * self.config.sample_rate / 1000.0)
        
        # Find contiguous regions above threshold
        transitions = np.diff(above_threshold.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        ends = np.where(transitions == -1)[0] + 1
        
        if len(starts) == 0:
            # No signal detected above threshold
            print(f"   ⚠️  No signal regions above {threshold_db} dB threshold")
            return waveform  # Return original (will trigger squelch in demodulate)
        
        # Merge adjacent regions and filter by minimum duration
        valid_regions = []
        for start, end in zip(starts, ends):
            duration = end - start
            if duration >= min_samples:
                valid_regions.append((start, end))
        
        if not valid_regions:
            print(f"   ⚠️  No signal regions with duration >= {min_duration_ms} ms")
            return waveform  # Return original (will trigger squelch in demodulate)
        
        # Merge overlapping regions with small gaps
        gap_threshold = int(self.config.sample_rate * 0.001)  # 1 ms gap
        merged_regions = [valid_regions[0]]
        for start, end in valid_regions[1:]:
            if start - merged_regions[-1][1] <= gap_threshold:
                merged_regions[-1] = (merged_regions[-1][0], end)
            else:
                merged_regions.append((start, end))
        
        # Concatenate signal regions (remove silence)
        cropped_samples = []
        for start, end in merged_regions:
            cropped_samples.extend(waveform[start:end])
        
        cropped_waveform = np.array(cropped_samples, dtype=waveform.dtype)
        
        print(f"   ✂️  Cropped to signal regions: {len(cropped_waveform)} samples ({len(merged_regions)} regions)")
        return cropped_waveform
    
    def _plot_denoising_results(self, noisy, denoised, control_stats, ai_stats, 
                                  control_bytes=None, ai_bytes=None):
        """Plot comprehensive OFDM comparison with images and noise metrics."""
        from PIL import Image
        import io
        
        model_name = Path(self.model_path).stem if self.model_path else "NoModel"
        
        # Extract symbols for constellation
        noisy_symbols = self._extract_symbols(noisy)
        denoised_symbols = self._extract_symbols(denoised)
        
        # Try to reconstruct images from bytes
        original_img = self.original_image if hasattr(self, 'original_image') else None
        noisy_img = None
        clean_img = None
        
        if control_bytes:
            try:
                noisy_img = Image.open(io.BytesIO(control_bytes))
            except:
                pass
        
        if ai_bytes:
            try:
                clean_img = Image.open(io.BytesIO(ai_bytes))
            except:
                pass
        
        # Calculate noise metrics if we have images
        noise_metrics = {}
        if original_img and noisy_img:
            noise_metrics = self._calculate_noise_metrics(original_img, noisy_img, clean_img)
        
        output_dir = Path('src/inference/plot')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive comparison plot (3x2 grid)
        fig = plt.figure(figsize=(16, 18))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
        
        # Row 1: Images (if available)
        if original_img or noisy_img or clean_img:
            # Original Image
            ax1 = fig.add_subplot(gs[0, 0])
            if original_img:
                ax1.imshow(original_img)
                ax1.set_title('📷 Original Image (Before TX)', fontweight='bold', fontsize=14)
            else:
                ax1.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=14)
                ax1.set_title('📷 Original Image', fontweight='bold', fontsize=14)
            ax1.axis('off')
            
            # Received Image (with noise metrics)
            ax2 = fig.add_subplot(gs[0, 1])
            if clean_img:
                ax2.imshow(clean_img)
                title = '📷 Received Image (After RX + AI)'
                if noise_metrics:
                    title += f"\nPSNR: {noise_metrics.get('psnr_clean', 0):.2f} dB"
                ax2.set_title(title, fontweight='bold', fontsize=14)
            elif noisy_img:
                ax2.imshow(noisy_img)
                title = '📷 Received Image (After RX, No AI)'
                if noise_metrics:
                    title += f"\nPSNR: {noise_metrics.get('psnr_noisy', 0):.2f} dB"
                ax2.set_title(title, fontweight='bold', fontsize=14)
            else:
                ax2.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=14)
                ax2.set_title('📷 Received Image', fontweight='bold', fontsize=14)
            ax2.axis('off')
        else:
            # No images - show noise metrics as text
            ax1 = fig.add_subplot(gs[0, :])
            ax1.text(0.5, 0.5, 'No image data available for visualization', 
                    ha='center', va='center', fontsize=14)
            ax1.axis('off')
        
        # QPSK reference points
        qpsk_ref = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
        
        # Row 2: QPSK Constellations
        # Left: BEFORE AI Denoising (Noisy RX)
        ax3 = fig.add_subplot(gs[1, 0])
        if noisy_symbols is not None and len(noisy_symbols) > 0:
            ax3.scatter(np.real(noisy_symbols), np.imag(noisy_symbols),
                       alpha=0.4, s=15, c='orange', label='Noisy RX Symbols')
            ax3.scatter(np.real(qpsk_ref), np.imag(qpsk_ref),
                       c='red', s=250, marker='x', linewidths=4, label='Ideal QPSK', zorder=5)
        ax3.axhline(0, color='k', linewidth=0.8, alpha=0.3)
        ax3.axvline(0, color='k', linewidth=0.8, alpha=0.3)
        ax3.set_title('🔴 BEFORE AI Denoising - QPSK Constellation', fontweight='bold', fontsize=14)
        ax3.set_xlabel('I (In-Phase)', fontsize=11)
        ax3.set_ylabel('Q (Quadrature)', fontsize=11)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.axis('equal')
        ax3.set_xlim(-1.5, 1.5)
        ax3.set_ylim(-1.5, 1.5)
        
        # Right: AFTER AI Denoising (Clean)
        ax4 = fig.add_subplot(gs[1, 1])
        if denoised_symbols is not None and len(denoised_symbols) > 0:
            ax4.scatter(np.real(denoised_symbols), np.imag(denoised_symbols),
                       alpha=0.4, s=15, c='blue', label='AI Denoised Symbols')
            ax4.scatter(np.real(qpsk_ref), np.imag(qpsk_ref),
                       c='red', s=250, marker='x', linewidths=4, label='Ideal QPSK', zorder=5)
        ax4.axhline(0, color='k', linewidth=0.8, alpha=0.3)
        ax4.axvline(0, color='k', linewidth=0.8, alpha=0.3)
        ax4.set_title('🟢 AFTER AI Denoising - QPSK Constellation', fontweight='bold', fontsize=14)
        ax4.set_xlabel('I (In-Phase)', fontsize=11)
        ax4.set_ylabel('Q (Quadrature)', fontsize=11)
        ax4.legend(loc='upper right', fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.axis('equal')
        ax4.set_xlim(-1.5, 1.5)
        ax4.set_ylim(-1.5, 1.5)
        
        # Row 3: Waveform and PSD
        # Left: Waveform Comparison
        ax5 = fig.add_subplot(gs[2, 0])
        time_axis = np.arange(1000)
        ax5.plot(time_axis, np.real(noisy[:1000]), label='Noisy I', 
                alpha=0.6, linewidth=1.2, color='orange')
        ax5.plot(time_axis, np.real(denoised[:1000]), label='Denoised I', 
                alpha=0.8, linewidth=1.0, color='blue', linestyle='--')
        ax5.set_title('Waveform Comparison (I channel)', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Sample', fontsize=11)
        ax5.set_ylabel('Amplitude', fontsize=11)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # Right: PSD Comparison
        ax6 = fig.add_subplot(gs[2, 1])
        try:
            ax6.psd(noisy, NFFT=1024, Fs=2.0, scale_by_freq=False, 
                   color='orange', alpha=0.7, label='Noisy')
            ax6.psd(denoised, NFFT=1024, Fs=2.0, scale_by_freq=False, 
                   color='blue', alpha=0.7, label='Denoised', linestyle='--')
        except Exception as e:
            print(f"⚠️  PSD plot warning (non-critical): {e}")
        
        ax6.set_title('Power Spectral Density', fontweight='bold', fontsize=12)
        ax6.set_xlabel('Frequency (MHz)', fontsize=11)
        ax6.set_ylabel('Power (dB/Hz)', fontsize=11)
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # Add stats as text (including noise metrics)
        stats_lines = [
            f"Control (No AI): BER={control_stats.get('ber', 0):.4f}, "
            f"Errors={control_stats.get('bit_errors', 0)}/{control_stats.get('total_bits', 0)}",
            f"AI Denoised:     BER={ai_stats.get('ber', 0):.4f}, "
            f"Errors={ai_stats.get('bit_errors', 0)}/{ai_stats.get('total_bits', 0)}"
        ]
        
        if noise_metrics:
            stats_lines.append("\n🔊 NOISE METRICS:")
            if 'snr_noisy' in noise_metrics:
                stats_lines.append(f"   Noisy RX:  SNR={noise_metrics['snr_noisy']:.2f} dB, "
                                 f"MSE={noise_metrics['mse_noisy']:.2f}, PSNR={noise_metrics['psnr_noisy']:.2f} dB")
            if 'snr_clean' in noise_metrics:
                stats_lines.append(f"   AI Clean:  SNR={noise_metrics['snr_clean']:.2f} dB, "
                                 f"MSE={noise_metrics['mse_clean']:.2f}, PSNR={noise_metrics['psnr_clean']:.2f} dB")
            if 'improvement' in noise_metrics:
                stats_lines.append(f"   🎯 Improvement: {noise_metrics['improvement']:.2f} dB SNR gain")
        
        stats_text = '\n'.join(stats_lines)
        fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='bottom')
        
        plt.suptitle(f'OFDM Transmission Analysis - Model: {model_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        filename = f'OFDM_Comparison_{model_name}.png'
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 OFDM comparison saved: {output_path}")
        
        # Print stats comparison
        print("\n📊 Statistics Comparison:")
        print(f"   Control: BER={control_stats.get('ber', 0):.4f}, "
              f"Errors={control_stats.get('bit_errors', 0)}/{control_stats.get('total_bits', 0)}")
        print(f"   AI:      BER={ai_stats.get('ber', 0):.4f}, "
              f"Errors={ai_stats.get('bit_errors', 0)}/{ai_stats.get('total_bits', 0)}")
        
        if noise_metrics:
            print("\n🔊 Noise Metrics:")
            if 'snr_noisy' in noise_metrics:
                print(f"   Noisy:  SNR={noise_metrics['snr_noisy']:.2f} dB, "
                      f"MSE={noise_metrics['mse_noisy']:.2f}, PSNR={noise_metrics['psnr_noisy']:.2f} dB")
            if 'snr_clean' in noise_metrics:
                print(f"   Clean:  SNR={noise_metrics['snr_clean']:.2f} dB, "
                      f"MSE={noise_metrics['mse_clean']:.2f}, PSNR={noise_metrics['psnr_clean']:.2f} dB")
            if 'improvement' in noise_metrics:
                print(f"   🎯 Improvement: {noise_metrics['improvement']:.2f} dB")
    
    def _plot_3way_constellation(self, symbols_noisy, symbols_ai, symbols_filtered, 
                                 control_stats, ai_stats, filtered_stats):
        """
        Plot 3-way QPSK constellation comparison.
        
        Args:
            symbols_noisy: Raw received QPSK symbols
            symbols_ai: AI denoised QPSK symbols
            symbols_filtered: Filter denoised QPSK symbols
            control_stats, ai_stats, filtered_stats: Statistics for each path
        """
        output_dir = Path('src/inference/plot')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = Path(self.model_path).stem if self.model_path else "NoModel"
        
        # QPSK reference points
        qpsk_ref = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Noisy (Raw RX)
        ax = axes[0]
        if symbols_noisy is not None and len(symbols_noisy) > 0:
            ax.scatter(np.real(symbols_noisy), np.imag(symbols_noisy),
                      alpha=0.3, s=20, c='orange', label='Noisy RX')
            ax.scatter(np.real(qpsk_ref), np.imag(qpsk_ref),
                      c='red', s=250, marker='x', linewidths=4, label='Ideal QPSK', zorder=5)
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_title('🔴 NOISY (No Denoising)', fontweight='bold', fontsize=12)
        ax.set_xlabel('I (In-Phase)')
        ax.set_ylabel('Q (Quadrature)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        
        # Add stats
        stats_text = f"BER: {control_stats.get('ber', 0):.4f}\nErrors: {control_stats.get('bit_errors', 0)}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Plot 2: AI Denoised
        ax = axes[1]
        if symbols_ai is not None and len(symbols_ai) > 0:
            ax.scatter(np.real(symbols_ai), np.imag(symbols_ai),
                      alpha=0.3, s=20, c='blue', label='AI Denoised')
            ax.scatter(np.real(qpsk_ref), np.imag(qpsk_ref),
                      c='red', s=250, marker='x', linewidths=4, label='Ideal QPSK', zorder=5)
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_title('🟦 AI DENOISED', fontweight='bold', fontsize=12)
        ax.set_xlabel('I (In-Phase)')
        ax.set_ylabel('Q (Quadrature)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        
        # Add stats
        stats_text = f"BER: {ai_stats.get('ber', 0):.4f}\nErrors: {ai_stats.get('bit_errors', 0)}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Plot 3: Filter Denoised
        ax = axes[2]
        if symbols_filtered is not None and len(symbols_filtered) > 0:
            ax.scatter(np.real(symbols_filtered), np.imag(symbols_filtered),
                      alpha=0.3, s=20, c='green', label='Filter Denoised')
            ax.scatter(np.real(qpsk_ref), np.imag(qpsk_ref),
                      c='red', s=250, marker='x', linewidths=4, label='Ideal QPSK', zorder=5)
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_title('🟩 FILTER DENOISED (Savitzky-Golay)', fontweight='bold', fontsize=12)
        ax.set_xlabel('I (In-Phase)')
        ax.set_ylabel('Q (Quadrature)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        
        # Add stats
        stats_text = f"BER: {filtered_stats.get('ber', 0):.4f}\nErrors: {filtered_stats.get('bit_errors', 0)}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.suptitle(f'3-Way QPSK Constellation Comparison - Model: {model_name}', 
                    fontsize=14, fontweight='bold')
        
        filename = f'OFDM_3Way_Constellation_{model_name}.png'
        output_path = output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 3-way constellation saved: {output_path}")
    
    def _calculate_noise_metrics(self, original_img, noisy_img, clean_img=None):
        """Calculate SNR, MSE, PSNR between images."""
        metrics = {}
        
        try:
            # Ensure same size
            if original_img.size != noisy_img.size:
                noisy_img = noisy_img.resize(original_img.size)
            if clean_img and clean_img.size != original_img.size:
                clean_img = clean_img.resize(original_img.size)
            
            # Convert to numpy arrays
            orig = np.array(original_img, dtype=np.float64)
            noisy = np.array(noisy_img, dtype=np.float64)
            
            # Calculate metrics for noisy image
            mse_noisy = np.mean((orig - noisy) ** 2)
            if mse_noisy > 0:
                psnr_noisy = 10 * np.log10(255.0 ** 2 / mse_noisy)
                signal_power = np.mean(orig ** 2)
                noise_power = mse_noisy
                snr_noisy = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
                
                metrics['mse_noisy'] = mse_noisy
                metrics['psnr_noisy'] = psnr_noisy
                metrics['snr_noisy'] = snr_noisy
            
            # Calculate metrics for clean image (if available)
            if clean_img:
                clean = np.array(clean_img, dtype=np.float64)
                mse_clean = np.mean((orig - clean) ** 2)
                if mse_clean > 0:
                    psnr_clean = 10 * np.log10(255.0 ** 2 / mse_clean)
                    noise_power_clean = mse_clean
                    snr_clean = 10 * np.log10(signal_power / noise_power_clean) if noise_power_clean > 0 else float('inf')
                    
                    metrics['mse_clean'] = mse_clean
                    metrics['psnr_clean'] = psnr_clean
                    metrics['snr_clean'] = snr_clean
                    
                    # Calculate improvement
                    if 'snr_noisy' in metrics:
                        metrics['improvement'] = snr_clean - snr_noisy
        
        except Exception as e:
            print(f"⚠️  Noise metrics calculation failed: {e}")
        
        return metrics
    
    def _extract_symbols(self, waveform):
        """Extract QPSK symbols from waveform for constellation plot.

        Uses config.data_carriers (not data_subcarriers) which holds the indices
        of active data subcarriers relative to DC.
        """
        try:
            # Remove CP and apply FFT
            symbol_size = self.config.fft_size + self.config.cp_len
            num_symbols = len(waveform) // symbol_size
            
            symbols_all = []
            for i in range(num_symbols):
                ofdm_symbol = waveform[i * symbol_size:(i + 1) * symbol_size]
                ofdm_symbol_no_cp = ofdm_symbol[self.config.cp_len:]
                freq_domain = np.fft.fft(ofdm_symbol_no_cp)
                
                # Extract data carriers (attribute is data_carriers)
                data_indices = self.config.data_carriers
                data_symbols = freq_domain[data_indices]
                symbols_all.extend(data_symbols)
            
            return np.array(symbols_all)
        except Exception as e:
            print(f"⚠️  Symbol extraction failed: {e}")
            return None
    
    def _extract_received_bits(self, waveform):
        """Extract received bits from waveform (replicates receiver logic exactly)."""
        try:
            # Split into symbols EXACTLY like receiver does
            sym_len = self.config.symbol_len
            num_symbols = len(waveform) // sym_len
            
            all_data_symbols = []
            
            # Process each symbol
            for i in range(num_symbols):
                time_sym = waveform[i*sym_len : (i+1)*sym_len]
                
                # CP Removal + FFT
                raw_data, raw_pilots = self.receiver.engine.process_received_symbol(time_sym)
                
                # Channel Equalization
                h_est = self.receiver.equalizer.estimate_channel(raw_pilots)
                corrected_data = self.receiver.equalizer.equalize(raw_data, h_est)
                
                all_data_symbols.extend(corrected_data)
            
            all_data_symbols = np.array(all_data_symbols)
            
            # Demodulate to bits using SAME modulator as receiver
            bits = self.receiver.modulator.demodulate(all_data_symbols)
            
            return bits
            
        except Exception as e:
            print(f"⚠️  Bit extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _compute_ber_stats(self, received_bits, original_bits):
        """
        Compute BER statistics by comparing received and original bits.
        
        NOTE: This measures PRE-FEC bit errors. After FEC correction,
        these errors may be corrected and the decoded data may be accurate.
        High BER here doesn't mean decoding failed!
        """
        if received_bits is None or original_bits is None or len(received_bits) == 0 or len(original_bits) == 0:
            return {'ber': 0.0, 'bit_errors': 0, 'total_bits': 0}
        
        # Compare the raw demodulated bits (this is PRE-FEC, so errors are expected to be corrected)
        min_len = min(len(original_bits), len(received_bits))
        if min_len == 0:
            return {'ber': 0.0, 'bit_errors': 0, 'total_bits': 0}
        
        # Compare only matching length
        bit_errors = np.sum(original_bits[:min_len] != received_bits[:min_len])
        total_bits = min_len
        ber = bit_errors / total_bits if total_bits > 0 else 0.0
        
        return {
            'ber': ber,
            'bit_errors': int(bit_errors),
            'total_bits': total_bits,
            'note': 'Pre-FEC measurements - errors corrected by FEC'
        }
