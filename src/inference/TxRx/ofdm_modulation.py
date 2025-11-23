"""
================================================================================
OFDM MODULATION CLASS - With AI Denoising Support
================================================================================
"""

import numpy as np
import torch
from pathlib import Path
import sys
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
    OFDM modulation/demodulation with optional AI denoising.
    """
    
    def __init__(self, use_ai=True, model_path=None, passthrough=False):
        """
        Initialize OFDM modulation.
        
        Args:
            use_ai: Enable AI denoising
            model_path: Path to trained model (if None, searches saved_models/OFDM/final_models)
            passthrough: Skip AI denoising (raw OFDM only)
        """
        self.use_ai = use_ai and not passthrough
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize OFDM transmitter and receiver
        self.config = OFDMConfig()
        self.transmitter = OFDMTransmitter(self.config)
        self.receiver = OFDMReceiver(self.config)
        
        # Load AI model if enabled
        if self.use_ai:
            self._load_model()
        
        print(f"🔧 OFDM Modulation initialized:")
        print(f"   AI Denoising: {self.use_ai}")
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
            ]
            
            for search_dir in search_paths:
                model_dir = Path(search_dir)
                if model_dir.exists():
                    # Look for best model
                    candidates = list(model_dir.glob('*best*.pth')) + \
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
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded: {Path(self.model_path).name}")
        except Exception as e:
            print(f"⚠️  Model loading failed: {e}")
            print(f"   AI denoising disabled.")
            self.use_ai = False
            self.model = None
    
    def modulate(self, data_bytes):
        """
        Modulate data to OFDM waveform.
        
        Args:
            data_bytes: Raw data bytes
            
        Returns:
            Complex IQ waveform
        """
        print(f"🔄 Modulating {len(data_bytes)} bytes...")
        
        # Transmit using OFDM transmitter
        waveform, info = self.transmitter.transmit(data_bytes)
        
        # Plot before TX
        model_name = Path(self.model_path).stem if self.model_path else "NoModel"
        SDRUtils.plot_waveform(
            waveform,
            "OFDM Modulated (Before TX)",
            f"BeforeTX_OFDM_{model_name}_waveform.png"
        )
        
        print(f"✅ Modulated to {len(waveform)} samples")
        print(f"   Symbols: {info.get('num_ofdm_symbols', 0)}, Power: {info.get('power', 0):.2f}")
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
                'waveform_denoised': denoised waveform (if AI used)
            }
        """
        print(f"🔄 Demodulating {len(waveform)} samples...")
        
        result = {
            'data': None,
            'control_data': None,
            'stats': {},
            'waveform_noisy': waveform,
            'waveform_denoised': None
        }
        
        # Control path (no AI)
        print("\n--- Control Path (No AI) ---")
        control_bytes, control_stats = self.receiver.receive(waveform)
        
        if control_bytes is not None and len(control_bytes) > 0:
            result['control_data'] = control_bytes
            print(f"✅ Control: Decoded {len(control_bytes)} bytes")
        else:
            print(f"❌ Control: Decoding failed")
        
        # AI denoising path
        if self.use_ai and self.model is not None:
            print("\n--- AI Denoising Path ---")
            
            # Denoise waveform
            denoised_waveform = self._denoise_waveform(waveform)
            result['waveform_denoised'] = denoised_waveform
            
            # Demodulate denoised waveform
            ai_bytes, ai_stats = self.receiver.receive(denoised_waveform)
            
            if ai_bytes is not None and len(ai_bytes) > 0:
                result['data'] = ai_bytes
                result['stats'] = ai_stats
                print(f"✅ AI Path: Decoded {len(ai_bytes)} bytes")
            else:
                print(f"❌ AI Path: Decoding failed")
                result['data'] = result['control_data']  # Fallback to control
                result['stats'] = control_stats
            
            # Plot comparison
            self._plot_denoising_results(waveform, denoised_waveform, control_stats, ai_stats)
        else:
            # No AI, use control path result
            result['data'] = result['control_data']
            result['stats'] = control_stats
        
        return result
    
    def _denoise_waveform(self, waveform):
        """
        Apply AI denoising to waveform.
        
        Args:
            waveform: Noisy complex IQ samples
            
        Returns:
            Denoised complex IQ samples
        """
        # Prepare input (I/Q channels)
        waveform_2ch = np.stack([np.real(waveform), np.imag(waveform)], axis=0)
        waveform_tensor = torch.from_numpy(waveform_2ch).float().unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            denoised_tensor = self.model(waveform_tensor)
        
        # Convert back to complex
        denoised_2ch = denoised_tensor.cpu().numpy()[0]
        denoised_waveform = denoised_2ch[0] + 1j * denoised_2ch[1]
        
        print(f"🧠 AI Denoised: {len(denoised_waveform)} samples")
        return denoised_waveform
    
    def _plot_denoising_results(self, noisy, denoised, control_stats, ai_stats):
        """Plot comprehensive OFDM comparison with before/after constellations."""
        model_name = Path(self.model_path).stem if self.model_path else "NoModel"
        
        # Extract symbols for constellation
        noisy_symbols = self._extract_symbols(noisy)
        denoised_symbols = self._extract_symbols(denoised)
        
        output_dir = Path('src/inference/plot')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive comparison plot (2x2 grid)
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        # QPSK reference points
        qpsk_ref = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
        
        # Top Left: Constellation BEFORE Denoising
        ax1 = fig.add_subplot(gs[0, 0])
        if noisy_symbols is not None and len(noisy_symbols) > 0:
            ax1.scatter(np.real(noisy_symbols), np.imag(noisy_symbols),
                       alpha=0.4, s=15, c='orange', label='Noisy Symbols')
            ax1.scatter(np.real(qpsk_ref), np.imag(qpsk_ref),
                       c='red', s=250, marker='x', linewidths=4, label='Ideal QPSK', zorder=5)
        ax1.axhline(0, color='k', linewidth=0.8, alpha=0.3)
        ax1.axvline(0, color='k', linewidth=0.8, alpha=0.3)
        ax1.set_title('BEFORE Denoising - Constellation', fontweight='bold', fontsize=14)
        ax1.set_xlabel('I (In-Phase)', fontsize=11)
        ax1.set_ylabel('Q (Quadrature)', fontsize=11)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axis('equal')
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        
        # Top Right: Constellation AFTER Denoising
        ax2 = fig.add_subplot(gs[0, 1])
        if denoised_symbols is not None and len(denoised_symbols) > 0:
            ax2.scatter(np.real(denoised_symbols), np.imag(denoised_symbols),
                       alpha=0.4, s=15, c='blue', label='Denoised Symbols')
            ax2.scatter(np.real(qpsk_ref), np.imag(qpsk_ref),
                       c='red', s=250, marker='x', linewidths=4, label='Ideal QPSK', zorder=5)
        ax2.axhline(0, color='k', linewidth=0.8, alpha=0.3)
        ax2.axvline(0, color='k', linewidth=0.8, alpha=0.3)
        ax2.set_title('AFTER Denoising - Constellation', fontweight='bold', fontsize=14)
        ax2.set_xlabel('I (In-Phase)', fontsize=11)
        ax2.set_ylabel('Q (Quadrature)', fontsize=11)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.axis('equal')
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        
        # Bottom Left: Waveform Comparison
        ax3 = fig.add_subplot(gs[1, 0])
        time_axis = np.arange(1000)
        ax3.plot(time_axis, np.real(noisy[:1000]), label='Noisy I', 
                alpha=0.6, linewidth=1.2, color='orange')
        ax3.plot(time_axis, np.real(denoised[:1000]), label='Denoised I', 
                alpha=0.8, linewidth=1.0, color='blue', linestyle='--')
        ax3.set_title('Waveform Comparison (I channel)', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Sample', fontsize=11)
        ax3.set_ylabel('Amplitude', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Bottom Right: PSD Comparison
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.psd(noisy, NFFT=1024, Fs=2.0, scale_by_freq=False, 
               color='orange', alpha=0.7, label='Noisy')
        ax4.psd(denoised, NFFT=1024, Fs=2.0, scale_by_freq=False, 
               color='blue', alpha=0.7, label='Denoised')
        ax4.set_title('Power Spectral Density', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Frequency (MHz)', fontsize=11)
        ax4.set_ylabel('Power (dB/Hz)', fontsize=11)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Add stats as text
        stats_text = (
            f"Control (No AI): BER={control_stats.get('ber', 0):.4f}, "
            f"Errors={control_stats.get('bit_errors', 0)}/{control_stats.get('total_bits', 0)}\n"
            f"AI Denoised:     BER={ai_stats.get('ber', 0):.4f}, "
            f"Errors={ai_stats.get('bit_errors', 0)}/{ai_stats.get('total_bits', 0)}"
        )
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'OFDM Denoising Comparison - Model: {model_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
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
    
    def _extract_symbols(self, waveform):
        """Extract QPSK symbols from waveform for constellation plot."""
        try:
            # Remove CP and apply FFT
            symbol_size = self.config.fft_size + self.config.cp_len
            num_symbols = len(waveform) // symbol_size
            
            symbols_all = []
            for i in range(num_symbols):
                ofdm_symbol = waveform[i * symbol_size:(i + 1) * symbol_size]
                ofdm_symbol_no_cp = ofdm_symbol[self.config.cp_len:]
                freq_domain = np.fft.fft(ofdm_symbol_no_cp)
                
                # Extract data carriers
                data_indices = self.config.data_subcarriers
                data_symbols = freq_domain[data_indices]
                symbols_all.extend(data_symbols)
            
            return np.array(symbols_all)
        except Exception as e:
            print(f"⚠️  Symbol extraction failed: {e}")
            return None
