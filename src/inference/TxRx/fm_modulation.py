"""
================================================================================
FM MODULATION CLASS - With AI Denoising Support
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

from fm.model_loader import FMModelLoader, get_model_for_inference
from inference.TxRx.sdr_utils import SDRUtils
from scipy.signal import resample_poly


class FM_Modulation:
    """
    FM modulation/demodulation with optional AI denoising.
    """
    
    def __init__(self, use_ai=True, model_path=None, passthrough=False, mode='general', architecture=None):
        """
        Initialize FM modulation.
        
        Args:
            use_ai: Enable AI denoising
            model_path: Path to trained model (if None, searches saved_models/FM)
            passthrough: Skip AI denoising (raw FM only)
            mode: Model mode ('general', 'music', 'speech')
            architecture: Model architecture ('1dunet', 'stft', None for auto)
        """
        self.use_ai = use_ai and not passthrough
        self.model_path = model_path
        self.model = None
        self.model_info = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
        self.architecture = architecture
        
        # FM parameters
        self.audio_rate = 44100  # Audio sample rate
        self.fm_deviation = 75000  # FM deviation (75 kHz for wideband FM)
        self.sdr_rate = 2000000  # SDR sample rate (2 MSPS)
        
        # Load AI model if enabled
        if self.use_ai:
            self._load_model()
        
        print(f"🔧 FM Modulation initialized:")
        print(f"   AI Denoising: {self.use_ai}")
        print(f"   Mode: {self.mode}")
        print(f"   Audio Rate: {self.audio_rate} Hz")
        print(f"   FM Deviation: {self.fm_deviation/1000:.0f} kHz")
        print(f"   SDR Rate: {self.sdr_rate/1e6:.1f} MSPS")
        if self.use_ai and self.model_info:
            print(f"   Model: {self.model_info['architecture']} - {self.model_info['mode']}")
            print(f"   Path: {Path(self.model_info['path']).name}")
    
    def _load_model(self):
        """Load trained FM model using dynamic model loader."""
        try:
            if self.model_path is not None:
                # Load specific model file
                self.model, self.model_info = FMModelLoader.load_model(
                    model_path=self.model_path,
                    architecture=self.architecture,
                    device=str(self.device)
                )
            else:
                # Auto-search for model
                self.model, self.model_info = get_model_for_inference(
                    mode=self.mode,
                    architecture=self.architecture,
                    device=str(self.device)
                )
            
            print(f"✅ FM Model loaded: {self.model_info['architecture']} - {self.model_info['mode']}")
            print(f"   Size: {self.model_info['size_mb']:.2f} MB")
            if 'val_loss' in self.model_info and self.model_info['val_loss'] != 'N/A':
                print(f"   Val Loss: {self.model_info['val_loss']}")
        
        except Exception as e:
            print(f"⚠️  FM model loading failed: {e}")
            print(f"   AI denoising disabled.")
            self.use_ai = False
            self.model = None
            self.model_info = None
    
    def modulate(self, audio_data, realtime=False, chunk_size=44100):
        """
        Modulate audio to FM waveform.
        
        Args:
            audio_data: Audio samples (can be bytes, numpy array, or file path)
            realtime: Enable real-time streaming mode (chunks)
            chunk_size: Samples per chunk for real-time mode
            
        Returns:
            Complex IQ FM waveform at SDR sample rate (or generator for realtime)
        """
        # Handle different input types
        if isinstance(audio_data, (str, Path)):
            # Load from file
            audio_path = audio_data
            if realtime:
                # Return generator for streaming
                return self._modulate_file_realtime(audio_path, chunk_size)
            else:
                audio_data = self._load_audio_file(audio_path)
                
                # Check if loading failed
                if audio_data is None:
                    print("❌ Audio loading failed. Cannot modulate.")
                    return None
        elif isinstance(audio_data, bytes):
            # Convert bytes to int16 audio
            audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(audio_data, np.ndarray):
            # Normalize if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
        
        print(f"🔄 FM Modulating {len(audio_data)} audio samples...")
        
        # Resample audio to SDR rate if needed
        if self.audio_rate != self.sdr_rate:
            audio_resampled = resample_poly(audio_data, self.sdr_rate, self.audio_rate)
        else:
            audio_resampled = audio_data
        
        # FM modulation
        # Phase = integral of (2*pi*deviation*audio)
        phase = np.cumsum(2 * np.pi * self.fm_deviation * audio_resampled / self.sdr_rate)
        fm_waveform = np.exp(1j * phase)
        
        # Boost signal power for better transmission (scale by 1.5x)
        fm_waveform *= 1.5
        
        # Plot before TX (only for non-realtime)
        model_name = Path(self.model_path).stem if self.model_path else "NoModel"
        SDRUtils.plot_waveform(
            fm_waveform[:100000],  # Plot first chunk only
            "FM Modulated (Before TX)",
            f"BeforeTX_FM_{model_name}_waveform.png",
            sample_rate=self.sdr_rate
        )
        
        print(f"✅ FM Modulated to {len(fm_waveform)} samples")
        return fm_waveform
    
    def _modulate_file_realtime(self, audio_path, chunk_size=44100):
        """
        Generator that yields FM-modulated chunks in real-time.
        
        Args:
            audio_path: Path to audio file (WAV, FLAC, MP3, etc.)
            chunk_size: Audio samples per chunk (default: 1 second at 44.1kHz)
            
        Yields:
            Complex IQ chunks at SDR rate
        """
        try:
            # Load audio using the same loader as non-realtime
            audio = self._load_audio_file(audio_path)
            
            if audio is None:
                print("❌ Failed to load audio for streaming")
                return
            
            # Audio is already normalized and at correct sample rate
            rate = self.audio_rate
            
            print(f"📁 Streaming audio: {Path(audio_path).name}")
            print(f"   Chunk size: {chunk_size} samples ({chunk_size/rate:.2f}s)")
            
            # Calculate SDR chunk size
            sdr_chunk_size = int(chunk_size * self.sdr_rate / self.audio_rate)
            
            # Phase accumulator for continuous phase across chunks
            phase_accum = 0.0
            
            # Stream in chunks
            total_chunks = int(np.ceil(len(audio) / chunk_size))
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                
                # Resample chunk to SDR rate
                if self.audio_rate != self.sdr_rate:
                    chunk_resampled = resample_poly(chunk, self.sdr_rate, self.audio_rate)
                else:
                    chunk_resampled = chunk
                
                # FM modulation with continuous phase
                phase_diff = 2 * np.pi * self.fm_deviation * chunk_resampled / self.sdr_rate
                phase = np.cumsum(phase_diff) + phase_accum
                phase_accum = phase[-1]  # Carry phase to next chunk
                
                fm_chunk = np.exp(1j * phase)
                
                # Boost signal power for better transmission
                fm_chunk *= 1.5
                
                chunk_num = i // chunk_size + 1
                print(f"   Streaming chunk {chunk_num}/{total_chunks} ({len(fm_chunk)} samples)")
                
                yield fm_chunk
            
            print(f"✅ Streaming complete: {total_chunks} chunks")
        
        except Exception as e:
            print(f"❌ Streaming failed: {e}")
            return
    
    def demodulate(self, waveform):
        """
        Demodulate FM waveform with optional AI denoising.
        
        Args:
            waveform: Received complex IQ samples at SDR rate
            
        Returns:
            dict: {
                'audio': denoised audio (or control if AI failed),
                'control_audio': audio without AI,
                'denoised_audio': audio after AI denoising,
                'stats': statistics
            }
        """
        print(f"🔄 FM Demodulating {len(waveform)} samples...")
        
        # FM demodulation (phase difference)
        phase_diff = np.angle(waveform[1:] * np.conj(waveform[:-1]))
        audio_demod = phase_diff * self.sdr_rate / (2 * np.pi * self.fm_deviation)
        
        # Resample to audio rate
        if self.sdr_rate != self.audio_rate:
            audio_demod = resample_poly(audio_demod, self.audio_rate, self.sdr_rate)
        
        # Normalize
        audio_demod = audio_demod / np.max(np.abs(audio_demod) + 1e-8)
        
        result = {
            'audio': audio_demod,
            'control_audio': audio_demod,
            'denoised_audio': None,
            'stats': {'samples': len(audio_demod)}
        }
        
        # AI denoising
        if self.use_ai and self.model is not None:
            print("\n--- AI Denoising ---")
            
            try:
                denoised_audio = self._denoise_audio(audio_demod)
                result['denoised_audio'] = denoised_audio
                result['audio'] = denoised_audio  # Use denoised as main output
                
                # Calculate SNR improvement
                snr_before = SDRUtils.calculate_snr(audio_demod, audio_demod)  # Placeholder
                print(f"✅ AI Denoised: {len(denoised_audio)} samples")
                
                # Plot comparison
                self._plot_fm_analysis(audio_demod, denoised_audio)
            
            except Exception as e:
                print(f"⚠️  AI denoising failed: {e}")
                print(f"   Using raw demodulated audio")
        
        return result
    
    def _denoise_audio(self, audio):
        """
        Apply AI denoising to audio.
        
        Args:
            audio: Noisy audio samples
            
        Returns:
            Denoised audio samples
        """
        # Check model architecture from model_info
        architecture = self.model_info.get('architecture', '1dunet').lower()
        
        if 'stft' in architecture or '2d' in architecture:
            # STFT 2D U-Net expects 4D input [batch, channels, freq, time]
            # Need to convert audio to spectrogram
            from scipy.signal import stft, istft
            
            # Compute STFT
            f, t, Zxx = stft(audio, fs=self.audio_rate, nperseg=1024, noverlap=512)
            
            # Magnitude and phase
            mag = np.abs(Zxx)
            phase = np.angle(Zxx)
            
            # Prepare input [batch, channels, freq, time]
            mag_tensor = torch.from_numpy(mag).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                denoised_mag_tensor = self.model(mag_tensor)
            
            # Get denoised magnitude
            denoised_mag = denoised_mag_tensor.cpu().numpy()[0, 0]
            
            # Reconstruct with original phase
            denoised_Zxx = denoised_mag * np.exp(1j * phase)
            
            # Inverse STFT
            _, denoised_audio = istft(denoised_Zxx, fs=self.audio_rate, nperseg=1024, noverlap=512)
            
            # Trim to original length
            denoised_audio = denoised_audio[:len(audio)]
            
        else:
            # 1D U-Net expects 3D input [batch, channels, samples]
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Pad to multiple of model's downsampling factor
            orig_len = audio_tensor.shape[-1]
            pad_len = (16 - (orig_len % 16)) % 16  # Assuming 4 levels (2^4=16)
            if pad_len > 0:
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_len))
            
            # Inference
            with torch.no_grad():
                denoised_tensor = self.model(audio_tensor)
            
            # Remove padding
            denoised_audio = denoised_tensor.cpu().numpy()[0, 0, :orig_len]
        
        return denoised_audio
    
    def _plot_fm_analysis(self, noisy_audio, denoised_audio):
        """Plot FM denoising analysis (similar to live_denoise.py)."""
        model_name = Path(self.model_path).stem if self.model_path else "NoModel"
        
        # Use SDRUtils FM plotting
        # Note: We need to convert audio back to FM waveform for full analysis
        # For now, just plot audio waveforms
        
        import matplotlib.pyplot as plt
        
        output_dir = Path('src/inference/plot')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Noisy audio waveform
        axes[0, 0].plot(noisy_audio[:10000], linewidth=0.8)
        axes[0, 0].set_title('After RX (Noisy)', fontweight='bold')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Noisy spectrum
        axes[0, 1].psd(noisy_audio, NFFT=1024, Fs=self.audio_rate/1000)
        axes[0, 1].set_title('Noisy Spectrum', fontweight='bold')
        axes[0, 1].set_xlabel('Frequency (kHz)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Denoised audio waveform
        axes[1, 0].plot(denoised_audio[:10000], linewidth=0.8)
        axes[1, 0].set_title('After AI Denoise', fontweight='bold')
        axes[1, 0].set_xlabel('Sample')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Denoised spectrum
        axes[1, 1].psd(denoised_audio, NFFT=1024, Fs=self.audio_rate/1000)
        axes[1, 1].set_title('Denoised Spectrum', fontweight='bold')
        axes[1, 1].set_xlabel('Frequency (kHz)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'FM Analysis - Model: {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = f'FM_Analysis_{model_name}.png'
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 FM analysis saved: {output_path}")
    
    def _load_audio_file(self, file_path):
        """Load audio from file (WAV, FLAC, MP3, etc.)."""
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        try:
            if file_ext == '.flac':
                # Load FLAC using soundfile
                try:
                    import soundfile as sf
                    audio, rate = sf.read(str(file_path), dtype='float32')
                    
                    # Convert to mono if stereo
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    
                    print(f"📁 Loaded FLAC: {file_path.name} ({rate} Hz, {len(audio)} samples)")
                
                except ImportError:
                    print("⚠️  soundfile not installed, trying scipy...")
                    # Fallback to scipy (may not support FLAC)
                    from scipy.io import wavfile
                    rate, audio = wavfile.read(str(file_path))
                    
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    
                    if audio.dtype == np.int16:
                        audio = audio.astype(np.float32) / 32768.0
                    elif audio.dtype == np.int32:
                        audio = audio.astype(np.float32) / 2147483648.0
                    else:
                        audio = audio.astype(np.float32)
            
            elif file_ext in ['.wav', '.wave']:
                # Load WAV using scipy
                from scipy.io import wavfile
                rate, audio = wavfile.read(str(file_path))
                
                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                # Normalize based on dtype
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                elif audio.dtype == np.int24:
                    audio = audio.astype(np.float32) / 8388608.0
                else:
                    audio = audio.astype(np.float32)
                
                print(f"📁 Loaded WAV: {file_path.name} ({rate} Hz, {len(audio)} samples)")
            
            elif file_ext in ['.mp3', '.m4a', '.aac', '.ogg']:
                # Try soundfile first (supports many formats via libsndfile)
                try:
                    import soundfile as sf
                    audio, rate = sf.read(str(file_path), dtype='float32')
                    
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    
                    print(f"📁 Loaded {file_ext.upper()}: {file_path.name} ({rate} Hz, {len(audio)} samples)")
                
                except:
                    print(f"⚠️  {file_ext} support requires soundfile library")
                    print(f"   Install: pip install soundfile")
                    return None
            
            else:
                print(f"⚠️  Unsupported format: {file_ext}")
                print(f"   Supported: .wav, .flac, .mp3, .m4a, .aac, .ogg")
                return None
            
            # Resample if needed
            if rate != self.audio_rate:
                audio = resample_poly(audio, self.audio_rate, rate)
                print(f"   Resampled to {self.audio_rate} Hz")
            
            # Normalize to [-1, 1] range
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio
        
        except Exception as e:
            print(f"❌ Failed to load audio file: {e}")
            print(f"   File: {file_path}")
            return None
