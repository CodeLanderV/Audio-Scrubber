import torch
import librosa
import soundfile as sf
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_stft import Paths, STFTSettings
from src.fm.model_stft.stft_unet2d import UNet2D_STFT

"""
================================================================================
MULTI-MODEL STFT AUDIO DENOISER - INFERENCE MODULE
================================================================================

Purpose:
    Denoise audio using ALL THREE trained STFT models (Speech, Music, General)
    and save results in separate output folders for comparison.

Models Used:
    1. STFT Speech Model (stft_speech_100_norm) - Optimized for voice/speech
    2. STFT Music Model (stft_music_100_norm) - Optimized for music
    3. STFT General Model (stft_general_100_norm) - Works on both speech and music

Output Structure:
    output_stft/
    ├── speech/          # Results from STFT speech model
    ├── music/           # Results from STFT music model
    └── general/         # Results from STFT general model

How it Works:
    1. Load audio → Convert to STFT spectrogram (magnitude + phase)
    2. Denoise magnitude with model
    3. Combine denoised magnitude with original phase
    4. Convert back to audio with inverse STFT

Usage Examples:

    1. DENOISE SINGLE FILE WITH ALL STFT MODELS:
       python src/inference_stft_all_models.py path/to/noisy_audio.wav
       → Creates: output_stft/speech/denoised_audio.wav
                  output_stft/music/denoised_audio.wav
                  output_stft/general/denoised_audio.wav

    2. BATCH DENOISE DIRECTORY:
       python src/inference_stft_all_models.py path/to/noisy_files/
       → Processes all audio files in directory with all 3 STFT models

    3. CUSTOM OUTPUT DIRECTORY:
       python src/inference_stft_all_models.py input.wav custom_output/
       → Creates: custom_output/speech/, custom_output/music/, custom_output/general/

Audio Configuration:
    - Sample Rate: 44100 Hz (CD quality)
    - STFT: n_fft=2048, hop_length=512
    - Supported Formats: .wav, .mp3, .flac, .m4a, .ogg

Created: 22/11/2025
================================================================================
"""

class MultiModelSTFTDenoiser:
    """
    Audio denoiser using all three trained STFT models (Speech, Music, General).
    """
    def __init__(self, device='cpu'):
        """
        Initialize the denoiser with all three STFT models.
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.sample_rate = STFTSettings.SAMPLE_RATE
        self.n_fft = STFTSettings.N_FFT
        self.hop_length = STFTSettings.HOP_LENGTH
        self.win_length = STFTSettings.WIN_LENGTH
        self.window = STFTSettings.WINDOW
        
        # Model paths
        self.model_paths = {
            'speech': str(Paths.MODEL_STFT_SPEECH_BEST),
            'music': str(Paths.MODEL_STFT_MUSIC_BEST),
            'general': str(Paths.MODEL_STFT_GENERAL_BEST)
        }
        
        # Load all three models
        self.models = {}
        self.model_info = {}
        
        print(f"\n{'='*70}")
        print("Loading STFT Models...")
        print(f"{'='*70}")
        
        for model_name, model_path in self.model_paths.items():
            if not Path(model_path).exists():
                print(f"⚠️  {model_name.upper()} STFT model not found at {model_path}")
                print(f"   Skipping {model_name} model...")
                continue
            
            # Load model
            model = UNet2D_STFT(in_channels=1, out_channels=1).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.models[model_name] = model
            self.model_info[model_name] = {
                'train_loss': checkpoint.get('train_loss', 'N/A'),
                'val_loss': checkpoint.get('val_loss', 'N/A'),
                'path': model_path
            }
            
            print(f"\n✅ {model_name.upper()} STFT Model Loaded")
            print(f"   Path: {model_path}")
            print(f"   Training Loss: {checkpoint.get('train_loss', 'N/A')}")
            print(f"   Validation Loss: {checkpoint.get('val_loss', 'N/A')}")
        
        print(f"\n{'='*70}")
        print(f"✅ {len(self.models)}/{len(self.model_paths)} STFT models loaded successfully")
        print(f"{'='*70}\n")
        
        if len(self.models) == 0:
            raise RuntimeError("No STFT models could be loaded. Please train models first.")
    
    def audio_to_stft(self, audio):
        """
        Convert audio to STFT magnitude and phase.
        
        Args:
            audio: Audio waveform (1D numpy array)
        
        Returns:
            magnitude: STFT magnitude spectrogram (freq_bins, time_frames)
            phase: STFT phase for reconstruction (freq_bins, time_frames)
        """
        # Compute STFT
        stft_complex = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window
        )
        
        # Extract magnitude and phase
        magnitude = np.abs(stft_complex)
        phase = np.angle(stft_complex)
        
        return magnitude, phase
    
    def stft_to_audio(self, magnitude, phase):
        """
        Convert STFT magnitude and phase back to audio.
        
        Args:
            magnitude: STFT magnitude spectrogram (freq_bins, time_frames)
            phase: STFT phase (freq_bins, time_frames)
        
        Returns:
            audio: Reconstructed audio waveform (1D numpy array)
        """
        # Reconstruct complex STFT
        stft_complex = magnitude * np.exp(1j * phase)
        
        # Inverse STFT
        audio = librosa.istft(
            stft_complex,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window
        )
        
        return audio
    
    def denoise_audio_with_model(self, audio, model):
        """
        Denoise an audio signal using a specific STFT model.
        
        Args:
            audio: Audio waveform (1D numpy array)
            model: PyTorch STFT model to use for denoising
        
        Returns:
            denoised_audio: Denoised audio waveform (1D numpy array)
        """
        # Convert audio to STFT
        noisy_magnitude, phase = self.audio_to_stft(audio)
        
        # Convert to tensor and add batch + channel dimensions
        noisy_mag_tensor = torch.FloatTensor(noisy_magnitude).unsqueeze(0).unsqueeze(0)  # (1, 1, freq, time)
        noisy_mag_tensor = noisy_mag_tensor.to(self.device)
        
        # Denoise magnitude with model
        with torch.no_grad():
            clean_mag_tensor = model(noisy_mag_tensor)
        
        # Convert back to numpy
        clean_magnitude = clean_mag_tensor.squeeze().cpu().numpy()
        
        # Reconstruct audio using denoised magnitude + original phase
        denoised_audio = self.stft_to_audio(clean_magnitude, phase)
        
        return denoised_audio
    
    def denoise_file_all_models(self, input_path, output_base_dir="output_stft"):
        """
        Denoise an audio file with all three STFT models and save results separately.
        
        Args:
            input_path: Path to noisy audio file
            output_base_dir: Base directory for output (creates subdirs for each model)
        
        Returns:
            Dictionary with paths to all denoised files
        """
        input_path = Path(input_path)
        output_base_dir = Path(output_base_dir)
        
        print(f"\n{'='*70}")
        print(f"Processing: {input_path.name}")
        print(f"{'='*70}")
        
        # Load audio
        print(f"\n📂 Loading audio...")
        audio, sr = librosa.load(str(input_path), sr=self.sample_rate)
        print(f"   ✓ Loaded: {len(audio)} samples at {sr} Hz ({len(audio)/sr:.2f} seconds)")
        
        # Denoise with each model
        output_paths = {}
        
        for model_name, model in self.models.items():
            print(f"\n🔧 Denoising with {model_name.upper()} STFT model...")
            
            # Denoise
            denoised = self.denoise_audio_with_model(audio, model)
            print(f"   ✓ Denoised: {len(denoised)} samples")
            
            # Create output directory
            output_dir = output_base_dir / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save with original extension
            output_path = output_dir / f"denoised_{input_path.name}"
            sf.write(str(output_path), denoised, sr)
            output_paths[model_name] = str(output_path)
            
            print(f"   ✓ Saved to: {output_path}")
        
        print(f"\n{'='*70}")
        print(f"✅ File denoised with {len(self.models)} STFT models!")
        print(f"{'='*70}\n")
        
        return output_paths
    
    def denoise_directory(self, input_dir, output_base_dir="output_stft"):
        """
        Denoise all audio files in a directory with all STFT models.
        
        Args:
            input_dir: Directory containing noisy audio files
            output_base_dir: Base directory for output
        
        Returns:
            Dictionary with statistics about processed files
        """
        input_dir = Path(input_dir)
        
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(input_dir.glob(f"*{ext}")))
        
        if not audio_files:
            print(f"❌ No audio files found in {input_dir}")
            print(f"   Supported formats: {', '.join(audio_extensions)}")
            return {}
        
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING")
        print(f"{'='*70}")
        print(f"Input Directory: {input_dir}")
        print(f"Output Directory: {output_base_dir}")
        print(f"Files Found: {len(audio_files)}")
        print(f"Models: {', '.join(self.models.keys())}")
        print(f"{'='*70}\n")
        
        # Process each file
        results = {}
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}]")
            output_paths = self.denoise_file_all_models(audio_file, output_base_dir)
            results[str(audio_file)] = output_paths
        
        # Summary
        print(f"\n{'='*70}")
        print(f"✅ BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Files Processed: {len(audio_files)}")
        print(f"Models Used: {len(self.models)}")
        print(f"Total Outputs: {len(audio_files) * len(self.models)}")
        print(f"\nOutput Structure:")
        for model_name in self.models.keys():
            output_dir = Path(output_base_dir) / model_name
            print(f"  {output_dir}/ ({len(audio_files)} files)")
        print(f"{'='*70}\n")
        
        return results


def main():
    """
    Main entry point for multi-model STFT inference.
    
    Usage:
        python src/inference_stft_all_models.py                           # Shows usage
        python src/inference_stft_all_models.py audio.wav                 # Denoise single file
        python src/inference_stft_all_models.py audio.wav custom_output/  # Custom output dir
        python src/inference_stft_all_models.py noisy_files/              # Denoise directory
    """
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🔧 Using device: {device.upper()}")
    
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("MULTI-MODEL STFT AUDIO DENOISER")
        print("="*70)
        print("\nUsage:")
        print("  python src/inference_stft_all_models.py <input_file_or_dir> [output_dir]")
        print("\nExamples:")
        print("  python src/inference_stft_all_models.py noisy_audio.wav")
        print("  python src/inference_stft_all_models.py noisy_audio.wav results/")
        print("  python src/inference_stft_all_models.py noisy_files/")
        print("\nOutput Structure:")
        print("  output_stft/")
        print("  ├── speech/    # Results from STFT speech model")
        print("  ├── music/     # Results from STFT music model")
        print("  └── general/   # Results from STFT general model")
        print("\nSTFT Configuration:")
        print(f"  Sample Rate: {STFTSettings.SAMPLE_RATE} Hz")
        print(f"  N_FFT: {STFTSettings.N_FFT}")
        print(f"  Hop Length: {STFTSettings.HOP_LENGTH}")
        print(f"  Frequency Bins: {STFTSettings.FREQ_BINS}")
        print("="*70 + "\n")
        return
    
    # Parse arguments
    input_path = Path(sys.argv[1])
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_stft"
    
    # Check if input exists
    if not input_path.exists():
        print(f"❌ Input not found: {input_path}")
        return
    
    # Create denoiser
    try:
        denoiser = MultiModelSTFTDenoiser(device=device)
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        print("Please train STFT models first using the training scripts:")
        print("  - python src/fm/model_stft/backshot_stft_speech.py")
        print("  - python src/fm/model_stft/backshot_stft_music.py")
        print("  - python src/fm/model_stft/backshot_stft_general.py")
        return
    
    # Process input
    if input_path.is_file():
        # Single file mode
        denoiser.denoise_file_all_models(input_path, output_dir)
    elif input_path.is_dir():
        # Directory mode
        denoiser.denoise_directory(input_path, output_dir)
    else:
        print(f"❌ Invalid input: {input_path}")
        print("Input must be a file or directory")


if __name__ == "__main__":
    main()
