"""
Configuration for STFT-based U-Net 2D Models
Created: 22/11/2025

STFT Parameters and Model Configuration for Spectrogram-based Denoising
"""

from pathlib import Path

# --- PROJECT PATHS ---
PROJECT_ROOT = Path(__file__).parent  # config_stft.py is in project root
DATASET_ROOT = PROJECT_ROOT / "dataset"

class Paths:
    """File paths for STFT models"""
    PROJECT_ROOT = PROJECT_ROOT
    DATASET_ROOT = DATASET_ROOT
    
    # Audio datasets
    LIBRISPEECH_ROOT = DATASET_ROOT / "LibriSpeech"
    LIBRISPEECH_DEV_CLEAN = LIBRISPEECH_ROOT / "dev-clean"
    LIBRISPEECH_DEV_OTHER = LIBRISPEECH_ROOT / "dev-other"
    MUSIC_ROOT = DATASET_ROOT / "music"
    
    # Noise file
    NOISE_FM = DATASET_ROOT / "noise" / "superNoiseFM.wav"
    
    # Model save paths
    MODEL_STFT_SPEECH_BEST = PROJECT_ROOT / "saved_models" / "stft_speech_100_norm" / "unet2d_best_speech.pth"
    MODEL_STFT_MUSIC_BEST = PROJECT_ROOT / "saved_models" / "stft_music_100_norm" / "unet2d_best_music.pth"
    MODEL_STFT_GENERAL_BEST = PROJECT_ROOT / "saved_models" / "stft_general_100_norm" / "unet2d_best_general.pth"


class STFTSettings:
    """
    STFT (Short-Time Fourier Transform) Parameters
    
    These parameters control how audio is converted to spectrograms:
    - n_fft: FFT window size (larger = better frequency resolution, worse time resolution)
    - hop_length: How much the window moves each step (smaller = better time resolution)
    - win_length: Window size for STFT (usually same as n_fft)
    - window: Window function ('hann' is standard for audio)
    """
    
    # STFT parameters
    N_FFT = 2048              # FFT size (gives 1025 frequency bins)
    HOP_LENGTH = 512          # Hop size (75% overlap is standard)
    WIN_LENGTH = 2048         # Window length (usually = n_fft)
    WINDOW = 'hann'           # Window type
    
    # Derived values
    FREQ_BINS = N_FFT // 2 + 1  # = 1025 frequency bins (0 to 22050 Hz)
    
    # Audio settings
    SAMPLE_RATE = 44100       # 44.1 kHz (CD quality)
    AUDIO_LENGTH = 88192      # ~2 seconds at 44.1 kHz (same as 1D models)
    
    # Expected time frames for AUDIO_LENGTH
    # time_frames = (AUDIO_LENGTH - N_FFT) // HOP_LENGTH + 1
    TIME_FRAMES = (AUDIO_LENGTH - N_FFT) // HOP_LENGTH + 1  # ≈ 168 frames


class TrainingSettings:
    """Training hyperparameters for STFT models"""
    
    # Training parameters
    NUM_EPOCHS = 100
    BATCH_SIZE = 8           # STFT needs more memory, reduce from 16
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.01
    
    # Optimizer settings
    OPTIMIZER = 'AdamW'
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-7
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 20
    
    # Data split
    TRAIN_SPLIT = 0.8        # 80% training, 20% validation
    
    # SNR settings (same as 1D models)
    SNR_RANGE_MIN = 5.0      # Minimum SNR in dB
    SNR_RANGE_MAX = 20.0     # Maximum SNR in dB
    RMS_NORMALIZATION = 0.1  # Normalize both clean and noise to RMS=0.1
    
    # Device
    DEVICE = 'cuda'          # or 'cpu'
    NUM_WORKERS = 4          # DataLoader workers


class ModelSettings:
    """U-Net 2D architecture settings"""
    
    IN_CHANNELS = 1          # Input channels (magnitude spectrogram)
    OUT_CHANNELS = 1         # Output channels (clean magnitude spectrogram)
    
    # U-Net channels at each level
    ENCODER_CHANNELS = [16, 32, 64, 128, 256]
    
    # Conv2D settings
    KERNEL_SIZE = 3
    PADDING = 1


# Quick info
if __name__ == "__main__":
    print("="*70)
    print("STFT U-Net 2D Configuration")
    print("="*70)
    print(f"\nSTFT Parameters:")
    print(f"  N_FFT: {STFTSettings.N_FFT}")
    print(f"  Hop Length: {STFTSettings.HOP_LENGTH}")
    print(f"  Frequency Bins: {STFTSettings.FREQ_BINS}")
    print(f"  Time Frames (for 88192 samples): {STFTSettings.TIME_FRAMES}")
    print(f"  Sample Rate: {STFTSettings.SAMPLE_RATE} Hz")
    print(f"  Audio Length: {STFTSettings.AUDIO_LENGTH} samples (~{STFTSettings.AUDIO_LENGTH/STFTSettings.SAMPLE_RATE:.2f}s)")
    
    print(f"\nTraining Settings:")
    print(f"  Epochs: {TrainingSettings.NUM_EPOCHS}")
    print(f"  Batch Size: {TrainingSettings.BATCH_SIZE}")
    print(f"  Learning Rate: {TrainingSettings.LEARNING_RATE}")
    print(f"  SNR Range: {TrainingSettings.SNR_RANGE_MIN}-{TrainingSettings.SNR_RANGE_MAX} dB")
    print(f"  RMS Normalization: {TrainingSettings.RMS_NORMALIZATION}")
    
    print(f"\nModel Paths:")
    print(f"  Speech: {Paths.MODEL_STFT_SPEECH_BEST}")
    print(f"  Music: {Paths.MODEL_STFT_MUSIC_BEST}")
    print(f"  General: {Paths.MODEL_STFT_GENERAL_BEST}")
    
    print(f"\nInput Shape: (batch, 1, {STFTSettings.FREQ_BINS}, {STFTSettings.TIME_FRAMES})")
    print("="*70)
