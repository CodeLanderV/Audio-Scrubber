import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import soundfile as sf

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config_stft import Paths, STFTSettings, TrainingSettings, ModelSettings
from src.fm.model_stft.stft_unet2d import UNet2D_STFT

"""
==============================================================================
STFT U-Net 2D Training Script - MUSIC MODEL
==============================================================================

Created: 22/11/2025

Trains STFT-based U-Net 2D on Music dataset (407 music files)
Uses spectrogram representation for better frequency-domain denoising

Key Features:
- On-the-fly STFT conversion (audio -> spectrogram)
- Magnitude + phase processing
- SNR normalization (RMS=0.1 for clean and noise before mixing)
- Real FM noise from superNoiseFM.wav
- Early stopping with checkpoints

Output: saved_models/stft_music_100_norm/unet2d_best_music.pth
==============================================================================
"""

# --- Configuration ---
class Config:
    # Paths
    CLEAN_AUDIO_DIRS = [
        str(Paths.MUSIC_ROOT),
    ]
    NOISE_FILE = str(Paths.NOISE_FM)
    MODEL_SAVE_PATH = str(Paths.MODEL_STFT_MUSIC_BEST)
    CHECKPOINT_DIR = str(Paths.MODEL_STFT_MUSIC_BEST.parent / "checkpoints")
    
    # STFT parameters
    N_FFT = STFTSettings.N_FFT
    HOP_LENGTH = STFTSettings.HOP_LENGTH
    WIN_LENGTH = STFTSettings.WIN_LENGTH
    WINDOW = STFTSettings.WINDOW
    
    # Audio parameters
    SAMPLE_RATE = STFTSettings.SAMPLE_RATE
    AUDIO_LENGTH = STFTSettings.AUDIO_LENGTH
    
    # Training hyperparameters
    BATCH_SIZE = TrainingSettings.BATCH_SIZE
    NUM_EPOCHS = TrainingSettings.NUM_EPOCHS
    LEARNING_RATE = TrainingSettings.LEARNING_RATE
    WEIGHT_DECAY = TrainingSettings.WEIGHT_DECAY
    TRAIN_SPLIT = TrainingSettings.TRAIN_SPLIT
    
    # SNR settings
    SNR_MIN = TrainingSettings.SNR_RANGE_MIN
    SNR_MAX = TrainingSettings.SNR_RANGE_MAX
    RMS_NORM = TrainingSettings.RMS_NORMALIZATION
    
    # Model parameters
    IN_CHANNELS = ModelSettings.IN_CHANNELS
    OUT_CHANNELS = ModelSettings.OUT_CHANNELS
    
    # Device
    DEVICE = TrainingSettings.DEVICE
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = TrainingSettings.EARLY_STOPPING_PATIENCE


class OnFlySTFTNoiseDataset(Dataset):
    """
    Dataset that generates noisy audio on-the-fly using STFT processing.
    
    Process:
    1. Load clean audio chunk
    2. Normalize clean audio to RMS=0.1
    3. Load noise chunk and normalize to RMS=0.1
    4. Mix at target SNR
    5. Convert noisy audio to STFT magnitude spectrogram
    6. Convert clean audio to STFT magnitude spectrogram
    
    Returns: (noisy_spectrogram, clean_spectrogram) pairs
    Shape: (1, freq_bins, time_frames)
    """
    
    def __init__(self, clean_dirs, noise_file, audio_length, sample_rate,
                 n_fft, hop_length, win_length, window='hann',
                 snr_min=5.0, snr_max=20.0, rms_norm=0.1):
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.rms_norm = rms_norm
        
        # Load noise file once
        print(f"Loading FM noise from: {noise_file}")
        self.noise, _ = sf.read(noise_file)
        
        # Convert stereo to mono if needed
        if len(self.noise.shape) > 1:
            self.noise = np.mean(self.noise, axis=1)
        
        print(f"Noise loaded: {len(self.noise)} samples, {len(self.noise)/sample_rate:.2f} seconds")
        
        # Collect all clean audio files (mp3, wav, flac for music)
        print("Scanning for clean audio files...")
        self.clean_files = []
        for clean_dir in clean_dirs:
            clean_path = Path(clean_dir)
            for ext in ['*.flac', '*.wav', '*.mp3']:
                audio_files = list(clean_path.rglob(ext))
                self.clean_files.extend(audio_files)
        
        print(f"Found {len(self.clean_files)} clean audio files")
    
    def __len__(self):
        return len(self.clean_files)
    
    def _audio_to_stft_magnitude(self, audio):
        """
        Convert audio waveform to STFT magnitude spectrogram.
        
        Args:
            audio: Audio array (audio_length,)
        
        Returns:
            magnitude: STFT magnitude spectrogram (freq_bins, time_frames)
            phase: STFT phase for reconstruction (freq_bins, time_frames)
        """
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window
        )
        
        # Extract magnitude and phase
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        return magnitude, phase
    
    def _add_noise_snr(self, clean, noise, snr_db):
        """
        Add noise to clean audio at specified SNR.
        
        IMPORTANT: Normalizes both signals to standard RMS (0.1) BEFORE calculating SNR.
        This ensures consistent SNR meaning across all audio files.
        
        Steps:
        1. Normalize clean audio to RMS = 0.1
        2. Normalize noise chunk to RMS = 0.1
        3. Calculate target signal and noise powers based on desired SNR
        4. Scale noise to achieve target SNR
        5. Mix clean + scaled noise
        
        Args:
            clean: Clean audio array
            noise: Noise audio array (same length as clean)
            snr_db: Target SNR in decibels
        
        Returns:
            noisy: Mixed audio with specified SNR
        """
        # Step 1: Normalize clean audio to RMS = 0.1
        clean_rms = np.sqrt(np.mean(clean ** 2)) + 1e-8
        clean_normalized = clean * (self.rms_norm / clean_rms)
        
        # Step 2: Normalize noise chunk to RMS = 0.1
        noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-8
        noise_normalized = noise * (self.rms_norm / noise_rms)
        
        # Step 3: Calculate target signal and noise powers
        # SNR = 10 * log10(P_signal / P_noise)
        # P_noise = P_signal / (10^(SNR/10))
        clean_power = np.mean(clean_normalized ** 2)
        noise_power_target = clean_power / (10 ** (snr_db / 10))
        
        # Step 4: Scale noise to achieve target SNR
        current_noise_power = np.mean(noise_normalized ** 2)
        noise_scale = np.sqrt(noise_power_target / (current_noise_power + 1e-8))
        noise_scaled = noise_normalized * noise_scale
        
        # Step 5: Mix clean and scaled noise
        noisy = clean_normalized + noise_scaled
        
        return noisy
    
    def __getitem__(self, idx):
        """
        Generate one training sample:
        - Load clean audio
        - Add noise at random SNR
        - Convert both to STFT magnitude spectrograms
        
        Returns:
            noisy_mag: Noisy magnitude spectrogram (1, freq_bins, time_frames)
            clean_mag: Clean magnitude spectrogram (1, freq_bins, time_frames)
        """
        # Load clean audio
        clean_path = self.clean_files[idx]
        clean_audio, sr = librosa.load(str(clean_path), sr=self.sample_rate)
        
        # Ensure audio is exactly audio_length
        if len(clean_audio) > self.audio_length:
            # Random crop
            start = np.random.randint(0, len(clean_audio) - self.audio_length)
            clean_audio = clean_audio[start:start + self.audio_length]
        elif len(clean_audio) < self.audio_length:
            # Pad with zeros
            clean_audio = np.pad(clean_audio, (0, self.audio_length - len(clean_audio)))
        
        # Get random noise chunk
        max_start = len(self.noise) - self.audio_length
        if max_start <= 0:
            # Noise file too short, tile it
            num_tiles = (self.audio_length // len(self.noise)) + 1
            noise_chunk = np.tile(self.noise, num_tiles)[:self.audio_length]
        else:
            start = np.random.randint(0, max_start)
            noise_chunk = self.noise[start:start + self.audio_length]
        
        # Random SNR
        snr_db = np.random.uniform(self.snr_min, self.snr_max)
        
        # Add noise at specified SNR (with normalization)
        noisy_audio = self._add_noise_snr(clean_audio, noise_chunk, snr_db)
        
        # Convert to STFT magnitude spectrograms
        noisy_mag, noisy_phase = self._audio_to_stft_magnitude(noisy_audio)
        clean_mag, clean_phase = self._audio_to_stft_magnitude(clean_audio)
        
        # Convert to tensors and add channel dimension
        noisy_mag = torch.FloatTensor(noisy_mag).unsqueeze(0)  # (1, freq_bins, time_frames)
        clean_mag = torch.FloatTensor(clean_mag).unsqueeze(0)
        
        return noisy_mag, clean_mag


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    with tqdm(dataloader, desc="Training") as pbar:
        for noisy_mag, clean_mag in pbar:
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predicted_mag = model(noisy_mag)
            
            # Loss (MSE between predicted and clean magnitude spectrograms)
            loss = criterion(predicted_mag, clean_mag)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Validation") as pbar:
            for noisy_mag, clean_mag in pbar:
                noisy_mag = noisy_mag.to(device)
                clean_mag = clean_mag.to(device)
                
                # Forward pass
                predicted_mag = model(noisy_mag)
                
                # Loss
                loss = criterion(predicted_mag, clean_mag)
                total_loss += loss.item()
                pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})
    
    return total_loss / len(dataloader)


def main():
    """Main training loop"""
    config = Config()
    
    print("="*70)
    print("STFT U-Net 2D Training - MUSIC MODEL")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"STFT: n_fft={config.N_FFT}, hop={config.HOP_LENGTH}")
    print(f"SNR Range: {config.SNR_MIN}-{config.SNR_MAX} dB")
    print(f"RMS Normalization: {config.RMS_NORM}")
    print("="*70 + "\n")
    
    # Create dataset
    print("Creating dataset...")
    dataset = OnFlySTFTNoiseDataset(
        clean_dirs=config.CLEAN_AUDIO_DIRS,
        noise_file=config.NOISE_FILE,
        audio_length=config.AUDIO_LENGTH,
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        window=config.WINDOW,
        snr_min=config.SNR_MIN,
        snr_max=config.SNR_MAX,
        rms_norm=config.RMS_NORM
    )
    
    # Train/val split
    train_size = int(config.TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    model = UNet2D_STFT(in_channels=config.IN_CHANNELS, 
                       out_channels=config.OUT_CHANNELS).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, 
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Create save directory
    Path(config.MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    print("="*70 + "\n")
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Learning Rate: {current_lr:.2e}\n")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, config.MODEL_SAVE_PATH)
            print(f"✅ Best model saved! Val Loss: {val_loss:.6f}\n")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}\n")
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(config.CHECKPOINT_DIR) / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('STFT U-Net 2D Training History (Music)')
    plt.legend()
    plt.grid(True)
    plot_path = Path(config.MODEL_SAVE_PATH).parent / "training_history.png"
    plt.savefig(plot_path)
    print(f"\n✅ Training plot saved to: {plot_path}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print("="*70)


if __name__ == "__main__":
    main()
