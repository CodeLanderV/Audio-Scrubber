"""
================================================================================
OFDM DENOISER TRAINING - CUDA OPTIMIZED
================================================================================
Trains 1D U-Net for OFDM waveform denoising with:
- Automatic CUDA/CPU detection
- Mixed precision training (faster on GPU)
- Memory-efficient streaming dataset
- Early stopping & checkpointing
- Progress tracking & visualization
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ofdm.model.neuralnet import OFDM_UNet


class OFDMDataset(Dataset):
    """Memory-efficient streaming dataset for IQ waveforms."""
    
    def __init__(self, clean_path, noisy_path, chunk_size=1024, max_chunks=None, preload=True):
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.chunk_size = chunk_size
        
        # Get file info
        clean_data = np.fromfile(clean_path, dtype=np.complex64)
        noisy_data = np.fromfile(noisy_path, dtype=np.complex64)
        
        assert len(clean_data) == len(noisy_data), "Clean and noisy files must match!"
        
        self.total_samples = len(clean_data)
        self.num_chunks = self.total_samples // chunk_size
        
        if max_chunks:
            self.num_chunks = min(self.num_chunks, max_chunks)
        
        # Preload if fits in RAM (< 4GB)
        file_size_gb = self.total_samples * 8 / (1024**3)
        self.preloaded = preload and file_size_gb < 4
        
        if self.preloaded:
            print(f"📦 Preloading dataset into RAM ({file_size_gb:.2f} GB)...")
            self.clean_data = clean_data
            self.noisy_data = noisy_data
        else:
            print(f"🌊 Using streaming mode ({file_size_gb:.2f} GB)")
        
        print(f"   Dataset: {self.num_chunks:,} chunks of {chunk_size} samples")
    
    def __len__(self):
        return self.num_chunks
    
    def __getitem__(self, idx):
        start = idx * self.chunk_size
        end = start + self.chunk_size
        
        if self.preloaded:
            clean = self.clean_data[start:end]
            noisy = self.noisy_data[start:end]
        else:
            # Stream from disk
            clean = np.fromfile(self.clean_path, dtype=np.complex64, 
                              count=self.chunk_size, offset=start*8)
            noisy = np.fromfile(self.noisy_path, dtype=np.complex64, 
                              count=self.chunk_size, offset=start*8)
        
        # Convert to [2, Length] format (I, Q channels)
        clean_tensor = torch.stack([
            torch.from_numpy(np.real(clean).astype(np.float32)),
            torch.from_numpy(np.imag(clean).astype(np.float32))
        ])
        
        noisy_tensor = torch.stack([
            torch.from_numpy(np.real(noisy).astype(np.float32)),
            torch.from_numpy(np.imag(noisy).astype(np.float32))
        ])
        
        return noisy_tensor, clean_tensor


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, use_amp=True):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for noisy, clean in pbar:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient
        
        if use_amp and scaler:
            with autocast():
                output = model(noisy)
                loss = criterion(output, clean)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for noisy, clean in tqdm(dataloader, desc="Validating", leave=False):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            
            output = model(noisy)
            loss = criterion(output, clean)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def plot_training_history(train_losses, val_losses, save_path):
    """Plot training curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('OFDM Denoiser Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Training plot saved: {save_path}")


def train_model(
    clean_path,
    noisy_path,
    epochs=100,
    batch_size=32,
    chunk_size=1024,
    lr=0.001,
    val_split=0.1,
    save_dir='saved_models/OFDM',
    use_amp=True,
    early_stopping_patience=20,
    num_workers=4
):
    """Main training loop with CUDA optimization."""
    
    print("="*80)
    print(" "*20 + "OFDM DENOISER TRAINING")
    print("="*80)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
    else:
        print("   ⚠️  No CUDA available - training will be slow!")
        use_amp = False
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    print(f"   Clean: {clean_path}")
    print(f"   Noisy: {noisy_path}")
    
    full_dataset = OFDMDataset(clean_path, noisy_path, chunk_size=chunk_size)
    
    # Split train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training: {train_size:,} chunks ({train_size * chunk_size:,} samples)")
    print(f"   Validation: {val_size:,} chunks ({val_size * chunk_size:,} samples)")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if device.type == 'cuda' and num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    print(f"\n🧠 Initializing model...")
    model = OFDM_UNet(in_channels=2, out_channels=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024**2:.2f} MB")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Chunk size: {chunk_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Val split: {val_split*100:.0f}%")
    print(f"   Early stopping patience: {early_stopping_patience}")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print("="*80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n📈 Results:")
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Val Loss:   {val_loss:.6f}")
        print(f"   Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            improvement = (best_val_loss - val_loss) / best_val_loss * 100 if best_val_loss != float('inf') else 0
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'chunk_size': chunk_size,
                'best_val_loss': best_val_loss
            }
            
            best_path = save_dir / 'ofdm_unet_best.pth'
            torch.save(checkpoint, best_path)
            print(f"   ✅ Best model saved ({improvement:.1f}% improvement): {best_path.name}")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{early_stopping_patience})")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path.name}")
    
    # Plot training history
    plot_path = save_dir / 'training_history.png'
    plot_training_history(train_losses, val_losses, plot_path)
    
    # Final summary
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📊 Final Results:")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    print(f"   Total epochs: {len(train_losses)}")
    print(f"   Best model: {save_dir / 'ofdm_unet_best.pth'}")
    print(f"   Training plot: {plot_path}")
    print(f"\n🎓 Ready for inference!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Train OFDM denoising model')
    
    parser.add_argument('--clean-data', type=str, required=True,
                       help='Path to clean IQ file')
    parser.add_argument('--noisy-data', type=str, required=True,
                       help='Path to noisy IQ file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--chunk-size', type=int, default=1024,
                       help='Chunk size in samples (default: 1024)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split (default: 0.1)')
    parser.add_argument('--save-dir', type=str, default='saved_models/OFDM',
                       help='Save directory (default: saved_models/OFDM)')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (default: 20)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loader workers (default: 4)')
    
    args = parser.parse_args()
    
    train_model(
        clean_path=args.clean_data,
        noisy_path=args.noisy_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        lr=args.lr,
        val_split=args.val_split,
        save_dir=args.save_dir,
        use_amp=not args.no_amp,
        early_stopping_patience=args.patience,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()
