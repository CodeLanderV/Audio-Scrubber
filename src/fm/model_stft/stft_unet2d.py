import torch
from torch import nn

"""
Created: 22/11/2025

STFT-based U-Net 2D for Audio Denoising

Architecture designed for spectrogram processing:
- Input: STFT magnitude spectrogram (batch, 1, freq_bins, time_frames)
- Output: Clean STFT magnitude spectrogram (same shape)
- Better for frequency-domain noise (FM interference, hum, etc.)

Key advantages over 1D U-Net:
1. Can "see" which specific frequencies are noisy
2. Better preserves harmonic structure in music
3. Better preserves formants in speech
4. More effective for stationary noise patterns
"""

class Conv2DBlock(nn.Module):
    """
    2D Convolution Block: Conv2D -> BatchNorm2D -> ReLU
    
    Used in both encoder and decoder paths of the U-Net.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UNet2D_STFT(nn.Module):
    """
    2D U-Net for STFT-based audio denoising.
    
    Architecture:
    - Encoder: 4 downsampling layers (16 -> 32 -> 64 -> 128 channels)
    - Bottleneck: 256 channels
    - Decoder: 4 upsampling layers with skip connections
    - Output: Single channel magnitude spectrogram
    
    Input shape: (batch_size, 1, freq_bins, time_frames)
    - freq_bins: 1025 (for n_fft=2048)
    - time_frames: ~168 (for 88192 samples with hop=512)
    
    Output shape: (batch_size, 1, freq_bins, time_frames)
    """

    def __init__(self, in_channels=1, out_channels=1):
        super(UNet2D_STFT, self).__init__()
        
        # --- ENCODER (Downsampling Path) ---
        # Each layer doubles channels and halves spatial dimensions
        self.enc1 = Conv2DBlock(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # /2 in both dims
        
        self.enc2 = Conv2DBlock(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = Conv2DBlock(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = Conv2DBlock(64, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- BOTTLENECK ---
        self.bottleneck = Conv2DBlock(128, 256, kernel_size=3, padding=1)
        
        # --- DECODER (Upsampling Path) ---
        # Each layer halves channels and doubles spatial dimensions
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = Conv2DBlock(256, 128, kernel_size=3, padding=1)  # 256 = 128 + 128 (skip)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = Conv2DBlock(128, 64, kernel_size=3, padding=1)   # 128 = 64 + 64 (skip)
        
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = Conv2DBlock(64, 32, kernel_size=3, padding=1)    # 64 = 32 + 32 (skip)
        
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = Conv2DBlock(32, 16, kernel_size=3, padding=1)    # 32 = 16 + 16 (skip)
        
        # --- OUTPUT LAYER ---
        # 1x1 convolution to get back to 1 channel
        self.out = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the 2D U-Net.
        
        Args:
            x: Input magnitude spectrogram (batch, 1, freq_bins, time_frames)
        
        Returns:
            Clean magnitude spectrogram (batch, 1, freq_bins, time_frames)
        """
        # --- ENCODER ---
        enc1 = self.enc1(x)        # Save for skip connection
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)        # Save for skip connection
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)        # Save for skip connection
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)        # Save for skip connection
        x = self.pool4(enc4)
        
        # --- BOTTLENECK ---
        x = self.bottleneck(x)
        
        # --- DECODER ---
        # Upsample and concatenate with encoder features (skip connections)
        x = self.upconv4(x)
        # Handle size mismatch from pooling
        if x.shape[2:] != enc4.shape[2:]:
            x = self._match_size(x, enc4)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        if x.shape[2:] != enc3.shape[2:]:
            x = self._match_size(x, enc3)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        if x.shape[2:] != enc2.shape[2:]:
            x = self._match_size(x, enc2)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        if x.shape[2:] != enc1.shape[2:]:
            x = self._match_size(x, enc1)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # --- OUTPUT ---
        x = self.out(x)
        
        return x
    
    def _match_size(self, x, target):
        """
        Pad or crop x to match target size.
        
        Args:
            x: Tensor to adjust
            target: Target tensor with desired size
        
        Returns:
            Adjusted tensor matching target's spatial dimensions
        """
        if x.shape[2] < target.shape[2] or x.shape[3] < target.shape[3]:
            # Need to pad
            pad_h = target.shape[2] - x.shape[2]
            pad_w = target.shape[3] - x.shape[3]
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        elif x.shape[2] > target.shape[2] or x.shape[3] > target.shape[3]:
            # Need to crop
            x = x[:, :, :target.shape[2], :target.shape[3]]
        
        return x


# --- Quick Test ---
if __name__ == "__main__":
    print("="*70)
    print("Testing STFT U-Net 2D Architecture")
    print("="*70)
    
    # Create dummy STFT spectrogram input
    # Shape: (batch=2, channels=1, freq_bins=1025, time_frames=168)
    # This matches our config: n_fft=2048 -> 1025 freq bins, 88192 samples -> ~168 frames
    batch_size = 2
    freq_bins = 1025      # n_fft//2 + 1 = 2048//2 + 1
    time_frames = 168     # (88192 - 2048) // 512 + 1
    
    dummy_input = torch.randn(batch_size, 1, freq_bins, time_frames)
    
    # Create the model
    model = UNet2D_STFT(in_channels=1, out_channels=1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Verify output matches input
    assert output.shape == dummy_input.shape, "Output shape mismatch!"
    
    print("\n✅ STFT U-Net 2D is working correctly!")
    print("="*70)
