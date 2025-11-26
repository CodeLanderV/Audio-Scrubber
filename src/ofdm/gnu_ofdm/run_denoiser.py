import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

# --- HARDCODED PATHS (YOURS) ---
# Input: The noisy file you recorded with GNU Radio
CAPTURE_FILE = r"D:\Bunker\OneDrive - Amrita vishwa vidyapeetham\BaseCamp\AudioScrubber\src\ofdm\captured.iq"

# Output: The clean file this script will create
DENOISED_FILE = r"D:\Bunker\OneDrive - Amrita vishwa vidyapeetham\BaseCamp\AudioScrubber\src\ofdm\denoised.iq"

# Model Path
MODEL_PATH = r"saved_models\OFDM\final_models\gnu_radio_model.pth"

# --- CONFIG ---
CHUNK_SIZE = 16384
SAMPLE_RATE = 2e6

# --- MODEL DEFINITION (UNet1D) ---
class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UNet1D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(UNet1D, self).__init__()
        self.enc1 = Conv1DBlock(in_channels, 32)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = Conv1DBlock(32, 64)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = Conv1DBlock(64, 128)
        self.pool3 = nn.MaxPool1d(2)
        self.enc4 = Conv1DBlock(128, 256)
        self.pool4 = nn.MaxPool1d(2)
        self.bottleneck = Conv1DBlock(256, 512)
        self.upconv4 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec4 = Conv1DBlock(512, 256)
        self.upconv3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec3 = Conv1DBlock(256, 128)
        self.upconv2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec2 = Conv1DBlock(128, 64)
        self.upconv1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec1 = Conv1DBlock(64, 32)
        self.final = nn.Conv1d(32, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.dec4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.upconv1(dec2), enc1), dim=1))
        return self.final(dec1)

def main():
    # 1. Load IQ Data
    if not os.path.exists(CAPTURE_FILE):
        print(f"File not found: {CAPTURE_FILE}")
        return

    print(f"Loading {CAPTURE_FILE}...")
    raw_signal = np.fromfile(CAPTURE_FILE, dtype=np.complex64)
    print(f"Loaded {len(raw_signal)} samples.")

    # 2. Prep Data
    num_chunks = len(raw_signal) // CHUNK_SIZE
    valid_len = num_chunks * CHUNK_SIZE
    raw_signal = raw_signal[:valid_len]
    
    real = raw_signal.real.reshape(num_chunks, CHUNK_SIZE)
    imag = raw_signal.imag.reshape(num_chunks, CHUNK_SIZE)
    tensor_input = np.stack([real, imag], axis=1)
    tensor_in = torch.from_numpy(tensor_input).float()

    # 3. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet1D(in_channels=2, out_channels=2).to(device)
    
    try:
        # Resolve path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up 3 levels to root (AudioScrubber), then down to saved_models
        full_model_path = os.path.abspath(os.path.join(script_dir, "../../..", MODEL_PATH))
        
        print(f"Loading model from: {full_model_path}")
        checkpoint = torch.load(full_model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return

    # 4. Run Inference (Whole file or large chunk)
    model.eval()
    tensor_in = tensor_in.to(device)
    
    print("Denoising...")
    with torch.no_grad():
        # Process ALL chunks (or batch if memory low)
        # If you have a GPU, processing all at once is usually fine for <100MB files
        # If OOM, slice this: tensor_in[:100]
        clean_out = model(tensor_in) 
        clean_out = clean_out.cpu().numpy()

    # 5. Reconstruct & SAVE
    flat_real = clean_out[:, 0, :].flatten()
    flat_imag = clean_out[:, 1, :].flatten()
    clean_signal = flat_real + 1j * flat_imag
    
    print(f"Saving denoised file to: {DENOISED_FILE}")
    clean_signal.astype(np.complex64).tofile(DENOISED_FILE)
    print("✅ Saved.")

    # 6. Plot Comparison
    noisy_slice = raw_signal[:100000]
    clean_slice = clean_signal[:100000]

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.title("Original Noisy Input (RTL-SDR)")
    plt.specgram(noisy_slice, NFFT=1024, Fs=SAMPLE_RATE)
    plt.subplot(2, 1, 2)
    plt.title("AI Denoised Output")
    plt.specgram(clean_slice, NFFT=1024, Fs=SAMPLE_RATE)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
