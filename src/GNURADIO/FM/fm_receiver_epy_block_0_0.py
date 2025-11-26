"""
Embedded Python Block for FM Audio Denoising
"""

import numpy as np
from gnuradio import gr
import torch
import sys
from pathlib import Path

# --- CONFIGURATION (Update these paths!) ---
# Path to your TRAINED FM MODEL
MODEL_PATH = r"D:\Bunker\OneDrive - Amrita vishwa vidyapeetham\BaseCamp\AudioScrubber\saved_models\FM\FinalModels\FM_Final_STFT\speech.pth"

# Path to your SOURCE CODE (where neuralnet.py is)
SRC_PATH = r"D:\Bunker\OneDrive - Amrita vishwa vidyapeetham\BaseCamp\AudioScrubber\src\fm\neuralnets"

# Add src to system path
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# Import your model architecture
try:
    from neuralnet import UNet2D_STFT  # Assuming class name is UNet1D for FM
except ImportError:
    print("AI Block Error: Could not import UNet1D. Check sys.path.")

class fm_ai_denoiser(gr.sync_block):  
    """
    FM AI Denoiser
    Takes float audio samples, runs them through 1D U-Net, outputs clean audio.
    """

    def __init__(self, chunk_size=16384): # 16000 is typical for speech models
        gr.sync_block.__init__(
            self,
            name='FM AI Denoiser',
            in_sig=[np.float32],  # Audio is Float (Orange)
            out_sig=[np.float32]  # Output is Float (Orange)
        )
        self.chunk_size = chunk_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"AI Block: Loading FM model from {MODEL_PATH}...")
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            # Initialize model (1 Channel for Audio)
            self.model = UNet1D(in_channels=1, out_channels=1).to(self.device)
            
            # Load weights
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.eval()
            print("AI Block: FM Model loaded successfully.")
            
        except Exception as e:
            print(f"AI Block Error: {e}")
            print("AI Block: Running in PASSTHROUGH mode.")
            self.model = None

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out0 = output_items[0]
        n_samples = len(in0)
        
        # Passthrough if model failed
        if self.model is None:
            out0[:] = in0
            return n_samples

        # Process in chunks
        num_chunks = n_samples // self.chunk_size
        valid_length = num_chunks * self.chunk_size
        
        if num_chunks > 0:
            # 1. Reshape
            data_chunks = in0[:valid_length].reshape(num_chunks, self.chunk_size)
            
            # 2. Tensor Prep (Batch, Channel, Length)
            # Add channel dimension: (Num_Chunks, 1, Chunk_Size)
            tensor_input = torch.from_numpy(data_chunks).float().unsqueeze(1).to(self.device)
            
            # 3. Inference
            with torch.no_grad():
                output_tensor = self.model(tensor_input)
            
            # 4. Output
            # Remove channel dimension and flatten
            clean_audio = output_tensor.squeeze(1).cpu().numpy().flatten()
            out0[:valid_length] = clean_audio
            
        # Pass remaining samples
        if valid_length < n_samples:
            out0[valid_length:] = in0[valid_length:]

        return n_samples