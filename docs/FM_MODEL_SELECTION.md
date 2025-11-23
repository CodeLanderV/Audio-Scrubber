# FM Model Selection Guide

## Overview
The system now supports dynamic FM model selection with multiple architectures and modes.

## Available Architectures

### 1. **1D U-Net** (`1dunet`)
- Time-domain audio processing
- Best for: General purpose, low-latency
- Model size: ~16 MB
- Input: Raw audio waveform

### 2. **STFT 2D U-Net** (`stft`)
- Frequency-domain (spectrogram) processing
- Best for: Music, complex noise patterns
- Model size: ~11 MB
- Input: STFT magnitude spectrogram

## Available Modes

- **`general`** - General purpose FM denoising
- **`music`** - Optimized for music broadcasts
- **`speech`** - Optimized for voice/talk radio

## Usage Examples

### Live Denoising (live_denoise.py)

#### List all available models:
```bash
python src/live_denoise.py --list-models
```

#### Use default (general, auto-architecture):
```bash
python src/live_denoise.py
```

#### Select specific mode:
```bash
python src/live_denoise.py --mode music
python src/live_denoise.py --mode speech
```

#### Force specific architecture:
```bash
# Use 1D U-Net for music
python src/live_denoise.py --mode music --architecture 1dunet

# Use STFT 2D for speech
python src/live_denoise.py --mode speech --architecture stft
```

#### Use specific model file:
```bash
python src/live_denoise.py --model saved_models/FM/FinalModels/FM_Final_STFT/music.pth
```

#### Run without AI (passthrough):
```bash
python src/live_denoise.py --passthrough
```

### SDR Inference (main_inference.py)

#### FM transmission with music model:
```bash
python src/inference/main_inference.py \
    --mode fm \
    --audio Tests/samples/music.wav \
    --fm-mode music \
    --fm-architecture stft
```

#### FM transmission with speech model (1D U-Net):
```bash
python src/inference/main_inference.py \
    --mode fm \
    --audio Tests/samples/speech.wav \
    --fm-mode speech \
    --fm-architecture 1dunet
```

#### FM with specific model file:
```bash
python src/inference/main_inference.py \
    --mode fm \
    --audio myaudio.wav \
    --model saved_models/FM/FinalModels/FM_Final_1DUNET/general.pth
```

## Model Auto-Detection

The system automatically searches for models in this order:

1. `saved_models/FM/FinalModels/FM_Final_1DUNET/`
2. `saved_models/FM/FinalModels/FM_Final_STFT/`
3. `saved_models/FM/models/`
4. `saved_models/FM/`

Architecture is detected from folder name:
- Folders with "1D", "1DUNET" → 1D U-Net
- Folders with "STFT", "2D" → STFT 2D U-Net

## Programmatic Usage

```python
from fm.model_loader import FMModelLoader, get_model_for_inference

# List all models
models = FMModelLoader.list_available_models()
FMModelLoader.print_available_models()

# Load model (auto-detect)
model, info = get_model_for_inference(
    mode='music',
    architecture='stft',
    device='cuda'
)

# Load specific file
model, info = FMModelLoader.load_model(
    model_path='saved_models/FM/FinalModels/FM_Final_STFT/music.pth',
    device='cpu'
)
```

## Model Information

When a model loads, you'll see:
```
✅ Model loaded!
   Architecture: stft
   Mode: music
   Size: 11.05 MB
   Val Loss: 1.505568
```

## Recommendations

| Use Case | Mode | Architecture |
|----------|------|--------------|
| General FM radio | `general` | Auto or `1dunet` |
| Music stations | `music` | `stft` |
| Talk radio/News | `speech` | `1dunet` |
| Low latency needed | Any | `1dunet` |
| Complex noise patterns | Any | `stft` |

## Troubleshooting

### "No FM model found"
```bash
# Check available models
python src/live_denoise.py --list-models

# If no models listed, train models first or check paths
```

### Model won't load
```bash
# Try forcing architecture
python src/live_denoise.py --mode general --architecture 1dunet

# Or specify exact file
python src/live_denoise.py --model saved_models/FM/FinalModels/FM_Final_1DUNET/general.pth
```

### Wrong architecture detected
```bash
# Always specify architecture explicitly
python src/live_denoise.py --mode music --architecture stft
```

## Adding New Models

1. Save your trained model (.pth file) to:
   - `saved_models/FM/FinalModels/FM_Final_1DUNET/` (for 1D U-Net)
   - `saved_models/FM/FinalModels/FM_Final_STFT/` (for STFT 2D)

2. Name it: `general.pth`, `music.pth`, or `speech.pth`

3. The system will auto-detect it on next run

## Performance Notes

- **1D U-Net**: Faster inference (~10ms/chunk), lower memory
- **STFT 2D**: Better quality, slower (~30ms/chunk), higher memory
- CUDA recommended for STFT models
- CPU sufficient for 1D U-Net models
