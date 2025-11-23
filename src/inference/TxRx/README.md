# SDR TX/RX System with AI Denoising

Complete rewrite of the transmission/reception system with modular architecture.

## 📁 File Structure

```
src/
├── inference/
│   ├── main_inference.py       # Main entry point
│   └── plot/                    # Generated plots saved here
│
└── ofdm/TxRx/
    ├── sdr_base.py              # Hardware classes (PlutoSDR, RTLSDR)
    ├── ofdm_modulation.py       # OFDM modulation with AI denoising
    ├── fm_modulation.py         # FM modulation with AI denoising
    └── sdr_utils.py             # Utilities (plotting, data conversion)
```

## 🏗️ Architecture

### 1. Hardware Classes (`sdr_base.py`)

#### **PlutoSDR**
- `check_device()` - Detect Adalm Pluto
- `configure(freq, rate, gain)` - Configure TX parameters
- `transmit_background(waveform)` - Start cyclic transmission
- `stop()` - Stop transmission

#### **RTLSDR**
- `check_device()` - Detect RTL-SDR
- `configure(freq, rate, gain)` - Configure RX parameters
- `receive(duration)` - Capture IQ samples
- `stop()` - Close device

**Shared Config Dictionary:**
```python
SDR_CONFIG = {
    'center_freq': 915e6,      # 915 MHz ISM
    'sample_rate': 2e6,        # 2 MSPS
    'tx_gain': -10,            # dB
    'rx_gain': 'auto',
    'pluto_ip': 'ip:192.168.2.1'
}
```

### 2. Modulation Classes

#### **OFDM_Modulation** (`ofdm_modulation.py`)
```python
modulation = OFDM_Modulation(
    use_ai=True,              # Enable AI denoising
    model_path=None,          # Auto-detect from saved_models/OFDM/
    passthrough=False         # Skip AI if True
)

# Transmit
waveform = modulation.modulate(data_bytes)

# Receive
result = modulation.demodulate(rx_waveform)
# Returns: {'data', 'control_data', 'stats', 'waveform_denoised'}
```

**Features:**
- Auto model loading from `saved_models/OFDM/final_models/`
- Control path (no AI) for comparison
- Comprehensive plotting at each stage
- Constellation diagrams (noisy vs denoised)

**Plots Generated:**
- `BeforeTX_OFDM_<modelname>_waveform.png`
- `BeforeTX_OFDM_<modelname>_constellation.png`
- `AfterRX_OFDM_<modelname>_noisy.png`
- `AfterRX_OFDM_<modelname>_denoised.png`
- `AfterRX_OFDM_<modelname>_constellation_noisy.png`
- `AfterRX_OFDM_<modelname>_constellation_denoised.png`

#### **FM_Modulation** (`fm_modulation.py`)
```python
modulation = FM_Modulation(
    use_ai=True,
    model_path=None,           # Auto-detect from saved_models/FM/
    passthrough=False,
    mode='general'             # 'general', 'music', or 'speech'
)

# Transmit
waveform = modulation.modulate(audio_file_or_array)

# Receive
result = modulation.demodulate(rx_waveform)
# Returns: {'audio', 'control_audio', 'denoised_audio', 'stats'}
```

**Features:**
- Auto model loading: `saved_models/FM/FM_Final_1DUNET/{mode}.pth`
- 75 kHz FM deviation (wideband FM)
- Audio rate: 44.1 kHz
- Auto resampling between audio/SDR rates
- Similar plotting to `live_denoise.py`

**Plots Generated:**
- `BeforeTX_FM_<modelname>_waveform.png`
- `FM_Analysis_<modelname>.png` (noisy vs denoised comparison)

### 3. Utilities (`sdr_utils.py`)

**SDRUtils** class provides:
- `load_data(file_path)` - Load any file type
- `save_data(data, output_path)` - Save data
- `plot_waveform(waveform, title, filename)` - Time + frequency plot
- `plot_constellation(symbols, title, filename)` - Constellation diagram
- `plot_fm_analysis(audio, demodulated, denoised, ...)` - FM analysis
- `normalize_power(waveform, target_power)` - Power normalization
- `scale_for_sdr(waveform, max_amplitude)` - Prevent clipping
- `calculate_snr(clean, noisy)` - SNR calculation

All plots saved to `src/inference/plot/` with Agg backend (no window display).

## 🚀 Usage Examples

### OFDM Transmission (with AI)
```bash
python src/inference/main_inference.py \
    --mode ofdm \
    --data dataset/instant/clean/sample.txt \
    --rx-duration 5 \
    --output received_data.bin
```

### OFDM Transmission (passthrough, no AI)
```bash
python src/inference/main_inference.py \
    --mode ofdm \
    --data dataset/instant/clean/sample.txt \
    --passthrough \
    --rx-duration 5
```

### FM Transmission (speech model)
```bash
python src/inference/main_inference.py \
    --mode fm \
    --audio Tests/samples/audio.wav \
    --fm-mode speech \
    --rx-duration 10
```

### FM Transmission (music model, custom frequency)
```bash
python src/inference/main_inference.py \
    --mode fm \
    --audio Tests/samples/music.wav \
    --fm-mode music \
    --freq 433 \
    --tx-gain 0 \
    --rx-duration 15
```

### Custom Model Selection
```bash
python src/inference/main_inference.py \
    --mode ofdm \
    --data myfile.bin \
    --model saved_models/OFDM/final_models/custom_model.pth
```

## 📊 Workflow

### OFDM Workflow
1. Load data bytes
2. **Modulate:** Data → Bits → QPSK → OFDM (IFFT + CP)
3. Plot: Waveform + Constellation (before TX)
4. **Transmit:** Pluto SDR cyclic transmission
5. **Receive:** RTL-SDR capture
6. **Control Path:** Direct demodulation (no AI)
7. **AI Path:** Denoise → Demodulate
8. Plot: Noisy vs Denoised (waveform + constellation)
9. Save decoded data + stats

### FM Workflow
1. Load audio file
2. **Modulate:** Audio → Resample → FM (phase modulation)
3. Plot: FM waveform (before TX)
4. **Transmit:** Pluto SDR cyclic transmission
5. **Receive:** RTL-SDR capture
6. **Demodulate:** Phase difference → Audio
7. **AI Denoise:** Neural network denoising (if enabled)
8. Plot: Noisy vs Denoised (time + frequency)
9. Save denoised audio (WAV)

## 🎯 Key Features

✅ **Modular Architecture** - Clean separation of concerns  
✅ **Multi-Model Support** - Auto-detect models from saved_models/  
✅ **Passthrough Mode** - Test without AI (--passthrough flag)  
✅ **Control Path** - Always run non-AI path for comparison  
✅ **Comprehensive Plotting** - Every stage visualized and saved  
✅ **Shared Config** - SDR_CONFIG dict for TX/RX synchronization  
✅ **Error Handling** - Graceful fallbacks if AI fails  
✅ **CLI Interface** - Easy command-line usage  

## 🔧 Configuration

Edit `SDR_CONFIG` in `sdr_base.py` to change defaults:
```python
SDR_CONFIG = {
    'center_freq': 915e6,      # Change frequency
    'sample_rate': 2e6,        # Change sample rate
    'tx_gain': -10,            # Change TX gain
    'rx_gain': 'auto',         # Change RX gain
    'pluto_ip': 'ip:192.168.2.1',
    'buffer_size': 65536,
}
```

Or use CLI arguments:
```bash
--freq 433            # 433 MHz
--tx-gain 0           # 0 dB
--rx-duration 10      # 10 seconds
```

## 📈 Model Auto-Detection

### OFDM Models
Searches in order:
1. `saved_models/OFDM/final_models/*best*.pth`
2. `saved_models/OFDM/final_models/ofdm_final*.pth`
3. `saved_models/OFDM/ofdm_unet_best.pth`

### FM Models
Searches:
1. `saved_models/FM/FM_Final_1DUNET/{mode}.pth` (mode = general/music/speech)
2. `saved_models/FM/FM_Final_1DUNET/*.pth` (any model)

## 🐛 Debugging

Check plots in `src/inference/plot/`:
- Before TX plots show clean modulated signal
- After RX plots show noisy + denoised comparison

Enable passthrough mode to test modulation without AI:
```bash
--passthrough
```

Compare control path vs AI path in console output:
```
📊 Statistics Comparison:
   Control: BER=0.0250, Errors=100/4000
   AI:      BER=0.0050, Errors=20/4000
```

## ⚠️ Known Issues

1. **RTL-SDR Buffer Errors** - Fixed with chunked reading (256K chunks)
2. **100% Packet Error** - Increase TX gain or move antennas closer:
   ```bash
   --tx-gain 0  # Instead of default -10 dB
   ```
3. **Model Not Found** - Specify manual path:
   ```bash
   --model saved_models/OFDM/ofdm_unet_best.pth
   ```

## 📝 Next Steps

1. **Train Production Models:**
   ```bash
   cd src/ofdm/model
   python generate_dataset.py  # 50M samples
   python train.py --epochs 150
   ```

2. **Test Hardware Loopback:**
   ```bash
   # Increase TX gain, move antennas closer
   python src/inference/main_inference.py --mode ofdm --data test.txt --tx-gain 0
   ```

3. **Optimize Models:**
   - Train on more data (100M+ samples)
   - Try different SNR ranges
   - Fine-tune for specific data types

4. **Add Features:**
   - Real-time streaming
   - Multi-channel support
   - Adaptive equalization
   - FEC (Forward Error Correction)
