"""
RTL-SDR Troubleshooting Script
Diagnoses RTL-SDR hardware and driver issues
"""
import sys
from pathlib import Path

print("\n" + "="*70)
print("RTL-SDR TROUBLESHOOTING DIAGNOSTIC")
print("="*70 + "\n")

# Step 1: Check pyrtlsdr installation
print("[1] Checking pyrtlsdr installation...")
try:
    from rtlsdr import RtlSdr
    import rtlsdr
    print("    ✓ pyrtlsdr is installed")
    print(f"    ✓ Version: {rtlsdr.__version__ if hasattr(rtlsdr, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"    ✗ FAILED: {e}")
    print("    Install with: pip install pyrtlsdr")
    sys.exit(1)

# Step 2: Check libusb
print("\n[2] Checking libusb...")
try:
    import usb1
    print("    ✓ libusb is available")
except ImportError:
    print("    ✗ libusb not found (pyUSB)")
    print("    Install with: pip install pyusb")

# Step 3: Try to detect RTL-SDR devices
print("\n[3] Scanning for RTL-SDR devices...")
try:
    import usb.core
    devices = usb.core.find(find_all=True, idVendor=0x0bda, idProduct=0x2838)
    device_list = list(devices)
    
    if device_list:
        print(f"    ✓ Found {len(device_list)} RTL-SDR device(s)")
        for i, dev in enumerate(device_list):
            print(f"      Device {i}: {dev.manufacturer} {dev.product}")
    else:
        print("    ✗ No RTL-SDR devices found via USB")
        print("    → Check if device is connected")
        print("    → Try different USB port")
except Exception as e:
    print(f"    ✗ USB scan failed: {e}")

# Step 4: Check device permissions
print("\n[4] Testing RTL-SDR device access...")
try:
    sdr = RtlSdr()
    print("    ✓ RTL-SDR device accessible!")
    print(f"    ✓ Sample rate: {sdr.sample_rate}")
    print(f"    ✓ Gain values: {sdr.gain_values}")
    sdr.close()
except Exception as e:
    error_str = str(e)
    print(f"    ✗ FAILED: {error_str}")
    
    if "Access denied" in error_str or "LIBUSB_ERROR_ACCESS" in error_str:
        print("\n    💡 ACCESS DENIED - This is a driver issue!")
        print("    Solution: Reinstall Zadig driver with WinUSB")
        print("    Steps:")
        print("      1. Download Zadig from: https://zadig.akeo.ie/")
        print("      2. Connect RTL-SDR device")
        print("      3. Run Zadig as Administrator")
        print("      4. Options → List All Devices")
        print("      5. Find 'Realtek RTL2832U EEPROM' or 'RTL2832 EEPROM'")
        print("      6. Driver: WinUSB → Install Driver")
        print("      7. Restart this script")
    elif "No such device" in error_str:
        print("\n    💡 DEVICE NOT FOUND - Check connection!")
        print("    Steps:")
        print("      1. Unplug RTL-SDR from all USB ports")
        print("      2. Wait 5 seconds")
        print("      3. Plug into a different USB port (preferably USB 2.0)")
        print("      4. Wait for driver to load")
        print("      5. Run this script again")
    else:
        print(f"\n    💡 Unknown error: {error_str}")

# Step 5: Check config files
print("\n[5] Checking configuration...")
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import AudioSettings, RTLSDRSettings, Paths
    print("    ✓ config.py loaded successfully")
    print(f"    ✓ Sample rate: {AudioSettings.SAMPLE_RATE}")
    print(f"    ✓ Model path: {Paths.MODEL_BEST}")
except Exception as e:
    print(f"    ✗ Config error: {e}")

# Step 6: Check saved model
print("\n[6] Checking AI model...")
try:
    model_path = Path(__file__).parent / "saved_models" / "unet1d_best.pth"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024*1024)
        print(f"    ✓ Model found: {model_path}")
        print(f"    ✓ Size: {size_mb:.1f} MB")
    else:
        print(f"    ✗ Model not found at: {model_path}")
except Exception as e:
    print(f"    ✗ Model check failed: {e}")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70 + "\n")
