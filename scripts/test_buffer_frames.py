#!/usr/bin/env python3
"""Test buffer frames feature: run transmission with/without buffer and compare results.

Outputs:
  - test_results_no_buffer.txt : Results without buffer frames
  - test_results_with_buffer.txt : Results with 3 buffer frames
  - comparison_summary.txt : Before/after comparison
"""

import subprocess
import os
import sys
from pathlib import Path
import time

# Setup paths
repo_root = Path(__file__).resolve().parent.parent
os.chdir(repo_root)

test_file = "src/inference/TxRx/content/testfile_small.txt"
output_dir = "output/buffer_frames_test"

if not Path(test_file).exists():
    print(f"❌ Test file not found: {test_file}")
    sys.exit(1)

os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("BUFFER FRAMES TEST - Comparing Before/After")
print("=" * 80)
print()

# Test 1: WITHOUT buffer
print("📡 TEST 1: Transmission WITHOUT buffer frames...")
print("-" * 80)
cmd1 = [
    sys.executable, "src/inference/main_inference.py",
    "--mode", "ofdm",
    "--data", test_file,
    "--buffer-frames", "0",
    "--tx-gain", "0"
]

with open(f"{output_dir}/test_results_no_buffer.txt", "w") as f:
    result1 = subprocess.run(cmd1, stdout=f, stderr=subprocess.STDOUT, text=True)

with open(f"{output_dir}/test_results_no_buffer.txt", "r") as f:
    output1 = f.read()
    # Print summary
    for line in output1.split('\n')[-20:]:
        if line.strip():
            print(f"  {line}")

print()
time.sleep(2)

# Test 2: WITH buffer
print("📡 TEST 2: Transmission WITH 3 buffer frames...")
print("-" * 80)
cmd2 = [
    sys.executable, "src/inference/main_inference.py",
    "--mode", "ofdm",
    "--data", test_file,
    "--buffer-frames", "3",
    "--tx-gain", "0"
]

with open(f"{output_dir}/test_results_with_buffer.txt", "w") as f:
    result2 = subprocess.run(cmd2, stdout=f, stderr=subprocess.STDOUT, text=True)

with open(f"{output_dir}/test_results_with_buffer.txt", "r") as f:
    output2 = f.read()
    # Print summary
    for line in output2.split('\n')[-20:]:
        if line.strip():
            print(f"  {line}")

print()

# Generate comparison
print("=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

def extract_ber(output):
    """Extract BER info from output."""
    lines = output.split('\n')
    for line in lines:
        if 'BER' in line or 'errors' in line or 'success' in line:
            return line.strip()
    return "Could not extract BER"

summary = f"""
WITHOUT BUFFER FRAMES (baseline):
  {extract_ber(output1)}

WITH 3 BUFFER FRAMES (improved):
  {extract_ber(output2)}

Recommendation:
  1. Compare the BER/error rates above.
  2. Buffer frames are MOST USEFUL if:
     - First few blocks show high error rates without buffer.
     - With buffer, early blocks show significant improvement.
  3. If no improvement, the issue is not startup transients.
     - Check: TX power, RX gain, clock sync, sample-rate mismatch.
  4. If improvement, use buffer-frames=2 or buffer-frames=3 for regular transmission.

Files:
  - {output_dir}/test_results_no_buffer.txt
  - {output_dir}/test_results_with_buffer.txt
  - {output_dir}/comparison_summary.txt
"""

with open(f"{output_dir}/comparison_summary.txt", "w") as f:
    f.write(summary)

print(summary)

print()
print("✅ Test complete. Check output/ directory for detailed logs.")
