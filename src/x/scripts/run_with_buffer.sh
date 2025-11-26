#!/bin/bash
# Example: Run OFDM transmission with warm-up buffer frames
# This helps the receiver sync before actual payload arrives

echo "OFDM Transmission with Warm-up Buffer"
echo "====================================="
echo ""
echo "Buffer frames help receiver sync and skip startup transients."
echo "Each frame is ~128 bytes of random data."
echo ""

# Example: 3 warm-up frames + testfile_small.txt
python src/inference/main_inference.py \
  --mode ofdm \
  --data "src/inference/TxRx/content/testfile_small.txt" \
  --buffer-frames 3 \
  --tx-gain 0 \
  --rx-duration 5.0

echo ""
echo "Transmission complete. Check output/ for results."
