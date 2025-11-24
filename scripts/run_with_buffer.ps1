# Example: Run OFDM transmission with warm-up buffer frames
# This helps the receiver sync and skip startup transients
# Each frame is ~128 bytes of random data

Write-Host "OFDM Transmission with Warm-up Buffer" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Buffer frames help receiver sync and skip startup transients." -ForegroundColor Yellow
Write-Host "Each frame is ~128 bytes of random data."
Write-Host ""

# Example: 3 warm-up frames + testfile_small.txt
$env:PYTHONIOENCODING='utf-8'
cd "d:\Bunker\OneDrive - Amrita vishwa vidyapeetham\BaseCamp\AudioScrubber"

python src/inference/main_inference.py `
  --mode ofdm `
  --data "src/inference/TxRx/content/testfile_small.txt" `
  --buffer-frames 3 `
  --tx-gain 0 `
  --rx-duration 5.0

Write-Host ""
Write-Host "Transmission complete. Check output/ for results." -ForegroundColor Green
