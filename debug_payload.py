"""Debug payload accuracy calculation"""
import sys
import numpy as np
sys.path.append('src')
sys.path.append('.')

# Test payload error calculation logic
test_data = b'HELLO'

# Simulate what happens in modulation
length_header = len(test_data).to_bytes(4, 'big')  # 4 bytes: 00 00 00 05
full_packet = length_header + test_data
print(f"Full packet: {full_packet.hex()}")
print(f"  Header (4 bytes): {full_packet[:4].hex()} (length={len(test_data)})")
print(f"  Payload (5 bytes): {full_packet[4:].hex()}")

# Generate original bits
original_bits = np.unpackbits(np.frombuffer(full_packet, dtype=np.uint8))
print(f"\nOriginal bits: {len(original_bits)} bits")
print(f"  Header bits (32): {original_bits[:32]}")
print(f"  Payload bits (40): {original_bits[32:]}")

# In demodulation, we receive the same
received_bits = original_bits.copy()  # Perfect channel
received_bytes = np.packbits(received_bits).tobytes()

print(f"\nReceived bytes: {received_bytes.hex()}")
print(f"  Header (4 bytes): {received_bytes[:4].hex()}")
print(f"  Payload (5 bytes): {received_bytes[4:].hex()}")

# Now compute payload accuracy like in code
header_size = 4
orig_payload_bits = original_bits[header_size*8:]
print(f"\nOriginal payload bits: {orig_payload_bits}")
orig_payload_bytes = np.packbits(orig_payload_bits).tobytes()
print(f"Repacked payload bytes: {orig_payload_bytes.hex()}")
print(f"Received payload bytes: {received_bytes[header_size:].hex()}")

# Compare
payload_errors = 0
for i, (orig, recv) in enumerate(zip(orig_payload_bytes, received_bytes[header_size:])):
    if orig != recv:
        print(f"  Byte {i}: {orig:02x} != {recv:02x}")
        payload_errors += 1

print(f"\nPayload errors: {payload_errors}/{len(received_bytes[header_size:])}")
print(f"Accuracy: {100.0 * (1 - payload_errors / len(received_bytes[header_size:])):.1f}%")
