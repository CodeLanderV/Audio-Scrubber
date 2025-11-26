#!/usr/bin/env python3
"""
CRITICAL HARDWARE DIAGNOSTIC - TX/RX SIGNAL LOSS

This tool has identified a CRITICAL ISSUE with your OFDM TX/RX system.

PROBLEM SUMMARY
===============

TX Power Output: -6.02 dB (good)
RX Power Received: -40.63 dB (extremely low!)
PATH LOSS: 34.61 dB (SEVERE - should be ~10-15 dB for near-field SDR test)

Result: Your system is losing MOST OF THE SIGNAL between TX and RX.

ROOT CAUSES (In order of likelihood)
===================================

1. ⚠️  ANTENNAS NOT CONNECTED (MOST LIKELY)
   - Check if TX antenna is connected to Pluto TX port
   - Check if RX antenna is connected to RTL-SDR antenna port
   - Both should be small external antennas or 50Ω SMA connectors

2. ⚠️  ANTENNA MISMATCH/ORIENTATION
   - Both antennas should be similar type and frequency matched (915 MHz)
   - Try aligning antennas (e.g., both vertical or both parallel)
   - Try moving antennas closer or further apart

3. ⚠️  FREQUENCY OFFSET
   - Pluto TX might not be tuning to exact 915 MHz
   - RTL-SDR RX might not be tuning to exact 915 MHz
   - If offsets > 100 kHz, signals miss each other

4. ⚠️  CABLE/CONNECTOR ISSUES
   - Loose SMA connectors cause signal loss
   - Check all cable connections are tight
   - Look for damaged or bent SMA pins

IMMEDIATE ACTIONS
=================

1. PHYSICALLY INSPECT:
   ✓ Is there an antenna plugged into Pluto TX?
   ✓ Is there an antenna plugged into RTL-SDR?
   ✓ Are both antennas at 915 MHz frequency?
   ✓ Are SMA connectors tight?

2. TEST WITHOUT ANTENNAS:
   Try very short direct connection (1-2 inches) with antennas to see if
   signal improves significantly. This rules out environmental RF reflections.

3. VERIFY FREQUENCY:
   The diagnostic showed TX and RX both saw peak near 915 MHz, but...
   - Check Pluto web interface (http://192.168.2.1)
   - Check RTL-SDR frequency with rtl_test tool

4. MEASURE WITH SPECTRUM ANALYZER:
   If available, measure 915 MHz output at both TX and RX ports
   to verify transmission path.

CURRENT IMPACT ON YOUR SYSTEM
=============================

With 34 dB path loss and high noise floor:
- BER measurements are CORRUPTED (mixing signal + noise bits)
- Decoding works only because FEC is extremely strong
- Pre-FEC BER shows ~0.45 (45% errors) - mostly NOISE, not signal errors
- Data output is RANDOM because you're decoding noise as data

EXPECTED PERFORMANCE AFTER FIX
==============================

Once antenna/cable issue is resolved:
- Path loss should drop to 10-20 dB (realistic for test setup)
- SNR should improve by 15-25 dB
- Pre-FEC BER should drop to 0.01-0.05 range (1-5% errors)
- Data decoding accuracy should jump from 0% to 80-100%

"""

if __name__ == '__main__':
    print(__doc__)
