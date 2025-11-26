#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: satya
# GNU Radio version: 3.10.12.0

from gnuradio import blocks
import pmt
from gnuradio import digital
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import threading




class file_process(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 2000000

        ##################################################
        # Blocks
        ##################################################

        self.digital_ofdm_rx_0_1 = digital.ofdm_rx(
            fft_len=64, cp_len=16,
            frame_length_tag_key='frame_'+"packet_len",
            packet_length_tag_key="packet_len",
            occupied_carriers=((-4, -3, -2, -1, 1, 2, 3, 4),),
            pilot_carriers=((-21, -7, 7, 21,),),
            pilot_symbols=((1, 1, 1, -1),),
            sync_word1=None,
            sync_word2=None,
            bps_header=2,
            bps_payload=2,
            debug_log=False,
            scramble_bits=False)
        self.digital_ofdm_rx_0_1.set_min_output_buffer(65536)
        self.digital_ofdm_rx_0 = digital.ofdm_rx(
            fft_len=64, cp_len=16,
            frame_length_tag_key='frame_'+"packet_len",
            packet_length_tag_key="packet_len",
            occupied_carriers=((-4, -3, -2, -1, 1, 2, 3, 4),),
            pilot_carriers=((-21, -7, 7, 21,),),
            pilot_symbols=((1, 1, 1, -1),),
            sync_word1=None,
            sync_word2=None,
            bps_header=2,
            bps_payload=2,
            debug_log=False,
            scramble_bits=False)
        self.digital_ofdm_rx_0.set_min_output_buffer(65536)
        self.digital_crc32_bb_1_0 = digital.crc32_bb(True, "packet_len", True)
        self.digital_crc32_bb_1_0.set_min_output_buffer(65536)
        self.digital_crc32_bb_1 = digital.crc32_bb(True, "packet_len", True)
        self.digital_crc32_bb_1.set_min_output_buffer(65536)
        self.blocks_throttle2_1 = blocks.throttle( gr.sizeof_gr_complex*1, samp_rate, True, 0 if "auto" == "auto" else max( int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1) )
        self.blocks_throttle2_0 = blocks.throttle( gr.sizeof_gr_complex*1, samp_rate, True, 0 if "auto" == "auto" else max( int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1) )
        self.blocks_file_source_0_0 = blocks.file_source(gr.sizeof_gr_complex*1, 'D:\\Bunker\\OneDrive - Amrita vishwa vidyapeetham\\BaseCamp\\AudioScrubber\\src\\ofdm\\gnu_ofdm\\denoised.iq', True, 0, 0)
        self.blocks_file_source_0_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, 'D:\\Bunker\\OneDrive - Amrita vishwa vidyapeetham\\BaseCamp\\AudioScrubber\\src\\ofdm\\gnu_ofdm\\captured.iq', True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_sink_1 = blocks.file_sink(gr.sizeof_char*1, 'D:\\Bunker\\OneDrive - Amrita vishwa vidyapeetham\\BaseCamp\\AudioScrubber\\Tests\\denoised.jpg.png', False)
        self.blocks_file_sink_1.set_unbuffered(False)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_char*1, 'D:\\Bunker\\OneDrive - Amrita vishwa vidyapeetham\\BaseCamp\\AudioScrubber\\Tests\\captured.jpg.png', False)
        self.blocks_file_sink_0.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.blocks_throttle2_0, 0))
        self.connect((self.blocks_file_source_0_0, 0), (self.blocks_throttle2_1, 0))
        self.connect((self.blocks_throttle2_0, 0), (self.digital_ofdm_rx_0, 0))
        self.connect((self.blocks_throttle2_1, 0), (self.digital_ofdm_rx_0_1, 0))
        self.connect((self.digital_crc32_bb_1, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.digital_crc32_bb_1_0, 0), (self.blocks_file_sink_1, 0))
        self.connect((self.digital_ofdm_rx_0, 0), (self.digital_crc32_bb_1, 0))
        self.connect((self.digital_ofdm_rx_0_1, 0), (self.digital_crc32_bb_1_0, 0))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle2_0.set_sample_rate(self.samp_rate)
        self.blocks_throttle2_1.set_sample_rate(self.samp_rate)




def main(top_block_cls=file_process, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.flowgraph_started.set()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
