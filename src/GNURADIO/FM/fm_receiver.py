#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: FM Receiver
# Author: Dusan Stokic
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import audio
from gnuradio import blocks
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import fm_receiver_epy_block_0 as epy_block_0  # embedded python block
import fm_receiver_epy_block_0_0 as epy_block_0_0  # embedded python block
import fm_receiver_epy_block_0_1 as epy_block_0_1  # embedded python block
import fm_receiver_epy_block_0_2 as epy_block_0_2  # embedded python block
import fm_receiver_epy_block_0_3 as epy_block_0_3  # embedded python block
import osmosdr
import time
import sip
import threading



class fm_receiver(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "FM Receiver", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("FM Receiver")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "fm_receiver")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.volume = volume = 0
        self.tuner = tuner = 87000000
        self.samp_rate = samp_rate = 2000000
        self.rfgain = rfgain = 15
        self.down_rate = down_rate = 250000

        ##################################################
        # Blocks
        ##################################################

        self._volume_range = qtgui.Range(0, 100, 1, 0, 200)
        self._volume_win = qtgui.RangeWidget(self._volume_range, self.set_volume, "Volume", "slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._volume_win, 0, 1, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._tuner_range = qtgui.Range(87000000, 108000000, 100000, 87000000, 200)
        self._tuner_win = qtgui.RangeWidget(self._tuner_range, self.set_tuner, "Station Select", "counter_slider", int, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._tuner_win, 1, 0, 1, 2)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._rfgain_range = qtgui.Range(10, 70, 5, 15, 20)
        self._rfgain_win = qtgui.RangeWidget(self._rfgain_range, self.set_rfgain, "RF Gain", "slider", int, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._rfgain_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.rtlsdr_source_0 = osmosdr.source(
            args="numchan=" + str(1) + " " + 'str(grgsm.device.get_default_args(args))'
        )
        self.rtlsdr_source_0.set_sample_rate(samp_rate)
        self.rtlsdr_source_0.set_center_freq(tuner, 0)
        self.rtlsdr_source_0.set_freq_corr(0, 0)
        self.rtlsdr_source_0.set_dc_offset_mode(2, 0)
        self.rtlsdr_source_0.set_iq_balance_mode(2, 0)
        self.rtlsdr_source_0.set_gain_mode(True, 0)
        self.rtlsdr_source_0.set_gain(rfgain, 0)
        self.rtlsdr_source_0.set_if_gain(20, 0)
        self.rtlsdr_source_0.set_bb_gain(20, 0)
        self.rtlsdr_source_0.set_antenna('', 0)
        self.rtlsdr_source_0.set_bandwidth(0, 0)
        self.rational_resampler_xxx_0 = filter.rational_resampler_fff(
                interpolation=24,
                decimation=250,
                taps=[],
                fractional_bw=0)
        self.qtgui_freq_sink_x_0_3 = qtgui.freq_sink_f(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            tuner, #fc
            samp_rate, #bw
            'Sampling Bandpass', #name
            2,
            None # parent
        )
        self.qtgui_freq_sink_x_0_3.set_update_time(0.1)
        self.qtgui_freq_sink_x_0_3.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0_3.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0_3.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0_3.enable_autoscale(False)
        self.qtgui_freq_sink_x_0_3.enable_grid(True)
        self.qtgui_freq_sink_x_0_3.set_fft_average(0.05)
        self.qtgui_freq_sink_x_0_3.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0_3.enable_control_panel(False)
        self.qtgui_freq_sink_x_0_3.set_fft_window_normalized(False)


        self.qtgui_freq_sink_x_0_3.set_plot_pos_half(not True)

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(2):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0_3.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0_3.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0_3.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0_3.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0_3.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_3_win = sip.wrapinstance(self.qtgui_freq_sink_x_0_3.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_3_win)
        self.low_pass_filter_0 = filter.fir_filter_ccf(
            (int(samp_rate/down_rate)),
            firdes.low_pass(
                2,
                samp_rate,
                100000,
                10000,
                window.WIN_HAMMING,
                6.76))
        self.epy_block_0_3 = epy_block_0_3.fm_ai_denoiser(chunk_size=16384)
        self.epy_block_0_2 = epy_block_0_2.fm_ai_denoiser(chunk_size=16384)
        self.epy_block_0_1 = epy_block_0_1.fm_ai_denoiser(chunk_size=16384)
        self.epy_block_0_0 = epy_block_0_0.fm_ai_denoiser(chunk_size=16384)
        self.epy_block_0 = epy_block_0.fm_ai_denoiser(chunk_size=16384)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_ff((volume/100))
        self.audio_sink_0_0_3 = audio.sink(24000, '', True)
        self.audio_sink_0_0_2 = audio.sink(24000, '', True)
        self.audio_sink_0_0_1 = audio.sink(24000, '', True)
        self.audio_sink_0_0_0 = audio.sink(24000, '', True)
        self.audio_sink_0_0 = audio.sink(24000, '', True)
        self.audio_sink_0 = audio.sink(24000, '', True)
        self.analog_wfm_rcv_0 = analog.wfm_rcv(
        	quad_rate=down_rate,
        	audio_decimation=1,
        )


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_wfm_rcv_0, 0), (self.rational_resampler_xxx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.audio_sink_0, 0))
        self.connect((self.epy_block_0, 0), (self.audio_sink_0_0, 0))
        self.connect((self.epy_block_0, 0), (self.qtgui_freq_sink_x_0_3, 0))
        self.connect((self.epy_block_0_0, 0), (self.audio_sink_0_0_0, 0))
        self.connect((self.epy_block_0_1, 0), (self.audio_sink_0_0_1, 0))
        self.connect((self.epy_block_0_2, 0), (self.audio_sink_0_0_2, 0))
        self.connect((self.epy_block_0_3, 0), (self.audio_sink_0_0_3, 0))
        self.connect((self.low_pass_filter_0, 0), (self.analog_wfm_rcv_0, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.epy_block_0, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.epy_block_0_0, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.epy_block_0_1, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.epy_block_0_2, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.epy_block_0_3, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.qtgui_freq_sink_x_0_3, 1))
        self.connect((self.rtlsdr_source_0, 0), (self.low_pass_filter_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "fm_receiver")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_volume(self):
        return self.volume

    def set_volume(self, volume):
        self.volume = volume
        self.blocks_multiply_const_vxx_0.set_k((self.volume/100))

    def get_tuner(self):
        return self.tuner

    def set_tuner(self, tuner):
        self.tuner = tuner
        self.qtgui_freq_sink_x_0_3.set_frequency_range(self.tuner, self.samp_rate)
        self.rtlsdr_source_0.set_center_freq(self.tuner, 0)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.low_pass_filter_0.set_taps(firdes.low_pass(2, self.samp_rate, 100000, 10000, window.WIN_HAMMING, 6.76))
        self.qtgui_freq_sink_x_0_3.set_frequency_range(self.tuner, self.samp_rate)
        self.rtlsdr_source_0.set_sample_rate(self.samp_rate)

    def get_rfgain(self):
        return self.rfgain

    def set_rfgain(self, rfgain):
        self.rfgain = rfgain
        self.rtlsdr_source_0.set_gain(self.rfgain, 0)

    def get_down_rate(self):
        return self.down_rate

    def set_down_rate(self, down_rate):
        self.down_rate = down_rate




def main(top_block_cls=fm_receiver, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
