"""
This module deals with all sorts of audio input and output.
"""

from nt.transform import filter
from nt.transform import fbank
# from nt.transform import mfcc
# from nt.transform import ssc
from nt.transform import mod_stft

# filter = filter
# fbank = fbank.fbank
#
# preemphasis = filter.preemphasis
#
# mfcc = mfcc.mfcc
#
# ssc = ssc.ssc
#
stft = mod_stft.stft
# istft = stft.istft
# stft_to_spectrogram = stft.stft_to_spectrogram
# plot_spectrogram = stft.plot_spectrogram
# plot_stft = stft.plot_stft
# spectrogram_to_energy_per_frame = stft.spectrogram_to_energy_per_frame
