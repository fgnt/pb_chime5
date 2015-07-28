"""
This module deals with all sorts of audio input and output.
"""
from nt.transform import fbank, filter, mfcc, ssc, stft

fbank = fbank.fbank

preemphasis = filter.preemphasis

mfcc = mfcc.mfcc

ssc = ssc.ssc

stft = stft.stft
istft = stft.istft
stft_to_spectrogram = stft.stft_to_spectrogram
plot_spectrogram = stft.plot_spectrogram
plot_stft = stft.plot_stft
spectrogram_to_energy_per_frame = stft.spectrogram_to_energy_per_frame
