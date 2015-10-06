"""
This file contains the STFT function and related helper functions.
"""
import numpy as np
import scipy

from scipy import signal

import pylab as plt
import seaborn as sns
sns.set_palette("deep", desat=.6)
COLORMAP = sns.diverging_palette(220, 20, n=7, as_cmap=True)

from numpy.fft import rfft, irfft

import string
from nt.utils.numpy_utils import segment_axis


def stft(time_signal, time_dim=None, size=1024, shift=256,
        window=signal.blackman, fading=True, window_length=None):
    """
    Calculates the short time Fourier transformation of a multi channel multi
    speaker time signal. It is able to add additional zeros for fade-in and
    fade out and should yield an STFT signal which allows perfect reconstruction.

    :param time_signal: multi channel time signal.
    :param time_dim: Scalar dim of time. Default: None means the biggest dimension
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift. Typically shift is a fraction of size.
    :param window: Window function handle.
    :param fading: Pads the signal with zeros for better reconstruction.
    :param window_length: Sometimes one desires to use a shorter window than
        the fft size. In that case, the window is padded with zeros.
        The default is to use the fft-size as a window size.
    :return: Single channel complex STFT signal
        with dimensions frames times size/2+1.
    """
    if time_dim is None:
        time_dim = np.argmax(time_signal.shape)

    # Pad with zeros to have enough samples for the window function to fade.
    if fading is True:
        pad = [(0, 0)]*time_signal.ndim
        pad[time_dim] = [size-shift, size-shift]
        time_signal = np.pad(time_signal, pad, mode='constant')

    # Pad with trailing zeros, to have an integral number of frames.
    frames = _samples_to_stft_frames(time_signal.shape[time_dim], size, shift)
    samples = _stft_frames_to_samples(frames, size, shift)
    pad = [(0, 0)]*time_signal.ndim
    pad[time_dim] = [0, samples - time_signal.shape[time_dim]]
    time_signal = np.pad(time_signal, pad, mode='constant')

    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = np.pad(window, (0, size-window_length), mode='constant')

    time_signal_seg = segment_axis(time_signal, size, size-shift, axis=time_dim)

    letters = string.ascii_lowercase
    mapping = letters[:time_signal_seg.ndim]+','+letters[time_dim+1]+'->'+letters[:time_signal_seg.ndim]

    # ToDo: Implement this more memory efficient
    return rfft(np.einsum(mapping, time_signal_seg, window), axis=time_dim+1)


def stft_single_channel(time_signal, size=1024, shift=256,
         window=signal.blackman, fading=True, window_length=None):
    """
    Calculates the short time Fourier transformation of a single channel time
    signal. It is able to add additional zeros for fade-in and fade out and
    should yield an STFT signal which allows perfect reconstruction.

    Up to now, only a single channel time signal is possible.

    :param time_signal: Single channel time signal.
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift. Typically shift is a fraction of size.
    :param window: Window function handle.
    :param fading: Pads the signal with zeros for better reconstruction.
    :param window_length: Sometimes one desires to use a shorter window than
        the fft size. In that case, the window is padded with zeros.
        The default is to use the fft-size as a window size.
    :return: Single channel complex STFT signal
        with dimensions frames times size/2+1.
    """
    assert len(time_signal.shape) == 1

    # Pad with zeros to have enough samples for the window function to fade.
    if fading is True:
        time_signal = np.pad(time_signal, size-shift, mode='constant')

    # Pad with trailing zeros, to have an integral number of frames.
    frames = _samples_to_stft_frames(len(time_signal), size, shift)
    samples = _stft_frames_to_samples(frames, size, shift)
    time_signal = np.pad(time_signal,
                         (0, samples - len(time_signal)), mode='constant')

    # The range object contains the sample index of the beginning of each frame.
    range_object = range(0, len(time_signal) - size + shift, shift)

    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = np.pad(window, (0, size-window_length), mode='constant')
    windowed = np.array([(window*time_signal[i:i+size]) for i in range_object])
    return rfft(windowed)


def _samples_to_stft_frames(samples, size, shift):
    """
    Calculates STFT frames from samples in time domain.
    :param samples: Number of samples in time domain.
    :param size: FFT size.
    :param shift: Hop in samples.
    :return: Number of STFT frames.
    """
    return np.ceil((samples - size + shift) / shift)


def _stft_frames_to_samples(frames, size, shift):
    """
    Calculates samples in time domain from STFT frames
    :param frames: Number of STFT frames.
    :param size: FFT size.
    :param shift: Hop in samples.
    :return: Number of samples in time domain.
    """
    return frames * shift + size - shift


def _biorthogonal_window_loopy(analysis_window, shift):
    """
    This version of the synthesis calculation is as close as possible to the
    Matlab impelementation in terms of variable names.

    The results are equal.

    The implementation follows equation A.92 in
    Krueger, A. Modellbasierte Merkmalsverbesserung zur robusten automatischen
    Spracherkennung in Gegenwart von Nachhall und Hintergrundstoerungen
    Paderborn, Universitaet Paderborn, Diss., 2011, 2011
    """
    fft_size = len(analysis_window)
    assert np.mod(fft_size, shift) == 0
    number_of_shifts = len(analysis_window) // shift

    sum_of_squares = np.zeros(shift)
    for synthesis_index in range(0, shift):
        for sample_index in range(0, number_of_shifts+1):
            analysis_index = synthesis_index + sample_index * shift

            if analysis_index + 1 < fft_size:
                sum_of_squares[synthesis_index] \
                    += analysis_window[analysis_index] ** 2

    sum_of_squares = np.kron(np.ones(number_of_shifts), sum_of_squares)
    synthesis_window = analysis_window / sum_of_squares / fft_size
    return synthesis_window


def _biorthogonal_window(analysis_window, shift):
    """
    This is a vectorized implementation of the window calculation. It is much
    slower than the variant using for loops.

    The implementation follows equation A.92 in
    Krueger, A. Modellbasierte Merkmalsverbesserung zur robusten automatischen
    Spracherkennung in Gegenwart von Nachhall und Hintergrundstoerungen
    Paderborn, Universitaet Paderborn, Diss., 2011, 2011
    """
    fft_size = len(analysis_window)
    assert np.mod(fft_size, shift) == 0
    number_of_shifts = len(analysis_window) // shift

    sum_of_squares = np.zeros(shift)
    for synthesis_index in range(0, shift):
        sample_index = np.arange(0, number_of_shifts+1)
        analysis_index = synthesis_index + sample_index * shift
        analysis_index = analysis_index[analysis_index + 1 < fft_size]
        sum_of_squares[synthesis_index] \
            = np.sum(analysis_window[analysis_index] ** 2)
    sum_of_squares = np.kron(np.ones(number_of_shifts), sum_of_squares)
    synthesis_window = analysis_window / sum_of_squares / fft_size
    return synthesis_window


def istft(stft_signal, size=1024, shift=256,
          window=signal.blackman, fading=True, window_length=None):
    """
    Calculated the inverse short time Fourier transform to exactly reconstruct
    the time signal.

    :param stft_signal: Single channel complex STFT signal
        with dimensions frames times size/2+1.
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift. Typically shift is a fraction of size.
    :param window: Window function handle.
    :param fading: Removes the additional padding, if done during STFT.
    :param window_length: Sometimes one desires to use a shorter window than
        the fft size. In that case, the window is padded with zeros.
        The default is to use the fft-size as a window size.
    :return: Single channel complex STFT signal
    :return: Single channel time signal.
    """
    assert stft_signal.shape[1] == 1024 // 2 + 1

    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = np.pad(window, (0, size-window_length), mode='constant')

    window = _biorthogonal_window_loopy(window, shift)

    # Why? Line created by Hai, Lukas does not know, why it exists.
    window = window * size

    time_signal = scipy.zeros(stft_signal.shape[0] * shift + size - shift)

    for j, i in enumerate(range(0, len(time_signal) - size + shift, shift)):
        time_signal[i:i + size] += window * np.real(irfft(stft_signal[j]))

    # Compensate fade-in and fade-out
    if fading:
        time_signal = time_signal[size-shift:len(time_signal)-(size-shift)]

    return time_signal


def stft_to_spectrogram(stft_signal):
    """
    Calculates the power spectrum (spectrogram) of an stft signal.
    The output is guaranteed to be real.

    :param stft: Complex STFT signal with dimensions
        #time_frames times #frequency_bins.
    :return: Real spectrogram with same dimensions as input.
    """
    spectrogram = np.abs(stft_signal * np.conjugate(stft_signal))
    return spectrogram


def plot_spectrogram(spectrogram, limits=None):
    """
    Plots a spectrogram from a spectrogram (power) as input.

    :param spectrogram: Real valued power spectrum
        with shape (frames, frequencies).
    :param limits: Color limits for clipping purposes.
    :return: None
    """
    if limits is None:
        limits = (np.min(spectrogram), np.max(spectrogram))

    plt.imshow(np.clip(np.log10(spectrogram).T, limits[0], limits[1]),
               interpolation='none', origin='lower', cmap=COLORMAP)
    plt.grid(False)
    plt.xlabel('Time frame')
    plt.ylabel('Frequency bin')
    cbar = plt.colorbar()
    cbar.set_label('Energy / dB')
    plt.show()


def plot_stft(stft_signal, limits=None):
    """
    Plots a spectrogram from an stft signal as input. This is a wrapper of the
    plot function for spectrograms.

    :param stft_signal: Complex valued stft signal.
    :param limits: Color limits for clipping purposes.
    :return: None
    """
    plot_spectrogram(stft_to_spectrogram(stft_signal), limits)


def spectrogram_to_energy_per_frame(spectrogram):
    """
    The energy per frame is sometimes used as an additional feature to the MFCC
    features. Here, it is caluclated from the power spectrum.

    :param spectrogram: Real valued power spectrum.
    :return: Real valued energy per frame.
    """
    energy = np.sum(spectrogram, 1)

    # If energy is zero, we get problems with log
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    return energy
