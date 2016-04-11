"""
This module deals with all sorts of acoustic features and transforms.
"""
from nt.transform.module_stft import stft, spectrogram
from nt.transform.module_stft import istft
from nt.transform.module_stft import stft_to_spectrogram
from nt.transform.module_stft import spectrogram_to_energy_per_frame
from nt.transform.module_stft import get_stft_center_frequencies
from nt.transform.module_filter import preemphasis
from nt.transform.module_filter import inverse_preemphasis
from nt.transform.module_fbank import fbank
from nt.transform.module_filter import offset_compensation
from nt.transform.module_mfcc import mfcc, mfcc_velocity_acceleration
from nt.transform.module_ssc import ssc
from nt.transform.module_bark_fbank import bark_fbank
from nt.transform.module_rastaplp import rasta_plp
from nt.transform.module_ams import ams
import numpy as np


def normalize_mean_variance(data, axis=0, eps=1e-6):
    """ Normalize features.

    :param data: Any feature
    :param axis: Time dimensions, default is 0
    :return: Normalized observation
    """
    return (data - np.mean(data, axis=axis, keepdims=True)) /\
        (np.std(data, axis=axis, keepdims=True) + eps)
