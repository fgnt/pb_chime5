"""
This module deals with all sorts of acoustic features and transforms.
"""
from nt.transform.module_stft import (
    stft,
    istft,
    spectrogram,
    stft_to_spectrogram,
    spectrogram_to_energy_per_frame,
    get_stft_center_frequencies,
)

# from pb_chime5.nt.transform.module_filter import preemphasis
# from pb_chime5.nt.transform.module_filter import inverse_preemphasis
# from pb_chime5.nt.transform.module_filter import offset_compensation
# from pb_chime5.nt.transform.module_filter import preemphasis_with_offset_compensation
# from pb_chime5.nt.transform.module_fbank import fbank, logfbank
# from pb_chime5.nt.transform.module_mfcc import mfcc, mfcc_velocity_acceleration
# from pb_chime5.nt.transform.module_ssc import ssc
# from pb_chime5.nt.transform.module_bark_fbank import bark_fbank
# from pb_chime5.nt.transform.module_rastaplp import rasta_plp
# from pb_chime5.nt.transform.module_ams import ams
# import numpy as np


# def normalize_mean_variance(data, axis=0, eps=1e-6):
#     """ Normalize features.
#
#     :param data: Any feature
#     :param axis: Time dimensions, default is 0
#     :return: Normalized observation
#     """
#     return (data - np.mean(data, axis=axis, keepdims=True)) /\
#         (np.std(data, axis=axis, keepdims=True) + eps)
