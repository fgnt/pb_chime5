import numpy as np
from scipy.io.wavfile import write as wav_write
import threading
from pathlib import Path

int16_max = np.iinfo(np.int16).max
int16_min = np.iinfo(np.int16).min


def audiowrite(data, path, sample_rate=16000, normalize=False, threaded=True):
    """ Write the audio data ``data`` to the wav file ``path``

    The file can be written in a threaded mode. In this case, the writing
    process will be started at a separate thread. Consequently, the file will
    not be written when this function exits.

    :param data: A numpy array with the audio data
    :param path: The wav file the data should be written to
    :param sample_rate: Sample rate of the audio data
    :param normalize: Normalize the audio first so that the values are within
        the range of [INTMIN, INTMAX]. E.g. no clipping occurs
    :param threaded: If true, the write process will be started as a separate
        thread
    :return: The number of clipped samples
    """
    assert data.dtype.kind == 'f', (data.shape, data.dtype)

    if isinstance(path, Path):
        path = str(path)

    data = data.copy()

    if normalize:
        if not data.dtype.kind == 'f':
            data = data.astype(np.float)
        data /= np.maximum(np.amax(np.abs(data)), 1e-6)

    if data.dtype.kind == 'f':
        data *= int16_max

    sample_to_clip = np.sum(data > int16_max)
    if sample_to_clip > 0:
        print('Warning, clipping {} sample{}.'.format(
            sample_to_clip, '' if sample_to_clip == 1 else 's'
            ))
    data = np.clip(data, int16_min, int16_max)
    data = data.astype(np.int16)

    if threaded:
        threading.Thread(target=wav_write, args=(path, sample_rate, data)
                         ).start()
    else:
        try:
            wav_write(path, sample_rate, data)
        except Exception:  #_struct.error
            if data.ndim == 2:
                assert data.shape[1] < 20, \
                    "channels bigger than 20 looks wrong. " \
                    "Maybe you must call audiowrite(data.T, ...)"
            raise

    return sample_to_clip
