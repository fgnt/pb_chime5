import logging
from pathlib import Path

import numpy as np

from pb_chime5.database import keys
from pb_chime5.io.audioread import load_audio

LOG = logging.getLogger('iterator')

def recursive_transform(func, dict_list_val, list2array=False):
    """
    Applies a function func to all leaf values in a dict or list or directly to
    a value. The hierarchy of dict_list_val is inherited. Lists are stacked
    to numpy arrays. This function can e.g. be used to recursively apply a
    transformation (e.g. audioread) to all audio paths in an example dict
    (see top of this file).
    :param func: a transformation function to be applied to the leaf values
    :param dict_list_val: dict list or value
    :param args: args for func
    :param kwargs: kwargs for func
    :param list2array:
    :return: dict, list or value with transformed elements
    """
    if isinstance(dict_list_val, dict):
        # Recursively call itself
        return {key: recursive_transform(func, val, list2array)
                for key, val in dict_list_val.items()}
    if isinstance(dict_list_val, (list, tuple)):
        # Recursively call itself
        l = type(dict_list_val)(
            [recursive_transform(func, val, list2array)
             for val in dict_list_val]
        )
        if list2array:
            return np.array(l)
        return l
    else:
        # applies function to a leaf value which is not a dict or list
        return func(dict_list_val)


class AudioReader:
    def __init__(self, src_key='audio_path', dst_key='audio_data',
                 audio_keys='observation', read_fn=lambda x: load_audio(x)[0]):
        """
        recursively read audio files and add audio
        signals to the example dict.
        :param src_key: key in an example dict where audio file paths can be
            found.
        :param dst_key: key to add the read audio to the example dict.
        :param audio_keys: str or list of subkeys that are relevant. This can be
            used to prevent unnecessary audioread.
        """
        self.src_key = src_key
        self.dst_key = dst_key
        if audio_keys is not None:
            self.audio_keys = to_list(audio_keys)
        else:
            self.audio_keys = None
        self._read_fn = read_fn

    def __call__(self, example):
        """
        :param example: example dict with src_key in it
        :return: example dict with audio data added
        """
        if self.audio_keys is not None:
            data = {
                audio_key: recursive_transform(
                    self._read_fn, example[self.src_key][audio_key],
                    list2array=True
                )
                for audio_key in self.audio_keys
            }
        else:
            data = recursive_transform(
                self._read_fn, example[self.src_key], list2array=True
            )

        if self.dst_key is not None:
            example[self.dst_key] = data
        else:
            example.update(data)
        return example


class IdFilter:
    def __init__(self, id_list):
        """
        A filter to filter example ids.
        :param id_list: list of valid ids, e.g. ids belonging to a specific
            dataset.

        An alternative with slicing:

        >>> it = ExamplesIterator({'a': {}, 'b': {}, 'c': {}})
        >>> it = it.items().map(lambda x: {'example_id': x[0], **x[1]})
        >>> list(it)
        [{'example_id': 'a'}, {'example_id': 'b'}, {'example_id': 'c'}]
        >>> it['a']
        {'example_id': 'a'}
        >>> it['a', 'b']  # doctest: +ELLIPSIS
              ExamplesIterator(len=3)
              ExamplesIterator(len=3)
            ZipIterator()
          MapIterator(<function <lambda> at 0x...>)
        SliceIterator(('a', 'b'))
        >>> list(it['a', 'b'])
        [{'example_id': 'a'}, {'example_id': 'b'}]

        >>> it.filter(IdFilter(('a', 'b')))  # doctest: +ELLIPSIS
              ExamplesIterator(len=3)
              ExamplesIterator(len=3)
            ZipIterator()
          MapIterator(<function <lambda> at 0x...>)
        FilterIterator(<nt.database.iterator.IdFilter object at 0x...>)
        >>> list(it.filter(IdFilter(('a', 'b'))))
        [{'example_id': 'a'}, {'example_id': 'b'}]
        """
        self.id_list = id_list

    def __call__(self, example):
        """
        :param example: example dict with example_id in it
        :return: True if example_id in id_list else False
        """
        return example[keys.EXAMPLE_ID] in self.id_list


def to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def remove_zero_length_example(example, audio_key='observation',
                               dst_key='audio_data'):

    if keys.NUM_SAMPLES in example:
        num_samples = example[keys.NUM_SAMPLES]
        if isinstance(num_samples, dict):
            num_samples = num_samples[keys.OBSERVATION]
        valid_ali = num_samples > 0
    else:
        valid_ali = len(example[dst_key][audio_key]) > 0
    if not valid_ali:
        LOG.warning(f'Skipping: Audio length '
                    f'example\n{example[keys.EXAMPLE_ID]} is 0')
        return False
    return True


class LimitAudioLength:
    def __init__(self, max_lengths=160000, audio_keys=('observation',),
                 dst_key='audio_data', frame_length=400, frame_step=160):
        self.max_lengths = max_lengths
        self.audio_keys = audio_keys
        self.dst_key = dst_key
        self.frame_length = frame_length
        self.frame_step = frame_step
        if self.max_lengths:
            LOG.info(f'Will limit audio length to {self.max_lengths}')

    def _sample_to_frame(self, s):
        return max(
            0,
            (s - self.frame_length + self.frame_step) // self.frame_step
        )

    def _frame_to_lfr_frame(self, f):
        return (f + np.mod(-f, 3)) // 3

    def __call__(self, example):
        valid_ex = keys.NUM_SAMPLES in example and \
            example[keys.NUM_SAMPLES] <= self.max_lengths
        if not valid_ex:
            delta = max(1, (example[keys.NUM_SAMPLES] - self.max_lengths) // 2)
            start = np.random.choice(delta, 1)[0]

            # audio
            def cut_fn(x): return x[..., start: start + self.max_lengths]
            if self.audio_keys is not None:
                example[keys.AUDIO_DATA] = {
                    audio_key: recursive_transform(
                        cut_fn, example[keys.AUDIO_DATA][audio_key],
                        list2array=True
                    )
                    for audio_key in self.audio_keys
                }
            else:
                example[keys.AUDIO_DATA] = recursive_transform(
                    cut_fn, example[keys.AUDIO_DATA], list2array=True
                )
            example[keys.NUM_SAMPLES] = self.max_lengths

            # alignment
            if keys.ALIGNMENT in example:
                num_frames_start = self._sample_to_frame(start)
                num_frames_length = self._sample_to_frame(self.max_lengths)
                # Check for LFR
                num_frames = (example[keys.NUM_SAMPLES] - 400 + 160) // 160
                num_frames_lfr = self._frame_to_lfr_frame(num_frames)
                if len(example[keys.ALIGNMENT]) == num_frames_lfr:
                    num_frames_start = self._frame_to_lfr_frame(num_frames_start)
                    num_frames_length = self._frame_to_lfr_frame(num_frames_length)
                # Adjust alignment
                example[keys.ALIGNMENT] = \
                    example[keys.ALIGNMENT][num_frames_start: num_frames_start
                                            + num_frames_length]
                example[keys.NUM_ALIGNMENT_FRAMES] = num_frames_length

            LOG.warning(f'Cutting example to length {self.max_lengths}'
                        f' :{example[keys.EXAMPLE_ID]}')
        return example


class Word2Id:
    def __init__(self, word2id_fn):
        self._word2id_fn = word2id_fn

    def __call__(self, example):
        def _w2id(s):
            return np.array([self._word2id_fn(w) for w in s.split()], np.int32)

        if not (keys.TRANSCRIPTION in example or
                keys.KALDI_TRANSCRIPTION in example):
            raise ValueError(
                'Could not find transcription for example id '
                f'{example[keys.EXAMPLE_ID]}'
            )
        if keys.TRANSCRIPTION in example:
            example[keys.TRANSCRIPTION + '_ids'] = recursive_transform(
                _w2id, example[keys.TRANSCRIPTION]
            )
        if keys.KALDI_TRANSCRIPTION in example:
            example[keys.KALDI_TRANSCRIPTION + '_ids'] = recursive_transform(
                _w2id, example[keys.KALDI_TRANSCRIPTION]
            )

        return example
