from datetime import datetime

import numpy as np
from nt.database import HybridASRJSONDatabaseTemplate
from nt.database import keys as K
from nt.database.chime5.create_json import CHiME5_Keys
from nt.database.iterator import AudioReader
from nt.io.audioread import audioread
from nt.io.data_dir import database_jsons

FORMAT_STRING = '%H:%M:%S.%f'


class Chime5(HybridASRJSONDatabaseTemplate):
    def __init__(self):
        path = database_jsons / 'chime5.json'
        super().__init__(path)

    @property
    def datasets_train(self):
        return ['train']

    @property
    def datasets_eval(self):
        return ['dev']

    @property
    def datasets_test(self):
        return ['test']

    def get_lengths(self, datasets, length_transform_fn=lambda x: x[0]):
        it = self.get_iterator_by_names(datasets)
        lengths = []
        for example in it:
            num_samples = example[K.NUM_SAMPLES]
            speaker_id = example[CHiME5_Keys.TARGET_SPEAKER]
            if isinstance(num_samples, dict):
                num_samples = num_samples[CHiME5_Keys.WORN][speaker_id]
            lengths.append(length_transform_fn(num_samples))
        return lengths


class Chime5AudioReader(AudioReader):
    def __init__(self, src_key='audio_path', dst_key='audio_data',
                 audio_keys='observation',
                 read_fn=lambda x, offset, duration:
                 audioread(path=x, offset=offset, duration=duration)[0]):
        super().__init__(src_key=src_key, dst_key=dst_key,
                         audio_keys=audio_keys,
                         read_fn=read_fn)

    def __call__(self, example):
        """
        :param example: example dict with src_key in it
        :return: example dict with audio data added
        """
        if self.audio_keys is not None:
            data = {
                audio_key: recursive_transform(
                    self._read_fn, example[self.src_key][audio_key],
                    example[K.START][audio_key],
                    example[K.END][audio_key], list2array=True
                )
                for audio_key in self.audio_keys
            }
        else:
            data = recursive_transform(
                self._read_fn, example[self.src_key], example[K.START],
                example[K.END], list2array=True
            )

        if self.dst_key is not None:
            example[self.dst_key] = data
        else:
            example.update(data)
        return example


def recursive_transform(func, dict_list_val, start, end, list2array=False):
    """
    Applies a function func to all leaf values in a dict or list or directly to
    a value. The hierarchy of dict_list_val is inherited. Lists are stacked
    to numpy arrays. This function can e.g. be used to recursively apply a
    transformation (e.g. audioread) to all audio paths in an example dict
    (see top of this file).
    :param func: a transformation function to be applied to the leaf values
    :param dict_list_val: dict list or value
    :param start: start time of example
    :param end: end time of example
    :param list2array:
    :return: dict, list or value with transformed elements
    """
    if isinstance(dict_list_val, dict):
        # Recursively call itself
        return {
            key: recursive_transform(func, val, start[key], end[key],
                                     list2array)
            for key, val in dict_list_val.items()}
    if isinstance(dict_list_val, (list, tuple)):
        # Recursively call itself
        l = [recursive_transform(func, dict_list_val[idx], start[idx], end[idx],
                                 list2array)
             for idx in range(len(dict_list_val))]
        if list2array:
            return np.array(l)
        return l
    else:
        # applies function to a leaf value which is not a dict or list
        offset = datetime.strptime(start, FORMAT_STRING) - datetime.strptime(
            '0:00:00.00', FORMAT_STRING)
        duration = datetime.strptime(end, FORMAT_STRING) - datetime.strptime(
            start, FORMAT_STRING)
        return func(dict_list_val, offset.total_seconds(),
                    duration.total_seconds())
