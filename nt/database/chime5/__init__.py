from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict

import numpy as np
from nt.database import HybridASRJSONDatabaseTemplate
from nt.database import keys as K
from nt.database.chime5.create_json import CHiME5_Keys, SAMPLE_RATE
from nt.database.chime5.get_speaker_activity import to_numpy, get_active_speaker
from nt.database.iterator import AudioReader
from nt.io.audioread import audioread
from nt.io.data_dir import database_jsons
from nt.io.json_module import load_json

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

    @property
    def map_dataset_to_sessions(self):
        return {'train': ['S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S12',
                          'S13', 'S16', 'S17', 'S18', 'S19', 'S20', 'S22',
                          'S23', 'S24'],
                'dev': ['S02', 'S09'],
                'test': ['S01', 'S21']
                }


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
        if type(start) == type(dict_list_val):
            # Recursively call itself
            l = [recursive_transform(func, dict_list_val[idx], start[idx], end[idx],
                                     list2array)
                 for idx in range(len(dict_list_val))]

            if list2array:
                return np.array(l)
            return l
        else:
            return recursive_transform(func, dict_list_val[0], start, end,
                                        list2array)
    elif isinstance(dict_list_val, (list, tuple)):
        l = start
        if list2array:
            return np.array(l)
        return l
    else:
        # applies function to a leaf value which is not a dict or list
        offset = start / SAMPLE_RATE
        duration = (end - start) / SAMPLE_RATE
        return func(dict_list_val, offset, duration)


class OverlapMapper:
    def __init__(self, dataset: str, json_path=database_jsons):
        """
        Add information on relative overlap to each example in `dataset`
        :param dataset: Chime5 dataset ('train', 'dev' or 'test')
        :param json_path: Path to json database. May be a string or Path object
        """

        if isinstance(json_path, str):
            json_path = Path(json_path)
        db = Chime5()
        sessions = db.map_dataset_to_sessions[dataset]
        self.crosstalk = dict()

        for session_id in sessions:
            json_sess = load_json(
                json_path / 'chime5_speech_activity' / f'{session_id}.json')

            crosstalk_at_target = {
                target: to_numpy(crosstalk_dict, 0, 10800*16000,
                                 # sample_step = 160 matches with kaldi
                                 # times in chime5.json
                                 sample_step=160)
                for target, crosstalk_dict
                in json_sess['cross_talk'].items()
            }

            self.crosstalk[session_id] = crosstalk_at_target

    def __call__(self, example: dict):
        """
        :param example: example_dict with example_id in it
        :return: example dict with relative overlap on example added
        """
        target, session, start, end = re.split('[_-]', example[K.EXAMPLE_ID])
        if target == 'original':
            example['overlap'] = np.NaN
            return example
        start_sample, end_sample = int(start), int(end)

        crosstalk_arr = self.crosstalk[session][target][start_sample:end_sample]
        overlap_samples = crosstalk_arr.sum()
        overlap_ratio = overlap_samples / (end_sample - start_sample)

        example['overlap'] = overlap_ratio
        return example


class CrossTalkFilter:
    def __init__(self, dataset: str, json_path=database_jsons,
                 with_crosstalk='no', min_overlap=0.0, max_overlap=0.0):
        """
        Filter examples according to with_crosstalk and left_overlap and
            right_overlap
        :param dataset: Chime5 dataset ('train', 'dev' or 'test')
        :param json_path: Path to json database. May be a string or Path object
        :param with_crosstalk: A string from ['yes', 'no', 'all']. If 'yes',
            return all utterances which have at least one sample overlap. If
            'no', return all samples which are overlap-free. If 'all', return
            all samples whose overlap is between `min_overlap` and
            `max_overlap`. By default, 'all' will return only the utterances
            with no overlap. Defaults to 'no'.
        :param min_overlap: If with_crosstalk is 'yes' or 'all', defines the
            lower bound the overlap on the utterance needs to have to return
            True. Ratio given in overlapping samples over total utterance
            samples
        :param max_overlap: If with_crosstalk is 'no' or 'all', defines the
            upper bound the overlap on the utterance is allowed to have to
            return True. Ratio given in overlapping samples over total utterance
            samples
        """

        assert with_crosstalk in ['yes', 'no', 'all'], \
            f'with_crosstalk must be a value from ["yes", "no", "all"], ' \
            'not {with_crosstalk}'

        self.mapper = OverlapMapper(dataset, json_path)

        self.with_crosstalk = with_crosstalk
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap

    def __call__(self, example):
        """
        :param example: example_dict with example_id in it
        :return: True if either
            1. self.with_crosstalk='no' and overlap_ratio in
                [0, self.max_overlap] or
            2. self.with_crosstalk='yes' and overlap_ratio in
                (self.min_overlap, 1] or
            3. self.with_crosstalk='all' and overlap_ratio in
                [self.min_overlap, self.max_overlap]
        """

        if 'overlap' not in example.keys():
            example = self.mapper(example)

        overlap_ratio = example['overlap']

        if self.with_crosstalk == 'no' and overlap_ratio <= self.max_overlap:
            return True
        elif self.with_crosstalk == 'yes' and overlap_ratio > self.min_overlap:
            return True
        elif self.with_crosstalk == 'all' and \
                (self.min_overlap <= overlap_ratio <= self.max_overlap):
            return True
        else:
            return False


class SessionFilter:
    def __init__(self, session_id):
        self.session_id = session_id

    def __call__(self, example):
        return example['session_id'] == self.session_id


class BadTranscriptionFilter:
    def __init__(self, bad_transcriptions=None):
        """
        Filter out all examples with bad transcriptions like [inaudible],
            [redacted], [laughs] or [noise]

        :param bad_transcriptions: A string of bad transcriptions that should be
            filtered out from the iterator. Transcriptions must be separated by
            a pipe (|). If None, a list of default bad transcriptions is used.
        """
        if not bad_transcriptions:
            self.bad_transcriptions = r'inaudible|redacted|laughs|noise'
        else:
            self.bad_transcriptions = bad_transcriptions

    def __call__(self, example):
        """

        :param example: example_dict with transcription in it
        :return: True if transcription is not in self.bad_transcriptions
        """
        return not all([re.match(self.bad_transcriptions, words) for words in
                        list(filter(lambda x: bool(x.strip()),
                                    re.split('[\[\]]', example[K.TRANSCRIPTION])
                                    )
                             )
                        ]
                       )
