from datetime import datetime
from pathlib import Path

import numpy as np
from nt.database import HybridASRJSONDatabaseTemplate
from nt.database import keys as K
from nt.database.chime5.create_json import CHiME5_Keys, SAMPLE_RATE
from nt.database.chime5.get_speaker_activity import to_numpy
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
            all samples whose overlap is between left_overlap and right_overlap.
            By default, 'all' will return only the utterances with no overlap.
        :param min_overlap: If with_crosstalk is 'yes' or 'all', defines the
            lower bound the overlap on the utterance needs to have to return
            True. Ratio given in overlapping samples over total utterance
            samples
        :param max_overlap: If with_crosstalk is 'no' or 'all', defines the
            upper bound the overlap on the utterance needs to have to return
            True. Ratio given in overlapping samples over total utterance
            samples
        """

        assert with_crosstalk in ['yes', 'no', 'all'], \
            f'with_crosstalk must be a value from ["yes", "no", "all"], ' \
            'not {with_crosstalk}'

        if isinstance(json_path, str):
            json_path = Path(json_path)
        db = Chime5()
        sessions = db.map_dataset_to_sessions[dataset]
        self.crosstalk_times = dict()

        for session_id in sessions:
            json_sess = load_json(
                json_path / 'chime5_speech_activity' / f'{session_id}time.json')

            target_speakers = sorted(
                [speaker for speaker in list(json_sess.keys())
                 if speaker.startswith('P')])

            crosstalk_times = set([(start, end) for spk in target_speakers for
                                   start, end in
                                   zip(json_sess[spk]['cross_talk']['start'],
                                       json_sess[spk]['cross_talk']['end'])
                                   ])

            crosstalk = {
                'start': [times[0] for times in crosstalk_times],
                'end': [times[1] for times in crosstalk_times]
            }
            self.crosstalk_times[session_id] = crosstalk

        self.with_crosstalk = 0 if with_crosstalk == 'no' else \
            1 if with_crosstalk == 'yes' else 2
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
        session, _, start, end = example[K.EXAMPLE_ID].split('_')
        crosstalk_times = self.crosstalk_times[session]
        crosstalk_start = np.array(crosstalk_times['start'])
        crosstalk_end = np.array(crosstalk_times['end'])
        start_sample, end_sample = (
            to_samples(start), to_samples(end)
        )
        crosstalk_idx = np.logical_and(crosstalk_start >= start_sample,
                                       crosstalk_end <= end_sample)
        has_crosstalk = crosstalk_idx.any()
        if has_crosstalk:
            # sample only every 0.01 second
            utt_samples = to_numpy({'start': crosstalk_start[crosstalk_idx],
                                    'end': crosstalk_end[crosstalk_idx]},
                                   start_sample, end_sample,
                                   sample_step=160)
            overlap_ratio = utt_samples.sum() / int(
                (end_sample - start_sample) / 160)
        else:
            overlap_ratio = 0
        return \
            (not self.with_crosstalk and overlap_ratio <= self.max_overlap) \
            or (
                not (self.with_crosstalk - 1)
                and overlap_ratio > self.min_overlap
                ) \
            or (
                not (self.with_crosstalk - 2)
                and self.max_overlap >= overlap_ratio >= self.min_overlap
                )
