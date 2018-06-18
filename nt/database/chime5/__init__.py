from pathlib import Path
import re

import numpy as np
from nt.database import HybridASRJSONDatabaseTemplate
from nt.database import keys as K
from nt.database.chime5.create_json import CHiME5_Keys, SAMPLE_RATE
from nt.database.chime5.get_speaker_activity import to_numpy, get_active_speaker
from nt.database.iterator import AudioReader
# from nt.io.audioread import audioread
from nt.io.data_dir import database_jsons
from nt.io import load_json
from nt.options import Options
from nt.utils.numpy_utils import segment_axis_v2
from nt.utils.numpy_utils import pad_axis
import nt.io

kaldi_root = Path('/net/vol/jenkins/kaldi/2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb/')

FORMAT_STRING = '%H:%M:%S.%f'


class Chime5(HybridASRJSONDatabaseTemplate):
    K = CHiME5_Keys

    def __init__(self, path=database_jsons / 'chime5.json'):
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

    @property
    def ali_path_train(self):
        """Path containing the kaldi alignments for train data."""
        return kaldi_root / 'egs' / 'chime5' / 's5' / 'exp' / 'tri3_all_train_worn_ali'


    @property
    def ali_path_eval(self):
        """Path containing the kaldi alignments for train data."""
        return kaldi_root / 'egs' / 'chime5' / 's5' / 'exp' / 'tri3_all_dev_worn_ali'

    @property
    def hclg_path(self):
        """Path to HCLG directory created by Kaldi."""
        return kaldi_root / 'egs' / 'chime5' / 's5' / 'exp' \
                   / 'tri3_all' / 'graph'

    @property
    def occs_file(self):
        return str(self.ali_path_train / 'final.occs')

    def get_lengths(self, datasets, length_transform_fn=lambda x: x[0]):
        # The output is a dictionary with the approximative lengths per example
        #  no exact length is necessary to specify the bucket boundaries
        it = self.get_iterator_by_names(datasets)
        lengths = dict()
        for example in it:
            num_samples = example[K.NUM_SAMPLES]
            speaker_id = example[CHiME5_Keys.TARGET_SPEAKER]
            if speaker_id == 'unknown':
                speaker_id = example[K.SPEAKER_ID][0]
            if isinstance(num_samples, dict):
                num_samples = num_samples[CHiME5_Keys.WORN][speaker_id]
            example_id = example[K.EXAMPLE_ID]
            lengths[example_id] = (length_transform_fn(num_samples))
        return lengths

    @property
    def map_dataset_to_sessions(self):
        return {'train': ['S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S12',
                          'S13', 'S16', 'S17', 'S18', 'S19', 'S20', 'S22',
                          'S23', 'S24'],
                'dev': ['S02', 'S09'],
                'test': ['S01', 'S21']
                }

    @property
    def example_id_map_fn(self):
        """
        >>> ex = {
        ...     K.EXAMPLE_ID: 'P05_S02_0004060-0004382',
        ...     K.DATASET_NAME: 'dev',
        ...     CHiME5_Keys.LOCATION: 'kitchen',
        ... }
        >>> Chime5.example_id_map_fn(ex)
        'P05_S02_KITCHEN.L-0004060-0004382'
        >>> ex = {
        ...     K.EXAMPLE_ID: 'P09_S03_0005948-0006038',
        ...     K.DATASET_NAME: 'train',
        ...     CHiME5_Keys.LOCATION: 'unkown',
        ... }
        >>> Chime5.example_id_map_fn(ex)
        'P09_S03_NOLOCATION.L-0005948-0006038'
        """
        def _map_example_id(example):
            speaker, session, time = example[K.EXAMPLE_ID].split('_')
            dataset_name = example[K.DATASET_NAME]
            location = example[CHiME5_Keys.LOCATION]
            if not location == 'unknown':
                return '_'.join([speaker, session, location.upper() + '.L-']) + time
            else:
                return '_'.join([speaker, session, 'NOLOCATION.L-']) + time

        return _map_example_id

    def add_num_samples(self, example):
        speaker = example[CHiME5_Keys.TARGET_SPEAKER]
        if speaker == 'unknown':
            speaker = example[K.SPEAKER_ID][0]
        example[K.NUM_SAMPLES] = example[K.NUM_SAMPLES][CHiME5_Keys.WORN][speaker]
        return example

    def word2id(self, word):
        """Returns the integer ID for a given word.

        If the word is not found, it returns the ID for `<UNK>`.
        """
        try:
            return self._word2id_dict[word]
        except KeyError:
            return self._word2id_dict['<eps>']

    def get_iterator_for_session(
            self,
            session,
            *,
            audio_read=False,
            drop_unknown_target_speaker=False,
            adjust_times=False,
    ):
        if isinstance(session, str):
            session = (session, )

        it = self.get_iterator_by_names(['train', 'dev']).filter(
            lambda ex: ex['session_id'] in session, lazy=False
        )
        if drop_unknown_target_speaker:
            it = it.filter(lambda ex: ex['target_speaker'] != 'unknown', lazy=False)

        if adjust_times:
            assert drop_unknown_target_speaker, (
                'adjust_times is undefined for '
                'ex["target_speaker"] == "unknown". '
                'Set adjust_times to True.'
            )
            it = it.map(adjust_start_end)

        if audio_read is False:
            pass
        elif audio_read is True:
            it = it.map(nt.database.chime5.Chime5AudioReader(audio_keys=None))
        else:
            raise TypeError(audio_read)

        return it


class Chime5AudioReader(AudioReader):
    def __init__(
            self,
            src_key='audio_path',
            dst_key='audio_data',
            audio_keys='observation',
            # audio_keys=None,  # is this not better for chime5? Low overhead.
            read_fn=nt.io.load_audio,
    ):
        super().__init__(src_key=src_key, dst_key=dst_key,
                         audio_keys=audio_keys,
                         read_fn=read_fn)

    def __call__(self, example):
        """
        :param example: example dict with src_key in it
        :return: example dict with audio data added
        """
        if self.audio_keys is not None:
            try:
                data = {
                    audio_key: recursive_transform(
                        self._read_fn, example[self.src_key][audio_key],
                        example[K.START][audio_key],
                        example[K.END][audio_key], list2array=True
                    )
                    for audio_key in self.audio_keys
                }
            except KeyError as e:
                raise KeyError(
                    f'{e} not in {example[self.src_key].keys()}'
                ) from e
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


def kaldi_to_nt_example_id(example_id):
    """
    >>> kaldi_to_nt_example_id('P28_S09_LIVING.R-0714562-0714764')
    'P28_S09_0714562-0714764'
    """
    P, S, remaining = example_id.split('_')
    _, start, end = remaining.split('-')
    return f'{P}_{S}_{start}-{end}'


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
            key: recursive_transform(
                func,
                val,
                start=start[key],
                end=end[key],
                list2array=list2array,
            )
            for key, val in dict_list_val.items()
        }
    elif isinstance(dict_list_val, (list, tuple)):
        if type(start) == type(dict_list_val):
            # Recursively call itself
            l = [
                recursive_transform(
                    func,
                    l,
                    start=s,
                    end=e,
                    list2array=list2array,
                )
                for l, s, e in zip(dict_list_val, start, end)
            ]
            if list2array:
                return np.array(l)
            return l
        else:
            assert False, \
                'CB: This branch has no valid code. Fix the code. ' \
                'I do not know when this branch is reached.'
            return recursive_transform(func, dict_list_val[0], start, end,
                                        list2array)
    elif isinstance(dict_list_val, (list, tuple)):
        assert False, \
            'CB: This branch is unreachable. ' \
            'This branch has no valid code. Fix the code.'
        l = start
        if list2array:
            return np.array(l)
        return l
    else:
        # applies function to a leaf value which is not a dict or list
        return func(dict_list_val, start=start, stop=end)


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


class SpeakerActivityMapper:
    def __init__(self, options: Options, context=0, json_path=database_jsons):
        """
        Add speaker activity per STFT frame to each example.
        :param options: nt.Options object containing arguments for STFT length
            and shift.
        :param context: Speaker activity added before and after actual
            utterance. Context is given in seconds.
        :param json_path: Path to database json files.
        """
        assert 'frame_length' in options.keys() and \
               'frame_step' in options.keys(), \
            'Options object must specify STFT parameters "frame_length" and ' \
            '"frame_step"'
        self.frame_length = options['frame_length']
        self.frame_step = options['frame_step']
        self.context = context
        self.speaker_activity = dict()

        if isinstance(json_path, str):
            json_path = Path(json_path)
        db = Chime5()
        sessions = [sess for sess_list in db.map_dataset_to_sessions.values()
                    for sess in sess_list]

        for session_id in sessions:
            try:
                json_sess = load_json(json_path / 'chime5_speech_activity'
                                      / f'{session_id}.json'
                                      )
            except FileNotFoundError:
                continue

            target_speakers = sorted([key for key in json_sess.keys() if
                                      key.startswith('P')]
                                     )
            speaker_activity = {
                target: to_numpy(json_sess[target][target], 0, 10800*16000,
                                 sample_step=1, dtype=np.int16)
                for target in target_speakers
            }
            self.speaker_activity[session_id] = speaker_activity

    def __call__(self, example):
        """

        :param example: example_dict
        :return: example_dict with add. key "speaker_activity_per_frame"
        """
        _, session_id, kaldi_start, kaldi_end = re.split('[_-]',
                                                         example[K.EXAMPLE_ID])
        start = (int(kaldi_start) - self.context * 100) * 160
        end = (int(kaldi_end) + self.context * 100) * 160

        speaker_activity = self.speaker_activity[session_id]
        pad_width = self.frame_length - self.frame_step  # Consider fading

        speaker_activity_per_frame = {
            target: segment_axis_v2(np.pad(activity[start:end], pad_width,
                                           mode='constant'),
                                    length=self.frame_length,
                                    shift=self.frame_step,
                                    end='pad'  # Consider padding
                                    ).any(1)
            for target, activity in speaker_activity.items()
        }

        example['speaker_activity_per_frame'] = speaker_activity_per_frame
        return example


def activity_frequency_to_time(
        frequency_activity,
        stft_window_length,
        stft_shift,
        stft_fading,
        time_length=None,
):
    """

    >>> from nt.transform import istft
    >>> vad = np.array(   [0, 1, 0, 1, 0, 0, 1, 0, 0])
    >>> np.set_printoptions(suppress=True)
    >>> activity_frequency_to_time(vad, stft_window_length=4, stft_shift=2, stft_fading=False)
    array([False, False,  True,  True,  True,  True,  True,  True,  True,
            True, False, False,  True,  True,  True,  True, False, False,
           False, False])
    >>> activity_frequency_to_time([vad, vad], stft_window_length=4, stft_shift=2, stft_fading=False)
    array([[False, False,  True,  True,  True,  True,  True,  True,  True,
             True, False, False,  True,  True,  True,  True, False, False,
            False, False],
           [False, False,  True,  True,  True,  True,  True,  True,  True,
             True, False, False,  True,  True,  True,  True, False, False,
            False, False]])

    """
    if stft_fading:
        raise NotImplementedError(stft_fading)

    frequency_activity = np.asarray(frequency_activity)
    # import from nt.transform import istft
    # cbj.istft
    # frequency_activity = frequency_activity
    frequency_activity = np.broadcast_to(
        frequency_activity[..., None], (*frequency_activity.shape, stft_window_length)
    )

    time_activity = np.zeros(
        (*frequency_activity.shape[:-2],
         frequency_activity.shape[-2] * stft_shift + stft_window_length - stft_shift)
    )

    # Get the correct view to time_signal
    time_signal_seg = segment_axis_v2(
        time_activity, stft_window_length, stft_shift, end=None
    )

    # Unbuffered inplace add
    # np.add.at(
    #     time_signal_seg,
    #     ...,
    #     frequency_activity
    # )
    # It is not nessesary to do a unbuffered assignment, because it is alwais
    # the same value that gets assigned.
    time_signal_seg[frequency_activity > 0] = 1
    time_activity = time_activity != 0

    if time_length is not None:
        if time_length == time_activity.shape[-1]:
            pass
        elif time_length < time_activity.shape[-1]:
            delta = time_activity.shape[-1] - time_length
            assert delta < stft_window_length - stft_shift, (delta, stft_window_length, stft_shift)
            time_activity = time_activity[..., :time_length]

        elif time_length > time_activity.shape[-1]:
            delta = time_length - time_activity.shape[-1]
            assert delta < stft_window_length - stft_shift, (delta, stft_window_length, stft_shift)

            time_activity = pad_axis(
                time_activity[..., :time_length],
                pad_width=(0, delta),
                axis=-1,
            )
        else:
            raise Exception('Can not happen')
        assert time_length == time_activity.shape[-1], (time_length, time_activity.shape)

    return time_activity != 0


def activity_time_to_frequency(
        time_activity,
        stft_window_length,
        stft_shift,
        stft_fading,
        stft_pad=True,
):
    """
    >>> from nt.transform import stft
    >>> signal = np.array([0, 0, 0, 0, 0, 1, -3, 0, 5, 0, 0, 0, 0, 0])
    >>> vad = np.array(   [0, 0, 0, 0, 0, 1,  1, 0, 1, 0, 0, 0, 0, 0])
    >>> np.set_printoptions(suppress=True)
    >>> print(stft(signal, size=4, shift=2, fading=True, window=np.ones))
    [[ 0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j]
     [ 1.+0.j  0.+1.j -1.+0.j]
     [-2.+0.j  3.-1.j -4.+0.j]
     [ 2.+0.j -8.+0.j  2.+0.j]
     [ 5.+0.j  5.+0.j  5.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j]]
    >>> activity_time_to_frequency(vad, stft_window_length=4, stft_shift=2, stft_fading=True)
    array([False, False,  True,  True,  True,  True, False, False])
    >>> activity_time_to_frequency([vad, vad], stft_window_length=4, stft_shift=2, stft_fading=True)
    array([[False, False,  True,  True,  True,  True, False, False],
           [False, False,  True,  True,  True,  True, False, False]])
    >>> print(stft(signal, size=4, shift=2, fading=False, window=np.ones))
    [[ 0.+0.j  0.+0.j  0.+0.j]
     [ 1.+0.j  0.+1.j -1.+0.j]
     [-2.+0.j  3.-1.j -4.+0.j]
     [ 2.+0.j -8.+0.j  2.+0.j]
     [ 5.+0.j  5.+0.j  5.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j]]
    >>> activity_time_to_frequency(vad, stft_window_length=4, stft_shift=2, stft_fading=False)
    array([False,  True,  True,  True,  True, False])
    >>> activity_time_to_frequency([vad, vad], stft_window_length=4, stft_shift=2, stft_fading=False)
    array([[False,  True,  True,  True,  True, False],
           [False,  True,  True,  True,  True, False]])


    >>> activity_time_to_frequency(np.zeros(200000), stft_window_length=1024, stft_shift=256, stft_fading=False, stft_pad=False).shape
    (778,)
    >>> from nt.transform import stft
    >>> stft(np.zeros(200000), size=1024, shift=256, fading=False, pad=False).shape
    (778, 513)
    """
    assert np.asarray(time_activity).dtype != np.object, (type(time_activity), np.asarray(time_activity).dtype)
    time_activity = np.asarray(time_activity)

    if stft_fading:
        pad_width = np.array([(0, 0)] * time_activity.ndim)
        pad_width[-1, :] = stft_window_length - stft_shift  # Consider fading
        time_activity = np.pad(
            time_activity,
            pad_width,
            mode='constant'
        )

    return segment_axis_v2(
        time_activity,
        length=stft_window_length,
        shift=stft_shift,
        end='pad' if stft_pad else 'cut'
    ).any(axis=-1)


class CrossTalkFilter:
    def __init__(self, dataset: str, json_path=database_jsons,
                 with_crosstalk='no', min_overlap=0.0, max_overlap=0.0):
        """
        Filter examples according to with_crosstalk and min_overlap and
            max_overlap
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


def _adjust_start_end(
        worn_start,
        worn_end,
        array_start,
        array_end,
):
    """

    >>> w_s = np.random.randint(0, 100)
    >>> w_e = w_s + np.random.randint(1, 100)
    >>> a_s = np.random.randint(0, 100)
    >>> a_e = a_s + w_e - w_s
    >>> def test(a_s, a_e, delta_s, delta_e):
    ...     res = _adjust_start_end(w_s, w_e, a_s, a_e)
    ...     if (a_s + delta_s, a_e + delta_e) != res:
    ...         raise AssertionError(f'Expected changes {delta_s} {delta_e}, got {res[0] - a_s} {res[1] - a_e}. {(w_s, w_e, a_s, a_e)}')
    >>> test(a_s, a_e, 0, 0)
    >>> test(a_s, a_e+1, 0, -1)
    >>> test(a_s, a_e-1, 0, +1)
    >>> test(a_s+1, a_e, 0, +1)
    >>> test(a_s-1, a_e, 0, -1)

    >>> test(a_s, a_e+2, 1, -1)
    >>> test(a_s, a_e-2, -1, 1)

    >>> test(a_s, a_e+3, 1, -2)
    >>> test(a_s, a_e-3, -1, +2)
    >>> test(a_s, a_e+4, 2, -2)
    >>> test(a_s, a_e-4, -2, +2)
    >>> test(a_s, a_e+5, 2, -3)
    >>> test(a_s, a_e-5, -2, +3)

    >>> _adjust_start_end(10, 20, 10, 19)
    (10, 20)
    >>> _adjust_start_end(10, 20, 10, 21)
    (10, 20)
    """
    worn_duration = worn_end - worn_start
    array_duration = array_end - array_start

    if worn_duration == array_duration:
        new_start, new_end = array_start, array_end
    elif worn_duration > array_duration:
        delta = worn_duration - array_duration
        delta_start = delta // 2
        delta_end = (delta + 1) // 2
        new_start, new_end = array_start - delta_start, array_end + delta_end
    elif worn_duration < array_duration:
        delta = array_duration - worn_duration
        delta_start = delta // 2
        delta_end = (delta + 1) // 2
        new_start, new_end = array_start + delta_start, array_end - delta_end
    else:
        raise Exception('Can not happen.')

    assert new_end - new_start == worn_duration, (
        f'worn: {worn_end} - {worn_start} = {worn_duration}, '
        f'array: {array_end} - {array_start} = {array_duration}, '
        f'new: {new_end} - {new_start} = {new_end - new_start}, \n'
        f'delta: {delta}, delta_start: {delta_start}, delta_end: {delta_end}'
    )

    return new_start, new_end


def adjust_start_end(ex):

    target_speaker = ex['target_speaker']

    worn_start = ex[K.START]['worn_microphone'][target_speaker]
    worn_end = ex[K.END]['worn_microphone'][target_speaker]

    for array_id in ex['audio_path']['observation'].keys():
        array_start = ex[K.START]['observation'][array_id]
        array_end = ex[K.END]['observation'][array_id]

        array_start, = list(set(array_start))  # assert len 1
        array_end, = list(set(array_end))  # assert len 1

        array_start, array_end = _adjust_start_end(
            worn_start,
            worn_end,
            array_start,
            array_end,
        )
        ex[K.START]['observation'][array_id] = [array_start] * 4
        ex[K.END]['observation'][array_id] = [array_end] * 4
        ex[K.NUM_SAMPLES]['observation'][array_id] = [array_end - array_start] * 4

    for mic_id in ex['audio_path']['worn_microphone'].keys():
        array_start, array_end = _adjust_start_end(
            worn_start,
            worn_end,
            ex[K.START]['worn_microphone'][mic_id],
            ex[K.END]['worn_microphone'][mic_id]
        )
        ex[K.START]['worn_microphone'][mic_id] = array_start
        ex[K.END]['worn_microphone'][mic_id] = array_end
        ex[K.NUM_SAMPLES]['worn_microphone'][mic_id] = array_end - array_start
    return ex


class SessionFilter:
    def __init__(self, session_id):
        self.session_id = session_id

    def __call__(self, example):
        return example['session_id'] == self.session_id


class BadTranscriptionFilter:
    def __init__(self, bad_transcriptions=None, keep_bad=False):
        """
        Filter out all examples with bad transcriptions like [inaudible],
            [redacted], [laughs] or [noise]

        :param bad_transcriptions: A list of bad transcriptions that should be
            filtered out from the iterator. Must be given as regular
            expressions. If None, a list of default bad transcriptions is used.
        :param keep_bad: If False, return examples with good transcriptions
            only. Else, return examples whose transcriptions match with
            `bad_transcriptions`. Defaults to False.
        """
        if not bad_transcriptions:
            self.bad_transcriptions = ['\[noise\]', '\[laughs\]',
                                       '\[redacted\]', '\[inaudible',
                                       '\w:\w\w:\w\w\.\w\w]'
                                       ]
        else:
            self.bad_transcriptions = bad_transcriptions
        self.keep_bad = keep_bad

    def __call__(self, example):
        """

        :param example: example_dict with transcription in it
        :return: True if transcription matches with `keep_bad`
        """
        return (all([any([re.match(p, word) for p in self.bad_transcriptions])
                    for word in example['transcription'].split()])
                == self.keep_bad)


# cyclic import, has to be at the end of the __init__ file
from nt.database.chime5.mapping import (
    session_speakers_mapping,
    session_dataset_mapping,
    session_array_to_num_samples_mapping,
)
