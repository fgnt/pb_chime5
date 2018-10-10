from pathlib import Path
import re
import functools

import numpy as np
from pb_chime5.nt.database import HybridASRJSONDatabaseTemplate
from pb_chime5.nt.database import keys as K
from pb_chime5.nt.database.chime5.create_json import CHiME5_Keys, SAMPLE_RATE
from pb_chime5.nt.database.chime5.get_speaker_activity import to_numpy, get_active_speaker
from pb_chime5.nt.database.iterator import AudioReader
# from pb_chime5.nt.io.audioread import audioread
from pb_chime5.nt.io.data_dir import database_jsons
from pb_chime5.nt.io import load_json
# from pb_chime5.nt.options import Options
from pb_chime5.nt.utils.numpy_utils import segment_axis_v2
from pb_chime5.nt.utils.numpy_utils import pad_axis
import pb_chime5.nt.io
import numbers

kaldi_root = Path('/net/vol/jenkins/kaldi/2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb/')

FORMAT_STRING = '%H:%M:%S.%f'


class Chime5(HybridASRJSONDatabaseTemplate):
    K = CHiME5_Keys

    def __init__(
            self,
            path  # =database_jsons / 'chime5.json',
    ):
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

    def write_text_file(self, filename, datasets):
        iterator = self.get_iterator_by_names(datasets)
        with open(filename, 'w') as fid:
            for example in iterator:
                if example[CHiME5_Keys.TARGET_SPEAKER] == 'unknown':
                    continue
                transcription = example[K.TRANSCRIPTION]
                transcription.replace('/', '')
                fid.write(
                    f'{example[K.EXAMPLE_ID]} '
                    f'{transcription}\n'
                )

    @property
    def map_dataset_to_sessions(self):
        return {'train': ['S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S12',
                          'S13', 'S16', 'S17', 'S18', 'S19', 'S20', 'S22',
                          'S23', 'S24'],
                'dev': ['S02', 'S09'],
                'test': ['S01', 'S21']
                }

    @staticmethod
    def example_id_map_fn(example):
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
        ...     CHiME5_Keys.LOCATION: 'unknown',
        ... }
        >>> Chime5.example_id_map_fn(ex)
        'P09_S03_NOLOCATION.L-0005948-0006038'
        """
        speaker, session, time = example[K.EXAMPLE_ID].split('_')
        location = example[CHiME5_Keys.LOCATION]
        if not location == 'unknown':
            return '_'.join([speaker, session, location.upper() + '.L-']) + time
        else:
            return '_'.join([speaker, session, 'NOLOCATION.L-']) + time

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
            context_samples=0,
            equal_start_context=False,
    ):
        if isinstance(session, str):
            session = (session, )

        it = self.get_iterator_by_names(session)

        if drop_unknown_target_speaker:
            it = it.filter(lambda ex: ex['transcription'] != '[redacted]', lazy=False)

        if context_samples is not 0 or adjust_times:
            it = it.map(backup_orig_start_end)

        if adjust_times:
            if adjust_times is True:
                assert drop_unknown_target_speaker, (
                    'adjust_times is undefined for '
                    'ex["target_speaker"] == "unknown". '
                    'Set adjust_times to True.'
                )
                it = it.map(adjust_start_end)
            elif adjust_times == 'js':
                it = it.map(get_adjust_time_js())
            else:
                raise ValueError(adjust_times)

        if context_samples is not 0:
            # adjust_times should be before AddContext, because AddContext
            # adds the new key start_context that
            it = it.map(AddContext(
                context_samples,
                equal_start_context=equal_start_context,
            ))

        if audio_read is False:
            pass
        elif audio_read is True:
            it = it.map(Chime5AudioReader(audio_keys=None))
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
            read_fn=pb_chime5.nt.io.load_audio,
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


def kaldi_to_nt_example_id(example_id: str):
    """
    >>> kaldi_to_nt_example_id('P28_S09_LIVING.R-0714562-0714764')
    'P28_S09_0714562-0714764'
    >>> kaldi_to_nt_example_id('P05_S02_U02_KITCHEN.ENH-0007012-0007298')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    NotImplementedError: Array IDs like "P05_S02_U02_KITCHEN.ENH-0007012-0007298"" are currently unsupported, becasuse they have different timestamps.
    >>> kaldi_to_nt_example_id('P09_S03_U01_NOLOCATION.CH1-0005948-0006038')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    NotImplementedError: Array IDs like "P09_S03_U01_NOLOCATION.CH1-0005948-0006038"" are currently unsupported, becasuse they have different timestamps.
    """
    try:
        split = example_id.split('_')
        if len(split) == 4:
            raise NotImplementedError(
                f'Array IDs like "{example_id}"" are currently unsupported, '
                f'becasuse they have different timestamps.'
            )
        P, S, remaining = split
        _, start, end = remaining.split('-')
        return f'{P}_{S}_{start}-{end}'
    except NotImplementedError as e:
        raise
    except Exception as e:
        raise ValueError(example_id) from e


def kaldi_id_to_channel(example_id):
    """
    >>> kaldi_id_to_channel('P28_S09_LIVING.R-0714562-0714764')
    'R'
    >>> kaldi_id_to_channel('P28_S09_LIVING.L-0714562-0714764')
    'L'
    >>> kaldi_id_to_channel('P05_S02_U02_KITCHEN.ENH-0007012-0007298')
    'ENH'
    >>> kaldi_id_to_channel('P09_S03_U01_NOLOCATION.CH1-0005948-0006038')
    'CH1'
    """
    try:
        _, post = example_id.split('.')
        channel, _, _ = post.split('-')
        return channel
    except Exception as e:
        raise ValueError(example_id) from e


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
        else:
            # Broadcast start and end for the array channels
            assert isinstance(start, numbers.Integral) and isinstance(end, numbers.Integral), (start, end, type(start), type(end))
            l = [
                recursive_transform(
                    func,
                    l,
                    start=start,
                    end=end,
                    list2array=list2array,
                )
                for l in dict_list_val
            ]
        if list2array:
            return np.array(l)
        return l
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
        start_sample, end_sample = int(start), int(end)

        try:
            crosstalk_arr = self.crosstalk[session][target][start_sample:
            end_sample]
            overlap_samples = crosstalk_arr.sum()
            overlap_ratio = overlap_samples / (end_sample - start_sample)
            example['overlap'] = overlap_ratio
        except KeyError:  # target == 'original' or target == 'unknown'
            example['overlap'] = np.NaN
        return example


class SpeakerActivityMapper:
    def __init__(self, options, context=0, json_path=database_jsons):
        """
        Add speaker activity per STFT frame to each example.
        :param options: pb_chime5.nt.Options object containing arguments for STFT length
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

    >>> from pb_chime5.nt.transform import istft
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
    # import from pb_chime5.nt.transform import istft
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
    >>> from pb_chime5.nt.transform import stft
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
    >>> from pb_chime5.nt.transform import stft
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

    worn_start = ex[K.START]['original']
    worn_end = ex[K.END]['original']

    for array_id in ex['audio_path']['observation'].keys():
        array_start = ex[K.START]['observation'][array_id]
        array_end = ex[K.END]['observation'][array_id]


        array_start, array_end = _adjust_start_end(
            worn_start,
            worn_end,
            array_start,
            array_end,
        )
        ex[K.START]['observation'][array_id] = array_start
        ex[K.END]['observation'][array_id] = array_end
        ex[K.NUM_SAMPLES]['observation'][array_id] = array_end - array_start

    for mic_id in ex['audio_path'].get('worn_microphone', {}).keys():
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


def get_adjust_time_js():

    @functools.lru_cache(1)
    def get_json(session_id):
        json = load_json(f'~/net/vol/jensheit/chime5_playground/data/chime5_js_jsons/{session_id}_js.json')
        json = {
            e['example_id']: e
            for e in json
        }
        return json

    def adjust_time_js(ex):
        session_id = ex['session_id']
        example_id = ex['example_id']
        times_js = get_json(session_id)[example_id]

        for array_id in times_js['start_time'].keys():
            # delta_start = times_js['start_time_js'][array_id] - times_js['start_time'][array_id]
            # delta_end = times_js['end_time_js'][array_id] - times_js['end_time'][array_id]

            if array_id[0] == 'P':
                array_type = 'worn_microphone'
            elif array_id[0] == 'U':
                array_type = 'observation'
            elif array_id == 'original':
                array_type = None
            else:
                raise ValueError(array_id, example_id)

            if array_type is not None:
                assert ex['start'][array_type][array_id] == times_js['start_time'][array_id], (ex['start'][array_type][array_id], times_js['start_time'][array_id], array_type, example_id)
                assert ex['end'][array_type][array_id] == times_js['end_time'][array_id], (ex['end'][array_type][array_id], times_js['end_time'][array_id], array_type, example_id)

                ex['start'][array_type][array_id] = times_js['start_time_js'][array_id]
                ex['end'][array_type][array_id] = times_js['end_time_js'][array_id]
                ex['num_samples'][array_type][array_id] = ex['end'][array_type][array_id] - ex['start'][array_type][array_id]
        return ex

    return adjust_time_js


def nest_broadcast(
        shallow_tree,
        input_tree,
        mapping_type=dict,
        sequence_type=(tuple, list),
):
    """

    >>> shallow_tree = {'a': [1, 2, (3, 4)], 'b': [5, (6,)]}
    >>> nest_broadcast(shallow_tree, 10)
    {'a': [10, 10, (10, 10)], 'b': [10, (10,)]}
    >>> nest_broadcast(shallow_tree, {'a': 11})
    Traceback (most recent call last):
    ...
    AssertionError: ({'a': 11}, {'a': [1, 2, (3, 4)], 'b': [5, (6,)]})
    >>> nest_broadcast(shallow_tree, {'a': 11, 'b': 12})
    {'a': [11, 11, (11, 11)], 'b': [12, (12,)]}
    >>> nest_broadcast(shallow_tree, {'a': 11, 'b': (13, 14)})
    Traceback (most recent call last):
    ...
    AssertionError: (<class 'tuple'>, <class 'list'>, (13, 14), [5, (6,)])
    >>> nest_broadcast(shallow_tree, {'a': 11, 'b': [13, 14]})
    {'a': [11, 11, (11, 11)], 'b': [13, (14,)]}
    >>> nest_broadcast(shallow_tree, (1, 2))
    Traceback (most recent call last):
    ...
    TypeError: (<class 'tuple'>, (1, 2), {'a': [1, 2, (3, 4)], 'b': [5, (6,)]})
    >>> nest_broadcast(shallow_tree, (1, 2), sequence_type=None)
    {'a': (1, 2), 'b': (1, 2)}


    """
    def inner(
        shallow_tree,
        input_tree,
    ):
        if mapping_type is not None and isinstance(shallow_tree, mapping_type):
            if isinstance(input_tree, mapping_type):
                assert set(input_tree.keys()) == set(shallow_tree.keys()), (input_tree, shallow_tree)
                return {
                    k: inner(v, input_tree[k])
                    for k, v in shallow_tree.items()
                }
            elif sequence_type is not None and isinstance(input_tree, sequence_type):
                raise TypeError(type(input_tree), input_tree, shallow_tree)
            else:
                return {
                    k: inner(v, input_tree)
                    for k, v in shallow_tree.items()
                }
        elif sequence_type is not None and isinstance(shallow_tree, sequence_type):
            if isinstance(input_tree, sequence_type):
                assert type(input_tree) == type(shallow_tree), (type(input_tree), type(shallow_tree), input_tree, shallow_tree)
                assert len(input_tree) == len(shallow_tree)
                return shallow_tree.__class__([
                    inner(s, i)
                    for s, i in zip(shallow_tree, input_tree)
                ])
            elif mapping_type is not None and isinstance(input_tree, mapping_type):
                raise TypeError(type(input_tree), input_tree, shallow_tree)
            else:
                return shallow_tree.__class__([
                    inner(s, input_tree)
                    for s in shallow_tree
                ])
        else:
            return input_tree
    return inner(shallow_tree, input_tree)


def backup_orig_start_end(ex):
    ex['start_orig'] = ex[K.START]
    ex['end_orig'] = ex[K.END]
    ex['num_samples_orig'] = ex[K.NUM_SAMPLES]
    return ex


def AddContext(samples, equal_start_context=False):
    """
    >>> from IPython.lib.pretty import pprint
    >>> from pb_chime5.nt.io.data_dir import database_jsons
    >>> db = Chime5(database_jsons / 'chime5.json')
    >>> it = db.get_iterator_for_session('S02')
    >>> pprint(it[0])  # doctest: +ELLIPSIS
    {...
     'end': {'observation': {'U01': [701587, 701587, 701587, 701587],
     ...
      'worn_microphone': {'P05': 701120,
      ...
     'num_samples': {'observation': {'U01': [51520, 51520, 51520, 51520],
     ...
      'worn_microphone': {'P05': 51520, 'P06': 51520, 'P07': 51520, 'P08': 51520}},
     ...
     'start': {'observation': {'U01': [650067, 650067, 650067, 650067],
      ...
      'worn_microphone': {'P05': 649600,
      ...
    >>> pprint(it.map(AddContext(100))[0])  # doctest: +ELLIPSIS
    {...
     'end': {'observation': {'U01': [701687, 701687, 701687, 701687],
      ...
      'worn_microphone': {'P05': 701220,
      ...
     'num_samples': {'observation': {'U01': [51720, 51720, 51720, 51720],
      ...
      'worn_microphone': {'P05': 51720, 'P06': 51720, 'P07': 51720, 'P08': 51720}},
    ...
     'start': {'observation': {'U01': [649967, 649967, 649967, 649967],
      ...
      'worn_microphone': {'P05': 649500,
      ...
    >>> pprint(it.map(AddContext(10**10))[0])  # doctest: +ELLIPSIS
    {...
     'num_samples': {'observation': {'U01': [10000701587,
     ...
     'start': {'observation': {'U01': [0, 0, 0, 0],
       'U02': [0, 0, 0, 0],
       'U03': [0, 0, 0, 0],
       'U04': [0, 0, 0, 0],
       'U05': [0, 0, 0, 0],
       'U06': [0, 0, 0, 0]},
      'worn_microphone': {'P05': 0, 'P06': 0, 'P07': 0, 'P08': 0}},
     ...
    >>> pprint(it.map(AddContext(10**10, equal_start_context=True))[0])  # doctest: +ELLIPSIS
    {...
     'end': {'observation': {'U01': [10000701587,...
     'num_samples': {'observation': {'U01': [10000701113,...
     'start': {'observation': {'U01': [474, 474, 474, 474],
       'U02': [285, 285, 285, 285],
       'U03': [152, 152, 152, 152],
       'U04': [109, 109, 109, 109],
       'U05': [23, 23, 23, 23],
       'U06': [3, 3, 3, 3]},
      'worn_microphone': {'P05': 7, 'P06': 4, 'P07': 0, 'P08': 9}},...
    >>> pprint(it.map(AddContext([100, 50]))[0])  # doctest: +ELLIPSIS
    {...
     'end': {'observation': {'U01': [701637, 701637, 701637, 701637],
      ...
      'worn_microphone': {'P05': 701170,
      ...
     'num_samples': {'observation': {'U01': [51670, 51670, 51670, 51670],
      ...
      'worn_microphone': {'P05': 51670, 'P06': 51670, 'P07': 51670, 'P08': 51670}},
     ...
     'start': {'observation': {'U01': [649967, 649967, 649967, 649967],
      ...
      'worn_microphone': {'P05': 649500,
      ...
    >>> pprint(it.map(AddContext({'worn_microphone': 0, 'observation': [100, 50]}))[0])  # doctest: +ELLIPSIS
    {...
     'end': {'observation': {'U01': [701637, 701637, 701637, 701637],
      ...
      'worn_microphone': {'P05': 701120,
      ...
     'num_samples': {'observation': {'U01': [51670, 51670, 51670, 51670],
      ...
      'worn_microphone': {'P05': 51520, 'P06': 51520, 'P07': 51520, 'P08': 51520}},
     ...
     'start': {'observation': {'U01': [649967, 649967, 649967, 649967],
      ...
      'worn_microphone': {'P05': 649600,
      ...
    >>> pprint(it.map(AddContext([100, -50]))[0])
    Traceback (most recent call last):
    ...
    AssertionError: Negative context value (-50) is not supported
    >>> pprint(it.map(AddContext([-100, 50]))[0])
    Traceback (most recent call last):
    ...
    AssertionError: Negative context value (-100) is not supported
    >>> pprint(it.map(AddContext([-100, -50]))[0])
    Traceback (most recent call last):
    ...
    AssertionError: Negative context value (-100) is not supported
    >>> pprint(it.map(AddContext(-50))[0])
    Traceback (most recent call last):
    ...
    AssertionError: Negative context value (-50) is not supported
    """
    from tensorflow.python.util import nest

    def split(samples):
        if isinstance(samples, dict):
            d = [(k, *split(v)) for k, v in samples.items()]
            # raise Exception(samples, d)
            keys, *values = list(zip(*d))
            ret = tuple([
                dict(zip(keys, v))
                for v in values
            ])
        elif isinstance(samples, (tuple, list)):
            if len(samples) == 1 and isinstance(samples[0], int):
                assert samples[0] >= 0, f'Negative context value ({samples}) is not supported'
                ret = samples[0], samples[0], samples[0] + samples[0]
            elif len(samples) == 2 and isinstance(samples[0], int):
                #  and isinstance(samples[1], int)
                start, end = samples
                assert start >= 0, f'Negative context value ({start}) is not supported'
                assert end >= 0, f'Negative context value ({end}) is not supported'
                ret = start, end, start + end
            elif len(samples) == 3 and isinstance(samples[0], int):
                raise NotImplementedError(samples)
            else:
                l = [
                    split(e) for e in samples
                ]
                ret = tuple(map(samples.__class__, zip(*l)))
                raise NotImplementedError(samples)
        elif isinstance(samples, int):
            assert samples >= 0, f'Negative context value ({samples}) is not supported'
            ret = samples, samples, samples + samples
        else:
            raise ValueError(samples, type(samples))

        # assert len(ret) == 3, ret
        return ret

    start_context, end_context, duration_context = split(samples)

    if isinstance(start_context, int):
        # Faster implementation than the else Branch
        def add_context(ex):
            assert 'start_orig' in ex, ex
            assert 'end_orig' in ex, ex
            assert 'num_samples_orig' in ex, ex

            ex[K.START] = nest.map_structure(
                lambda time: max(time - start_context, 0),
                ex[K.START],
            )
            if equal_start_context:
                start_flat = nest.flatten(ex[K.START])
                start_orig_flat = nest.flatten(ex['start_orig'])
                smallest_start_context = np.min(np.array(start_orig_flat) - np.array(start_flat))

                ex[K.START] = nest.map_structure(
                    lambda time: max(time - smallest_start_context, 0),
                    ex['start_orig'],
                )

            ex[K.END] = nest.map_structure(
                lambda time: time + end_context,
                ex[K.END],
            )

            # ToDo: Fix NUM_SAMPLES for start < context
            # ex[K.NUM_SAMPLES] = nest.map_structure(
            #     lambda time: time + duration_context,
            #     ex[K.NUM_SAMPLES],
            # )
            ex[K.NUM_SAMPLES] = nest.map_structure(
                lambda start, end: end - start,
                ex[K.START],
                ex[K.END],
            )
            return ex
    else:
        def add_context(ex):
            assert 'start_orig' in ex, ex
            assert 'end_orig' in ex, ex
            assert 'num_samples_orig' in ex, ex

            bc_start_context = nest_broadcast(ex[K.START], start_context)
            bc_end_context = nest_broadcast(ex[K.END], end_context)
            bc_duration_context = nest_broadcast(ex[K.NUM_SAMPLES], duration_context)

            ex[K.START] = nest.map_structure(
                lambda time, start_context: max(time - start_context, 0),
                ex[K.START],
                bc_start_context,
            )
            # ex['start_context'] = nest.map_structure(
            #     lambda time, start_context: start_context + max(time - start_context, 0),
            #     ex[K.START],
            #     bc_start_context,
            # )
            ex[K.END] = nest.map_structure(
                lambda time, end_context: time + end_context,
                ex[K.END],
                bc_end_context,
            )
            # ToDo: end_context
            # ex['num_samples_orig'] = ex[K.NUM_SAMPLES]
            # ToDo: Fix NUM_SAMPLES for start < context
            # ex[K.NUM_SAMPLES] = nest.map_structure(
            #     lambda time, duration_context: time + duration_context,
            #     ex[K.NUM_SAMPLES],
            #     bc_duration_context,
            # )
            ex[K.NUM_SAMPLES] = nest.map_structure(
                lambda start, end: end - start,
                ex[K.START],
                ex[K.END],
            )
            return ex

    return add_context


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
from pb_chime5.nt.database.chime5.mapping import (
    session_speakers_mapping,
    session_dataset_mapping,
    session_array_to_num_samples_mapping,
)
