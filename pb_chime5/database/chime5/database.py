
import numpy as np
from pb_chime5.database import JsonDatabase
from pb_chime5.database import keys as K

from pb_chime5.database.iterator import AudioReader
from pb_chime5.io import load_audio
from pb_chime5 import git_root
from pb_chime5.utils.numpy_utils import segment_axis_v2
from pb_chime5.utils.numpy_utils import pad_axis
import numbers


FORMAT_STRING = '%H:%M:%S.%f'

class CHiME5_Keys:
    """
    >>> print(dir(K))
    """
    WORN = 'worn'
    TARGET_SPEAKER = 'target_speaker'
    NOTES = 'notes'
    SESSION_ID = 'session_id'
    LOCATION = 'location'
    REFERENCE_ARRAY = 'reference_array'


class Chime5(JsonDatabase):
    K = CHiME5_Keys

    def __init__(
            self,
            path=git_root / 'cache' / 'chime5.json',
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

        it = self.get_datasets(session)

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
            read_fn=load_audio,
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


def kaldi_id_to_parts(example_id):
    """
    ToDo: start and end

    >>> kaldi_id_to_parts('P28_S09_LIVING.R-0714562-0714764')
    {'speaker_id': 'P28', 'session_id': 'S09', 'array_id': 'P28', 'location': 'LIVING', 'channel': 'R'}
    >>> kaldi_id_to_parts('P28_S09_LIVING.L-0714562-0714764')
    {'speaker_id': 'P28', 'session_id': 'S09', 'array_id': 'P28', 'location': 'LIVING', 'channel': 'L'}
    >>> kaldi_id_to_parts('P05_S02_U02_KITCHEN.ENH-0007012-0007298')
    {'speaker_id': 'P05', 'session_id': 'S02', 'array_id': 'P05', 'location': 'KITCHEN', 'channel': 'ENH'}
    >>> kaldi_id_to_parts('P09_S03_U01_NOLOCATION.CH1-0005948-0006038')
    {'speaker_id': 'P09', 'session_id': 'S03', 'array_id': 'P09', 'location': 'NOLOCATION', 'channel': 'CH1'}
    """
    try:
        pre, post = example_id.split('.')
        speaker_id, session_id, *array, location = pre.split('_')
        channel, start, end = post.split('-')
        if len(array) == 0:
            array = speaker_id
        elif len(array) == 1:
            array = speaker_id
        else:
            raise ValueError(array, example_id)
        return {
            'speaker_id': speaker_id,
            'session_id': session_id,
            'array_id': array,
            'location': location,
            'channel': channel,
        }
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


def activity_frequency_to_time(
        frequency_activity,
        stft_window_length,
        stft_shift,
        stft_fading,
        time_length=None,
):
    """

    >>> from nara_wpe.utils import istft
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
    >>> from nara_wpe.utils import stft
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
    >>> from nara_wpe.utils import stft
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
    >>> db = Chime5()
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

