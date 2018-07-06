import numpy as np

from nt.utils.mapping import Dispatcher

from nt.database.chime5.mapping import (
    session_array_to_num_samples_mapping,
    session_speakers_mapping,
    # session_dataset_mapping,
    session_array_mapping,
)
from chime5.decode.alignment import get_phone_alignment
from nt.database.chime5 import activity_frequency_to_time, adjust_start_end

from chime5.util.intervall_array import ArrayIntervall


def get_function_example_to_non_sil_alignment(
        ali_path,
        channel_preference,
        add_statistics=False,
):
    alignment = get_phone_alignment(
        ali_path,
        channel_preference=channel_preference
    )
    non_sil_alignment_dict = Dispatcher({k: v != 'sil' for k, v in alignment.items()})

    # Because of perspective_mic_array is is usefull to use a cache
    last = None

    import collections

    statistics = collections.defaultdict(set)

    def example_to_non_sil_alignment(
            ex,
            perspective_mic_array,
    ):
        nonlocal last
        nonlocal statistics
        if last is not None:
            example_id, value = last
            if example_id == ex['example_id']:
                return value

        # ignore perspective_mic_array
        if ex['example_id'] in non_sil_alignment_dict:
            ret = activity_frequency_to_time(
                non_sil_alignment_dict[ex['example_id']],
                stft_window_length=400,
                stft_shift=160,
                stft_fading=False,
                time_length=ex['num_samples']['worn_microphone'][
                    ex['target_speaker']],
            )
        else:
            print(
                f"Warning: Could not find {ex['example_id']} in non_sil_alignment."
            )
            ret = 1
            if add_statistics:
                session_id = ex['session_id']
                target_speaker = ex['target_speaker']
                statistics[f'{target_speaker}_{session_id}'].add(ex['example_id'])

        last = ex['example_id'], ret
        return ret

    example_to_non_sil_alignment.statistics = statistics

    return example_to_non_sil_alignment


from chime5.io.file_cache import file_cache

from nt.io.data_dir import database_jsons


@file_cache('get_all_activity_for_worn_alignments')
def get_all_activity_for_worn_alignments(
        perspective,
        *,
        garbage_class,
        ali_path,
        # use_ArrayIntervall=True,
        session_id=None,
        channel_preference=None,
        database_path=database_jsons / 'chime5.json',
        add_statistics=False,
):
    """
    session_id: When None than all sessions, when given only one session

    >>> get_all_activity_for_worn_alignments('worn', None, None)
    >>> ali_path = ['/net/vol/jenkins/kaldi/2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb/egs/chime5/s5/exp/tri3_all_dev_worn_ali']
    >>> get_all_activity_for_worn_alignments('worn', None, ali_path)
    """
    from nt.database.chime5 import Chime5
    from nt.database.chime5.mapping import session_dataset_mapping
    if session_id is None:
        it = Chime5(database_path).get_iterator_by_names(['train', 'dev'])
    else:
        it = Chime5(database_path).get_iterator_by_names(
            session_dataset_mapping[session_id]
        )
        it = it.filter(lambda ex: ex['session_id'] == session_id, lazy=False)

    it = it.filter(lambda ex: ex['target_speaker'] != 'unknown', lazy=False)

    if ali_path:
        non_sil_alignment_fn = get_function_example_to_non_sil_alignment(
            ali_path,
            channel_preference=channel_preference,
            add_statistics=add_statistics,
        )
        it = it.map(adjust_start_end)
    else:
        non_sil_alignment_fn = None

    activity = get_activity(
        it,
        perspective=perspective,
        garbage_class=garbage_class,
        dtype=np.bool,
        non_sil_alignment_fn=non_sil_alignment_fn,
        use_ArrayIntervall=True,
    )
    if add_statistics:
        print('get_phone_alignment statistics')
        for k, v in non_sil_alignment_fn.statistics.items():
            print(k, len(v))
    return activity


def get_activity(
        iterator,
        # session_id,
        *,

        perspective,
        garbage_class,
        dtype=np.bool,
        non_sil_alignment_fn=None,
        debug=False,
        use_ArrayIntervall=False,
):
    """

    perspective:
        Example:
            'global_worn' -- global perspective for worn
            'worn' --
            'array' --


    returns:
        dict[session_id][mic_perspective][speaker_id] = array(dtype=bool)
        session_id e.g.: 'S02', ...
        mic_perspective e.g.: 'global_worn', 'P05', 'U01', ...
        speaker_id e.g.: 'P05', ...


    """

    dict_it_S = iterator.groupby(lambda ex: ex['session_id'])

    all_acitivity = {}
    for session_id, it_S in dict_it_S.items():

        if perspective == 'worn':
            perspective_tmp = session_speakers_mapping[session_id]
        elif perspective == 'global_worn':
            perspective_tmp = [perspective]
        elif perspective == 'array':
            perspective_tmp = session_array_mapping[session_id]
            # perspective_tmp = [f'U0{i}' for i in range(1, 7)]
        else:
            perspective_tmp = perspective

            if not isinstance(perspective_tmp, (tuple, list)):
                perspective_tmp = [perspective_tmp, ]

        # num_samples = session_array_to_num_samples_mapping[f'{session_id}_P']
        speaker_ids = session_speakers_mapping[session_id]

        if use_ArrayIntervall:
            assert dtype == np.bool, dtype
            zeros = ArrayIntervall
        else:
            import functools
            zeros = functools.partial(np.zeros, dtype=dtype)

        all_acitivity[session_id] = {
            p: {
                s: zeros(shape=[session_array_to_num_samples_mapping[f'{session_id}_{p}']])
                # s: ArrayIntervall(shape=[num_samples])
                for s in speaker_ids
            }
            for p in perspective_tmp
        }
        if garbage_class is True:
            for p in perspective_tmp:
                num_samples = session_array_to_num_samples_mapping[
                    f'{session_id}_{p}']
                if use_ArrayIntervall:
                    noise = zeros(shape=[num_samples])
                    noise[:] = 1
                else:
                    noise = np.ones(shape=[num_samples], dtype=dtype)

                all_acitivity[session_id][p]['Noise'] = noise
        elif garbage_class is False:
            noise = zeros(shape=[num_samples])
            for p in perspective_tmp:
                all_acitivity[session_id][p]['Noise'] = noise
        elif garbage_class is None:
            pass
        else:
            raise ValueError(garbage_class)

        for ex in it_S:
            for pers in perspective_tmp:
                target_speaker = ex['target_speaker']
                # example_id = ex['example_id']

                if pers == 'global_worn':
                    perspective_mic_array = target_speaker
                else:
                    perspective_mic_array = pers

                if target_speaker == 'unknown' and ex['transcription'] == '[redacted]':
                    continue

                if perspective_mic_array.startswith('P'):
                    start = ex['start']['worn_microphone'][perspective_mic_array]
                    end = ex['end']['worn_microphone'][perspective_mic_array]
                else:
                    if not perspective_mic_array in ex['audio_path']['observation']:
                        continue
                    start = ex['start']['observation'][perspective_mic_array]
                    start, = list(set(start))
                    end = ex['end']['observation'][perspective_mic_array]
                    end, = list(set(end))

                start = start
                end = end

                if non_sil_alignment_fn is None:
                    value = 1
                else:
                    # value = ex['non_sil_alignment']
                    value = non_sil_alignment_fn(ex, perspective_mic_array)

                if debug:
                    all_acitivity[session_id][pers][target_speaker][start:end] += value
                else:
                    all_acitivity[session_id][pers][target_speaker][start:end] = value

        del it_S
    return all_acitivity


def _dummy():
    """
    >>> from IPython.lib.pretty import pprint
    >>> from nt.database.chime5 import Chime5
    >>> db = Chime5()
    >>> it = db.get_iterator_by_names(['dev'])
    >>> ali_path = ['/net/vol/jenkins/kaldi/2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb/egs/chime5/s5/exp/tri3_all_dev_worn_ali']
    >>> activity = get_activity(
    ...     it.filter(lambda ex: ex['target_speaker'] != 'unknown', lazy=False).map(adjust_start_end),
    ...     perspective='worn',
    ...     garbage_class=None,
    ...     dtype=np.bool,
    ...     non_sil_alignment_fn=get_function_example_to_non_sil_alignment(ali_path),
    ...     use_ArrayIntervall=True,
    ... )  #doctest: +ELLIPSIS
    Warning: Could not find P05_S02_0038340-0039534 ...
    >>> from cbj.mem import get_size_bosswissam, get_size_hall
    >>> get_size_hall(activity)
    ByteSize('10_628_415 B')
    >>> activity_S02_P05_P05 = []
    >>> activity_S02_P05_P05 += [activity['S02']['P05']['P05'][:]]
    >>> np.mean(activity['S02']['P05']['P05'][:])
    0.27750043800342317
    >>> activity_S02_P05_P05 += [get_all_activity_for_worn_alignments(perspective='worn', garbage_class=None, ali_path=ali_path[0])['S02']['P05']['P05'][:]]
    >>> np.mean(get_all_activity_for_worn_alignments(perspective='worn', garbage_class=None, ali_path=ali_path[0])['S02']['P05']['P05'][:])
    0.27750043800342317
    >>> activity = get_activity(
    ...     it.filter(lambda ex: ex['target_speaker'] != 'unknown', lazy=False).map(adjust_start_end),
    ...     perspective='worn',
    ...     garbage_class=None,
    ...     dtype=np.bool,
    ...     non_sil_alignment_fn=get_function_example_to_non_sil_alignment(ali_path),
    ... )  #doctest: +ELLIPSIS
    Warning: Could not find P05_S02_0038340-0039534 ...
    >>> activity_S02_P05_P05 += [activity['S02']['P05']['P05']]
    >>> for i in range(len(activity_S02_P05_P05)):
    ...     np.testing.assert_equal(activity_S02_P05_P05[0], activity_S02_P05_P05[1], err_msg=f'{i}')
    >>> get_size_hall(activity)
    ByteSize('4_112_776_792 B')
    >>> activity.keys()
    dict_keys(['S02', 'S09'])
    >>> activity['S02'].keys()
    dict_keys(['P05', 'P06', 'P07', 'P08'])
    >>> activity['S02']['P05'].keys()
    dict_keys(['P05', 'P06', 'P07', 'P08'])
    >>> np.set_string_function(lambda a: f'array(mean={np.mean(a)})')
    >>> pprint(activity)
    {'S02': {'P05': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.33762876177555357),
       'P07': array(mean=0.2070824030440115),
       'P08': array(mean=0.2496243278332083)},
      'P06': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.33762876177555357),
       'P07': array(mean=0.2070824030440115),
       'P08': array(mean=0.2496243278332083)},
      'P07': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.33762876177555357),
       'P07': array(mean=0.2070824030440115),
       'P08': array(mean=0.2496243278332083)},
      'P08': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.33762876177555357),
       'P07': array(mean=0.2070824030440115),
       'P08': array(mean=0.2496243278332083)}},
     'S09': {'P25': {'P25': array(mean=0.24691648502332622),
       'P26': array(mean=0.17939263866217411),
       'P27': array(mean=0.1926943769924331),
       'P28': array(mean=0.22845449328140732)},
      'P26': {'P25': array(mean=0.24691648502332622),
       'P26': array(mean=0.17939263866217411),
       'P27': array(mean=0.1926943769924331),
       'P28': array(mean=0.22845449328140732)},
      'P27': {'P25': array(mean=0.24691648502332622),
       'P26': array(mean=0.17939263866217411),
       'P27': array(mean=0.1926943769924331),
       'P28': array(mean=0.22845449328140732)},
      'P28': {'P25': array(mean=0.24691648502332622),
       'P26': array(mean=0.17939263866217411),
       'P27': array(mean=0.1926943769924331),
       'P28': array(mean=0.22845449328140732)}}}

    >>> activity = get_activity(
    ...     it.filter(lambda ex: ex['target_speaker'] != 'unknown', lazy=False).map(adjust_start_end),
    ...     perspective='worn',
    ...     garbage_class=False,
    ...     dtype=np.bool,
    ...     # non_sil_alignment_fn=get_function_add_non_sil_alignment(ali_path),
    ... )  #doctest: +ELLIPSIS
    >>> pprint(activity)
    {'S02': {'P05': {'P05': array(mean=0.4502383187856299),
       'P06': array(mean=0.476081643838078),
       'P07': array(mean=0.31896532360591373),
       'P08': array(mean=0.3996792116275309),
       'Noise': array(mean=0.0)},
      'P06': {'P05': array(mean=0.45023821349634546),
       'P06': array(mean=0.476081643838078),
       'P07': array(mean=0.31896532360591373),
       'P08': array(mean=0.3996791133575321),
       'Noise': array(mean=0.0)},
      'P07': {'P05': array(mean=0.45023813628420356),
       'P06': array(mean=0.47608122268094033),
       'P07': array(mean=0.31896532360591373),
       'P08': array(mean=0.3996790150875333),
       'Noise': array(mean=0.0)},
      'P08': {'P05': array(mean=0.4502383187856299),
       'P06': array(mean=0.476081643838078),
       'P07': array(mean=0.31896532360591373),
       'P08': array(mean=0.399679246723959),
       'Noise': array(mean=0.0)}},
     'S09': {'P25': {'P25': array(mean=0.35883135724927984),
       'P26': array(mean=0.2900560656541185),
       'P27': array(mean=0.2554168348118473),
       'P28': array(mean=0.3141167508207114),
       'Noise': array(mean=0.0)},
      'P26': {'P25': array(mean=0.35883135724927984),
       'P26': array(mean=0.29005637110816634),
       'P27': array(mean=0.2554168348118473),
       'P28': array(mean=0.3141170126384667),
       'Noise': array(mean=0.0)},
      'P27': {'P25': array(mean=0.3588308947045788),
       'P26': array(mean=0.29005573401829515),
       'P27': array(mean=0.2554168348118473),
       'P28': array(mean=0.314116384275854),
       'Noise': array(mean=0.0)},
      'P28': {'P25': array(mean=0.35883135724927984),
       'P26': array(mean=0.29005637110816634),
       'P27': array(mean=0.2554168348118473),
       'P28': array(mean=0.3141170737292763),
       'Noise': array(mean=0.0)}}}
    >>> activity = get_activity(
    ...     it.filter(lambda ex: ex['target_speaker'] != 'unknown', lazy=False).map(adjust_start_end),
    ...     perspective='array',
    ...     garbage_class=True,
    ...     dtype=np.bool,
    ...     non_sil_alignment_fn=get_function_example_to_non_sil_alignment(ali_path),
    ... )  #doctest: +ELLIPSIS
    Warning: Could not find P05_S02_0038340-0039534 ...
    >>> pprint(activity)
    {'S02': {'U01': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.33762820023270335),
       'P07': array(mean=0.20708050783689202),
       'P08': array(mean=0.24962333109464918),
       'Noise': array(mean=1.0)},
      'U02': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.3376276176319963),
       'P07': array(mean=0.20707807916406484),
       'P08': array(mean=0.2496243278332083),
       'Noise': array(mean=1.0)},
      'U03': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.3376279756155633),
       'P07': array(mean=0.2070824030440115),
       'P08': array(mean=0.2496243278332083),
       'Noise': array(mean=1.0)},
      'U04': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.3376276176319963),
       'P07': array(mean=0.20707770012264096),
       'P08': array(mean=0.24962263618537203),
       'Noise': array(mean=1.0)},
      'U05': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.337627905422707),
       'P07': array(mean=0.2070824030440115),
       'P08': array(mean=0.2496243278332083),
       'Noise': array(mean=1.0)},
      'U06': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.33762788436485014),
       'P07': array(mean=0.2070824030440115),
       'P08': array(mean=0.24962399090749818),
       'Noise': array(mean=1.0)}},
     'S09': {'U01': {'P25': array(mean=0.24691648502332622),
       'P26': array(mean=0.17939263866217411),
       'P27': array(mean=0.19269398426580017),
       'P28': array(mean=0.22845449328140732),
       'Noise': array(mean=1.0)},
      'U02': {'P25': array(mean=0.24691648502332622),
       'P26': array(mean=0.17939263866217411),
       'P27': array(mean=0.19269401917483422),
       'P28': array(mean=0.22845440600882222),
       'Noise': array(mean=1.0)},
      'U03': {'P25': array(mean=0.24691595266055713),
       'P26': array(mean=0.17939263866217411),
       'P27': array(mean=0.1926943769924331),
       'P28': array(mean=0.22845449328140732),
       'Noise': array(mean=1.0)},
      'U04': {'P25': array(mean=0.24691648502332622),
       'P26': array(mean=0.17939263866217411),
       'P27': array(mean=0.1926943769924331),
       'P28': array(mean=0.22845355946474677),
       'Noise': array(mean=1.0)},
      'U06': {'P25': array(mean=0.24691605738765923),
       'P26': array(mean=0.17939263866217411),
       'P27': array(mean=0.1926943769924331),
       'P28': array(mean=0.22845449328140732),
       'Noise': array(mean=1.0)}}}
    >>> activity = get_activity(
    ...     it.filter(lambda ex: ex['target_speaker'] != 'unknown', lazy=False).map(adjust_start_end),
    ...     perspective='global_worn',
    ...     garbage_class=None,
    ...     dtype=np.bool,
    ...     non_sil_alignment_fn=get_function_example_to_non_sil_alignment(ali_path),
    ... )  #doctest: +ELLIPSIS
    Warning: Could not find P05_S02_0038340-0039534 ...
    >>> pprint(activity)
    {'S02': {'global_worn': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.33762876177555357),
       'P07': array(mean=0.2070824030440115),
       'P08': array(mean=0.2496243278332083)}},
     'S09': {'global_worn': {'P25': array(mean=0.24691648502332622),
       'P26': array(mean=0.17939263866217411),
       'P27': array(mean=0.1926943769924331),
       'P28': array(mean=0.22845449328140732)}}}
    >>> from cbj.mem import get_size_bosswissam
    >>> get_size_bosswissam(activity)
    ByteSize('1_028_195_238 B')

    """
