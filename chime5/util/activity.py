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


def get_function_add_non_sil_alignment(ali_path):
    alignment = get_phone_alignment(ali_path)
    non_sil_alignment_dict = Dispatcher({k: v != 'sil' for k, v in alignment.items()})

    def transform_add_non_sil_alignment(
            ex
    ):
        if ex['example_id'] in non_sil_alignment_dict:
            ex['non_sil_alignment'] = activity_frequency_to_time(
                non_sil_alignment_dict[ex['example_id']],
                stft_window_length=400,
                stft_shift=160,
                stft_fading=False,
                time_length=ex['num_samples']['worn_microphone'][
                    ex['target_speaker']],
            )
        else:
            print(
                f"Warning: Could not find {ex['example_id']} in non_sil_alignment.")
            ex['non_sil_alignment'] = 1

        return ex

    return transform_add_non_sil_alignment


def get_activity(
        iterator,
        # session_id,
        *,

        perspective,
        garbage_class,
        dtype=np.bool,
        non_sil_alignment=False,
        debug=False
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

        num_samples = session_array_to_num_samples_mapping[f'{session_id}_P']
        speaker_ids = session_speakers_mapping[session_id]

        acitivity = {
            p: {
                s: np.zeros(shape=[num_samples], dtype=dtype)
                for s in speaker_ids
            }
            for p in perspective_tmp
        }
        if garbage_class is True:
            noise = np.ones(shape=[num_samples], dtype=dtype)
            for p in perspective_tmp:
                acitivity[p]['Noise'] = noise
        elif garbage_class is False:
            noise = np.ones(shape=[num_samples], dtype=dtype)
            for p in perspective_tmp:
                acitivity[p]['Noise'] = noise
        elif garbage_class is None:
            pass
        else:
            raise ValueError(garbage_class)

        for ex in it_S:

            for pers in perspective_tmp:
                target_speaker = ex['target_speaker']

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
                    start = ex['start']['observation'][perspective_mic_array]
                    start, = list(set(start))
                    end = ex['end']['observation'][perspective_mic_array]
                    end, = list(set(end))

                start = start
                end = end

                if non_sil_alignment:
                    value = ex['non_sil_alignment']
                else:
                    value = 1

                if debug:
                    acitivity[pers][target_speaker][start:end] += value
                else:
                    acitivity[pers][target_speaker][start:end] = value
        all_acitivity[session_id] = acitivity
        del acitivity
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
    ...     it.filter(lambda ex: ex['target_speaker'] != 'unknown', lazy=False).map(adjust_start_end).map(get_function_add_non_sil_alignment(ali_path)),
    ...     perspective='worn',
    ...     garbage_class=None,
    ...     dtype=np.bool,
    ...     non_sil_alignment=True,
    ... )  #doctest: +ELLIPSIS
    Warning: Could not find P05_S02_0038340-0039534 ...
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
    ...     it.filter(lambda ex: ex['target_speaker'] != 'unknown', lazy=False).map(adjust_start_end).map(get_function_add_non_sil_alignment(ali_path)),
    ...     perspective='array',
    ...     garbage_class=None,
    ...     dtype=np.bool,
    ...     non_sil_alignment=True,
    ... )  #doctest: +ELLIPSIS
    Warning: Could not find P05_S02_0038340-0039534 ...
    >>> pprint(activity)
    {'S02': {'U01': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.33762820023270335),
       'P07': array(mean=0.20708050783689202),
       'P08': array(mean=0.24962333109464918)},
      'U02': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.3376276176319963),
       'P07': array(mean=0.20707807916406484),
       'P08': array(mean=0.2496243278332083)},
      'U03': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.3376279756155633),
       'P07': array(mean=0.2070824030440115),
       'P08': array(mean=0.2496243278332083)},
      'U04': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.3376276176319963),
       'P07': array(mean=0.20707770012264096),
       'P08': array(mean=0.24962263618537203)},
      'U05': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.337627905422707),
       'P07': array(mean=0.2070824030440115),
       'P08': array(mean=0.2496243278332083)},
      'U06': {'P05': array(mean=0.27750043800342317),
       'P06': array(mean=0.33762788436485014),
       'P07': array(mean=0.2070824030440115),
       'P08': array(mean=0.24962399090749818)}},
     'S09': {'U01': {'P25': array(mean=0.24691648502332622),
       'P26': array(mean=0.17939263866217411),
       'P27': array(mean=0.19269398426580017),
       'P28': array(mean=0.22845449328140732)},
      'U02': {'P25': array(mean=0.24691648502332622),
       'P26': array(mean=0.17939263866217411),
       'P27': array(mean=0.19269401917483422),
       'P28': array(mean=0.22845440600882222)},
      'U03': {'P25': array(mean=0.24691595266055713),
       'P26': array(mean=0.17939263866217411),
       'P27': array(mean=0.1926943769924331),
       'P28': array(mean=0.22845449328140732)},
      'U04': {'P25': array(mean=0.24691648502332622),
       'P26': array(mean=0.17939263866217411),
       'P27': array(mean=0.1926943769924331),
       'P28': array(mean=0.22845355946474677)},
      'U06': {'P25': array(mean=0.24691605738765923),
       'P26': array(mean=0.17939263866217411),
       'P27': array(mean=0.1926943769924331),
       'P28': array(mean=0.22845449328140732)}}}
    >>> activity = get_activity(
    ...     it.filter(lambda ex: ex['target_speaker'] != 'unknown', lazy=False).map(adjust_start_end).map(get_function_add_non_sil_alignment(ali_path)),
    ...     perspective='global_worn',
    ...     garbage_class=None,
    ...     dtype=np.bool,
    ...     non_sil_alignment=True,
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
    """

