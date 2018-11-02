import numpy as np

from pb_chime5 import mapping
from pb_chime5.util import ArrayIntervall
from pb_chime5.nt.utils.mapping import Dispatcher


def get_activity(
        iterator,
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
            'global_worn' -- global perspective for worn ('P')
            'worn' -- return perspective for each speaker ('P01', ...)
            'array' -- return perspective for each array ('U01', ...)
    garbage_class: True, False, None
        True: garbage_class is always one
        False: garbage_class is always zero
        None: the number of classes is 4 and not 5
    non_sil_alignment_fn: None or a function with the signature:
        value = non_sil_alignment_fn(ex, perspective_mic_array)
        where
            ex is one example in iterator
            perspective_mic_array is in ['U01', ..., 'P01', ..., 'P']
            value is a 1d array indicating if at a sample the source is active
                or not
        use_ArrayIntervall: ArrayIntervall is a special datatype to reduce
            memory usage

    returns:
        dict[session_id][mic_perspective][speaker_id] = array(dtype=bool)
        session_id e.g.: 'S02', ...
        mic_perspective e.g.: 'P', 'P05', 'U01', ...
        speaker_id e.g.: 'P05', ...

    >>> from pb_chime5.nt.database.chime5 import Chime5
    >>> import textwrap
    >>> db = Chime5()
    >>> def display_activity(activity):
    ...     print(tuple(activity.keys()))
    ...     print(' '*2, tuple(activity['S02'].keys()))
    ...     print(' '*4, tuple(activity['S02']['P'].keys()))
    ...     print(' '*6, activity['S02']['P']['P05'])
    ...     print(' '*6, activity['S02']['P']['Noise'])
    >>> def display_activity(activity, indent=0):
    ...     indent_print = lambda x: print(textwrap.indent(str(x), ' '*indent))
    ...     if isinstance(activity, dict):
    ...         for i, (k, v) in enumerate(activity.items()):
    ...             if i == 0 or k in ['Noise']:
    ...                 indent_print(f'{k}:')
    ...                 display_activity(v, indent=indent+2)
    ...             else:
    ...                 indent_print(f'{k}: ...')
    ...     else:
    ...         indent_print(activity)
    >>> activity = get_activity(db.get_iterator_by_names('S02'), perspective='global_worn', garbage_class=True)
    >>> display_activity(activity)
    S02:
      P:
        P05:
          [False False False ... False False False]
        P06: ...
        P07: ...
        P08: ...
        Noise:
          [ True  True  True ...  True  True  True]
    >>> activity = get_activity(db.get_iterator_by_names('S02'), perspective='worn', garbage_class=False)
    >>> display_activity(activity)
    S02:
      P05:
        P05:
          [False False False ... False False False]
        P06: ...
        P07: ...
        P08: ...
        Noise:
          [False False False ... False False False]
      P06: ...
      P07: ...
      P08: ...
    >>> activity = get_activity(db.get_iterator_by_names('S02'), perspective='array', garbage_class=None)
    >>> display_activity(activity)
    S02:
      U01:
        P05:
          [False False False ... False False False]
        P06: ...
        P07: ...
        P08: ...
      U02: ...
      U03: ...
      U04: ...
      U05: ...
      U06: ...

    """

    dict_it_S = iterator.groupby(lambda ex: ex['session_id'])

    # Dispatcher is a dict with better KeyErrors
    all_acitivity = Dispatcher()
    for session_id, it_S in dict_it_S.items():

        if perspective == 'worn':
            perspective_tmp = mapping.session_to_speakers[session_id]
        elif perspective == 'global_worn':
            perspective_tmp = ['P']  # Always from target speaker
        elif perspective == 'array':
            # The mapping considers missing arrays
            perspective_tmp = mapping.session_to_arrays[session_id]
        else:
            perspective_tmp = perspective

            if not isinstance(perspective_tmp, (tuple, list)):
                perspective_tmp = [perspective_tmp, ]

        speaker_ids = mapping.session_to_speakers[session_id]

        if use_ArrayIntervall:
            assert dtype == np.bool, dtype
            zeros = ArrayIntervall

            def ones(shape):
                arr = zeros(shape=shape)
                arr[:] = 1
                return arr
        else:
            import functools
            zeros = functools.partial(np.zeros, dtype=dtype)
            ones = functools.partial(np.ones, dtype=dtype)

        all_acitivity[session_id] = Dispatcher({
            p: Dispatcher({
                s: zeros(shape=[mapping.session_array_to_num_samples[f'{session_id}_{p}']])
                # s: ArrayIntervall(shape=[num_samples])
                for s in speaker_ids
            })
            for p in perspective_tmp
        })

        if garbage_class is True:
            for p in perspective_tmp:
                num_samples = mapping.session_array_to_num_samples[
                    f'{session_id}_{p}']
                all_acitivity[session_id][p]['Noise'] = ones(
                    shape=[num_samples],
                )
        elif garbage_class is False:
            for p in perspective_tmp:
                num_samples = mapping.session_array_to_num_samples[
                    f'{session_id}_{p}']
                all_acitivity[session_id][p]['Noise'] = zeros(
                    shape=[num_samples]
                )
        elif garbage_class is None:
            pass
        elif isinstance(garbage_class, int) and garbage_class > 0:
            for noise_idx in range(garbage_class):
                for p in perspective_tmp:
                    num_samples = mapping.session_array_to_num_samples[
                        f'{session_id}_{p}'
                    ]
                    all_acitivity[session_id][p][f'Noise{noise_idx}'] = ones(
                        shape=[num_samples]
                    )
        else:
            raise ValueError(garbage_class)

        for ex in it_S:
            for pers in perspective_tmp:
                if ex['transcription'] == '[redacted]':
                    continue

                target_speaker = ex['speaker_id']
                # example_id = ex['example_id']

                if pers == 'P':
                    perspective_mic_array = target_speaker
                else:
                    perspective_mic_array = pers

                if perspective_mic_array.startswith('P'):
                    start = ex['start']['worn'][perspective_mic_array]
                    end = ex['end']['worn'][perspective_mic_array]
                else:
                    if not perspective_mic_array in ex['audio_path']['observation']:
                        continue
                    start = ex['start']['observation'][perspective_mic_array]
                    end = ex['end']['observation'][perspective_mic_array]

                if non_sil_alignment_fn is None:
                    value = 1
                else:
                    value = non_sil_alignment_fn(ex, perspective_mic_array)

                if debug:
                    all_acitivity[session_id][pers][target_speaker][start:end] += value
                else:
                    all_acitivity[session_id][pers][target_speaker][start:end] = value

        del it_S

    return all_acitivity
