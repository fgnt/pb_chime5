import numpy as np
from nt.utils.mapping import Dispatcher
from nt.database.chime5 import activity_frequency_to_time, adjust_start_end
from pb_chime5.util.alignment import get_phone_alignment


def get_non_sil_alignment_fn_from_kalid(
        ali_path,
        unique_alignments=True,
        channel_preference=None,
        # use_kaldi_id=False,
        add_statistics=False,
):
    """

    Args:
        ali_path:
        channel_preference:
        unique_alignments: (Only True is implemented)
         - When True assume that the alignments in ali_path are unique for a
           single utterance, e.g. only worn alignments or ref array alignments.
            - When more than one alignment is present, a non None
              channel_preference gives the priority what to choose.
         - When False assume there is an alignment for each array and/or
           speaker. For array enhancement: U01 - U06.
        use_kaldi_id:
            Distinguish different kaldi IDs.
            When False worn alignments can be used for the array.
        add_statistics:


    Returns:

    >>> worn_ali_path = '~/net/vol/jenkins/kaldi/2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb/egs/chime5/s5/exp/tri3_all_dev_worn_ali'
    >>> array_ali_path = '~/net/storage/jheymann/__share/jensheit/chime5/kaldi/arrayBSS_v5/exp/tri3_u_bss_js_cleaned_dev_new_bss_beam_39_ali'

    >>> from nt.database.chime5 import Chime5
    >>> db = Chime5()
    >>> it = db.get_iterator_for_session('S02', drop_unknown_target_speaker=True)
    >>> ex = it[1]
    >>> worn_non_sil_alignment_fn = get_non_sil_alignment_fn_from_kalid(worn_ali_path)
    >>> array_non_sil_alignment_fn = get_non_sil_alignment_fn_from_kalid(array_ali_path)
    >>> reference_array = ex['reference_array']
    >>> reference_array
    'U02'
    >>> worn_non_sil_alignment_fn(ex, reference_array).shape
    (45760,)
    >>> worn_non_sil_alignment_fn(ex, 'U06').shape
    (45760,)
    >>> worn_non_sil_alignment_fn(ex, 'P05').shape
    (45760,)
    >>> array_non_sil_alignment_fn(ex, reference_array).shape
    >>> array_non_sil_alignment_fn(ex, 'U06').shape
    >>> array_non_sil_alignment_fn(ex, 'P05').shape


    """

    if not unique_alignments:
        raise NotImplementedError('ToDo: Implement')

    alignment = get_phone_alignment(
        ali_path,
        # use_kaldi_id=use_kaldi_id,
        use_kaldi_id=True,
        unique_per_utt=unique_alignments,
        channel_preference=channel_preference,
    )
    non_sil_alignment_dict = Dispatcher({k: v != 'sil' for k, v in alignment.items()})

    # Because of perspective_mic_array is is usefull to use a cache
    # last = None

    import collections

    statistics = collections.defaultdict(set)

    from nt.database.chime5 import kaldi_id_to_parts
    from chime5.scripts.create_mapping_json import Chime5KaldiIdMapping, kaldi_to_nt_example_id
    nt_to_kaldi_id = Chime5KaldiIdMapping()

    # non_sil_alignment_dict

    source_key_mapping = {
        kaldi_to_nt_example_id(k): k
        for k in non_sil_alignment_dict.keys()
    }

    def example_to_non_sil_alignment(
            ex,
            perspective_mic_array,
    ):
        nonlocal statistics

        example_id = ex['example_id']

        # if use_kaldi_id:
        #     example_id, = nt_to_kaldi_id.get_array_ids_from_nt_id(
        #         example_id,
        #         arrays=perspective_mic_array,
        #         channels='ENH'
        #     )

        if perspective_mic_array[0] == 'P':
            target_time_length = ex['num_samples']['worn'][
                perspective_mic_array]
        elif perspective_mic_array[0] == 'U':
            target_time_length = ex['num_samples']['observation'][
                perspective_mic_array]
        else:
            raise NotImplementedError(perspective_mic_array)

        # ignore perspective_mic_array
        # if example_id in non_sil_alignment_dict:
        if example_id in source_key_mapping:
            source_example_id = source_key_mapping[example_id]

            array_id = kaldi_id_to_parts(source_example_id)['array_id']

            if array_id.startswith('U'):
                source_time_length = ex['num_samples']['observation'][array_id]
            elif array_id.startswith('P'):
                source_time_length = ex['num_samples']['worn'][array_id]
            else:
                raise ValueError(array_id, source_example_id, example_id)

            # if use_kaldi_id:
            #
            #
            #     source_time_length = target_time_length
            # else:
            #     source_time_length = ex['num_samples']['original']

            ret = activity_frequency_to_time(
                non_sil_alignment_dict[source_example_id],
                stft_window_length=400,
                stft_shift=160,
                stft_fading=False,
                time_length=source_time_length,
            )

            delta = target_time_length - source_time_length
            if delta > 0:
                start_pad = delta // 2
                end_pad = delta - start_pad
                ret = np.pad(ret, [start_pad, end_pad], mode='constant', constant_values=0)
            elif delta < 0:
                start_cut = abs(delta) // 2
                end_cut = abs(delta) - start_cut
                assert end_cut > 0, end_cut
                ret = ret[start_cut:-end_cut]
            else:
                pass  # same length

        else:
            # raise Exception(example_id, list(source_key_mapping.keys())[:10])
            print(
                f"Warning: Could not find {example_id} in non_sil_alignment."
            )
            ret = 1
            if add_statistics:
                session_id = ex['session_id']
                target_speaker = ex['target_speaker']
                statistics[f'{target_speaker}_{session_id}'].add(ex['example_id'])

        # if not use_kaldi_id:
            # last = ex['example_id'], ret
        return ret

    example_to_non_sil_alignment.statistics = statistics
    example_to_non_sil_alignment.non_sil_alignment_dict = non_sil_alignment_dict

    return example_to_non_sil_alignment


if __name__ == '__main__':
    import pickle
    from pathlib import Path

    from nt.database.chime5 import Chime5

    from pb_chime5.activity import get_activity
    from pb_chime5 import git_root

    db = Chime5()
    it = db.get_iterator_for_session(
        ['train', 'dev', 'eval'], drop_unknown_target_speaker=True
    )

    data = get_activity(
        iterator=it,
        perspective='array',
        garbage_class=True,
        dtype=np.bool,
        non_sil_alignment_fn=None,
        debug=False,
        use_ArrayIntervall=True,
    )

    path: Path = git_root / 'cache' / 'annotation'

    for session_id, v in data.items():
        path.mkdir(exist_ok=True, parents=True)
        with open(path / f'{session_id}.pkl', 'wb') as fd:
            pickle.dump(v, fd)

    worn_ali_path = '~/net/vol/jenkins/kaldi/2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb/egs/chime5/s5/exp/tri3_all_dev_worn_ali'

    if Path(worn_ali_path).exists():
        worn_non_sil_alignment_fn = get_non_sil_alignment_fn_from_kalid(
            worn_ali_path
        )

        it = db.get_iterator_for_session(
            ['dev'], drop_unknown_target_speaker=True
        )
        data = get_activity(
            iterator=it,
            perspective='array',
            garbage_class=True,
            dtype=np.bool,
            non_sil_alignment_fn=worn_non_sil_alignment_fn,
            debug=False,
            use_ArrayIntervall=True,
        )
        path: Path = git_root / 'cache' / 'word_non_sil_alignment'

        for session_id, v in data.items():
            path.mkdir(exist_ok=True, parents=True)
            with open(path / f'{session_id}.pkl', 'wb') as fd:
                pickle.dump(v, fd)
