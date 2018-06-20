

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

    # it_S = iterator.filter((lambda ex: ex['session_id'] == session_id), lazy=True)

    dict_it_S = iterator.groupby(lambda ex: ex['session_id'])

    all_acitivity = {}
    for session_id, it_S in dict_it_S.items():

        if perspective == 'worn':
            perspective_tmp = session_speakers_mapping[session_id]
        elif perspective == 'global_worn':
            perspective_tmp = [perspective]
        elif perspective == 'array':
            perspective_tmp = [f'U0{i}' for i in range(1, 7)]
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
                    end = ex['end']['observation'][perspective_mic_array]

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