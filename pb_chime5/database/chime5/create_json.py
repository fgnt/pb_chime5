"""
ToDo:
 - remove redundant information for array start, end and num samples
 - add worn eval when they get released
 -
"""

import logging
import os
from datetime import datetime
from functools import partial

import click

from pathlib import Path
from pb_chime5.database import keys
from pb_chime5.database.helper import (
    dump_database_as_json,
    click_common_options,
    check_audio_files_exist,
    click_convert_to_path,
)
from pb_chime5.database.chime5.get_speaker_activity import to_samples
from pb_chime5.database.chime5 import CHiME5_Keys

from pb_chime5 import git_root
from pb_chime5.io.json_module import load_json
from pb_chime5.mapping import Dispatcher


EVAL_TRANSCRIPTIONS_MISSING = False


for k in dir(keys):
    if k == k.upper():
        setattr(CHiME5_Keys, k, getattr(keys, k))


CH_K = CHiME5_Keys

FEMALE_SPEAKER = [
    'P14', 'P15', 'P17', 'P19', 'P36', 'P49', 'P52', 'P41', 'P43', 'P44',
    'P53', 'P56', 'P05', 'P08', 'P25', 'P26', 'P27', 'P28', 'P01', 'P02',
    'P08', 'P46', 'P48'
]
NUM_ARRAYS = 6
NUM_MICS = 4
SAMPLE_RATE = 16000
FORMAT_STRING = '%H:%M:%S.%f'
WORN_MICS = ['left', 'right']
NOTES_DICT = dict(SO3='P11 dropped from min ~15 to ~30',
                  S12='Last 15 minutes of U05 missing',
                  S19='P52 mic unreliable',
                  S23='Neighbour interrupts',
                  S24='P54 mic unreliable, P53 disconnects for bathroom',
                  S01='No registration tone')

# Note for CHiME-6
# The audio-sync version of the evaluation data (S01) is not released yet (Nov 2019)
# and we only use the training and development data sets at this moment.
# This will be fixed once the evaluation data is released
set_length_chime5 = dict(
    train=dict(S03=4090, S04=5563, S05=4939, S06=5097, S07=3656, S17=5892,
               S08=6175, S16=5004, S12=3300, S13=4193, S19=4292, S20=5365,
               S18=4907, S22=4758, S23=7054, S24=5695),
    dev=dict(S02=3822, S09=3618),
    eval=dict(S01=5797, S21=5231)
)
set_length_chime6 = dict(
    train=dict(S03=4090, S04=5563, S05=4939, S06=5097, S07=3656, S17=5892,
               S08=6175, S16=5004, S12=3300, S13=4193, S19=4292, S20=5365,
               S18=4907, S22=4758, S23=7054, S24=5695),
    dev=dict(S02=3822, S09=3618)
    # eval=dict(S01=5797, S21=5231)
)


def create_database(database_path, transcription_realigned_path, chime6):
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    datasets = dict()
    alias = dict()

    transcription_realigned_pathes = Dispatcher({
        p.name: p
        for p in Path(transcription_realigned_path).glob('**/*.json')
    })

    kaldi_transcriptions = dict()

    if chime6:
        set_length = set_length_chime6
    else:
        set_length = set_length_chime5

    for dataset in set_length.keys():
        out_dict = get_dataset(database_path, dataset, transcription_realigned_pathes, kaldi_transcriptions, chime6)
        for session_id, v in out_dict.items():
            datasets[session_id] = v
        alias[dataset] = list(out_dict.keys())
    return {keys.DATASETS: datasets, 'alias': alias}


def transform_transciption_list(transciption_list, chime6=False):
    """
    >>> transciption_list = [{
    ...     "end_time": {
    ...         "original": "0:01:0.3800000",
    ...         "U01": "0:01:0.3753125",
    ...         "U02": "0:01:0.3863125",
    ...         "U03": "0:01:0.3755625",
    ...         "U04": "0:01:0.3663125",
    ...         "U05": "0:01:0.3827500",
    ...         "U06": "0:01:0.3565625",
    ...         "P09": "0:01:0.3802500",
    ...         "P10": "0:01:0.3799375",
    ...         "P11": "0:01:0.3795000",
    ...         "P12": "0:01:0.3800000"
    ...     },
    ...     "start_time": {
    ...         "original": "0:00:57.5400000",
    ...         "U01": "0:00:57.5353125",
    ...         "U02": "0:00:57.5463750",
    ...         "U03": "0:00:57.5355000",
    ...         "U04": "0:00:57.5271250",
    ...         "U05": "0:00:57.5426875",
    ...         "U06": "0:00:57.5176875",
    ...         "P09": "0:00:57.5402500",
    ...         "P10": "0:00:57.5399375",
    ...         "P11": "0:00:57.5395625",
    ...         "P12": "0:00:57.5400000"
    ...     },
    ...     "words": "[noise] What were we talking about again? [inaudible 0:00:58.96]",
    ...     "speaker": "P12",
    ...     "session_id": "S03"
    ... }]
    >>> from IPython.lib.pretty import pprint
    >>> pprint(transform_transciption_list(transciption_list))
    [{'end_time': {'original': 966080,
       'U01': 966005,
       'U02': 966181,
       'U03': 966009,
       'U04': 965861,
       'U05': 966124,
       'U06': 965705,
       'P09': 966084,
       'P10': 966079,
       'P11': 966072,
       'P12': 966080},
      'start_time': {'original': 920640,
       'U01': 920565,
       'U02': 920742,
       'U03': 920568,
       'U04': 920434,
       'U05': 920683,
       'U06': 920283,
       'P09': 920644,
       'P10': 920639,
       'P11': 920633,
       'P12': 920640},
      'words': '[noise] What were we talking about again? [inaudible 0:00:58.96]',
      'speaker': 'P12',
      'session_id': 'S03'}]

    >>> chime6_transciption_list = [{
    ...     "end_time": "0:01:0.3800000",
    ...     "start_time": "0:00:57.5400000",
    ...     "words": "[noise] What were we talking about again? [inaudible 0:00:58.96]",
    ...     "speaker": "P12",
    ...     "session_id": "S03"
    ... }]
    >>> from IPython.lib.pretty import pprint
    >>> pprint(transform_transciption_list(chime6_transciption_list, chime6=True))
    [{'end_time': 966080,
      'start_time': 920640,
      'words': '[noise] What were we talking about again? [inaudible 0:00:58.96]',
      'speaker': 'P12',
      'session_id': 'S03'}]
      
      
    >>> transform_transciption_list(transciption_list, chime6=True) # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: The transcriptions have CHiME5 style (start_time has type dict), but the chime6 option was set.
    Example transcription:
    {'end_time': {'original': '0:01:0.3800000', ...}, ...}
    >>> transform_transciption_list(chime6_transciption_list, chime6=False) # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: The transcriptions have CHiME6 style (start_time has type int), but the chime6 option was not set.
    Example transcription:
    {'end_time': '0:01:0.3800000', ...}


    """
    if chime6:
        if not isinstance(transciption_list[0]['start_time'], str):
            if isinstance(transciption_list[0]['start_time'], dict):
                raise ValueError(
                    'The transcriptions have CHiME5 style '
                    '(start_time has type dict), but the chime6 option was '
                    'set.\n'
                    'Example transcription:\n'
                    f'{transciption_list[0]}'
                )
            else:
                raise RuntimeError(
                    'start_time in the transcriptions has the wrong datatype.\n'
                    'Expected a str, got:\n'
                    f'{transciption_list[0]}'
                )

        return [
            {
                key: to_samples(value) if key.endswith('time') else value
                for key, value in example.items()
            }
            for example in transciption_list
        ]

    if not isinstance(transciption_list[0]['start_time'], dict):
        if isinstance(transciption_list[0]['start_time'], str):
            raise ValueError(
                'The transcriptions have CHiME6 style '
                '(start_time has type int), but the chime6 option was not '
                'set.\n'
                'Example transcription:\n'
                f'{transciption_list[0]}'
            )
        else:
            raise RuntimeError(
                'start_time in the transcriptions has the wrong datatype.\n'
                'Expected a dict of str, got:\n'
                f'{transciption_list[0]}'
            )

    def transform(entry):
        return {
            k: {
                array_id: to_samples(time)
                for array_id, time in v.items()
            } if k.endswith('time') else v
            for k, v in entry.items()
        }

    return list(map(transform, transciption_list))


def load_transciption_json(path, chime6):
    try:
        return transform_transciption_list(load_json(path), chime6)
    except Exception as e:
        raise RuntimeError(
            'See above exception msg.\n'
            f'The problematic json file is {path}.'
        ) from e


def get_dataset(database_path, dataset, transcription_realigned_path, kaldi_transcriptions, chime6):
    # database_path = database_path / 'CHiME5'
    dataset_transciption_path = database_path / 'transcriptions' / dataset
    dataset_transciption_realigned_path = transcription_realigned_path #  / dataset
    dataset_audio_path = database_path / 'audio' / dataset
    json_dict = dict()
    for session_path in dataset_transciption_path.glob('*.json'):
        session_id = session_path.name.split('.')[0]
        json_dict[session_id] = {}
        trans = load_transciption_json(session_path, chime6=chime6)
        trans_realigned = load_transciption_json(dataset_transciption_realigned_path[session_path.name], chime6=chime6)

        total = len(trans)
        assert len(trans) == len(trans_realigned), (len(trans), len(trans_realigned))
        # ToDo: Fix this exception to test equality.
        #       In chime6 the number of utterances changed -> Disabled at the moment
        if not chime6 and total < set_length_chime5[dataset][session_id]:
            raise ValueError(
                f'missing utterances in session {session_id} expected length'
                f' {set_length_chime5[dataset][session_id]} available length {total}')

        assert total > 0, ('Each session should have at least one example', dataset_transciption_realigned_path[session_path.name])
        # elif total > set_length[dataset][session_id]:
        #     warn(f'there are {total - set_length[dataset][session_id]} examples'
        #           f' more than expected in session {session_id}')
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(os.cpu_count()) as ex:
            for example_id, example in ex.map(
                    partial(
                        get_example,
                        audio_path=dataset_audio_path,
                        kaldi_transcriptions=kaldi_transcriptions,
                        chime6=chime6,
                    ),
                    trans, trans_realigned
            ):
                if example_id in ['P45_S21_0356170-0356149']:
                    # The number of samples is negative
                    continue

                json_dict[session_id][example_id] = example
    return json_dict


def get_example(transcription, transcription_realigned, audio_path, kaldi_transcriptions, chime6):
    from pb_chime5.database.chime5.mapping import session_speakers_mapping

    session_id = transcription['session_id']

    from .mapping import session_dataset_mapping
    dataset = session_dataset_mapping[session_id]

    notes = list()
    if chime6:
        if not isinstance(transcription['start_time'], int):
            if isinstance(transcription['start_time'], dict):
                raise ValueError(
                    'You requested the CHiME6 json, but the transcriptions '
                    'have CHiME5 style.\n'
                    'In CHiME6 start_time and end_time are no longer dict.\n'
                    f'Got:\n{transcription}'
                )
            else:
                raise RuntimeError(
                    'Something went wrong. Expected that start_time is an '
                    'integer that represents the number of samples.'
                    f'Got:\n{transcription}'
                )
        speaker_ids = session_speakers_mapping[session_id]
    else:
        speaker_ids = [
            key
            for key in transcription['start_time'].keys()
            if 'P' in key
        ]
        if EVAL_TRANSCRIPTIONS_MISSING and session_id in ['S01', 'S21']:
            # eval
            assert speaker_ids == [], (speaker_ids, session_id)
            speaker_ids = session_speakers_mapping[session_id]
        else:
            if speaker_ids == [] and session_id in ['S01', 'S21']:
                raise AssertionError(
                    'The eval transcriptions are missing.\n'
                    'The eval transcriptions were released after the challenge.\n'
                    'When you do not want to download them, you can try to '
                    'set `EVAL_TRANSCRIPTIONS_MISSING` in this file to True.\n'
                    'But the code is not tested, to work with missing eval'
                    'transcriptions.'
                )
            assert speaker_ids == session_speakers_mapping[session_id], (speaker_ids, session_id)

    try:
        target_speaker_id = transcription['speaker']
    except KeyError as e:
        target_speaker_id = 'unknown'
        notes.append('target_speaker_id is missing')

    if chime6:
        start_sample = transcription['start_time']
        end_sample = transcription['end_time']
    else:
        start_sample = transcription['start_time']['original']
        end_sample = transcription['end_time']['original']

    example_id = get_example_id(
        start_sample=start_sample,
        end_sample=end_sample,
        session_id=session_id,
        speaker_id=target_speaker_id,
        chime6=chime6,
    )

    arrays = [f'U0{array+1}' for array in range(NUM_ARRAYS)]
    if session_id in ['S05']:
        if chime6:
            del arrays[3]
            del arrays[2]
            notes.append('Array U03 and U04 are missing, this is expected')
        else:
            del arrays[2]
            notes.append('Array U03 is missing, this is expected')
    elif session_id in ['S22']:
        del arrays[2]
        notes.append('Array U03 is missing, this is expected')
    elif session_id == 'S09':
        del arrays[4]
        notes.append('Array U05 is missing, this is expected')

    audio_path_dict = get_audio_path_dict(
        arrays,
        speaker_ids,
        session_id,
        audio_path,
        dataset,
    )

    if chime6:
        start_time_dict = transcription_realigned['start_time']
        end_time_dict = transcription_realigned['end_time']
        num_samples = end_time_dict - start_time_dict
    else:
        start_time_dict = get_time_from_dict(
            transcription_realigned['start_time'],
            speaker_ids,
            arrays,
            dataset,
        )
        end_time_dict = get_time_from_dict(
            transcription_realigned['end_time'],
            speaker_ids,
            arrays,
            dataset,
        )
        num_samples = get_num_samples(start_time_dict, end_time_dict)

    empty_keys = [
        key for key1, mic_samples in audio_path_dict.items()
        if key1 != 'original'
        for key, value in mic_samples.items()
        if (isinstance(value, int) and value == 0) or
           (isinstance(value, list) and value[0] == 0)
    ]
    for key in empty_keys:
        if not chime6:
            del num_samples[keys.OBSERVATION][key]
            del start_time_dict[keys.OBSERVATION][key]
            del end_time_dict[keys.OBSERVATION][key]
        del audio_path_dict[keys.OBSERVATION][key]
        notes.append(f'Array {key} is missing, this may be expected')

    gender = {id: 'male' for id in speaker_ids}
    gender.update({id: 'female' for id in speaker_ids if id in FEMALE_SPEAKER})
    if 'location' in transcription:
        location = transcription['location']
        ref_array = transcription['ref']
    else:
        location = 'unknown'
        ref_array = 'unknown'
    if session_id in NOTES_DICT:
        notes.append(NOTES_DICT[session_id])

    d = {
        CH_K.SESSION_ID: session_id,
        # CH_K.TARGET_SPEAKER: target_speaker_id,
        keys.NUM_SAMPLES: num_samples,
        keys.AUDIO_PATH: audio_path_dict,
        CH_K.NOTES: notes,
        keys.START: start_time_dict,
        keys.END: end_time_dict,
        # CH_K.REFERENCE_ARRAY: ref_array,
        keys.TRANSCRIPTION: transcription['words'],
    }
    if target_speaker_id == 'unknown':
        pass
    else:
        d[keys.SPEAKER_ID] = target_speaker_id
        d[keys.GENDER] = gender[target_speaker_id]

    if example_id in kaldi_transcriptions:
        d[keys.KALDI_TRANSCRIPTION] = kaldi_transcriptions[example_id]

    if location != 'unknown':
        d[CH_K.LOCATION] = location

    if ref_array != 'unknown':
        d[CH_K.REFERENCE_ARRAY] = ref_array

    return example_id, d


def time_to_string_format(time):
    # format to time string to fit time in kaldi example_ids
    return str(int(to_samples(time) * 100 / SAMPLE_RATE)).zfill(7)


def get_example_id(
        start_sample,
        end_sample,
        speaker_id,
        session_id,
        chime6,
):
    """
    >>> get_example_id(45963520, 45987360, 'P09', 'S03')
    'P09_S03_0287272-0287421'

    """
    start_sample_str = str(int(start_sample * 100 / SAMPLE_RATE)).zfill(7)
    end_sample_str = str(int(end_sample * 100 / SAMPLE_RATE)).zfill(7)
    if chime6:
        return f'{speaker_id}_{session_id}-{start_sample_str}-{end_sample_str}'
    else:
        return f'{speaker_id}_{session_id}_{start_sample_str}-{end_sample_str}'


from functools import lru_cache


def get_audio_path_dict(arrays, speaker_ids, session_id, audio_path, dataset):
    observation = {
        keys.OBSERVATION: {
            array: [
                str(audio_path / f'{session_id}_{array}.CH{mic}.wav')
                for mic in range(1, 1+NUM_MICS)
            ]
            for array in arrays
        }
    }
    if dataset == 'eval':
        worn_microphone = {}
    else:
        worn_microphone = {
            CH_K.WORN: {
                speaker: str(audio_path / f'{session_id}_{speaker}.wav')
                for speaker in speaker_ids
            }
        }

    audio_path_dict = {**observation, **worn_microphone}

    return audio_path_dict


def get_num_samples(start_time_dict, end_time_dict):
    def get_num_samples_recursive(start_time, end_time):
        if isinstance(start_time, dict):
            return {
                key: get_num_samples_recursive(value, end_time[key])
                for key, value in start_time.items()
            }
        elif isinstance(start_time, list):
            return [
                e_time - s_time
                for s_time, e_time in zip(start_time, end_time)
            ]
        elif isinstance(start_time, int):
            return end_time - start_time
        else:
            raise ValueError('expected dict or list')

    return get_num_samples_recursive(start_time_dict, end_time_dict)


def get_duration(start_time, end_time):
    duration = datetime.strptime(end_time, FORMAT_STRING) - datetime.strptime(
        start_time, FORMAT_STRING)
    return int(duration.total_seconds() * SAMPLE_RATE)


def get_time_from_dict(time, speaker_ids, arrays, dataset):
    observation = {
        keys.OBSERVATION: {
            array: time[array]
            # array: [time[array]] * NUM_MICS
            for array in arrays
        }
    }
    if dataset == 'eval':
        worn_microphone = {}
    else:
        worn_microphone = {
            CH_K.WORN: {
                speaker: time[speaker]
                for speaker in speaker_ids
            }
        }

    time_dict = {
        **observation,
        **worn_microphone,
        'original': time['original']
    }

    return time_dict


@click.command()
@click_common_options('chime5.json', git_root / 'CHiME5')
@click.option(
    '--transcription-path',
    type=click.Path(),
    callback=click_convert_to_path
)
@click.option('--chime6', default=False, is_flag=True)
def main(
        database_path: Path,
        json_path: Path,
        transcription_path: Path,
        chime6,
):
    database_path = database_path.expanduser().resolve()
    assert database_path.exists(), database_path
    json_path = json_path.expanduser().resolve()
    transcription_path = transcription_path.expanduser().resolve()
    assert transcription_path.exists(), database_path

    json = create_database(database_path, transcription_path, chime6)

    print('Check that all wav files in the json exsist.')
    check_audio_files_exist(json, speedup='thread')
    print('Finished check. Write json to disk:')
    dump_database_as_json(json_path, json)
    print('Finished write')


if __name__ == '__main__':
    main()
