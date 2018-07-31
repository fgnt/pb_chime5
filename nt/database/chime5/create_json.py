import logging
import os
from datetime import datetime
from functools import partial
from warnings import warn

import click
import tqdm
from pathlib import Path
from nt.database import keys
from nt.database.helper import (
    dump_database_as_json,
    click_common_options,
    check_audio_files_exist,
    click_convert_to_path,
)
from nt.database.chime5.get_speaker_activity import to_samples
from nt.io.data_dir import chime_5
from nt.io.json_module import load_json
from nt.utils.mapping import Dispatcher


EVAL_TRANSCRIPTIONS_MISSING = True


class CHiME5_Keys:
    """
    >>> print(dir(keys))
    """
    WORN = 'worn_microphone'
    TARGET_SPEAKER = 'target_speaker'
    NOTES = 'notes'
    SESSION_ID = 'session_id'
    LOCATION = 'location'
    REF_ARRAY = 'reference_array'


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

set_length = dict(
    train=dict(S03=4090, S04=5563, S05=4939, S06=5097, S07=3656, S17=5892,
               S08=6175, S16=5004, S12=3300, S13=4193, S19=4292, S20=5365,
               S18=4907, S22=4758, S23=7054, S24=5695),
    dev=dict(S02=3822, S09=3618),
    eval=dict(S01=5797, S21=5231)
)


def create_database(database_path, transcription_realigned_path):
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    datasets = dict()

    transcription_realigned_pathes = Dispatcher({
        p.name: p
        for p in Path(transcription_realigned_path).glob('**/*.json')
    })

    for dataset in set_length.keys():
        out_dict = get_dataset(database_path, dataset, transcription_realigned_pathes)
        datasets[dataset] = out_dict
    return {keys.DATASETS: datasets}


def transform_transciption_list(transciption_list):
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

    """

    def transform(entry):

        return {
            k: {
                array_id: to_samples(time)
                for array_id, time in v.items()
            } if isinstance(v, dict) else v
            for k, v in entry.items()
        }

    return list(map(transform, transciption_list))


def load_transciption_json(path):
    return transform_transciption_list(load_json(path))


def get_dataset(database_path, dataset, transcription_realigned_path):
    database_path = database_path / 'CHiME5'
    dataset_transciption_path = database_path / 'transcriptions' / dataset
    dataset_transciption_realigned_path = transcription_realigned_path #  / dataset
    dataset_audio_path = database_path / 'audio' / dataset
    json_dict = dict()
    for session_path in dataset_transciption_path.glob('*.json'):
        session_id = session_path.name.split('.')[0]
        trans = load_transciption_json(session_path)
        trans_realigned = load_transciption_json(dataset_transciption_realigned_path[session_path.name])

        total = len(trans)
        assert len(trans) == len(trans_realigned), (len(trans), len(trans_realigned))
        if total < set_length[dataset][session_id]:
            raise ValueError(
                f'missing utterances in session {session_id} expected length'
                f' {set_length[dataset][session_id]} available length {total}')
        # elif total > set_length[dataset][session_id]:
        #     warn(f'there are {total - set_length[dataset][session_id]} examples'
        #           f' more than expected in session {session_id}')
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(os.cpu_count()) as ex:
            for example_id, example in ex.map(
                    partial(get_example, audio_path=dataset_audio_path),
                    trans, trans_realigned
            ):
                json_dict[example_id] = example
    return json_dict


def get_example(transcription, transcription_realigned, audio_path):
    from nt.database.chime5.mapping import session_speakers_mapping

    session_id = transcription['session_id']

    from .mapping import session_dataset_mapping
    dataset = session_dataset_mapping[session_id]

    notes = list()
    speaker_ids = [
        key
        for key in transcription['start_time'].keys()
        if 'P' in key
    ]
    if session_id in ['S01', 'S21']:
        # eval
        assert speaker_ids == [], (speaker_ids, session_id)
        speaker_ids = session_speakers_mapping[session_id]
    else:
        assert speaker_ids == session_speakers_mapping[session_id], (speaker_ids, session_id)

    try:
        target_speaker_id = transcription['speaker']
    except KeyError as e:
        target_speaker_id = 'unknown'
        notes.append('target_speaker_id is missing')

    start_sample = transcription['start_time']['original']
    end_sample = transcription['end_time']['original']

    example_id = get_example_id(
        start_sample=start_sample,
        end_sample=end_sample,
        session_id=session_id,
        speaker_id=target_speaker_id,
    )

    arrays = [f'U0{array+1}' for array in range(NUM_ARRAYS)]
    if session_id in ['S05', 'S22']:
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
        key for key1, mic_samples in num_samples.items()
        if key1 != 'original'
        for key, value in mic_samples.items()
        if (isinstance(value, int) and value == 0) or
           (isinstance(value, list) and value[0] == 0)
    ]
    for key in empty_keys:
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

    if EVAL_TRANSCRIPTIONS_MISSING and session_id == 'eval':
        assert transcription['words'] == "", transcription['words']
    else:
        words_dict = {keys.TRANSCRIPTION: transcription['words']}

    return example_id, {
        CH_K.SESSION_ID: session_id,
        CH_K.TARGET_SPEAKER: target_speaker_id,
        keys.SPEAKER_ID: speaker_ids,
        keys.GENDER: gender,
        keys.NUM_SAMPLES: num_samples,
        keys.AUDIO_PATH: audio_path_dict,
        CH_K.NOTES: notes,
        keys.START: start_time_dict,
        keys.END: end_time_dict,
        CH_K.REF_ARRAY: ref_array,
        CH_K.LOCATION: location,
        **words_dict,
    }


def time_to_string_format(time):
    # format to time string to fit time in kaldi example_ids
    return str(int(to_samples(time) * 100 / SAMPLE_RATE)).zfill(7)


def get_example_id(
        start_sample,
        end_sample,
        speaker_id,
        session_id,
):
    """
    >>> get_example_id(45963520, 45987360, 'P09', 'S03')
    'P09_S03_0287272-0287421'

    """
    start_sample_str = str(int(start_sample * 100 / SAMPLE_RATE)).zfill(7)
    end_sample_str = str(int(end_sample * 100 / SAMPLE_RATE)).zfill(7)
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
            array: [time[array]] * NUM_MICS
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
@click_common_options('chime5.json', chime_5)
@click.option(
    '--transcription-path',
    # '-j',
    default='/net/vol/boeddeker/deploy/git/hub/chime5-synchronisation/transcriptions_aligned',
    type=click.Path(),
    callback=click_convert_to_path
)
def main(database_path, json_path, transcription_path):

    json = create_database(database_path, transcription_path)

    print('Check that all wav files in the json exsist.')
    check_audio_files_exist(json, speedup='thread')
    print('Finished check.')
    dump_database_as_json(json_path, json)


if __name__ == '__main__':
    main()
