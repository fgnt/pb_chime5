import logging
import os
from datetime import datetime
from functools import partial
from warnings import warn

import click
import tqdm
from pathlib import Path
from nt.database import keys
from nt.database.helper import dump_database_as_json, click_common_options
from nt.database.chime5.get_speaker_activity import to_samples
from nt.io.data_dir import chime_5
from nt.io.json_module import load_json

class CHiME5_Keys:
    WORN = 'worn_microphone'
    TARGET_SPEAKER = 'target_speaker'
    NOTES = 'notes'
    SESSION_ID = 'session_id'
    LOCATION = 'location'
    REF_ARRAY = 'reference_array'


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


def create_database(database_path):
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    datasets = dict()
    for dataset in set_length.keys():
        out_dict = get_dataset(database_path, dataset)
        datasets[dataset] = out_dict
    return {keys.DATASETS: datasets}


def get_dataset(database_path, dataset):
    database_path = database_path / 'CHiME5'
    dataset_transciption_path = database_path / 'transcriptions' / dataset
    dataset_audio_path = database_path / 'audio' / dataset
    json_dict = dict()
    for session_path in dataset_transciption_path.glob('*.json'):
        session_id = session_path.name.split('.')[0]
        trans = load_json(session_path)
        total = len(trans)
        if total < set_length[dataset][session_id]:
            raise ValueError(
                f'missing utterances in session {session_id} expected length'
                f' {set_length[dataset][session_id]} available length {total}')
        elif total > set_length[dataset][session_id]:
            warn(f'there are {total - set_length[dataset][session_id]} examples'
                  f' more than expected in session {session_id}')
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(os.cpu_count()) as ex:
            for example_id, example in tqdm.tqdm(
                    ex.map(
                        partial(get_example, audio_path=dataset_audio_path),
                        trans
                    ),
                    total=total,
                    desc=dataset
            ):
                json_dict[example_id] = example
    return json_dict


def get_example(transcription, audio_path):
    session_id = transcription['session_id']
    notes = list()
    speaker_ids = [key for key in transcription['start_time'].keys()
                   if 'P' in key]
    try:
        target_speaker_id = transcription['speaker']
    except KeyError as e:
        warn(f'{e}; keys allowed are: {transcription.keys()} for '
             f'session_id: {transcription["session_id"]}')
        target_speaker_id = [
            ids for ids in speaker_ids
            if transcription['start_time'][ids] == transcription['start_time'][
                'original']
            if transcription['end_time'][ids] == transcription['end_time'][
                   'original']
        ]
        if len(target_speaker_id) == 1:
            target_speaker_id = target_speaker_id[0]
            notes.append('target_speaker_id is missing and was inferred')
        else:
            target_speaker_id = 'original'
            notes.append('target_speaker_id is missing')
    # Format time to fit kaldi example ids
    example_time = '-'.join([
        time_to_string_format(transcription['start_time'][target_speaker_id]),
        time_to_string_format(transcription['end_time'][target_speaker_id])
    ])
    example_id = '_'.join(
        [target_speaker_id, session_id, example_time]
    )
    arrays = [f'U0{array+1}' for array in range(NUM_ARRAYS)]
    if session_id in ['S05', 'S22']:
        del arrays[2]
        notes.append('Array U03 is missing, this is expected')
    elif session_id == 'S09':
        del arrays[4]
        notes.append('Array U05 is missing, this is expected')
    audio_path_dict = get_audio_path_dict(arrays, speaker_ids,
                                          session_id, audio_path)
    start_time_dict = get_time_from_dict(transcription['start_time'],
                                         speaker_ids, arrays)
    end_time_dict = get_time_from_dict(transcription['end_time'],
                                       speaker_ids, arrays)
    num_samples = get_num_samples(start_time_dict, end_time_dict)
    gender = {id: 'male' for id in speaker_ids}
    gender.update({id: 'female' for id in speaker_ids if id in FEMALE_SPEAKER})
    if 'location' in transcription:
        location = transcription['location']
        ref_array = transcription['ref']
    else:
        location = 'unkown'
        ref_array = 'unkown'
    if session_id in NOTES_DICT:
        notes.append(NOTES_DICT[session_id])
    return example_id, {CH_K.SESSION_ID: session_id,
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
                        keys.TRANSCRIPTION: transcription['words']}


def time_to_string_format(time):
    # format to time string to fit time in kaldi example_ids
    return str(int(to_samples(time) * 100 / SAMPLE_RATE)).zfill(7)

def get_audio_path_dict(arrays, speaker_ids, session_id, audio_path):
    audio_path_dict = {keys.OBSERVATION: {array: [
        str(audio_path / '_'.join([session_id, array])) + f'.CH{mic+1}.wav'
        for mic in range(NUM_MICS)] for array in arrays}}
    audio_path_dict.update({CH_K.WORN: {speaker: [
        str(audio_path / '_'.join([session_id, speaker])) + '.wav'] for speaker in speaker_ids}})
    for key, audio_path in audio_path_dict.items():
        assert [Path(p).is_file() for path in audio_path.values()
                for p in path], f'For {key} at least one audio_path is no file'
    return audio_path_dict

def get_num_samples(start_time_dict, end_time_dict):
    def get_num_samples_recursive(start_time, end_time):
        if isinstance(start_time, dict):
            return {key: get_num_samples_recursive(value, end_time[key])
                    for key, value in start_time.items()}
        elif isinstance(start_time, list):
            return [end_time[idx]- start_time[idx]
                    for idx in range(len(start_time))]
        elif isinstance(start_time, int):
            return end_time - start_time
        else:
            raise ValueError('expected dict or list')

    return get_num_samples_recursive(start_time_dict, end_time_dict)


def get_duration(start_time, end_time):
    duration = datetime.strptime(end_time, FORMAT_STRING) - datetime.strptime(
        start_time, FORMAT_STRING)
    return int(duration.total_seconds() * SAMPLE_RATE)


def get_time_from_dict(time, speaker_ids, arrays):
    time_dict = {keys.OBSERVATION: {array: [to_samples(
        time[array])for mic in range(NUM_MICS)] for array in arrays}}
    time_dict.update({CH_K.WORN: {speaker: to_samples(
        time[speaker]) for speaker in speaker_ids}})
    return time_dict


@click.command()
@click_common_options('chime.json', chime_5)
def main(database_path, json_path):
    json = create_database(database_path)
    dump_database_as_json(json_path, json)


if __name__ == '__main__':
    main()
