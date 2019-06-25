import logging
import os
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import tqdm
from pb_chime5.database.helper import click_convert_to_path
from pb_chime5.io.json_module import load_json, dump_json
from itertools import combinations
FORMAT_STRING = '%H:%M:%S.%f'
JSON_PATH = Path('/net/vol/jenkins/jsons/chime5_speech_activity')
NUM_MICS = 4
SAMPLE_RATE = 16000
NUM_ARRAYS = 6
TIME_ZERO = datetime.strptime('0:00:00.00', FORMAT_STRING)


def create_cross_talk_database(database_path, json_path):
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    for dataset in ['train', 'dev']:
        get_cross_talk(database_path, dataset, json_path)


def get_cross_talk(database_path, dataset, json_path):
    database_path = database_path / 'CHiME5'
    dataset_transciption_path = database_path / 'transcriptions' / dataset
    json_dict = dict()
    for session_path in dataset_transciption_path.glob('*.json'):
        trans = load_json(session_path)
        session_id = trans[0]['session_id']
        json_dict[session_id] = dict()
        total = len(trans)
        speaker_ids = [key for key in trans[0]['start_time'].keys() if
                       'P' in key]
        out_dict = {
                        speaker: {speaker: dict(start=[], end=[]) for speaker in
                                  speaker_ids} for speaker in speaker_ids}
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(os.cpu_count()) as ex:
            for speaker, example_dict in tqdm.tqdm(
                    ex.map(
                        get_dict_speaker,
                        trans
                    ),
                    total=total,
                    desc=dataset + '_' + session_id
            ):
                if example_dict is not None:
                    out_dict[speaker] = combine_dicts(example_dict,
                                                      out_dict[speaker])
        out_dict['cross_talk'] = get_cross_talk_per_mic(out_dict)

        dump_json(out_dict,
                  str(json_path / session_id) + '.json')


def get_dict_speaker(example):
    speaker_ids = [key for key in example['start_time'].keys()
                   if 'P' in key]
    try:
        speaker_id = example['speaker']
    except KeyError:
        return None, None
    time_dict = get_time_from_dict(example['start_time'], example['end_time'],
                                   speaker_ids)
    return speaker_id, time_dict


def get_time_from_dict(start, end, speaker_ids):
    time_dict = {
        speaker: dict(start=to_samples(start[speaker]),
                     end=to_samples(end[speaker])) for speaker in speaker_ids}
    return time_dict

from decimal import Decimal


def to_samples(time: str):
    """
    >>> def from_samples(samples):
    ...     hours = samples // (60 * 60 * 16000)
    ...     samples = samples - hours * 60 * 60 * 16000
    ...     minutes = samples // (60 * 16000)
    ...     samples = samples - minutes * 60 * 16000
    ...     seconds = samples / 16000
    ...     return f'{hours}:{minutes}:{seconds}'
    >>> to_samples('0:47:52.708375')
    45963334
    >>> from_samples(45963334)
    '0:47:52.708375'
    >>> to_samples('0:47:52.7083750')
    45963334
    >>> to_samples('0:47:54.1956875')
    45987131
    >>> from_samples(45987131)
    '0:47:54.1956875'
    >>> to_samples('1000:47:54.1956875')
    57645987131
    >>> from_samples(57645987131)
    '1000:47:54.1956875'
    >>> from_samples(57645987130)
    '1000:47:54.195625'
    >>> to_samples('1000:47:54.195625')
    57645987130
    >>> to_samples('0:01:04.62')
    1033920
    >>> from_samples(1033920)
    '0:1:4.62'
    """
    # datetime.strptime can only handle 6 digits after the comma, but 16000 Hz
    # requires a resolution of 7 digits after the comma

    hours, minutes, seconds = [t for t in time.split(':')]

    hours = int(hours)
    minutes = int(minutes)
    seconds = Decimal(seconds)

    seconds_samples = seconds * SAMPLE_RATE

    assert seconds_samples == int(seconds_samples), (seconds_samples, seconds, time)

    samples = (
        hours * 3600 * SAMPLE_RATE
        + minutes * 60 * SAMPLE_RATE
        + seconds_samples
    ) 

    # samples = (
    #                   datetime.strptime(time.rstrip('0'), FORMAT_STRING) - TIME_ZERO
    #           ).total_seconds() * SAMPLE_RATE
    assert samples == int(samples), (samples, time)
    return int(samples)


def combine_dicts(speaker_dict, org_dict):
    def combine_iterativ(in_dict, out_dict):
        if isinstance(in_dict, dict):
            out_dict.update({key: combine_iterativ(value, out_dict[key])
                             for key, value in in_dict.items()})
            return out_dict
        elif isinstance(in_dict, int):
            out_dict.append(in_dict)
            return out_dict
        else:
            raise ValueError('expected dict or int')

    return combine_iterativ(speaker_dict, org_dict)


def get_cross_talk_per_mic(speaker_dict):
    cross_talk = {speaker: dict(start=list(), end=list())
                  for speaker in speaker_dict.keys()}
    speaker_combinations = [list(map(str, comb))
                            for comb in combinations(speaker_dict.keys(), 2)]
    speaker_combinations += [comb[::-1] for comb in speaker_combinations]
    for active_speaker_id, second_speaker_id in speaker_combinations:
        active_speaker_mics = speaker_dict[active_speaker_id]
        second_speaker_mics = speaker_dict[second_speaker_id]
        for idx in range(len(active_speaker_mics[active_speaker_id]['start'])):
            start_sp1 = active_speaker_mics[active_speaker_id]['start'][idx]
            end_sp1 = active_speaker_mics[active_speaker_id]['end'][idx]
            second_speaker = second_speaker_mics[active_speaker_id]
            for idy in range(len(second_speaker['start'])):
                start_sp2 = second_speaker['start'][idy]
                end_sp2 = second_speaker['end'][idy]
                if start_sp1 <= end_sp2 and end_sp1 >= start_sp2:
                    for speaker_mic in speaker_dict.keys():
                        if start_sp1 > start_sp2:
                            cross_talk[speaker_mic]['start'].append(
                                active_speaker_mics[speaker_mic]['start'][idx])
                        else:
                            cross_talk[speaker_mic]['start'].append(
                                second_speaker_mics[speaker_mic]['start'][idy])

                        if end_sp1 < end_sp2:
                            cross_talk[speaker_mic]['end'].append(
                                active_speaker_mics[speaker_mic]['end'][idx])
                        else:
                            cross_talk[speaker_mic]['end'].append(
                                second_speaker_mics[speaker_mic]['end'][idy])
    return cross_talk


def get_active_speaker(start_sample, end_sample, session_id, mic_id,
                       json_path=None, speaker_json=None, sample_step=1,
                       dtype=bool):
    if json_path is not None:
        speaker_json = load_json(str(json_path / session_id) + '.json')
    elif speaker_json is None:
        raise ValueError('Either json_path or speaker_json have to be defined')
    out_dict = dict()
    for key, value in speaker_json['cross_talk'].items():
        cross_talk = to_numpy(value, start_sample, end_sample,
                              sample_step=sample_step, dtype=dtype)
        activity = to_numpy(speaker_json[key][mic_id], start_sample, end_sample,
                           sample_step, dtype)
        out_dict[key] = dict(cross_talk=cross_talk, activity=activity)
    return out_dict


def to_numpy(in_dict, start_sample, end_sample, sample_step=1, dtype=bool):
    num_samples = end_sample - start_sample
    array = np.zeros(int(num_samples / sample_step), dtype=dtype)
    for idx, start in enumerate(in_dict['start']):
        end = in_dict['end'][idx]
        if start > end_sample:
            break
        if end < start_sample:
            continue
        array[max(start - start_sample, 0) // sample_step:((end - start_sample)
                                                           // sample_step)] = 1
    return array


@click.command()
@click.option(
    '--database-path', type=click.Path(),
    callback=click_convert_to_path
)
@click.option(
    '--json-path', type=click.Path(),
    callback=click_convert_to_path
)
def main(database_path, json_path):
    create_cross_talk_database(database_path, json_path)


if __name__ == '__main__':
    main()
