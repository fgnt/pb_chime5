import logging
import os
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import tqdm
from nt.database.helper import click_convert_to_path
from nt.io.json_module import load_json, dump_json

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
        in_tuple = list(out_dict.items())
        with ThreadPoolExecutor(os.cpu_count()) as ex:
            for mic_id, cross_talk in tqdm.tqdm(
                    ex.map(get_cross_talk_per_mic,
                           in_tuple),
                    total=len(in_tuple),
                    desc=dataset + '_' + session_id
            ):
                out_dict[mic_id]['cross_talk'] = cross_talk
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


def to_samples(time):
    return int((
                   datetime.strptime(time, FORMAT_STRING) - TIME_ZERO
                ).total_seconds() * SAMPLE_RATE)


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


def get_cross_talk_per_mic(input_tuple):
    # input is item of time_dict
    speaker, mic_times = input_tuple
    cross_talk = dict(end=list(), start=list())
    speaker_times = mic_times[speaker]
    start_list = sorted(speaker_times['start'])
    end_list = sorted(speaker_times['end'])
    for idx in range(len(start_list)):
        start = start_list[idx]
        end = end_list[idx]
        for speaker_2, speaker_times_2 in mic_times.items():
            start_list_2 = sorted(speaker_times_2['start'])
            end_list_2 = sorted(speaker_times_2['end'])
            for idx2 in range(len(start_list_2)):
                start_2 = start_list_2[idx2]
                end_2 = end_list_2[idx2]
                if start_2 >= end:
                    break
                if end_2 <= start or (start==start_2 and end==end_2):
                    continue
                if end < end_2:
                    cross_talk['end'].append(end)
                else:
                    cross_talk['end'].append(end_2)
                if start > start_2:
                    cross_talk['start'].append(start)
                else:
                    cross_talk['start'].append(start_2)

    return speaker, cross_talk


def get_active_speaker(start_sample, end_sample, session_id,
                       speaker_json=None, sample_step=1, dtype=bool):
    # speaker_json = load_json(str(json_path / session_id) + '.json')
    out_dict = dict()
    for key, value in speaker_json.items():
        cross_talk = to_numpy(value['cross_talk'], start_sample, end_sample,
                              sample_step=sample_step, dtype=dtype)
        speech_activity = np.array([to_numpy(activity, start_sample,
                                             end_sample,
                                             sample_step=sample_step,
                                             dtype=dtype)
                                    for _, activity in sorted(value.items())])
        out_dict[key] = dict(cross_talk=cross_talk, activity=speech_activity)
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
        array[int(max(start - start_sample, 0) / sample_step):int(
            (end - start_sample) / sample_step)] = 1
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
