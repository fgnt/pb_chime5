from nt.io.json_module import load_json, dump_json
import click
import tqdm
import logging
import os
import numpy as np
from nt.io.audioread import audio_length
from nt.database.helper import click_convert_to_path
from datetime import datetime
from pathlib import Path
FORMAT_STRING = '%H:%M:%S.%f'
JSON_PATH = Path('/net/vol/jensheit/chime5/train')
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
    dataset_audio_path = database_path / 'audio' / dataset
    json_dict = dict()
    for session_path in dataset_transciption_path.glob('*.json'):
        trans = load_json(session_path)
        session_id = trans[0]['session_id']
        json_dict[session_id] = dict()
        num_samples = audio_length(str(dataset_audio_path / session_id) +'_U01.CH1.wav')
        total = len(trans)
        speaker_ids = [key for key in trans[0]['start_time'].keys() if
                       'P' in key]
        arrays = [f'U0{array+1}' for array in range(NUM_ARRAYS)]
        out_dict = {array: {speaker: dict(start=[], end=[]) for speaker in speaker_ids} for array in arrays}
        out_dict.update({speaker: {speaker: dict(start=[], end=[]) for speaker in speaker_ids} for speaker in speaker_ids})

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(os.cpu_count()) as ex:
            for example_dict in tqdm.tqdm(
                    ex.map(
                        get_dict_speaker,
                        trans
                    ),
                    total=total,
                    desc=dataset+'_'+session_id
            ):
                if not example_dict is None:
                    out_dict = combine_dicts(example_dict, out_dict)
        in_tuple = list(out_dict.items())
        with ThreadPoolExecutor(os.cpu_count()) as ex:
            for mic_id, cross_talk in tqdm.tqdm(
                ex.map(get_cross_talk_per_mic,
                       in_tuple),
                total=len(in_tuple),
                desc=dataset+'_'+session_id
            ):
                out_dict[mic_id]['cross_talk'] = cross_talk
        dump_json(out_dict,
                  str(json_path / session_id) + 'time.json')




def get_dict_speaker(example):
    session_id = example['session_id']
    speaker_ids = [key for key in example['start_time'].keys()
                   if 'P' in key]
    try:
        speaker_id = example['speaker']
    except KeyError:
        return None

    arrays = [f'U0{array+1}' for array in range(NUM_ARRAYS)]
    if session_id in ['S05', 'S22']:
        del arrays[2]
    elif session_id == 'S09':
        del arrays[4]
    time_dict = get_time_from_dict(example['start_time'], example['end_time'],
                                   speaker_ids, arrays, speaker_id)

    return time_dict


def get_time_from_dict(start, end, speaker_ids, arrays, target):
    time_dict = {array:
        {target: dict(start=to_samples(start[array]),end=to_samples(end[array]))} for array in arrays}
    time_dict.update({speaker: {target: dict(start=to_samples(start[speaker]), end=to_samples(end[speaker]))}
                      for speaker in speaker_ids})
    return time_dict


def to_samples(time):
    return int((datetime.strptime(time,
                                  FORMAT_STRING) - TIME_ZERO).total_seconds() * SAMPLE_RATE)


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
    mic, mic_times = input_tuple
    speaker_list = list()
    cross_talk = dict(end=list(), start=list())
    for speaker, speaker_times in mic_times.items():
        speaker_list.append(speaker)
        start_list = sorted(speaker_times['start'])
        end_list = sorted(speaker_times['end'])
        for idx in range(len(start_list)):
            start = start_list[idx]
            end = end_list[idx]
            for speaker_2, speaker_times_2 in mic_times.items():
                if speaker_2 in speaker_list:
                    continue
                start_list_2 = sorted(speaker_times_2['start'])
                end_list_2 = sorted(speaker_times_2['end'])
                for idx2 in range(len(start_list_2)):
                    start_2 = start_list_2[idx2]
                    end_2 = end_list_2[idx2]
                    if start_2 >= end:
                        break
                    if end_2 <= start:
                        continue
                    if end < end_2:
                        cross_talk['end'].append(end)
                    else:
                        cross_talk['end'].append(end_2)
                    if start > start_2:
                        cross_talk['start'].append(start)
                    else:
                        cross_talk['start'].append(start_2)

    return mic, cross_talk


def get_active_speaker(start_sample, end_sample, session_id,
                       json_path=JSON_PATH):
    speaker_json = load_json(str(json_path / session_id) + '.json')
    out_dict = dict()
    for key, value in speaker_json.items():
        cross_talk = to_numpy(value['cross_talk'], start_sample, end_sample)
        speech_activity = np.array([to_numpy(activity, start_sample, end_sample)
                                    for _, activity in sorted(value.items())])
        out_dict[key] = dict(cross_talk=cross_talk, activity=speech_activity)
    return out_dict


def to_numpy(in_dict, start_sample, end_sample):
    num_samples = end_sample - start_sample
    array = np.array([False] * num_samples)
    for idx, start in enumerate(in_dict['start']):
        end = in_dict['end'][idx]
        if start > end_sample:
            break
        if end < start_sample:
            continue
        array[max(start - start_sample, 0): end - end_sample] = True
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
