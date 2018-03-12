import click
import logging
import os
import tqdm
from functools import partial
from warnings import warn
from nt.io.json_module import load_json, dump_json
from nt.database import keys
from nt.io.data_dir import chime_5
from nt.database.helper import dump_database_as_json, click_common_options
from nt.io.audioread import audio_length

class CHiME5_Keys:
    WORN='worn_microphone'
    TARGET_SPEAKER='target_speaker'
    NOTES = 'notes'

CH_K = CHiME5_Keys

NUM_ARRAYS = 6
NUM_MICS = 4
NOTES_DICT = dict(SO3='P11 dropped from min ~15 to ~30',
                  S12='Last 15 minutes of U05 missing',
                  S19='P52 mic unreliable',
                  S23='Neighbour interrupts',
                  S24='P54 mic unreliable, P53 disconnects for bathroom',
                  S01='No registration tone')

set_length=dict(train=dict(S03=4090, S04=5563, S05=4939, S06=5097, S07=3656,
                           S17=5892, S08=6175, S16=5004, S12=3300, S13=4193,
                           S19=4292,  S20=5365, S18=4907, S22=4758, S23=7054,
                           S24=5695),
                dev=dict(S02=3822, S09=3618),
                eval=dict(S01=5797, S21=5231))

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
    json_dict= dict()
    error_list = list()
    for session_path in dataset_transciption_path.glob('*.json'):
        trans = load_json(session_path)
        total = len(trans)
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
        sesson_id = transcription['session_id']
        notes = list()
        try:
            active_speaker_id = transcription['speaker']
        except KeyError as e:
            warn(f'{e}; keys allowed are: {transcription.keys()} for '
                 f'session_id: {transcription["session_id"]}')
            transcription['error'] = e
            active_speaker_id = 'original'
            notes.append('target_speaker_id is missing')
        example_id = '_'.join(
            [sesson_id, active_speaker_id,
             transcription['start_time'][active_speaker_id],
             transcription['end_time'][active_speaker_id]]
        )
        speaker_ids = [key for key in transcription['start_time'].keys()
                       if 'P' in key]
        arrays = [f'U0{array+1}' for array in range(NUM_ARRAYS)]
        if sesson_id in ['S05', 'S22']:
            del arrays[2]
            notes.append('Array U03 is missing, this is expected')
        elif sesson_id=='S09':
            del arrays[4]
            notes.append('Array U05 is missing, this is expected')
        audio_path_dict = {keys.OBSERVATION: {array:[
            str(audio_path / '_'.join([sesson_id, array])) + f'.CH{mic+1}.wav'
            for mic  in range(NUM_MICS)] for array in arrays}}
        audio_path_dict.update({CH_K.WORN: {
            speaker: str(audio_path / '_'.join([sesson_id, speaker])) + '.wav'
            for speaker in speaker_ids}})
        start_time_dict = get_time_from_dict(transcription['start_time'],
                                             speaker_ids, arrays)
        end_time_dict = get_time_from_dict(transcription['end_time'],
                                           speaker_ids, arrays)
        num_samples = get_num_samples(audio_path_dict)
        if sesson_id in NOTES_DICT:
            notes.append(NOTES_DICT[sesson_id])
        return example_id, {keys.EXAMPLE_ID: example_id,
                            CH_K.TARGET_SPEAKER: active_speaker_id,
                            keys.SPEAKER_ID: speaker_ids,
                            keys.NUM_SAMPLES: num_samples,
                            keys.AUDIO_PATH: audio_path_dict,
                            CH_K.NOTES: notes,
                            keys.START: start_time_dict,
                            keys.END: end_time_dict,
                            keys.TRANSCRIPTION: transcription['words']}


def get_num_samples(audio_dict):
    def get_num_samples_recursive(fn_input):
        if isinstance(fn_input, dict):
            return {key: get_num_samples_recursive(value)
                    for key, value in fn_input.items()}
        elif isinstance(fn_input, list):
            return [audio_length(path) for path in fn_input]
        elif isinstance(fn_input, str):
            return audio_length(fn_input)
        else:
            raise ValueError('expected dict, list or str')
    return get_num_samples_recursive(audio_dict)

def get_time_from_dict(time, speaker_ids, arrays):
    time_dict = {keys.OBSERVATION: {array: [
        time[array] for mic in range(NUM_MICS)] for array in arrays}}
    time_dict.update({CH_K.WORN: {speaker: time[speaker]
                                  for speaker in speaker_ids}})
    return time_dict


@click.command()
@click_common_options('chime.json', chime_5)
def main(database_path, json_path):
    json = create_database(database_path)
    dump_database_as_json(json_path, json)


if __name__ == '__main__':
    main()