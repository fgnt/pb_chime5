"""
[mpirun -np $(nproc --all)] python -m pb_chime5.scripts.kaldi_run_rttm_libri_css with storage_dir=<...> database_rttm=<...> [activity_rttm=<...>] [session_id=dev] [session_to_audio_paths=<...>.{yaml,json}] job_id=1 number_of_jobs=1

 - [mpirun -np $(nproc --all)]: Use all loacl cores
 - storage_dir: Where to store the files:
        <storage_dir>/audio/<fileID>/<spk>_<startSample>-<endSample>
 - database_rttm: RTTM files that contains the utterance/sentence start and end times.
 - activity_rttm: RTTM files that contains the word boundaries (default database_rttm)
 - session_to_audio_paths: yaml or json file
        Contains the mapping from session/file ID to to the actual files.
 - job_id=1: Kaldi style parallel option
 - number_of_jobs=1: Kaldi style parallel option
"""

from pathlib import Path
import inspect

import sacred
from sacred.commands import print_config
from sacred.observers import FileStorageObserver

import dlp_mpi

from pb_chime5.core_chime6_rttm import get_enhancer

experiment = sacred.Experiment('Chime5 Array Enhancement')


@experiment.config
def config():
    locals().update({k: v.default for k, v in inspect.signature(get_enhancer).parameters.items()})

    session_id = None  # Default: All sessions from database_rttm
    storage_dir: str = None

    database_rttm: str = None
    activity_rttm: str = database_rttm

    # session_to_audio_paths: A file with a mapping from session ID (or file ID) used
    #                   in rttm to the actual audio file.
    # e.g. session_to_audio_paths = {
    #          session_1: multichannel.wav,
    #          session_1: [ch1.wav, ch2.wav, ...],
    # }
    # Supported are json and yaml.
    session_to_audio_paths = None

    job_id = 1
    number_of_jobs = 1

    assert storage_dir is not None, (storage_dir, 'overwrite the storage_dir from the command line')
    assert database_rttm is not None, (database_rttm, 'overwrite the database_rttm from the command line')
    assert activity_rttm is not None, (database_rttm, 'overwrite the activity_rttm from the command line')

    if dlp_mpi.IS_MASTER:
        experiment.observers.append(FileStorageObserver.create(str(
            Path(storage_dir).expanduser().resolve() / 'sacred'
        )))

@experiment.named_config
def my_test_rttm():
    database_rttm = '/scratch/hpc-prf-nt1/cbj/net/vol/boeddeker/chime6/kaldi/egs/chime6/s5_track2_download/data/dev_beamformit_dereverb_stats_seg/rttm.U06'


# get_enhancer = experiment.capture(get_enhancer)

@experiment.capture
def get_sessions(database_rttm):
    database_rttm = Path(database_rttm)
    database_rttm_text = database_rttm.read_text()
    sessions: set = {line.split()[1] for line in database_rttm_text.splitlines()}
    return sessions

@experiment.capture
def get_enhancer(
    database_rttm,
    activity_rttm,

    # session_to_audio_paths: A file with a mapping from session ID (or file ID) used
    #                   in rttm to the actual audio file.
    # e.g. session_to_audio_paths = {
    #          session_1: multichannel.wav,
    #          session_1: [ch1.wav, ch2.wav, ...],
    # }
    # Supported are json and yaml.
    session_to_audio_paths=None,

    # chime6_dir='/net/fastdb/chime6/CHiME6',
    # multiarray='outer_array_mics',
    context_samples=240000,

    wpe=True,
    wpe_tabs=10,
    wpe_delay=2,
    wpe_iterations=3,
    wpe_psd_context=0,

    activity_garbage_class=True,

    stft_size=1024,
    stft_shift=256,
    stft_fading=True,

    bss_iterations=20,
    bss_iterations_post=1,

    bf_drop_context=True,

    bf='mvdrSouden_ban',
    postfilter=None,
):
    from pb_chime5.core_chime6_rttm import (
        Enhancer,
        get_database,
        WPE,
        Activity,
        GSS,
        Beamformer,
        RTTMDatabase,
    )
    if session_to_audio_paths is None:
        from paderbox.array import intervall as array_intervall

        sessions: set = get_sessions(database_rttm)
        assert len(sessions) == 1, sessions

        files = list(Path(database_rttm).parent.glob('*.wav'))
        if len(files) == 1:
            # Assume multichannel file
            files, = files

        session_to_audio_paths = {
            sessions.pop(): files
        }
    elif isinstance(session_to_audio_paths, str):
        import paderbox as pb
        # Load file, load detects yaml, json, ... (pkl is not allowed -> unsafe)
        session_to_audio_paths = pb.io.load(session_to_audio_paths)
    elif isinstance(session_to_audio_paths, dict):
        session_to_audio_paths = session_to_audio_paths
    else:
        raise NotImplementedError(type(session_to_audio_paths), session_to_audio_paths)

    assert wpe is True or wpe is False, wpe

    class MyRTTMDatabase(RTTMDatabase):
        @staticmethod
        def example_id(file_id, speaker_id, start, end):
            # Don't use the strange CHiME-6 pattern for the example ID.
            max_digits = len(str(16000 * 60 * 60 * 10))  # 10h
            start = str(start).zfill(max_digits)
            end = str(end).zfill(max_digits)

            # return f'{file_id}_{speaker_id}-{start}_{end}'
            return f'{speaker_id}_{start}-{end}'

    db = MyRTTMDatabase(
        database_rttm,
        session_to_audio_paths,
        # rttm, audio_paths, alias=alias
    )

    return Enhancer(
        db=db,
        context_samples=context_samples,
        wpe_block=WPE(
            taps=wpe_tabs,
            delay=wpe_delay,
            iterations=wpe_iterations,
            psd_context=wpe_psd_context,
        ) if wpe else None,
        activity=Activity(
            garbage_class=activity_garbage_class,
            rttm=activity_rttm,
        ),
        gss_block=GSS(
            iterations=bss_iterations,
            iterations_post=bss_iterations_post,
            verbose=False,
        ),
        bf_drop_context=bf_drop_context,
        bf_block=Beamformer(
            type=bf,
            postfilter=postfilter,
        ),
        stft_size=stft_size,
        stft_shift=stft_shift,
        stft_fading=stft_fading,
    )


@experiment.main
def main(_run, storage_dir):
    run(_run, storage_dir=storage_dir)


@experiment.command
def test_run(_run, storage_dir, test_run=True):
    assert test_run is not False, test_run
    run(_run, storage_dir=storage_dir, test_run=test_run)


@experiment.capture
def run(_run, storage_dir, job_id, number_of_jobs, session_id, test_run=False):
    if dlp_mpi.IS_MASTER:
        print_config(_run)

    assert job_id >= 1 and job_id <= number_of_jobs, (job_id, number_of_jobs)

    enhancer = get_enhancer()

    if test_run:
        print('Database', enhancer.db)

    if test_run is False:
        dataset_slice = slice(job_id - 1, None, number_of_jobs)
    else:
        dataset_slice = test_run

    if dlp_mpi.IS_MASTER:
        print('Enhancer:', enhancer)
        print(session_id)

    if session_id is None:
        session_ids = sorted(get_sessions())
    elif isinstance(session_id, str):
        session_ids = [session_id]
    elif isinstance(session_id, (tuple, list)):
        session_ids = session_id
    else:
        raise TypeError(type(session_id), session_id)

    for session_id in session_ids:
        enhancer.enhance_session(
            session_id,
            Path(storage_dir) / 'audio',
            dataset_slice=dataset_slice,
            audio_dir_exist_ok=True,
            is_chime=False,
        )

    if dlp_mpi.IS_MASTER:
        print('Finished experiment dir:', storage_dir)


if __name__ == '__main__':
    experiment.run_commandline()
