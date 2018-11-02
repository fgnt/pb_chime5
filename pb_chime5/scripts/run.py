import sys
from pathlib import Path
import inspect
import itertools

import sacred
from sacred.commands import print_config
from sacred.observers import FileStorageObserver

from pb_chime5.core import get_enhancer
from pb_chime5 import mapping
from pb_chime5.util import mpi

experiment = sacred.Experiment('Chime5 Array Enhancement')


@experiment.config
def config():
    locals().update({k: v.default for k, v in inspect.signature(get_enhancer).parameters.items()})

    session_id = 'dev'


@experiment.capture
def get_dir(
        _run,
):
    assert len(_run.observers) == 1, len(_run.observers)
    _dir = Path(_run.observers[0].basedir) / str(_run._id)
    return _dir


@experiment.capture
def get_session_ids(session_id):

    if isinstance(session_id, str):
        session_id = [session_id]

    dataset_to_session = mapping.Dispatcher({
        dataset: [session for session, _ in list(session_dataset)]
        for dataset, session_dataset in itertools.groupby(
            sorted(
                mapping.session_to_dataset.items(),
                key=lambda x: (x[1], x[0])
            ),
            key=lambda x: x[1]
        )
    })
    dataset_to_session['all'] = [
        session
        for v in dataset_to_session.values()
        for session in v
    ]

    return list(sorted({
        sess
        for key in session_id
        for sess in dataset_to_session.get(key, [key])
    }))


get_enhancer = experiment.capture(get_enhancer)


@experiment.main
def main(_run):
    run(_run)


@experiment.command
def test_run(_run, test_run=True):
    assert test_run is not False, test_run
    run(_run, test_run=test_run)


def run(_run, test_run=False):
    if mpi.IS_MASTER:
        print_config(_run)
        _dir = get_dir()
        print('Experiment dir:', _dir)
    else:
        _dir = None

    _dir = mpi.bcast(_dir, mpi.MASTER)

    enhancer = get_enhancer()

    session_ids = get_session_ids()
    if mpi.IS_MASTER:
        print('Enhancer:', enhancer)
        print(session_ids)

    enhancer.enhance_session(
        session_ids,
        _dir / 'audio',
        test_run=test_run,
    )
    if mpi.IS_MASTER:
        print('Finished experiment dir:', _dir)


if __name__ == '__main__':

    # Custom parsing of sacred --file_storage option.
    # This allows to give this option a default.
    argv = [*sys.argv]
    import argparse
    from pb_chime5 import git_root

    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--file_storage',
                        default=git_root / 'sacred',
                        help='add a file storage observer')

    parsed, args = parser.parse_known_args()
    argv = argv[:1] + args

    if mpi.IS_MASTER:
        path = Path(parsed.file_storage).expanduser().resolve()
        experiment.observers.append(FileStorageObserver.create(str(path)))

    experiment.run_commandline(argv)
