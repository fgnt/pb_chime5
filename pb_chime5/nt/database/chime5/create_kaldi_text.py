import tempfile
from pathlib import Path

from IPython.lib.pretty import pprint

from pb_chime5.nt.io.data_dir import chime_5 as dir_chime_5
from pb_chime5.nt.utils.process_caller import run_process


def get_kaldi_transcriptions(
        json2text=dir_chime_5 / 'kaldi_tools' / 'json2text.py'
):
    if json2text is None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_process(
                'wget https://raw.githubusercontent.com/kaldi-asr/kaldi/master/egs/chime5/s5/local/json2text.py',
                stdout=None,
                stderr=None,
                cwd=tmp_dir,
            )
            return get_kaldi_transcriptions(Path(tmp_dir) / 'json2text.py')

    all_kaldi_transcriptions = {}

    for file in sorted(Path(
            '/net/fastdb/chime5/CHiME5/transcriptions'
    ).glob('*/*.json')):
        kaldi_transcriptions = run_process(
            [
                'python',
                json2text,
                '--mictype=worn',
                file,
            ],
            stderr=None,
        ).stdout

        for line in kaldi_transcriptions.strip().split('\n'):
            example_id, transcription = map(str.strip, (line + ' ').split(' ', maxsplit=1))

            example_id_split = example_id.split('_')
            speaker_id, session_id, remaining = example_id_split
            location, start, end = remaining.split('-')
            example_id = f'{speaker_id}_{session_id}_{start}-{end}'

            assert example_id not in all_kaldi_transcriptions, (example_id, all_kaldi_transcriptions)

            all_kaldi_transcriptions[example_id] = transcription

    return all_kaldi_transcriptions


if __name__ == '__main__':
    pprint(list(get_kaldi_transcriptions().items())[:5])
    pprint(list(get_kaldi_transcriptions(None).items())[:5])
