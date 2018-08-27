from pathlib import Path
from nt.utils.mapping import Dispatcher


def word2id(words_txt):
    with open(words_txt) as fid:
        return {
            line.strip().split(' ')[0]: int(line.split(' ')[1])
            for line in fid if len(line.split(' ')) == 2
        }


def id2word(words_txt):
    with open(words_txt) as fid:
        return {
            int(line.strip().split(' ')[1]): line.split(' ')[0]
            for line in fid if len(line.split(' ')) == 2
        }


def read_keyed_text_file(text_file: Path, to_list=True):
    """
    Often used to read e.g. Kaldi `text`, `wav.scp` or `spk2utt`.

    Args:
        text_file: Path with file in format: <utterance_id> <else>
        to_list: If true, all items after the first will be split to list.

    Returns:

    """
    text_file = Path(text_file)
    with text_file.open() as f:
        if to_list:
            return {
                line.strip().split()[0]:
                    line.strip().split()[1:] for line in f
            }
        else:
            return {
                line.strip().split()[0]: line.split(' ', maxsplit=1)[1].strip()
                for line in f.readlines()
            }


def write_keyed_text_file(text_file: Path, data_dict):
    """
    Often used to write e.g. Kaldi `text`, `wav.scp` or `spk2utt`.
    Sorting is enforced here to avoid subsequent calls to fix_data_dir.sh

    For some file names, it tries to perform some kind of sanity check to match
    the Kaldi file standards.

    Args:
        text_file: Path with file in format: <utterance_id> <else>

    Returns:

    """
    text_file = Path(text_file)
    data = []
    for k, text in sorted(data_dict.items()):
        if isinstance(text, list):
            text = ' '.join(text)
        if text_file.name == 'utt2dur':
            try:
                text_number = float(text)
            except Exception:
                raise ValueError(
                    f'The text "{text}" for {k} that should be written to '
                    f'{text_file} does not represent a number.'
                )
            else:
                assert 0. < text_number < 1000., f'Strange duration: {k}: {text_number} s'
        elif text_file.name == 'spk2gender':
            text = Dispatcher(male='m', female='f', m='m', f='f',)[text]
        else:
            pass

        data.append(f'{k} {text}')

    text_file.write_text('\n'.join(data))
