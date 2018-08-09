from pathlib import Path


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
                line.strip().split()[0]:
                    line.strip().split(maxsplit=1)[1] for line in f
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
    with text_file.open('w') as f:
        for k, text in sorted(data_dict.items()):
            if isinstance(text, list):
                text = ' '.join(text)
            if text_file.name == 'utt2dur':
                text = float(text)
                assert 0. < text < 1000., f'Strange duration: {k}: {text} s'
                f.write(f'{k} {text:.2f}\n')
            elif text_file.name == 'spk2gender':
                text = dict(male='m', female='f', m='m', f='f',)[text]
                f.write(f'{k} {text}\n')
            else:
                f.write(f'{k} {text}\n')
