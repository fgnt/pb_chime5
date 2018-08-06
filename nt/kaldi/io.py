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


def write_keyed_text_file(text_file: Path, data_dict, from_list=True):
    """
    Often used to write e.g. Kaldi `text`, `wav.scp` or `spk2utt`.
    Sorting is enforced here to avoid subsequent calls to fix_data_dir.sh

    Args:
        text_file: Path with file in format: <utterance_id> <else>
        from_list: Asserts each value is a list. Merges values with space.

    Returns:

    """
    # ToDo: CB: Why "from_list" and not automatic conversion?

    text_file = Path(text_file)
    with text_file.open('w') as f:
        for k, v in sorted(data_dict.items()):
            if from_list:
                assert isinstance(v, list)
                text = ' '.join(v)
            else:
                text = v
            f.write(f'{k} {text}\n')
