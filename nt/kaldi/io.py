import ast
import re

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


def _parse_kaldi_best_wer_text(text):
    """
    >>> from IPython.lib.pretty import pprint
    >>> text = '%WER 72.78 [ 42853 / 58881, 2518 ins, 22449 del, 17886 sub ] /net/vol/boeddeker/chime5/pc2/arrayBSS/ali_sweep/39/kaldi/baseline/exp/chain_train_worn_u100k/tdnn1a_sp/decode_bss_beam/wer_8_0.5'
    >>> pprint(_parse_kaldi_best_wer_text(text))
    {'wer': 72.78,
     'word_errors': 42853,
     'words': 58881,
     'ins': 2518,
     'del': 22449,
     'sub': 17886,
     'flags': '',
     'file': '/net/vol/boeddeker/chime5/pc2/arrayBSS/ali_sweep/39/kaldi/baseline/exp/chain_train_worn_u100k/tdnn1a_sp/decode_bss_beam/wer_8_0.5'}

    """
    content = text.strip()

    r = re.compile(r'%WER (?P<wer>\d*\.\d*) \[ ('
                   r'?P<word_errors>\d*) '
                   r'/ (?P<words>\d*), (?P<ins>\d*) ins, '
                   r'(?P<del>\d*) del, (?P<sub>\d*) sub ]'
                   r' ?(?P<flags>[^\n]*) (?P<file>[^\n]+)')

    match = r.search(content).groupdict()
    for k in match.keys():
        try:
            # https://stackoverflow.com/a/9510585
            match[k] = ast.literal_eval(match[k])
        except (SyntaxError, ValueError):
            pass

    # if not ignore_missing and match['missing']:
    #     raise RuntimeError(match)

    return match



def parse_kaldi_wer_file(path, allow_best_wer=False, ignore_missing=False):
    """
    # example
    ```
    compute-wer --text --mode=present ark:exp/tri2a_mc/decode_bg_5k_REVERB_dt_SimData_dt_for_1ch_far_room2_A/scoring/test_filt.txt ark,p:-
    %WER 45.11 [ 1830 / 4057, 36 ins, 638 del, 1156 sub ]
    %SER 97.57 [ 241 / 247 ]
    Scored 247 sentences, 0 not present in hyp.

    compute-wer --text --mode=present ark:...
    %WER 16.80 [ 653 / 3886, 82 ins, 98 del, 473 sub ] [PARTIAL]
    %SER 78.57 [ 187 / 238 ]
    Scored 238 sentences, 10 not present in hyp.
    ```

    >>> from IPython.lib.pretty import pprint
    >>> f = '/net/vol/boeddeker/sacred/90/kaldi/exp/tri2a_mc/decode_bg_5k_REVERB_dt_new_SimData_dt_for_1ch_near_room1_A/wer_15'
    >>> pprint(parse_kaldi_wer_file(f))
    {'wer': 15.83,
     'word_errors': 644,
     'words': 4068,
     'ins': 80,
     'del': 106,
     'sub': 458,
     'flags': '',
     'ser': 76.61,
     'sentence_errors': 190,
     'sentences': 248,
     'missing': 0}
    >>> f = '/net/vol/boeddeker/sacred/93/kaldi/exp/tri2a_mc/decode_bg_5k_REVERB_dt_new_SimData_dt_for_1ch_near_room1_A/wer_15'
    >>> pprint(parse_kaldi_wer_file(f, ignore_missing=True))
    {'wer': 16.8,
     'word_errors': 653,
     'words': 3886,
     'ins': 82,
     'del': 98,
     'sub': 473,
     'flags': '[PARTIAL]',
     'ser': 78.57,
     'sentence_errors': 187,
     'sentences': 238,
     'missing': 10}
    >>> f = '/net/vol/boeddeker/sacred/plain_wpe/34/kaldi/exp/tri2a_mc/decode_bg_5k_REVERB_dt_new_SimData_dt_for_1ch_near_room1_A/wer_15'
    >>> pprint(parse_kaldi_wer_file(f, ignore_missing=True))
    {'wer': 19.81,
     'word_errors': 797,
     'words': 4024,
     'ins': 108,
     'del': 130,
     'sub': 559,
     'flags': '[PARTIAL]',
     'ser': 82.86,
     'sentence_errors': 203,
     'sentences': 245,
     'missing': 3}
    >>> pprint(parse_kaldi_wer_file(f))
    Traceback (most recent call last):
    ...
    RuntimeError: {'wer': 19.81, 'word_errors': 797, 'words': 4024, 'ins': 108, 'del': 130, 'sub': 559, 'flags': '[PARTIAL]', 'ser': 82.86, 'sentence_errors': 203, 'sentences': 245, 'missing': 3}
    >>> f = '/net/vol/boeddeker/chime5/pc2/arrayBSS/ali_sweep/39/kaldi/baseline/exp/chain_train_worn_u100k/tdnn1a_sp/decode_bss_beam/scoring_kaldi/best_wer'
    >>> pprint(parse_kaldi_wer_file(f))
    {'wer': 72.78,
     'word_errors': 42853,
     'words': 58881,
     'ins': 2518,
     'del': 22449,
     'sub': 17886,
     'flags': '',
     'ser': 84.73,
     'sentence_errors': 6301,
     'sentences': 7437,
     'missing': 0}
    >>> pprint(parse_kaldi_wer_file(f, allow_best_wer=True))

    """

    with Path(path).open() as f:
        content = f.read()

    if len(content.strip().split('\n')) == 1:
        # assume best wer file
        if allow_best_wer:
            return _parse_kaldi_best_wer_text(content)
        content = Path(content.split(']', maxsplit=1)[-1].strip()).read_text()

    r = re.compile(r'%WER (?P<wer>\d*\.\d*) \[ ('
                   r'?P<word_errors>\d*) '
                   r'/ (?P<words>\d*), (?P<ins>\d*) ins, '
                   r'(?P<del>\d*) del, (?P<sub>\d*) sub ]'
                   r' ?(?P<flags>[^\n]*)\n%SER '
                   r'(?P<ser>\d*\.\d*) \[ (?P<sentence_errors>\d*) '
                   r'/ (?P<sentences>\d*) ]\n'
                   r'Scored \d+ sentences, (?P<missing>\d*) not '
                   r'present in hyp.')

    match = r.search(content).groupdict()
    for k in match.keys():
        try:
            # https://stackoverflow.com/a/9510585
            match[k] = ast.literal_eval(match[k])
        except (SyntaxError, ValueError):
            pass

    if not ignore_missing and match['missing']:
        raise RuntimeError(match)

    return match
