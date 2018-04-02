from nt.kaldi import helper
import re
import os
from nt.io import mkdir_p
from collections import defaultdict
import warnings
import pandas
import pickle
import logging
import glob
import numpy as np


def _build_rescale_lattice_cmd(decode_dir, hclg_dir, lmwt,
                               word_insertion_penalty=0):
    lattice_rescale_cmd = \
        f'''lattice-scale --inv-acoustic-scale={lmwt} ark:"cat {decode_dir}/lats/*.lat|" ark:- | \
        lattice-add-penalty --word-ins-penalty={word_insertion_penalty} ark:- ark:- | \
        lattice-best-path --word-symbol-table={hclg_dir}/words.txt \
        ark:- ark,t:{decode_dir}/scoring/{lmwt}.tra'''
    return lattice_rescale_cmd


def _build_compute_WER_command(decode_dir, hclg_dir, lmwt, strict=False):
    if strict:
        strict = '--mode=strict'
    else:
        strict = '--mode=present'
    cmd = \
        f'''cat {decode_dir}/scoring/{lmwt}.tra | sort -u -k1,1 | \
        {helper.INT2SYM} -f 2- {hclg_dir}/words.txt | sed 's:<UNK>::g' | \
        compute-wer --text {strict}\
        ark:{decode_dir}/scoring/test_filt.txt ark,p:- > {decode_dir}/wer_{lmwt}'''
    return cmd


def _parse_wer_lines(lines):
    wer_parser = re.compile(
        "\%WER\s([0-9]*\.?[0-9]*) \[ ([0-9]*) \/ ([0-9]*), ([0-9]*) ins, ([0-9]*) del, ([0-9]*) sub.*")
    if isinstance(lines, list):
        for line in lines:
            if line.startswith('%WER'):
                wer, errors, words, ins, del_, sub = wer_parser.match(
                    line).groups()
                return float(wer), int(errors), int(words), int(ins), int(
                    del_), int(sub)
    elif isinstance(lines, str):
        wer, errors, words, ins, del_, sub = wer_parser.match(
            lines).groups()
        return float(wer), int(errors), int(words), int(ins), int(del_), int(
            sub)
    return np.nan, 0, 0, 0, 0, 0


def parse_wer_file(wer_file):
    try:
        with open(wer_file) as fid:
            lines = fid.readlines()
            return _parse_wer_lines(lines)
    except Exception as e:
        warnings.warn('Exception during parsing of WER file: {}'.format(e))
    return np.nan, 0, 0, 0, 0, 0


def _tra_complete(tra_file, ref_file):
    if not os.path.exists(tra_file):
        return False
    with open(tra_file) as fid:
        tra = set([l.split()[0] for l in fid.readlines()])
    with open(ref_file) as fid:
        ref = set([l.split()[0] for l in fid.readlines()])
    diff = ref - tra
    if not len(diff):
        return True
    else:
        logging.getLogger('_tra_complete').warn(
            f'{tra_file} is missing {len(diff)} utts. '
            f'Samples: {list(diff)[:min(len(diff), 5)]}'
        )
    return False


def _lattices_exists(ref_file, lat_dir):
    with open(ref_file) as fid:
        ref = set([l.split()[0] for l in fid.readlines()])
    lat_files = set(
        l.split('/')[-1].replace('.lat', '')
        for l in glob.glob(f'{lat_dir}/*.lat')
    )
    diff = ref - lat_files
    if not len(diff):
        return True
    else:
        logging.getLogger('_lattices_exists').warn(
            f'{lat_dir} is missing {len(diff)} utts.\n'
            f'Samples: {list(diff)[:min(len(diff), 5)]}\n'
            f'Files: {list(lat_files)[:min(len(lat_files), 5)]}'
        )
    return False


def compute_scores(decode_dir, hclg_dir, ref_text, min_lmwt=8, max_lmwt=18,
                   force_scoring=False, build_tra=True, strict=True,
                   ignore_return_codes=True):
    LOG = logging.getLogger('computer_scores')
    decode_dir = os.path.abspath(decode_dir)
    mkdir_p(os.path.join(decode_dir, 'scoring'))
    ref_file = f'{decode_dir}/scoring/test_filt.txt'
    cmd = (f"cat {ref_text} | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' "
           f"> {ref_file}")
    helper.excute_kaldi_commands(
        [cmd], 'copying reference transcription', log_dir=decode_dir + '/logs',
        ignore_return_code=ignore_return_codes
    )
    cmds = list()
    tra_written = dict()
    for lmwt in range(min_lmwt, max_lmwt + 1):
        tra_file = f'{decode_dir}/scoring/{lmwt}.tra'
        rescale = not os.path.exists(tra_file)
        rescale &= not _tra_complete(tra_file, ref_file)
        rescale &= _lattices_exists(ref_file, f'{decode_dir}/lats')
        rescale &= build_tra
        rescale |= force_scoring
        if rescale:
            LOG.info(f'Rescaling lattice for lmwt {lmwt}')
            cmds.append(
                _build_rescale_lattice_cmd(decode_dir, hclg_dir, lmwt))
            tra_written[lmwt] = True
    if len(cmds):
        helper.excute_kaldi_commands(
            cmds, 'rescaling lattice', log_dir=decode_dir + '/logs',
            ignore_return_code=ignore_return_codes
        )
    else:
        LOG.info('All utts already rescaled - skipping')
    cmds = list()
    for lmwt in range(min_lmwt, max_lmwt + 1):
        if lmwt in tra_written:
            LOG.info(f'Computing WER for lmwt {lmwt}')
            cmds.append(
                _build_compute_WER_command(
                    decode_dir, hclg_dir, lmwt, strict=strict))
    if len(cmds):
        helper.excute_kaldi_commands(
            cmds, 'computing WER', log_dir=decode_dir + '/logs',
            ignore_return_code=ignore_return_codes
        )
    result = defaultdict(list)
    for lmwt in range(min_lmwt, max_lmwt + 1):
        wer, errors, words, ins, del_, sub = parse_wer_file(
            decode_dir + '/wer_{}'.format(lmwt)
        )
        result['wer'].append(float(wer))
        result['errors'].append(int(errors))
        result['words'].append(int(words))
        result['ins'].append(int(ins))
        result['del'].append(int(del_))
        result['sub'].append(int(sub))
        result['decode_dir'].append(decode_dir)
        result['lmwt'].append(int(lmwt))
    res = pandas.DataFrame(result)
    with open(decode_dir + '/result.pkl', 'wb') as fid:
        pickle.dump(res, fid)
    return result.copy()
