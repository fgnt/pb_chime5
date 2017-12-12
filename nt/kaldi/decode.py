from nt.kaldi import helper
import re
import os
from nt.io import mkdir_p
from collections import defaultdict
import warnings
import pandas
import pickle


def _build_rescale_lattice_cmd(decode_dir, hclg_dir, lmwt,
                               word_insertion_penalty=0):
    lattice_rescale_cmd = \
        '''lattice-scale --inv-acoustic-scale={LMWT} ark:"cat {decode_dir}/lats/*.lat|" ark:- | \
        lattice-add-penalty --word-ins-penalty={word_ins_penalty} ark:- ark:- | \
        lattice-best-path --word-symbol-table={hclg_dir}/words.txt \
        ark:- ark,t:{decode_dir}/scoring/{LMWT}.tra || exit 1;'''.format(
            LMWT=lmwt, decode_dir=decode_dir, hclg_dir=hclg_dir,
            word_ins_penalty=word_insertion_penalty
        )
    return lattice_rescale_cmd


def _build_compute_WER_command(decode_dir, hclg_dir, lmwt, strict=False):
    if strict:
        strict = '--mode=strict'
    else:
        strict = '--mode=present'
    cmd = \
        '''cat {decode_dir}/scoring/{LMWT}.tra | sort -u -k1,1 | \
        {int2sym} -f 2- {hclg_dir}/words.txt | sed 's:<UNK>::g' | \
        compute-wer --text {strict}\
        ark:{decode_dir}/scoring/test_filt.txt ark,p:- > {decode_dir}/wer_{LMWT}'''.format(
            int2sym=helper.INT2SYM, LMWT=lmwt, decode_dir=decode_dir,
            hclg_dir=hclg_dir, strict=strict
        )
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
    return 100, 0, 0, 0, 0, 0


def parse_wer_file(wer_file):
    try:
        with open(wer_file) as fid:
            lines = fid.readlines()
            return _parse_wer_lines(lines)
    except Exception as e:
        warnings.warn('Exception during parsing of WER file: {}'.format(e))
    return 100, 0, 0, 0, 0, 0


def compute_scores(decode_dir, hclg_dir, ref_text, min_lmwt=8, max_lmwt=18,
                   force_scoring=False, build_tra=True, strict=True):
    decode_dir = os.path.abspath(decode_dir)
    mkdir_p(os.path.join(decode_dir, 'scoring'))
    cmd = "cat {data} | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' " \
          "> {decode_dir}/scoring/test_filt.txt".format(
        data=ref_text, decode_dir=decode_dir)
    helper.excute_kaldi_commands(
        [cmd], 'copying reference transcription', log_dir=decode_dir + '/logs'
    )
    cmds = list()
    for lmwt in range(min_lmwt, max_lmwt + 1):
        if (not os.path.exists('{decode_dir}/scoring/{LMWT}.tra'.format(
                decode_dir=decode_dir, LMWT=lmwt
        )) or force_scoring) and build_tra:
            cmds.append(
                _build_rescale_lattice_cmd(decode_dir, hclg_dir, lmwt))
    helper.excute_kaldi_commands(
        cmds, 'rescaling lattice', log_dir=decode_dir + '/logs'
    )
    cmds = list()
    for lmwt in range(min_lmwt, max_lmwt + 1):
        if not os.path.exists('{decode_dir}/wer_{LMWT}'.format(
                decode_dir=decode_dir, LMWT=lmwt
        )) or force_scoring:
            cmds.append(
                _build_compute_WER_command(
                    decode_dir, hclg_dir, lmwt, strict=strict))
    helper.excute_kaldi_commands(
        cmds, 'computing WER', log_dir=decode_dir + '/logs'
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
    return dict(**result)
