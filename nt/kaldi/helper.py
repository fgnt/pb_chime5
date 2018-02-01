from pathlib import Path
import subprocess
import os
from nt.io.data_dir import kaldi_root as KALDI_ROOT

WSJ_EG = '{}/egs/wsj/s5'.format(KALDI_ROOT)
INT2SYM = '{}/egs/wsj/s5/utils/int2sym.pl'.format(KALDI_ROOT)


def get_kaldi_env():
    env = os.environ.copy()
    env['PATH'] += ':{}/src/bin'.format(KALDI_ROOT)
    env['PATH'] += ':{}/tools/openfst/bin'.format(KALDI_ROOT)
    env['PATH'] += ':{}/src/fstbin/'.format(KALDI_ROOT)
    env['PATH'] += ':{}/src/gmmbin/'.format(KALDI_ROOT)
    env['PATH'] += ':{}/src/featbin/'.format(KALDI_ROOT)
    env['PATH'] += ':{}/src/lm/'.format(KALDI_ROOT)
    env['PATH'] += ':{}/src/sgmmbin/'.format(KALDI_ROOT)
    env['PATH'] += ':{}/src/sgmm2bin/'.format(KALDI_ROOT)
    env['PATH'] += ':{}/src/fgmmbin/'.format(KALDI_ROOT)
    env['PATH'] += ':{}/src/latbin/'.format(KALDI_ROOT)
    env['PATH'] += ':{}/src/nnetbin/'.format(KALDI_ROOT)
    env['PATH'] += ':{}/src/nnet2bin/'.format(KALDI_ROOT)
    env['PATH'] += ':{}/src/kwsbin/'.format(KALDI_ROOT)
    env['PATH'] += ':{}/src/online2bin/'.format(KALDI_ROOT)
    env['PATH'] += ':{}/src/ivectorbin/'.format(KALDI_ROOT)
    env['PATH'] += ':{}/src/lmbin/'.format(KALDI_ROOT)
    if 'LD_LIBRARY_PATH' in env.keys():
        env["LD_LIBRARY_PATH"] += ":{}/tools/openfst/lib".format(KALDI_ROOT)
    else:
        env["LD_LIBRARY_PATH"] = ":{}/tools/openfst/lib".format(KALDI_ROOT)
    env['LC_ALL'] = 'C'
    env['OMP_NUM_THREADS'] = '1'
    utils_path = '{}/egs/wsj/s5/utils'.format(KALDI_ROOT)
    steps_path = '{}/egs/wsj/s5/steps'.format(KALDI_ROOT)
    env['PATH'] += utils_path
    env['PATH'] += steps_path
    env['PATH'] += WSJ_EG
    return env


def excute_kaldi_commands(
        cmds, name='kaldi_cmd', env=None, log_dir=None, inputs=None,
        ignore_return_code=False
    ):
    p_list = list()
    std_out_list = list()
    std_err_list = list()
    return_codes_list = list()

    cmds = cmds if isinstance(cmds, (tuple, list)) else [cmds]
    if inputs is None:
        inputs = len(cmds) * [None]
    else:
        inputs = inputs if isinstance(inputs, (tuple, list)) else [inputs]

    for cmd in cmds:
        kaldi_env = get_kaldi_env()
        if env is not None:
            kaldi_env.update(env)
        if isinstance(cmd, list):
            p = subprocess.Popen(cmd, env=kaldi_env, universal_newlines=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 cwd=WSJ_EG)
        else:
            p = subprocess.Popen(cmd, shell=True, env=kaldi_env,
                                 universal_newlines=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 cwd=WSJ_EG)
        p_list.append(p)
    for idx, p in enumerate(p_list):
        std_out, std_err = p.communicate(inputs[idx])
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / '{}.{}.stdout'.format(name, idx), 'w') as fid:
                fid.write(std_out)
            with open(log_dir / '{}.{}.stderr'.format(name, idx), 'w') as fid:
                fid.write(std_err)
        returncode = p.returncode
        if returncode != 0 and not ignore_return_code:
            print('Error excuting {}. Output was:'.format(name))
            print('Stdout: {}'.format(std_out))
            print('Stderr: {}'.format(std_err))
            print('Command was {}'.format(cmds[idx]))
            raise ValueError('Kaldi error')
        return_codes_list.append(returncode)
        std_out_list.append(std_out)
        std_err_list.append(std_err)
    return std_out_list, std_err_list, return_codes_list
