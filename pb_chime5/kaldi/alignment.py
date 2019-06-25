from pathlib import Path
import numpy as np
from tempfile import NamedTemporaryFile
from pb_chime5.kaldi.helper import get_kaldi_env, excute_kaldi_commands
from pb_chime5.kaldi import helper as kaldi_helper
import logging
import sys

LOG = logging.getLogger('Kaldi')


def _import_alignment(ark, model_file, extract_cmd, extract_cmd_finish,
                      is_zipped=True, import_options=None):
    """ Read alignment data file.

        Can read either phones or pdfs depending on the copy_cmd.

        :param ark: The ark file to read
        :param model_file: Model file used to create the alignments. This is needed
            to extract the pdf ids
        :param extract_cmd: Command to extract the alignment. Can be either
            ali-to-pdf or ali-to-phones
        :param extract_cmd_finish: Success output of the extraction command
            (i.e. Done or Converted)
        :param copy_feats: The location of the kaldi tool `copy-feats`
        :return: A dictionary with the file ids as keys and their data as values
        """
    data = dict()
    if is_zipped:
        src_param = 'ark:gunzip -c {ark} |'.format(ark=ark)
    else:
        src_param = 'ark:{ark}'.format(ark=ark)
    dest_param = 'ark,t:-'
    if import_options is None:
        import_options = []
    from pb_chime5.utils.process_caller import run_process

    completed_process = run_process([extract_cmd, *import_options, model_file, src_param, dest_param],
                                    environment=get_kaldi_env())
    out = completed_process.stdout
    err = completed_process.stderr
    pos = err.find(extract_cmd_finish) + 1 + len(extract_cmd_finish)
    matrix_number = int(err[pos:].split()[0])
    for line in out.split('\n'):
        split = line.split()
        if len(split) > 0:
            utt_id = split[0]
            ali = np.asarray(split[1:], dtype=np.int32)
            data[utt_id] = ali

    assert len(data) == matrix_number, \
        '{cmd} converted {num_matrix} alignments, ' \
        'but we read {num_data}'. \
        format(cmd=extract_cmd,
               num_matrix=matrix_number, num_data=len(data))
    return data


def import_pdf_alignment_from_file(ark, model_file, is_zipped=True):
    """Import alignments as pdf ids

    Since the binary form is not documented and may change in future release,
    a kaldi tool (ali-to-pdf) is used to first create a ark file in text mode.

    :param ark: The ark file to read
    :param model_file: Model file used to create the alignments. This is needed
        to extract the pdf ids
    :return: A dictionary with the file ids as keys and their data as values
    """
    _cmd = 'ali-to-pdf'
    return _import_alignment(
        str(ark), str(model_file), _cmd, 'Converted', is_zipped
    )


def import_phone_alignment_from_file(
        ark, model_file, is_zipped=True, per_frame=False):
    """Import alignments as phone ids

    Since the binary form is not documented and may change in future release,
    a kaldi tool (ali-to-pdf) is used to first create a ark file in text mode.

    :param ark: The ark file to read
    :param model_file: Model file used to create the alignments. This is needed
        to extract the pdf ids
    :return: A dictionary with the file ids as keys and their data as values
    """
    _cmd = 'ali-to-phones'
    import_options = []
    if per_frame:
        import_options += ['--per-frame']
    return _import_alignment(
        str(ark), str(model_file), _cmd, 'Done', is_zipped, import_options)


def import_alignment_data(
    ali_dir, model_name='final.alimdl',
    import_fn=import_pdf_alignment_from_file,
    **import_kwargs
):
    """ Import alignment data from a directory.

    Kaldi splits the alignments across several files. This function reads the
    data from all file starting with "ali" in a directory and combines them
    into one dictionary.

    Args:
        ali_dir: Directory containing the alignment files
        model_name: Name of the model to use for importing. Must be within
            the alignment directory
        import_fn: Import function to use for import the alignments. Can be
            `import_pdf_alignment_from_file` or
            `import_phone_alignment_from_file`

    Returns: dict containing the alignments with the utterance ids as keys.

    """
    ali_dir = Path(ali_dir).resolve()
    data_dict = dict()
    for file in ali_dir.glob('*'):
        if file.name.startswith('ali'):
            zipped = file.name.endswith('.gz')
            ali_file = ali_dir / file
            model_file = ali_dir / model_name
            imported = import_fn(
                ali_file, model_file, is_zipped=zipped, **import_kwargs)
            data_dict.update(imported)
    return data_dict


def import_occs(occs_file):
    """ Reads data from an oocs file
    """
    try:
        with open(occs_file) as fid:
            occs = fid.readline().strip()
    except UnicodeDecodeError:
        with NamedTemporaryFile() as tmpfile:
            excute_kaldi_commands(
                [f'copy-vector --binary=false {occs_file} {tmpfile.name}'],
                'convert occs'
            )
            with open(tmpfile.name) as fid:
                occs = fid.readline().strip()
    occs = occs.replace('[', '').replace(']', '').split()
    occs = [occ.split('.')[0] for occ in occs]
    return np.array(occs, dtype=np.int32)


def write_occs(occs, occs_file):
    """ Writes data to an oocs file

    """
    with open(occs_file, 'w') as fid:
        fid.write('[')
        fid.write(' '.join(map(str, occs)))
        fid.write(']')


def compile_train_graphs(
        tree_file: Path,
        model_file: Path,
        lexicon_fst_file: Path,
        integer_transcription_file: Path,
        output_graphs_file: Path
):
    """
    Initial step to prepare for forced alignment.

    Args:
        tree_file: E.g. `s5/exp/tri4b/tree`
        model_file: E.g. `s5/exp/tri4b/final.mdl`
        lexicon_fst_file: E.g. `lang_path / 'L.fst'`
        integer_transcription_file: E.g. `train.tra`
        output_graphs_file: E.g. `graphs.fsts`

    Returns:

    """
    command = (
        f"compile-train-graphs "
        f"{tree_file.resolve().absolute()} "
        f"{model_file.resolve().absolute()} "
        f"{lexicon_fst_file.resolve().absolute()} "
        f"ark:{integer_transcription_file.resolve().absolute()} "
        f"ark:{output_graphs_file.resolve().absolute()}"
    )

    # Why does this execute in `.../egs/wsj/s5`?
    env = kaldi_helper.get_kaldi_env()
    _, std_err_list, _ = kaldi_helper.excute_kaldi_commands(
        command,
        name=sys._getframe().f_code.co_name,
        env=env
    )

    for line in std_err_list[0].split('\n'):
        LOG.info(line)


def forced_alignment(
        log_posteriors_ark_file: Path,
        graphs_file: Path,
        model_file: Path,
        alignment_dir: Path,
        beam: int=200,
        retry_beam: int=400,
        part=1
):
    """

    Args:
        log_posteriors_ark_file: E.g. `log_posteriors.ark`
        graphs_file: E.g. `graphs.fsts`
        model_file: E.g. `s5/exp/tri4b/final.mdl`
        alignment_dir:
        beam: Kaldi recipes (e.g. WSJ) typically use 10.
        retry_beam: Kaldi recipes (e.g. WSJ) typically use 40.
        part: Could be used for parallel processing.

    Returns:

    """
    if not part == 1:
        raise NotImplementedError(
            "I believe that the `log_posteriors_ark_file` and the "
            "`graphs_file` already needs to be chunked to support parallelism."
        )

    command = (
        'align-compiled-mapped '
        f'--beam={beam} '
        f'--retry-beam={retry_beam} '
        f'{model_file} '
        f'ark:{graphs_file} '
        f'ark:{log_posteriors_ark_file} '
        f'ark,t:|gzip -c > {alignment_dir}/ali.{part}.gz'
    )

    # Why does this execute in `.../egs/wsj/s5`?
    env = kaldi_helper.get_kaldi_env()
    _, std_err_list, _ = kaldi_helper.excute_kaldi_commands(
        command,
        name=sys._getframe().f_code.co_name,
        env=env
    )

    for line in std_err_list[0].split('\n'):
        LOG.info(line)
