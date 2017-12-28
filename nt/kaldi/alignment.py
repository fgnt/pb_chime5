from pathlib import Path
import subprocess
import numpy as np
from nt.kaldi.helper import get_kaldi_env


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
    copy_process = subprocess.Popen(
        [extract_cmd, *import_options, model_file, src_param, dest_param],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=get_kaldi_env())
    out, err = copy_process.communicate()
    try:
        if copy_process.returncode != 0:
            raise ValueError("Returncode of{} was != 0. Stderr "
                             "output is:\n{}".format(extract_cmd, err))
        out = out.decode('utf-8')
        err = err.decode('utf-8')
        pos = err.find(extract_cmd_finish) + 1 + len(extract_cmd_finish)
        matrix_number = int(err[pos:].split()[0])
        for line in out.split('\n'):
            split = line.split()
            if len(split) > 0:
                utt_id = split[0]
                ali = np.asarray(split[1:], dtype=np.int32)
                data[utt_id] = ali
    except Exception as e:
        print('Exception during reading the alignments: {}'.format(e))
        print('Stderr: {}'.format(err))
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
    with open(occs_file) as fid:
        occs = fid.readline().strip().replace('[', '').replace(']', '').split()
    return np.array(occs, dtype=np.int32)
