from pathlib import Path
# from nt.io.data_dir import kaldi_root
from pb_chime5.kaldi import helper as kaldi_helper
import logging

LOG = logging.getLogger('Kaldi')


def write_transcription_file(
    out_of_vocabulary_mapping_file: Path,
    word_mapping_file: Path,
    word_transcription_file: Path,
    mapped_transcription_file: Path,
):
    """
    Code tested with WSJ database and derived databases.

    Args:
        out_of_vocabulary_mapping_file: Contains an integer to which all OOVs
            are going to be mapped.
            Typically: `db.lang_path / 'oov.int'`
        word_mapping_file:
            It has this form:
                <eps> 0
                !EXCLAMATION-POINT 1
                !SIL 2
        word_transcription_file: If you want to align own data, you need
            to write this file, first.
            Typically: `db.lang_path / 'words.txt'`
            It has this form:
                011c0201 THE SALE OF THE HOTELS ...
        mapped_transcription_file: Output file.
            Typically: `*.tra`
            It has this form:
                011c0201 110920 96431 79225 110920 52031 ...
    Returns:

    """
    sym2int_pl_file = (
        kaldi_root / "egs" / "wsj" / "s5" / "utils" / "sym2int.pl"
    )

    for file in (
        sym2int_pl_file,
        out_of_vocabulary_mapping_file,
        word_mapping_file,
        word_transcription_file
    ):
        assert file.is_file(), file
    assert mapped_transcription_file.parent.is_dir(), mapped_transcription_file

    with out_of_vocabulary_mapping_file.open() as f:
        oov = f.read().strip()

    command = (
        f"{sym2int_pl_file.resolve().absolute()} "
        f"--map-oov {oov} "
        f"-f 2- "  # Will map from second item onwards (skipping utt id).
        f"{word_mapping_file.resolve().absolute()} "
        f"{word_transcription_file.resolve().absolute()} "
        f"> {mapped_transcription_file.resolve().absolute()}"
    )

    # Why does this execute in `.../egs/wsj/s5`?
    env = kaldi_helper.get_kaldi_env()
    _, std_err_list, _ = kaldi_helper.excute_kaldi_commands(command, env=env)

    for line in std_err_list[0].split('\n'):
        LOG.info(line)
