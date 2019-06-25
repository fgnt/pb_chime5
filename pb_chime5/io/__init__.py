"""
This module deals with all sorts of input and output.

There is special focus on audio, but there are also some convenience imports
i.e. for load_json() and similar functions.

The file path is called `path` just as it has been done in ``audioread``.
The path should either be a ``pathlib.Path`` object or a string.
"""

from pb_chime5.io import audioread
from pb_chime5.io.json_module import (
    load_json,
    loads_json,
    dump_json,
    dumps_json,
)
from pb_chime5.io.json_module import SummaryEncoder
from pb_chime5.io.audioread import load_audio as _load_audio
from pb_chime5.io.audiowrite import dump_audio
from pb_chime5.io.file_handling import (
    mkdir_p,
    symlink,
)
from pb_chime5.io.load_decorator import recursive_load_decorator
load_audio = recursive_load_decorator(default_list_to='array')(_load_audio)

__all__ = [
    "load_audio",
    "dump_audio",
    "load_json",
    "loads_json",
    "dump_json",
    "dumps_json",
    "mkdir_p",
    "symlink",
    "SummaryEncoder",
]

