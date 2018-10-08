from nt.io import load_audio as _load_audio
from nt.io.recusive import recursive_load_decorator as _recursive_load_decorator

load_audio = _recursive_load_decorator(default_list_to='array')(_load_audio)
