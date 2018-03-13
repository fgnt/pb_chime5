from nt.database import HybridASRJSONDatabaseTemplate
from nt.database import HybridASRKaldiDatabaseTemplate
from nt.database import keys as K
from nt.io.data_dir import kaldi_root, database_jsons, chime_5
import numpy as np
from pathlib import Path
from nt.database import keys
from collections import defaultdict


class Chime5(HybridASRJSONDatabaseTemplate):
    def __init__(self):
        path = database_jsons / 'chime5.json'
        super().__init__(path)

    @property
    def datasets_train(self):
        return ['train']

    @property
    def datasets_eval(self):
        return ['dev']

    @property
    def datasets_test(self):
        return ['test']
