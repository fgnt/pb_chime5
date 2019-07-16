# pb_chime5: Front-End Processing for the CHiME-5 Dinner Party Scenario [\[pdf\]](http://spandh.dcs.shef.ac.uk/chime_workshop/papers/CHiME_2018_paper_boeddecker.pdf)

This repository includes all components of the CHiME-5 front-end presented by Paderborn University on the CHiME-5 workshop [PB2018CHiME5].
Using the baseline backend provided by the challenge organizers on the data enhanced with this multi-array front-end using the default parameters which differ slightly from the original paper a WER of 60.89 % was achieved on the development set.
In combination with an acoustic model presented by the RWTH Aachen this multi-array front-end achieved the third best results during the challenge with 54.56 % on the development and 55.30 % on the evaluation set.
A later cooperation with Hitachi led to WER of 39.94 % on the development and 41.64 % on the evaluation set, using the multi-array front-end presented in this repository.

The front-end consists out of WPE, a spacial mixture model that uses time annotations (GSS), beamforming and masking:

![(System Overview)](doc/images/system.svg)

The core code is located in the file `pb_chime5/core.py`.
An example script to run the enhancement is in `pb_chime5/scripts/run.py` and can be executed with `python -m pb_chime5.scripts.run with session_id=dev wpe=True wpe_tabs=2`.

Challenge website: http://spandh.dcs.shef.ac.uk/chime_challenge/

Workshop website: http://spandh.dcs.shef.ac.uk/chime_workshop/

If you are using this code please cite the following paper:

```
@Article{PB2018CHiME5,
  author    = {Boeddeker, Christoph and Heitkaemper, Jens and Schmalenstroeer, Joerg and Drude, Lukas and Heymann, Jahn and Haeb-Umbach, Reinhold},
  title     = {{Front-End Processing for the CHiME-5 Dinner Party Scenario}},
  year      = {2018},
  booktitle = {CHiME5 Workshop},
}
```

ToDos:

- [x] core enhancement code
- [x] remove dependencies from our infrastructure
- [x] launch script
- [x] manual script
- [x] code cleanup, remove all unnessesary code and rearrange the code files

## Installation

Does not work with Windows.

Clone the repo with submodules
```bash
$ git clone https://github.com/fgnt/pb_chime5.git
$ cd pb_chime5
$ # Download submodule dependencies  # https://stackoverflow.com/a/3796947/5766934
$ git submodule init  
$ git submodule update
```
Use the environmental variable CHIME5_DIR to direct the repository to your chime5 data:
```bash
$ export CHIME5_DIR=/path/to/chime5/data/CHiME5
```

Install this package and pb_bss 
```bash
$ pip install --user -e pb_bss/
$ pip install --user -e .
```

Create the database description file
```bash
$ make cache/chime5.json
```

It is assumed that the folder `sacred` in this git is the simulation folder.
If you want to change the simulation dir, add a symlink to the folder where you want to store the simulation results: `ln -s /path/to/sim/dir sacred`

Start a testrun with
```bash
$ python -m pb_chime5.scripts.run test_run with session_id=dev
```

Start a simulation with 9 mpi workers (1 scheduler and 8 actual worker)
```bash
$ mpiexec -np 9 python -m pb_chime5.scripts.run with session_id=dev
```
You can replace `mpiexec -np 9` with your HPC command to start a MPI program.
It scalles up very well and is tested with 600 distributed cores.

# FAQ

#### Q: I ran `mpiexec -np 9 python -m pb_chime5.scripts.run with session_id=dev wpe=True wpe_tabs=2` and it generated 9 folders and the estimated duration is around 100 h. Is this right?
A: It is likely that your mpi4py installation does not work. Execute the following command and check if the output is correct:
```bash
$ mpiexec -np 3 python -c 'from mpi4py import MPI; print("My worker rank:", MPI.COMM_WORLD.rank, "Total workers:", MPI.COMM_WORLD.size)'
My worker rank: 2 Total workers: 3
My worker rank: 0 Total workers: 3
My worker rank: 1 Total workers: 3
```

#### Q: I want to use my own source activity detector. Can you give me a hint where to start?
At the end of `pb_chime5/activity_alignment.py` is some code how to generate finetuned time annotations from kaldi worn alignments.
You have to change the `worn_ali_path` to worn alignments from kaldi and it will generate files (`cache/word_non_sil_alignment/S??.pkl`) for finetuned oracle time annotations.
Using them for enhancement you have to change the `activity_type` to `path` and `activity_path` to the path of the finetuned time annotations
e.g. `python -m pb_chime5.scripts.run with activity_type=path activity_path=cache/word_non_sil_alignment`.




