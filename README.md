# pb_chime5: Front-End Processing for the CHiME-5 Dinner Party Scenario [\[pdf\]](http://spandh.dcs.shef.ac.uk/chime_workshop/papers/CHiME_2018_paper_boeddecker.pdf)

This repository includes all components of the CHiME-5 front-end presented by Paderborn University on the CHiME-5 workshop [PB2018CHiME5].
Using the baseline backend provided by the challenge organizers on the data enhanced with this multi-array front-end using the default parameters which differ slightly from the original paper a WER of 60.89 % was achieved on the development set.
In combination with an acoustic model presented by the RWTH Aachen [Kitza2018] this multi-array front-end achieved the third best results during the challenge with 54.56 % on the development and 55.30 % on the evaluation set.

A later cooperation with Hitachi [Kanda2019] led to WER of 39.94 % on the development and 41.64 % on the evaluation set, using the multi-array front-end presented in this repository.

The best single system WERs with this enhancement are 41.6 % on the development and 43.2 % on the evaluation set reported in [Zorila2019].

The front-end consists out of WPE, a spacial mixture model that uses time annotations (GSS), beamforming and masking:

![(System Overview)](doc/images/system.svg)

The core code is located in the file `pb_chime5/core.py`.
An example script to run the enhancement is in `pb_chime5/scripts/run.py` and can be executed with `python -m pb_chime5.scripts.run with session_id=dev wpe=True wpe_tabs=2`.

Challenge website: http://spandh.dcs.shef.ac.uk/chime_challenge/

Workshop website: http://spandh.dcs.shef.ac.uk/chime_workshop/

If you are using this code please cite the following paper ([pdf](https://groups.uni-paderborn.de/nt/pubs/2018/INTERSPEECH_2018_Heitkaemper_Paper.pdf), [poster](https://groups.uni-paderborn.de/nt/pubs/2018/INTERSPEECH_2018_Heitkaemper_Poster.pdf)):
```
@inproceedings{PB2018CHiME5,
  author    = {Boeddeker, Christoph and Heitkaemper, Jens and Schmalenstroeer, Joerg and Drude, Lukas and Heymann, Jahn and Haeb-Umbach, Reinhold},
  title     = {{Front-End Processing for the CHiME-5 Dinner Party Scenario}},
  year      = {2018},
  booktitle = {CHiME5 Workshop},
}
```

Related work:

The RWTH/UPB System Combination for the CHiME 2018 Workshop ([pdf](https://groups.uni-paderborn.de/nt/pubs/2018/INTERSPEECH_2018_Heitkaemper_RWTH_Paper.pdf))
```
@inproceedings{Kitza2018,
  author    = {Kitza, Markus and Michel, Wilfried and Boeddeker, Christoph and Heitkaemper, Jens and Menne, Tobias and Schl{\"u}ter, Ralf and Ney, Hermann and Schmalenstroeer, Joerg and Drude, Lukas and Heymann, Jahn and others},
  title     = {The RWTH/UPB system combination for the CHiME 2018 workshop},
  year      = {2018}
  booktitle = {CHiME-5 Workshop},
}
```
Guided Source Separation Meets a Strong ASR Backend: Hitachi/Paderborn University Joint Investigation for Dinner Party ASR ([pdf](https://arxiv.org/pdf/1905.12230.pdf), [slides](https://groups.uni-paderborn.de/nt/pubs/2018/INTERSPEECH_2019_Boeddeker_Slides.pdf))
```
@Article{Kanda2019,
  author    = {Kanda, Naoyuki and Boeddeker, Christoph and Heitkaemper, Jens and Fujita, Yusuke and Horiguchi, Shota and Nagamatsu, Kenji and Haeb-Umbach, Reinhold},
  title     = {{Guided Source Separation Meets a Strong ASR Backend: Hitachi/Paderborn University Joint Investigation for Dinner Party ASR}},
  year      = {2019},
  booktitle = {Interspeech},
}
```
An Investigation into the Effectiveness of Enhancement in ASR Training and Test for CHiME-5 Dinner Party Transcription ([pdf](https://arxiv.org/pdf/1909.12208.pdf))
```
@inproceedings{Zorila2019,
  title     = {An Investigation into the Effectiveness of Enhancement in ASR Training and Test for CHiME-5 Dinner Party Transcription},
  author    = {Zoril\u{a}, C\u{a}t\u{a}lin and Boeddeker, Christoph and Doddipatla, Rama and Haeb-Umbach, Reinhold},
  year={2019},
  booktitle = {2019 IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU)},
}
```

Towards a speaker diarization system for the CHiME 2020 dinner party transcription ([pdf](https://chimechallenge.github.io/chime2020-workshop/abstracts/CHiME_2020_abstract_boeddeker.pdf), [slides](https://chimechallenge.github.io/chime2020-workshop/presentations/CHiME_2020_slides_boeddeker.pdf), [video](https://www.youtube.com/watch?v=tQhrqhVtsQI&feature=youtu.be))
```
@inproceedings{Boeddeker2018CHiME6,
  author    = {Boeddeker, Christoph and Cord-Landwehr, Tobias and Heitkaemper, Jens and Zoril\u{a}, C\u{a}t\u{a}lin and Hayakawa, Daichi and Li, Mohan and Liu, Min and Doddipatla, Rama and Haeb-Umbach, Reinhold},
  title     = {{Towards a speaker diarization system for the CHiME 2020 dinner party transcription}},
  year      = {2020},
  booktitle = {CHiME-6 Workshop},
}
```

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

# CHiME-6 Track2: RTTM files

In Track 2 of CHiME-6 it is not allowed to use the human annotations for
utterance starts and ends.
Instead they must be estimated. As format they used RTTM files
(For a description see https://github.com/nryant/dscore#rttm).
Here an example line for such a file:
```
SPEAKER S09 1 65.58 1.75 <NA> <NA> P25 <NA> <NA>
```

Once you have an estmate for the utterance starts and ends, you can enhance the
data with the following code:

```bash
python -m pb_chime5.scripts.kaldi_run_rttm with \
  storage_dir=path/to/save/enhanced/data \
  chime6_dir='/net/fastdb/chime6/CHiME6' \
  database_rttm="https://raw.githubusercontent.com/nateanl/chime6_rttm/master/dev_rttm" \
  activity_rttm="https://raw.githubusercontent.com/nateanl/chime6_rttm/master/dev_rttm" \
  session_id=dev \
  job_id=1 \
  number_of_jobs=1 \
  context_samples=160000 \
  bss_iterations=5 \
  multiarray='outer_array_mics'
```
 - `storage_dir`: Path where to store the enhanced data (`<storage_dir>/audio/<dataset>/*.wav`)
   - The enhanced data will be written to `<storage_dir>/audio/<dataset>`.
 - `chime6_dir`: Path to the CHiME-6 folder.
 - `session_id` dataset/session to enhance, e.g. `dev`, `eval`, `train`, `S02`, ...
 - `database_rttm` must contain the utterance starts and ends for the selected `session_id`. These starts and ends are used to write the audio files that can be used for ASR.
 - `activity_rttm`: Default is the same as `database_rttm`. May have more silence than `database_rttm`. e.g. `activity_rttm` has word start and end, while `database_rttm` has sentence start and end.
 - `job_id` and `number_of_jobs`: Control the subset you want to calculate. This option is intended for kaldi (e.g. `run.pl`). Alternatively, you could use `mpiexec -np $number_of_jobs` in front of the call to parallelize the enhancement. This should be slightly faster than kaldi.


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
