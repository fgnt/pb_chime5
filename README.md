# pb_chime5: Front-End Processing for the CHiME-5 Dinner Party Scenario

This repository is not finished (i.e. work in progress).
Open points:

- [x] core enhancement code
- [ ] remove dependencies from our infrastructure
- [ ] launch script
- [ ] manual script
- [ ] code cleanup, remove all unnessesary code and rearrange the code files

Challenge website: http://spandh.dcs.shef.ac.uk/chime_challenge/

Workshop website: http://spandh.dcs.shef.ac.uk/chime_workshop/

This repository includes all components of the CHiME-5 front-end presented by Paderborn University on the CHiME-5 workshop. In combination with an acoustic model presented by the RWTH Aachen this multi-channel front-end achieved the third best results during the challenge with 54.56 %. 

The core code for this enhamcement is located in the file `pb_chime5/core.py`.
Run the enhancement with something like `python -m pb_chime5.scripts.run with session_id=dev wpe=True wpe_tabs=2`


If you are using this code please cite the following paper:

```
@Article{PB2018CHiME5,
  author    = {Boeddeker, Christoph and Heitkaemper, Jens and Schmalenstroeer, Joerg and Drude, Lukas and Heymann, Jahn and Haeb-Umbach, Reinhold},
  title     = {{Front-End Processing for the CHiME-5 Dinner Party Scenario}},
  year      = {2018},
  booktitle = {CHiME5 Workshop},
}
```
