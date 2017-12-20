def dump_database_as_json(filename, obj):
    with open(filename, 'w') as fid:
        json.dump(obj, fid, sort_keys=True, indent=4, ensure_ascii=False)


import json
from nt.database.keys import *

def print_template():
    """ Prints the template used for the json file

    :return:
    """
    print('<root>\n'
          '..<train>\n'
          '....annotations\n'
          '......<scenario>\n'
          '........<utterance_id>\n'
          '..........nsamples: <nsamples>\n'
          '....<flists>\n'
          '......<file_type> (z.B. wav)\n'
          '..........<scenario> (z.B. tr05_simu, tr05_real...)\n'
          '............<utterance_id>\n'
          '..............<observed>\n'
          '................<A>\n'
          '..................path\n'
          '................<B>\n'
          '..................path\n'
          '..............<image>\n'
          '................<A>\n'
          '..................path\n'
          '................<B>\n'
          '..................path\n'
          '..............<source>\n'
          '................path\n'
          '\n'
          '..<dev>\n'
          '..<test>\n'
          '..<orth>\n'
          '....<word>\n'
          '......<utterance_id>\n'
          '....<phoneme>\n'
          '......<utterance_id>\n'
          '........string\n'
          '..<flists>\n'
          '....Flist_1\n'
          '....Flist_2\n')


def print_old_template():
    """ Prints the template used for the json file

    :return:
    """
    print('<root>\n'
          '..<step_name>\n'
          '....<log>\n'
          '......list of strings\n'
          '....<config>\n'
          '......dict\n'
          '....<git_hash>\n'
          '......string\n'
          '....<date>\n'
          '......string\n'
          '....<comment>\n'
          '......string\n'
          '..<train>\n'
          '....<step_name>\n'
          '......Database / Feature extraction:\n'
          '......<flists>\n'
          '........<file_type> (z.B. wav)\n'
          '..........<channels>\n'
          '............<scenario>\n'
          '..............<utterance_id>\n'
          '................<observed>\n'
          '..................<A>\n'
          '....................string\n'
          '..................<B>\n'
          '....................string\n'
          '................<image>\n'
          '..................<A>\n'
          '....................string\n'
          '..................<B>\n'
          '....................string\n'
          '................<source>\n'
          '..................string\n'
          '......Beamformer:\n'
          '......<flists>\n'
          '........<file_type> (z.B. wav)\n'
          '..........<channels>\n'
          '............<scenario>\n'
          '..............<utterance_id>\n'
          '................<observed>\n'
          '....................string\n'
          '......<scores>\n'
          '..........<channels>\n'
          '............<scenario>\n'
          '..............<utterance_id>\n'
          '................<score_type>\n'
          '....................0 -> float (Wert vorher)\n'
          '....................1 -> float (Wert nachher)\n'
          '\n'
          '..<dev>\n'
          '..<test>\n'
          '..<orth>\n'
          '....<utterance_id>\n'
          '......string\n'
          '..<flists>\n'
          '....Flist_1\n'
          '....Flist_2\n')


def traverse_to_dict(data, path, delimiter='/'):
    """ Returns the dictionary at the end of the path defined by `path`

    :param data: A dict with the contents of the json file
    :param path: A string defining the path with or without
        leading and trailing slashes
    :param delimiter: The delimiter to convert the string to a list
    :return: dict at the end of the path
    """

    path = path.strip('/').split(delimiter)
    cur_dict = data[path[0]]
    for next_level in path[1:]:
        try:
            cur_dict = cur_dict[next_level]
        except KeyError as e:
            print('Error: {k} not found. Possible keys are {keys}'
                  .format(k=next_level, keys=cur_dict.keys()))
            raise e
    return cur_dict


def get_available_channels(flist):
    """ Returns all available channels in the format *type/channel_no*
    inferred from the first utterance.

    :param flist: A dictionary with ids as keys and file lists as values
    :type flist: dict

    :return: A list of available channels
    """

    if len(flist) == 0:
        return list()

    utt = list(flist.keys())[0]
    channels = list()
    for src in flist[utt]:
        if type(flist[utt][src]) is dict:
            for ch in flist[utt][src]:
                channels.append('{src}/{ch}'.format(src=src, ch=ch))
        else:
            channels.append(src)
    return channels


def get_flist_for_channel(flist, ch):
    """ Returns a flist containing only the files for a specific channel

    :param flist: A dict representing a file list
    :param ch: The channel to get

    :return: A dict with the ids and the files for the specific channel
    """

    if not ch in get_available_channels(flist):
        raise KeyError(
            'Could not find channel {ch}. Available channels are {chs}'
                .format(ch=ch, chs=get_available_channels(flist)))

    ret_flist = dict()
    for utt in flist:
        val = flist[utt]
        for branch in ch.split('/'):
            if branch in val:
                val = val[branch]
            else:
                return []
        ret_flist[utt] = val

    assert len(ret_flist) > 0, \
        'Could not find any files for channel {c}'.format(c=str(ch))
    return ret_flist


def get_channel_for_utt(flist, ch, utt):
    """ Returns a specific channel for one utterance.

    Raises a KeyError if the channel does not exist

    :param flist: A dict representing a file list, i.e. the keys are the utt ids
    :param ch: The channel to fetch (i.e. X/CH1). Separator must be `/`
    :param utt: The utterance to fetch
    :return: Path to the file
    """
    val = flist[utt]
    for branch in ch.split('/'):
        if branch in val:
            val = val[branch]
        else:
            raise KeyError('No channel {} for {}'.format(ch, utt))
    return val


def add_flist(flist, progress_json, scenario, stage='train',
              file_type='wav', channel_type='observed', channel='CH1'):
    """ Adds a file list to the current progress_json object

    Example::

    ....<flists>
    ......<file_type> (z.B. wav)
    ........<scenario> (z.B. tr05_simu, tr05_real...)
    ..........<utterance_id>
    ............<observed>
    ..............<A>

    :param flist: A dictionary acting as a file list
    :param progress_json: The current json object
    :param scenario: Name for the file list
    :param stage: [train, dev, test]
    :param file_type: Type of the referenced files. e.g. wav, mfcc, ...
    :return:
    """

    def _get_next_dict(cur_dict, key):
        try:
            return cur_dict[key]
        except KeyError:
            cur_dict[key] = dict()
            return cur_dict[key]

    cur_dict = progress_json[stage]
    flists_dict = _get_next_dict(cur_dict, 'flists')
    file_type_dict = _get_next_dict(flists_dict, file_type)
    scenario_dict = _get_next_dict(file_type_dict, scenario)

    for utt_id in flist:
        utt_id_dict = _get_next_dict(scenario_dict, utt_id)
        channel_type_dict = _get_next_dict(utt_id_dict, channel_type)
        channel_type_dict[channel] = flist[utt_id]

def add_listing(flist, progress_json, scenario):


    def _get_next_dict(cur_dict, key):
        try:
            return cur_dict[key]
        except KeyError:
            cur_dict[key] = dict()
            return cur_dict[key]

    cur_dict = progress_json
    dataset_dict = _get_next_dict(cur_dict, DATASETS)
    dataset_dict[scenario] = list(flist.keys())

def add_examples(flist, orth, progress_json, scenario,
               channel_type='observed', channel=None):
    """ Adds a file list to the current progress_json object

    datasets:
    <unique example id 1>
        audio_path:
            observation:
                a0:
                    <path to observation of array 0 and channel 0>
                    <path to observation of array 0 and channel 0>
                    ...
                a1:
                    c0: <path to observation of array 1 and channel 0>
                    c99: <path to observation of array 1 and channel 99>



    :param flist: A dictionary acting as a file list
    :param progress_json: The current json object
    :param scenario: Name for the file list
    :param stage: [train, dev, test]
    :param file_type: Type of the referenced files. e.g. wav, mfcc, ...
    :return:
    """

    def _get_next_dict(cur_dict, key):
        try:
            return cur_dict[key]
        except KeyError:
            cur_dict[key] = dict()
            return cur_dict[key]

    cur_dict = progress_json
    flists_dict = _get_next_dict(cur_dict, DATASETS)
    scenario_dict = _get_next_dict(flists_dict, scenario)

    for utt_id in flist:
        utt_id_dict = _get_next_dict(scenario_dict, utt_id)
        utt_id_dict.update({TRANSCRIPTION : orth[utt_id.split('_')[0]]})
        audio_path_dict = _get_next_dict(utt_id_dict, AUDIO_PATH)
        channel_type_dict = _get_next_dict(audio_path_dict, channel_type)
        if channel is None:
            if not audio_path_dict[channel_type]:
                audio_path_dict[channel_type] = []
            audio_path_dict[channel_type].append(flist[utt_id])
        else:
            channel_type_dict[channel] = flist[utt_id]


def combine_flists(data, flist_1_path, flist_2_path, flist_path,
                   postfix_1='', postfix_2='', delimiter='/',
                   only_common_channels=False):
    """ Combines two file lists into a new file list ``flist_name``

    The new file list will only have those channels, which are present in both
    file lists.

    :param flist_1_path: Path to the first file list
    :param flist_2_path: Path to the second file list
    :param flist_path: Path to the new file list
    """

    flist_1 = traverse_to_dict(data, flist_1_path, delimiter)
    flist_2 = traverse_to_dict(data, flist_2_path, delimiter)

    if postfix_1 == '' and postfix_2 == '':
        assert len(set(list(flist_1.keys()) + list(flist_2.keys()))) \
               == len(flist_1) + len(flist_2), \
            'The ids in the file lists must be unique.'

    channels_flist_1 = get_available_channels(traverse_to_dict(
        data, flist_1_path, delimiter
    ))
    channels_flist_2 = get_available_channels(traverse_to_dict(
        data, flist_2_path, delimiter
    ))

    if only_common_channels:
        common_channels = set((ch.split('/')[0]
                               for ch in channels_flist_1
                               if ch in channels_flist_2))

    new_flist = dict()
    for flist, postfix in zip([flist_1, flist_2], [postfix_1, postfix_2]):
        for id in flist.keys():
            new_id = id if len(postfix) == 0 else id + '_' + postfix
            new_flist[new_id] = dict()
            for ch in flist[id]:
                if only_common_channels:
                    if ch in common_channels:
                        new_flist[new_id][ch] = flist[id][ch]
                else:
                    new_flist[new_id][ch] = flist[id][ch]

    flist_name = flist_path.split(delimiter)[-1]
    flist_parent_path = delimiter.join(flist_path.split(delimiter)[:-1])

    flist_parent = traverse_to_dict(data, flist_parent_path, delimiter)
    flist_parent[flist_name] = new_flist
