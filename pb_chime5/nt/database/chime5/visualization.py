"""
    This module contains all functions related to visualizing data from the
    CHiME 5 challenge (plots, pandas, ...)
"""

import itertools
from operator import itemgetter
import datetime
from collections import Counter
from pathlib import Path
import re

from pb_chime5.nt.visualization import matplotlib_fix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pb_chime5.nt.io.json_module import load_json
from pb_chime5.nt.io.data_dir import database_jsons
from pb_chime5.nt.database.chime5 import Chime5, CrossTalkFilter, SessionFilter
from pb_chime5.nt.database.chime5.get_speaker_activity import get_active_speaker


def speaker_activity_per_sess(sessions: list, sample_step=160):
    """

    :param sessions: Calculate speaker activity for sessions given in
        `sessions`.
    :param sample_step: Consider only every i-th sample in speaker activity
        array for calculation. This may lower accuracy but can greatly speed
        up calculations (e.g. for notebooks). Defaults to 160, i.e. take a
        sample every 0.01 seconds.
    :return: df: pd.DataFrame
        An tabular overview of relative speaker activity per session.
    """

    sample_rate = 16000
    start_sample = 0
    end_sample = 10800*sample_rate  # 3 hours

    speaker_activity = dict()
    speakers = list()

    for sess_id in sessions:

        speaker_json = load_json(str(database_jsons /
                                     'chime5_speech_activity' /
                                     f'{sess_id}.json')
                                 )

        target_speakers = sorted([spk for spk in speaker_json.keys()
                                  if spk.startswith('P')])

        speakers += target_speakers

        active_speaker_samples = dict()

        for spk_id in target_speakers:

            activity = get_active_speaker(start_sample, end_sample,
                                          sess_id, spk_id,
                                          speaker_json=speaker_json,
                                          sample_step=sample_step,
                                          )

            active_speaker_samples[spk_id] = activity[spk_id]['activity']

        speaker_activity_arr = np.array(list(active_speaker_samples.values()))

        sess_start_sample, sess_end_sample = np.squeeze(np.argwhere(
            speaker_activity_arr.any(0)))[[0, -1]]

        # Discard samples at beginning and end of session where no speaker is
        # active
        speaker_activity_arr = speaker_activity_arr[
                               :, sess_start_sample:sess_end_sample+1
                               ]

        speaker_activity[sess_id] = {spk: row.sum()/row.size for spk, row in
                                     zip(target_speakers, speaker_activity_arr)}
        speaker_activity[sess_id]['Total Activity per Session'] = \
            np.sum(speaker_activity_arr.any(0))/speaker_activity_arr.shape[1]

    # Generate pandas.DataFrame
    df = pd.DataFrame.from_dict(speaker_activity)
    df.index = sorted(set(speakers)) + ['Total Activity per Session']
    df = df.sort_values(by='Total Activity per Session', axis=1,
                        ascending=False)
    df = (df.style
          .format(lambda x: f"{x:.2%}" if x > 0 else x)
          .applymap(lambda x: 'background: yellow' if x > 0 else '')
          )
    return df


def plot_speaker_timelines(speaker_activity: dict, left_init, sample_step=1,
                           highlight_example=None):
    """
    Prepare time line for every speaker in the session

    :param speaker_activity: Contains for each speaker a np.ndarray where an
        entry is either set to True if the speaker is active or False if the
        speaker is inactive at the current time sample
    :param left_init: Start offset (in samples)
    :return: ax: plt.axes()
        Raw speaker time lines
    """

    if highlight_example:
        target, _, kaldi_start, kaldi_end = re.split(
            '[_-]', highlight_example['example_id']
        )
        target_start = int(kaldi_start) * 160 // sample_step - left_init
        target_end = int(kaldi_end) * 160 // sample_step - left_init
    else:
        target, target_start, target_end = None, None, None


    plot_list = dict()
    plot_color = dict()
    speaker_numbers = [enum[0] for enum in enumerate(speaker_activity.keys())]
    speaker_size = len(speaker_numbers)

    fig, ax = plt.subplots(figsize=(20, 10))
    for idx, spk in enumerate(speaker_activity.keys()):
        plot_list[idx] = list()
        plot_color[idx] = list()
        for k, v in itertools.groupby(enumerate(speaker_activity[spk]),
                                      key=itemgetter(1)):
            v = list(v)
            if spk == target and target_start >= v[0][0] and \
                            target_end <= v[-1][0] + 1:
                plot_list[idx].extend([
                    target_start - v[0][0],
                    target_end - target_start,
                    v[-1][0] + 1 - target_end
                ])
                plot_color[idx].extend(['g', 'y', 'g'])
            else:
                plot_list[idx].append(v[-1][0] - v[0][0] + 1)
                if k:
                    plot_color[idx].append('g')
                else:
                    plot_color[idx].append('w')

    max_bins = max([len(bin_list) for bin_list in plot_list.values()])

    activity_matrix = np.zeros((4, max_bins), dtype=tuple)
    for idx, durations, colors in zip(plot_list.keys(), plot_list.values(),
                                     plot_color.values()):
        activity_matrix[idx] = list(zip(
            np.pad(durations, (0, max_bins-len(durations)), 'constant',
                   constant_values=0),
            colors + ['w'] * (max_bins - len(colors))
        ))

    num_active_speakers = np.sum(np.asarray(list(speaker_activity.values())), 0)

    left_val = np.repeat(left_init, speaker_size)

    # Plot time bars
    for plot_values in activity_matrix.T:
        durations = list(map(itemgetter(0), plot_values))
        colors = list(map(itemgetter(1), plot_values))
        ax.barh(speaker_numbers, durations, 1, color=colors,
                left=left_val)
        left_val += durations

    # Add visualization for start and end of time frames where only one speaker
    # is active
    for k, v in itertools.groupby(enumerate(num_active_speakers),
                                  key=itemgetter(1)):
        v = list(v)
        if k == 1:
            ax.axvline(x=left_init + v[0][0], color='blue', linewidth=1.5)
            ax.axvline(x=left_init + v[-1][0], color='red', linewidth=1.5)

    return fig, ax


def sess_timeline(session: str, start=60.0, end=None, duration=120.0,
                  sample_step=1, verbose=True, highlight_example=None):
    """
    Plot timeline for specified session

    :param session: The session ID, e.g. 'S02'
    :param start: Start time of the time line (in seconds)
    :param end: End time of the time line (in seconds). If not provided, end
        time is set to `start` + `duration`
    :param duration: Duration of time frame (in seconds)
    :param sample_step: Consider only every i-th sample in speaker activity
        array for calculation. This may lower accuracy but can greatly speed
        up calculations (e.g. for notebooks). Defaults to 1, i.e. take every
        sample.
    :return: ax: plt.axes()
        Time line ready to plot
    """

    sample_rate = 16000
    if not end:
        end = start + duration

    speaker_json = load_json(str(database_jsons /
                                 'chime5_speech_activity' /
                                 f'{session}.json')
                             )

    speakers = sorted([spk for spk in speaker_json.keys()
                       if spk.startswith('P')])

    speaker_activity = dict()

    for spk_id in speakers:
        activity = get_active_speaker(int(start * sample_rate),
                                      int(end * sample_rate),
                                      session, spk_id,
                                      speaker_json=speaker_json,
                                      sample_step=sample_step
                                      )

        speaker_activity[spk_id] = activity[spk_id]['activity']

    if verbose:
        print(f"Start time @ {datetime.timedelta(seconds=start)}")
        for k, v in reversed(list(speaker_activity.items())):
            print(f"Speaker {k} active: {100*np.sum(v)/len(v):.2f} % "
                  f"in time frame")

    plt.ioff()

    fig, ax = plot_speaker_timelines(speaker_activity,
                                     int(start * sample_rate) // sample_step,
                                     sample_step=sample_step,
                                     highlight_example=highlight_example
                                     )

    # Format plot and add information
    green_patch = mpatches.Patch(color='green', label='Active')
    if highlight_example:
        yellow_patch = mpatches.Patch(color='yellow', label='Target')
        patches = [green_patch, yellow_patch]
    else:
        patches = [green_patch]
    ax.set_xticks(np.linspace(start * sample_rate // sample_step,
                              end * sample_rate // sample_step, 8
                              )
                  )
    ax.set_xticklabels(list(map(lambda x: f'{x:.2f}',
                                np.linspace(start, end, 8, dtype=np.float32))
                            )
                       )
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Speaker ID')
    ax.set_yticks(np.arange(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.legend(handles=patches)
    ax.set_title(f'Speaker Activity over Time, Session {session}')
    return fig


def _highlight_min(s):
    is_min = s == s.min()
    return ['background-color: green' if v else '' for v in is_min]


def _highlight_max(s):
    is_max = s == s.max()
    return ['background-color: red' if v else '' for v in is_max]


def plot_histogram(rel_olap_per_utt_per_sess: dict, ncols=3):
    sessions = list(rel_olap_per_utt_per_sess.keys())

    nrows = int(np.ceil(len(sessions) / ncols))
    hist_plot, axes = plt.subplots(nrows, ncols, squeeze=False,
                                   figsize=(ncols * 5, nrows * 5))

    blue_patch = mpatches.Patch(color='blue', label='Utterances with overlap')
    red_patch = mpatches.Patch(color='red', label='Overlap-free utterances')

    for idx, dict_items in enumerate(rel_olap_per_utt_per_sess.items()):
        sess_id = dict_items[0]
        data = np.array(dict_items[1])
        num_olap_free = np.array(len(data) - len(data[[data > 0]]))

        occurences = dict(
            Counter(np.array(data[[data > 0]] * 10, dtype=np.int32) * 10))
        if 90 and 100 in occurences:
            occurences[90] += occurences[100]
            occurences.pop(100)

        row = int(np.floor(idx / ncols))
        col = idx % ncols

        # Generate histogram for subplot
        axes[row, col].bar(np.array(list(occurences.keys())) + 5,
                           list(occurences.values()), width=5,
                           color='blue')
        axes[row, col].bar(0, num_olap_free, width=5, color='red')
        axes[row, col].set_title(f'Session {sess_id}')
        axes[row, col].set_xticks(np.arange(0, 101, 10))
        axes[row, col].set_xlabel('Relative overlap per utterance (in %)')
        axes[row, col].set_ylabel('Utterances per bin')
        axes[row, col].legend(handles=[red_patch, blue_patch])

    return hist_plot


def calculate_overlap(dataset, sessions, json_path=database_jsons,
                      plot_hist=True, ncols=3):

    if isinstance(json_path, str):
        json_path = Path(json_path)

    db = Chime5()
    if dataset == 'train':
        iterator = db.get_iterator_by_names(db.datasets_train)
    elif dataset == 'dev':
        iterator = db.get_iterator_by_names(db.datasets_eval)
    elif dataset == 'test':
        iterator = db.get_iterator_by_names(db.datasets_test)
    else:
        raise ValueError(f'Datset {dataset} unknown')
    iterator = iterator.filter(CrossTalkFilter(dataset, json_path,
                                               with_crosstalk='all',
                                               min_overlap=0.0,
                                               max_overlap=1.0)
                               )

    overlap_durations = dict()
    rel_overlap_per_utt_per_sess = dict()
    utt_overlap_per_sess = dict()
    num_overlapping_utts_per_sess = dict()

    for session_id in sessions:
        session_it = iterator.filter(SessionFilter(session_id))

        relative_overlaps = np.array([ex['overlap'] for ex in session_it])
        utterance_durations = np.array([
            int(re.split('[_-]', ex['example_id'])[3]) -
            int(re.split('[_-]', ex['example_id'])[2]) for ex in session_it
        ])

        olap_duration = utterance_durations * relative_overlaps
        overlap_durations[session_id] = olap_duration[olap_duration > 0]
        num_overlapping_utts_per_sess[session_id] = np.sum(relative_overlaps > 0)
        utt_overlap_per_sess[session_id] = \
            num_overlapping_utts_per_sess[session_id] / len(list(session_it))
        rel_overlap_per_utt_per_sess[session_id] = relative_overlaps

    # Convert kaldi time to seconds
    data = {
        '#Overlapping Utterances': list(num_overlapping_utts_per_sess.values()),
        'Relative Overlap': list(utt_overlap_per_sess.values()),
        'Minimum Overlap': [np.min(durations / 100) for durations in
                            overlap_durations.values()],
        'Average Overlap': [np.average(durations / 100) for durations in
                            overlap_durations.values()],
        'Maximum Overlap': [np.max(durations / 100) for durations in
                            overlap_durations.values()]
    }

    # Generate pandas.DataFrame
    olap_df = (pd.DataFrame(data=data, index=sessions,
                            columns=['#Overlapping Utterances',
                                     'Minimum Overlap',
                                     'Average Overlap',
                                     'Maximum Overlap',
                                     'Relative Overlap'
                                     ]
                            )
               .sort_values(by='Relative Overlap', ascending=True)
               .style
               .format(
        {
            'Relative Overlap': "{:.2%}",
            'Minimum Overlap': "{:.2f} s",
            'Average Overlap': "{:.2f} s",
            'Maximum Overlap': "{:.2f} s"
        }
    )
               .apply(_highlight_min,
                      subset=['#Overlapping Utterances', 'Average Overlap'])
               .apply(_highlight_max,
                      subset=['#Overlapping Utterances', 'Average Overlap'])
               .set_properties(**{'text-align': 'center'})
               )

    # Get histogram
    if plot_hist:
        hist = plot_histogram(rel_overlap_per_utt_per_sess, ncols=ncols)
        return olap_df, hist

    return olap_df, None
