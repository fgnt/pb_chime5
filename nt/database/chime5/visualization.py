import itertools
from operator import itemgetter
from functools import lru_cache
import datetime
from collections import Counter
from pathlib import Path
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from nt.io.json_module import load_json
from nt.io.data_dir import database_jsons
from nt.database.chime5 import Chime5, CrossTalkFilter, SessionFilter
from nt.database.chime5.get_speaker_activity import get_active_speaker, \
    to_numpy

"""
    This module contains all functions related to visualizing data from the 
    CHiME 5 challenge (plots, pandas, ...)
"""


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


def plot_speaker_timelines(speaker_activity: dict, left_init):
    """
    Prepare time line for every speaker in the session

    :param speaker_activity: Contains for each speaker a np.ndarray where an
        entry is either set to True if the speaker is active or False if the
        speaker is inactive at the current time sample
    :param left_init: Start offset (in samples)
    :return: ax: plt.axes()
        Raw speaker time lines
    """

    plot_list = dict()
    plot_color = dict()
    speaker_numbers = [enum[0] for enum in enumerate(speaker_activity.keys())]
    speaker_size = len(speaker_numbers)
    _, ax = plt.subplots(figsize=(20, 10))
    for idx, spk in enumerate(speaker_activity.keys()):
        bin_id = 0
        for k, v in itertools.groupby(enumerate(speaker_activity[spk]),
                                      key=itemgetter(1)):
            if bin_id not in plot_list.keys():
                plot_list[bin_id] = np.zeros(speaker_size, np.int64)
                plot_color[bin_id] = np.repeat('w', speaker_size)
            v = list(v)
            plot_list[bin_id][idx] = (v[-1][0] - v[0][0] + 1)
            if k:
                plot_color[bin_id][idx] = 'g'
            bin_id += 1

    num_active_speakers = np.sum(np.asarray(list(speaker_activity.values())), 0)

    left_val = np.repeat(left_init, speaker_size)

    # Plot time bars
    for bin_id, dur in plot_list.items():
        ax.barh(speaker_numbers, dur, 1, color=plot_color[bin_id],
                left=left_val)
        left_val += dur

    # Add visualization for start and end of time frames where only one speaker
    # is active
    for k, v in itertools.groupby(enumerate(num_active_speakers),
                                  key=itemgetter(1)):
        v = list(v)
        if k == 1:
            ax.axvline(x=left_init + v[0][0], color='blue', linewidth=1.5)
            ax.axvline(x=left_init + v[-1][0], color='red', linewidth=1.5)

    return ax


@lru_cache(maxsize=None)
def sess_timeline(session: str, start=60, duration=120, sample_step=1):
    """
    Plot timeline for specified session from `start` to `start`+`duration`

    :param session: The session ID, e.g. 'S02'
    :param start: Start time of the time line (in seconds)
    :param duration: Duration of time frame (in seconds)
    :param sample_step: Consider only every i-th sample in speaker activity
        array for calculation. This may lower accuracy but can greatly speed
        up calculations (e.g. for notebooks). Defaults to 1, i.e. take every
        sample.
    :return: ax: plt.axes()
        Time line ready to plot
    """

    sample_rate = 16000
    end = start + duration

    speaker_json = load_json(str(database_jsons /
                                 'chime5_speech_activity' /
                                 f'{session}.json')
                             )

    print(f"Start time @ {datetime.timedelta(seconds=start)}")

    speakers = sorted([spk for spk in speaker_json.keys()
                       if spk.startswith('P')])

    speaker_activity = dict()

    for spk_id in speakers:
        activity = get_active_speaker(start * sample_rate, end * sample_rate,
                                      session, spk_id,
                                      speaker_json=speaker_json,
                                      sample_step=sample_step
                                      )

        speaker_activity[spk_id] = activity[spk_id]['activity']

    for k, v in reversed(list(speaker_activity.items())):
        print(f"Speaker {k} active: {100*np.sum(v)/len(v):.2f} % in time frame")

    ax = plot_speaker_timelines(speaker_activity, start * sample_rate)

    # Format plot and add information
    green_patch = mpatches.Patch(color='green', label='Active')
    ax.set_xticks(np.linspace(start * sample_rate, end * sample_rate, 8))
    ax.set_xticklabels(np.linspace(start, end, 8, dtype=np.int64))
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Speaker ID')
    ax.set_yticks(np.arange(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.legend(handles=[green_patch])
    ax.set_title(f'Speaker Activity over Time, Session {session}')
    return ax


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


def get_crosstalk_examples(example_ids, session_id, crosstalk_times,
                           with_crosstalk=True, min_overlap=0.0,
                           max_overlap=0.0):
        """
        Return examples of a session according to flag `with_crosstalk`

        :param example_ids: List of example IDs of data set iterator
        :param session_id: The session whose utterances should be filtered.
        :param crosstalk_times: A dictionary which specifies start and end times
            of cross talk
        :param with_crosstalk: If False, return all utterances which have 0%
            overlap with any other utterance.
            If True, return all utterances which have at least one sample
            overlap with any other utterance.
        :param min_overlap: If `with_crosstalk` is True, filter only utterances
            whose overlap ratio is greater than `min_overlap`. Then only these
            utterances are considered to have overlap.
        :param max_overlap: If `with_crosstalk` is False, filter all utterances
            whose overlap_ratio is lower or equal to `max_overlap`. These
            additional filtered utterances will then be treated as
            "overlap-free".

        :return: (filtered_examples: list, filter_ratio: float,
                relative_overlap_per_utt: list)
            filtered_examples is a list of examples IDs which satisfy filter
                criterion `with_crosstalk`.
            filter_ratio is the number of examples in filtered_examples over
                number of total examples in the session.
            relative_overlap_per_utt is a list of relative overlap an utterance
                exhibits for each utterance
        """

        session_examples = list(
            filter(lambda x: x[4:7] == session_id, example_ids))
        num_examples = len(session_examples)

        crosstalk_start = np.array(crosstalk_times['start'])
        crosstalk_end = np.array(crosstalk_times['end'])

        filtered_examples = list()
        relative_overlap_per_utt = list()

        for example_id in session_examples:
            _, _, start_time, end_time = re.split('[_-]', example_id)
            # convert from kaldi time to samples
            start_sample, end_sample = (int(start_time)*160,
                                        int(end_time)*160)
            crosstalk_idx = np.logical_and(crosstalk_start >= start_sample,
                                           crosstalk_end <= end_sample)
            has_crosstalk = crosstalk_idx.any()
            if has_crosstalk:
                # sample only every 0.01 second
                utt_samples = to_numpy({'start': crosstalk_start[crosstalk_idx],
                                        'end': crosstalk_end[crosstalk_idx]},
                                       start_sample, end_sample,
                                       sample_step=160)
                overlap_ratio = utt_samples.sum() / int(
                    (end_sample - start_sample) / 160)
            else:
                overlap_ratio = 0
            relative_overlap_per_utt.append(overlap_ratio)
            if not with_crosstalk and overlap_ratio <= max_overlap:
                filtered_examples.append(example_id)
            elif with_crosstalk and overlap_ratio > min_overlap:
                filtered_examples.append(example_id)

        filter_ratio = len(filtered_examples) / num_examples

        # sort examples by start time
        return \
            sorted(filtered_examples, key=lambda x: re.split('[_-]', x)[2]), \
            filter_ratio, \
            relative_overlap_per_utt


def calculate_overlap(dataset, sessions, json_path=database_jsons,
                      with_crosstalk='all', min_overlap=0.0, max_overlap=1.0,
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
    filtered = iterator.filter(CrossTalkFilter(dataset, json_path,
                                               with_crosstalk=with_crosstalk,
                                               min_overlap=min_overlap,
                                               max_overlap=max_overlap)
                               )

    overlap_durations = dict()
    rel_overlap_per_utt_per_sess = dict()
    utt_overlap_per_sess = dict()
    num_overlapping_utts_per_sess = dict()

    for session_id in sessions:
        session_it = iterator.filter(SessionFilter(session_id))
        session_filtered = filtered.filter(SessionFilter(session_id))

        relative_overlaps = np.array([ex['overlap'] for ex in session_filtered])
        utterance_durations = np.array([
            int(re.split('[_-]', ex['example_id'])[3]) -
            int(re.split('[_-]', ex['example_id'])[2]) for ex in
            session_filtered
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
