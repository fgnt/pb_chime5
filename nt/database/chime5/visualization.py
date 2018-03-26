import itertools
from operator import itemgetter
from functools import lru_cache
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from nt.database.chime5.get_speaker_activity import get_active_speaker

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

        activity = get_active_speaker(start_sample, end_sample,
                                      f'{sess_id}time',
                                      sample_step=sample_step,
                                      )

        target_speakers = sorted([speaker for speaker in list(activity.keys())
                                  if speaker.startswith('P')])

        speakers += target_speakers

        active_speaker_samples = dict()

        for idx, spk in enumerate(target_speakers):
            active_speaker_samples[spk] = activity[spk]['activity'][idx]

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
def sess_timeline(session: str, start=60, duration=120):
    """
    Plot timeline for specified session from `start` to `start`+`duration`

    :param session: The session ID, e.g. 'S02'
    :param start: Start time of the time line (in seconds)
    :param duration: Duration of time frame (in seconds)
    :return: ax: plt.axes()
        Time line ready to plot
    """

    sample_rate = 16000
    end = start + duration

    print(f"Start time @ {datetime.timedelta(seconds=start)}")

    activity = get_active_speaker(start * sample_rate, end * sample_rate,
                                  f'{session}time')

    speakers = sorted([speaker for speaker in list(activity.keys()) if
                       speaker.startswith('P')])

    speaker_activity = dict()

    for idx, spk in enumerate(speakers):
        speaker_activity[spk] = activity[spk]['activity'][idx]

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
