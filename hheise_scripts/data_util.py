#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 04/08/2021 18:21
@author: hheise

Utility functions specifically for data handling, transformations and preprocessing
"""
from typing import Tuple, List, Optional, Iterable
import numpy as np
from schema import hheise_behav, hheise_placecell


def dict_to_array(data: dict, order: Iterable[str]) -> np.ndarray:
    """
    Transform a dict holding several 1D ndarrays of the same length into a 2D np.ndarray, column-wise.

    Args:
        data: Dict with multiple 1D ndarrays of the same length.
        order: List of strings with keys, in the desired column order of the key-value pairs.

    Returns:
        Content of the dict in np.ndarray of shape (#samples, #fields)
    """
    return np.vstack([data[key] for key in order]).T


def get_accepted_trials(key: dict, trace_type: str, get_avg: Optional[bool] = True) -> Tuple:
    """
    Classifies and separates trials in a session based on their context (corridor types).
    Used to process different contexts separately, if the context changed during a session, e.g. for validation trials.

    Args:
        key: Primary keys of the session. Has to include all PKs of hheise_placecell.Synchronization().
        trace_type: Activity trace that should be classified. Has to be an attribute in
            hheise_placecell.BinnedActivity.ROI().
        get_avg: Bool flag whether trial-averaged traces should be returned or not.

    Returns:
        Three lists, with one entry per corridor type encountered in the session
            - corridor_types: Corridor type integer labels of the included trials
                0 = only normal trials
                1 = all trials
                2 = only second context (e.g. no tone)
                3 = only third context (if two context changes per session, e.g. No pattern, then no tone)
            - trace_list: Activity traces of included trials. Shape (n_cells, n_bins) if get_avg=True,
                (n_cells, n_bins, n_accepted_trials) if get_avg=False.
            - accepted_trials: Trial IDs (for this session) of the included trials. None if all trials are included.

    """
    if len(hheise_placecell.Synchronization & key) == 1:
        n_trials = len(hheise_placecell.Synchronization.VRTrial & key)
    else:
        raise IndexError('More than one session in placecell.Synchronization for {}'.format(key))

    # Check if the corridor condition changed during the session (validation trials), and trials have to be treated separately
    switch = (hheise_behav.VRSessionInfo & key).fetch1('condition_switch')

    # Store BinnedActivity entry in shorter variable name
    if get_avg:
        entry = hheise_placecell.BinnedActivity & key
    else:
        entry = hheise_placecell.BinnedActivity.ROI & key

    trial_mask = np.ones(n_trials, dtype=bool)  # by default, all trials will be processed

    # No switch, include all trials
    if switch == [-1]:
        corridor_types = [0]
        if get_avg:
            trace_list = [np.stack(entry.get_trial_avg(trace_type))]
        else:
            trace_list = [np.stack(entry.fetch(trace_type))]
        accepted_trials = [None]

    # One condition switch occurred, process conditions separately
    elif len(switch) == 1:
        corridor_types = [0, 1, 2]  # corridor_type label of the following traces arrays (0=only normal, 1=all trials, 2=only changed condition 1, 3=only changed condition 2)
        trial_mask[switch[0]:] = False  # Exclude trials with different condition

        if get_avg:
            trace_list = [entry.get_trial_avg(trace_type, trial_mask=trial_mask),     # First array includes all normal trials (mask)
                          entry.get_trial_avg(trace_type),                            # Second array includes all trials (no mask)
                          entry.get_trial_avg(trace_type, trial_mask=~trial_mask)]    # Third array includes all changed trials (inverse mask)
        else:
            trace_list = [np.stack(entry.fetch(trace_type))[:, :, trial_mask],     # First array includes all normal trials (mask)
                          np.stack(entry.fetch(trace_type)),                       # Second array includes all trials (no mask)
                          np.stack(entry.fetch(trace_type))[:, :, ~trial_mask]]    # Third array includes all changed trials (inverse mask)

        accepted_trials = [np.where(trial_mask)[0],
                           None,
                           np.where(~trial_mask)[0]]

    # Two switches occurred
    elif len(switch) == 2:
        corridor_types = [0, 1, 2, 3]  # corridor_type label of the following traces arrays
        trial_mask[switch[0]:] = False  # Only normal trials
        cond1 = np.zeros(trial_mask.shape, dtype=bool)
        cond1[switch[0]:switch[1]] = True  # Only trials with no pattern
        cond2 = np.zeros(trial_mask.shape, dtype=bool)
        cond2[switch[1]:] = True  # Only trials with no tone and no pattern
        if get_avg:
            trace_list = [entry.get_trial_avg(trace_type, trial_mask=trial_mask),
                          entry.get_trial_avg(trace_type),
                          entry.get_trial_avg(trace_type, trial_mask=cond1),
                          entry.get_trial_avg(trace_type, trial_mask=cond2)]
        else:
            trace_list = [np.stack(entry.fetch(trace_type))[:, :, trial_mask],
                          np.stack(entry.fetch(trace_type)),
                          np.stack(entry.fetch(trace_type))[:, :, cond1],
                          np.stack(entry.fetch(trace_type))[:, :, cond2]]
        accepted_trials = [np.where(trial_mask)[0],
                           None,
                           np.where(cond1)[0],
                           np.where(cond2)[0]]
    else:
        raise IndexError(f"Trial {key}:\nCondition switch {switch} not recognized.")

    return corridor_types, trace_list, accepted_trials


def get_binned_licking(data: np.ndarray, bin_size: int = 1, normalized: bool = False) -> np.ndarray:
    """
    Spatially bin licking data of one trial to create a histogram of licking vs. VR position.

    Args:
        data: Array of the trial behavior data with shape (n_samples, n_attributes). Format: position - licking - everything else
        bin_size: Bin size in VR units for binned licking performance analysis
        normalized: Bool flag whether lick counts should be returned normalized (sum of bins = 1)

    Returns:
        1D np.array with length 120/bin_size holding (normalized) binned lick counts per position bin
    """

    # select only time points where the mouse started to lick 
    lick_idx = np.where(data[:-1, 1] < data[1:, 1])[0]
    lick_pos = data[lick_idx, 0]

    # bin lick positions
    hist, _ = np.histogram(np.digitize(lick_pos, np.arange(start=-10, stop=111, step=bin_size)),
                                   bins=np.arange(start=1, stop=int(120/bin_size+2)), density=normalized)
    return hist