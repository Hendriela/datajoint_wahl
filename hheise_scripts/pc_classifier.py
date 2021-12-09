#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 07/12/2021 15:32
@author: hheise

Functions that constitute the place cell classification pipeline for hheise_placecell.PlaceCells.
"""
import numpy as np
from typing import List, Optional, Tuple

from schema import hheise_placecell


def smooth_trace(trace: np.ndarray, bin_window: int) -> np.ndarray:
    """
    Smooths a trace (usually binned, but can also be used unbinned) by averaging each time point across adjacent values.

    Args:
        trace:      2D np.ndarray with shape (n_neurons, n_timepoints) containing data points.
        bin_window: Half-size of sliding window

    Returns:
        Array of the same size as trace, but smoothed
    """

    smoothed = np.zeros(trace.shape) * np.nan
    n_bins = trace.shape[1]
    for bin_idx in range(n_bins):
        # get the frame windows around the current time point i
        if bin_idx < bin_window:
            curr_left_bin = trace[:, :bin_idx]
        else:
            curr_left_bin = trace[:, bin_idx - bin_window:bin_idx]
        if bin_idx + bin_window > n_bins:
            curr_right_bin = trace[:, bin_idx:]
        else:
            curr_right_bin = trace[:, bin_idx:bin_idx + bin_window]
        curr_bin = np.hstack((curr_left_bin, curr_right_bin))

        smoothed[:, bin_idx] = np.mean(curr_bin, axis=1)

    return smoothed


def pre_screen_place_fields(trace: np.ndarray, bin_baseline: float,
                            placefield_threshold: float) -> List[List[Optional[np.ndarray]]]:
    """
    Performs pre-screening of potential place fields in a trace. A potential place field is any bin/point that
    has a higher dF/F value than 'placefield_threshold' % (default 25%) of the difference between the baseline and
    maximum dF/F of this trace. The baseline dF/F is the mean of the 'bin_baseline' % (default 25%) least active bins.

    Args:
        trace: 2D array with shape (n_neurons, n_bins) of the data that should be screened for place fields,
                e.g. smoothed binned trial-averaged data
        bin_baseline: Fraction of least active bins whose mean is defined as baseline dF/F
        placefield_threshold: Fraction difference between baseline and max dF/F above which a bin counts as place field

    Returns:
        List (one entry per cell) of lists, which hold arrays containing separate potential place fields
            (empty list if there are no place fields)
    """

    f_max = np.max(trace, axis=1)  # get maximum DF/F value of each neuron

    # get baseline dF/F value from the average of the 'bin_base' % least active bins (default 25% of n_bins)
    f_base = np.mean(np.sort(trace, axis=1)[:, :int(np.round((trace.shape[1] * bin_baseline)))], axis=1)

    # get threshold value above which a point is considered part of the potential place field (default 25%)
    f_thresh = ((f_max - f_base) * placefield_threshold) + f_base

    # get indices where the smoothed trace is above threshold
    rows, cols = np.where(np.greater_equal(trace, f_thresh[:, np.newaxis]))
    pot_place_idx = [cols[rows == i] for i in np.unique(rows)]

    # Split consecutive potential place field indices into blocks to get separate fields
    pot_pfs = [np.split(pot_pf, np.where(np.diff(pot_pf) != 1)[0] + 1)
               if len(pot_pf) > 0 else [] for pot_pf in pot_place_idx]

    return pot_pfs


def apply_pf_criteria(trace: np.ndarray, place_blocks: List[np.ndarray], trans: np.ndarray, params: dict,
                      sess_key: dict) -> List[Tuple[np.ndarray, bool, bool, bool]]:
    """
    Applies the criteria of place fields to potential place fields of a single neuron. A place field is accepted when...
        1) it stretches at least 'min_bin_size' bins (default 10)
        2) its mean dF/F is larger than outside the field by a factor of 'fluo_infield'
        3) during 'trans_time'% of the time the mouse is in the field, the signal consists of significant transients
    Place fields that pass these criteria have to have a p-value < 0.05 to be fully accepted. This is checked in
    the bootstrap() function.

    Args:
        trace: Spatially binned dF/F trace in which the potential place fields are located, shape (n_bins,)
        place_blocks: Bin indices of potential place fields, one array per field
        trans: Transient-only dF/F trace from the current neuron, shape (n_frames_in_session)
        params: Current hheise_placecell.PlaceCellParameter() entry
        sess_key: Primary keys of the current hheise_placecell.PlaceCells make() call

    Returns:
        List of results, Tuple with (place_field_idx, criterion1_result, criterion2_result, criterion3_result)
    """
    results = []
    for pot_place in place_blocks:
        bin_size = is_large_enough(pot_place, params['min_bin_size'])
        intensity = is_strong_enough(trace, pot_place, place_blocks, params['fluo_infield'])
        transients = has_enough_transients(trans, pot_place, sess_key, params['trans_time'])

        results.append((pot_place, bin_size, intensity, transients))

    return results


def is_large_enough(place_field: np.ndarray, min_bin_size: int) -> bool:
    """
    Checks if the potential place field is large enough according to 'min_bin_size' (criterion 1).

    Args:
        place_field:    1D array of potential place field indices
        min_bin_size:   Minimum number of bins for a place field

    Returns:
        Flag whether the criterion is passed or not (place field larger than minimum size)
    """
    return place_field.size >= min_bin_size


def is_strong_enough(trace: np.ndarray, place_field: np.ndarray, all_fields: List[np.ndarray],
                     fluo_factor: float) -> bool:
    """
    Checks if the place field has a mean dF/F that is 'fluo_infield'x higher than outside the field (criterion 2).
    Other potential place fields are excluded from this analysis.

    Args:
        trace: 1D array of the trace data, shape (n_bins,)
        place_field: 1D array of indices of data points that form the potential place field
        all_fields: 1D array of indices of all place fields in this trace
        fluo_factor: Threshold factor of mean dF/F in the place field compared to outside the field

    Returns:
        Flag whether the criterion is passed or not (place field more active than rest of trace)
    """
    pot_place_idx = np.in1d(range(trace.shape[0]), place_field)  # get an idx mask for the potential place field
    all_place_idx = np.in1d(range(trace.shape[0]), np.concatenate(all_fields))  # get an idx mask for all place fields
    return np.mean(trace[pot_place_idx]) >= fluo_factor * np.mean(trace[~all_place_idx])


def has_enough_transients(trans: np.ndarray, place_field: np.ndarray, key: dict, trans_time: float) -> bool:
    """
    Checks if of the time during which the mouse is located in the potential field, at least 'trans_time'%
    consist of significant transients (criterion 3).

    Args:
        trans: Transient-only trace of the current neuron, shape (n_frames_in_session,)
        place_field: 1D array of indices of data points that form the potential place field
        key: Primary keys of the current hheise_placecell.PlaceCells make() call
        trans_time: Fraction of the time spent in the place field that should consist of significant transients.

    Returns:
        Flag whether the criterion is passed or not (place field consists of enough significant transients)
    """

    trial_mask = (hheise_placecell.PCAnalysis & key).fetch1('trial_mask')
    frames_per_bin, running_masks = (hheise_placecell.Synchronization.VRTrial & key).fetch('aligned_frames',
                                                                                           'running_mask')

    place_frames_trace = []  # stores the trace of all trials when the mouse was in a place field as one data row
    for trial in np.unique(trial_mask):

        # Get frame indices of first and last place field bin
        frame_borders = (np.sum(frames_per_bin[trial][:place_field[0]]),
                         np.sum(frames_per_bin[trial][:place_field[-1]+1]))
        # Mask transient-only trace for correct trial and running-only frames (like frames_per_bin)
        trans_masked = trans[trial_mask == trial][running_masks[trial]]
        # Add frames that were in the bin to the list
        place_frames_trace.append(trans_masked[frame_borders[0]:frame_borders[1] + 1])

    # create one big 1D array that includes all frames during which the mouse was located in the place field.
    place_frames_trace = np.hstack(place_frames_trace)
    # check if at least 'trans_time' percent of the frames are part of a significant transient
    return np.sum(place_frames_trace) >= trans_time * place_frames_trace.shape[0]
