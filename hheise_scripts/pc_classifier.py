#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 07/12/2021 15:32
@author: hheise

Functions that constitute the place cell classification pipeline for hheise_placecell.PlaceCells.
"""
import numpy as np
from typing import List, Optional, Tuple, Iterable, Union
import random

from schema import common_img, hheise_placecell


def bin_activity_to_vr(traces: np.ndarray, spikes: np.ndarray, n_bins: int,
                       trial_mask: np.ndarray, running_masks: Union[np.recarray, list], bin_frame_counts: Iterable,
                       key: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spatially bins the dF/F and deconvolved traces of many neurons to the VR position. Extracted from
    BinnedActivity.make() because it is also used to bin the shuffled trace during place cell bootstrapping.

    Args:
        traces: dF/F traces with shape (n_neurons, n_frames_in_session), queried from common_img.Segmentation.ROI().
        spikes: CASCADE spike prediction of the trace, same shape and source as races.
        n_bins: Number of bins into which the trace should be binned, queried from PCAnalysis().
        trial_mask: 1D array with length n_frames_in_session, queried from PCAnalysis().
        running_masks: One element per trial, usually queried from Synchronization.VRTrial()
        bin_frame_counts: Same as running_masks, same as "aligned_frames" from Synchronization.VRTrial()
        key: Primary keys of the current query

    Returns:
        Three ndarrays with shape (n_neurons, n_bins, n_trials), spatially binned activity metrics for N neurons
            - dF/F, spikes, spikerate, fitting for entry of BinnedActivity.ROI().
    """

    n_trials = len(running_masks)

    binned_trace = np.zeros((traces.shape[0], n_bins, n_trials))
    binned_spike = np.zeros((spikes.shape[0], n_bins, n_trials))
    binned_spikerate = np.zeros((spikes.shape[0], n_bins, n_trials))

    for trial_idx, (running_mask, bin_frame_count) in enumerate(zip(running_masks, bin_frame_counts)):
        # Create bin mask from frame counts
        bin_masks = []
        for idx, n_frames in enumerate(bin_frame_count):
            bin_masks.append(np.full(n_frames, idx))
        bin_mask = np.concatenate(bin_masks)

        # Get section of current trial from the session-wide trace and filter out non-running frames
        trial_trace = traces[:, trial_mask == trial_idx][:, running_mask]
        trial_spike = spikes[:, trial_mask == trial_idx][:, running_mask]

        # Iteratively for all bins, average trace and sum spike probabilities
        for bin_idx in range(n_bins):
            bin_trace = trial_trace[:, bin_mask == bin_idx]
            bin_spike = trial_spike[:, bin_mask == bin_idx]

            if bin_trace.shape[1]:  # Test if there is data for the current bin, otherwise raise error
                binned_trace[:, bin_idx, trial_idx] = np.mean(bin_trace, axis=1)
                # sum instead of mean (CASCADE's spike probability is cumulative)
                binned_spike[:, bin_idx, trial_idx] = np.nansum(bin_spike, axis=1)
            else:
                if key is None:
                    key = {'-- No primary keys provided': '--'}
                raise IndexError("Entry {}:\n\t Bin {} returned empty array, could not bin trace.".format(key,
                                                                                                          bin_idx))

        # Smooth average spike rate and transform values into mean firing rates by dividing by the time in s
        # occupied by the bin (from number of samples * sampling rate)

        # Todo: Discuss if smoothing the binned spikes across spatial bins (destroying temporal resolution) is
        #  actually necessary
        # smooth_binned_spike = gaussian_filter1d(binned_spike, 1)
        bin_times = bin_frame_count / (common_img.ScanInfo & key).fetch1('fr')
        binned_spikerate[:, :, trial_idx] = binned_spike[:, :, trial_idx] / bin_times

    return binned_trace, binned_spike, binned_spikerate


def smooth_trace(trace: np.ndarray, bin_window: int) -> np.ndarray:
    """
    Smooths traces of neurons (usually binned, but can also be used unbinned) by averaging each time point across
    adjacent values.

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
    Performs pre-screening of potential place fields in traces. A potential place field is any bin/point that
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
                      sess_key: dict, accepted_trials: Optional[Iterable[int]] = None) \
        -> List[Tuple[np.ndarray, bool, bool, bool]]:
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
        accepted_trials: List of trial IDs that should be used. If None, all IDs will be used.

    Returns:
        List of results, Tuple with (place_field_idx, criterion1_result, criterion2_result, criterion3_result)
    """
    results = []
    for pot_place in place_blocks:
        bin_size = is_large_enough(pot_place, params['min_bin_size'])
        intensity = is_strong_enough(trace, pot_place, place_blocks, params['fluo_infield'])
        transients = has_enough_transients(trans, pot_place, sess_key, params['trans_time'], accepted_trials)

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


def has_enough_transients(trans: np.ndarray, place_field: np.ndarray, key: dict, trans_time: float,
                          accepted_trials: Optional[Iterable[int]]) -> bool:
    """
    Checks if of the time during which the mouse is located in the potential field, at least 'trans_time'%
    consist of significant transients (criterion 3).

    Args:
        trans: Transient-only trace of the current neuron, shape (n_frames_in_session,)
        place_field: 1D array of indices of data points that form the potential place field
        key: Primary keys of the current hheise_placecell.PlaceCells make() call
        trans_time: Fraction of the time spent in the place field that should consist of significant transients.
        accepted_trials: List of trial IDs that should be used. If None, all IDs will be used.

    Returns:
        Flag whether the criterion is passed or not (place field consists of enough significant transients)
    """

    trial_mask = (hheise_placecell.PCAnalysis & key).fetch1('trial_mask')
    frames_per_bin, running_masks = (hheise_placecell.Synchronization.VRTrial & key).fetch('aligned_frames',
                                                                                           'running_mask')

    place_frames_trace = []  # stores the trace of all trials when the mouse was in a place field as one data row
    for trial in np.unique(trial_mask):
        if (accepted_trials is None) or ((accepted_trials is not None) and (trial in accepted_trials)):
            # Get frame indices of first and last place field bin
            frame_borders = (np.sum(frames_per_bin[trial][:place_field[0]]),
                             np.sum(frames_per_bin[trial][:place_field[-1] + 1]))
            # Mask transient-only trace for correct trial and running-only frames (like frames_per_bin)
            trans_masked = trans[trial_mask == trial][running_masks[trial]]
            # Add frames that were in the bin to the list
            place_frames_trace.append(trans_masked[frame_borders[0]:frame_borders[1] + 1])

    # create one big 1D array that includes all frames during which the mouse was located in the place field.
    place_frames_trace = np.hstack(place_frames_trace)
    # check if at least 'trans_time' percent of the frames are part of a significant transient
    return np.sum(place_frames_trace) >= trans_time * place_frames_trace.shape[0]


def run_classifier(traces: np.ndarray, trans_only: np.ndarray, key: dict, accepted_trials: Optional[Iterable[int]],
                   params: Optional[dict] = None) -> dict:
    """
    Prepares binned and trial-averaged traces for place cell classification (smoothing) and runs classification by
    applying place field criteria. Called by hheise_placecell.PlaceCells().make() for initial classification and by
    perform_bootstrapping() for each shuffle iteration for validation.

    Args:
        traces:     2D array with shape (n_neurons, n_bins), spatially binned and trial-averaged traces. Can be queried
                    from hheise_placecell.BinnedActivity.get_trial_avg('bin_activity').
        trans_only: 2D array with shape (n_neurons, n_frames_in_session), transient-only traces from
                    hheise_placecell.TransientOnly().
        key:        Primary keys of the current BinnedActivity() entry.
        accepted_trials: List of trial IDs that should be used. If None, all IDs will be used.
        params:     Current parameter set, from hheise_placecell.PlaceCellParameter(). If not provided, function
                    attempts to query it itself.

    Returns:
        Dictionary with passed cell IDs as keys and a list with potential place fields and boolean flag of their passing
        the three criteria.
    """
    # Try to fetch parameters if they were not provided
    if params is None:
        params = (hheise_placecell.PlaceCellParameter & key).fetch1()

    # Smooth binned data
    smooth = smooth_trace(traces, params['bin_window_avg'])

    # Screen for potential place fields
    potential_pf = pre_screen_place_fields(smooth, params['bin_base'], params['place_thresh'])

    passed_cells = {}
    # For each cell, apply place cell criteria on the potential place fields, and do bootstrapping if necessary
    for neuron_id, (neuron_pf, neuron_trace, neuron_trans_only) in enumerate(zip(potential_pf, smooth, trans_only)):

        # Apply criteria
        results = apply_pf_criteria(neuron_trace, neuron_pf, neuron_trans_only, params, key, accepted_trials)
        # If any place field passed all three criteria (bool flags sum to 3), save data for later bootstrapping
        if any([sum(entry[1:]) == 3 for entry in results]):
            passed_cells[neuron_id] = results

    return passed_cells


def perform_bootstrapping(pc_traces: np.ndarray, pc_trans_only: np.ndarray, accepted_trials: Optional[Iterable[int]],
                          key: dict, n_iter: int, split_size: int) -> np.ndarray:
    """
    Performs bootstrapping on unbinned dF/F traces and returns p-values of putative place cells.
    For bootstrapping, the trace is divided in parts with 'split_size' length which are randomly shuffled n_iter times.
    Then, place cell detection is performed on each shuffled trace. The p-value is defined as the ratio of place cells
    detected in the shuffled traces versus number of shuffles (default 1000; see Dombeck et al., 2010). If this neuron's
    trace gets a p-value of p < 0.05 (place fields detected in less than 50 shuffles), the place field is accepted.

    Args:
        pc_traces: 2D array with shape (n_passed_cells, n_frames_in_session), unbinned dF/F traces of all ROIs that
            passed the initial PC classification and have to be validated. Queried from common_img.Segmentation.ROI().
        pc_trans_only: Same shape as pc_traces, contains transient-only traces of only passed ROIs from
            hheise_placecell.TransientOnly.ROI().
        accepted_trials: List of trial IDs that should be used. If None, all IDs will be used.
        key: Primary keys of the current BinnedActivity() entry. Passed down from hheise_placecell.PlaceCells().make().
        n_iter: Number of bootstrap iterations. Passed down from current PlaceCellParameter() entry.
        split_size: Size in frames of shuffle segments. Passed down like n_iter.

    Returns:
        1D array with shape (n_passed_cells,), holding p-values for all passed ROIs
    """

    n_bins, trial_mask = (hheise_placecell.PCAnalysis & key).fetch1('n_bins', 'trial_mask')
    running_masks, bin_frame_counts = (hheise_placecell.Synchronization.VRTrial & key).fetch('running_mask',
                                                                                             'aligned_frames')
    n_trials = len(running_masks)

    # This array keeps track of whether a shuffled trace passed the PC criteria
    p_counter = np.zeros(pc_traces.shape[0])
    for i in range(n_iter):
        # Shuffle pc_traces for every trial separately
        shuffle = []
        for trial_id in np.unique(trial_mask):
            if (accepted_trials is None) or ((accepted_trials is not None) and (trial_id in accepted_trials)):

                # TODO: When trials are ignored, "shuffle" only is appended with accepted trials and its length is reduced.
                #   Running masks are not, and thus indexing during binning does not work.
                #   Adjust algorithm to also split and shuffle "running_mask" in the same way as "trial".
                #   This problem likely appears also for "bin_frame_counts", so remove certain trials from it as well

                trial = pc_traces[:, trial_mask == trial_id]

                # divide the trial trace into splits of 'split_size' size and manually append the remainder
                clean_div_length = trial.shape[1] - trial.shape[1] % split_size
                split_trace = np.split(trial[:, :clean_div_length], clean_div_length / split_size, axis=1)
                split_trace.append(trial[:, clean_div_length:])

                # shuffle the list of trial-bits with random.sample, and concatenate to a new, now shuffled, array
                shuffle.append(np.hstack(random.sample(split_trace, len(split_trace))))

        # The binning function requires the whole session in one row, so we stack the single-trial-arrays
        shuffle = np.hstack(shuffle)

        # bin trials to VR position
        # parse shuffle twice as a spike rate and discard spike rate output
        bin_act, _, _ = bin_activity_to_vr(shuffle, shuffle, n_bins, n_trials, trial_mask, running_masks,
                                           bin_frame_counts, key)

        # Average binned activity across trials
        bin_avg_act = np.vstack([np.mean(x, axis=1) for x in bin_act])

        # Run the classifier on the shuffled traces
        passed_shuffles = run_classifier(bin_avg_act, pc_trans_only, key, accepted_trials)
        # Add 1 to the counter of each cell that passed the criteria
        if passed_shuffles:
            p_counter[np.array(list(passed_shuffles.keys()))] += 1

    return p_counter / n_iter  # return p-value of all neurons (ratio of accepted place fields out of n_iter shuffles)
