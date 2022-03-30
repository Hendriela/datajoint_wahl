#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 10/11/2021 15:06
@author: hheise

Schema that combines behavioral and imaging data of Hendriks VR task for place cell analysis
"""

# imports
import os
import yaml
import numpy as np
from scipy import stats
from typing import Optional, Tuple, Iterable

import datajoint as dj
import login

login.connect()

from schema import common_img, hheise_behav
from hheise_scripts import pc_classifier

schema = dj.schema('hheise_placecell', locals(), create_tables=True)


@schema
class PlaceCellParameter(dj.Manual):
    definition = """ # Parameters for place cell classification
    place_cell_id           : smallint              # index for unique parameter set, base 0
    ----
    description             : varchar(1024)         # Short description of the effect of this parameter set
    exclude_rest = 1        : tinyint   # bool flag whether immobile periods of the mouse should be excluded from analysis
    encoder_unit = 'raw'    : enum('raw', 'speed')  # Which value to use to determine resting frames (encoder data or cm/s)
    running_thresh = 3.0    : float     # Running speed threshold under which a frame counts as "resting", calculated from time points between the previous to the current frame. If encoder_unit = 'raw', value is summed encoder data. If encoder_unit = 'speed', value is average speed [cm/s].
    trans_length = 0.5      : float     # minimum length in seconds of a significant transient
    trans_thresh = 4        : tinyint   # factor of sigma above which a dF/F transient is considered significant
    bin_length = 5          : tinyint   # Spatial bin length for dF/F traces [cm]. Has to be divisor of track length. 
    bin_window_avg = 3      : tinyint   # half-size of symmetric sliding window of position bins for binned trace smoothing
    bin_base = 0.25         : float     # fraction of lowest bins that are averaged for baseline calculation
    place_thresh = 0.25     : float     # place field threshold, factor for difference between max and baseline dF/F
    min_pf_size = 15        : tinyint   # minimum size [cm] for a place field
    fluo_infield = 7        : tinyint   # threshold factor of mean DF/F in the place field compared to outside the field
    trans_time = 0.2        : float     # fraction of the (unbinned) signal while the mouse is located in the place field that should consist of significant transients
    split_size = 50         : int       # Number of frames in bootstrapping segments
    boot_iter = 1000        : int       # Number of shuffles for bootstrapping (default 1000, after Dombeck et al., 2010)
    min_bin_size            : int       # Min_pf_size transformed into number of bins (rounded up). Calculated before insertion and raises an error if given by user.
    """

    def helper_insert1(self, entry: dict) -> None:
        """
        Extended insert1() method that also creates a backup YAML file for every parameter set.

        Args:
            entry: Content of the new PlaceCellParameter() entry.
        """

        if (170 % entry['bin_length'] != 0) or (400 % entry['bin_length'] != 0):
            print("Warning:\n\tParameter 'bin_length' = {} cm is not a divisor of common track lengths 170 and 400 cm."
                  "\n\tProblems might occur in downstream analysis.".format(entry['bin_length']))

        if 'min_bin_size' not in entry:
            entry['min_bin_size'] = int(np.ceil(entry['min_pf_size'] / entry['bin_length']))
        else:
            raise KeyError("Parameter 'min_bin_size' will be calculated before insertion and should not be given by the"
                           "user!")

        self.insert1(entry)

        # Query full entry in case some default attributes were not set
        full_entry = (self & f"place_cell_id = {entry['place_cell_id']}").fetch1()

        # TODO: remove hard-coding of folder location
        REL_BACKUP_PATH = "Datajoint/manual_submissions"

        identifier = f"placecell_{full_entry['place_cell_id']}_{login.get_user()}"

        # save dictionary in a backup YAML file for faster re-population
        filename = os.path.join(login.get_neurophys_wahl_directory(), REL_BACKUP_PATH, identifier + '.yaml')
        with open(filename, 'w') as outfile:
            yaml.dump(full_entry, outfile, default_flow_style=False)


@schema
class PCAnalysis(dj.Computed):
    definition = """ # Session-wide parameters for combined VR and imaging analysis, like place cell analysis.  
    -> common_img.Segmentation
    -> PlaceCellParameter
    ------    
    n_bins      : tinyint   # Number of VR position bins to combine data. Calculated from track length and bin length.
    trial_mask  : longblob  # 1D bool array with length (nr_session_frames) holding the trial ID for each frame. 
    """

    def make(self, key: dict) -> None:
        """
        Compute metrics and parameters that are common for a whole session of combined VR imaging data.

        Args:
            key: Primary keys of the current Session() entry.
        """

        # print('Populating PCAnalysis for {}'.format(key))

        # Get current parameter set
        params = (PlaceCellParameter & key).fetch1()

        # Compute number of bins, depends on the track length and user-parameter bin_length
        track_length = (hheise_behav.VRSessionInfo & key).fetch1('length')
        if track_length % params['bin_length'] == 0:
            n_bins = int(track_length / params['bin_length'])
        else:
            raise Exception('Bin_length has to be a divisor of track_length!')

        ### Create trial_mask to split session-wide activity traces into trials

        # Get frame counts for all trials of the current session
        frame_count = (common_img.RawImagingFile & key).fetch('nr_frames')

        # Make arrays of the trial's length with the trial's ID and concatenate them to one mask for the whole session
        trial_masks = []
        for idx, n_frame in enumerate(frame_count):
            trial_masks.append(np.full(n_frame, idx))
        trial_mask = np.concatenate(trial_masks)

        # Enter data into table
        self.insert1(dict(**key, n_bins=n_bins, trial_mask=trial_mask))


@schema
class TransientOnly(dj.Computed):
    definition = """ # Transient-only thresholded traces of dF/F traces
    -> PCAnalysis
    ------
    time_transient = CURRENT_TIMESTAMP : timestamp   # automatic timestamp
    """

    class ROI(dj.Part):
        definition = """ # Data of single neurons
        -> TransientOnly
        mask_id : smallint   #  Mask index (as in Segmentation.ROI, base 0)
        -----
        sigma   : float      # Noise level of dF/F determined by FWHM of Gaussian KDE (from Koay et al. 2019)
        trans   : longblob   # 1d array with shape (n_frames,) with transient-only thresholded dF/F
        """

    def make(self, key: dict) -> None:
        """
        Automatically threshold dF/F for all traces of Segmentation.ROI() using FWHM from Koay et al. (2019)

        Args:
            key: Primary keys of the current PCAnalysis() (and by inheritance common_img.Segmentation()) entry.
        """

        # print('Populating TransientOnly for {}'.format(key))

        traces, unit_ids = (common_img.Segmentation & key).get_traces(include_id=True)
        params = (PlaceCellParameter & key).fetch1()

        # Create part table entries
        part_entries = []
        for i, unit_id in enumerate(unit_ids):

            # Get noise level of current neuron
            kernel = stats.gaussian_kde(traces[i])
            x_data = np.linspace(min(traces[i]), max(traces[i]), 1000)
            y_data = kernel(x_data)
            y_max = y_data.argmax()  # get idx of half maximum

            # get points above/below y_max that is closest to max_y/2 by subtracting it from the data and
            # looking for the minimum absolute value
            nearest_above = (np.abs(y_data[y_max:] - max(y_data) / 2)).argmin()
            nearest_below = (np.abs(y_data[:y_max] - max(y_data) / 2)).argmin()
            # get FWHM by subtracting the x-values of the two points
            fwhm = x_data[nearest_above + y_max] - x_data[nearest_below]
            # noise level is FWHM/2.3548 (https://en.wikipedia.org/wiki/Full_width_at_half_maximum)
            sigma = fwhm / 2.3548

            # Get time points where dF/F is above the threshold
            if sigma <= 0:
                raise ValueError('Sigma estimation of {} failed.'.format(dict(**key, mask_id=unit_id)))
            else:
                idx = np.where(traces[i] >= params['trans_thresh'] * sigma)[0]

            # Find blocks of sufficient length for a significant transient
            if idx.size > 0:
                blocks = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
                duration = int(params['trans_length'] / (1 / (common_img.ScanInfo & key).fetch1('fr')))
                try:
                    transient_idx = np.concatenate([x for x in blocks if x.size >= duration])
                except ValueError:
                    transient_idx = []
            else:
                transient_idx = []

            trans_only = traces[i].copy()
            select = np.in1d(range(trans_only.shape[0]), transient_idx)  # create mask for trans-only indices
            trans_only[~select] = 0  # set everything outside of this mask to 0

            new_part = dict(**key,
                            mask_id=unit_id,
                            sigma=sigma,
                            trans=trans_only)
            part_entries.append(new_part)

        # Enter master table entry
        self.insert1(key)

        # Enter part-table entries
        self.ROI().insert(part_entries)


@schema
class Synchronization(dj.Computed):
    definition = """ # Synchronized frame times binned to VR position of this session, trial data in part table
    -> PCAnalysis
    -> hheise_behav.VRSession
    ------
    time_sync = CURRENT_TIMESTAMP    : timestamp     # automatic timestamp
    """

    class VRTrial(dj.Part):
        definition = """ # Frame numbers aligned to VR position and spatially binned for individual trials
        -> Synchronization
        trial_id        : tinyint   # Counter of the trial in the session, same as RawImagingFile's 'part', base 0
        ---
        running_mask    : longblob  # Bool array with shape (n_frames), False if mouse was stationary during this frame
        aligned_frames  : longblob  # np.array with shape (n_bins), number of frames averaged in each VR position bin
        """

    def make(self, key: dict) -> None:
        """
        Align frame times with VR position and bin frames into VR position bins for each trial in a session.

        Args:
            key: Primary keys of the current PCAnalysis() entry.
        """

        # print('Populating Synchronization for {}'.format(key))

        # Load parameters and data
        params = (PlaceCellParameter & key).fetch1()
        params['n_bins'] = (PCAnalysis & key).fetch1('n_bins')
        trial_ids, frame_counts = (common_img.RawImagingFile & key).fetch('part', 'nr_frames')
        # The given attributes will be columns in the array, next to time stamp in col 0
        behavior = (hheise_behav.VRSession.VRTrial & key).get_arrays(attr=['pos', 'lick', 'frame', 'enc', 'valve'])

        trial_entries = []

        for trial_idx, trial_id in enumerate(trial_ids):
            # If necessary, translate encoder data to running speed in cm/s
            if params['encoder_unit'] == 'speed':
                curr_speed = (hheise_behav.VRSession.VRTrial & dict(**key, trial_id=trial_id)).enc2speed()

            # Make mask with length n_frames that is False for frames where the mouse was stationary (or True everywhere if
            # exclude_rest = 0).
            running_mask = np.ones(frame_counts[trial_idx], dtype=bool)
            if params['exclude_rest']:
                frame_idx = np.where(behavior[trial_idx][:, 3] == 1)[0]  # find idx of all frames

                # Because data collection starts at the first frame, there is no running data available before it.
                # Mice usually run at the beginning of the trial, so we assume that the frame is not stationary and just
                # skip the first frame and start with i=1.
                for i in range(1, len(frame_idx)):
                    # TODO: implement smoothing speed (2 s window) before removing (after Harvey 2009)
                    if params['encoder_unit'] == 'speed':
                        if np.mean(curr_speed[frame_idx[i - 1]:frame_idx[i]]) <= params['running_thresh']:
                            # set index of mask to False (excluded in later analysis)
                            running_mask[i] = False
                            # set the bad frame in the behavior array to 0 to skip it during bin_frame_counting
                            behavior[trial_idx][frame_idx[i], 3] = np.nan
                    elif params['encoder_unit'] == 'raw':
                        if np.sum(behavior[trial_idx][frame_idx[i - 1]:frame_idx[i], 4]) < params['running_thresh']:
                            # set index of mask to False (excluded in later analysis)
                            running_mask[i] = False
                            # set the bad frame in the behavior array to 0 to skip it during bin_frame_counting
                            behavior[trial_idx][frame_idx[i], 3] = np.nan
                    else:
                        raise ValueError(f"Encoder unit {params['encoder_unit']} not recognized, behavior not aligned.")

            # Get frame counts for each bin for complete trial (moving and resting frames)
            bin_frame_count = np.zeros((params['n_bins']), 'int')

            # bin data in distance chunks
            bin_borders = np.linspace(-10, 110, params['n_bins'])
            idx = np.digitize(behavior[trial_idx][:, 1], bin_borders)  # get indices of bins

            # check how many frames are in each bin
            for i in range(params['n_bins']):
                bin_frame_count[i] = np.nansum(behavior[trial_idx][np.where(idx == i + 1), 3])

            # check that every bin has at least one frame in it
            if np.any(bin_frame_count == 0):
                all_zero_idx = np.where(bin_frame_count == 0)[0]
                # if not, take a frame of the next bin. If the mouse is running that fast, the recorded calcium will lag
                # behind the actual activity in terms of mouse position, so spikes from a later time point will probably be
                # related to an earlier actual position. (or the previous bin in case its the last bin)
                for zero_idx in all_zero_idx:
                    # If the bin with no frames is the last bin, take one frame from the second-last bin
                    if zero_idx == 79 and bin_frame_count[78] > 1:
                        bin_frame_count[78] -= 1
                        bin_frame_count[79] += 1
                    # Otherwise, take it from the next bin, but only if the next bin has more than 1 frame itself
                    elif zero_idx < 79 and bin_frame_count[zero_idx + 1] > 1:
                        bin_frame_count[zero_idx + 1] -= 1
                        bin_frame_count[zero_idx] += 1
                    # This error is raised if two consecutive bins have no frames
                    else:
                        raise ValueError('Error in {}:\nNo frame in this bin, could not be corrected: {}'.format(key,
                                                                                                                 zero_idx))

            # Save trial entry for later combined insertion
            trial_entries.append(dict(**key, trial_id=trial_id, running_mask=running_mask,
                                      aligned_frames=bin_frame_count))

        # After all trials are processed, make entry into master table
        self.insert1(key)

        # And enter trial data into part table
        self.VRTrial().insert(trial_entries)


@schema
class BinnedActivity(dj.Computed):
    definition = """ # Spatially binned dF/F traces to VR position, one entry per session
    -> Synchronization
    ------
    time_bin_act = CURRENT_TIMESTAMP : timestamp   # automatic timestamp
    """

    class ROI(dj.Part):
        definition = """ # Data of single neurons, trials stacked as axis 1 (columns) in np.array
        -> BinnedActivity
        mask_id         : int       # Mask index (as in Segmentation.ROI, base 0)
        -----
        bin_activity   : longblob   # Array with shape (n_bins, n_trials), spatially binned single-trial dF/F trace
        bin_spikes     : longblob   # Same as bin_activity, but with estimated CASCADE spike probabilities
        bin_spikerate  : longblob   # Same as bin_activity, but with estimated CASCADE spikerates
        """

    def make(self, key: dict) -> None:
        """
        Spatially bin dF/F trace of every trial for each neuron and thus align it to VR position.

        Args:
            key: Primary keys of the current Synchronization() entry (one per session).
        """

        # from scipy.ndimage.filters import gaussian_filter1d

        # print('Populating BinnedActivity for {}'.format(key))

        # Fetch activity traces and parameter sets
        traces, unit_ids = (common_img.Segmentation & key).get_traces(include_id=True)
        spikes = (common_img.Segmentation & key).get_traces(trace_type='decon')
        n_bins, trial_mask = (PCAnalysis & key).fetch1('n_bins', 'trial_mask')
        running_masks, bin_frame_counts = (Synchronization.VRTrial & key).fetch('running_mask', 'aligned_frames')
        n_trials = len(running_masks)

        # Bin neuronal activity for all neurons
        binned_trace, binned_spike, binned_spikerate = pc_classifier.bin_activity_to_vr(traces, spikes, n_bins,
                                                                                        n_trials, trial_mask,
                                                                                        running_masks, bin_frame_counts,
                                                                                        key)

        # Create part entries
        part_entries = [dict(**key, mask_id=unit_id,
                             bin_activity=np.array(binned_trace[unit_idx], dtype=np.float32),
                             bin_spikes=np.array(binned_spike[unit_idx], dtype=np.float32),
                             bin_spikerate=np.array(binned_spikerate[unit_idx], dtype=np.float32))
                        for unit_idx, unit_id in enumerate(unit_ids)]

        # Enter master table entry
        self.insert1(key)

        # Enter part table entries
        self.ROI().insert(part_entries)

    def get_trial_avg(self, trace: str, trial_mask: Optional[np.ndarray] = None) -> np.array:
        """
        Compute trial-averaged VR position bin values for a given trace of one queried session.

        Args:
            trace: Trace type. Has to be attr of self.ROI(): bin_activity (dF/F), bin_spikes (spikes, decon),
                    bin_spikerate (spikerate).
            trial_mask: Optional boolean array which specifies which trials to include in the averaging. Used to
                    separate trials with condition switches. If not provided, all trials will be used.

        Returns:
            Numpy array with shape (n_neurons, n_bins) with traces averaged over queried trials (one session).
        """

        # Accept multiple inputs
        if trace in ['bin_activity', 'dff']:
            trace = 'bin_activity'
        elif trace in ['bin_spikes', 'spikes', 'decon']:
            trace = 'bin_spikes'
        elif trace in ['bin_spikerate', 'spikerate']:
            trace = 'bin_spikerate'
        else:
            raise ValueError('Trace has invalid value.\nUse bin_activity, bin_spikes or bin_spikerate.')

        # Check that only one entry has been queried
        if len(self) > 1:
            raise dj.errors.QueryError('You have to query a single session when computing trial averages. '
                                       f'{len(self)} sessions queried.')

        data = self.ROI().fetch(trace)  # Fetch requested data arrays from all neurons

        if trial_mask is None:
            trial_mask = np.ones(data[0].shape[1], dtype=bool)

        if len(trial_mask) != data[0].shape[1]:
            raise IndexError(f"Provided trial mask has {len(trial_mask)} entries, but traces have {data[0].shape[1]} trials.")
        else:
            # Take average across trials (axis 1) and return array with shape (n_neurons, n_bins)
            return np.vstack([np.mean(x[:, trial_mask], axis=1) for x in data])


@schema
class PlaceCell(dj.Computed):
    definition = """ # Place cell analysis and results (PC criteria mainly from Hainmüller (2018) and Dombeck/Tank lab)
    -> BinnedActivity
    -> TransientOnly
    corridor_type   : tinyint   # Code for including different corridors in one session in the analysis. 0=only standard corridor; 1=both; 2=only changed condition 1; 3=only changed condition 2
    ------
    place_cell_ratio                    : float         # Ratio of accepted place cells to total detected components
    time_place_cell = CURRENT_TIMESTAMP : timestamp     # automatic timestamp
    """

    class ROI(dj.Part):
        definition = """ # Data of ROIs that have place fields which passed all three criteria.
        -> PlaceCell
        mask_id         : int       # Mask index (as in Segmentation.ROI, base 0).
        -----
        is_place_cell   : int       # Boolean flag whether the cell is classified as a place cell (at least one place
                                    #  field passed all three criteria, and bootstrapping p-value is < 0.05).
        p               : float     # P-value of bootstrapping.
        """

    class PlaceField(dj.Part):
        definition = """ # Data of all place fields from ROIs with at least one place field that passed all 3 criteria.
        -> PlaceCell.ROI
        place_field_id  : int       # Index of the place field, base 0
        -----
        bin_idx         : longblob  # 1D array with the bin indices of the place field.
        large_enough    : tinyint   # Boolean flag of the 1. criterion (PF is large enough).
        strong_enough   : tinyint   # Boolean flag of the 2. criterion (PF is much stronger than the rest of the trace).
        transients      : tinyint   # Boolean flag of the 3. criterion (Time in PF consists of enough transients).
        """

    def make(self, key: dict) -> None:
        """
        Perform place cell classification on one session, after criteria from Hainmüller (2018) and Tank lab.
        Args:
            key: Primary keys of the current BinnedActivity() entry (one per session).
        """

        # print(f"Classifying place cells for {key}.")

        # Fetch data and parameters of the current session
        # traces = (BinnedActivity & key).get_trial_avg('bin_activity')  # Get spatially binned dF/F (n_cells, n_bins)
        mask_ids = (BinnedActivity.ROI & key).fetch('mask_id')
        trans_only = np.vstack((TransientOnly.ROI & key).fetch('trans'))  # Get transient-only dF/F (n_cells, n_frames)
        params = (PlaceCellParameter & key).fetch1()
        n_trials = len(common_img.RawImagingFile & key)

        # Check if the corridor condition changed during the session (validation trials), and trials have to be treated separately
        switch = (hheise_behav.VRSessionInfo & key).fetch1('condition_switch')

        trial_mask = np.ones(n_trials, dtype=bool)      # by default, all trials will be processed

        # No switch, include all trials
        if switch == [-1]:
            corridor_types = [0]
            trace_list = [(BinnedActivity & key).get_trial_avg('bin_activity')]
            accepted_trials = [None]

        # One condition switch occurred, process conditions separately
        elif len(switch) == 1:
            corridor_types = [0, 1, 2]          # corridor_type label of the following traces arrays (0=only normal, 1=all trials, 2=only changed condition 1, 3=only changed condition 2)
            trial_mask[switch[0]:] = False    # Exclude trials with different condition
            trace_list = [(BinnedActivity.ROI & key).fetch('mask_id', trial_mask=trial_mask),   # First array includes all normal trials (mask)
                          (BinnedActivity.ROI & key).fetch('mask_id'),                          # Second array includes all trials (no mask)
                          (BinnedActivity.ROI & key).fetch('mask_id', trial_mask=~trial_mask)]  # Third array includes all changed trials (inverse mask)
            accepted_trials = [np.where(trial_mask)[0],
                               None,
                               np.where(~trial_mask)[0]]

        # Two switches occurred
        elif len(switch) == 2:
            corridor_types = [0, 1, 2, 3]       # corridor_type label of the following traces arrays
            trial_mask[switch[0]:] = False      # Only normal trials
            cond1 = np.zeros(trial_mask.shape, dtype=bool)
            cond1[switch[0]:switch[1]] = True       # Only trials with no pattern
            cond2 = np.zeros(trial_mask.shape, dtype=bool)
            cond2[switch[1]:] = True       # Only trials with no tone and no pattern
            trace_list = [(BinnedActivity.ROI & key).fetch('mask_id', trial_mask=trial_mask),
                          (BinnedActivity.ROI & key).fetch('mask_id'),
                          (BinnedActivity.ROI & key).fetch('mask_id', trial_mask=cond1),
                          (BinnedActivity.ROI & key).fetch('mask_id', trial_mask=cond2)]
            accepted_trials = [np.where(trial_mask)[0],
                               None,
                               np.where(cond1)[0],
                               np.where(cond2)[0]]

        else:
            raise IndexError(f"Trial {key}:\nCondition switch {switch} not recognized.")

        # Make separate entries for each corridor type
        for corridor_type, traces, accepted_trial in zip(corridor_types, trace_list, accepted_trials):
            print('\tProcessing place cells for the following corridor type:', corridor_type)
            # Smooth binned data
            smooth = pc_classifier.smooth_trace(traces, params['bin_window_avg'])

            # Screen for potential place fields
            potential_pf = pc_classifier.pre_screen_place_fields(smooth, params['bin_base'], params['place_thresh'])

            passed_cells = {}
            # For each cell, apply place cell criteria on the potential place fields, and do bootstrapping if necessary
            for neuron_id, (neuron_pf, neuron_trace, neuron_trans_only) in enumerate(zip(potential_pf, smooth, trans_only)):

                # Apply criteria
                results = pc_classifier.apply_pf_criteria(neuron_trace, neuron_pf, neuron_trans_only, params, key,
                                                          accepted_trial)
                # If any place field passed all three criteria (bool flags sum to 3), save data for later bootstrapping
                if any([sum(entry[1:]) == 3 for entry in results]):
                    passed_cells[neuron_id] = results

            print(f"\t{len(passed_cells)} potential place cells found. Performing bootstrapping...")
            # Perform bootstrapping on all cells with passed place fields
            pc_traces = (common_img.Segmentation & key).get_traces()[np.array(list(passed_cells.keys()))]
            pc_trans_only = trans_only[np.array(list(passed_cells.keys()))]
            p_values = pc_classifier.perform_bootstrapping(pc_traces, pc_trans_only, accepted_trial, key,
                                                           n_iter=params['boot_iter'], split_size=params['split_size'])
            print(f"\tBootstrapping complete. {np.sum(p_values <= 0.05)} cells with p<=0.05.")
            # Prepare single-ROI entries
            pf_roi_entries = []
            pf_entries = []
            for idx, (cell_id, place_fields) in enumerate(passed_cells.items()):
                pf_roi_entries.append(dict(**key, corridor_type=corridor_type, mask_id=mask_ids[cell_id],
                                           is_place_cell=int(p_values[idx] <= 0.05), p=p_values[idx]))
                for field_idx, field in enumerate(place_fields):
                    pf_entries.append(dict(**key, corridor_type=corridor_type, mask_id=mask_ids[cell_id],
                                           place_field_id=field_idx, bin_idx=field[0], large_enough=int(field[1]),
                                           strong_enough=int(field[2]), transients=int(field[3])))

            # Insert entries into tables
            self.insert1(dict(**key, corridor_type=corridor_type, place_cell_ratio=np.sum(p_values < 0.05)/len(traces)))
            self.ROI().insert(pf_roi_entries)
            self.PlaceField().insert(pf_entries)

    def get_placecell_ids(self) -> np.ndarray:
        """
        Returns ROI IDs of accepted place cells from the queried entry(s)
        Returns:
            1D ndarray with the mask_id of accepted place cells (p < 0.5)
        """

        ids, p = self.ROI().fetch('mask_id', 'is_place_cell')
        return ids[np.array(p, dtype=bool)]
