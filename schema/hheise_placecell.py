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
from typing import Optional, Tuple, Iterable, Union
from scipy.ndimage import gaussian_filter1d

import datajoint as dj
import login

login.connect()

from schema import common_img, hheise_behav
from hheise_scripts import pc_classifier, data_util

schema = dj.schema('hheise_placecell', locals(), create_tables=True)


@schema
class PlaceCellParameter(dj.Manual):
    definition = """ # Parameters for place cell classification
    place_cell_id           : smallint              # index for unique parameter set, base 0
    ----
    description             : varchar(1024)         # Short description of the effect of this parameter set
    exclude_rest = 1        : tinyint   # bool flag whether immobile periods of the mouse should be excluded from analysis
    encoder_unit = 'speed'    : enum('raw', 'speed')  # Which value to use to determine resting frames (encoder data or cm/s)
    running_thresh = 5.0    : float     # Running speed threshold under which a frame counts as "resting", calculated from time points between the previous to the current frame. If encoder_unit = 'raw', value is summed encoder data. If encoder_unit = 'speed', value is average speed [cm/s].
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
    min_bin_size            : int       # Min_pf_size transformed into number of bins (rounded up). Depends on bin_length, and is calculated before insertion. Raises an error if given by user.
    sigma = 1               : tinyint   # Standard deviation of Gaussian kernel used to smooth activity trace.
    """

    def helper_insert1(self, entry: dict) -> None:
        """
        Extended insert1() method that also creates a backup YAML file for every parameter set.

        Args:
            entry: Content of the new PlaceCellParameter() entry.
        """

        if ('bin_length' in entry) and ((170 % entry['bin_length'] != 0) or (400 % entry['bin_length'] != 0)):
            print("Warning:\n\tParameter 'bin_length' = {} cm is not a divisor of common track lengths 170 and 400 cm."
                  "\n\tProblems might occur in downstream analysis.".format(entry['bin_length']))

        # set default values if not given
        if 'min_pf_size' not in entry:
            entry['min_pf_size'] = 15
        if 'bin_length' not in entry:
            entry['bin_length'] = 5

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

    def get_traces(self, include_id: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], None]:
        """
        Main function to get fluorescent traces in format (nr_traces, timepoints)
        Adrian 2020-03-16

        Args:
            trace_type      : Type of the trace: 'dff', 'trace' (absolute signal values), 'decon' (Cascade spike rates)
            include_id      : Flag to return a second argument with the ROI ID's of the returned signals
            decon_id        : Additional restriction, in case trace_type 'decon' is selected and multiple deconvolution
                                models have been run. In case of only one model, function selects this one.

        Returns:
            2D numpy array (nr_traces, timepoints): Fluorescent traces
            optional: 1D numpy array (nr_traces): Only if include_id==True, contains mask ID's of the rows i
        """

        # check if multiple caiman_ids are selected with self
        caiman_ids = self.fetch('caiman_id')
        if len(set(caiman_ids)) != 1:  # set returns unique entries in list
            raise Exception('You requested traces from more the following caiman_ids: {}\n'.format(set(caiman_ids)) + \
                            'Choose only one of them with & "caiman_id = ID"!')

        selected_rois = TransientOnly.ROI() & self

        traces_list = selected_rois.fetch('trans', order_by='mask_id')

        # some more sanity checks to catch common errors
        if len(traces_list) == 0:
            print('Warning: The query hheise_placecell.TransientOnly().get_traces() resulted in no traces!')
            return None
        # check if all traces have the same length and can be transformed into 2D array
        if not all(len(trace) == len(traces_list[0]) for trace in traces_list):
            raise Exception(
                'Error: The traces in traces_list had different lengths (probably from different recordings)!')

        traces = np.array([trace for trace in traces_list])  # (nr_traces, timepoints) array

        if not include_id:
            return traces

        else:  # include mask_id as well
            mask_ids = selected_rois.fetch('mask_id', order_by='mask_id')
            return traces, mask_ids

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
            bin_frame_count = np.zeros((params['n_bins']), dtype=int)
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
                    if zero_idx == params['n_bins']-1 and bin_frame_count[-2] > 1:
                        bin_frame_count[-2] -= 1
                        bin_frame_count[-1] += 1
                    # Otherwise, take it from the next bin, but only if the next bin has more than 1 frame itself
                    elif zero_idx < params['n_bins']-1 and bin_frame_count[zero_idx + 1] > 1:
                        bin_frame_count[zero_idx + 1] -= 1
                        bin_frame_count[zero_idx] += 1
                    # This error is raised if two consecutive bins have no frames
                    else:
                        raise ValueError('Error in {}:\nNo frame in this bin, '
                                         'could not be corrected: {}'.format(key, zero_idx))

            if not np.sum(bin_frame_count) == np.sum(running_mask):
                raise ValueError('Error in {}, trial {}:\nBinning failed, found {} running frames, but {} frames in '
                                 'bin_frame_count'.format(key, trial_idx, np.sum(running_mask), np.sum(bin_frame_count)))

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

        # print('Populating BinnedActivity for {}'.format(key))

        # Fetch activity traces and parameter sets
        traces, unit_ids = (common_img.Segmentation & key).get_traces(include_id=True)
        spikes = (common_img.Segmentation & key).get_traces(trace_type='decon')
        n_bins, trial_mask = (PCAnalysis & key).fetch1('n_bins', 'trial_mask')
        running_masks, bin_frame_counts = (Synchronization.VRTrial & key).fetch('running_mask', 'aligned_frames')

        # Bin neuronal activity for all neurons
        binned_trace, binned_spike, binned_spikerate = pc_classifier.bin_activity_to_vr(traces, spikes, n_bins,
                                                                                        trial_mask, running_masks,
                                                                                        bin_frame_counts, key)

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
            trial_mask: Optional boolean array with size (n_trials) which specifies which trials to include in the
                    averaging. Used to split trials with condition switches. If not provided, all trials will be used.

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

        data = (self.ROI() & self.restriction).fetch(trace)  # Fetch requested data arrays from all neurons

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
    corridor_type   : tinyint   # allows different corridors in one session in the analysis. 0=only standard corridor; 1=both; 2=only changed condition 1; 3=only changed condition 2
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

        print(f"Classifying place cells for {key}.")

        # Fetch data and parameters of the current session
        # traces = (BinnedActivity & key).get_trial_avg('bin_activity')  # Get spatially binned dF/F (n_cells, n_bins)
        mask_ids = (BinnedActivity.ROI & key).fetch('mask_id')
        trans_only = np.vstack((TransientOnly.ROI & key).fetch('trans'))  # Get transient-only dF/F (n_cells, n_frames)
        params = (PlaceCellParameter & key).fetch1()

        corridor_types, trace_list, accepted_trials = data_util.get_accepted_trials(key, 'bin_activity')

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

            # Perform bootstrapping on all cells with passed place fields
            if len(passed_cells) > 0:
                print(f"\t{len(passed_cells)} potential place cells found. Performing bootstrapping...")
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
            else:
                # Make dummy entries if no place fields passed all criteria
                p_values = np.ndarray([1])
                pf_roi_entries = None
                pf_entries = None

            # Insert entries into tables
            self.insert1(dict(**key, corridor_type=corridor_type, place_cell_ratio=np.sum(p_values < 0.05)/len(traces)))

            if len(passed_cells) > 0:
                self.ROI().insert(pf_roi_entries)
                self.PlaceField().insert(pf_entries)

    def get_placecell_ids(self) -> np.ndarray:
        """
        Returns ROI IDs of accepted place cells from the queried entry(s)
        Returns:
            1D ndarray with the mask_id of accepted place cells (p < 0.5)
        """

        ids, p = (self.ROI() & self.restriction).fetch('mask_id', 'is_place_cell')
        return ids[np.array(p, dtype=bool)]


@schema
class SpatialInformation(dj.Computed):
    definition = """ # Place cell classification using spatial information in deconvolved spikerates. Adapted from Shuman (2020).
    -> BinnedActivity
    corridor_type   : tinyint   # allows different corridors in one session in the analysis. 0=only standard corridor; 1=both; 2=only changed condition 1; 3=only changed condition 2
    ------
    place_cell_ratio            : float         # Ratio of accepted place cells to total detected components
    time_si = CURRENT_TIMESTAMP : timestamp     # automatic timestamp
    """

    class ROI(dj.Part):
        definition = """ # Spatial information data of all ROIs in the session.
        -> SpatialInformation
        mask_id         : int       # Mask index (as in Segmentation.ROI, base 0).
        -----
        si              : float     # Spatial information content per spike of this cell. Better performance than si_skaggs in detecting valid place cells.
        p_si            : float     # P-value of SI value after bootstrapping.
        si_skaggs       : float     # Total spatial information content of this cell's activity map. Not used for further analysis.
        p_si_skaggs     : float     # P-value of Skaggs' SI value after bootstrapping.
        stability       : float     # Within-session stability, derived from correlating trial traces. See compute_within_session_stability() for further details.
        p_stability     : float     # P-value of stability value after bootstrapping.
        place_fields    : longblob  # List with indices of consecutive high-activity bins. Empty if no bins were active enough.
        pf_threshold    : float     # Threshold to accept place fields (95th percentile of binned, trial-averaged, circularly shuffled activity).
        is_pc           : tinyint   # Boolean flag whether the cell passed all three criteria and is classified as a place cell.
        """

    def make(self, key):

        def compute_spatial_info(act_map: np.ndarray, occupancy: np.ndarray, sigma: int) -> Tuple[np.ndarray, np.ndarray]:
            """
            Computes total spatial information (Skaggs) and spatial information per spike (Shuman) for all cells.
            Formulae are from Skaggs (1992) mutual information and adapted to calcium imaging data from Shuman (2020).

            Args:
                act_map: Spatially binned spikerates with shape (n_cells, n_bins, n_trials). From
                    data_util.get_accepted_trials().
                occupancy: Frame counts of spatial bins with shape (n_trials, n_bins). From Synchronization.VRTrial().
                sigma: Standard deviation of Gaussian kernel for binned spikerate smoothing.

            Returns:
                Numpy array with shape (n_cells) with the spatial information values per cell after Skaggs and Shuman.
            """
            p_occ = np.sum(occupancy, axis=0) / np.sum(occupancy)  # Occupancy probability per bin p(i)
            p_occ = p_occ[None, :]
            act_bin = np.mean(gaussian_filter1d(act_map, sigma, axis=1), axis=2)  # Activity rate per bin lambda(i) (first smoothed)
            act_rel = act_bin.T / np.sum(p_occ * act_bin, axis=1)  # Normalized activity rate lambda(i) by lambda-bar
            skaggs = np.sum(p_occ * act_bin * np.log2(act_rel.T), axis=1)  # Skaggs computes total mutual info
            shuman = np.sum(p_occ * act_rel.T * np.log2(act_rel.T), axis=1)  # Shuman scales SI by activity level to make SI value more comparable between cells
            return skaggs, shuman

        def compute_within_session_stability(act_map: np.ndarray, sigma: int) -> np.ndarray:
            """
            Computes within-session stability of spikerates across trials after Shuman (2020). Trials are averaged and
            correlated across two timescales: First vs. second half of the session and even vs. odd trials. The Pearson
            correlation coefficients are Fisher z-scored to make them comparable, and their average is the stability
            value of the cell.

            Args:
                act_map: Spatially binned spikerates with shape (n_cells, n_bins, n_trials). From
                    data_util.get_accepted_trials().
                sigma: Standard deviation of Gaussian kernel for binned spikerate smoothing.

            Returns:
                Np.ndarray with shape (n_cells) with stability value of each cell.
            """
            smoothed = gaussian_filter1d(act_map, sigma, axis=1)
            # First, correlate trials in the first vs second half of the session
            half_point = int(np.round(smoothed.shape[2] / 2))
            first_half = np.mean(smoothed[:, :, :half_point], axis=2)
            second_half = np.mean(smoothed[:, :, half_point:], axis=2)
            r_half = np.vstack([np.corrcoef(first_half[x], second_half[x])[0, 1] for x in range(len(smoothed))])
            fisher_z_half = np.arctanh(r_half)

            # Then, correlate even and odd trials
            even = np.mean(smoothed[:, :, ::2], axis=2)
            odd = np.mean(smoothed[:, :, 1::2], axis=2)
            r_even = np.vstack([np.corrcoef(even[x], odd[x])[0, 1] for x in range(len(smoothed))])
            fisher_z_even = np.arctanh(r_even)

            # Within-session stability is the average of the two measures
            stab = np.mean(np.hstack((fisher_z_half, fisher_z_even)), axis=1)

            return stab

        def circular_shuffle(data: np.ndarray, t_mask: np.ndarray, r_mask: np.ndarray, occupancy: np.ndarray,
                             sess_key: dict, n_bins: int, n_iter: int) -> np.ndarray:
            """
            Performs circular shuffling of activity data to creating surrogate data for significance testing (after
            Shuman 2020). Each trial is circularly shifted by +/- trial length. Traces from adjacent trials shifts in
            and out of view at the ends. For the first and last trial, the trial trace is shifted inside itself.
            The shifted data is binned with the original occupancy data, so that each frame now is associated with a
            different position.

            Args:
                data: Raw, unbinned activity data with shape (n_cells, n_frames). Irrelevant sessions (other context)
                    has to be already removed. From common_img.Segmentation().
                t_mask: Trial mask with shape (n_frames) with accepted trial identity for every frame. From
                    PCAnalysis().
                r_mask: Boolean Running mask with shape (n_frames) with True for frames where the mouse was running.
                    From Synchronization().
                occupancy:  Frame counts of spatial bins with shape (n_trials, n_bins). From Synchronization.VRTrial().
                sess_key: Primary keys of the current session. Only used for error reporting.
                n_bins: Number of spatial bins in which the signal should be binned. From PlaceCellParameter().
                n_iter: Number of iterations of shuffling. From PlaceCellParameter

            Returns:
                Np.array with shape (n_iter, n_cells, n_bins, n_trials) with the shuffled activity data.
            """

            # Shuffled traces are stored in this array with shape (n_iter, n_rois, n_bins, n_trials)
            shuffle_data = np.zeros((n_iter, len(data), n_bins, len(r_mask))) * np.nan

            for shuff in range(n_iter):
                shift = []                  # The shifted traces for the current shuffle
                bf_counts = []              # Holds bin frame counts for accepted trials
                trial_mask_accepted = []    # Holds trial mask for accepted trials
                dummy_running_masks = []    # Hold running masks for accepted trials

                for rel_idx, trial_id in enumerate(np.unique(t_mask)):
                    curr_trace = data[:, t_mask == trial_id][:, r_mask[rel_idx]]
                    # Possible shifts are +/- half of the trial length (Aleksejs suggestion)
                    d = np.random.randint(-data.shape[1] // 2, data.shape[1] // 2 + 1)

                    dummy_running_masks.append(np.ones(curr_trace.shape[1], dtype=bool))

                    # Circularly shift traces
                    if trial_id == np.unique(t_mask)[0] or trial_id == np.unique(t_mask)[-1]:
                        # The first and last trials have to be treated differently: traces are circulated in-trial
                        shift.append(np.roll(curr_trace, d, axis=1))
                    else:
                        # For all other trials, we shift them together with the previous and next trial
                        prev_trial = data[:, t_mask == trial_id - 1]
                        next_trial = data[:, t_mask == trial_id + 1]
                        # Make the previous, current and next trials into one array
                        neighbor_trials = np.hstack((prev_trial, curr_trace, next_trial))
                        # Roll that array and take the values at the indices of the current trial
                        shift.append(np.roll(neighbor_trials, d, axis=1)[:, prev_trial.shape[1]:-next_trial.shape[1]])

                    # Add entries of bin_frame_counts and trial_mask for accepted trials
                    bf_counts.append(occupancy[rel_idx])
                    trial_mask_accepted.append(np.array([rel_idx] * curr_trace.shape[1], dtype=int))

                # The binning function requires the whole session in one row, so we stack the single-trial-arrays
                shift = np.hstack(shift)
                trial_mask_accepted = np.hstack(trial_mask_accepted)

                _, _, bin_shift = pc_classifier.bin_activity_to_vr(shift, shift, n_bins, trial_mask_accepted,
                                                                   dummy_running_masks, bf_counts, sess_key)
                shuffle_data[shuff] = bin_shift

            return shuffle_data

        print('Starting to populate SpatialInformation for', key)

        # Fetch relevant data
        corridor_types, trace_list, accepted_trials = data_util.get_accepted_trials(key, 'bin_spikerate', get_avg=False)
        running_masks, bin_frame_counts = (Synchronization.VRTrial & key).fetch('running_mask', 'aligned_frames')
        n_bins, trial_mask = (PCAnalysis & key).fetch1('n_bins', 'trial_mask')
        decon = (common_img.Segmentation & key).get_traces('decon')
        bin_frame_counts = np.vstack(bin_frame_counts)
        sigma_gauss, n_iter, min_bin_size = (PlaceCellParameter & key).fetch1('sigma', 'boot_iter', 'min_bin_size')

        # Process each corridor condition separately
        for corridor_type, traces, accepted_trial in zip(corridor_types, trace_list, accepted_trials):
            print('\tProcessing corridor type', corridor_type)
            # Restrict data and masks to accepted trials
            if accepted_trial is not None:
                curr_decon = decon[:, np.in1d(trial_mask, accepted_trial)]
                curr_trial_mask = trial_mask[np.in1d(trial_mask, accepted_trial)]
                curr_bin_frame_counts = bin_frame_counts[accepted_trial]
                curr_running_masks = running_masks[accepted_trial]
            else:
                curr_decon = decon
                curr_trial_mask = trial_mask
                curr_bin_frame_counts = bin_frame_counts
                curr_running_masks = running_masks

            ### SPATIAL INFORMATION ###
            # Compute SI of real data
            real_skaggs, real_shuman = compute_spatial_info(traces, curr_bin_frame_counts, sigma_gauss)
            real = [real_skaggs, real_shuman]

            # Perform circular shuffling and get SI of shuffled data
            shuffled_data = circular_shuffle(data=curr_decon, t_mask=curr_trial_mask, r_mask=curr_running_masks,
                                             occupancy=curr_bin_frame_counts, sess_key=key, n_bins=n_bins, n_iter=n_iter)

            shuffle_skaggs = np.zeros((shuffled_data.shape[0], shuffled_data.shape[1])) * np.nan
            shuffle_shuman = np.zeros((shuffled_data.shape[0], shuffled_data.shape[1])) * np.nan
            shuffles = [shuffle_skaggs, shuffle_shuman]

            for i, shuffle in enumerate(shuffled_data):
                s_skaggs, s_shuman = compute_spatial_info(shuffle, curr_bin_frame_counts, sigma_gauss)
                shuffles[0][i] = s_skaggs
                shuffles[1][i] = s_shuman

            # Find percentile -> SI of how many shuffles were higher than the real SI
            si_percs = [np.sum(shuf > r[None, :], axis=0) / n_iter for r, shuf in zip(real, shuffles)]

            ### WITHIN-SESSION STABILITY ###
            # Compute stability of real data
            real_stab = compute_within_session_stability(act_map=traces, sigma=sigma_gauss)

            # Perform circular shuffling and get stability of shuffled data
            shuffled_data = circular_shuffle(curr_decon, curr_trial_mask, curr_running_masks, curr_bin_frame_counts,
                                             key, n_bins, n_iter)
            shuffle_stab = np.zeros((shuffled_data.shape[0], shuffled_data.shape[1])) * np.nan
            for i, shuffle in enumerate(shuffled_data):
                shuffle_stab[i] = compute_within_session_stability(shuffle, sigma_gauss)

            # Find percentile -> stability of how many shuffles were higher than the real stability
            stab_perc = np.sum(shuffle_stab > real_stab[None, :], axis=0) / n_iter

            ### PLACE FIELD ACTIVITY ###
            # Create shuffled dataset again
            shuffled_data = circular_shuffle(curr_decon, curr_trial_mask, curr_running_masks, curr_bin_frame_counts,
                                             key, n_bins, n_iter)
            # Average data across trials
            shuffled_data = np.mean(shuffled_data, axis=3)
            # Get 95th percentile for each neuron's binned activity across shuffles
            perc95 = np.percentile(shuffled_data, 95, axis=(0,2))
            # Find bins with higher activity than perc95
            above_95 = np.mean(traces, axis=2) >= perc95[:, None]
            active_bin_coords = np.where(above_95)

            # Check for consecutive bins of at least "min_bin_size" size
            large_fields = [[] for i in range(len(traces))]
            for cell in np.unique(active_bin_coords[0]):
                curr_bin_idx = active_bin_coords[1][active_bin_coords[0] == cell]
                curr_bin_list = np.split(curr_bin_idx, np.where(np.diff(curr_bin_idx) != 1)[0]+1)
                large_fields[cell] = [x for x in curr_bin_list if len(x) >= min_bin_size]

            # Apply place cell criteria
            spatial_info = [perc < 0.05 for perc in si_percs]
            stability = stab_perc < 0.05
            place_fields = [len(x) > 0 for x in large_fields]
            # criteria_skaggs = np.vstack((spatial_info[0], stability, place_fields))
            criteria_shuman = np.vstack((spatial_info[1], stability, place_fields))
            # place_cell_skaggs = np.sum(criteria_skaggs, axis=0) == 3
            place_cell_shuman = (np.sum(criteria_shuman, axis=0) == 3).astype(int)

            # Create part entries
            entries = []
            for cell_id in range(len(traces)):
                entries.append(dict(**key,
                                    mask_id=cell_id,
                                    corridor_type=corridor_type,
                                    si=real_shuman[cell_id],
                                    p_si=si_percs[1][cell_id],
                                    si_skaggs=real_skaggs[cell_id],
                                    p_si_skaggs=si_percs[0][cell_id],
                                    stability=real_stab[cell_id],
                                    p_stability=stab_perc[cell_id],
                                    place_fields=large_fields[cell_id],
                                    pf_threshold=perc95[cell_id],
                                    is_pc=place_cell_shuman[cell_id]))

            # Insert master entry
            self.insert1(dict(**key, corridor_type=corridor_type,
                              place_cell_ratio=len(np.where(place_cell_shuman)[0]) / len(traces)))
            # Insert part entries
            self.ROI().insert(entries)

            ### Plotting code for checking data
            # fig, ax = plt.subplots(bin_shift[0].shape[1] + 2, sharex='all', sharey='all', figsize=(6, 9))
            # for i in range(bin_shift[0].shape[1] + 2):
            #
            #     if i == 0:
            #         ax[i].plot(np.mean(rate[0], axis=1))
            #         ax[i].set_ylabel('avg unshuffled')
            #     elif i == 8:
            #         ax[i].plot(np.mean(bin_shift[0], axis=1))
            #         ax[i].set_ylabel('avg shuffled')
            #     else:
            #         ax[i].plot(bin_shift[0, :, i-1])
            #     ax[i].spines['right'].set_visible(False)
            #     ax[i].spines['top'].set_visible(False)
            # plt.figure()
            # plt.plot(act_bin[0], label='binned activity')
            # plt.plot(act_rel.T[0], label='normalized activity')
            # plt.legend()
            # plt.twinx()
            # plt.plot(i[0], c='r', label='spatial info')
            # plt.plot(i_s[0], c='y', label='skaggs')
            # plt.legend()
            # mask_id, is_pc, p = (hheise_placecell.PlaceCell.ROI & key).fetch('mask_id', 'is_place_cell', 'p')
            # bin_avg_act = (hheise_placecell.BinnedActivity & key).get_trial_avg('bin_activity')
            # bin_avg_rate = gaussian_filter1d(np.mean(np.stack(bin_rate), axis=2), 1, axis=1)
            #
            # np.sum(is_pc)
            #
            # nrows = 8
            # ncols = 3
            #
            # fig, ax = plt.subplots(nrows, ncols, figsize=(23,13))
            # random_ids = np.sort(np.random.choice(has_si, nrows*ncols, replace=False))
            # i = 0
            # for row in range(nrows):
            #     for col in range(ncols):
            #         m_id = random_ids[i]
            #         ax[row, col].plot(bin_avg_act[m_id])
            #         twinx = ax[row, col].twinx()
            #         twinx.plot(bin_avg_rate[m_id], color='orange')
            #         ax[row, col].spines['right'].set_visible(False)
            #         ax[row, col].spines['top'].set_visible(False)
            #         if m_id in mask_id:
            #             ax[row, col].set_title('{} - SI: {:.2f}, p={} - PC_p: {}'.format(m_id, real[m_id], perc[m_id],
            #                                                                                p[mask_id==m_id][0]))
            #         else:
            #             ax[row, col].set_title('{} - SI: {:.2f}, p={} - PC_p: None'.format(m_id, real[m_id], perc[m_id]))
            #         i += 1
            # plt.tight_layout()

            # ### CHECK DIFFERENCES BETWEEN PLACE CELL CLASSIFICATION METHODS
            # diff_pc = np.where(place_cell_skaggs != place_cell_shuman)[0]
            # diff_pc = np.where(place_cell_orig != place_cell_shuman)[0]
            #
            # # Differences between Skaggs and Shuman method
            # for idx in diff_pc:
            #     fig, ax = plt.subplots(traces.shape[2]+1, 1, sharex='all', sharey='all', figsize=(7,8))
            #     fig.suptitle(f"{idx} - Skaggs {place_cell_skaggs[idx]}, Shuman {place_cell_shuman[idx]}, Orig {place_cell_orig[idx]}")
            #     for row in range(traces.shape[2]):
            #         ax[row].plot(traces[idx, :, row])
            #         ax[row].spines['right'].set_visible(False)
            #         ax[row].spines['top'].set_visible(False)
            #         for field in large_fields[idx]:
            #             ax[row].axvspan(field[0], field[-1], color='r', alpha=0.3)
            #     ax[-1].plot(np.mean(traces[idx], axis=1))
            #     ax[-1].spines['right'].set_visible(False)
            #     ax[-1].spines['top'].set_visible(False)
            #     for field in large_fields[idx]:
            #         ax[-1].axvspan(field[0], field[-1], color='r', alpha=0.3)
            #     plt.tight_layout()
            #     fname = r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Tests\data_analysis\spatial_info\M82_20210709_0_shuman_bartos_diff"
            #     plt.savefig(os.path.join(fname, f"{idx}.png"))
            #     plt.close()
            #
            # # Differences between SI and original PC-Classifier
            # mask_id, is_pc, p = (hheise_placecell.PlaceCell.ROI & key).fetch('mask_id', 'is_place_cell', 'p')
            # place_cell_orig = np.zeros(len(common_img.Segmentation.ROI & key), dtype=bool)
            # place_cell_orig[mask_id[is_pc == 1]] = True



