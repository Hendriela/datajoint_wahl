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


import datajoint as dj
import login
login.connect()

from schema import common_img, hheise_behav

schema = dj.schema('hheise_placecell', locals(), create_tables=True)


@schema
class PlaceCellParameter(dj.Manual):
    definition = """ # Parameters for place cell classification
    place_cell_id       : smallint              # index for unique parameter set, base 0
    ----
    description         : varchar(1024)         # Short description of the effect of this parameter set
    encoder_unit = 'raw': enum('raw', 'speed')  # Which value to use to determine resting frames (encoder data or cm/s)
    running_thresh = 3.0: float     # Running speed threshold under which a frame counts as "resting", calculated from
                                    # time points between the previous to the current frame.
                                    # If encoder_unit = 'raw', value is summed encoder data.
                                    # If encoder_unit = 'speed', value is average speed [cm/s].
    exclude_rest = 1    : tinyint   # bool flag whether immobile periods of the mouse should be included in analysis
    trans_length = 0.5  : float     # minimum length in seconds of a significant transient
    trans_thresh = 4    : tinyint   # factor of sigma above which a dF/F transient is considered significant
    bin_length = 5      : tinyint   # Spatial bin length for dF/F traces [cm]. Has to be divisor of track length. 
    bin_window_avg = 3  : tinyint   # half-size of symmetric sliding window of position bins for binned trace smoothing
    bin_base = 0.25     : float     # fraction of lowest bins that are averaged for baseline calculation
    place_thresh = 0.25 : float     # place field threshold, factor for difference between max and baseline dF/F
    min_pf_size = 15    : tinyint   # minimum size [cm] for a place field
    fluo_infield = 7    : tinyint   # threshold factor of mean DF/F in the place field compared to outside the field
    trans_time = 0.2    : float     # fraction of the (unbinned) signal while the mouse is located in the place field 
                                    # that should consist of significant transients
    split_size = 50     : int       # Number of frames in bootstrapping segments
    min_bin_size        : int       # Min_pf_size transformed into number of bins (rounded up). Calculated before 
                                    # insertion and raises an error if given by user.
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

        print('Populating TransientOnly for {}'.format(key))

        traces, unit_ids = (common_img.Segmentation & key).get_traces(include_id=True)
        params = (PlaceCellParameter & key).fetch1()

        # Enter master table entry
        self.insert1(key)

        # Create part table entries
        part_entries = []
        for i, unit_id in enumerate(unit_ids):

            # Get noise level of current neuron
            kernel = stats.gaussian_kde(traces[i])
            x_data = np.arange(min(traces[i]), max(traces[i]), 0.02)
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

        # Enter part-table entries
        self.ROI().insert(part_entries)


@schema
class Synchronization (dj.Computed):
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

        print('Populating Synchronization for {}'.format(key))

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
                    elif zero_idx < 79 and bin_frame_count[zero_idx+1] > 1:
                        bin_frame_count[zero_idx+1] -= 1
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

        print('Populating BinnedActivity for {}'.format(key))

        # Fetch activity traces and parameter sets
        traces, unit_ids = (common_img.Segmentation & key).get_traces(include_id=True)
        spikes = (common_img.Segmentation & key).get_traces(trace_type='decon')
        n_bins, trial_mask = (PCAnalysis & key).fetch1('n_bins', 'trial_mask')
        running_masks, bin_frame_counts = (Synchronization.VRTrial & key).fetch('running_mask', 'aligned_frames')
        n_trials = len(running_masks)

        # Create part table entries
        part_entries = []   # Store entries for each neuron in a list
        for unit_id, trace, spike in zip(unit_ids, traces, spikes):

            binned_trace = np.zeros((n_bins, n_trials))
            binned_spike = np.zeros((n_bins, n_trials))
            binned_spikerate = np.zeros((n_bins, n_trials))

            for trial_idx, (running_mask, bin_frame_count) in enumerate(zip(running_masks, bin_frame_counts)):
                # Create bin mask from frame counts
                bin_masks = []
                for idx, n_frames in enumerate(bin_frame_count):
                    bin_masks.append(np.full(n_frames, idx))
                bin_mask = np.concatenate(bin_masks)

                # Get section of current trial from the session-wide trace and filter out non-running frames
                trial_trace = trace[trial_mask == trial_idx][running_mask]
                trial_spike = spike[trial_mask == trial_idx][running_mask]

                # Iteratively for all bins, average trace and sum spike probabilities
                for bin_idx in range(n_bins):
                    bin_trace = trial_trace[bin_mask == bin_idx]
                    bin_spike = trial_spike[bin_mask == bin_idx]

                    if len(bin_trace):       # Test if there is data for the current bin, otherwise raise error
                        binned_trace[bin_idx, trial_idx] = np.mean(bin_trace)
                        # sum instead of mean (CASCADE's spike probability is cumulative)
                        binned_spike[bin_idx, trial_idx] = np.nansum(bin_spike)
                    else:
                        raise IndexError("Entry {}:\n\tNeuron {}, in {} returned empty array, "
                                         "could not bin trace.".format(key, unit_id, bin_idx))

                # Smooth average spike rate and transform values into mean firing rates by dividing by the time in s
                # occupied by the bin (from number of samples * sampling rate)

                # Todo: Discuss if smoothing the binned spikes across spatial bins (destroying temporal resolution) is
                #  actually necessary
                # smooth_binned_spike = gaussian_filter1d(binned_spike, 1)
                bin_times = bin_frame_count / (common_img.ScanInfo & key).fetch1('fr')
                binned_spikerate[:, trial_idx] = binned_spike[:, trial_idx]/bin_times

            part_entries.append(dict(**key, mask_id=unit_id,
                                     bin_activity=np.array(binned_trace, dtype=np.float32),
                                     bin_spikes=np.array(binned_spike, dtype=np.float32),
                                     bin_spikerate=np.array(binned_spikerate, dtype=np.float32)))

        # Enter master table entry
        self.insert1(key)

        # Enter part table entries
        self.ROI().insert(part_entries)

    def get_trial_avg(self, trace: str) -> np.array:
        """
        Compute trial-averaged VR position bin values for a given trace of one queried session.

        Args:
            trace: Trace type. Has to be attr of self.ROI(): bin_activity (dF/F), bin_spikes (spikes, decon),
                    bin_spikerate (spikerate).

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

        data = self.ROI().fetch(trace)          # Fetch requested data arrays from all neurons

        # Take average across trials (axis 1) and return array with shape (n_neurons, n_bins)
        return np.vstack([np.mean(x, axis=1) for x in data])



