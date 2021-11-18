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
    place_cell_id   : smallint  # index for unique parameter set, base 0
    ----
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
    """

    def helper_insert1(self, entry: dict) -> None:
        """
        Extended insert1() method that also creates a backup YAML file for every parameter set.

        Args:
            entry: Content of the new PlaceCellParameter() entry.
        """

        self.insert1(entry)

        full_entry = (self & entry).fetch1()  # Query full entry in case some default attributes were not set

        # TODO: remove hard-coding of folder location
        REL_BACKUP_PATH = "Datajoint/manual_submissions"

        identifier = f"placecell_{full_entry['place_cell_id']}_{full_entry['username']}_M{full_entry['mouse_id']}"

        # save dictionary in a backup YAML file for faster re-population
        filename = os.path.join(login.get_neurophys_wahl_directory(), REL_BACKUP_PATH, identifier + '.yaml')
        with open(filename, 'w') as outfile:
            yaml.dump(full_entry, outfile, default_flow_style=False)


@schema
class TransientOnly(dj.Computed):
    definition = """ # Transient-only thresholded traces of dF/F traces
    -> common_img.Segmentation
    -> PlaceCellParameter
    ------    
    time_transient = CURRENT_TIMESTAMP : timestamp   # automatic timestamp
    """

    class ROI(dj.Part):
        definition = """ # Data of single neurons
        -> TransientOnly
        mask_id : int        #  Mask index (as in Segmentation.ROI, base 0)
        -----
        sigma   : float      # Noise level of dF/F determined by FWHM of Gaussian KDE (from Koay et al. 2019)
        trans   : longblob   # 1d array with shape (n_frames,) with transient-only thresholded dF/F
        """

    def make(self, key: dict) -> None:
        """
        Automatically threshold dF/F for all traces of Segmentation.ROI() using FWHM from Koay et al. (2019)

        Args:
            key: Primary keys of the current Segmentation() entry.
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
        self.ROI.insert(part_entries)


@schema
class Synchronization (dj.Computed):
    definition = """ # Synchronized frame times binned to VR position
    -> hheise_behav.VRTrial
    -> PlaceCellParameter
    ------
    running_mask    : longblob      # Bool array with shape (n_frames), False if mouse was stationary during this frame
    aligned_frames  : longblob      # np.array with shape (n_bins), number of frames averaged in each VR position bin
    """

    def make(self, key: dict) -> None:
        """
        Align frame times with VR position and bin frames into VR position bins for each trial in a session.

        Args:
            key: Primary keys of the current Segmentation() entry.
        """

        # Load and calculate necessary parameters
        params = (PlaceCellParameter & key).fetch1()
        track_length = (hheise_behav.VRSession & key).fetch1('length')
        frame_count = (common_img.RawImagingFile & dict(**key, part=key['trial_id'])).fetch1('nr_frames')

        # The given attributes will be columns in the array, next to time stamp in col 0
        behavior = (hheise_behav.VRTrial & key).get_array(attr=['pos', 'lick', 'frame', 'enc', 'valve'])

        if params['encoder_unit'] == 'speed':
            curr_speed = (hheise_behav.VRTrial & key).enc2speed()

        if track_length % params['bin_length'] == 0:
            params['n_bins'] = int(track_length / params['bin_length'])
        else:
            raise Exception('Bin_length has to be a divisor of track_length!')

        # Make mask with length n_frames that is False for frames where the mouse was stationary (or True everywhere if
        # exclude_rest = 0).
        running_mask = np.ones(frame_count, dtype=bool)
        if params['exclude_rest']:
            frame_idx = np.where(behavior[:, 3] == 1)[0]  # find idx of all frames

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
                        behavior[frame_idx[i], 3] = np.nan
                elif params['encoder_unit'] == 'raw':
                    if np.sum(behavior[frame_idx[i - 1]:frame_idx[i], 4]) < params['running_thresh']:
                        # set index of mask to False (excluded in later analysis)
                        running_mask[i] = False
                        # set the bad frame in the behavior array to 0 to skip it during bin_frame_counting
                        behavior[frame_idx[i], 3] = np.nan
                else:
                    raise ValueError(f"Encoder unit {params['encoder_unit']} not recognized, behavior not aligned.")

        # Get frame counts for each bin for complete trial (moving and resting frames)
        bin_frame_count = np.zeros((params['n_bins']), 'int')

        # bin data in distance chunks
        bin_borders = np.linspace(-10, 110, params['n_bins'])
        idx = np.digitize(behavior[:, 1], bin_borders)  # get indices of bins

        # check how many frames are in each bin
        for i in range(params['n_bins']):
            bin_frame_count[i] = np.nansum(behavior[np.where(idx == i + 1), 3])

        # check that every bin has at least one frame in it
        if np.any(bin_frame_count == 0):
            all_zero_idx = np.where(bin_frame_count == 0)
            # if not, take a frame of the next bin. If the mouse is running that fast, the recorded calcium will lag
            # behind the actual activity in terms of mouse position, so spikes from a later time point will probably be
            # related to an earlier actual position. (or the previous bin in case its the last bin)
            for i in range(len(all_zero_idx[0])):
                zero_idx = (all_zero_idx[0][i], all_zero_idx[1][i])
                if zero_idx[0] == 79 and bin_frame_count[78, zero_idx[1]] > 1:
                    bin_frame_count[78, zero_idx[1]] -= 1
                    bin_frame_count[79, zero_idx[1]] += 1
                elif zero_idx[0] < 79 and bin_frame_count[zero_idx[0]+1, zero_idx[1]] > 1:
                    bin_frame_count[zero_idx[0]+1, zero_idx[1]] -= 1
                    bin_frame_count[zero_idx[0], zero_idx[1]] += 1
                else:
                    raise ValueError('No frame in these bins (#bin, #trial): {}'.format(*zip(zero_idx[0], zero_idx[1])))

        # Enter data into table
        self.insert1(dict(**key, running_mask=running_mask, aligned_frames=bin_frame_count))

