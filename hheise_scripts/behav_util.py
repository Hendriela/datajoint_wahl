#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 02/11/2021 10:28
@author: hheise

Utility functions that deal with VR behavior data engineering.
"""
import login

login.connect()

import os
from glob import glob
import ast
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime, timedelta

from schema import hheise_behav, common_mice, common_exp, common_img

SAMPLE = 0.008  # hardcoded sample rate of merged behavioral data in seconds


def create_vrsession_entry(key: dict) -> dict:
    """
    Gather basic info about the session, mainly from Excel file, and return entry dict for VRSession table.
    The entry dict has to be returned instead of inserted here because DJ only lets you insert entries into Imported
    or Computed tables inside the make() function call.

    Args:
        key: Primary keys to query each entry of common_exp.Session() to be populated

    Returns:
        Complete entry for VRSession with primary keys as well as data about the session
    """
    # Save original key
    new_key = key.copy()

    # Get current mouse
    mouse = (common_mice.Mouse & key).fetch1()

    # Load info from the Excel file
    excel_path = os.path.join(login.get_neurophys_data_directory(),
                              (hheise_behav.BatchData & {"batch_id": mouse['batch']}).fetch1('behav_excel'))
    excel = pd.read_excel(excel_path, sheet_name="M{}".format(mouse['mouse_id']))
    # Day is returned as date, has to be cast as datetime for pandas comparison
    sess_entry = excel.loc[excel['Date'] == datetime(key['day'].year, key['day'].month, key['day'].day)]

    # Fill in info from Excel entry
    new_key['valve_duration'] = sess_entry['Water'].values[0].split()[1][:3]
    new_key['length'] = sess_entry['Track length'].values[0]
    new_key['running'] = sess_entry['Running'].values[0]
    new_key['licking'] = sess_entry['Licking'].values[0]
    new_key['deprivation'] = sess_entry['Deprivation'].values[0]
    new_key['vr_notes'] = sess_entry['Notes'].values[0]

    # Enter weight if given
    if not pd.isna(sess_entry['weight [g]'].values[0]):
        common_mice.Weight().insert1({'username': key['username'], 'mouse_id': key['mouse_id'],
                                      'date_of_weight': key['day'], 'weight': sess_entry['weight [g]'].values[0]})

    # Get block and condition switch from session_notes string
    note_dict = ast.literal_eval((common_exp.Session & key).fetch1('session_notes'))
    new_key['block'] = note_dict['block']
    new_key['condition_switch'] = eval(note_dict['switch'])  # eval turns string into list

    # Check if this is an imaging session (session has to be inserted into hheise_img.Scan() first)
    if len((common_img.Scan & key).fetch()) == 1:
        new_key['imaging_session'] = 1
    else:
        new_key['imaging_session'] = 0

    return new_key


def create_vrlogfile_entry(key: dict) -> dict:
    """
    Find LOG file for a session and return entry dict for VRLogFile() table.

    The entry dict has to be returned instead of inserted here because DJ only lets you insert entries into Imported
    or Computed tables inside the make() function call.

    Args:
        key: Primary keys to query each entry of VRSession() to be populated

    Returns:
        Complete entry for VRLogFile() with primary keys as well as the LOG file path
    """
    mouse_id = (common_mice.Mouse & key).fetch1('mouse_id')

    # Get filename of this session's LOG file (path is relative to the session directory)
    log_name = glob(os.path.join(login.get_neurophys_data_directory(),
                                 (common_exp.Session & key).fetch1('session_path'),
                                 'TDT LOG_*'))

    if len(log_name) == 0:
        raise Warning('No LOG file found for M{} session {}!'.format(mouse_id, key['day']))
    elif len(log_name) > 1:
        raise Warning('{} LOG files found for M{} session {}!'.format(len(log_name), mouse_id, key['day']))
    else:
        return dict(key, log_filename=os.path.basename(log_name[0]))


def align_behavior_files(trial_key: dict, enc_path: str, pos_path: str, trig_path: str, imaging: bool = False,
                         frame_count: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Align behavioral data from three text files to a common master time frame provided by LabView. Data are
    re-sampled at the rate of the encoder (125 Hz), as the encoder is a summed data collection and is difficult to
    resample.

    Args:
        trial_key:      Dict with primary keys for current trial to query LOG file from VRLog()
        enc_path:       Path to the Encoder.txt file (running speed)
        pos_path:       Path to the TCP.txt file (VR position)
        trig_path:      Path to the TDT.txt file (licking and frame trigger)
        imaging:        Bool flag whether the behavioral data is accompanied by an imaging movie (defaults to False)
        frame_count:    Frame count of the imaging movie (only needed if imaging=True)

    Returns:
        Aligned data in np.array with columns 'time', 'position', 'licking', 'trigger', 'encoder', 'speed', 'water'
        Returns None if a fault with the data is caught, exception is printed and trial is skipped
    """

    pd.options.mode.chained_assignment = None  # Disable false positive SettingWithCopyWarning

    # Load behavioral files
    encoder = np.loadtxt(enc_path)
    position = np.loadtxt(pos_path)
    trigger = np.loadtxt(trig_path)
    raw_trig = trigger.copy()

    try:
        # Separate licking and trigger signals (different start times)
        licking = trigger[:, :2].copy()
        licking[0, 0] = licking[0, 1]
        licking[0, 1] = encoder[0, 1]
        trigger = np.delete(trigger, 1, axis=1)
    except IndexError:
        # catch error if a file is empty
        print('File seems to be empty, alignment skipped.')
        return None

    data = [trigger, licking, encoder, position]

    if imaging and frame_count is None:
        print(f'Error in trial {enc_path}: provide frame count if imaging=True!')
        return None

    # check if the trial might be incomplete (VR not run until the end or TDT file incomplete)
    if max(position[:, 1]) < 110 or abs(position[-1, 0] - trigger[-1, 0]) > 2:
        print(f'Trial {trig_path} incomplete, please remove file!')
        with open(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\bad_trials.txt', 'a') as bad_file:
            out = bad_file.write(f'{trig_path}\n')
        return None

    ### check if a file was copied from the previous one (bug in LabView), if the start time stamp differs by >2s
    # transform the integer time stamps plus the date from the TDT file into datetime objects
    time_format = '%Y%m%d%H%M%S%f'
    date = trig_path.split('_')[-2]
    for f in data:
        if str(int(f[0, 0]))[4:] == '60000':
            f[0, 0] -= 1
    start_times = np.array([datetime.strptime(date + str(int(x[0, 0])), time_format) for x in data])

    # calculate absolute difference in seconds between the start times
    max_diff = np.max(np.abs(start_times[:, None] - start_times)).total_seconds()
    if max_diff > 2:
        print(f'Faulty trial (TDT file copied from previous trial), time stamps differed by {int(max(max_diff))}s!')
        with open(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\bad_trials.txt', 'a') as bad_file:
            out = bad_file.write(f'{trig_path}\tTDT from previous trial, diff {int(max(max_diff))}s\n')
        return None

    ### preprocess frame trigger signal
    if imaging:
        frames_to_prepend = 0
        # get a list of indices for every time stamp a frame was acquired
        trig_blocks = np.split(np.where(trigger[1:, 1])[0] + 1,
                               np.where(np.diff(np.where(trigger[1:, 1])[0]) != 1)[0] + 1)
        # take the middle of each frame acquisition as the unique time stamp of that frame, save trigger idx in a list
        trig_idx = []
        for block in trig_blocks:
            trigger[block, 1] = 0  # set the whole period to 0
            if np.isnan(np.mean(block)):
                print(f'No frame trigger in {trig_path}. Check file!')
                # return None, None
            trigger[int(round(np.mean(block))), 1] = 1
            trig_idx.append(int(round(np.mean(block))))

        # check if imported frame trigger matches frame count of .tif file and try to fix it
        more_frames_in_TDT = int(np.sum(trigger[1:, 1]) - frame_count)  # pos if TDT, neg if .tif had more frames
        if more_frames_in_TDT < 0:
            # first check if TDT has been logging shorter than TCP
            tdt_offset = position[-1, 0] - trigger[-1, 0]
            # if all missing frames would fit in the offset (time where tdt was not logging), print out warning
            if tdt_offset / 0.033 > abs(more_frames_in_TDT):
                print('TDT not logging long enough, too long trial? Check trial!')
            # if TDT file had too little frames, they are assumed to have been recorded before TDT logging
            # these frames are added after merge array has been filled
            frames_to_prepend = abs(more_frames_in_TDT)

        elif more_frames_in_TDT > 0:
            # if TDT included too many frames, its assumed that the false-positive frames are from the end of recording
            if more_frames_in_TDT < 5:
                for i in range(more_frames_in_TDT):
                    trigger[trig_blocks[-i], 1] = 0
            else:
                print(f'{more_frames_in_TDT} too many frames imported from TDT, could not be corrected!')
                with open(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\bad_trials.txt',
                          'a') as bad_file:
                    out = bad_file.write(f'{trig_path}\n')
                return None

        if frames_to_prepend > 0:
            first_frame = np.where(trigger[1:, 1] == 1)[0][0] + 1
            # first_frame_start = np.where(raw_trig[:, 2] == 1)[0][0]
            median_frame_time = int(np.median([len(frame) for frame in trig_blocks]))
            if median_frame_time > 70:
                median_frame_time = 66
            if first_frame > frames_to_prepend * median_frame_time:
                # if frames would fit in the merge array before the first recorded frame, prepend them with proper steps
                # make a list of the new indices (in steps of median_frame_time before the first frame)
                idx_start = first_frame - frames_to_prepend * median_frame_time
                # add 1 to indices because we do not count the first index of the trigger signal
                idx_list = np.arange(start=idx_start + 1, stop=first_frame, step=median_frame_time)
                if idx_list.shape[0] != frames_to_prepend:
                    print(f'Frame correction failed for {trig_path}!')
                    return None
                trigger[idx_list, 1] = 1

            # if frames dont fit, and less than 30 frames missing, put them in steps of 2 equally in the start and end
            elif frames_to_prepend < 30:
                for i in range(1, frames_to_prepend + 1):
                    if i % 2 == 0:  # for every even step, put the frame in the beginning
                        if trigger[i * 2, 1] != 1:
                            trigger[i * 2, 1] = 1
                        else:
                            trigger[i + 2 * 2, 1] = 1
                    else:  # for every uneven step, put the frame in the end
                        if trigger[-(i * 2), 1] != 1:
                            trigger[-(i * 2), 1] = 1
                        else:
                            trigger[-((i + 1) * 2), 1] = 1
            else:
                # correction does not work if the whole log file is not large enough to include all missing frames
                print(f'{int(abs(more_frames_in_TDT))} too few frames imported from TDT, could not be corrected.')
                with open(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch2\bad_trials.txt',
                          'a') as bad_file:
                    out = bad_file.write(f'{trig_path}\t{more_frames_in_TDT}\n')
                return None

    ### Preprocess position signal
    pos_to_be_del = np.arange(np.argmax(position[:, 1]) + 1,
                              position.shape[0])  # Get indices after the max position
    position = np.delete(position, pos_to_be_del,
                         0)  # remove all data points after maximum position (end of corridor)
    position[position[:, 1] < -10, 1] = -10  # cap position values to -10 and 110
    position[position[:, 1] > 110, 1] = 110
    data[3] = position  # Put preprocessed positions back into data list

    # Transform relative time to absolute time for all measurements
    fixed_times = []
    for idx, dataset in enumerate(data):
        timesteps = np.array([start_times[idx] + timedelta(seconds=x) for x in dataset[1:, 0]])
        fixed_times.append(np.array([timesteps, dataset[1:, 1]]).T)

    # Transform data to pandas dataframes with timestamps indices
    datasets = ['trigger', 'licking', 'encoder', 'position']
    df_list = []
    for name, dataset in zip(datasets, fixed_times):
        df = pd.DataFrame(dataset[:, 1], index=dataset[:, 0], columns=[name], dtype=float)
        df_list.append(df)

    ### Resample data to encoder sampling rate (125 Hz, every 8 ms), as encoder data is difficult to extrapolate

    # 1 if any of grouped values are 1 avoids loosing frame. Sometimes, even downsampling can create NaNs.
    # They are forward filled for now, which will fail (create an unwanted frame trigger) if the timepoint before
    # the NaN happens to be a frame trigger. Lets hope that never happens.
    df_list[0] = df_list[0].resample("8L").max().fillna(method='pad').astype(int)
    df_list[1] = (df_list[1].resample("8L").mean() > 0.5).astype(int)  # 1 if half of grouped values are 1
    df_list[2] = df_list[2].resample("8L").sum().astype(int)  # sum encoder, a summed rotation value
    df_list[3] = df_list[3].resample("8L").ffill()  # Forward fill missing position values

    # Sometimes pos is shifted during resampling and creates a NaN. In that case shift it back.
    if df_list[3].iloc[0].isna():
        df_list[3] = df_list[3].shift(periods=-1, fill_value=110)

    # Serially merge dataframes sorted by earliest data point (usually trigger) to not miss any data
    data_times = np.argsort(start_times)
    merge = pd.merge_asof(df_list[data_times[0]], df_list[data_times[1]], left_index=True, right_index=True)
    merge = pd.merge_asof(merge, df_list[data_times[2]], left_index=True, right_index=True)
    merge = pd.merge_asof(merge, df_list[data_times[3]], left_index=True, right_index=True)

    ### Get valve opening times from LOG file
    # Load LOG file TODO maybe make another boolean column with "being in reward zone"
    log = (hheise_behav.VRLog & trial_key).get_dataframe()
    if log is not None:
        # Filter out bad lines if Datetime column could not be parsed
        if log['log_time'].dtype == 'object':
            log = log.loc[~np.isnan(log['log_trial'])]
            log['log_time'] = pd.to_datetime(log['log_time'])
        # Extract data for the current trial based on the first and last times of the trigger timestamps
        trial_log = log.loc[(log['log_time'] > merge.index[0]) & (log['log_time'] < merge.index[-1])]
        # Get times when the valve opened
        water_times = trial_log.loc[trial_log['log_event'].str.contains('Dev1/port0/line0-B'), 'log_time']
        # Initialize empty water column and set to '1' for every water valve opening timestamp
        merge['water'] = 0
        for water_time in water_times:
            merge.loc[merge.index[merge.index.get_loc(water_time, method='nearest')], 'water'] = 1
    else:
        merge['water'] = -1

    # Delete rows before the first frame (don't delete anything if no frame trigger)
    if merge['trigger'].sum() > 0:
        first_frame = merge.index[np.where(merge['trigger'] == 1)[0][0]]
    else:
        first_frame = merge.index[0]
    merge_filt = merge[merge.index >= first_frame]

    # Fill in NaN values
    merge_filt['position'].fillna(-10, inplace=True)
    merge_filt['encoder'].fillna(0, inplace=True)
    merge_filt['licking'].fillna(0, inplace=True)

    # check frame count again
    merge_trig = np.sum(merge_filt['trigger'])
    if imaging and merge_trig != frame_count:
        print(f'Frame count matching unsuccessful: \n{merge_trig} frames in merge, should be {frame_count} frames.')
        return None

    # transform back to numpy array for saving
    time_passed = merge_filt.index - merge_filt.index[0]  # transfer timestamps to
    seconds = np.array(time_passed.total_seconds())  # time change in seconds
    array_df = merge_filt[['position', 'licking', 'trigger', 'encoder', 'water']]  # change column order
    array = np.hstack((seconds[..., np.newaxis], np.array(array_df)))  # combine both arrays

    return array


def compute_performances(curr_trial, data: np.ndarray, params: dict) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes lick, binned lick and stop performance of a single trial. Called during VRPerformance.populate().

    Args:
        curr_trial: Query of the current trial from VRSession.VRTrial()
        data: Behavioral data of this trial (time - lick - position - encoder)
        params: Current entry of PerformanceParameters()

    Returns:
        Three 1D arrays with different performance metrics: Binned lick ratio, lick count ratio, stop ratio
    """
    # TODO: Alternative approaches to calculate performances
    #  - only count the ONSET of licks per bin, not if a lick went over a bin border (especially impactful if a mouse is running fast while licking)
    #  - if one bin of a RZ is licked, count it for all bins of that RZ
    #  - check for mean distance of licked bins to the next RZ
    # Get reward zone borders for the current trial and add the buffer
    zone_borders = curr_trial.get_zone_borders()
    zone_borders[:, 0] -= params['vrzone_buffer']
    zone_borders[:, 1] += params['vrzone_buffer']

    # Find out which reward zones were passed (reward given) if parameter is set (default no)
    reward_from_merged = False
    if params['valve_for_passed']:
        rz_passed = np.zeros(len(zone_borders))
        for idx, zone in enumerate(zone_borders):
            # Get the reward entries at indices where the mouse is in the current RZ
            valve = curr_trial.fetch1('valve')
            rz_data = valve[np.where(np.logical_and(data[:, 2] >= zone[0], data[:, 2] <= zone[1]))]
            # Cap reward at 1 per reward zone (ignore possible manual water rewards given)
            if rz_data.sum() >= 1:
                rz_passed[idx] = 1
            else:
                rz_passed[idx] = 0

        passed_rz = rz_passed.sum() / len(zone_borders)
        reward_from_merged = True

    ### GET LICKING DATA ###
    # select only time point where the mouse licked
    lick_only = data[np.where(data[:, 1] == 1)]

    if lick_only.shape[0] == 0:
        lick_count_ratio = np.nan  # set nan, if there was no licking during the trial
        if not reward_from_merged:
            passed_rz = 0
    else:
        # remove continuous licks that were longer than 5 seconds
        diff = np.round(np.diff(lick_only[:, 0]) * 1000).astype(int)  # get an array of time differences in ms
        licks = np.split(lick_only, np.where(diff > SAMPLE * 1000)[0] + 1)  # split where difference > sample rate
        licks = [i for i in licks if i.shape[0] <= int(5 / SAMPLE)]  # only keep licks shorter than 5 seconds
        if len(licks) > 0:
            licks = np.vstack(licks)  # put list of arrays together to one array
            # out of these, select only time points where the mouse was in a reward zone
            lick_zone_only = []
            for zone in zone_borders:
                lick_zone_only.append(licks[(zone[0] <= licks[:, 2]) & (licks[:, 2] <= zone[1])])
            zone_licks = np.vstack(lick_zone_only)
            # the length of the zone-only licks divided by the all-licks is the zone-lick ratio
            lick_count_ratio = zone_licks.shape[0] / lick_only.shape[0]

            # correct by fraction of reward zones where the mouse actually licked
            if not reward_from_merged:
                passed_rz = len([x for x in lick_zone_only if len(x) > 0]) / len(zone_borders)
            lick_count_ratio = lick_count_ratio * passed_rz

            # # correct by the fraction of time the mouse spent in reward zones vs outside
            # rz_idx = 0
            # for zone in zone_borders:
            #     rz_idx += len(np.where((zone[0] <= data[:, 1]) & (data[:, 1] <= zone[1]))[0])
            # rz_occupancy = rz_idx/len(data)
            # lick_ratio = lick_ratio/rz_occupancy

        else:
            lick_count_ratio = np.nan
            if not reward_from_merged:
                passed_rz = 0

    ### GET BINNED LICKING PERFORMANCE
    licked_rz_bins = 0
    licked_nonrz_bins = 0
    bins = np.arange(start=-10, stop=111, step=1)  # create bin borders for position bins (2 steps/6cm per bin)
    zone_bins = []
    for zone in zone_borders:
        zone_bins.extend(np.arange(start=zone[0], stop=zone[1] + 1, step=params['bin_size']))
    bin_idx = np.digitize(data[:, 2], bins)
    # Go through all position bins
    for curr_bin in np.unique(bin_idx):
        # Check if there was any licking at the current bin
        if sum(data[np.where(bin_idx == curr_bin)[0], 1]) >= 1:
            # If yes, check if the bin is part of a reward zone
            if bins[curr_bin - 1] in zone_bins:
                licked_rz_bins += 1  # if yes, the current bin was RZ and thus correctly licked in
            else:
                licked_nonrz_bins += 1  # if no, the current bin was not RZ and thus incorrectly licked in
    try:
        # Ratio of RZ bins that were licked vs total number of licked bins, normalized by factor of passed RZs
        binned_lick_ratio = (licked_rz_bins / (licked_rz_bins + licked_nonrz_bins)) * passed_rz
    except ZeroDivisionError:
        binned_lick_ratio = 0

    ### GET STOPPING DATA ###
    # select only time points where the mouse was not running (from params (in cm/s) divided by encoder factor)
    stop_only = data[(-params['velocity_thresh'] / 2.87 <= data[:, 3]) &
                     (data[:, 3] <= params['velocity_thresh'] / 2.87)]
    # split into discrete stops
    diff = np.round(np.diff(stop_only[:, 0]) * 1000).astype(int)  # get an array of time differences in ms
    stops = np.split(stop_only, np.where(diff > SAMPLE * 1000)[0] + 1)  # split where difference > sample gap
    # select only stops that were longer than the specified stop time
    stops = [i for i in stops if i.shape[0] >= params['stop_time'] / (SAMPLE * 1000)]
    # select only stops that were inside a reward zone (min or max position was inside a zone border)
    zone_stop_only = []
    for zone in zone_borders:
        zone_stop_only.append([i for i in stops if zone[0] <= np.max(i[:, 1]) <= zone[1] or
                               zone[0] <= np.min(i[:, 1]) <= zone[1]])
    # the number of the zone-only stops divided by the number of the total stops is the zone-stop ratio
    zone_stops = np.sum([len(i) for i in zone_stop_only])
    stop_ratio = zone_stops / len(stops)

    return binned_lick_ratio, lick_count_ratio, stop_ratio


def compute_time_metrics(curr_trial, params: dict, timestamps: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean speed, running speed and trial duration of a single trial.

    Args:
        curr_trial: Query of the current trial from VRSession.VRTrial()
        params: Current entry of PerformanceParameters()
        timestamps: Time stamps of behavioral data points

    Returns:
        Three 1D arrays with different time metrics: mean speed, mean running speed and trial duration
    """

    # Get mean speed by taking track length / max time stamp. Slighty more accurate than mean(vel) because ITI
    # running is ignored, but included in vel
    time = max(timestamps)
    length = (hheise_behav.VRSession & curr_trial.restriction[0]).fetch1('length')
    mean_speed = length / time

    # Get mean running speed by filtering out time steps where mouse is stationary
    vel = curr_trial.enc2speed()  # Get velocity in cm/s
    running_vel = vel[vel >= params['velocity_thresh']]
    mean_running_speed = np.mean(running_vel)

    return mean_speed, mean_running_speed, time
