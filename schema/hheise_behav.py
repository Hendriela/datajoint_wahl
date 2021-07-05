#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 01/06/2021 13:25
@author: hheise

Schema to store behavioral data of Hendriks VR task
"""

import login
login.connect()
import datajoint as dj
from schema import common_exp, common_mice #common_img
from hheise_scripts import util

from datetime import datetime, timedelta
import ast
import os
from glob import glob
import numpy as np
import pandas as pd
import logging
from copy import deepcopy

schema = dj.schema('hheise_behav', locals(), create_tables=True)
# logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

SAMPLE = 0.008      # hardcoded sample rate of merged behavioral data in milliseconds

@schema
class BatchData(dj.Manual):
    definition = """ # Filename of the unprocessed LOG file for each session
    batch_id            : tinyint           # ID of the batch, corresponds to "batch" attribute in common_mice.Mouse
    ---
    behav_excel         : varchar(128)      # relative filepath (from neurophys data) of the behavior_evaluation Excel
    """


@schema
class CorridorPattern(dj.Lookup):
    definition = """ # Different types of VR corridor patterns and RZ locations
    pattern             : varchar(128)      # Description of pattern
    ---
    positions           : longblob          # 2D np.array of start and end of RZs in VR coordinates
    """
    contents = [
        ['none', np.array([[-6, 4], [26, 36], [58, 68], [90, 100]])],
        ['none_shifted', np.array([[-6, 4], [34, 44], [66, 76], [90, 100]])],
        ['training', np.array([[-6, 4], [26, 36], [58, 68], [90, 100]])],
        ['training_shifted', np.array([[-6, 4], [34, 44], [66, 76], [90, 100]])],
        ['novel', np.array([[9, 19], [34, 44], [59, 69], [84, 94]])]
    ]


@schema
class VRSession(dj.Imported):
    definition = """ # Info about the VR Session
    -> common_exp.Session
    ---
    imaging_session     : tinyint           # bool flag whether imaging was performed during this session
    condition_switch    : longblob          # List of ints indicating the first trial(s) of the new condition
    valve_duration      : smallint          # Duration of valve opening during reward in ms
    length              : smallint          # Track length in cm
    running             : enum('none', 'very bad', 'bad', 'medium', 'okay', 'good', 'very good') 
    licking             : enum('none', 'very bad', 'bad', 'medium', 'okay', 'good', 'very good') 
    deprivation         : varchar(256)      # Water deprivation status before the session
    block               : tinyint           # sessions of one batch can be sub-grouped in blocks (e.g. multiple strokes)
    vr_notes            : varchar(1024)     # Notes about the session
    """

    class VRTrial(dj.Part):
        definition = """ # Single trial of VR data
        -> VRSession
        trial_id            : tinyint           # Counter of the trial in 
        ---
        -> CorridorPattern
        tone                : tinyint           # bool flag whether the RZ tone during the trial was on (1) or off (0)
        pos                 : longblob          # 1d array with VR position sampled every 8 ms
        lick                : longblob          # 1d array with licks sampled every 8 ms
        frame               : longblob          # 1d array with frame triggers sampled every 8 ms
        enc                 : longblob          # 1d array with raw encoder ticks sampled every 8 ms
        valve               : longblob          # 1d array with valve openings (reward) sampled every 8 ms
        """

        def enc2speed(self):
            """Transform encoder ticks to speed in cm/s"""
            # Hard-coded constant properties of encoder wheel
            SAMPLE_RATE = 0.008  # sample rate of encoder in s
            D_WHEEL = 10.5  # wheel diameter in cm
            N_TICKS = 1436  # number of ticks in a full wheel rotation
            DEG_DIST = (D_WHEEL * np.pi) / N_TICKS  # distance in cm the band moves for each encoder tick

            # TODO: How to deal with the encoder artifact of "catching up" ticks from the ITI?
            # translate encoder data into velocity [cm/s]
            speed = self.fetch1('enc') * DEG_DIST / SAMPLE_RATE
            speed[speed == -0] = 0
            return speed

        def get_zone_borders(self):
            return deepcopy((self * CorridorPattern).fetch1('positions'))

    def make(self, key):

        # Safety check that only my sessions are processed (should be restricted during the populate() call)
        if key['username'] != login.get_user():
            return

        # print(f'Start to populate key: {key}')

        # First, create entries of VRSession and VRLogFile and insert them
        vrsession_entry = self.create_vrsession_entry(key)
        self.insert1(vrsession_entry, allow_direct_insert=True)

        vrlogfile_entry = self.create_vrlogfile_entry(key)
        VRLogFile().insert1(row=vrlogfile_entry, skip_duplicates=True)
        VRLog().helper_insert1(key=vrlogfile_entry, skip_duplicates=True)  # Load the LOG file into the other table

        # Then, create entries of single trials
        data = self.create_VRTrial_entries(key)
        # Enter the single trial entries one by one
        for i in range(len(data['trial'])):
            VRSession.VRTrial().insert1(data['trial'][i])
            RawEncFile().insert1(data['enc'][i])
            RawTDTFile().insert1(data['trig'][i])
            RawTCPFile().insert1(data['pos'][i])

    @staticmethod
    def create_vrsession_entry(key):
        """Gather basic info about the session, mainly from Excel file, and return entry dict for VRSession table."""
        # Save original key
        new_key = key.copy()

        # Get current mouse
        mouse = (common_mice.Mouse & key).fetch1()

        # Load info from the Excel file
        excel_path = os.path.join(login.get_neurophys_data_directory(),
                                  (BatchData & {"batch_id": mouse['batch']}).fetch1('behav_excel'))
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
        # if len(img.Scan & key).fetch() == 1:
        #     new_key['imaging_session'] = 1
        # else:
        #     new_key['imaging_session'] = 0
        new_key['imaging_session'] = 0

        return new_key

    @staticmethod
    def create_vrlogfile_entry(key):
        """Find LOG file for a session and return entry dict for VRLogFile table."""
        mouse_id = (common_mice.Mouse & key).fetch1('mouse_id')
        ### FILL VRLOG INFO
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

    def create_VRTrial_entries(self, key):
        """
        Find raw behavior files, load data, and align it. Return entry dicts for VRTrial, RawEncFile, RawTDTFile and
        RawTCPFile in a dict, each entry corresponding to a list of trial dicts for each table.
        """

        def find_file(tstamp, file_list):
            """Finds a file with the same timestamp from a list of files."""
            time_format = '%H%M%S'
            time_stamp = datetime.strptime(str(tstamp), time_format)
            matched_file = []
            for filename in file_list:
                curr_stamp = datetime.strptime(filename.split('_')[-1][:-4], time_format)
                diff = time_stamp - curr_stamp
                if abs(diff.total_seconds()) < 3:
                    matched_file.append(filename)
            if len(matched_file) == 0:
                print(f'No files with timestamp {tstamp} found in {root}!')
                return
            elif len(matched_file) > 1:
                print(f'More than one file with timestamp {tstamp} found in {root}!')
                return
            else:
                return matched_file[0]

        def get_behavior_file_paths(root_path, is_imaging):
            """Get lists of encoder, position and trigger files of all trials in this session."""
            if is_imaging:
                encoder_files = glob(root_path + r'\\**\\Encoder*.txt')
                position_files = glob(root_path + r'\\**\\TCP*.txt')
                trigger_files = glob(root_path + r'\\**\\TDT TASK*.txt')
            else:
                encoder_files = glob(root_path + r'\\Encoder*.txt')
                position_files = glob(root_path + r'\\TCP*.txt')
                trigger_files = glob(root_path + r'\\TDT TASK*.txt')

            encoder_files = util.numerical_sort(encoder_files)
            position_files = util.numerical_sort(position_files)
            trigger_files = util.numerical_sort(trigger_files)

            if (len(encoder_files) == len(position_files)) & (len(encoder_files) == len(trigger_files)):
                return encoder_files, position_files, trigger_files
            else:
                print(f'Uneven numbers of encoder, position and trigger files in folder {root_path}!')
                return

        # Get complete path of the current session
        root = os.path.join(login.get_neurophys_data_directory(), (common_exp.Session & key).fetch1('session_path'))

        # Find out if this session is an imaging session
        imaging = bool((self & key).fetch1('imaging_session'))

        # Get the task condition of this session
        cond = (common_exp.Session & key).fetch1('task')
        cond_switch = (self & key).fetch1('condition_switch')

        # Get paths of all behavior files in this session and process them sequentially
        enc_files, pos_files, trig_files = get_behavior_file_paths(root, imaging)
        counter = 1

        # Initialize dict that holds all entries
        data = {'trial': [], 'enc': [], 'pos': [], 'trig': []}

        for enc_file in enc_files:

            # Initialize relevant variables
            trial_key = dict(key, trial_id=counter)     # dict containing the values of all trial attributes
            frame_count = None                      # frame count of the imaging file of that trial
            if imaging:
                frame_count = (img.RawImagingFile & trial_key).fetch1('frame_count')

            # Find out which condition this trial was
            trial_key['pattern'], trial_key['tone'] = self.get_condition(trial_key, cond, cond_switch)

            ### ALIGN BEHAVIOR ###
            # Find the files of the same trial in the session folder
            timestamp = int(os.path.basename(enc_file).split('_')[1][:-4])  # here the 0 in front of early times is removed
            pos_file = find_file(timestamp, pos_files)
            trig_file = find_file(timestamp, trig_files)
            if pos_file is None or trig_file is None:
                print(f'Could not find all three files for timestamp {timestamp}!')
                # return

            merge = self.align_behavior_files(trial_key, enc_file, pos_file, trig_file,
                                              imaging=imaging, frame_count=frame_count)

            if merge is not None:
                # parse columns into entry dict
                # TODO: make "frame" and "valve" to event-times rather than continuous sampling
                trial_key['pos'] = merge[:, 1]
                trial_key['lick'] = merge[:, 2].astype(int)
                trial_key['frame'] = merge[:, 3].astype(int)
                trial_key['enc'] = -merge[:, 4].astype(int)         # encoder is installed upside down, so reverse sign
                trial_key['valve'] = merge[:, 5].astype(int)

                # Collect data of the current trial
                data['trial'].append(trial_key)
                data['enc'].append(dict(key, trial_id=counter, enc_filename=util.remove_session_path(key, enc_file)))
                data['pos'].append(dict(key, trial_id=counter, tcp_filename=util.remove_session_path(key, pos_file)))
                data['trig'].append(dict(key, trial_id=counter, tdt_filename=util.remove_session_path(key, trig_file)))

                counter += 1

        return data

    def align_behavior_files(self, trial_key, enc_path, pos_path, trig_path, imaging=False, frame_count=None):
        """
        Main function that aligns behavioral data from three text files to a common master time frame provided by
        LabView. Data are re-sampled at the rate of the encoder (125 Hz), as the encoder is a summed data collection and
        is difficult to resample.
        :param enc_path: str, path to the Encoder.txt file (running speed)
        :param pos_path: str, path to the TCP.txt file (VR position)
        :param trig_path: str, path to the TDT.txt file (licking and frame trigger)
        :param imaging: bool flag whether the behavioral data is accompanied by an imaging movie
        :param frame_count: int, frame count of the imaging movie (if imaging=True)
        :return: merge, np.array with columns '', 'position', 'licking', 'trigger', 'encoder', 'speed', 'water'
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
            return None, None

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
                first_frame_start = np.where(raw_trig[:, 2] == 1)[0][0]
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

        ### Resample data to encoder sampling rate (125 Hz), as encoder data is difficult to extrapolate

        # 1 if any of grouped values are 1 avoids loosing frame. Sometimes, even downsampling can create NaNs.
        # They are forward filled for now, which will fail (create an unwanted frame trigger) if the timepoint before
        # the NaN happens to be a frame trigger. Lets hope that never happens.
        df_list[0] = df_list[0].resample("8L").max().fillna(method='pad').astype(int)
        df_list[1] = (df_list[1].resample("8L").mean() > 0.5).astype(int)       # 1 if half of grouped values are 1
        df_list[2] = df_list[2].resample("8L").sum()                            # sum encoder, a summed rotation value
        df_list[3] = df_list[3].resample("8L").ffill()                          # Forward fill missing position values

        # Serially merge dataframes sorted by earliest data point (usually trigger) to not miss any data
        data_times = np.argsort(start_times)
        merge = pd.merge_asof(df_list[data_times[0]], df_list[data_times[1]], left_index=True, right_index=True)
        merge = pd.merge_asof(merge, df_list[data_times[2]], left_index=True, right_index=True)
        merge = pd.merge_asof(merge, df_list[data_times[3]], left_index=True, right_index=True)

        ### Get valve opening times from LOG file
        # Load LOG file TODO maybe make another boolean column with "being in reward zone"
        log = (VRLog & trial_key).get_dataframe()
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
        time_passed = merge_filt.index - merge_filt.index[0]    # transfer timestamps to
        seconds = np.array(time_passed.total_seconds())         # time change in seconds
        array_df = merge_filt[['position', 'licking', 'trigger', 'encoder', 'water']]  # change column order
        array = np.hstack((seconds[..., np.newaxis], np.array(array_df)))  # combine both arrays

        return array

    def is_session_novel(self, sess_key):
        """Checks whether the session of 'sess_key' is in the novel corridor (from VR zone borders)"""
        # Get event log
        log_events= (VRLog & sess_key).get_dataframe()['log_event']
        # Get the rounded position of the first reward zone
        rz_pos = int(np.round(float(log_events[log_events.str.contains('VR enter Reward Zone:')].iloc[0].split(':')[1])))
        if rz_pos == -6:
            return False
        elif rz_pos == 9:
            return True
        else:
            print(f'Could not determine context in session {self.key}!\n')

    def get_condition(self, key, task, condition_switch):
        """Test which condition this trial was"""

        # No condition switches in novel corridor
        if self.is_session_novel(key):
            return 'novel', 1

        # No condition switch or before first switch
        if (condition_switch == [-1]) or key['trial_id'] < condition_switch[0]:
            if ((task == 'Active') or (task == 'Passive')) or key['trial_id'] < condition_switch[0]:
                pattern = 'training'
                tone = 1
            else:
                raise Exception(f'Error at {key}:\nTask is not Active or Passive, but no condition switch given.')

        # One condition switch in this session, and the current trial is after the switch
        elif (len(condition_switch) == 1) and key['trial_id'] >= condition_switch[0]:
            if task == 'No tone':
                pattern = 'training'
                tone = 0
            elif task == 'No pattern':
                pattern = 'none'
                tone = 1
            elif task == 'Changed distances':
                pattern = 'training_shifted'
                tone = 1
            elif task == 'No reward at RZ3':
                pattern = 'training'
                tone = 1
            else:
                raise Exception('Error at {}:\n'
                                'Task condition could not be determined for trial nb {}.'.format(key, key['trial_id']))

        # Two condition switches, and the trial is after the first but before the second, or after the second switch
        elif task == 'No pattern and tone' and (len(condition_switch) == 2) and (key['trial_id'] >= condition_switch[1]):
            pattern = 'none'
            tone = 0
        elif task == 'No pattern and tone' and (len(condition_switch) == 2) and (key['trial_id'] < condition_switch[1]):
            pattern = 'none'
            tone = 1
        else:
            raise Exception('Error at {}:\n'
                            'Task condition could not be determined for trial nb {}.'.format(key, key['trial_id']))

        return pattern, tone


@schema
class RawTCPFile(dj.Manual):
    definition = """ # Filename of the unprocessed TCP (VR position) file for each trial
    -> VRSession.VRTrial
    ---
    tcp_filename        : varchar(128)      # filename of the TCP file
    """


@schema
class RawTDTFile(dj.Manual):
    definition = """ # Filename of the unprocessed TDT (licking and frame trigger) file for each trial
    -> VRSession.VRTrial
    ---
    tdt_filename        : varchar(128)      # filename of the TDT file
    """


@schema
class RawEncFile(dj.Manual):
    definition = """ # Filename of the unprocessed Encoder (running speed) file for each trial
    -> VRSession.VRTrial
    ---
    enc_filename        : varchar(128)      # filename of the Enc file
    """


@schema
class VRLogFile(dj.Manual):
    definition = """ # Filename of the unprocessed LOG file for each session
    -> VRSession
    ---
    log_filename        : varchar(128)      # filename of the LOG file (should be inside the session directory)
    """


@schema
class VRLog(dj.Imported):
    definition = """ # Processed LOG data
    -> VRLogFile
    ---
    log_time            : longblob          # np.array of time stamps (datetime64[ns])
    log_trial           : longblob          # np.array of trial numbers (int)
    log_event           : longblob          # np.array of event log (str)
    """

    def make(self, key):
        self.helper_insert1(key)

    def helper_insert1(self, key, skip_duplicates):
        # Load LOG file
        log = pd.read_csv(os.path.join(login.get_neurophys_data_directory(),
                                       (common_exp.Session & key).fetch1('session_path'), key['log_filename']),
                          sep='\t', parse_dates=[[0, 1]])

        # Validate mouse and track length info
        line = log['Event'].loc[log['Event'].str.contains("VR Task start, Animal:")].values[0]
        log_length = int(line.split('_')[1])
        log_mouse = int(line.split('_')[0].split()[-1][1:])
        tab_length = (VRSession & key).fetch1('length')
        if log_length != tab_length:
            raise Warning('Session {}:\nTrack length {} in LOG file does not correspond to length {} in '
                          'database.'.format(key, log_length, tab_length))
        if log_mouse != key['mouse_id']:
            raise Warning('Session {}: Mouse ID M{} in LOG file does not correspond to ID in '
                          'database M{}'.format(key['day'], log_mouse, key['mouse_id']))

        # Remove "log_filename" keyword that is not needed for this table (from an untouched copy of 'key')
        insert_dict = dict(key)
        insert_dict.pop('log_filename')

        # Parse fields as separate np.arrays
        insert_dict['log_time'] = np.array(log['Date_Time'], dtype=str)  # str because DJ doesnt like datetime[ns]
        insert_dict['log_trial'] = np.array(log['Trial'])
        insert_dict['log_event'] = np.array(log['Event'])
        self.insert1(insert_dict, allow_direct_insert=True, skip_duplicates=skip_duplicates)

    def get_dataframe(self):
        """Fetch LOG data of one session re-assembled as pandas DataFrame"""
        return pd.DataFrame({'log_time': self.fetch1('log_time'),
                             'log_trial': self.fetch1('log_trial'),
                             'log_event': self.fetch1('log_event')})


@schema
class PerformanceParameters(dj.Lookup):
    definition = """ # Different parameters for VR performance analysis
    perf_param_id       : tinyint       # ID of parameter sets
    ---
    vrzone_buffer       : tinyint       # number of position bins around the RZ that are still counted as RZ for licking
    valve_for_passed    : tinyint       # 0 or 1 whether valve openings should be used to compute number of passed RZs. 
                                        # More sensitive for well performing mice, but vulnerable against manual valve 
                                        # openings and useless for autoreward trials.
    bin_size            : tinyint       # size of position bins (in VR coordinates) for binned licking computation
    stop_time           : smallint      # time in ms of stationary encoder above which it the period counts as a "stop"
    """
    contents = [
        [0, 2, 0, 1, 100]
    ]


@schema
class VRPerformance(dj.Computed):
    definition = """ # Performance analysis data of VR behavior, one list per attribute/session with individ. trial data
    -> VRSession
    -> PerformanceParameters
    ---
    binned_lick_ratio           : longblob          # np.array of binned lick performance (how many positions bins, 
                                                    # where the mouse licked, were in a RZ, preferred metric)
    lick_count_ratio            : longblob          # np.array of lick count performance (how many individual licks were
                                                    # in a RZ, old metric)
    stop_ratio                  : longblob          # np.array of stops in RZs divided by total number of stops
    mean_speed                  : longblob          # np.array of mean velocity (in cm/s), basically track length/time
    mean_running_speed          : longblob          # np.array of mean running velocity (in cm/s), w/o non-moving times
    trial_duration              : longblob          # np.array of trial durations
    performance_trend           : float             # slope of regression line through trial performances (indicates if
                                                    # a mouse performed better/worse at the end vs start of a session)
    """

    def make(self, key):
        """
        Extracts behavior data from one merged_behavior.txt file (acquired through behavior_import.py).
        :param data: pd.DataFrame of the merged_behavior*.txt file
        :param novel: bool, flag whether file was performed in novel corridor (changes reward zone location)
        :param buffer: int, position bins around the RZ that are still counted as RZ for licking
        :param valid: bool, flag whether trial was a RZ position validation trial (training corridor with shifted RZs)
        :param bin_size: int, bin size in VR units for binned licking performance analysis (divisible by zone borders)
        :param use_reward: bool flag whether to use valve openings to calculate number of passed reward zones. More
                         sensitive for well performing mice, but vulnerable against manual valve openings and
                         useless for autoreward trials.
        :returns lick_ratio: float, ratio between individual licking bouts that occurred in reward zones div. by all licks
        :returns stop_ratio: float, ratio between stops in reward zones divided by total number of stops
        """

        # Get current set of parameters
        params = (PerformanceParameters & key).fetch1()

        # Process every trial (start with 1 because trial_id is 1-based
        for trial_id in range(1, len(VRSession.VRTrial & key)+1):

            # Store query of current trial
            curr_trial = (VRSession.VRTrial & key & 'trial_id={}'.format(trial_id))

            # Compute lick and stop performances
            binned_lick_ratio, lick_count_ratio, stop_ratio = self.compute_performances(curr_trial, params)

            # Compute time metrics

        return

    @staticmethod
    def compute_performances(curr_trial, params):
        """
        Computes lick, binned lick and stop performance of a single trial. Called during VRPerformance.populate().
        :param curr_trial: query of the current trial from VRSession.VRTrial()
        :param params: dict of current entry of PerformanceParameters()
        :return: binned_lick_ratio, lick_count_ratio, stop_ratio
        """
        # Get reward zone borders for the current trial and add the buffer
        zone_borders = curr_trial.get_zone_borders()
        zone_borders[:, 0] -= params['vrzone_buffer']
        zone_borders[:, 1] += params['vrzone_buffer']

        # Get relevant behavioral data of the current trial
        lick, pos, enc, valve = curr_trial.fetch1('lick', 'pos', 'enc', 'valve')

        # Find out which reward zones were passed (reward given) if parameter is set (default no)
        reward_from_merged = False
        if params['valve_for_passed']:
            rz_passed = np.zeros(len(zone_borders))
            for idx, zone in enumerate(zone_borders):
                # Get the reward entries at indices where the mouse is in the current RZ
                rz_data = valve[np.where(np.logical_and(pos >= zone[0], pos <= zone[1]))]
                # Cap reward at 1 per reward zone (ignore possible manual water rewards given)
                if rz_data.sum() >= 1:
                    rz_passed[idx] = 1
                else:
                    rz_passed[idx] = 0

            passed_rz = rz_passed.sum() / len(zone_borders)
            reward_from_merged = True

        # Get indices of proper columns and transform DataFrame to numpy array for easier processing
        time = np.arange(start=0, stop=len(lick) * SAMPLE, step=SAMPLE)
        data = np.vstack((time, lick, pos, enc)).T

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
        # select only time points where the mouse was not running (encoder between -2 and 2)
        stop_only = data[(-2 <= data[:, 3]) & (data[:, 3] <= 2)]
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

    def get_mean(self, attribute):
        """Get the mean of given attribute of the queried session(s)"""
        sess = self.fetch(attribute)
        means = [np.mean(x) for x in sess]
        if len(means) == 1:
            return means[0]
        else:
            return means