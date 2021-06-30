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
from schema import common_exp as exp
from schema import common_mice as mice
# from schema import common_img as img
from hheise_scripts import util

from datetime import datetime, timedelta
import ast
import os
from glob import glob
import numpy as np
import pandas as pd

schema = dj.schema('hheise_behav', locals(), create_tables=True)

# Hard-coded constant properties of encoder wheel
SAMPLE_RATE = 0.008                                 # sample rate of encoder in s
D_WHEEL = 10.5                                      # wheel diameter in cm
N_TICKS = 1436                                      # number of ticks in a full wheel rotation
DEG_DIST = (D_WHEEL * np.pi) / N_TICKS              # distance in cm the band moves for each encoder tick

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
    positions           : longblob          # Start and end of RZs in VR coordinates
    """
    contents = [
        ['none', [[-6, 4], [26, 36], [58, 68], [90, 100]]],
        ['none_shifted', [[-6, 4], [34, 44], [66, 76], [90, 100]]],
        ['training', [[-6, 4], [26, 36], [58, 68], [90, 100]]],
        ['training_shifted', [[-6, 4], [34, 44], [66, 76], [90, 100]]],
        ['novel', [[9, 19], [34, 44], [59, 69], [84, 94]]]
    ]

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
class VRSession(dj.Imported):
    definition = """ # Info about the VR Session
    -> exp.Session
    ---
    imaging_session     : tinyint           # bool flag whether imaging was performed during this session
    condition_switch    : longblob          # List of ints indicating trial numbers at which the condition switched
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
        pos                 : longblob          # 1d array with VR position sampled every 0.5 ms
        lick                : longblob          # 1d array with licks sampled every 0.5 ms
        frame               : longblob          # 1d array with frame triggers sampled every 0.5 ms
        enc                 : longblob          # 1d array with encoder data sampled every 0.5 ms
        valve               : longblob          # 1d array with valve openings (reward) sampled every 0.5 ms
        """

    def make(self, key):

        # Safety check that only my sessions are processed (should be restricted during the populate() call)
        if key['username'] != login.get_user():
            return

        # First, insert session information into the VRSession table
        self.insert_vr_info(key)

        # self.populate_trials(key)

    def insert_vr_info(self, key):
        """Fills VRSession table with basic info about the session, mainly from Excel file"""
        # Save original key
        new_key = key.copy()

        # Get current mouse
        mouse = (mice.Mouse & key).fetch1()

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
            mice.Weight().insert1({'username': key['username'], 'mouse_id': key['mouse_id'],
                                   'date_of_weight': key['day'], 'weight': sess_entry['weight [g]'].values[0]})

        # Get block and condition switch from session_notes string
        note_dict = ast.literal_eval((exp.Session & key).fetch1('session_notes'))
        new_key['block'] = note_dict['block']
        new_key['condition_switch'] = note_dict['switch']

        # Check if this is an imaging session (session has to be inserted into hheise_img.Scan() first)
        # if len(img.Scan & key).fetch() == 1:
        #     new_key['imaging_session'] = 1
        # else:
        #     new_key['imaging_session'] = 0
        new_key['imaging_session'] = 0

        # Insert final dict into the table
        self.insert1(new_key)

        ### FILL VRLOG INFO
        # Get filename of this session's LOG file (path is relative to the session directory)
        log_name = glob(os.path.join(login.get_neurophys_data_directory(),
                                     (exp.Session & key).fetch1('session_path'),
                                     'TDT LOG_*'))

        if len(log_name) == 0:
            raise Warning('No LOG file found for M{} session {}!'.format(mouse['mouse_id'], key['day']))
        elif len(log_name) > 1:
            raise Warning('{} LOG files found for M{} session {}!'.format(len(log_name), mouse['mouse_id'], key['day']))
        else:
            # Insert the filename (without path) into the responsible table
            row_dict = dict(key, log_filename=os.path.basename(log_name[0]))
            VRLogFile.insert1(row=row_dict, skip_duplicates=True)
            VRLog.make(self=VRLog, key=row_dict)     # Load the LOG file into the other table

    def populate_trials(self, key):
        """
        Find raw behavior files, insert filenames into respective tables, load data, align it and insert it into the
        VRTrial table.
        """

        def find_file(tstamp, file_list):
            """
            Finds a file with the same timestamp from a list of files.
            """
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

        # path to the session folder (includes behavioral .txt files)
        root = os.path.join(login.get_neurophys_data_directory(), (exp.Session & key).fetch1('session_path'))
        imaging = bool((VRSession & key).fetch1('imaging_session'))

        if imaging:
            enc_files = glob(root + r'\\**\\Encoder*.txt')
            pos_files = glob(root + r'\\**\\TCP*.txt')
            trig_files = glob(root + r'\\**\\TDT TASK*.txt')

        else:
            enc_files = glob(root + r'\\Encoder*.txt')
            pos_files = glob(root + r'\\TCP*.txt')
            trig_files = glob(root + r'\\TDT TASK*.txt')

        enc_files = util.numerical_sort(enc_files)
        pos_files = util.numerical_sort(pos_files)
        trig_files = util.numerical_sort(trig_files)

        if not len(enc_files) == len(pos_files) & len(enc_files) == len(trig_files):
            print(f'Uneven numbers of encoder, position and trigger files in folder {root}!')
            return

        counter = 1

        for enc_file in enc_files:
            # Find the files of the same trial in the session folder
            timestamp = int(os.path.basename(enc_file).split('_')[1][:-4])  # here the 0 in front of early times is removed
            pos_file = find_file(timestamp, pos_files)
            trig_file = find_file(timestamp, trig_files)
            if pos_file is None or trig_file is None:
                print(f'Could not find all three files for timestamp {timestamp}!')
                return

            trial_key = dict(key, part=counter)
            frame_count = None
            if imaging:
                frame_count = (img.RawImagingFile & trial_key).fetch1('frame_count')

            merge = self.align_behavior_files(trial_key, enc_file, pos_file, trig_file,
                                              imaging=imaging, frame_count=frame_count)

            if merge is not None:
                # save file (4 decimal places for time (0.5 ms), 2 dec for position, ints for lick, trigger, encoder)
                # TODO:
                #   - change sampling time to 8 ms
                #   - make "frame" and "valve" to event-times rather than continuous sampling
                #   - decide on raw or speed encoder saves, and make function to translate from one to the other
                trial_key['pos'] = merge[:, 1]
                trial_key['lick'] = merge[:, 2]
                trial_key['frame'] = merge[:, 3]
                trial_key['enc'] = merge[:, 4]
                trial_key['valve'] = merge[:, 6]

                if imaging:
                    file_path = os.path.join(str(Path(enc_file).parents[0]), f'merged_behavior_{str(timestamp)}.txt')
                else:
                    file_path = os.path.join(root, f'merged_behavior_{str(timestamp)}.txt')
                np.savetxt(file_path, merge, delimiter='\t',
                           fmt=['%.5f', '%.3f', '%1i', '%1i', '%1i', '%.2f', '%1i'],
                           header='Time\tVR pos\tlicks\tframe\tencoder\tcm/s\treward')

                # Insert name of behavior files into the database Todo not basename, but including trial folder if imaging
                RawEncFile.insert1(row=dict(key, trial_id=counter, enc_filename=os.path.basename(enc_file)))
                RawTCPFile.insert1(row=dict(key, trial_id=counter, enc_filename=os.path.basename(pos_file)))
                RawTDTFile.insert1(row=dict(key, trial_id=counter, enc_filename=os.path.basename(trig_file)))

                if verbose:
                    print(f'Done! \nSaving merged file to {file_path}...\n')

            else:
                print(f'Skipped trial {enc_file}, please check!')



            counter += 1
        print('Done!\n')

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
            print('Trial incomplete, please remove file!')
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
            if more_frames_in_TDT == 0 and verbose:
                print('Frame count matched, no correction necessary.')
            elif more_frames_in_TDT < 0:
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
                    if verbose:
                        print(f'Imported frame count missed {frames_to_prepend}, corrected by prepending them to the'
                              f'start of the file in 30Hz distances.')

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
                    if verbose:
                        print(f'Imported frame count missed {frames_to_prepend}, corrected by adding frames regularly'
                              f'at start and end of file.')
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

        # Resample data to encoder sampling rate (125 Hz), as encoder data is difficult to extrapolate
        df_list[0] = df_list[0].resample("8L").max().astype(int)             # 1 if any of grouped values are 1 avoids loosing frame
        df_list[1] = (df_list[1].resample("8L").mean() > 0.5).astype(int)    # 1 if half of grouped values are 1
        df_list[2] = df_list[2].resample("8L").sum()                         # sum up encoder values as it is a summed rotation value
        df_list[3] = df_list[3].resample("8L").ffill()                       # Forward fill missing position values

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

        # Set proper data types (float for position, int for the rest)
        merge_filt['trigger'] = merge_filt['trigger'].astype(int)
        merge_filt['licking'] = merge_filt['licking'].astype(int)
        merge_filt['encoder'] = merge_filt['encoder'].astype(int)
        merge_filt['water'] = merge_filt['water'].astype(int)

        # TODO: How to deal with the encoder artifact of "catching up" ticks from the ITI?
        # translate encoder data into velocity [cm/s] if enc_unit = 'speed'
        speed = -merge_filt.loc[:, 'encoder'] * DEG_DIST / SAMPLE_RATE  # speed in cm/s for each sample
        speed[speed == -0] = 0
        merge_filt['speed'] = speed

        # check frame count again
        merge_trig = np.sum(merge_filt['trigger'])
        if imaging and merge_trig != frame_count:
            print(f'Frame count matching unsuccessful: \n{merge_trig} frames in merge, should be {frame_count} frames.')
            return None

        # transform back to numpy array for saving
        time_passed = merge_filt.index - merge_filt.index[0]    # transfer timestamps to
        seconds = np.array(time_passed.total_seconds())         # time change in seconds
        array_df = merge_filt[['position', 'licking', 'trigger', 'encoder', 'speed', 'water']]  # change column order
        array = np.hstack((seconds[..., np.newaxis], np.array(array_df)))  # combine both arrays

        return array


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
        # Load LOG file
        log = pd.read_csv(os.path.join(login.get_neurophys_data_directory(),
                                       (exp.Session & key).fetch1('session_path'), key['log_filename']),
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
        self.insert1(insert_dict, allow_direct_insert=True)

    def get_dataframe(self):
        """Fetch LOG data of one session re-assembled as pandas DataFrame"""
        return pd.DataFrame({'log_time': self.fetch1('log_time'),
                             'log_trial': self.fetch1('log_trial'),
                             'log_event': self.fetch1('log_event')})

@schema
class Behavior(dj.Imported):
    definition = """ # Processed and merged TCP, TDT and Encoder data
    -> RawTCPFile
    -> RawTDTFile
    -> RawEncFile
    ---
    pos                 : longblob          # 1d array with VR position sampled every 0.5 ms
    lick                : longblob          # 1d array with licks sampled every 0.5 ms
    frame               : longblob          # 1d array with frame triggers sampled every 0.5 ms
    enc                 : longblob          # 1d array with encoder data sampled every 0.5 ms
    valve               : longblob          # 1d array with valve openings (reward) sampled every 0.5 ms
    """
    #todo: Try to reduce sampling rate to reduce file sizes?