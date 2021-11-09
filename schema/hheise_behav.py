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
from schema import common_mice, common_exp, common_img

from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple, Dict
import os
import ast
from glob import glob
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

schema = dj.schema('hheise_behav', locals(), create_tables=True)
# logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

SAMPLE = 0.008  # hardcoded sample rate of merged behavioral data in seconds


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
    definition = """ # Info about the VR Session, mostly read from "behavioral evaluation" Excel file
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

    def make(self, key: dict) -> None:
        """
        Populates VRSession() for every entry of common_exp.Session().

        Args:
            key: Primary keys to query each entry of common_exp.Session() to be populated
        """
        # Safety check that only my sessions are processed (should be restricted during the populate() call)
        if key['username'] != login.get_user():
            return

        print(f'Start to populate key: {key}')

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

        # Check if this is an imaging session (session has to be inserted into common_img.Scan() first)
        if len((common_img.Scan & key).fetch()) == 1:
            new_key['imaging_session'] = 1
        else:
            new_key['imaging_session'] = 0

        self.insert1(new_key)


@schema
class RawBehaviorFile(dj.Imported):
    definition = """ # File names (relative to session folder) of raw VR behavior files (3 separate files per trial)
    -> VRSession
    trial_id            : smallint          # Counter for file sets (base 0)
    ---
    tdt_filename        : varchar(256)      # filename of the TDT file (licking and frame trigger)
    tcp_filename        : varchar(256)      # filename of the TCP file (VR position)
    enc_filename        : varchar(128)      # filename of the Enc file (running speed)
    """

    def make(self, key: dict) -> None:
        """
        Automatically looks up file names for behavior files of a single VRSession() entry.

        Args:
            key: Primary keys of the queried VRSession() entry.
        """

        print("Finding raw behavior files for session {}".format(key))

        # Check if RawImagingFile has already been filled for this session
        if len(common_img.RawImagingFile() & key) == 0:
            raise ImportError("No entries for session {} in common_img.RawImagingFile. Fill table before populating "
                              "RawBehaviorFile.")

        # Get complete path of the current session
        root = os.path.join(login.get_neurophys_data_directory(), (common_exp.Session & key).fetch1('session_path'))

        # Find all behavioral files; for now both file systems (in trial subfolder and outside)
        encoder_files = glob(root + '\\Encoder*.txt')
        encoder_files.extend(glob(root + '\\**\\Encoder*.txt'))
        position_files = glob(root + '\\TCP*.txt')
        position_files.extend(glob(root + '\\**\\TCP*.txt'))
        trigger_files = glob(root + '\\TDT TASK*.txt')
        trigger_files.extend(glob(root + '\\**\\TDT TASK*.txt'))

        # Sort them by time stamp (last part of filename, separated by underscore
        encoder_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        position_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        trigger_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

        ### FILTER OUT BAD TRIALS ###
        # Catch an uneven number of files
        if not (len(encoder_files) == len(position_files)) & (len(encoder_files) == len(trigger_files)):
            raise ImportError(f'Uneven numbers of encoder, position and trigger files in folder {root}!')

        # Catch different numbers of behavior files and raw imaging files
        if len(encoder_files) != len(common_img.RawImagingFile() & key):
            raise ImportError(f'Different numbers of behavior and imaging files in folder {root}!')

        # Check some common bugs of individual trials
        for i in range(len(encoder_files)):
            position = np.loadtxt(position_files[i])
            trigger = np.loadtxt(trigger_files[i])
            encoder = np.loadtxt(encoder_files[i])
            data = [trigger, position]

            if (len(position) == 0) or len(trigger) == 0 or len(encoder) == 0:
                raise ImportError("File in trial {} seems to be empty.".format(position_files[i]))

            # check if the trial might be incomplete (VR not run until the end or TDT file incomplete)
            if max(position[:, 1]) < 110 or abs(position[-1, 0] - trigger[-1, 0]) > 2:
                raise IndexError(f'Trial {trigger_files[i]} incomplete, please remove file!')

            # Check if a file was copied from the previous one (bug in LabView), if the start time stamp differs by >2s
            # transform the integer time stamps plus the date from the TDT file into datetime objects
            time_format = '%Y%m%d%H%M%S%f'
            date = trigger_files[i].split('_')[-2]
            for f in data:
                if str(int(f[0, 0]))[4:] == '60000':
                    f[0, 0] -= 1
            start_times = np.array([datetime.strptime(date + str(int(x[0, 0])), time_format) for x in data])

            # calculate absolute difference in seconds between the start times
            max_diff = np.max(np.abs(start_times[:, None] - start_times)).total_seconds()
            if max_diff > 2:
                raise ValueError(
                    f'Faulty trial (TDT file copied from previous trial), time stamps differed by {int(max(max_diff))}s!')

        # Manually curate sessions to weed out trials with e.g. buggy lick sensor
        bad_trials = self.plot_screening(trigger_files, encoder_files, key)

        if len(bad_trials) > 0:
            print("Session {}:\nThe following trials will be excluded from further analysis.\n"
                  "DELETE THE CORRESPONDING TIFF FILES!!".format(key))
            for index in sorted(bad_trials, reverse=True):
                print(trigger_files[index])
                del encoder_files[index]
                del trigger_files[index]
                del position_files[index]

        # If everything is fine, insert the behavioral file paths, relative to session folder, sorted by time
        session_path = (common_exp.Session() & key).get_absolute_path()
        for idx in range(len(encoder_files)):
            new_entry = dict(
                **key,
                trial_id=idx,
                tdt_filename=os.path.relpath(trigger_files[idx], session_path),
                tcp_filename=os.path.relpath(position_files[idx], session_path),
                enc_filename=os.path.relpath(encoder_files[idx], session_path)
            )

            # Last sanity check: Time stamps of the three files should not differ more than 2 seconds
            time_format = '%H%M%S'
            times = [datetime.strptime(new_entry['tdt_filename'].split('_')[-1][:-4], time_format),
                     datetime.strptime(new_entry['tcp_filename'].split('_')[-1][:-4], time_format),
                     datetime.strptime(new_entry['enc_filename'].split('_')[-1][:-4], time_format)]

            import itertools
            for subset in itertools.combinations(times, 2):
                if (subset[0] - subset[1]).seconds > 2:
                    raise ValueError("Files for trial {} do not have matching time stamps!".format(new_entry))

            self.insert1(new_entry)

        print("Done!")

    def load_data(self) -> Dict[str, np.ndarray]:
        """
        Loads data of a single queried trial and returns it in dict form.

        Returns:
            Data dict with keys "tdt", "tcp" and "enc", with np.ndarrays of raw behavioral data
        """

        # Get the session of the current query
        session_path = (common_exp.Session() & self).get_absolute_path()

        # Load data into a dict
        data = dict(
            tdt=np.loadtxt(os.path.join(session_path, self.fetch1('tdt_filename'))),
            tcp=np.loadtxt(os.path.join(session_path, self.fetch1('tcp_filename'))),
            enc=np.loadtxt(os.path.join(session_path, self.fetch1('enc_filename'))),
        )
        return data

    @staticmethod
    def plot_screening(tdt_list: List[str], enc_list: List[str], trial_key: dict) -> List[int]:
        """
        Displays the running and licking data of all trials of one session in an interactive pyplot. Each subplot is
        clickable, upon which it is marked red and added to a list of trials that should not be analysed further.
        Clicking again will turn the plot white again and remove the trial from that list. The sorted list of bad
        trial IDs is returned when the figure is closed.

        Args:
            tdt_list: Absolute file names of TDT files for the current session
            enc_list: Absolute file names of Encoder files for the current session
            trial_key: Primary keys of the queried VRSession() entry.

        Returns:
            List with trial IDs that should NOT be entered into CuratedBehaviorFile() and not be used further.
        """
        n_trials = len(tdt_list)
        bad_trials = []

        nrows = int(np.ceil(n_trials / 3))
        ncols = 3

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))
        count = 0

        for row in range(nrows):
            for col in range(ncols):
                if count < n_trials or (row == nrows and col == 0):
                    curr_ax = ax[row, col]

                    # Load the data, (ignore first time stamp)
                    curr_enc = -np.loadtxt(enc_list[count])[1:, 1]
                    curr_lick = np.loadtxt(tdt_list[count])[1:, 1]

                    # only plot every 5th sample for performance
                    curr_enc = curr_enc[::5]
                    curr_lick = curr_lick[::5]

                    # Rescale speed to fit on y-axis
                    curr_enc = curr_enc/max(curr_enc)

                    # plot behavior
                    curr_ax.plot(curr_enc, color='tab:red')  # plot running
                    curr_ax.spines['top'].set_visible(False)
                    curr_ax.spines['right'].set_visible(False)
                    curr_ax.set_xticks([])
                    ax2 = curr_ax.twiny()   # make new plot with independent x axis in the same subplot
                    ax2.plot(curr_lick, color='tab:blue')  # plot licking
                    ax2.set_ylim(-0.1, 1.1)
                    ax2.axis('off')         # Turn of axis spines for both new axes

                    # Make curr_ax (where axis is not turned off and you can see the background color) pickable
                    curr_ax.set_picker(True)
                    # Save the index of the current trial in the URL field of the axes to recall it later
                    curr_ax.set_url(count)
                    # Put curr_ax on top to make it reachable through clicking (only the top-most axes is pickable)
                    curr_ax.set_zorder(ax2.get_zorder() + 1)
                    # Make the background color of curr_ax completely transparent to keep ax2 visible
                    curr_ax.set_facecolor((0, 0, 0, 0))

                    count += 1

        def onpick(event):
            """
            When a subplot is selected/clicked, add the trial's index to the bad_trials list and shade the trial plot
            red. If it is clicked again, clear plot and remove trial index from the list.

            Args:
                event: Event handler from the pick event
            """
            clicked_ax = event.artist  # save artist (axis) where the pick was triggered
            trial = clicked_ax.get_url()  # Get the picked trial from the URL field
            if trial not in bad_trials:
                # If the trial was not clicked before, add it to the list and make the background red
                bad_trials.append(trial)
                clicked_ax.set_facecolor((1, 0, 0, 0.2))
                fig.canvas.draw()
            else:
                # If the trial was clicked before, remove it from the list and make the background transparent again
                bad_trials.remove(trial)
                clicked_ax.set_facecolor((0, 0, 0, 0))
                fig.canvas.draw()

        # Connect the "Pick" event (Subplot is clicked/selected) with the function describing what happens then
        fig.canvas.mpl_connect('pick_event', onpick)
        fig.suptitle('Curate trials for mouse {}, session {}'.format(trial_key['mouse_id'], trial_key['day']),
                     fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=True)  # Block=True is important to block other code execution until figure is closed

        return sorted(bad_trials)


@schema
class VRLogFile(dj.Imported):
    definition = """ # Filename of the unprocessed LOG file for each session
    -> VRSession
    ---
    log_filename        : varchar(128)      # filename of the LOG file (should be inside the session directory)
    """

    def make(self, key: dict) -> None:
        """
        Automatically looks up file name for LOG file of a single VRSession() entry.

        Args:
            key: Primary keys of the queried VRSession() entry.
        """
        mouse_id = (common_mice.Mouse & key).fetch1('mouse_id')

        # Get filename of this session's LOG file
        log_name = glob(os.path.join((common_exp.Session & key).get_absolute_path(), 'TDT LOG_*.txt'))

        if len(log_name) == 0:
            raise Warning('No LOG file found for M{} session {}!'.format(mouse_id, key['day']))
        elif len(log_name) > 1:
            raise Warning('{} LOG files found for M{} session {}!'.format(len(log_name), mouse_id, key['day']))
        else:
            self.insert1(dict(**key, log_filename=log_name[0]))


@schema
class VRLog(dj.Imported):
    definition = """ # Processed LOG data
    -> VRLogFile
    ---
    log_time            : longblob          # np.array of time stamps (datetime64[ns])
    log_trial           : longblob          # np.array of trial numbers (int)
    log_event           : longblob          # np.array of event log (str)
    """

    def make(self, key: dict) -> None:
        """
        Populates VRLog for every entry of VRLogFile() with processed log data.

        Args:
            key: Primary keys of the current VRLogFile() entry.
        """

        path = os.path.join((common_exp.Session & key).get_absolute_path(), (VRLogFile & key).fetch1('log_filename'))

        # Load LOG file
        log = pd.read_csv(path, sep='\t', parse_dates=[[0, 1]])

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

        insert_dict = dict(**key)

        # Parse fields as separate np.arrays
        insert_dict['log_time'] = np.array(log['Date_Time'], dtype=str)  # str because DJ doesnt like datetime[ns]
        insert_dict['log_trial'] = np.array(log['Trial'])
        insert_dict['log_event'] = np.array(log['Event'])
        self.insert1(insert_dict)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Fetch LOG data of the queried SINGLE session re-assembled as pandas DataFrame.

        Returns:
            DataFrame with time stamp, trial number and event of the queried LOG file.
        """
        return pd.DataFrame({'log_time': self.fetch1('log_time'),
                             'log_trial': self.fetch1('log_trial'),
                             'log_event': self.fetch1('log_event')})


@schema
class VRTrial(dj.Computed):
    definition = """ # Aligned trials of VR behavioral data
    -> VRSession
    trial_id            : tinyint           # Counter of the trial in the session, same as RawBehaviorFile()
    ---
    timestamp           : time              # Start time of the trial
    -> CorridorPattern
    tone                : tinyint           # bool flag whether the RZ tone during the trial was on or off
    pos                 : longblob          # 1d array with VR position sampled every 8 ms
    lick                : longblob          # 1d array with licks sampled every 8 ms
    frame               : longblob          # 1d array with frame triggers sampled every 8 ms
    enc                 : longblob          # 1d array with raw encoder ticks sampled every 8 ms
    valve               : longblob          # 1d array with valve openings (reward) sampled every 8 ms
    """

    def enc2speed(self) -> np.ndarray:
        """
        Transform encoder ticks of the queried SINGLE trial to speed in cm/s.

        Returns:
            1D numpy array of encoder data transformed to cm/s
        """
        # Hard-coded constant properties of encoder wheel
        D_WHEEL = 10.5  # wheel diameter in cm
        N_TICKS = 1436  # number of ticks in a full wheel rotation
        DEG_DIST = (D_WHEEL * np.pi) / N_TICKS  # distance in cm the band moves for each encoder tick

        # TODO: How to deal with the encoder artifact of "catching up" ticks from the ITI?
        # Query encoder data from queried trial and translate encoder data into velocity [cm/s]
        speed = self.fetch1('enc') * DEG_DIST / SAMPLE
        speed[speed == -0] = 0
        return speed

    def get_zone_borders(self) -> np.ndarray:
        """
        Return a deepcopy of the queried SINGLE trial's reward zone borders. The deepcopy is necessary to edit the
        zone borders without changing the data in the database.

        Returns:
            A numpy array with dimensions (2, 4), start and end position of all four RZs
        """
        return deepcopy((self * CorridorPattern).fetch1('positions'))

    def get_array(self, attr: Iterable[str] = None) -> np.ndarray:
        """
        Combine individual attribute data with reconstructed time stamp to a common array for processing.

        Args:
            attr: List of attributes from the behavior dataset that should be combined. Default is all attributes.

        Returns:
            A numpy array (# samples, # attributes + 1) with single attributes as columns, the common
            time stamp as first column.
        """
        if attr is None:
            attr = ['pos', 'lick', 'frame', 'enc', 'valve']

        # Fetch behavioral data of the trial, add time scale and merge into np.array
        data = self.fetch1(*attr)
        time = self.get_timestamps()
        return np.vstack((time, *data)).T

    def get_timestamps(self) -> np.ndarray:
        """
        Returns np.array of time stamps in seconds for behavioral data points.

        Returns:
            Np.ndarray with shape (n_datapoints,) of time stamps in seconds
        """
        n_samples = len(self.fetch1('lick'))
        # To avoid floating point rounding errors, first create steps in ms (*1000), then divide by 1000 for seconds
        return np.array(range(0, n_samples * int(SAMPLE * 1000), int(SAMPLE * 1000))) / 1000

    def make(self, key: dict) -> None:
        """
        Fills VRTrial with temporally aligned behavior parameters for all trials of one queried VRSession() entry.

        Args:
            key: Primary keys of current VRSession() entry
        """

        # Fetch data about the session
        trial_ids = RawBehaviorFile().fetch('trial_id')  # Trial IDs (should be regularly counting up)
        imaging = bool((VRSession & key).fetch1('imaging_session'))  # Flag if frame trigger should be created
        cond = (common_exp.Session & key).fetch1('task')  # Task condition
        cond_switch = (VRSession & key).fetch1('condition_switch')  # First trial of new condition

        for trial_id in trial_ids:

            # Initialize relevant variables
            trial_key = dict(**key, trial_id=trial_id)  # dict containing the values of all trial attributes
            frame_count = None  # frame count of the imaging file of that trial
            if imaging:
                frame_count = (common_img.RawImagingFile & dict(**key, part=trial_id)).fetch1('nr_frames')

            # Find out which condition this trial was
            trial_key['pattern'], trial_key['tone'] = self.get_condition(trial_key, cond, cond_switch)

            # Get time stamp from filename
            time_format = '%H%M%S'
            timestamp = datetime.strptime((RawBehaviorFile & trial_key).fetch1('tdt_filename').split('_')[-1][:-4],
                                          time_format)
            trial_key['timestamp'] = timestamp.time()

            # Get arrays of the current trial raw data
            data = (RawBehaviorFile & trial_key).load_data()

            # ALIGN BEHAVIOR
            # print("Starting to align behavior files of {}".format(trial_key))
            # Create array containing merged and time-aligned behavior data.
            # Returns None if there was a problem with the data, and the trial will be skipped (like incomplete trials)
            merge = self.align_behavior_files(trial_key, data['enc'], data['tcp'], data['tdt'],
                                              imaging=imaging, frame_count=frame_count)

            if merge is not None:
                # parse columns into entry dict and typecast to int to save disk space
                # TODO: make "frame" and "valve" to event-times rather than continuous sampling?
                trial_key['pos'] = merge[:, 1].astype(np.float32)
                trial_key['lick'] = merge[:, 2].astype(int)
                trial_key['frame'] = merge[:, 3].astype(int)
                trial_key['enc'] = -merge[:, 4].astype(int)  # encoder is installed upside down, so reverse sign
                trial_key['valve'] = merge[:, 5].astype(int)
            else:
                raise ValueError("Alignment returned None, check trial!")

            # Insert entry of the trial into the database
            self.insert1(trial_key)

    def get_condition(self, key: dict, task: str, condition_switch: Iterable[int]) -> Tuple[str, int]:
        """
        Returns condition (RZ position, corridor pattern, tone) of a single trial.
        Args:
            key: Primary keys of the queried trial
            task: Type of task, manually entered in common_exp.Session
            condition_switch: Trial ID(s) at which the new condition in this session appears. [-1] for no change.

        Returns:
            Corridor pattern at that trial (corresponds to CorridorPattern()), and if the tone was on (1) or off (0).
        """

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
        elif task == 'No pattern and tone' and (len(condition_switch) == 2) and (
                key['trial_id'] >= condition_switch[1]):
            pattern = 'none'
            tone = 0
        elif task == 'No pattern and tone' and (len(condition_switch) == 2) and (key['trial_id'] < condition_switch[1]):
            pattern = 'none'
            tone = 1
        else:
            raise Exception('Error at {}:\n'
                            'Task condition could not be determined for trial nb {}.'.format(key, key['trial_id']))

        return pattern, tone

    def is_session_novel(self, sess_key: dict) -> bool:
        """
        Checks whether the session of 'sess_key' is in the novel corridor (from VR zone borders)
        Args:
            sess_key: Primary keys to query entry of VRLog()

        Returns:
            Boolean flag whether the queried session is in the novel (True) or training (False) corridor
        """
        # Get event log
        log_events = (VRLog & sess_key).get_dataframe()['log_event']
        # Get the rounded position of the first reward zone
        rz_pos = int(
            np.round(float(log_events[log_events.str.contains('VR enter Reward Zone:')].iloc[0].split(':')[1])))
        if rz_pos == -6:
            return False
        elif rz_pos == 9:
            return True
        else:
            print(f'Could not determine context in session {self.key}!\n')

    @staticmethod
    def align_behavior_files(trial_key: dict, encoder: np.ndarray, position: np.ndarray, trigger: np.ndarray,
                             imaging: bool = False, frame_count: Optional[int] = None) -> np.ndarray:
        """
        Align behavioral data from three text files to a common master time frame provided by LabView. Data are
        re-sampled at the rate of the encoder (125 Hz), as the encoder is a summed data collection and is difficult to
        resample.

        Args:
            trial_key   :   Dict with primary keys for current trial to query LOG file from VRLog()
            encoder     :   Encoder.txt file data (running speed)
            position    :   TCP.txt file data (VR position)
            trigger     :   TDT.txt file data (licking and frame trigger)
            imaging     :   Bool flag whether the behavioral data is accompanied by an imaging movie (defaults to False)
            frame_count :   Frame count of the imaging movie (only needed if imaging=True)

        Returns:
            Aligned data in np.array with columns 'time', 'position', 'licking', 'trigger', 'encoder', 'speed', 'water'
        """

        pd.options.mode.chained_assignment = None  # Disable false positive SettingWithCopyWarning

        # Separate licking and trigger signals (different start times)
        licking = trigger[:, :2].copy()
        licking[0, 0] = licking[0, 1]
        licking[0, 1] = encoder[0, 1]
        trigger = np.delete(trigger, 1, axis=1)

        data = [trigger, licking, encoder, position]

        if imaging and frame_count is None:
            raise ValueError(f'Error in trial {trial_key}: provide frame count if imaging=True!')

        # transform the integer time stamps plus the date of the session into datetime objects
        time_format = '%Y%m%d%H%M%S%f'
        date = datetime.strftime(trial_key['day'], '%Y%m%d')
        for f in data:
            if str(int(f[0, 0]))[4:] == '60000':
                f[0, 0] -= 1
        start_times = np.array([datetime.strptime(date + str(int(x[0, 0])), time_format) for x in data])

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
                    print(f'No frame trigger in {trial_key}. Check file!')
                    # return None, None
                trigger[int(round(np.mean(block))), 1] = 1
                trig_idx.append(int(round(np.mean(block))))

            # check if imported frame trigger matches frame count of .tif file and try to fix it
            more_frames_in_TDT = int(np.sum(trigger[1:, 1]) - frame_count)  # pos if TDT, neg if .tif had more frames

            if abs(more_frames_in_TDT) > 5:
                print("Trial {}:\n{} more frames imported from TDT than in raw imaging file "
                      "({} frames)".format(trial_key, more_frames_in_TDT, frame_count))

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
                    raise ImportError(f'{more_frames_in_TDT} too many frames imported from TDT, could not be corrected!')

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
                        raise ValueError(f'Frame correction failed for {trial_key}!')
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
                    raise ImportError(f'{int(abs(more_frames_in_TDT))} too few frames imported from TDT, could not be corrected.')

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
        if any(df_list[3].iloc[0].isna()):
            df_list[3] = df_list[3].shift(periods=-1, fill_value=110)

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
            raise ValueError(f'Frame count matching unsuccessful: \n{merge_trig} frames in merge, should be {frame_count} frames.')

        # transform back to numpy array for saving
        time_passed = merge_filt.index - merge_filt.index[0]  # transfer timestamps to
        seconds = np.array(time_passed.total_seconds())  # time change in seconds
        array_df = merge_filt[['position', 'licking', 'trigger', 'encoder', 'water']]  # change column order
        array = np.hstack((seconds[..., np.newaxis], np.array(array_df)))  # combine both arrays

        return array

    def compute_performances(self, params: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes lick, binned lick and stop performance of a single trial. Called by VRPerformance.make().

        Args:
            params: Current entry of PerformanceParameters()

        Returns:
            Three 1D arrays with different performance metrics: Binned lick ratio, lick count ratio, stop ratio
        """
        # TODO: Alternative approaches to calculate performances
        #  - only count the ONSET of licks per bin, not if a lick went over a bin border (especially impactful if a mouse is running fast while licking)
        #  - if one bin of a RZ is licked, count it for all bins of that RZ
        #  - check for mean distance of licked bins to the next RZ

        # Fetch required behavior data as array (columns: time - lick - pos - enc)
        data = self.get_array(attr=('lick', 'pos', 'enc'))

        # Get reward zone borders for the current trial and add the buffer
        zone_borders = self.get_zone_borders()
        zone_borders[:, 0] -= params['vrzone_buffer']
        zone_borders[:, 1] += params['vrzone_buffer']

        # Find out which reward zones were passed (reward given) if parameter is set (default no)
        reward_from_merged = False
        if params['valve_for_passed']:
            rz_passed = np.zeros(len(zone_borders))
            for idx, zone in enumerate(zone_borders):
                # Get the reward entries at indices where the mouse is in the current RZ
                valve = self.fetch1('valve')
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

    def compute_time_metrics(self, params: dict) -> Tuple[float, float, float]:
        """
        Compute mean speed, running speed and trial duration of a single trial.

        Args:
            params: Current entry of PerformanceParameters()

        Returns:
            Three different time metrics: mean speed, mean running speed and trial duration of the queried trial
        """

        # Get mean speed by taking track length / max time stamp. Slightly more accurate than mean(vel) because ITI
        # running is ignored, but included in vel
        time = max(self.get_timestamps())
        length = (VRSession & self.restriction[0]).fetch1('length')
        mean_speed = length / time

        # Get mean running speed by filtering out time steps where mouse is stationary
        vel = self.enc2speed()  # Get velocity in cm/s
        running_vel = vel[vel >= params['velocity_thresh']]
        mean_running_speed = np.mean(running_vel)

        return mean_speed, mean_running_speed, time

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
    velocity_thresh     : tinyint       # speed in cm/s below which mouse counts as "stationary"
    stop_time           : smallint      # time in ms above which the period counts as a "stop" for stopping performance
    metric_for_trend    : varchar(64)   # name of the performance metric that is used to compute performance trend
    """
    contents = [
        [0, 2, 0, 1, 5, 100, 'binned_lick_ratio']
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
    trial_duration              : longblob          # np.array of trial durations (in s)
    """

    def make(self, key: dict) -> None:
        """
        Computes general performance metrics from individual trials of a session with a certain set of parameters.
        Args:
            key: Primary keys of the union of the current VRSession and PerformanceParameters entry.
        """

        # Get current set of parameters
        params = (PerformanceParameters & key).fetch1()

        # Initialize dict that will hold single-trial lists (one per (non-primary) attribute)
        trial_data = {key: [] for key in self.heading if key not in self.primary_key}

        trial_ids = (VRTrial & key).fetch('trial_id')

        # Process every trial (start with 1 because trial_id is 1-based
        for trial_id in trial_ids:
            # Store query of current trial
            trial = (VRTrial & key & 'trial_id={}'.format(trial_id))

            # # Fetch behavioral data of the current trial, add time scale and merge into np.array
            # lick, pos, enc = trial.fetch1('lick', 'pos', 'enc')
            # # To avoid floating point rounding errors, first create steps in ms (*1000), then divide by 1000 for seconds
            # time = np.array(range(0, len(lick) * int(SAMPLE * 1000), int(SAMPLE * 1000))) / 1000
            # data = np.vstack((time, lick, pos, enc)).T

            # Compute lick and stop performances
            binned_lick_ratio, lick_count_ratio, stop_ratio = trial.compute_performances(params)

            # Compute time metrics
            mean_speed, mean_running_speed, trial_duration = trial.compute_time_metrics(params)

            # Add trial metrics to data dict
            trial_data['binned_lick_ratio'].append(binned_lick_ratio)
            trial_data['lick_count_ratio'].append(lick_count_ratio)
            trial_data['stop_ratio'].append(stop_ratio)
            trial_data['mean_speed'].append(mean_speed)
            trial_data['mean_running_speed'].append(mean_running_speed)
            trial_data['trial_duration'].append(trial_duration)

        # Combine primary dict "key" with attributes "trial_data" and insert entry
        self.insert1({**key, **trial_data})

    def get_mean(self, attr: str) -> List[float]:
        """
        Get a list of the mean of a given performance attribute of the queried sessions.
        
        Args:
            attr: Performance metric, must be attribute of VRPerformance()

        Returns:
            Means of a performance attribute of the queried sessions
        """
        sess = self.fetch(attr)
        return [np.mean(x) for x in sess]

    def plot_performance(self, attr: str = 'binned_lick_ratio') -> None:
        """
        Plots performance across time for the queried sessions.
        
        Args:
            attr: Performance metric, must be attribute of VRPerformance()
        """
        mouse_id, day, behav = self.fetch('mouse_id', 'day', attr)
        df = pd.DataFrame(dict(mouse_id=mouse_id, day=day, behav=behav))
        df_new = df.explode('behav')

        df_new['behav'] = df_new['behav'].astype(float)
        df_new['day'] = pd.to_datetime(df_new['day'])

        grid = sns.FacetGrid(df_new, col='mouse_id', col_wrap=3, height=3, aspect=2)
        grid.map(sns.lineplot, 'day', 'behav')

        for ax in grid.axes.ravel():
            ax.axhline(0.75, linestyle='--', color='r', alpha=0.5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylabel(attr)


@schema
class PerformanceTrend(dj.Computed):
    definition = """ # Trend analysis metrics of performance across trials of a session. LinReg via statsmodels.OLS()
    -> VRPerformance
    ---
    p_normality = NULL    : float           # p-value of stats.normaltest (D'Agostino + Pearson's omnibus test)
                                            # (can only be reliably determined for >20 trials, otherwise p=1).
    perf_corr = NULL      : float           # Correlation coefficient, through Pearson or Spearman (dep. on p_normality)
    p_perf_corr = NULL    : float           # p-value of correlation coefficient (strongly depends on sample size)
    perf_r2 = NULL        : float           # R-squared value of the OLS model (how much is y explained by x?)
    prob_lin_reg = NULL   : float           # Probability of F-statistic (likeliness that x's effect on y is 0)
    perf_intercept = NULL : float           # Intercept of the modelled linear regression
    perf_slope = NULL     : float           # Slope of the fitted line (neg = worse, pos = better performance over time)
    perf_ols_x = NULL     : longblob        # X (trial numbers) used to fit OLS (X is the SECOND argument in sm.OLS!)
    perf_ols_y = NULL     : longblob        # Y (performance data) used to fit OLS (y is the FIRST argument in sm.OLS!)
    """

    def make(self, key: dict) -> None:
        """
        Compute time-dependent trends across trials of a VRPerformance() session entry and populate PerformanceTrend().
        
        Args:
            key: Primary keys of the queried VRPerformance() entry.
        """
        # Get parameter set
        params = (PerformanceParameters & key).fetch1()

        # Get the appropriate performance dataset
        perf = (VRPerformance & key).fetch1(params['metric_for_trend'])

        if len(perf) > 1:
            # If there are at least 20 trials in the session, test for normality, otherwise assume non-normality
            normality = 1
            if len(perf) >= 20:
                k2, normality = stats.normaltest(perf)

            # Create x axis out of trial IDs
            x_vals = np.arange(len(perf))

            # If normal, perform Pearson correlation, otherwise Spearman
            if normality < 0.05:
                corr, p = stats.pearsonr(x=x_vals, y=perf)
            else:
                corr, p = stats.spearmanr(a=x_vals, b=perf)

            # For special cases (e.g. only 2 trials per session), p cannot be determined and reverts to 1
            if not np.isnan(corr) and np.isnan(p):
                p = 1

            # Perform linear regression with ordinary least squares (OLS) from statsmodels and extract relevant metrics
            x_fit = sm.add_constant(x_vals)
            ols = sm.OLS(perf, x_fit).fit()
            r2 = ols.rsquared
            p_r2 = ols.f_pvalue
            intercept = ols.params[0]
            slope = ols.params[1]
        # If there is only 1 trial, none of the parameters can be calculated, and should revert to None
        else:
            normality = corr = p = r2 = p_r2 = intercept = slope = x_fit = perf = None

        # TODO: maybe pickle the OLS object to store it directly in the database
        # Insert entry into the table
        self.insert1(dict(key, p_normality=normality, perf_corr=corr, p_perf_corr=p, perf_r2=r2, prob_lin_reg=p_r2,
                          perf_intercept=intercept, perf_slope=slope, perf_ols_x=x_fit, perf_ols_y=perf))
