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
from hheise_scripts import util, behav_util

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

    def load_data(self) -> Dict[str, List[np.ndarray]]:
        """
        Loads data of queried trials and returns it in dict form.

        Returns:
            Data dict with keys "tdt", "tcp" and "enc", holding lists of np arrays of the respective data
        """

        # Get the session of the current query
        session_path = (common_exp.Session() & self).get_absolute_path()

        # Load data into a dict
        data = dict(
            tdt=[np.loadtxt(os.path.join(session_path, fname)) for fname in self.fetch['tdt_filename']],
            tcp=[np.loadtxt(os.path.join(session_path, fname)) for fname in self.fetch['tcp_filename']],
            enc=[np.loadtxt(os.path.join(session_path, fname)) for fname in self.fetch['enc_filename']],
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
    -> CorridorPattern
    tone                : bool              # bool flag whether the RZ tone during the trial was on or off
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
        # To avoid floating point rounding errors, first create steps in ms (*1000), then divide by 1000 for seconds
        time = np.array(range(0, len(data[0]) * int(SAMPLE * 1000), int(SAMPLE * 1000))) / 1000
        return np.vstack((time, *data)).T

    def make(self, key: dict) -> None:
        """
        Find raw behavior files, load data, and align it. Return entry dicts for VRTrial(), RawEncFile(), RawTDTFile()
        and RawTCPFile() in a dict, each entry corresponding to a list of trial dicts for each table.

        The entry dict has to be returned instead of inserted here because DJ only lets you insert entries into Imported
        or Computed tables inside the make() function call.

        Args:
            key: Primary keys to query each entry of VRSession() to be populated

        Returns:
            Each key ('trial', 'enc', 'pos' and 'trig') holds one list, which contains entry dicts for every trial of
            the current session for tables VRTrial(), RawEncFile(), RawTCPFile(), and RawTDTFile() respectively.
        """

        counter = 0

        # Initialize dict that holds all entries
        data = {'trial': [], 'enc': [], 'pos': [], 'trig': []}

        # Get IDs of good trials
        trial_ids = RawBehaviorFile().fetch('trial_id')

        # Get flag if current session is an imaging session and frame trigger should be created
        imaging = bool((VRSession & key).fetch1('imaging_session'))

        for trial_id in trial_ids:

            # Initialize relevant variables
            trial_key = dict(key, trial_id=counter)  # dict containing the values of all trial attributes
            frame_count = None  # frame count of the imaging file of that trial
            if imaging:
                frame_count = (common_img.RawImagingFile & trial_key).fetch1('nr_frames')

            # Find out which condition this trial was
            trial_key['pattern'], trial_key['tone'] = self.get_condition(trial_key, cond, cond_switch)

            ### ALIGN BEHAVIOR ###
            # Find the files of the same trial in the session folder
            timestamp = int(
                os.path.basename(enc_file).split('_')[1][:-4])  # here the 0 in front of early times is removed
            pos_file = find_file(timestamp, pos_files)
            trig_file = find_file(timestamp, trig_files)
            if pos_file is None or trig_file is None:
                print(f'Could not find all three files for timestamp {timestamp}!')
                # return

            print("Starting to align behavior files of {}".format(trial_key))
            # Create array containing merged and time-aligned behavior data.
            # Returns None if there was a problem with the data, and the trial will be skipped (like incomplete trials)
            merge = behav_util.align_behavior_files(trial_key, enc_file, pos_file, trig_file,
                                                    imaging=imaging, frame_count=frame_count)

            if merge is not None:
                print("Typecasting merged data into the Datajoint format.")
                # parse columns into entry dict and typecast to int to save disk space
                # TODO: make "frame" and "valve" to event-times rather than continuous sampling?
                trial_key['pos'] = merge[:, 1]
                trial_key['lick'] = merge[:, 2].astype(int)
                trial_key['frame'] = merge[:, 3].astype(int)
                trial_key['enc'] = -merge[:, 4].astype(int)  # encoder is installed upside down, so reverse sign
                trial_key['valve'] = merge[:, 5].astype(int)

                # Collect data of the current trial
                data['trial'].append(trial_key)
                data['enc'].append(dict(key, trial_id=counter, enc_filename=util.remove_session_path(key, enc_file)))
                data['pos'].append(dict(key, trial_id=counter, tcp_filename=util.remove_session_path(key, pos_file)))
                data['trig'].append(dict(key, trial_id=counter, tdt_filename=util.remove_session_path(key, trig_file)))

                counter += 1
            else:
                print("Alignment returned None, skipping trial.")

        return data

    # Enter the single trial entries one by one
    for i in range(len(data['trial'])):
        VRSession.VRTrial().insert1(data['trial'][i])
        RawEncFile().insert1(data['enc'][i])
        RawTDTFile().insert1(data['trig'][i])
        RawTCPFile().insert1(data['pos'][i])

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
class ManuallyValidatedSessions(dj.Manual):
    definition = """ # Holds sessions that have been manually validated. Only these sessions will be analysed further.
    -> VRSession
    ---
    date_of_validation          : date              # date at which the validation occurred
    """


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

        # Process every trial (start with 1 because trial_id is 1-based
        for trial_id in range(1, len(VRSession.VRTrial & key) + 1):
            # Store query of current trial
            trial = (VRSession.VRTrial & key & 'trial_id={}'.format(trial_id))

            # Fetch behavioral data of the current trial, add time scale and merge into np.array
            lick, pos, enc = trial.fetch1('lick', 'pos', 'enc')
            # To avoid floating point rounding errors, first create steps in ms (*1000), then divide by 1000 for seconds
            time = np.array(range(0, len(lick) * int(SAMPLE * 1000), int(SAMPLE * 1000))) / 1000
            data = np.vstack((time, lick, pos, enc)).T

            # Compute lick and stop performances
            binned_lick_ratio, lick_count_ratio, stop_ratio = self.compute_performances(trial, data, params)

            # Compute time metrics
            mean_speed, mean_running_speed, trial_duration = self.compute_time_metrics(trial, params, data[:, 0])

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
