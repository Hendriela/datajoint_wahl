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
from hheise_scripts import data_util

from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple, Dict, Union
import os
import ast
from glob import glob
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy import stats

try:
    import statsmodels.api as sm
    import seaborn as sns
except ModuleNotFoundError:
    print("Import hheise_behav with read-only access.")

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
class VRSessionInfo(dj.Imported):
    definition = """ # Info about the VR Session, mostly read from "behavioral evaluation" Excel file
    -> common_exp.Session
    ---
    imaging_session     : tinyint           # bool flag whether imaging was performed during this session
    condition_switch    : longblob          # List of trial IDs of the first trial(s) of the new condition (base 0, -1 if no switch)
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
        Populates VRSessionInfo() for every entry of common_exp.Session().

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
        if ('weight [g]' in sess_entry.columns) and not pd.isna(sess_entry['weight [g]'].values[0]):
            try:
                common_mice.Weight().insert1({'username': key['username'], 'mouse_id': key['mouse_id'],
                                              'date_of_weight': key['day'],
                                              'weight': sess_entry['weight [g]'].values[0]})
            except dj.errors.DuplicateError:
                pass

        # Get block and condition switch from session_notes string
        note_dict = ast.literal_eval((common_exp.Session & key).fetch1('session_notes'))
        new_key['block'] = note_dict['block']
        new_key['condition_switch'] = eval(note_dict['switch'])  # eval turns string into list

        # Turn manual ID (base 1) into pythonic ID (base 0)
        if new_key['condition_switch'] != [-1]:
            new_key['condition_switch'] = [x - 1 for x in new_key['condition_switch']]

        # Check if this is an imaging session (session has to be inserted into common_img.Scan() first)
        if len((common_img.Scan & key).fetch()) == 1:
            new_key['imaging_session'] = 1
        else:
            new_key['imaging_session'] = 0
            print(f"Could not find session {key} in common_img.Scan, thus assuming that the session is not an imaging"
                  f" session. If it is, enter session into Scan() before populating VRSession()!")

        # Replace NaNs with empty strings
        new_key = {k: ('' if v is np.nan else v) for k, v in new_key.items()}

        self.insert1(new_key)


@schema
class RawBehaviorFile(dj.Imported):
    definition = """ # File names (relative to session folder) of raw VR behavior files (3 separate files per trial)
    -> VRSessionInfo
    trial_id            : smallint          # Counter for file sets (base 0)
    ---
    tdt_filename        : varchar(256)      # filename of the TDT file (licking and frame trigger)
    tcp_filename        : varchar(256)      # filename of the TCP file (VR position)
    enc_filename        : varchar(128)      # filename of the Enc file (running speed)
    """

    def make(self, key: dict, skip_curation: bool = False) -> None:
        """
        Automatically looks up file names for behavior files of a single VRSessionInfo() entry.

        Args:
            key: Primary keys of the queried VRSessionInfo() entry.
            skip_curation: Optional bool flag passed down from populate(). If True, all trials are automatically
                            accepted and not shown to the user. Useful if a session has already been curated.
        """

        print("Finding raw behavior files for session {}".format(key))

        is_imaging_session = (VRSessionInfo & key).fetch1('imaging_session') == 1

        # Get complete path of the current session
        bases = [login.get_neurophys_data_directory(), *login.get_alternative_data_directories()]

        for base in bases:
            root = os.path.join(base, (common_exp.Session & key).fetch1('session_path'))

            # Find all behavioral files; for now both file systems (in trial subfolder and outside)
            encoder_files = glob(root + '\\Encoder*.txt')
            encoder_files.extend(glob(root + '\\**\\Encoder*.txt'))
            position_files = glob(root + '\\TCP*.txt')
            position_files.extend(glob(root + '\\**\\TCP*.txt'))
            trigger_files = glob(root + '\\TDT TASK*.txt')
            trigger_files.extend(glob(root + '\\**\\TDT TASK*.txt'))

            if len(encoder_files) > 0:
                # If the list has entries, we found the directory where the files are stored and set this as the cwd
                login.set_working_directory(base)
                break

        if len(encoder_files) == 0:
            raise ImportError(f'Could not find behavior files for {key} \nin these directories:\n{bases}')

        # Sort them by time stamp (last part of filename, separated by underscore
        encoder_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        position_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        trigger_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

        ### FILTER OUT BAD TRIALS ###
        # Catch an uneven number of files
        if not (len(encoder_files) == len(position_files)) & (len(encoder_files) == len(trigger_files)):
            raise ImportError(f'Uneven numbers of encoder, position and trigger files in folder {root}!')

        # Commented out because RawImagingFile should be filled AFTER trials are curated
        # # Catch different numbers of behavior files and raw imaging files
        # if (len(encoder_files) != len(common_img.RawImagingFile() & key)) and is_imaging_session:
        #     raise ImportError(f'Different numbers of behavior and imaging files in folder {root}!')

        # Check if RawImagingFile has already been filled for this session
        if (len(common_img.RawImagingFile() & key) > 0) and is_imaging_session:
            print(f"TIFF files for session {key} already in common_img.RawImagingFile.\n"
                  f"If faulty trials are found, you have to delete these entries and potentially rerun the imaging pipeline.")

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
                raise ValueError(f'Faulty trial (TDT file copied from previous trial), time stamps differed by '
                                 f'{int(max(max_diff))}s!')

        if not skip_curation:
            # Manually curate sessions to weed out trials with e.g. buggy lick sensor
            bad_trials = self.plot_screening(trigger_files, encoder_files, key)
        else:
            bad_trials = []

        if len(bad_trials) > 0:
            print("Session {}:\nThe following trials will be excluded from further analysis.\n"
                  "DELETE THE CORRESPONDING TIFF FILES!!".format(key))
            for index in sorted(bad_trials, reverse=True):
                print(trigger_files[index])
                del encoder_files[index]
                del trigger_files[index]
                del position_files[index]
        else:
            print("All trials accepted.")

        # If everything is fine, insert the behavioral file paths, relative to session folder, sorted by time
        session_path = (common_exp.Session() & key).get_absolute_path()
        import itertools

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

            for subset in itertools.combinations(times, 2):
                if abs(subset[0] - subset[1]).seconds > 2:
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
            trial_key: Primary keys of the queried VRSessionInfo() entry.

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
                    try:
                        curr_ax = ax[row, col]
                    except IndexError:
                        curr_ax = ax[col]

                    # Load the data, (ignore first time stamp)
                    curr_enc = -np.loadtxt(enc_list[count])[1:, 1]
                    curr_lick = np.loadtxt(tdt_list[count])[1:, 1]

                    # only plot every 5th sample for performance
                    curr_enc = curr_enc[::5]
                    curr_lick = curr_lick[::5]

                    # Rescale speed to fit on y-axis
                    curr_enc = curr_enc / max(curr_enc)

                    # plot behavior
                    curr_ax.plot(curr_enc, color='tab:red')  # plot running
                    curr_ax.spines['top'].set_visible(False)
                    curr_ax.spines['right'].set_visible(False)
                    curr_ax.set_xticks([])
                    ax2 = curr_ax.twiny()  # make new plot with independent x axis in the same subplot
                    ax2.plot(curr_lick, color='tab:blue')  # plot licking
                    ax2.set_ylim(-0.1, 1.1)
                    ax2.axis('off')  # Turn of axis spines for both new axes

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
    -> VRSessionInfo
    ---
    log_filename        : varchar(128)      # filename of the LOG file (should be inside the session directory)
    """

    def make(self, key: dict) -> None:
        """
        Automatically looks up file name for LOG file of a single VRSessionInfo() entry.

        Args:
            key: Primary keys of the queried VRSessionInfo() entry.
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
        tab_length = (VRSessionInfo & key).fetch1('length')
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

    def is_session_novel(self) -> bool:
        """
        Checks whether the queried session is in the novel corridor (from VR zone borders)

        Returns:
            Boolean flag whether the queried session is in the novel (True) or training (False) corridor
        """
        # Get event log
        log_events = self.get_dataframe()['log_event']

        # Get the rounded position of the first reward zone
        rz_pos = int(np.round(float(log_events[log_events.str.contains('VR enter Reward Zone:')].iloc[0].split(':')[1])))
        if rz_pos == -6:
            return False
        elif rz_pos == 9:
            return True
        else:
            raise ValueError(f'Could not determine context in session {self.fetch1("KEY")}!\n')


@schema
class VRSession(dj.Computed):
    definition = """ # Session-specific table holding part-tables with trials of aligned VR behavioral data
    -> VRSessionInfo
    ---
    time_vr_align = CURRENT_TIMESTAMP    : timestamp     # automatic timestamp
    """

    class VRTrial(dj.Part):
        definition = """ # Aligned trials of VR behavioral data
        -> VRSession
        trial_id            : tinyint           # Counter of the trial in the session, same as RawBehaviorFile(), base 0
        ---
        timestamp           : time              # Start time of the trial
        -> CorridorPattern
        tone                : tinyint           # bool flag whether the RZ tone during the trial was on or off
        pos                 : longblob          # 1d array with VR position sampled every 8 ms (125 Hz)
        lick                : longblob          # 1d array with binary licks sampled every 8 ms (125 Hz)
        frame               : longblob          # 1d array with binary frame trigger sampled every 8 ms (125 Hz)
        enc                 : longblob          # 1d array with raw encoder ticks sampled every 8 ms (125 Hz)
        valve               : longblob          # 1d array with binary valve openings (reward) sampled at 125 Hz
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
            #
            # def idx2arr(indices: Iterable[int], length: int) -> np.ndarray:
            #     """
            #     Transform a list of array of indices into a 1D binary array with 1 at these indices.
            #
            #     Args:
            #         indices: Indices in final array at which positions the value will be 1
            #         length: Length of the final array. Has to be larger than the largest value in indices
            #
            #     Returns:
            #         1D binary array with 1 at specified indices and 0 everywhere else.
            #     """
            #
            #     if max(indices) >= length:
            #         raise IndexError(f"Error in VRSession.VRTrial().get_array() of entry {self.fetch1('KEY')}:\n"
            #                          f"Highest index {max(indices)} cannot fit into an array of length {length}!")
            #     else:
            #         bin_array = np.zeros(length, dtype=np.int8)
            #         bin_array[indices] = 1
            #
            #         return bin_array

            if attr is None:
                attr = ['pos', 'lick', 'frame', 'enc', 'valve']

            # Fetch behavioral data of the trial, add time scale and merge into np.array
            data = self.fetch1(*attr)
            #
            # # If queried, transform lick and valve indices into an continuously sampled array
            # if 'lick' in attr:
            #     data[attr.index('lick')] = data[attr.index('lick')]

            time = self.get_timestamps()
            return np.vstack((time, *data)).T

        def get_arrays(self, attr: Iterable[str] = None) -> List[np.ndarray]:
            """
            Wrapper function of self.get_array() for more than one trial. Returns list of arrays of queried entries.

            Args:
                attr: List of attributes from the behavior dataset that should be combined. Default is all attributes.

            Returns:
                A list of ndarrays (# samples, # attributes + 1) with single attributes as columns, the common
                    time stamp as first column.
            """

            trial_ids = self.fetch('trial_id')
            data = [(self & {'trial_id': trial_id}).get_array(attr) for trial_id in trial_ids]

            return data

        def get_timestamps(self) -> np.ndarray:
            """
            Returns np.array of time stamps in seconds for behavioral data points.

            Returns:
                Np.ndarray with shape (n_datapoints,) of time stamps in seconds
            """
            n_samples = len(self.fetch1('lick'))
            # To avoid floating point rounding errors, first create steps in ms (*1000), then divide by 1000 for seconds
            return np.array(range(0, n_samples * int(SAMPLE * 1000), int(SAMPLE * 1000))) / 1000

        def get_binned_licking(self, bin_size: int) -> np.ndarray:
            """
            Bin individual licks to VR position. Used to draw licking histograms for raw behavior analysis.

            Args:
                bin_size: Size of VR position bins (in arbitrary VR units) in which to bin the licks.

            Returns:
                1D array with length 120/bin_size, each element containing the number of individual licks at that bin.
            """

            trial = self.get_array(('pos', 'lick'))
            lick_only = trial[np.where(trial[:, 2] == 1)]

            if lick_only.shape[0] == 0:
                hist = np.zeros(int(120 / bin_size))
            else:
                # split licking track into individual licks and get VR position
                diff = np.round(np.diff(lick_only[:, 0]) * 1000).astype(int)    # get array of time differences (in ms)
                licks = np.split(lick_only, np.where(diff > SAMPLE * 1000)[0] + 1)  # split at points where difference > 8 ms (sample gap)
                licks = [i for i in licks if i.shape[0] <= 5//SAMPLE]           # only keep licks shorter than 5 seconds (longer licks might indicate problems with sensor)
                lick_pos = [x[0, 1] for x in licks]                             # get VR position when each lick begins

                # bin lick positions (arbitrary VR positions from -10 to 110 are hard-coded)
                hist, _ = np.histogram(np.digitize(lick_pos, np.arange(start=-10, stop=111, step=bin_size)),
                                       bins=np.arange(start=1, stop=int(120 / bin_size + 2)), density=False)
            return hist

        def compute_performances(self, params: dict) -> Tuple[float, float, float, float]:
            """
            Computes lick, binned lick and stop performance of a single trial. Called by VRPerformance.make().

            Args:
                params: Current entry of PerformanceParameters()

            Returns:
                Four different performance metrics: Distance to next RZ, Binned lick ratio, lick count ratio, stop ratio
            """
            # TODO: Alternative approaches to calculate performances
            #  - if one bin of a RZ is licked, count it for all bins of that RZ

            def get_lick_count_ratio(lick_data: np.ndarray, borders: np.ndarray, passed_zones: Optional[float] = None) \
                    -> Union[float, Tuple[float, float]]:
                """
                Compute performance value of one trial based on ratio of individual licks.

                Args:
                    lick_data: 2D Numpy array with shape (n_samples, n_metrics) of behavioral data. From get_array().
                        Columns have to be time - lick - pos - enc.
                    borders: 2D Numpy array with shape (n_zones, 2), containing buffed RZ borders in VR coordinates.
                        From get_zone_borders()
                    passed_zones: Fraction of reward zones successfully passed in this trial. If given, use that value
                        (computed from valve openings). If not given, compute here based on licks.

                Returns:
                    lick_count_ratio: float, ratio of in-RZ-licks vs total licks, corrected by passed reward zones
                    If passed_zones was not provided, also return lick-based passed_rzs
                """
                # select only time point where the mouse licked
                lick_only = lick_data[np.where(lick_data[:, 1] == 1)]

                if lick_only.shape[0] == 0:
                    lick_count_ratio = np.nan  # set nan, if there was no licking during the trial
                    passed_rzs = 0
                else:
                    # remove continuous licks that were longer than 5 seconds
                    lick_diff = np.round(np.diff(lick_only[:, 0]) * 1000).astype(int)  # get an array of time differences in ms
                    licks = np.split(lick_only, np.where(lick_diff > SAMPLE * 1000)[0] + 1)  # split where difference > sample rate
                    licks = [i[0] for i in licks if i.shape[0] <= int(5 / SAMPLE)]  # only keep licks shorter than 5 seconds, and only take lick onset (to get one row per lick)
                    if len(licks) > 0:
                        licks = np.vstack(licks)  # put list of arrays together to one array
                        # out of these, select only time points where the mouse was in a reward zone
                        lick_zone_only = []
                        for z in borders:
                            lick_zone_only.append(licks[(z[0] <= licks[:, 2]) & (licks[:, 2] <= z[1])])
                        zone_licks = np.vstack(lick_zone_only)
                        # the length of the zone-only licks divided by the all-licks is the zone-lick ratio
                        lick_count_ratio = zone_licks.shape[0] / licks.shape[0]

                        # correct by fraction of reward zones where the mouse actually licked
                        passed_rzs = len([x for x in lick_zone_only if len(x) > 0]) / len(borders)

                        # If passed_zones was not provided, use lick-based correction
                        if passed_zones is None:
                            lick_count_ratio = lick_count_ratio * passed_rzs
                        # Otherwise, use the valve-based correction
                        else:
                            lick_count_ratio = lick_count_ratio * passed_zones

                    else:
                        lick_count_ratio = np.nan
                        passed_rzs = 0

                # If passed zones was not provided, return the lick-based correction
                if passed_zones is None:
                    return lick_count_ratio, passed_rzs
                else:
                    return lick_count_ratio

            def get_binned_ratio(lick_data: np.ndarray, orig_borders: np.ndarray, borders: np.ndarray,
                                 passed_zones: float, bin_size: float) -> Tuple[float, float]:
                """
                Compute performance value of one trial based on ratio of binned licks, and get mean distance of licking
                bins to the next reward zone.

                Args:
                    lick_data: 2D Numpy array with shape (n_samples, n_metrics) of behavioral data. From get_array().
                        Columns have to be time - lick - pos - enc.
                    orig_borders: 2D Numpy array with shape (n_zones, 2), containing RZ borders in VR coordinates.
                        From get_zone_borders().
                    borders: Same as orig_borders, but zones are extended by the buffer set in PerformanceParameter().
                    passed_zones: Fraction of reward zones successfully passed in this trial.
                    bin_size: Size of position bins in VR coordinates. From PerformanceParameter().

                Returns:
                    Ratio of licked bins inside RZs vs total licked bins, corrected by passed reward zones
                    Mean distance of licking bins to next RZ (in VR units), corrected by passed reward zones
                """
                licked_rz_bins = 0
                licked_nonrz_bins = 0

                bins = np.arange(start=-10, stop=110, step=bin_size)  # create bin borders for position bins

                # Get first bins of each reward zone
                start_bins = np.array([x[0]+11 for x in orig_borders])  # +10 to offset -10 start, +1 because bins start counting at 1
                # Extrapolate next RZ after last one (1st in next trial) for licks that are after the last RZ
                start_bins = np.array((*start_bins, start_bins[-1]+(120-start_bins[-1]+start_bins[0])))

                zone_bins = []
                for z in borders:
                    zone_bins.extend(np.arange(start=z[0], stop=z[1], step=bin_size))
                bin_idx = np.digitize(lick_data[:, 2], bins)

                # Go through all position bins
                min_dist = []
                for curr_bin in np.unique(bin_idx):
                    # Check if any lick started at the current bin
                    if np.any(np.diff(lick_data[np.where(bin_idx == curr_bin)[0], 1]) == 1):
                        # If yes, check if the bin is part of a reward zone
                        if bins[curr_bin - 1] in zone_bins:
                            licked_rz_bins += 1     # if yes, the current bin was RZ and thus correctly licked in
                            dist_to_next = 0        # the distance to the next RZ is 0
                        else:
                            licked_nonrz_bins += 1  # if no, the current bin was not RZ and thus incorrectly licked in
                            dist = start_bins - curr_bin
                            dist_to_next = np.min(dist[dist > 0])    # Get distance to next RZ (only positive distances)
                        min_dist.append(dist_to_next)
                try:
                    # Ratio of RZ bins that were licked vs total number of licked bins, normalized by factor of passed RZs
                    binned_lick_ratio = (licked_rz_bins / (licked_rz_bins + licked_nonrz_bins)) * passed_zones
                except ZeroDivisionError:
                    binned_lick_ratio = 0

                # Calculate mean distance. Correct by number of passed reward zones
                distance_to_next = np.mean(min_dist) / passed_zones

                return binned_lick_ratio, distance_to_next

            def get_stop_ratio(lick_data: np.ndarray, borders: np.ndarray, vel_thresh: int, stop_time: int) -> float:
                """
                Compute performance value of one trial based on ratio of stops in and outside of RZs.

                Args:
                    lick_data: 2D Numpy array with shape (n_samples, n_metrics) of behavioral data. From get_array().
                        Columns have to be time - lick - pos - enc.
                    borders: 2D Numpy array with shape (n_zones, 2), containing buffed RZ borders in VR coordinates.
                        From get_zone_borders()
                    vel_thresh: Velocity threshold under which it counts as "stop". From PerformanceParameter().
                    stop_time: Duration threshold of low speed above which it counts as "stop".
                        From PerformanceParameter().

                Returns:
                    float, ratio of stops inside RZs vs total stops, corrected by passed reward zones
                """
                # select only time points where the mouse was not running (from params (in cm/s) divided by encoder factor)
                stop_only = lick_data[(-vel_thresh / 2.87 <= lick_data[:, 3]) & (lick_data[:, 3] <= vel_thresh / 2.87)]
                # split into discrete stops
                diff = np.round(np.diff(stop_only[:, 0]) * 1000).astype(int)  # get an array of time differences in ms
                stops = np.split(stop_only,np.where(diff > SAMPLE * 1000)[0] + 1)  # split where difference > sample gap
                # select only stops that were longer than the specified stop time
                stops = [i for i in stops if i.shape[0] >= stop_time / (SAMPLE * 1000)]
                # select only stops that were inside a reward zone (min or max position was inside a zone border)
                zone_stop_only = []
                for z in borders:
                    zone_stop_only.append([i for i in stops if z[0] <= np.max(i[:, 1]) <= z[1] or
                                           z[0] <= np.min(i[:, 1]) <= z[1]])
                # the number of the zone-only stops divided by the number of the total stops is the zone-stop ratio
                zone_stops = np.sum([len(i) for i in zone_stop_only])
                stopping_ratio = zone_stops / len(stops)

                return stopping_ratio

            # Fetch required behavior data as array (columns: time - lick - pos - enc)
            data = self.get_array(attr=('lick', 'pos', 'enc'))

            # Get reward zone borders for the current trial and add the buffer
            zone_borders = self.get_zone_borders()
            buf_zone_borders = zone_borders.copy()
            buf_zone_borders[:, 0] -= params['vrzone_buffer']
            buf_zone_borders[:, 1] += params['vrzone_buffer']

            # Find out which reward zones were passed (reward given) if parameter is set (default no)
            if params['valve_for_passed']:
                rz_passed = np.zeros(len(buf_zone_borders))
                for idx, zone in enumerate(buf_zone_borders):
                    # Get the reward entries at indices where the mouse is in the current RZ
                    valve = self.fetch1('valve')
                    rz_data = valve[np.where(np.logical_and(data[:, 2] >= zone[0], data[:, 2] <= zone[1]))]
                    # Cap reward at 1 per reward zone (ignore possible manual water rewards given)
                    if rz_data.sum() >= 1:
                        rz_passed[idx] = 1
                    else:
                        rz_passed[idx] = 0

                passed_rz = rz_passed.sum() / len(buf_zone_borders)

                ### GET LICKING DATA ###
                count_ratio = get_lick_count_ratio(data, buf_zone_borders, passed_rz)
            else:
                count_ratio, passed_rz = get_lick_count_ratio(data, buf_zone_borders)

            ### GET BINNED LICKING PERFORMANCE
            bin_ratio, distance = get_binned_ratio(lick_data=data, orig_borders=zone_borders, borders=buf_zone_borders,
                                                   passed_zones=passed_rz, bin_size=params['bin_size'])

            ### GET STOPPING DATA ###
            stop_ratio = get_stop_ratio(lick_data=data, borders=zone_borders, vel_thresh=params['velocity_thresh'],
                                        stop_time=params['stop_time'])

            return distance, bin_ratio, count_ratio, stop_ratio

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
            length = (VRSessionInfo & self.restriction[0]).fetch1('length')
            mean_speed = length / time

            # Get mean running speed by filtering out time steps where mouse is stationary
            vel = self.enc2speed()  # Get velocity in cm/s
            running_vel = vel[vel >= params['velocity_thresh']]
            mean_running_speed = np.mean(running_vel)

            return mean_speed, mean_running_speed, time

        @staticmethod
        def get_condition(key: dict, task: str, condition_switch: List[int]) -> Tuple[str, int]:
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
            if (VRLog & key).is_session_novel():
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
                                    'Task condition could not be determined for trial nb {}.'.format(key,
                                                                                                     key['trial_id']))

            # Two condition switches, and the trial is after the first but before the second, or after the second switch
            elif task == 'No pattern and tone' and (len(condition_switch) == 2) and (
                    key['trial_id'] >= condition_switch[1]):
                pattern = 'none'
                tone = 0
            elif task == 'No pattern and tone' and (len(condition_switch) == 2) and (
                    key['trial_id'] < condition_switch[1]):
                pattern = 'none'
                tone = 1
            else:
                raise Exception('Error at {}:\n'
                                'Task condition could not be determined for trial nb {}.'.format(key, key['trial_id']))

            return pattern, tone

    def make(self, key: dict) -> None:
        """
        Fills VRSession and VRSession.VRTrial with temporally aligned behavior parameters for all trials of each
        VRSessionInfo() entry.

        Args:
            key: Primary keys of current VRSessionInfo() entry
        """

        print('Starting to align trials for session {}'.format(key))

        # Fetch data about the session
        trial_ids = (RawBehaviorFile() & key).fetch('trial_id')  # Trial IDs (should be regularly counting up)
        imaging = bool((VRSessionInfo & key).fetch1('imaging_session'))  # Flag if frame trigger should be created
        cond = (common_exp.Session & key).fetch1('task')  # Task condition
        cond_switch = (VRSessionInfo & key).fetch1('condition_switch')  # First trial of new condition

        # Detect if there are different numbers of behavior vs imaging trials
        if imaging and len(trial_ids) != len(common_img.RawImagingFile & key):
            raise ValueError(f'Found {len(trial_ids)} behavior trials, but {len(common_img.RawImagingFile & key)} '
                             f'imaging trials in an imaging session!')

        trial_entries = []

        for trial_id in trial_ids:
            # Initialize relevant variables
            trial_key = dict(**key, trial_id=trial_id)  # dict containing the values of all trial attributes
            frame_count = None  # frame count of the imaging file of that trial
            if imaging:
                frame_count = (common_img.RawImagingFile & dict(**key, part=trial_id)).fetch1('nr_frames')

            # Find out which condition this trial was
            trial_key['pattern'], trial_key['tone'] = self.VRTrial().get_condition(trial_key, cond, cond_switch)

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
            merge = self.align_behavior_files(trial_key, encoder=data['enc'], position=data['tcp'], trigger=data['tdt'],
                                              imaging=imaging, frame_count=frame_count)

            # Transform lick and valve to event-time indices instead of continuous sampling to save disk space
            # lick_idx = np.where(merge[:, 2] == 1)[0]
            # frame_idx = np.where(merge[:, 3] == 1)[0]
            # valve_idx = np.where(merge[:, 5] == 1)[0]

            # Disk usages for sample session M93, 20210708: (with dj.size_on_disk)
            # All continuous, int8: 65536
            # All continuous, int: 65536
            # All idx, int32: 98304
            # Frame continuous, rest idx: 65536
            # Valve continuous, rest idx: 98304
            # Lick continuous, rest idx: 81920
            # -> Datajoint stores small integers more efficiently than large integers, so continuous sampling is better

            if merge is not None:
                # parse columns into entry dict and typecast to int to save disk space
                trial_key['pos'] = merge[:, 1].astype(np.float32)
                trial_key['lick'] = merge[:, 2].astype(int)
                trial_key['frame'] = merge[:, 3].astype(int)
                trial_key['enc'] = -merge[:, 4].astype(int)  # encoder is installed upside down, so reverse sign
                trial_key['valve'] = merge[:, 5].astype(int)
            else:
                raise ValueError("Alignment returned None, check trial!")

            # Add entry to entry list
            trial_entries.append(trial_key)

        # Insert entry of the master table into the database
        self.insert1(key)

        # Insert trial entries into the part table
        self.VRTrial().insert(trial_entries)

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
        if not type(trial_key['day']) == str:
            date = datetime.strftime(trial_key['day'], '%Y%m%d')
        else:
            date = trial_key['day'].replace('-', '')
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

            if np.sum(trigger[1:, 1]) * 2 == frame_count:
                raise IndexError(f'Trigger file has only half the frames than the TIFF file '
                                 f'({int(np.sum(trigger[1:, 1]))} vs. {frame_count})\nThis is most likely due to the '
                                 f'TIFF file having two channels, where only one is expected.\nDelete the entries in '
                                 f'RawImagingFile for this session, deinterleave the two channels in the TIFF file, and'
                                 f' repopulate the RawImagingFile entry.')

            if abs(more_frames_in_TDT) > 5:
                print("Trial {}:\n{} more frames imported from TDT than in raw imaging file "
                      "({} frames)".format(trial_key, more_frames_in_TDT, frame_count))


            # This means that the TDT file has less frames than the TIFF file (common)
            if more_frames_in_TDT < 0:
                # first check if TDT stopped logging earlier than TCP
                tdt_offset = position[-1, 0] - trigger[-1, 0]
                # if all missing frames would fit in the offset (time where tdt was not logging), print out warning
                if tdt_offset / 0.033 > abs(more_frames_in_TDT):
                    print('TDT not logging long enough, too long trial? Check trial!')
                # if TDT file had too little frames, they are assumed to have been recorded before TDT logging
                # these frames are added after merge array has been filled
                frames_to_prepend = abs(more_frames_in_TDT)

            # This means that the TDT file has more frames than the TIFF file (rare)
            elif more_frames_in_TDT > 0:
                # if TDT included too many frames, its assumed that the false-positive frames are from the end of recording
                if more_frames_in_TDT < 5:
                    for i in range(more_frames_in_TDT):
                        trigger[trig_blocks[-i], 1] = 0
                else:
                    raise ImportError(
                        f'{more_frames_in_TDT} too many frames imported from TDT, could not be corrected!')

            # This section deals with how to add the missing frames to the trigger array
            if frames_to_prepend > 0:

                # We first have to do some preprocessing to figure out which method of frame insertion is best

                # Get the index of the first recorded frame
                first_frame = np.where(trigger[1:, 1] == 1)[0][0] + 1

                # Get the median distance between frames, and correct if its off
                median_frame_time = int(np.median([len(frame) for frame in trig_blocks]))
                if median_frame_time > 70:
                    print(f'Median distance between frames is {median_frame_time}! Maybe check file for errors.')
                    median_frame_time = 68

                # Get 99th quantile of TDT sample time to conservatively estimate if 2 frames would be within the
                # 8 ms bin window (and thus lost during resampling)
                tdt_time_diff = data[0][2:, 0] - data[0][1:-1, 0]
                tdt_sample_time = np.quantile(tdt_time_diff, 0.99)
                tdt_step_limit = int(np.ceil(0.008 / tdt_sample_time))

                # Find gaps in frame trigger where frames might have been missed (distance between frames > 45 ms)
                frames_idx = np.where(trigger[1:, 1] == 1)[0]
                frame_dist = frames_idx[1:] - frames_idx[:-1]

                # 95 sample steps (~45 ms) is much higher than two normal frames should be apart
                # The gap is AFTER the frames at these indices, +1 because we ignore first line during slicing
                frame_gap_idx = frames_idx[np.where(frame_dist > 95)[0]] + 1

                if len(frame_gap_idx) > 0:
                    # Fill large gaps if any were found
                    # First, fill all large gaps with new frames (or enough to offset the mismatch)
                    frames_to_add = frames_to_prepend if len(frame_gap_idx) >= frames_to_prepend else len(frame_gap_idx)

                    # Get the indices of gaps, sorted by gap size
                    frame_dist_argsort = np.argsort(frame_dist)
                    # Get the index of the start of the largest gaps in "trigger" array (again +1)
                    largest_gap_start = frames_idx[frame_dist_argsort[-frames_to_add:]] + 1
                    # Get the distance in samples of these gaps and divide them by 2 to find middle point
                    largest_gap_middle = frame_dist[frame_dist_argsort[-frames_to_add:]] // 2

                    # Safety check that the smallest gap is still larger than the step limit, then insert frames there
                    # Only use gaps whose middle point is larger than the step limit
                    gap_mask = largest_gap_middle > tdt_step_limit
                    # Add gap start and half-size of gap to get indices of new frames
                    new_idx = largest_gap_start[gap_mask] + largest_gap_middle[gap_mask]
                    trigger[new_idx, 1] = 1

                    # Re-calculate frames_to_prepend to see if any more frames have to be added
                    frames_to_prepend = int(frame_count - np.sum(trigger[1:, 1]))

                # If there are still frames missing, prepend frames to the beginning of the file
                if frames_to_prepend > 0:
                    if first_frame > frames_to_prepend * median_frame_time:
                        # if frames would fit in the merge array before the first recorded frame, prepend them with proper steps
                        # make a list of the new indices (in steps of median_frame_time before the first frame)
                        idx_start = first_frame - frames_to_prepend * median_frame_time
                        # add 1 to indices because we do not count the first index of the trigger signal
                        idx_list = np.arange(start=idx_start + 1, stop=first_frame, step=median_frame_time)
                        if idx_list.shape[0] != frames_to_prepend:
                            raise ValueError(f'Frame correction failed for {trial_key}!')
                        else:
                            trigger[idx_list, 1] = 1

                    # Try to prepend frames to the beginning of the file
                    elif frames_to_prepend < 30:

                        # int rounds down and avoids overestimation of step size
                        max_step_size = int(first_frame / frames_to_prepend)

                        # If the time before the first frame is not enough to fit all frames in there without crossing the
                        # step limit, we interleave the new frames in the beginning half-way between the original frames
                        if max_step_size <= tdt_step_limit:

                            # Check where the actual frame triggers are and find half-way points
                            actual_frame_idx = np.where(trigger[1:, 1] == 1)[
                                                   0] + 1  # +1 because we ignore the first row
                            # Get the distance between the first "n_frames_to_prepend" frames where we have to interleave fake frames
                            interframe_idx = actual_frame_idx[1:frames_to_prepend + 1] - actual_frame_idx[
                                                                                         :frames_to_prepend]
                            # The new indices are the first actual frame indices plus half of the interframe indices
                            idx_list = actual_frame_idx[:frames_to_prepend] + interframe_idx // 2

                        # Otherwise, put them before the beginning at the biggest possible step size
                        else:
                            # Create indices of new frame triggers with given step size, going backwards from the first real frame
                            idx_list = np.array([i for i in range(first_frame - max_step_size, 0, -max_step_size)])

                        # Exit condition: New indices are somehow not enough
                        if len(idx_list) != frames_to_prepend:
                            raise ValueError(f"Could not prepend {frames_to_prepend} frames for {trial_key}, because "
                                             f"only {len(idx_list)} new indices could be generated.")
                        else:
                            trigger[idx_list, 1] = 1

                    else:
                        # correction does not work if the whole log file is not large enough to include all missing frames
                        raise ImportError(
                            f'{int(abs(more_frames_in_TDT))} too few frames imported from TDT, could not be corrected.')
                else:
                    if abs(more_frames_in_TDT) > 5:
                        print('Found enough gaps in frame trigger to add missing frames')

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

        # 1 if any of grouped values are 1 avoids loosing frames. Sometimes, even downsampling can create NaNs.
        # They are forward filled for now, which will fail (create an unwanted frame trigger) if the timepoint before
        # the NaN happens to be a frame trigger. Lets hope that never happens.
        df_list[0] = df_list[0].resample("8L").max().fillna(0).astype(int)
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
        # Load LOG file
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
            raise ValueError(
                f'Frame count matching unsuccessful: {merge_trig} frames in merge, should be {frame_count} frames.')

        # transform back to numpy array for saving
        time_passed = merge_filt.index - merge_filt.index[0]  # transfer timestamps to
        seconds = np.array(time_passed.total_seconds())  # time change in seconds
        array_df = merge_filt[['position', 'licking', 'trigger', 'encoder', 'water']]  # change column order
        array = np.hstack((seconds[..., np.newaxis], np.array(array_df)))  # combine both arrays

        return array

    def get_normal_trials(self) -> np.ndarray:
        """
        Convenience function that returns of one queried session the trial IDs with normal corridor (training pattern,
        with tone). Used to more easily compute performances, which generally should exclude validation trials.

        Returns:
            Array with trial IDs of normal (no validation condition) trials.
        """
        if len(self) > 1:
            raise IndexError('Query only one session at a time.')

        trial_ids = np.sort((self.VRTrial() & self.restriction).fetch('trial_id'))  # For some reason, trial_ids are not sorted after querying
        switch = (VRSessionInfo & self.restriction).fetch1('condition_switch')

        if switch == [-1]:
            return trial_ids
        else:
            return trial_ids[:switch[0]]

    def plot_lick_histogram(self, bin_size: int = 1, ignore_validation: bool = True) -> None:
        """
        Create a figure with multiple licking histograms of a single mouse. Each session has its own subplot. Each bin's
        value represents the percentage of trials during which the mouse licked at least once in this position. Reward
        zones are marked in red overlays. This plot is useful to examine the raw licking behavior instead of a single
        number like performance.

        Args:
            bin_size: Size of VR position bins (in arbitrary VR units) in which to bin the licks.
            ignore_validation: Bool flag whether to ignore validation trials.
        """

        def draw_single_histogram(sess_key: dict, curr_ax: plt.Axes, bin_s: int, label_ax: bool) -> None:
            """
            Subfunction that draws a single licking histogram (of a single session) onto a given axis. Used to construct
            larger figures.

            Args:
                sess_key: Primary keys of the session that should be drawn.
                curr_ax: Pyplot Axes object where the histogram will be drawn in.
                bin_s: Size of VR position bins (in arbitrary VR units) in which to bin the licks.
                label_ax: Bool flag whether X and Y axes should be labelled.
            """
            # Select appropriate trials
            if ignore_validation:
                trials = (VRSession & sess_key).get_normal_trials()
            else:
                trials = (VRSession.VRTrial & sess_key).fetch('trial_id')

            # Bin licking data from these trials into one array and binarize per-trial
            data = [(VRSession.VRTrial & sess_key & f'trial_id={idx}').get_binned_licking(bin_size=bin_s) for idx in trials]
            data = np.vstack(data)
            data[data > 0] = 1  # we only care about that a bin had a lick, not how many

            track_len = (VRSessionInfo & sess_key).fetch1('length')     # We need the track length for X-axis scaling

            # Draw the histogram, with the weights being
            curr_ax.hist(np.linspace(0, track_len, data.shape[1]), bins=data.shape[1],
                         weights=(np.sum(data, axis=0) / len(data)) * 100, facecolor='black', edgecolor='black')

            # Zone borders have to be adjusted for the unbinned x axis
            zone_borders = (VRSession.VRTrial & f'trial_id={trials[0]}' & sess_key).get_zone_borders()
            zone_borders = zone_borders + 10
            for zone in zone_borders * (track_len / 120):
                curr_ax.axvspan(zone[0], zone[1], color='red', alpha=0.3)

            curr_ax.set_ylim(0, 105)
            curr_ax.set_xlim(0, track_len)
            if label_ax:
                curr_ax.set_xlabel('VR position')
                curr_ax.set_ylabel('Licks [%]')
            curr_ax.spines['top'].set_visible(False)
            curr_ax.spines['right'].set_visible(False)

            curr_ax.set_title("M{}, {}, {:.2f}%".format(sess_key['mouse_id'], sess_key['day'],
                                                        (VRPerformance & sess_key).get_mean()[0]*100))

        # Get primary keys of queried sessions
        keys = self.fetch('KEY')
        mouse_ids = np.unique([x['mouse_id'] for x in keys])

        # Make a separate figure for each mouse
        for mouse in mouse_ids:
            curr_keys = [x for x in keys if x['mouse_id'] == mouse]

            # Go through queried sessions
            ncols = 3
            nrows = int(np.ceil(len(curr_keys) / ncols))
            count = 0

            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))
            for row in range(nrows):
                for col in range(ncols):
                    if count < len(curr_keys):
                        if row == nrows - 1:
                            label_axes = True
                        else:
                            label_axes = False
                        draw_single_histogram(curr_keys[count], curr_ax=axes[row, col], bin_s=bin_size, label_ax=label_axes)
                        count += 1
            fig.canvas.set_window_title(f"M{mouse} - Binned licking per session")
            fig.tight_layout()


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
    distance                    : longblob          # np.array of avg distance of licked bins to next reward zone 
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
            key: Primary keys of the union of the current VRSessionInfo and PerformanceParameters entry.
        """

        # Get current set of parameters
        params = (PerformanceParameters & key).fetch1()

        # Initialize dict that will hold single-trial lists (one per (non-primary) attribute)
        trial_data = {key: [] for key in self.heading if key not in self.primary_key}

        trial_ids = (VRSession.VRTrial & key).fetch('trial_id')

        # Process every trial (start with 1 because trial_id is 1-based
        for trial_id in trial_ids:
            # Store query of current trial
            trial = (VRSession.VRTrial & key & 'trial_id={}'.format(trial_id))

            # Compute lick and stop performances
            distance, binned_lick_ratio, lick_count_ratio, stop_ratio = trial.compute_performances(params)

            # Compute time metrics
            mean_speed, mean_running_speed, trial_duration = trial.compute_time_metrics(params)

            # Add trial metrics to data dict, convert to float32 before to save disk space
            trial_data['distance'].append(np.float32(distance))
            trial_data['binned_lick_ratio'].append(np.float32(binned_lick_ratio))
            trial_data['lick_count_ratio'].append(np.float32(lick_count_ratio))
            trial_data['stop_ratio'].append(np.float32(stop_ratio))
            trial_data['mean_speed'].append(np.float32(mean_speed))
            trial_data['mean_running_speed'].append(np.float32(mean_running_speed))
            trial_data['trial_duration'].append(np.float32(trial_duration))

        # Combine primary dict "key" with attributes "trial_data" and insert entry
        self.insert1({**key, **trial_data})

    def get_mean(self, attr: str = 'binned_lick_ratio', ignore_validation: bool = True) -> List[float]:
        """
        Get a list of the mean of a given performance attribute of the queried sessions.
        
        Args:
            attr: Performance metric, must be attribute of VRPerformance()
            ignore_validation: Bool flag whether trials with changed conditions (validation trials) should be excluded
                from the mean. For normal behavior analysis this should be true, as validation trials are to be handled
                separately.

        Returns:
            Means of the performance attribute over the queried sessions, one value per session
        """
        sess = self.fetch(attr)

        if len(np.unique(self.fetch('mouse_id'))) > 1:
            raise dj.errors.QueryError('More than one mouse queried! Only one mouse allowed for VRPerformance.get_mean().')

        if ignore_validation:
            mean_attr = []
            cond_switches = (VRSessionInfo() & self.restriction).fetch('condition_switch')
            for cond_switch, curr_sess in zip(cond_switches, sess):
                if cond_switch == [-1]:
                    mean_attr.append(np.mean(curr_sess))
                else:
                    mean_attr.append(np.mean(curr_sess[:cond_switch[0]]))
            return mean_attr
        else:
            return [np.mean(x) for x in sess]

    def plot_performance(self, attr: str = 'binned_lick_ratio', threshold: Optional[float] = 0.75,
                         mark_spheres: bool = True, x_tick_label: bool = True) -> None:
        """
        Plots performance across time for the queried sessions. Multiple mice included in the query are split up
        across subplots.
        
        Args:
            attr: Performance metric, must be attribute of VRPerformance()
            threshold: Y-intercept of a red dashed line to indicate performance threshold. Must be between 0 and 1. If None, no line is drawn.
            mark_spheres: Bool flag whether to draw a vertical line at the date of microsphere injection.
            x_tick_label: Bool flag whether X-ticks date labels should be drawn. Turning it off can make graphs tidier.
        """
        mouse_id, day, behav = self.fetch('mouse_id', 'day', attr)
        mouse_ids = np.unique(mouse_id)
        df = pd.DataFrame(dict(mouse_id=mouse_id, day=day, behav=behav))
        df_new = df.explode('behav')

        df_new['behav'] = df_new['behav'].astype(float)
        df_new['day'] = pd.to_datetime(df_new['day'])

        if len(mouse_ids) > 2:
            col_wrap = 3
        else:
            col_wrap = len(mouse_ids)
        grid = sns.FacetGrid(df_new, col='mouse_id', col_wrap=col_wrap, height=3, aspect=2, sharex=False)
        grid.map(sns.lineplot, 'day', 'behav')

        for idx, ax in enumerate(grid.axes.ravel()):
            if threshold is not None:
                ax.axhline(threshold, linestyle='--', color='black', alpha=0.5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylabel(attr)
            if mark_spheres:
                try:
                    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' &
                                   f'mouse_id={mouse_ids[idx]}').fetch1('surgery_date').date()
                    ax.axvline(surgery_day, linestyle='--', color='r', alpha=1)
                except dj.errors.DataJointError:
                    print(f'No microsphere surgery on record for mouse {mouse_ids[idx]}')
            if not x_tick_label:
                ax.set_xticklabels([])


@schema
class PerformanceTrend(dj.Computed):
    definition = """ # Trend analysis metrics of performance across trials of a session. LinReg via statsmodels.OLS()
    -> VRPerformance
    ---
    p_normality = NULL    : float           # p-value whether single-trial performance datapoints are normal distributed
                                            # (D'Agostino + Pearson's omnibus test). Can only be reliably determined for 
                                            # >20 trials, otherwise p=1 and assumed non-normality.
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

        # Replace infinity values that cannot be stored in the database
        if r2 == -np.inf:
            r2 = 0

        # Insert entry into the table
        entry = dict(key, p_normality=normality, perf_corr=corr, p_perf_corr=p, perf_r2=r2, prob_lin_reg=p_r2,
                     perf_intercept=intercept, perf_slope=slope, perf_ols_x=x_fit, perf_ols_y=perf)

        self.insert1(entry)
