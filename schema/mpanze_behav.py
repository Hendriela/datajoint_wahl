"""
Schema for processing behavioural data from reach-to-grasp experiments
"""

import datajoint as dj
from schema import common_exp
from mpanze_scripts import utils
import numpy as np
import pathlib
import login
import json
login.connect()

schema = dj.schema('mpanze_behav', locals(), create_tables=True)


@schema
class JoystickSessionInfo(dj.Imported):
    definition = """ # Reference table for joystick experiments, automatically computed (mostly) from parameter files
    -> common_exp.Session
    ----
    cue_type            : varchar(40)       # cue type
    task                : varchar(40)       # task selected from vi
    visual_cue_length   : int               # duration of visual cue, in ms
    upper_threshold     : float             # raw reward threshold in joystick sensor units
    lower_threshold     : float             # start zone boundary, in joystick sensor units
    total_intertrial_time : float           # pre_trial_time + post_trial_time in seconds, time for which the joystick is removed from reach
    pre_trial_time      : float             # baseline duration, in seconds
    post_trial_time     : float             # time elapsed after end of trial, in seconds
    valve_opening_time  : int               # reward duration, in milliseconds
    no_touch_required   : tinyint           # =1 if the mouse must not be touching the joystick to start the task
    sampling_rate       : float             # sampling rate for data acquisition through the NI card
    filename_params     : varchar(256)      # filename of the parameters file, relative to session folder
    """

    def make(self, key):
        # load parameter file
        p_session = pathlib.Path(login.get_working_directory(), (common_exp.Session() & key).fetch1()["session_path"])
        filename_params = pathlib.Path(p_session, utils.filename_from_session(key) + "_params.json")
        with open(str(filename_params)) as f:
            data = json.load(f)

        key_entry = {**key}
        # check if pre_trial/post_trial times are available, if not hardcode them
        # TODO: pre/post trial times as parameters in joystick vi
        if "pre_trial_time" in data:
            key_entry["pre_trial_time"] = data["pre_trial_time"]
            key_entry["post_trial_time"] = data["post_trial_time"]
        else:
            key_entry["pre_trial_time"] = 2
            key_entry["post_trial_time"] = 3

        # populate rest of key
        key_entry["cue_type"] = data["Cue type"]
        key_entry["task"] = data["Task"]
        key_entry["visual_cue_length"] = data["Visual cue length (ms)"]
        key_entry["upper_threshold"] = data["Upper threshold"]
        key_entry["lower_threshold"] = data["Lower threshold"]
        key_entry["total_intertrial_time"] = data["Intertrial time (s)"]
        key_entry["valve_opening_time"] = data["Valve opening  time (ms)"]
        key_entry["no_touch_required"] = int(data["no_touch_required"])
        key_entry["sampling_rate"] = data["sampling rate"]
        key_entry["filename_params"] = filename_params.stem + filename_params.suffix

        # add to table
        self.insert1(key_entry)


@schema
class RawBehaviouralData(dj.Imported):
    definition = """ # Files generated by the joystick vi
    -> JoystickSessionInfo
    ---
    filename_touch          : varchar(128)          # filename of touch readout file, relative to session folder
    filename_joystick       : varchar(128)          # filename of joystick readout file, relative to session folder
    filename_events         : varchar(128)          # filename of events log file, relative to session folder
    """

    def make(self, key):
        key_entry = {**key}
        # generate file names
        session_stem = utils.filename_from_session(key)
        key_entry["filename_touch"] = session_stem + "_touch.bin"
        key_entry["filename_joystick"] = session_stem + "_joystick.bin"
        key_entry["filename_events"] = session_stem + "_events.txt"
        # populate row
        self.insert1(key_entry)

    def get_paths(self, filename):
        return utils.get_paths(self, filename)

    def get_path(self, filename, check_existence=False):
        return utils.get_path(self, filename, check_existence=check_existence)

    def load_events(self):
        """ Read event data from file, return as a single ndarray """
        path_file = self.get_path("filename_events")
        events = np.genfromtxt(path_file, skip_header=1, delimiter="\t", dtype="f8,U15", names="timestamps,strings")
        return events

    def load_touch(self):
        """ Read data from file, return two arrays: one of timestamps and one of touch readouts """
        path_file = self.get_path("filename_touch")
        data = np.fromfile(path_file)
        return data[::2], data[1::2]

    def load_joystick(self):
        """ Read data from file, return three arrays: timestamp, x, y """
        path_file = self.get_path("filename_joystick")
        data = np.fromfile(path_file)
        return data[::3], data[1::3], data[2::3]


@schema
class JoystickSession(dj.Computed):
    definition = """ # Session-specific table, holding part-tables
    -> JoystickSessionInfo
    ---
    time_computed = CURRENT_TIMESTAMP   : timestamp     # automatic timestamp
    n_trials                            : smallint      # total trials in session
    """

    class JoystickTrial(dj.Part):
        definition = """ # Trial data
        -> JoystickSession
        trial_id            : smallint      # Counter of the trial in the session, base 0
        ---
        t_start             : float         # start of trial baseline
        t_cue               : float         # time of auditory cue
        t_end               : float         # time of trial end (either by success or time out)
        t_intertrial         : float         # timestamp of intertrial end
        successful          : tinyint       # bool flag whether the animal received a reward
        timestamps          : longblob      # 1d time axis, sampled at 100 Hz
        touch               : longblob      # 1d array with touch sensor readout sampled at 100 Hz
        x                   : longblob      # 1d array of x-axis readout from joystick, sampled at 100 Hz
        y                   : longblob      # 1d array of x-axis readout from joystick, sampled at 100 Hz
        """

    @staticmethod
    def get_trials(events, t_pre, t_post):
        # return trial timestamps as follows (start of pretrial (2s) - start of trial - end of trial - end of intertrial (+3s), flag (whether or not trial was completed))
        # skip 1st as there is no baseline,and cue doesn't work properly on 1st trial due to Labview bug
        # count number of trials
        trial_start_idx = np.where(events["strings"] == "Trial start")[0]
        trial_completed_idx = np.where(events["strings"] == "Trial completed")[0]
        trial_failed_idx = np.where(events["strings"] == "Trial failed")[0]
        trial_end_idx = np.sort(list(trial_completed_idx) + list(trial_failed_idx))

        n_trials = len(trial_start_idx)

        # cut off 1st trial (bugged audio cue) & last trial (may be cut-off early)
        trial_timestamps = []
        for idx in trial_start_idx[1:-1]:
            timestamp = 5 * [0]
            timestamp[1] = events["timestamps"][idx]
            timestamp[0] = timestamp[1] - t_pre  # 2 seconds of baseline
            # find 1st trial end after idx
            idx_end = trial_end_idx[trial_end_idx > idx][0]
            timestamp[2] = events["timestamps"][idx_end]
            timestamp[3] = timestamp[2] + t_post
            if idx_end in trial_completed_idx:
                timestamp[4] = 1
            else:
                timestamp[4] = 0

            trial_timestamps.append(timestamp)

        return np.array(trial_timestamps)

    def make(self, key):
        # load events and get event structure
        events = (RawBehaviouralData() & key).load_events()
        session_info = (JoystickSessionInfo() & key).fetch1()
        trial_timestamps = self.get_trials(events, session_info["pre_trial_time"], session_info["post_trial_time"])
        n_trials = trial_timestamps.shape[0]

        # load and resample traces
        t, x, y = (RawBehaviouralData() & key).load_joystick()
        t, touch = (RawBehaviouralData() & key).load_touch()
        from scipy.signal import resample
        sampling_rate = session_info["sampling_rate"]
        n_samples = int(len(t)*100/sampling_rate)
        x_resample, t_resample = resample(x.astype(np.float32), n_samples, t)
        y_resample, t_resample = resample(y.astype(np.float32), n_samples, t)
        touch_resample = (resample(touch, n_samples).astype(np.float32) > 0.5).astype(np.uint8)
        x_resample = x_resample[1:-1]
        y_resample = y_resample[1:-1]
        touch_resample = touch_resample[1:-1]
        t_resample = t_resample[1:-1].astype(np.float32)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(t, touch)
        # plt.plot(t_resample, touch_resample)
        # plt.figure()
        # plt.plot(t, x)
        # plt.plot(t_resample, x_resample)
        # plt.figure()
        # plt.plot(t, y)
        # plt.plot(t_resample, y_resample)
        # print(x_resample.dtype, y_resample.dtype, t_resample.dtype, touch_resample.dtype)

        trial_keys = []
        # load trials
        for i in range(n_trials):
            trial_key = {**key}
            trial_key["trial_id"] = i
            trial_key["t_start"] = trial_timestamps[i, 0]
            trial_key["t_cue"] = trial_timestamps[i, 1]
            trial_key["t_end"] = trial_timestamps[i, 2]
            trial_key["t_intertrial"] = trial_timestamps[i, 3]
            trial_key["successful"] = int(trial_timestamps[i, 4])

            # slice arrays
            idx = (t_resample >= trial_timestamps[i, 0]) & (t_resample <= trial_timestamps[i, 3])
            trial_key["timestamps"] = t_resample[idx]
            trial_key["touch"] = touch_resample[idx]
            trial_key["x"] = x_resample[idx]
            trial_key["y"] = y_resample[idx]

            trial_keys.append(trial_key)

        key["n_trials"] = n_trials
        self.insert1(key)
        self.JoystickTrial().insert(trial_keys)
