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
            key["pre_trial_time"] = 2
            key["post_trial_time"] = 3

        # populate rest of key
        key_entry["cue_type"] = data["Cue type"]
        key_entry["task"] = data["Task"]
        key_entry["visual_cue_length"] = data["Visual cue length (ms)"]
        key_entry["upper_threshold"] = data["Upper threshold"]
        key_entry["lower_threshold"] = data["Lower threshold"]
        key_entry["total_intertrial_time"] = data["Intertrial time (s)"]
        key_entry["valve_opening_time"] = data[""]
        key_entry["filename_params"] = filename_params.stem + filename_params.suffix

        # add to table
        self.insert1(key_entry)


@schema
class RawJoystickFile(dj.Manual):
    definition = """ # Raw .bin files containing timestamped raw traces from joystick sensor
    -> JoystickSession
    -----
    filename_joystick : varchar(128)        # filename relative to session folder
    """

    def get_paths(self):
        return utils.get_paths(self, "filename_joystick")

    def get_path(self, check_existence=False):
        return utils.get_path(self, "filename_joystick", check_existence=check_existence)

    def get_raw_data(self):
        """ Read data from file, return three arrays: timestamp, x, y """
        path_file = self.get_path()
        data = np.fromfile(path_file)
        return data[::3], data[1::3], data[2::3]


@schema
class RawEventsFile(dj.Manual):
    definition = """ # .txt files containing timestamped events from labview vi
    -> JoystickSession
    -----
    filename_events : varchar(128)        # filename relative to session folder
    """

    def get_paths(self):
        return utils.get_paths(self, "filename_events")

    def get_path(self, check_existence=False):
        return utils.get_path(self, "filename_events", check_existence=check_existence)

    def get_raw_data(self):
        """ Read data from file, return as a single ndarray """
        path_file = self.get_path()
        events = np.genfromtxt(path_file, skip_header=1, delimiter="\t", dtype="f8,U15", names="timestamps,strings")
        return events
