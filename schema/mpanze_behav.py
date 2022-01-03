"""
Schema for processing behavioural data from reach-to-grasp experiments
"""

import datajoint as dj
from schema import common_exp
from mpanze_scripts import utils
import numpy as np
import login
login.connect()

schema = dj.schema('mpanze_behav', locals(), create_tables=True)


@schema
class JoystickSession(dj.Manual):
    definition = """ # Reference table for joystick experiments
    -> common_exp.Session
    -----
    """


@schema
class RawTouchFile(dj.Manual):
    definition = """ # Raw .bin files containing timestamped raw traces from touch sensor
    -> JoystickSession
    -----
    filename_touch          : varchar(128)        # filename relative to session folder
    """

    def get_paths(self):
        return utils.get_paths(self, "filename_touch")

    def get_path(self, check_existence=False):
        return utils.get_path(self, "filename_touch", check_existence=check_existence)

    def get_raw_data(self):
        """ Read data from file, return two arrays: one of timestamps and one of touch readouts """
        path_file = self.get_path()
        data = np.fromfile(path_file)
        return data[::2], data[1::2]


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
