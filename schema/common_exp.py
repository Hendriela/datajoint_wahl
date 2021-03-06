"""Schema for experimental information"""

import datajoint as dj
import login
from schema import common_mice
from pathlib import Path
import os
from datetime import datetime
from datetime import date as datetime_date
from typing import Union, Optional

schema = dj.schema('common_exp', locals(), create_tables=True)


@schema
class Anesthesia(dj.Lookup):
    definition = """ # Anesthesia Info
    anesthesia               : varchar(128)   # Anesthesia short name
    ---
    anesthesia_details       : varchar(1024)  # Longer description
    """

    contents = [
        ['Awake', 'Mouse is awake'],
        ['Isoflurane', 'Mouse is under isoflurane gas anesthesia, ~1 Percent maintaince'],
        ['Wakeup', 'During the session, the mouse wakes up from anesthesia']
    ]


@schema
class Setup(dj.Lookup):
    definition = """ # Details of the setup
    setup          : varchar(128)   # Unique name of the setup
    ---
    setup_details  : varchar(1024)  # More info about the components
    """

    contents = [
        ['VR', 'Mouse is headfixed in the VR setup located outside the 2p microscope (no resonant scanner sound)'],
        ['VR_2p', 'Mouse is headfixed in the VR setup located under the 2p microscope (with resonant scanner sound)'],
        ['Mapping', 'Mouse is headfixed in the sensory mapping setup'],
        ['Grasping_widefield', 'Mouse is headfixed in the grasping setup, in the widefield imaging soundproof box']
    ]


@schema
class Task(dj.Lookup):
    definition = """ # Experimental task for the mouse
    task            : varchar(128)      # Unique name of the task
    ---
    -> common_mice.Investigator         # Keep track of whose task it is for GUI
    stage           : tinyint           # Counter for e.g. difficulty in case of learning task
    task_details    : varchar(1048)     # Task description
    """

    contents = [
        # Hendriks VR tasks
        ['Passive', 'hheise', 0, 'Water reward is given passively upon entering a reward zone.'],
        ['Active', 'hheise', 1, 'Water reward is given after active licking in a reward zone.'],
        ['No tone', 'hheise', 2, 'Like active, but tone cue is removed after X trials.'],
        ['No pattern', 'hheise', 2, 'Like "active", but tone cue is removed after X trials.'],
        ['No pattern and tone', 'hheise', 2, 'Like "No pattern", but tone cue is removed after Y trials.'],
        ['No pattern, tone and shifted', 'hheise', 2, 'Like "No pattern and tone", but reward zones shifted additionally.'],
        ['No reward at RZ3', 'hheise', 2, 'Like active, but water reward is disabled at RZ3.'],
        ['Changed distances', 'hheise', 2, 'Like active, but the distances between reward zones are changed.'],
        # Todo: enter Jithins wheel tasks
        # Jithins tasks
        # Todo: enter Matteos wheel tasks
        # Matteos tasks
        ['Sensory mapping', 'mpanze', 0, 'Various stimuli are presented to the anesthetized mouse'],
        ["Joystick_with_cue", "mpanze", 1, "Mouse pulls joystick when auditory cue is presented"],
    ]


@schema
class Session(dj.Manual):
    definition = """ # Information about the session and experimental setup
    -> common_mice.Mouse
    day             : date           # Date of the experimental session (YYYY-MM-DD)
    session_num     : tinyint        # Counter of experimental sessions on the same day (base 1)
    ---
    session_id      : varchar(128)   # Unique identifier
    session_path    : varchar(256)   # Path of this session relative to the Neurophysiology-Storage1 DATA directory
    session_counter : smallint       # Overall counter of all sessions across mice (base 0)
    experimenter    : varchar(128)   # Who actually performed the experiment, must be a username from Investigator()
    -> Anesthesia
    -> Setup
    -> Task
    session_notes   : varchar(2048)  # description of important things that happened
    """

    @staticmethod
    def create_id(investigator_name: str, mouse_id: int, date: Union[datetime, str], session_num: int) -> str:
        """
        Create unique session id with the format inv_MXXX_YYYY-MM-DD_ZZ:
        inv: investigator shortname (called 'experimenter' in Adrians GUI)
        MXXX: investigator-specific mouse number (M + 0-padded three-digit number, e.g. M018)
        YYYY-MM-DD: date of session
        ZZ: 0-padded two-digit counter of the sessions on that day
        Note that trials are with base 1
        Adrian 2019-08-12
        Adapted by Hendrik 2021-05-04
        ------------------------------------------------------------------------------------

        Args:
            investigator_name:  Shortname of the investigator for this session (from common_mice.Investigator)
            mouse_id:           Investigator-specific mouse ID (from common_mice.Mice)
            date:               Datetime object of the session date, or string with format YYYY-MM-DD
            session_num:        Iterator for number of sessions on that day

        Returns:
            Unique string ID for this session

        """

        # first part: mouse identifier
        mouse_id_str = 'M{:03d}'.format(int(mouse_id))
        first_part = 'session_' + investigator_name + '_' + mouse_id_str

        # second: Transform datetime object to string, while removing the time stamp
        if type(date) == datetime or type(date) == datetime_date:
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = date

        # third: trial with leading zeros
        trial_str = '{:02d}'.format(session_num)

        # combine and return the unique session id
        return first_part + '_' + date_str + '_' + trial_str

    @staticmethod
    def get_relative_path(abs_path: Union[Path, str]) -> str:
        """
        Removes the user-specific Neurophysiology-Server DATA path from an absolute path and returns the relative path

        Args:
            abs_path: Absolute path of a directory in the user's Neurophysiology-Storage1 server DATA directory

        Returns:
            Relative path with the machine-specific Neurophysiology-DATA-Path removed
        """

        # Get main data directory on the Neurophys-Server, and possible other data directories
        cwd = [Path(login.get_neurophys_data_directory()), *[Path(x) for x in login.get_alternative_data_directories()]]

        # Typecast absolute path to a Path object to easily get parents
        if type(abs_path) == str:
            abs_path = Path(abs_path)

        # Look through possible directories for the session folder
        for wd in cwd:
            if wd in abs_path.parents:
                # If the session path is inside the Neurophys data directory, transform it to a relative path
                return os.path.relpath(abs_path, wd)
            elif wd == abs_path:
                raise NameError('\nAbsolute session path {} cannot be the same as the Neurophys data directory.\nCreate'
                                ' a subfolder inside the Neurophys data directory for the session.'.format(abs_path))

        raise ImportError(
            '\nAbsolute session path {} \ndoes not seem to be in any of the possible DATA directories:\n{}. '
            'Make sure that the session path, the \nlocal server directory in '
            'login.get_neurophys_data_directory() and other possible locations in\n'
            'login.get_alternative_data_directories() are set correctly.'.format(abs_path, cwd))

    def helper_insert1(self, entry_dict: dict) -> str:
        """
        Simplified insert function that takes care of id and counter values.
        Adrian 2019-08-19

        Args:
            entry_dict: Dictionary containing all key, value pairs for the session except for
                                the id and counter

        Returns:
            Status update string confirming successful insertion.
        """

        # Make copy so that changes do not affect original dict
        new_entry_dict = entry_dict.copy()

        sess_id = self.create_id(new_entry_dict['username'], new_entry_dict['mouse_id'], new_entry_dict['day'],
                                 new_entry_dict['session_num'])
        if len(self.fetch('session_counter')) == 0:
            counter = 0
        else:
            counter = max(self.fetch('session_counter')) + 1

        # Transform absolute path from the GUI to the relative path on the Neurophys-Storage1 server
        new_entry_dict['session_path'] = self.get_relative_path(new_entry_dict['session_path'])

        # add automatically computed values to the dictionary
        entry = dict(**new_entry_dict, session_id=sess_id, session_counter=counter)

        self.insert1(entry)
        # Only print out primary keys
        key_dict = {your_key: entry[your_key] for your_key in ['username', 'mouse_id', 'day', 'session_num']}
        return 'Inserted new session: {}'.format(key_dict)

    def get_absolute_path(self, filename: Optional[str] = None) -> str:
        """
        Return the folder on the current user's Neurophys data directory or any other possible directories for this
        session on the current PC.
        Adrian 2020-12-07

        Returns:
            Absolute path on the Neurophys data directory of the current 'session_path'.
        """

        # Get all possible data directories
        roots = [login.get_neurophys_data_directory(), *login.get_alternative_data_directories()]
        rel_path = self.fetch1('session_path')

        for root in roots:
            path = os.path.join(root, rel_path)
            if filename is None:
                if os.path.isdir(path):
                    # print(f'Found session folder {rel_path} in root {root}.')
                    return path
            else:
                if os.path.isfile(os.path.join(path, filename)):
                    # print(f'Found file {filename} in session folder {rel_path} in root {root}.')
                    return path

        # If we went through all root directories without hitting a return, the given folder cannot be found anywhere
        raise NameError(f'Cannot find directory {rel_path} in any of the possible root directories:\n{roots}')

    # Commented out because we are (currently) not grouping sessions this way
    # def get_group(self, group_name='?'):
    #     """Return keys of Sessions belonging to a certain group.
    #     Parameters
    #     ---------
    #     group_name: str or list of str
    #         Allowed group names are defined in shared.Group
    #     Adrian 2020-03-23
    #     """
    #     if type(self) in [str, list]:
    #         group_name = self  # argument self was not given
    #
    #     if group_name == '?':
    #         print('Available groups:', shared.Group.fetch('group'))
    #         return
    #
    #     if type(group_name) == str:
    #         group_name = [group_name]  # convert to list for compatibility with lists
    #
    #     keys = list()
    #     for name in group_name:
    #         if name not in shared.Group.fetch('group'):
    #             raise Exception('The group name "{}" is not an entry of shared.Group.'.format(name))
    #
    #         keys.extend((Session() & (shared.GroupAssignment() & {'group': name})).fetch(dj.key))
    #     return keys

    def get_key(self, counter: int) -> dict:
        """
        Return uniquely identifying primary keys that corresponds to global counter of sessions.

        Args:
            counter: Overall counter of all sessions across mice (base 0).

        Returns:
            Primary keys of the session with the provided counter.
        """
        return (self & {'session_counter': counter}).fetch1('KEY')
