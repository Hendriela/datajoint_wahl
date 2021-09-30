"""Schema for experimental information"""

import datajoint as dj
import login
from schema import common_mice
from pathlib import Path
import os
from datetime import datetime

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
        ['No reward at RZ3', 'hheise', 2, 'Like active, but water reward is disabled at RZ3.'],
        ['Changed distances', 'hheise', 2, 'Like active, but the distances between reward zones are changed.'],
        # Todo: enter Jithins wheel tasks
        # Jithins tasks
        # Todo: enter Matteos wheel tasks
        # Matteos tasks
        ['Sensory mapping', 'mpanze', 0, 'Various stimuli are presented to the anestethized mouse'],
    ]


@schema
class Session(dj.Manual):
    definition = """ # Information about the session and experimental setup
    -> common_mice.Mouse
    day             : date           # Date of the experimental session (YYYY-MM-DD)
    session_num     : tinyint        # Counter of experimental sessions on the same day (base 1)
    ---
    session_id      : varchar(128)   # Unique identifier
    session_path    : varchar(256)   # Relative path of this session on the Neurophysiology-Storage1 server
    session_counter : smallint       # Overall counter of all sessions across mice (base 0)
    experimenter    : varchar(128)   # Who actually performed the experiment, must be a username from Investigator()
    -> Anesthesia
    -> Setup
    -> Task
    session_notes   : varchar(2048)  # description of important things that happened
    """

    def create_id(self, investigator_name: str, mouse_id: int, date: datetime, session_num: int) -> str:
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
            date:               Date of the session in format YYYY-MM-DD
            session_num:        Iterator for number of sessions on that day

        Returns:
            Unique string ID for this session

        """

        # first part: mouse identifier
        mouse_id_str = 'M{:03d}'.format(int(mouse_id))
        first_part = 'session_' + investigator_name + '_' + mouse_id_str

        # second: date to string
        date_str = str(date)

        # third: trial with leading zeros
        trial_str = '{:02d}'.format(session_num)

        # combine and return the unique session id
        return first_part + '_' + date_str + '_' + trial_str

    def get_relative_path(self, abs_path: str) -> str:
        """
        Removes the Neurophysiology-Server path from an absolute path and returns the relative path

        Args:
            abs_path: Absolute path of a directory on the Neurophysiology-Storage1 server

        Returns:
            Relative path with the machine-specific Neurophysiology-Path removed
        """

        dir = Path(login.get_working_directory())  # get current working directory (defaults to local path to neurophys)

        if dir in abs_path.parents:
            # If the session path is inside the Neurophys data directory, transform it to a relative path
            return Path(os.path.relpath(abs_path, dir))
        elif dir == abs_path:
            raise NameError('\nAbsolute session path {} cannot be the same as the Neurophys data directory.\n '
                            'Create a subdfolder inside the Neurophys data directory for the session.'.format(abs_path))
        else:
            raise Warning('\nAbsolute session path {} \ndoes not seem to be on the main Neurophys server directory. '
                          'Make sure that the session path and the \nlocal server directory in '
                          'login.get_neurophys_data_directory() are set correctly.\n'
                          'Absolute path used for now.'.format(abs_path))

    def helper_insert1(self, new_entry_dict: dict) -> str:
        """
        Simplified insert function that takes care of id and counter values.
        Adrian 2019-08-19

        Args:
            new_entry_dict: Dictionary containing all key, value pairs for the session except for
                                the id and counter

        Returns:
            Status update string confirming successful insertion.

        """

        id = self.create_id(new_entry_dict['username'], new_entry_dict['mouse_id'], new_entry_dict['day'],
                            new_entry_dict['session_num'])
        if len(Session.fetch('session_counter')) == 0:
            counter = 0
        else:
            counter = max(Session.fetch('session_counter')) + 1

        # Transform absolute path from the GUI to the relative path on the Neurophys-Storage1 server
        new_entry_dict['session_path'] = self.get_relative_path(new_entry_dict['session_path'])

        # add automatically computed values to the dictionary
        entry = dict(**new_entry_dict, session_id=id, session_counter=counter)

        self.insert1(entry)
        # Only print out primary keys
        key_dict = {your_key: entry[your_key] for your_key in ['username', 'mouse_id', 'day', 'session_num']}
        return 'Inserted new session: {}'.format(key_dict)

    def get_folder(self) -> str:
        """
        Return the folder on the current user's Neurophys data directory for this session on the current PC
        Adrian 2020-12-07

        Returns:
            Absolute path on the Neurophys data directory of the current 'session_path'.
        """

        # In the current version we save the relative path (excluding base directory) which the user saves in the GUI,
        # including the leading directory separator ('\\') so the absolute path can be recovered by adding both strings
        base_directory = login.get_working_directory()
        path = self.fetch1('session_path')
        return base_directory + path

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