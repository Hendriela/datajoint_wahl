"""Schema for experimental information"""

import datajoint as dj
import common_mice as mice
import os
import login

schema = dj.schema('common_exp', locals(), create_tables=True)


@schema
class Anesthesia(dj.Lookup):
    definition = """ # Anesthesia Info
    anesthesia               : varchar(128)   # Anesthesia short name
    ---
    anesthesia_details       : varchar(1024) # Longer description
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
    ]


@schema
class Task(dj.Lookup):
    definition = """ # Experimental task for the mouse
    task        : varchar(128)     # Unique name of the task
    stage       : int              # Counter for e.g. difficulty in case of learning task
    ---
    task_details : varchar(1048)   # Task description
    """

    contents = [
        # Hendriks VR tasks
        ['Passive', 0, 'Water reward is given passively upon entering a reward zone.'],
        ['Active', 1, 'Water reward is given after active licking in a reward zone.'],
        ['No tone', 2, 'Like active, but tone cue is removed after X trials.'],
        ['No pattern', 2, 'Like "active", but tone cue is removed after X trials.'],
        ['No pattern and tone', 2, 'Like "No pattern", but tone cue is removed after Y trials.'],
        ['No reward at RZ3', 2, 'Like active, but water reward is disabled at RZ3.'],
        ['Changed distances', 2, 'Like active, but the distances between reward zones are changed.'],
        # Todo: enter Jithins wheel tasks
        # Jithins tasks
        # Todo: enter Matteos wheel tasks
        # Matteos tasks
    ]


@schema
class Session(dj.Manual):
    definition = """ # Information about the session and experimental setup
    -> mice.Mouse
    day     : date           # Date of the experimental session (YYYY-MM-DD)
    trial   : int            # Counter of experimental sessions on the same day (base 1)
    ---
    id      : varchar(128)   # Unique identifier
    path    : varchar(256)   # Relative path of this session on the Neurophysiology-Storage1 server
    counter : int            # Overall counter of all sessions across mice (base 0)
    -> Anesthesia
    -> Setup
    -> Task
    notes   : varchar(2048)  # description of important things that happened
    """

    def create_id(self, investigator_name, mouse_id, date, trial):
        """Create unique session id with the format inv_MXXX_YYYY-MM-DD_ZZ:
        inv: investigator shortname (called 'experimenter' in Adrians GUI)
        MXXX: investigator-specific mouse number (M + 0-padded three-digit number, e.g. M018)
        YYYY-MM-DD: date of session
        ZZ: 0-padded two-digit counter of the sessions on that day
        Note that trials are with base 1
        Adrian 2019-08-12
        Adapted by Hendrik 2021-05-04
        ------------------------------------------------------------------------------------
        :param investigator_name: str, shortname of the investigator for this session (from common_mice.Investigator)
        :param mouse_id: int, investigator-specific mouse ID (from common_mice.Mice)
        :param date: datetime, date of the session in format YYYY-MM-DD
        :param trial: int, iterator for number of sessions on that day
        :return: unique string ID for this session
        """

        # Todo: I do not check whether the values for investigator and mouse ID are valid (if this mouse exists in the
        #  Mouse() table. This should be enforced when entering the session through the GUI!

        # first part: mouse identifier
        mouse_id_str = 'M{:03d}'.format(mouse_id)
        first_part = investigator_name + '_' + mouse_id_str

        # second: date to string
        date_str = str(date)

        # third: trial with leading zeros
        trial_str = '{:02d}'.format(trial)

        # combine and return the unique session id
        return first_part + '_' + date_str + '_' + trial_str


    def get_relative_path(self, abs_path):
        """Removes the Neurophysiology-Server path from an absolute path and returns the relative path
        :param abs_path: str, absolute path of a directory on the Neurophysiology-Storage1 server
        :return: relative path with the machine-specific Neurophysiology-Path removed
        """

        dir = login.get_neurophys_directory()   # get machine-specific path from local login file

        if dir in abs_path:
            return abs_path.replace(dir, '')    # the first character is a leading \\, be careful when using it
        else:
            raise Warning('\nAbsolute session path {} \ndoes not seem to be on the main Neurophys server directory. '
                          'Make sure that the session path and the \nlocal server directory in '
                          'login.get_neurophys_directory() are set correctly.\n'
                          'Absolute path used for now.'.format(abs_path))


    def helper_insert1(self, new_entry_dict):
        """Simplified insert function that takes care of id and counter values
        Parameters
        ---------
        new_entry_dict : dict
            Dictionary containing all key, value pairs for the session except for
            the id and counter
        Adrian 2019-08-19
        """

        id = self.create_id(new_entry_dict['experimenter'], new_entry_dict['name'], new_entry_dict['day'],
                            new_entry_dict['trial'])
        counter = max(Session.fetch('counter')) + 1

        # Transform absolute path from the GUI to the relative path on the Neurophys-Storage1 server
        new_entry_dict['path'] = self.get_relative_path(new_entry_dict['path'])

        # add automatically computed values to the dictionary
        entry = dict(**new_entry_dict, id=id, counter=counter)

        self.insert1(entry)
        return 'Inserted new entry: {}'.format(entry)

    def get_folder(self):
        """ Return the folder on neurophys for this session on the current PC
        Adrian 2020-12-07 """

        # In the current version we save the relative path (excluding base directory) which the user saves in the GUI,
        # including the leading directory separator ('\\') so the absolute path can be recovered by adding both strings
        base_directory = login.get_neurophys_directory()
        path = self.fetch1('path')
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

    def get_key(self, counter):
        """Return key that corresponds to counter of sessions """
        return (self & {'counter': counter}).fetch1('KEY')