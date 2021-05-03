"""Schema for experimental information"""

import datajoint as dj
import common_mice as mice
import os

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
    ]


@schema
class Session(dj.Manual):
    definition = """ # Information about the session and experimental setup
    -> mice.Mouse
    day     : date           # Date of the experimental session (YYYY-MM-DD)
    trial   : int            # Counter of experimental sessions on the same day (base 1)
    ---
    id      : varchar(128)   # Unique identifier, e.g. 0A_2019-06-26_01  (0A: mouse Adam (first time A), date, first trial)
    counter : int            # Overall counter of all sessions across mice (base 0)
    -> Anesthesia
    -> Setup
    -> Task
    -> Experimenter
    notes   : varchar(2048)  # description of important things that happened
    """

    def create_id(self, mouse_name, date, trial):
        """Create unique session id from given information (e.g. 0A_2019-06-26_01  (0A: mouse Adam (first time A), date, first trial))
        Note that trials are with base 1
        Adrian 2019-08-12
        """

        # first part: mouse identifier
        letter = mouse_name[0]  # first letter
        mice_with_letter = mice.Mouse() & 'name like "{}%%"'.format(letter)
        if len(mice_with_letter) == 0:
            raise Exception('No mouse found that starts with this letter.')
        nr = len(mice_with_letter) - 1
        first_part = str(nr) + letter

        # second: date to string
        date_str = str(date)

        # third: trial with leading zeros
        trial_str = '{:02d}'.format(trial)

        # combine and return the unique session id
        return first_part + '_' + date_str + '_' + trial_str

    def helper_insert1(self, new_entry_dict):
        """Simplified insert function that takes care of id and counter values
        Parameters
        ---------
        new_entry_dict : dict
            Dictionary containing all key, value pairs for the session except for
            the id and counter
        Adrian 2019-08-19
        """

        id = self.create_id(new_entry_dict['name'], new_entry_dict['day'], new_entry_dict['trial'])
        counter = max(Session.fetch('counter')) + 1

        # add automatically computed values to the dictionary
        entry = dict(**new_entry_dict,
                     id=id,
                     counter=counter)

        self.insert1(entry)
        return 'Inserted new entry: {}'.format(entry)

    def get_folder(self):
        """ Return the folder on neurophys for this session on the current PC
        Adrian 2020-12-07 """
        base_directory = login.get_neurophys_directory()
        id = self.fetch1('id')
        return os.path.join(base_directory, id)

    def get_group(self, group_name='?'):
        """Return keys of Sessions belonging to a certain group.
        Parameters
        ---------
        group_name: str or list of str
            Allowed group names are defined in shared.Group
        Adrian 2020-03-23
        """
        if type(self) in [str, list]:
            group_name = self  # argument self was not given

        if group_name == '?':
            print('Available groups:', shared.Group.fetch('group'))
            return

        if type(group_name) == str:
            group_name = [group_name]  # convert to list for compatibility with lists

        keys = list()
        for name in group_name:
            if name not in shared.Group.fetch('group'):
                raise Exception('The group name "{}" is not an entry of shared.Group.'.format(name))

            keys.extend((Session() & (shared.GroupAssignment() & {'group': name})).fetch(dj.key))
        return keys

    def get_key(self, counter):
        """Return key that corresponds to counter of sessions """
        return (self & {'counter': counter}).fetch1('KEY')