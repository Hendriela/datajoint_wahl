"""Schema for experimental information"""

import datajoint as dj
from . import common_mice, shared
import login
import os

schema = dj.schema('exp', locals(), create_tables=True)


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
        ['Wheel_v1', 'Basic wheel, no additional whisker stimulation or tones'],
        ['Wheel_3mod_v1', 'Wheel with whisker stimulation (galvo with tape), LED and sound'],
        ['Wheel_3mod_v2',
         'Wheel with 3 modalities for stimulation: Whisker, LED and Sound. The whisker stimulator is carbon stick with T shape and can be moved to positions between -5 and 5.'],
        ['Head-fixed', 'Mouse is headfixed under the microscope'],
        ['Ephys_wheel', 'Mouse on wheel, while recording e-phys signals at the same time with Intan box.'],
        ['Deluxe_wheel_v1', 'Wheel with soft cover, with whisker pole, sound and LED.'],
        ['Deluxe_wheel_v2', 'Wheel with soft cover, with whisker pole and sound. Light stimulus by illumination LED.'],
        ['Deluxe_wheel_ephys',
         'Wheel with soft cover, with whisker pole and sound. Light stimulus by illumination LED. Simultaneous ephys recording.'],
        ['Deluxe_wheel_widefield',
         'Wheel with soft cover, with whisker pole, sound and LED. Setup in widefield microscope'],
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
        ['None', 0, 'No task, mouse is freely running on the wheel.'],
        ['Manual_stim', 0,
         'Mouse is running on wheel, 3 modality stimulation triggered independently by clicking buttons.'],
        ['Pairing_v1', 0, 'Independent stimulation with 3 modalities, blocks with pauses.'],
        ['Pairing_v1', 1, 'Sound preceeding the whisker touch, blocks with pauses.'],
        ['Pairing_v1', 2, 'Sound that preceeds whisker touch sometimes omitted, blocks with pauses.'],
        ['Positions_and_events_v1', 0, 'Automatic protocoll with different positions and sensory events'],
        ['5Pos_dark', 0,
         'Automatic protocol with 5 positions (0, -2, ..., -8) in the darkness all the time (five_pos_stim_15min_v01.txt)'],
        ['5Pos_light', 0,
         'Automatic protocol with 5 positions (0, -2, ..., -8) with some trials with blue light on (five_pos_stim_wLight_15min_v02.txt)'],
        ['5Pos_control', 0, 'Same as 5Pos_light, but the whiskers are shortened, no contact to pole possible.'],
        ['5Pos_other', 0, 'Some variation of the task, manual check necessary what was going on (see exp.Session)'],

    ]


@schema
class Experimenter(dj.Lookup):
    definition = """ # Details about experimenter
    experimenter    : varchar(128)      # Short name of experimenter
    ---
    full_name       : varchar(256)      # Full name
    email           : varchar(256)      # Email address
    update_contact  : varchar(256)      # Contact information for automatic updates
    """

    contents = [
        ['Adrian', 'Adrian Hoffmann', 'hoffmann@hifo.uzh.ch', 'None'],
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