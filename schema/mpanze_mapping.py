"""Schema for processing sensory mapping data"""

import datajoint as dj
import login
login.connect()
from schema import common_exp

schema = dj.schema('mpanze_mapping', locals(), create_tables=True)

@schema
class Stimulation(dj.Lookup):
    definition = """ # Table storing various stimulus configurations
    stim_bodypart               : varchar(128)      # stimulated body part
    stim_config                 : int               # id to keep track of different stim parameters
    ---
    vibration_amplitude = NULL  : float             # vibration amplitude (volts)
    vibration_frequency = NULL  : int               # vibration frequency (hertz)
    tone_frequency = NULL       : int               # auditory tone frequency (hertz)
    stim_type                   : enum("Visual", "Somatosensory", "Auditory") # stimulus type
    stim_description            : varchar(1024)         # additional info about the stimulus
    """
    contents = [
        {"stim_bodypart": "forelimb_right", "stim_config": 0, "vibration_amplitude": 0.2, "vibration_frequency": 100,
         "stim_type": "Somatosensory", "stim_description": "vibrating bar placed under right forelimb"},
        {"stim_bodypart": "forelimb_left", "stim_config": 0, "vibration_amplitude": 0.2, "vibration_frequency": 100,
         "stim_type": "Somatosensory", "stim_description": "vibrating bar placed under left forelimb"},
        {"stim_bodypart": "hindlimb_right", "stim_config": 0, "vibration_amplitude": 0.2, "vibration_frequency": 100,
         "stim_type": "Somatosensory", "stim_description": "vibrating bar placed under right hindlimb"},
        {"stim_bodypart": "hindlimb_left", "stim_config": 0, "vibration_amplitude": 0.2, "vibration_frequency": 100,
         "stim_type": "Somatosensory", "stim_description": "vibrating bar placed under left hindlimb"}
    ]


@schema
class MappingSession(dj.Manual):
    definition = """ # Reference Table for mapping sessions
    -> common_exp.Session
    ---
    -> Stimulation
    """


@schema
class RawSynchronisationFile(dj.Manual):
    definition = """ # Synchronisation files for aligning imaging data to behaviour
    -> MappingSession
    ---
    filename_sync                : varchar(512)  # name of the sync file, relative to the session folder
    """


@schema
class RawParameterFile(dj.Manual):
    definition = """ # Files containing session parameters
    -> MappingSession
    ---
    filename_params              : varchar(512)  # name of the sync file, relative to the session folder
    """