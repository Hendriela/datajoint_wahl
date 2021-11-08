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


@schema
class MappingInfo(dj.Computed):
    definition = """ # Automatically computed parameters from the sesssion file
    -> MappingSession
    ---
    pre_stim                        : float     # pre stimulus interval in seconds
    stim_duration                   : float     # stimulus duration in seconds
    post_stim                       : float     # post stimulus interval in seconds
    repeats                         : int       # number of times the stimulus is delivered
    stim_timestamps                 : longblob  # stimulation timestamps in seconds
    sampling_rate                   : int       # Vi sampling rate
    stop_time                       : float     # total duration of the session, in seconds
    session                         : int       # counter for repeated sessions with the same body part
    """


class ProcessedMappingFile(dj.Computed):
    definition = """ # Computes and saves aligned,smoothed, pixelwise DF/F as a tiff file.
    -> MappingInfo
    ---
    n_frames_pre                    : int        # number of pre stim (baseline) frames
    n_frames_stim                   : int        # number of frames while the stimulus is active
    n_frames_post                   : int        # number of frames post stimulation
    n_frames_trial                  : int        # total number of frames in a trial
    -> mpanze_widefield.SmoothingKernel
    imaging_fps                     : float      # detected frames per second from sync path
    frame_timestamps                : longblob   # (repeats x n_frames_trial) matrix of frame timestamps in seconds
    frames_indices                  : longblob   # (repeats x n_frames_trial) matrix of frame indices in tiff file
    filename_processed              : varchar(512)  # name of the pre-processed file, relative to the session folder
    """