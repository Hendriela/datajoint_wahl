"""Schema for processing sensory mapping data"""
import json

import datajoint as dj
import login
login.connect()
from schema import common_exp
import pathlib
import json
from mpanze_scripts.widefield import utils

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

    def get_paths(self):
        """Construct full paths to raw synchronisation file"""
        path_neurophys = login.get_working_directory()  # get data directory path on local machine
        # find sessions corresponding to current files
        sessions = (self * common_exp.Session())

        # iterate over sessions
        paths = []
        for session in sessions:
            # obtain full path
            path_session = session["session_path"]
            path_file = session["filename_sync"]
            paths.append(pathlib.Path(path_neurophys, path_session, path_file))
        return paths

    def get_path(self, check_existence=False):
        """
        Construct full path to a raw synchronisation file.
        Method only works for single-element query.
        Args:
            check_existence: is True, method throws an exception if the file does not exist
        """
        if len(self) == 1:
            p = self.get_paths()[0]
            if check_existence:
                if not p.exists():
                    raise Exception("The file was not found at %s" % str(p))
            return p
        else:
            raise Exception("This method only works for a single entry! For multiple entries use get_paths")


@schema
class RawParameterFile(dj.Manual):
    definition = """ # Files containing session parameters
    -> MappingSession
    ---
    filename_params              : varchar(512)  # name of the sync file, relative to the session folder
    """

    def get_paths(self):
        """Construct full paths to raw parameter files"""
        path_neurophys = login.get_working_directory()  # get data directory path on local machine
        # find sessions corresponding to current files
        sessions = (self * common_exp.Session())

        # iterate over sessions
        paths = []
        for session in sessions:
            # obtain full path
            path_session = session["session_path"]
            path_file = session["filename_params"]
            paths.append(pathlib.Path(path_neurophys, path_session, path_file))
        return paths

    def get_path(self, check_existence=False):
        """
        Construct full path to a raw parameter file.
        Method only works for single-element query.
        Args:
            check_existence: is True, method throws an exception if the file does not exist
        """
        if len(self) == 1:
            p = self.get_paths()[0]
            if check_existence:
                if not p.exists():
                    raise Exception("The file was not found at %s" % str(p))
            return p
        else:
            raise Exception("This method only works for a single entry! For multiple entries use get_paths")


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
    stim_session_count              : int       # counter for repeated sessions with the same body part
    file_timestamp                  : datetime  # timestamp generated while saving the file
    """
    def make(self, key):
        """
        Loads session parameters from the mapping .json file. Raises an exception if the file is not found
        """
        # find mapping file. get_path already catches most likely exceptions
        p_json = (RawParameterFile & key).get_path(check_existence=True)
        with open(p_json) as f:
            data_json = json.load(f)
        key["pre_stim"] = data_json["Pre_stim (s)"]
        key["stim_duration"] = data_json["Stimulus Duration (s)"]
        key["post_stim"] = data_json["Post_stim (s)"]
        key["repeats"] = data_json["Repeats"]
        if data_json["Stim Type"] == "Sensory":
            key["stim_timestamps"] = data_json["Sensory Timestamps"]
        else:
            raise Exception("Invalid or unimplemented stiumuls type: %s" % data_json["Stim Type"])
        key["sampling_rate"] = data_json["Sampling Rate"]
        key["stop_time"] = data_json["Stop Time"]
        key["stim_session_count"] = data_json["Session"]
        key["file_timestamp"] = utils.mapping_datetime_from_dict(data_json).strftime("%Y-%m-%d %H:%M:%S")
        self.insert1(key)



# class ProcessedMappingFile(dj.Computed):
#     definition = """ # Computes and saves aligned,smoothed, pixelwise DF/F as a tiff file.
#     -> MappingInfo
#     ---
#     n_frames_pre                    : int        # number of pre stim (baseline) frames
#     n_frames_stim                   : int        # number of frames while the stimulus is active
#     n_frames_post                   : int        # number of frames post stimulation
#     n_frames_trial                  : int        # total number of frames in a trial
#     -> mpanze_widefield.SmoothingKernel
#     imaging_fps                     : float      # detected frames per second from sync path
#     frame_timestamps                : longblob   # (repeats x n_frames_trial) matrix of frame timestamps in seconds
#     frames_indices                  : longblob   # (repeats x n_frames_trial) matrix of frame indices in tiff file
#     filename_processed              : varchar(512)  # name of the pre-processed file, relative to the session folder
#     """
