"""
Schema for processing video files using deeplabcut
Adapted from Adrain Hoffmann's pipeline
"""

import datajoint as dj
import deeplabcut
import os
import numpy as np
import login
import pathlib
from schema import common_exp
import cv2
login.connect()


schema = dj.schema('common_dlc', locals(), create_tables=True)


# =============================================================================
# VIDEO
# =============================================================================

@schema
class CameraPosition(dj.Lookup):
    definition = """ # Lookup table for camera positions
    camera_position : varchar(128)      # attribute for distinguishing between separate camera angles
    ---
    position_details : varchar(1048)    # longer description of camera position and properties
    """
    contents = [
        ['Right_Forelimb_Side_View', 'camera placed orthogonally, with view of right forelimb from hand to shoulder'],
        ['Right_Forelimb_Front_View',
         'camera recording the mouse front-on, with focus between the resting position and the joystick']
    ]


@schema
class Video(dj.Manual):
    definition = """ # Info about the acquired video
    -> common_exp.Session
    -> CameraPosition
    ---
    """


@schema
class RawVideoFile(dj.Manual):
    definition = """ # File names of raw video files
    -> Video
    part = 0   : int           # Counter of video parts, base 0
    ---
    filename_video  : varchar(256)  # Name of the video file, relative to session folder
    """

    def get_paths(self):
        """Construct full paths to raw video files"""
        path_neurophys = login.get_working_directory()  # get data directory path on local machine
        # find sessions corresponding to current files
        sessions = (self * common_exp.Session())

        # iterate over sessions
        paths = []
        for session in sessions:
            # obtain full path
            path_session = session["session_path"]
            path_file = session["filename_video"]
            paths.append(pathlib.Path(path_neurophys, path_session, path_file))
        return paths

    def get_path(self, check_existence=False):
        """
        Construct full path to a raw video file.
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
class VideoInfo(dj.Computed):
    definition = """ # Info about video from file metadata
    -> Video
    ---
    nr_frames      : int        # Number of frames in the video
    width          : int        # Width of frame in pixel
    height         : int        # Height of each frame in pixel
    fps            : float      # Recording's frames per second
    bitrate        : float      # Video bitrate in kb/s
    first_frame    : longblob   # first frame of the video
    """

    def make(self, key):
        """ Populate the VideoInfo and ExampleFrame tables
        Adrian 2019-08-19
        """
        # open file
        video_path = (RawVideoFile & key).get_path(check_existence=True)
        cap = cv2.VideoCapture(str(video_path))

        # get properties from cv2 video object
        nr_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        bitrate = cap.get(cv2.CAP_PROP_BITRATE)
        ret, first_frame = cap.read()
        cap.release()

        # insert new entries into dict
        entry_info = {**key, "nr_frames": nr_frames, "width": width, "height": height, "fps": fps, "bitrate": bitrate,
                      "first_frame": first_frame}

        # insert the entries in the table (master first, then part)
        self.insert1(entry_info)


@schema
class FrameCountVideoTimeFile(dj.Manual):
    definition = """ # Binary file (.bin) containing time-stamped frame counts for synchronisation
    -> Video
    -----
    filename_binary : varchar(128)        # filename of synchronisation file relative to session folder
    """

    def get_paths(self):
        """Construct full paths to raw synchronisation files"""
        path_neurophys = login.get_working_directory()  # get data directory path on local machine
        # find sessions corresponding to current files
        sessions = (self * common_exp.Session())

        # iterate over sessions
        paths = []
        for session in sessions:
            # obtain full path
            path_session = session["session_path"]
            path_file = session["filename_binary"]
            paths.append(pathlib.Path(path_neurophys, path_session, path_file))
        return paths

    def get_path(self, check_existence=False):
        """
        Construct full path to a raw video file.
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

    def get_raw_data(self):
        """ Read data from file, return two arrays: one of timestamps and one of frame counts """
        path_file = self.get_path()
        data = np.fromfile(path_file)
        return data[::2], data[1::2]


# @schema
# class VideoTime(dj.Computed):
#     definition = """ # Compute frame timestamps from synchronisation file
#     -> Video
#     --------
#     avg_frame_rate  : float         # Measured average frame rate
#     video_time      : longblob      # 1d array, time of each frame in seconds of behavior PC
#     """
