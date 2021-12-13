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
import matplotlib.pyplot as plt
import os
import yaml
import subprocess
import mpanze_scripts.utils as utils
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

    def get_first_image(self):
        f_vid = (RawVideoFile & self).get_path(check_existence=True)
        cap = cv2.VideoCapture(str(f_vid))
        ret, frame = cap.read()
        cap.release()
        return frame

    def plot_cropping(self, crop_param=0):
        import matplotlib
        matplotlib.use("Qt5Agg")
        for vid in self:
            crop = (FFMPEGParameter() & "crop_id=%i" % crop_param).fetch1()
            frame = (self & vid).get_first_image()
            w, h, x, y = crop["crop_w"], crop["crop_h"], crop["crop_x"], crop["crop_y"]
            scale_w, scale_h = crop["scale_w"], crop["scale_h"]
            frame_crop = frame[y:y+h, x:x+h]
            iw = int(w / scale_w)
            ih = int(h / scale_h)
            if iw % 2 == 1:
                iw = iw + 1
            if ih % 2 == 1:
                ih = ih + 1
            frame_rescale = cv2.resize(frame_crop, (iw, ih), cv2.INTER_CUBIC)
            p_vid = (RawVideoFile & vid).get_path()
            plt.figure("%i, %s, %s" % (crop["crop_id"], vid["camera_position"], p_vid))
            plt.subplot(122)
            plt.imshow(frame_rescale, "Greys_r")
            plt.subplot(121)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))
            plt.imshow(frame)
            plt.tight_layout()
            plt.show()


@schema
class RawVideoFile(dj.Manual):
    definition = """ # File names of raw video files
    -> Video
    part = 0   : int           # Counter of video parts, base 0
    ---
    filename_video  : varchar(256)  # Name of the video file, relative to session folder
    """

    def get_paths(self):
        return utils.get_paths(self, "filename_video")

    def get_path(self, check_existence=False):
       return utils.get_path(self, "filename_video", check_existence=check_existence)


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
        return utils.get_paths(self, "filename_binary")

    def get_path(self, check_existence=False):
        return utils.get_path(self, "filename_binary", check_existence=check_existence)

    def get_raw_data(self):
        """ Read data from file, return two arrays: one of timestamps and one of frame counts """
        path_file = self.get_path()
        data = np.fromfile(path_file)
        return data[::2], data[1::2]


@schema
class VideoTime(dj.Computed):
    definition = """ # Compute frame timestamps from synchronisation file
    -> Video
    --------
    nr_frames_counted : int         # number of frames that could effectively be counted from sync file
    video_duration    : float       # measured video duration in seconds
    estim_frame_rate  : float       # Average frame rate estimated from video time file
    frame_offset      : int         # frame offset between raw video and detected frames in sync file
    video_time        : longblob    # 1d array, time of each frame in seconds of behavior PC
    """

    def make(self, key):
        # load data from counts file
        t, counts = (FrameCountVideoTimeFile & key).get_raw_data()

        nr_frames_counted = int(counts[-1])
        nr_frames_video = (VideoInfo & key).fetch1()["nr_frames"]
        delta_frames = nr_frames_video - nr_frames_counted
        try:
            t0 = t[counts>0][0]
            tf = t[-1]
            video_duration = tf-t0
            video_time = np.linspace(t0, tf, nr_frames_counted)
            estim_frame_rate = nr_frames_counted/video_duration
            frame_offset = delta_frames - 3
        except IndexError:
            # assume start and end points coincide with acquisition time
            t0 = t[0]
            tf = t[-1]
            video_duration = tf - t0
            video_time = np.linspace(t0, tf, nr_frames_video-3)
            estim_frame_rate = (nr_frames_video-3)/video_duration
            frame_offset = 0

        # populate new entry
        new_key = {**key, "nr_frames_counted": nr_frames_counted, "video_duration": video_duration,
                   "estim_frame_rate": estim_frame_rate, "frame_offset": frame_offset, "video_time": video_time}
        self.insert1(new_key)


@schema
class FFMPEGParameter(dj.Lookup):
    definition = """ # Parameters for cropping, rescaling and recompressing videos using ffmpeg and h264 codec
    crop_id            : int           # id of this parameter set
    ---
    preset              : enum('ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow')    # ratio of encoding speed to compression ratio
    crf                 : tinyint       # compression quality from 0 (no compression) to 31. typical values around 15-17
    scale_w             : float         # scale factor for video width
    scale_h             : float         # scale factor for video height
    crop_w              : int           # width of cropped region, in pixels
    crop_h              : int           # height of cropped region, in pixels
    crop_x              : int           # x offset of crop region, in pixels
    crop_y              : int           # y offset of crop region, in pixels 
    """
    contents = [{"crop_id": 0, "preset": "faster", "crf": 15, "scale_w": 1.5, "scale_h": 1.5,
                 "crop_w": 840, "crop_h": 524, "crop_x": 440, "crop_y": 500},
                {"crop_id": 1, "preset": "faster", "crf": 17, "scale_w": 1, "scale_h": 1,
                 "crop_w": 1280, "crop_h": 1024, "crop_x": 0, "crop_y": 0}
                ]


@schema
class CroppedVideo(dj.Computed):
    definition = """ # Generates cropped video file using a given set of parameters. requires ffmpeg and h264
    -> Video
    -> FFMPEGParameter
    ---
    pixel_w             : int   # actual width of the video in pixels. should always be even due to h264 requirements
    pixel_h             : int   # actual height of the video in pixels. should always be even due to h264 requirements
    filename_cropped    : varchar(256)  # Name of the cropped video file, relative to session folder
    """

    def make(self, key):
        # set path to ffmpeg executable
        ffmpeg_dir = "C:/Users/mpanze/Documents/ffmpeg/bin/"
        # get new path
        p_video = (RawVideoFile & key).get_path(check_existence=True)
        p_cropped = pathlib.Path(p_video.parent, p_video.stem + "_cropped_%i.avi" % (key["crop_id"]))
        # read video info
        vid_info = (VideoInfo & key).fetch1()
        w_0, h_0 = vid_info["width"], vid_info["height"]
        # get crop parameters
        crop = (FFMPEGParameter & key).fetch1()
        w, h, x, y = crop["crop_w"], crop["crop_h"], crop["crop_x"], crop["crop_y"]
        scale_w, scale_h = crop["scale_w"], crop["scale_h"]
        # actual frame dimensions must be divisible by 2 to be used with h264 codec
        iw = int(w / scale_w)
        ih = int(h / scale_h)
        if iw % 2 == 1:
            iw = iw + 1
        if ih % 2 == 1:
            ih = ih + 1

        # cropped dimensions same as vid dimensions, keep video as is
        if (w_0 == w) and (h_0 == h) and (x == 0) and (y == 0) and (scale_h == 1) and (scale_w == 1):
            p_video_rel = (RawVideoFile & key).fetch1["filename_video"]
            new_entry = {**key, "pixel_w": w_0, "pixel_h": h_0, "filename_cropped": p_video_rel}
        else:
            # make ffmpeg command
            ffmpeg_command = [
                'ffmpeg',
                '-i', str(p_video),
                '-c:v', 'libx264',
                '-preset', crop["preset"],
                '-crf', "%i" % (crop["crf"]),
                '-vf', 'crop=w=%i:h=%i:x=%i:y=%i, scale=%i:%i' % (w, h, x, y, iw, ih),
                '-n', '-c:a', 'copy', str(p_cropped)
            ]
            # execute command
            os.chdir(ffmpeg_dir)
            print(ffmpeg_command)
            subprocess.run(ffmpeg_command)
            # create entry
            p_cropped_rel = p_cropped.relative_to(p_video.parent)
            new_entry = {**key, "pixel_w": iw, "pixel_h": ih, "filename_cropped": str(p_cropped_rel)}
        # insert data
        self.insert1(new_entry)

    def get_paths(self):
        return utils.get_paths(self, "filename_cropped")

    def get_path(self, check_existence=False):
        return utils.get_path(self, "filename_cropped", check_existence=check_existence)


@schema
class DLCModel(dj.Lookup):
    definition = """ # lookup table pointing to Deeplabcut projects
    model_name          : varchar(128)      # model name == project folder name for clarity
    iteration           : smallint          # current iteration of the model
    ---
    config_path         : varchar(512)      # absolute path to the config.yaml file for the project
    """
    contents = [{"model_name": "DigitsLateralView-mpanze-2021-10-27", "iteration": 2,
                 "config_path": "W:\\Neurophysiology-Storage1\\Wahl\\Matteo\\deeplabcut\\DigitsLateralView-mpanze-2021-10-27\\config.yaml"}]

    def get_path(self):
        if len(self) != 1:
            raise Exception("Please select a single DLCModel")
        return self.fetch1()["config_path"]


@schema
class VideoPredictions(dj.Computed):
    definition = """ # runs video through a deeplabcut model and generates a .h5 with predictions
    -> CroppedVideo
    -> DLCModel
    ---
    filename_pred       : varchar(512)      # location of .h5 with predicted labels, relative to session folder
    """

    def make(self, key):
        # check that model iteration matches selected iteration
        p_config = (DLCModel() & key).get_path()
        with open(p_config) as file:
            cfg_dict = yaml.load(file, Loader=yaml.FullLoader)
        if cfg_dict["iteration"] != key["iteration"]:
            raise Exception("DLC model iterations do not match!")

        # get path to video
        p_video = (CroppedVideo() & key).get_path(check_existence=True)

        # generate subfolder for storing results. subfolders are separated by iteration
        iteration_folder = pathlib.Path(p_video.parent, "iteration_%i" % key["iteration"])
        if not iteration_folder.exists():
            os.mkdir(str(iteration_folder))

        # run analysis on video
        deeplabcut.analyze_videos(str(p_config), [str(p_video)], destfolder=str(iteration_folder))

        # find file in subfolder
        p_h5 = list(iteration_folder.glob(str(p_video.stem) + "*.h5"))
        if len(p_h5) != 1:
            raise Exception("Failed to find .h5 file")

        # generate attributes and add to table
        filename_pred = p_h5[0].relative_to(p_video.parent)
        new_entry = {**key, "filename_pred": str(filename_pred)}
        self.insert1(new_entry)
