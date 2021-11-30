""" Schema for processing video files using deeplabcut"""

import datajoint as dj
from . import common_exp as exp #, img

# from .utils.common import log    # standardized logging
# from .utils import analysis

import login
import os
import numpy as np

schema = dj.schema('common_dlc', locals(), create_tables=True)


# =============================================================================
# VIDEO
# =============================================================================

@schema
class CameraPosition(dj.Lookup):
    definition = """ # Lookup table for camera positions
    camera_position : varchar(128)
    ---
    details : varchar(1048)
    """
    contents = [
        ['Closeup_Face', '12mm objective with closeup of the face'],
        ['Far_Whole_Mouse', '50mm objective from far with the whole mouse in frame'],
        ['Face_and_Body', 'Face in focus with part of the body slightly defocussed, pupil also visible'],
    ]


@schema
class Video(dj.Manual):
    definition = """ # Info about the acquired video
    -> exp.Session
    camera_nr = 0 : int          # Additional key for multiple cameras at once
    ---
    -> CameraPosition
    frame_rate : int             # Selected frame rate of the recording
    """

    # def get_video_object(self, with_time=True):
    #     """ Returns an object of utils.video.BodyVideo to access frames from videos
    #     To get a specific frame, use video_object.get_frame(frame_nr)
    #     Adrian 2020-12-14 """
    #
    #     video_paths = (RawVideoFile & self).get_paths()
    #
    #     # optional include video time
    #     if with_time:
    #         video_time = (VideoTime & self).fetch1('video_time')
    #     else:
    #         video_time = None
    #
    #     # create the BodyVideo object to hide multiple part files from user
    #     from .utils import video
    #     video_object = video.BodyVideo(video_paths=video_paths, video_time=video_time)
    #
    #     return video_object

@schema
class RawVideoFile(dj.Manual):
    definition = """ # File names of raw video files
    -> Video
    part = 0   : int           # Counter of video parts, base 0
    ---
    file_name  : varchar(256)  # Name of the video file
    """

    def get_path(self):
        """Return the absolute path of the file on the current system"""

        if len(self) != 1:
            raise Exception('Only videos with a single part can use the get_path(), ' +\
                            'not with {} parts. Use get_paths instead.'.format(len(self)) )

        base_directory = login.get_neurophys_directory()
        folder = ( exp.Session() & self).fetch1('id')
        file = self.fetch1('file_name')

        return os.path.join( base_directory, folder, file )

    def get_paths(self):
        """Returns list of paths from multiple video parts from the same recording """

        # sanity check that all video parts are from the same session
        if len( exp.Session() & self ) != 1:
            raise Warning('This call of RawVideoFiles.get_paths resulted in videos ' +\
                          'from not exactly one session, but {}'.format(len( exp.Session() & self )))

        path_list = list()
        for part_key in self.fetch('KEY', order_by='part'):
            path_list.append( (RawVideoFile & part_key).get_path() )

        return path_list

    # def get_raw_data(self):
    #     """Return the video data as numpy array (frames, height, width)
    #     Returns
    #     -------
    #     video_frames : 3d numpy array
    #         All frames in shape (frames, height, width)
    #     # TODO: select only a subset of frames
    #     # TODO: write better ways to access the video with new table
    #     Adrian 2019-08-19
    #     """
    #
    #     # load video
    #     import pims    # if this throws error: pip install pims
    #     path = self.get_path()
    #     video = pims.Video( path )
    #
    #     # prepare numpy array to hold the video data
    #     nr_frames = video.reader.get_length()
    #     example_frame = np.array(video.get_frame( 0 ))[:,:,0]   # only gray scale
    #     height, width = example_frame.shape
    #
    #     video_frames = np.zeros( (nr_frames, height, width), dtype=np.uint8 )
    #
    #     for i in range( nr_frames ):
    #         video_frames[i,:,:] = np.array( video.get_frame( i ))[:,:,0]   # only gray scale
    #
    #     return video_frames


@schema
class VideoInfo(dj.Computed):
    definition = """ # Additional automatically generated information about the video
    -> Video
    ---
    nr_frames      : int     # Number of frames in the video
    width          : int     # Width of frame in pixel
    height         : int     # Height of each frame in pixel
    # Add: dropped_frames : int     # Number of dropped frames
    """

    class ExampleFrame(dj.Part):
        definition = """ # Example frames of the video in external storage
        -> master
        ---
        example_frame : longblob    # (width, height) example frame
        """

    def make(self, key):
        """ Populate the VideoInfo and ExampleFrame tables
        Adrian 2019-08-19
        """
        # Todo: set up logging for us
        # log('Populating VideoInfo for key: {}'.format(key))

        video_obj = (RawVideoFile() & key).get_video_object()

        # get info about video
        nr_frames = video_obj.nr_frames
        example_frame = video_obj.get_frame(100)
        height, width = example_frame.shape

        entry_info = dict( **key,
                            nr_frames = nr_frames,
                            width = width,
                            height = height)

        entry_example = dict( **key,
                                example_frame = example_frame)

        # insert the entries in the table (master first, then part)
        self.insert1( entry_info )
        VideoInfo.ExampleFrame.insert1( entry_example )

        # log('Finished populating VideoInfo for key: {}'.format(key))

@schema
class LEDLocation(dj.Manual):
    definition = """ # Location of the synchronization LED in the video
    -> Video
    -----
    led_left  : int    # LED position from the left side of the frame in pixel (dim 2)
    led_top   : int    # LED position from the top of the frame in pixel (dim 1)
    led_width : int    # Width of LED in pixel
    """

    # to enter the location of the LED, use the GUI provided in:
    #                pipeline_populate_behvior.ipynb

@schema
class RawVideoTimeFile(dj.Manual):
    definition = """ # File name of the recorded exposure on signal from camera (only for recordings > 2020-10-20)
    -> Video
    -----
    file_name : varchar(128)        # File name, use function get_path for path on your machine
    """

    def get_path(self):
        """Return the absolute path of the file on the current system"""
        return os.path.join( (exp.Session & self).get_folder(), self.fetch1('file_name') )
    def get_raw_data(self):
        """ Return data of the file (1d array) """
        # Signal sampled at 1kHz with onset together with other behavioral variables
        path = self.get_path()
        return np.fromfile( path, dtype='>d')


@schema
class VideoTime(dj.Computed):
    definition = """ # Time of the video recording from blinking LED
    -> Video
    --------
    avg_frame_rate  : float         # Measured average frame rate
    video_time      : longblob      # 1d array, time of each frame in seconds of behavior PC
    """

    def make(self, key):
        """Automatically populate the VideoTime (old: from LED, new: from exposure times file)
        Adrian 2019-08-22
        """
        # log('Populating VideoTime for key: {}'.format(key))

        if len( LEDLocation & key ) > 0:
            type = "LED"
            # extract timing from blinking LED every 5 seconds
            video = (RawVideoFile() & key).get_raw_data()  # (frames, height, width)

            led_x = (LEDLocation & key).fetch1('led_top')  # x ... top to bottom
            led_y = (LEDLocation & key).fetch1('led_left') # y ... left to right
            d = int( (LEDLocation & key).fetch1('led_width') / 2 )  # half LED width

            roi_x = slice( led_x-d, led_x+d)
            roi_y = slice( led_y-d, led_y+d)

            nr_frames = video.shape[0]

            # extract intensity of LED over time
            intensity = np.zeros( nr_frames )
            for i in range( nr_frames ):
                intensity[i] = np.mean( video[i, roi_x, roi_y] )

        elif len( RawVideoTimeFile & key ) > 0:
            type = "FILE"
            nr_frames = (VideoInfo & key).fetch1('nr_frames')
            intensity = (RawVideoTimeFile & key).get_raw_data()
        else:
            raise Exception('Either a LED location or RawVideoTimeFile must be given.')

        # detect threshold crossing
        thres = (np.min(intensity) + np.max(intensity) ) / 2  # threshold value between min and max

        cross = (intensity[:-1] < thres) & (intensity[1:] > thres)  # True/False with threshold crossing
        cross_index = np.where( cross )[0]

        if type == 'LED':
            # calculate average frame rate (one blink every 5 seconds)
            avg_frame_rate = np.mean( np.diff(cross_index)) / 5

            # Calculate the time of the frames based on the detected blink times
            # The LED blinks every 5 seconds and the first blink occurs directly when
            # the recording of the wheel position is started on the behavior PC (time 0)
            nr_blinks = len( cross_index )
            blink_time = np.arange( nr_blinks ) * 5     # blinking times in seconds

            # linearly interpolate between known timepoints of the blinks
            # this function transforms frame indicies to seconds of the wheel/galvo recording
            import scipy
            frames_to_seconds = scipy.interpolate.interp1d(cross_index, blink_time, fill_value='extrapolate')

            video_time = frames_to_seconds( np.arange( nr_frames) )

        else:
            avg_frame_rate = np.mean( 1000 / np.diff(cross_index) )
            video_time = cross_index / 1000   # time in seconds

            # remove additional few frames at the end that were not saved anymore
            l = len(video_time)
            if l == nr_frames:
                pass  # everything fine
            elif ( l > nr_frames ) and ( l <= nr_frames+2 ):
                # accept 1 or 2 difference
                print("Warning: Removed 1 or 2 frames in the VideoTime. Nothing to worry about.")
                video_time = video_time[:nr_frames]
            elif ( l < nr_frames ) and ( l >= nr_frames-2 ):
                # accept 1 or 2 difference
                print("Warning: VideoTime too short by 1 or 2 frames. Nothing to worry about?")
            else:
                raise Exception('Mismatch between recorded frames ({}) and detected ({})'.format(nr_frames,l))

        # insert variables into this table
        new_entry = dict( **key,
                          avg_frame_rate = avg_frame_rate,
                          video_time = video_time)
        self.insert1( new_entry )

        # log('Inserted VideoTime entry for key: {}'.format(key))
