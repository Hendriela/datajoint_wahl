""" Schema to store behavioral data """

import datajoint as dj
from . import common_exp #, img

# from .utils.common import log    # standardized logging
# from .utils import analysis

import login
import os
import numpy as np

schema = dj.schema('behav', locals(), create_tables = True)


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

    def get_video_object(self, with_time=True):
        """ Returns an object of utils.video.BodyVideo to access frames from videos
        To get a specific frame, use video_object.get_frame(frame_nr)
        Adrian 2020-12-14 """

        video_paths = (RawVideoFile & self).get_paths()

        # optional include video time
        if with_time:
            video_time = (VideoTime & self).fetch1('video_time')
        else:
            video_time = None

        # create the BodyVideo object to hide multiple part files from user
        from .utils import video
        video_object = video.BodyVideo(video_paths=video_paths, video_time=video_time)

        return video_object

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
        folder = (exp.Session() & self).fetch1('id')
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

    def get_raw_data(self):
        """Return the video data as numpy array (frames, height, width)
        Returns
        -------
        video_frames : 3d numpy array
            All frames in shape (frames, height, width)
        # TODO: select only a subset of frames
        # TODO: write better ways to access the video with new table
        Adrian 2019-08-19
        """

        # load video
        import pims    # if this throws error: pip install pims
        path = self.get_path()
        video = pims.Video( path )

        # prepare numpy array to hold the video data
        nr_frames = video.reader.get_length()
        example_frame = np.array(video.get_frame( 0 ))[:,:,0]   # only gray scale
        height, width = example_frame.shape

        video_frames = np.zeros( (nr_frames, height, width), dtype=np.uint8 )

        for i in range( nr_frames ):
            video_frames[i,:,:] = np.array( video.get_frame( i ))[:,:,0]   # only gray scale

        return video_frames


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
        log('Populating VideoInfo for key: {}'.format(key))

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

        log('Finished populating VideoInfo for key: {}'.format(key))

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
        log('Populating VideoTime for key: {}'.format(key))

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

        log('Inserted VideoTime entry for key: {}'.format(key))



# =============================================================================
# WHEEL
# =============================================================================


@schema
class WheelType(dj.Lookup):
    definition = """ # Lookup table for wheel types
    wheel_type   : varchar(128)      # Short name for wheel
    ---
    details      : varchar(1024)     # Description of the used wheel
    """
    contents = [
        ['Wheel_Asli', "Asli's wheel with rugs at equal spacing, no additional textures."],
        ['Lasercut_v1', 'White round wheel with laser cut rugs. Phidgets encoder attached, 4 ticks per degree.'],
        ['Lasercut_smooth', 'White round wheel with laser cut rugs, covered with soft mat. Phidgets encoder attached, 4 ticks per degree.'],
    ]


@schema
class Wheel(dj.Manual):
    definition = """ # Basic information about the wheel recording
    -> exp.Session
    ---
    -> WheelType
    """

    def get(self):
        """ Wrapper function: Returns two 1d arrays (time and filtered, downsampled wheel position) """
        return (FilteredWheel & self).get_filtered_position()

    def get_pos(self):
        """ Wrapper function: Returns two 1d arrays (time and filtered, downsampled wheel position) """
        return (FilteredWheel & self).get_filtered_position()

    def get_vel(self):
        """ Wrapper function: Returns two 1d arrays (time and filtered, downsampled wheel velocity) """
        return (FilteredWheel & self).get_filtered_velocity()


@schema
class RawWheelFile(dj.Manual):
    definition = """ # Link to data file in which the wheel recording is saved
    -> Wheel
    ---
    file_name   : varchar(256)     # File name of the wheel recording
    """

    def get_path(self):
        """Return the absolute path of the file on the current system"""

        # Quick check if only one file is selected
        if len(self) != 1: raise Exception('Only length one allowed (not {})'.format(len(self)))

        base_directory = login.get_neurophys_directory()
        folder = (exp.Session() & self).fetch1('id')
        file = self.fetch1('file_name')

        return os.path.join( base_directory, folder, file )


    def get_raw_data(self):
        """Return the raw time and wheel position for one given recording
        Returns
        ---------
        time : 1d array
            Time in seconds recorded by the LabView program of behavior PC
        wheel_pos : 1d array
            Position of the wheel at the timepoints of 'time'
        Adrian 2019-08-19
        """

        # Data in the RawWheelFile is saved as doubles alternating between time
        # and the wheel position, sampling based on events every ~8ms
        path = self.get_path()
        data = np.fromfile( path, dtype='>d')
        time = data[::2]         # select every second element, starting at the first
        wheel_pos = data[1::2]   #                          ... starting at the second

        return time, wheel_pos

@schema
class FilteredWheel(dj.Computed):
    definition = """ # Table to store filtered and equidistant sampled wheel position
    -> RawWheelFile
    ---
    filtered_pos   : longblob     # 1d array filtered and sampled every 10 ms from raw file
    filtered_vel   : longblob     # 1d array of velocity filtered and sampled every 10 ms from raw file
    """

    def make(self, key):
        """Automatically populate this table with data from RawWheelFile
        Adrian 2020-03-19
        """
        log('Populating FilteredWheel for key: {}'.format(key))
        raw_t, raw_pos = (RawWheelFile & key).get_raw_data()

        # First, sample wheel data in regular intervals (in raw data there are some jumps in t)
        import scipy.interpolate as interpolate
        f = interpolate.interp1d(raw_t, raw_pos, kind='linear',
                                 bounds_error=False, fill_value="extrapolate")
        new_t = np.arange( 0, raw_t[-1], step=0.01)  # sample wheel recording at 100 Hz
        new_pos = f(new_t)  # mapping function f for linear interpolation

        # Then filter this signal with Savitzky-Golay filter (polynom 3rd order)
        import scipy.signal as sig

        window_size = 21
        polynom_order = 3
        filt_pos = sig.savgol_filter(new_pos, window_size, polynom_order)
        filt_vel = sig.savgol_filter(new_pos, window_size, polynom_order, deriv=1) * 100

        key['filtered_pos'] = filt_pos
        key['filtered_vel'] = filt_vel

        self.insert1( key )

    def get_filtered_position(self):
        """ Returns two 1d arrays: time and filtered wheel position  """
        filtered_pos = self.fetch1('filtered_pos')      # signal sampled every 10ms
        time = np.arange( len(filtered_pos) ) / 100     # in seconds
        return time, filtered_pos

    def get_filtered_velocity(self):
        """ Returns two 1d arrays: time and filtered wheel velocity  """
        filtered_vel = self.fetch1('filtered_vel')      # signal sampled every 10ms
        time = np.arange( len(filtered_vel) ) / 100     # in seconds
        return time, filtered_vel


    def plot_quality_control(self):
        """ Plots the filtered and raw data above each other to double check filtering parameters
        Adrian 2020-03-19
        """
        import matplotlib.pyplot as plt
        t_f, pos_f = self.get_filtered_position()
        _, vel_f = self.get_filtered_velocity()
        t_r, pos_r = (RawWheelFile & self).get_raw_data()

        plt.figure()
        ax1 = plt.subplot(2,1,1)
        plt.plot(t_r, pos_r, label='Raw position')
        plt.plot(t_f, pos_f, label='Filtered, downsampled')
        plt.legend()

        plt.subplot(2,1,2, sharex=ax1)
        plt.plot(t_r[1:], np.diff(pos_r)*100, label='Raw velocity')
        plt.plot(t_f, vel_f, label='Filtered, downsampled')
        plt.xlabel('Time [s]')
        plt.legend()



# =============================================================================
# SYNCHRONISATION WITH IMAGING
# =============================================================================

@schema
class SynchronizationType(dj.Lookup):
    definition = """ # Lookup table for types of synchronization
    sync_type    : varchar(128)      # Short name for the synchronization signal
    ---
    details      : varchar(1024)     # Description of the used wheel
    """

    contents = [
        ['Galvo_Y', 'The analog voltage which is also sent to the galvo y scanner.'],
        ['Galvo_Y_Clipped', 'The analog voltage which is also sent to the galvo y scanner, passed through Hans-Jörg element and clipped due to Intan amplifier.'],
        ['Frame_TTL', 'TTL signal for the pockels cell which blanks only during galvo y flyback time (e.g. in H45)'],
        ['Orca_Frame_Onset', 'Signal from Orca camera to mark the onset of each imaging frame']
    ]


@schema
class Synchronization(dj.Manual):
    definition = """ # Table for basic information about the synchronization
    -> exp.Session
    ---
    -> SynchronizationType
    """

@schema
class RawSynchronizationFile(dj.Manual):
    definition = """ # Link to synchronization files
    -> Synchronization
    ---
    file_name     : varchar(256)    # File name of the synchronization file
    """

    def get_path(self):
        """Return the absolute path of the file on the current system"""

        # Quick check if only one file is selected
        if len(self) != 1: raise Exception('Only length one allowed (not {})'.format(len(self)))

        base_directory = login.get_neurophys_directory()
        folder = (exp.Session() & self).fetch1('id')
        file = self.fetch1('file_name')

        return os.path.join( base_directory, folder, file )


    def get_raw_data(self, include_time=False):
        """Return the time and synchronization signal (e.g. galvo_y)
        Returns
        ---------
        time : 1d array (only if include_time==True)
            Time in seconds recorded by the LabView program of behavior PC
        sync_signal : 1d array
            Recorded voltage (e.g. galvo_y output signal)
        Adrian 2019-08-19
        """

        # The signal is saved at 1kHz and all times in behav are relative to the
        # start of this recording
        path = self.get_path()
        sync_signal = np.fromfile( path, dtype='>d')

        if include_time == False:
            return sync_signal
        else:
            time = np.arange( len(sync_signal) ) / 1000     # in seconds
            return time, sync_signal


@schema
class ScanTime(dj.Computed):
    definition = """ # Time of each 2p imaging frame in behavior time
    -> Synchronization
    ------
    scan_time          : longblob    # 1d array with (behavior) time of 2p imaging frames in seconds
    nr_detected_peaks  : int         # Number of detected peaks in the synchronization signal (e.g. galvo)
    avg_frame_rate     : float       # Measured average frame rate of the scanning
    """

    def make(self, key):
        """Automatically populate the ScanTimes from recorded synchronization signal
        # TODO: use only second or third peak depending on the plane hopping?
        Adrian 2019-08-22
        """
        log('Populating ScanTime for key: {}'.format(key))
        plotting = False

        time, signal = ( RawSynchronizationFile() & key).get_raw_data(include_time=True)

        sync_type = (Synchronization & key).fetch1('sync_type')

        if sync_type == 'Frame_TTL':
            min_, max_ = np.percentile( signal, [1,99])
            thres = (min_ + max_) / 2  # threshold between min and max
            # Find index of threshold crossing during upwards movement
            cross = (signal[1:] > thres) & (signal[:-1] <= thres)
            peak_index = np.where( cross )[0] + 1  # go to index above threshold

        elif (sync_type == 'Galvo_Y') or (sync_type == 'Galvo_Y_Clipped'):
            thres = np.percentile( signal, 90)  # for plotting later
            # Find index of the peak of galvo signal
            from .utils.common import find_galvo_peaks
            peak_index = find_galvo_peaks(signal)
        elif sync_type == 'Orca_Frame_Onset':
            thres = 2

            cross = (signal[1:] > thres) & (signal[:-1] <= thres)
            cross_index = np.where( cross )[0] + 1  # go to index above threshold

            # remove first crossing (which corresponds to the initialization of the camera)
            cross_index = cross_index[1:]
            # remove last frames which were not saved anymore (assuming requested frames are divisible by 100)
            requested_frames = (len( cross_index ) // 100) * 100
            peak_index = cross_index[:requested_frames]    # time in ms (in terms of behavior time)
        else:
            raise Exception('Syncronization not implemented for ' + sync_type)

        # go from peak half a cycle back in time and in seconds (or forward for widefield and multi-area)
        dt = np.median( np.diff(peak_index) )
        if (sync_type == 'Galvo_Y') or (sync_type == 'Galvo_Y_Clipped'):
            scan_time = (peak_index - int(dt/2) ) / 1000
        else:
            scan_time = (peak_index + int(dt/2) ) / 1000

        nr_detected_peaks = len(peak_index)
        avg_frame_rate = 1 / dt * 1000  # average frame rate in Hz

        # check if the last frame was maybe not saved => drop last value of scan_time
        nr_frames_tif = (img.ScanInfo & key).fetch1('nr_frames')
        if len(scan_time) == nr_frames_tif:
            print('Detected peaks match with ScanInfo.')
        elif (len(scan_time) > nr_frames_tif) and (len(scan_time) < nr_frames_tif+5):
            print('Removing {} frames at the end.'.format( len(scan_time)-nr_frames_tif ))
            scan_time = scan_time[:nr_frames_tif]

        else:
            print('There is a large mismatch between nr_frames_tif {} and detected peaks {}'.format(
                            nr_frames_tif,  len(scan_time) ))
            raise Exception()

        # enter variables into datajoint table
        new_entry = dict( **key,
                          scan_time = scan_time,
                          nr_detected_peaks = nr_detected_peaks,
                          avg_frame_rate = avg_frame_rate)
        self.insert1( new_entry )

        if plotting == True:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(2,1,1)

            plt.plot(time, signal)
            plt.xlim( (time[-1]/2, time[-1]/2 + 0.5) )  # show 500ms from the middle
            plt.xlabel('Time [s]')

            plt.axhline(thres, color='C1')

            for peak in peak_index:
                plt.axvline( peak/1000, color='k')   # peak position in seconds

            plt.legend(['Signal', 'Threshold', 'Detected peaks'])

            plt.subplot(2,1,2)
            plt.plot( np.diff(peak_index) )
            plt.title('Time between crossings')
            plt.ylabel('Time [ms]')

            plt.tight_layout()
            plt.show()

        log('Inserted ScanTime entry for key: {}'.format(key))


@schema
class SensoryEventType(dj.Lookup):
    definition = """ # Lookup table for different whisker stimulators
    sensory_event_type   : varchar(128)   # Short name for used sensory events
    ---
    details      : varchar(1024)          # Description
    """
    contents = [
        ['Basic_events', "Events for LED, sound and whisker stimulation without any additional information"],
        ['Events_with_pos', 'Events for LED and sound as for Basic_events, but whisker stim has Galvo In Out or Galvo movement to: -2.00'],
    ]


@schema
class SensoryEvents(dj.Manual):
    definition = """ # Timing of sensory events (e.g. LED, tone, whisker stimulation)
    -> exp.Session
    ---
    -> SensoryEventType
    """

    def get(self):
        """ Returns dictionary with event times in seconds """
        # Refer to table which stores the dict (function here for easier access)
        return (SensoryEventsDict & self).get_events()

    def get_descriptive(self, with_light=False, exclude_sweep=True):
        """ Returns dictionary with descriptive keys like 'caudal' or 'small forward' """
        event_dict = (SensoryEventsDict & self).get_events()
        return analysis.create_descriptive_event_dict(event_dict, with_light=with_light, exclude_sweep=exclude_sweep)

@schema
class RawSensoryEventsFile(dj.Manual):
    definition = """ # Link to data file in which sensory events are saved
    -> SensoryEvents
    ---
    file_name   : varchar(256)     # File name of the sensory event file
    """

    def get_path(self):
        """Return the absolute path of the file on the current system"""

        # Quick check if only one file is selected
        if len(self) != 1: raise Exception('Only length one allowed (not {})'.format(len(self)))

        base_directory = login.get_neurophys_directory()
        folder = (exp.Session() & self).fetch1('id')
        file = self.fetch1('file_name')

        return os.path.join( base_directory, folder, file )

    def get_raw_data(self):
        """Return a dictionary with the event times in ms for 'LED', 'Whisker' and 'Sound'
        Returns
        ---------
        event_times : Dictionary of 1d arrays
            Event times in ms recorded by the LabView program of behavior PC
            The keys are 'LED', 'Whisker' and 'Sound'
        Adrian 2019-10-21
        """

        # Data in the RawSensoryEventsFile is saved as a tab seperated list of
        # events which have one entry at the 'Time [ms]' and another for 'Event'
        import pandas as pd
        path = self.get_path()
        events = pd.read_csv( path, sep='\t')

        if (SensoryEvents() & self).fetch1('sensory_event_type') == 'Basic_events':
            # create dictionary for easier access to the data
            # events is pandas table with 'Time [ms]' and 'Event' as keys
            stims = ['LED', 'Whisker', 'Sound']
            event_times = dict()

            for stim in stims:
                table = events[events['Event'] == stim]
                event_times[stim] = np.array( table['Time [ms]'])

            return event_times

        else:  # new file format
            stims = ['LED', 'Sound', 'Galvo In Out', 'Galvo movement']
            event_times = dict()

            for stim in stims:
                table = events[events['Event'].str.contains(stim)]
                event_times[stim] = np.array( table['Time [ms]'])

                if stim == 'Galvo movement':  # use 2d array with time and position
                    # transform the string after : to a float and save a list
                    skip = len('Galvo movement to:')
                    list_movement_target = [ float( string[skip:]) for string in table['Event'] ]
                    event_times['Galvo movement target'] = np.array( list_movement_target )

            # Add information about light on or off
            if len( events[events['Event'] == 'Light_on']) > 0:
                for stim in ['Light_on', 'Light_off']:
                    table = events[events['Event'] == stim]
                    event_times[stim] = np.array( table['Time [ms]'])

                # create array with True for light on during the movement
                move_t = event_times['Galvo movement']
                movement_light_on = np.zeros( len(move_t), dtype=int )

                for on, off in zip( event_times['Light_on'], event_times['Light_off']):
                    movement_light_on[ (move_t > on) & (move_t < off) ] = 1

                event_times['Galvo movement light'] = movement_light_on > 0 # True/False

            # dictionary with arrays of event times, in case of 'Galvo movement target' array of target pos
            return event_times

@schema
class SensoryEventsDict(dj.Computed):
    definition = """ # Table to store events as dictionary
    -> RawSensoryEventsFile
    ---
    event_dict   : longblob     # Dictionary with events as keys and times in seconds as values
    """

    def make(self, key):
        """Automatically populate this table with data from RawSensoryEventsFile
        Adrian 2020-03-17
        """
        log('Populating SensoryEventsDict for key: {}'.format(key))
        dic = (RawSensoryEventsFile() & key).get_raw_data()

        for key_ in dic:
            if key_ not in ['Galvo movement target', 'Galvo movement light']:
                dic[key_] = dic[key_] / 1000   # transform to seconds

        key['event_dict'] = dic
        self.insert1( key )   # Dictionary will be stored as np.rec.array

    def get_events(self):
        """ Returns dictionary with event times in seconds """

        dic = self.fetch1('event_dict')

        if len(dic) == 1:
            print('Warning: Dictionary saved in old format. Transforming it now.')
            dic = dict(zip( dic.dtype.names, dic[0]))

        return dic


@schema
class StimulatorType(dj.Lookup):
    definition = """ # Lookup table for different whisker stimulators
    stimulator_type   : varchar(128)      # Short name for whisker stimulator
    ---
    details      : varchar(1024)          # Description of the used whisker stimulator
    """
    contents = [
        ['Galvo_Tape_v1', "Whisker stimulator with tape extension, controlled without using full range"],
        ['Galvo_carbon_T', 'Galvo with two carbon sticks attached in form of a T. Range from -5 to 5 of positions.'],
        ['Galvo_Box_Martin', 'Galvo with two carbon sticks to form a T. Stick in front white. Controlled by Martins Galvo Box.']
    ]


@schema
class WhiskerStimulator(dj.Manual):
    definition = """ # Position of the galvo whisker stimulator
    -> exp.Session
    ---
    -> StimulatorType
    """

    def get(self):
        """ Wrapper function: Returns two 1d arrays (time and filtered, downsampled whisker stim pos) """
        return (FilteredWhiskerStimulator & self).get_filtered_position()

@schema
class RawWhiskerStimulatorFile(dj.Manual):
    definition = """ # Link to data file in which the trace of the galvo position is saved
    -> WhiskerStimulator
    ---
    file_name   : varchar(256)     # File name of the stimulator galvo trace
    """

    def get_path(self):
        """Return the absolute path of the file on the current system"""

        # Quick check if only one file is selected
        if len(self) != 1: raise Exception('Only length one allowed (not {})'.format(len(self)))

        base_directory = login.get_neurophys_directory()
        folder = (exp.Session() & self).fetch1('id')
        file = self.fetch1('file_name')

        return os.path.join( base_directory, folder, file )

    def get_raw_data(self, include_time=False):
        """Return a 1d array with the recorded voltage of the galvo feedback position
        Returns
        ---------
        time : 1d array (only if include_time==True)
            Time in seconds recorded by the LabView program of behavior PC
        stimulator_position : 1d array
            Galvo position sampled at 1kHz at the behavior time
        Adrian 2019-10-21
        """
        # The signal is saved at 1kHz and all times in behav are relative to the
        # start of this recording
        path = self.get_path()
        stimulator_position = np.fromfile( path, dtype='>d')

        if include_time == False:
            return stimulator_position
        else:
            time = np.arange( len(stimulator_position) ) / 1000     # in seconds
            return time, stimulator_position


@schema
class FilteredWhiskerStimulator(dj.Computed):
    definition = """ # Table to store filtered and downsampled whisker stimulator (galvo) positions
    -> RawWhiskerStimulatorFile
    ---
    filtered_pos   : longblob     # 1d array filtered and sampled every 10 ms from raw file
    """

    def make(self, key):
        """Automatically populate this table with data from RawWhiskerStimulatorFile
        Adrian 2020-03-18
        """
        log('Populating FilteredWhiskerStimulator for key: {}'.format(key))
        raw_pos = (RawWhiskerStimulatorFile & key).get_raw_data()

        window_size = 41
        polynom_order = 3

        import scipy.signal as sig
        pos_f = sig.savgol_filter(raw_pos, window_size, polynom_order)
        key['filtered_pos'] = pos_f[::10]    # downsample to 100 Hz

        self.insert1( key )

    def get_filtered_position(self):
        """ Returns two 1d arrays: time and filtered, downsampled whisker stimulator position  """

        filtered_pos = self.fetch1('filtered_pos')      # signal sampled every 10ms
        time = np.arange( len(filtered_pos) ) / 100     # in seconds

        return time, filtered_pos

    def plot_quality_control(self):
        """ Plots the filtered and raw data above each other to double check filtering parameters
        Adrian 2020-03-18
        """

        import matplotlib.pyplot as plt
        t_f, pos_f = self.get_filtered_position()
        t_r, pos_r = (RawWhiskerStimulatorFile & self).get_raw_data(include_time=True)

        plt.figure()
        plt.plot(t_r, pos_r, label='Raw')
        plt.plot(t_f, pos_f, label='Filtered, downsampled')
        plt.xlabel('Time [s]')
        plt.legend()
