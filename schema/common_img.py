#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 25/06/2021 10:21
@author: hheise

Schemas for the CaImAn 2-photon image analysis pipeline
"""

# imports
import datajoint as dj
from . import exp, shared, alg
import login
from .utils.common import log

from .utils import scanimage
from .utils import scope

import os
import numpy as np


schema = dj.schema('img', locals(), create_tables = True)


CURRENT_VERSION = 0   # identifier for management of different versions
VERBOSE = 5           # control level of debug messages

@schema
class Version(dj.Lookup):
    definition = """ # Version control for the imaging pipeline
    version      : int    # version index (base 0)
    ---
    description  : varchar(256)      # Notes on this version
    """
    contents = [ [0, 'Setting up pipeline'] ]


@schema
class Microscope(dj.Lookup):
    definition = """ # Used microscope for scanning
    microscope         : varchar(128)     # Microsope short name
    ---
    microscope_details : varchar(1048)    # Additional details
    """

    contents = [
        ['H45', '4-area-microscope in H45 in the front'],
        ['Scientifica', 'Scientifica microscope in H37 (right)'],
    ]

@schema
class Laser(dj.Lookup):
    definition = """ # Laser used for scanning
    laser    : varchar(128)   # Laser short name
    ---
    laser_details : varchar(1024)   # Additional details
    """

    contents = [
        ['OneFive', 'OneFive laser, 1050nm, 100fs, 5W'],
        ['MaiTai', 'MaiTai laser, tunable, power depending on wavelength'],
        ['1040', 'add details...']
        # TODO: add 1040 laser from scientifica
    ]


@schema
class Layer(dj.Lookup):
    definition = """ # Layers of the scanning
    layer : varchar(128)    # Short identifier of layer
    """
    contents = [
        ['L1'],
        ['L2/3'],
        ['L4'],
        ['L5'],
        ['L6'],
        ['Multi'],
    ]

@schema
class Scan(dj.Manual):
    definition = """ # Basic info about the recorded scan
    -> exp.Session
    ---
    -> Microscope
    -> Laser
    -> Layer
    objective = '16x' : varchar(64)  # Used objective for the scan
    nr_channels = 1   : int          # Number of recorded channels (1 or 2)
    nr_areas = 1      : int          # Number of simultaneous areas (1 to 4)
    nr_planes = 1     : int          # Number of successively scanned planes by hopping (1 to N)
    """

    def select_channel_if_necessary(self, key, return_channel):
        """ Return None if only 1 channel, otherwise the return_channel is used
        Adrian 2021-05-21 """
        if (Scan & key).fetch1('nr_channels') == 1:
            return None
        else:
            return return_channel

@schema
class BrainRegionAssignment(dj.Manual):
    definition = """ # Brain region for the selected area
    -> Scan
    area = 0        : int     # Same id as used for RawImagingFiles
    -----
    -> shared.BrainRegion
    """


@schema
class RawParameterFile(dj.Manual):
    definition = """ # File name of the parameters.xml file for H45 scans
    -> Scan
    ---
    file_name  : varchar(512)    # File name of parameters.xml file
    """
    def get_path(self):
        """Return the absolute path of the file on the current system"""

        if len(self) != 1: raise Exception('Only length one allowed (not {})'.format(len(self)))

        base_directory = login.get_neurophys_data_directory()
        folder = (exp.Session() & self).fetch1('id')
        file = self.fetch1('file_name')

        return os.path.join( base_directory, folder, file )

@schema
class RawImagingFile(dj.Manual):
    definition = """ # File names of the acquired .tif raw imaging files
    -> Scan
    part = 1    : tinyint                   # Counter for parts of the same scan (base 1)
    ---
    file_name   : varchar(512)              # File name, for the path use the functions get_file_path(s)
    frames      : smallint unsigned         # Number of frames in this file
    """

    def get_path(self):
        """ Returns a string with the absolute file path for a single file on given system
        This function uses the file paths defined in login.py to find the location
        of files for the current machine on which the code is executed.
        Adrian 2019-08-12
        """
        # Quick check if only one file is selected
        if len(self) != 1: raise Exception('Only length one allowed (not {})'.format(len(self)))

        # Return file at remote location
        base_directory = login.get_neurophys_directory()
        folder = (exp.Session() & self).fetch1('id')
        file = self.fetch1('file_name')

        return os.path.join( base_directory, folder, file )

    def get_paths(self):
        """ Return multiple files (in case of scan split up in multiple files) """
        keys = self.fetch(dj.key)
        path_list = list()
        for key in keys:
            path_list.append( (self & key).get_path() )
        return path_list


@schema
class ScanInfo(dj.Computed):
    definition = """ # Scan specific information common to all planes and channels
    -> Scan
    -> Version      # Include version for all following computed tables
    ---
    fs             : float         # Sampling rate in Hz
    zoom           : float         # Selected zoom setting (higher: more magnification)
    nr_lines       : int           # Number of lines in the scan field (Height)
    pixel_per_line : int           # Number of pixel per line (Width)
    scanning       : varchar(128)  # Scanning type (Resonant or Galvo)
    pockels        : float         # Setting on pockels cell to modulate power
    x_motor = -1.  : float         # X Motor position of microscope (relative to zeroed position)
    y_motor = -1.  : float         # Y Motor position of microscope (relative to zeroed position)
    z_motor = -1.  : float         # Z Motor position of microscope (relative to zeroed position)
    nr_frames = -1 : int           # Total number of frames in this recording
    """

    def make(self, key):
        """Automatically populate the ScanInfo table
        # TODO: Save locations of scan fields in part tables
                Create part tables for channel information
        Adrian 2019-08-21
        """
        log('Populating ScanInfo for key: {}'.format(key))

        if (Scan & key).fetch1('microscope') == 'Scientifica':
            # Extract meta-information from imaging .tif file

            path = (RawImagingFile & key).get_paths()[0]   # Extract only from first file
            info = scanimage.get_meta_info_as_dict( path )
            info['pockels'] = info['powers'][0] # TODO: remove hardcoding of MaiTai laser

        elif (Scan & key).fetch1('microscope') == 'H45':
            # get meta-info from parameter file
            path = (RawParameterFile & key).get_paths()[0]
            info = scope.get_meta_info_as_dict( path )

        else:
            raise Exception('Only Scientifica and H45 micrsocope supported so far.')

        new_entry = dict( **key,
                        fs = info['fs'],
                        zoom = info['zoom'],
                        nr_lines = info['nr_lines'],
                        pixel_per_line = info['pixel_per_line'],
                        scanning = info['scanning'],
                        pockels = info['pockels'],
                        )
        # Add positions for now just for Scientifica
        if (Scan & key).fetch1('microscope') == 'Scientifica':
            new_entry = dict( **new_entry,
                             x_motor = info['motor_pos'][0],
                             y_motor = info['motor_pos'][1],
                             z_motor = info['motor_pos'][2],
                            )

        # Add number of frames
        import tifffile as tif
        paths = (RawImagingFile & key & {'area':0}).get_paths()
        nr_frames = 0
        for path in paths:
            # get length without loading whole stack in memory
            nr_frames += len( tif.TiffFile(path).pages )

        if (Scan & key).fetch1('nr_channels') == 2:
            nr_frames = int( nr_frames / 2 ) # double amout due to interleaved saving

        new_entry['nr_frames'] = nr_frames

        self.insert1( new_entry )
        log('Finished populating ScanInfo for key: {}'.format(key))


@schema
class MotionCorrection(dj.Computed):
    definition = """ # Motion correction of the scan, each area in part table
    -> ScanInfo
    motion_id = 0  : int      # Additional key for different motion correction algorithms
    ------
    avg_shifts     : longblob       # 2d array (xy, nr_frames) of average shift of all sub-areas
    avg_x_std      : float          # Average standard deviation of shifts in x (left/right)
    avg_y_std      : float          # Average standard deviation of shifts in y (top/bottom)
    avg_x_max      : int            # Maximum of averge shifts in x
    avg_y_max      : int            # Maximum of average shifts in y
    outlier_frames : longblob       # 1d array with detected outlier in motion correction
    align_time=CURRENT_TIMESTAMP : timestamp     # Automatic timestamp of alignment
    with_params = 0 : int           # Flag to indicate whether the following additional parameters were set.
    max_shift   = 0 : int           # Maximum shift value allowed by the motion correction.
    n_iter      = 1 : int           # Number of iterations of the motion correction.
    crop_left   = 0 : int           # Number of pixels to cut away on the left to remove scanning artifacts before the motion correction.
    crop_right  = 0 : int           # Number of pixels to cut away on the right to remove scanning artifacts before the motion correction.
    offset      = 0 : int           # Fixed value that is added to all pixels in the scan to make values positive everywhere.
    line_shift  = 0 : int           # Detected shift between even and odd lines.
    """

    class Area(dj.Part):
        definition = """ # Motion correction for each area
        -> master
        -> shared.Area
        ------
        shifts               : longblob     # 2d array (xy, nr_frames) of shift of sub-area
        x_std                : float        # Standard deviation of shifts in x (left/right)
        y_std                : float        # Standard deviation of shifts in y (top/bottom)
        x_max                : int          # Maximal shift in x for this area
        y_max                : int          # Maximal shift in x for this area
        template             : longblob     # 2d image of used template
        template_correlation : longblob     # 1d array (nr_frames) with correlations with the template
        """

    def make(self, key):
        """ Automatically populate the MotionCorrection for all areas of this scan
        TODO:   - include motion correction of the second channel with the same parameters as primary channel
                - for multiple planes, remove the black stripe in the middle
        Adrian 2019-08-21
        """
        log('Populating MotionCorrection for key: {}'.format(key))

        # Imports (this code has to be run in a CaImAn environment)
        import caiman as cm
        from caiman.motion_correction import MotionCorrect
        from caiman.source_extraction.cnmf import params as params
        from .utils import motion_correction

        n_processes = 4       # TODO: remove hardcoding and choose as function of recording and RAM size?

        # start the cluster (if a cluster already exists terminate it)
        if 'dview' in locals():
            cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=n_processes, single_thread=False)

        ## get the parameters for the motion correction (hardcoded, not datajoint style)
        CROP_LEFT = 10
        CROP_RIGHT = 10
        OFFSET = 220
        MAX_SHIFT = 15
        N_ITER = 2
        # Select second channel in case there are 2 (base 0 index)
        CHANNEL_SELECT = Scan().select_channel_if_necessary(key, 1)


        opts_dict = {
            'max_shifts': (MAX_SHIFT,MAX_SHIFT),
            'pw_rigid': False,
            'nonneg_movie': False,
            'niter_rig':N_ITER}

        opts = params.CNMFParams(params_dict=opts_dict)

        # perform motion correction area by area
        nr_areas = (Scan & key).fetch1('nr_areas')
        new_part_entries = list()    # save new entries for part tables
        part_mmap_files = list()

        for area in range( nr_areas ):  # TODO: fix if area 1 and 3 are recorded...
            # get path to file for this area and locally cache files
            paths = (RawImagingFile & key & {'area':area}).get_paths()
            local_cache = login.get_cache_directory()

            local_paths = motion_correction.cache_files(paths, local_cache)
            paths = None   # make sure that the files at neurophysiology are not used by accident

            corrected_files = list()

            # If recording was done with scope, then the image has to be warped
            # first to get same x and y resolution
            if (Scan & key).fetch1('microscope') == 'H45':
                # this results in a temporary file, which will be deleted later
                for path in local_paths:
                    new_path = scope.warp_resonance_recording(path)
                    corrected_files.append( new_path )

                # set unused variables to 0 for compatibility with Scientifica part
                line_shift, CROP_LEFT, CROP_RIGHT, OFFSET = (0,0,0,0)

            elif (Scan & key).fetch1('microscope') == 'Scientifica':
                log('Calculating shift between even and odd lines...')
                # For multiple channels, take images from both (offset should be the same)
                line_shift = motion_correction.find_shift_multiple_stacks( local_paths )
                log('Detected line shift of {} pixel'.format(line_shift))

                for path in local_paths:
                    # apply raster and offset correction and save as new file
                    new_path = motion_correction.create_raster_and_offset_corrected_file(
                                local_file=path, line_shift=line_shift, offset=OFFSET,
                                crop_left=CROP_LEFT, crop_right=CROP_RIGHT,
                                channel=CHANNEL_SELECT )
                    corrected_files.append( new_path )

            else:
                raise Exception('This should not occur...')

            # delete not corrected files from cache to save storage
            log('Deleting raw cached files...')
            motion_correction.delete_cache_files(local_paths)

            # perform actual motion correction
            mc = MotionCorrect(corrected_files, **opts.get_group('motion'), dview=dview)

            log('Starting CaImAn motion correction for area {}...'.format(area) )
            log('Used parameters: {}'.format( opts.get_group('motion') ))
            mc.motion_correct(save_movie=True)
            log('Finished CaImAn motion correction for area {}.'.format(area) )

            log('Remove temporary created files (H45: warped, Scientifica: raster+edge correction)')
            for file in corrected_files:
                 os.remove( file )

            # the result of the motion correction is saved in a memory mapped file
            mmap_files = mc.mmap_file   # list of files
            part_mmap_files.append( mc.mmap_file )

            # extract and calculate information about the motion correction
            shifts = np.array( mc.shifts_rig ).T # caiman output: list with x,y shift tuples
            template = mc.total_template_rig

            log('Calculate correlation between template and frames...')
            template_correlations = list()
            for mmap_file in mmap_files:
                template_correlations.append( motion_correction.calculate_correlation_with_template(
                                            mmap_file, template, sigma=2) )
            template_correlation = np.concatenate( template_correlations )

            # delecte memory mapped files
            for file in mmap_files:
                os.remove( file )

            new_part_entries.append(
                    dict( **key,
                          motion_id = 0,
                          area = area,
                          shifts = shifts,
                          x_std = np.std( shifts[0,:]),
                          y_std = np.std( shifts[1,:]),
                          x_max = int( np.max( np.abs( shifts[0,:] )) ),
                          y_max = int( np.max( np.abs( shifts[1,:] )) ),

                          template = template,
                          template_correlation = template_correlation)
                                   )
        # stop cluster
        cm.stop_server(dview=dview)

        # After all areas have been motion corrected, calculate overview stats

        # TODO: Implement both functions (currently just placeholders)
        outlier_frames = motion_correction.find_outliers( new_part_entries )
        avg_shifts = motion_correction.calculate_average_shift( new_part_entries, outlier_frames )

        # insert MotionCorretion main table
        new_main_entry = dict( **key,
                               avg_shifts = avg_shifts,
                               avg_x_std = np.std( avg_shifts[0,:]),
                               avg_y_std = np.std( avg_shifts[1,:]),
                               avg_x_max = int( np.max( np.abs( avg_shifts[0,:] )) ),
                               avg_y_max = int( np.max( np.abs( avg_shifts[1,:] )) ),

                               # new: parameters of motion correction
                               with_params = 1,
                               max_shift = MAX_SHIFT,
                               n_iter = N_ITER,
                               crop_left = CROP_LEFT,
                               crop_right = CROP_RIGHT,
                               offset = OFFSET,
                               line_shift = line_shift,

                               outlier_frames = outlier_frames)
        self.insert1( new_main_entry )

        # insert information about areas in part tables
        for area in range( nr_areas ):
            self.Area.insert1( new_part_entries[area] )


        log('Finished populating MotionCorrection for key: {}'.format(key))


@schema
class MemoryMappedFile(dj.Manual):
    definition = """ # Table to store motion corrected memory mapped file (C-order) used for ROI detection
    -> MotionCorrection.Area
    -----
    mmap_path : varchar(256)    # path to the cached motion corrected memory mapped file
    """

    ## entries are inserted during population of the motion correction table
    def create(self, key, channel=None):
        """ Creates a memory mapped file with raster and motion correction and cropping
        If channel is given (0 or 1), the stack is deinterleaved before corrections.
        Adrian 2020-07-22 """
        from .utils import motion_correction
        import caiman as cm

        log('Creating memory mapped file...')

        if len( MemoryMappedFile() & key ) != 0:
            raise Exception('The memory mapped file already exists!')

        # get parameter from motion correction
        line_shift = (MotionCorrection() & key).fetch1('line_shift')
        offset = (MotionCorrection() & key).fetch1('offset')
        max_shift_allowed = (MotionCorrection() & key).fetch1('max_shift')
        xy_shift = (MotionCorrection.Area() & key).fetch1('shifts')  # (2 x nr_frames)

        # save raw recordings locally in cache
        paths = (RawImagingFile & key).get_paths()
        local_cache = login.get_cache_directory()

        local_paths = motion_correction.cache_files(paths, local_cache)
        paths = None   # make sure that the files at neurophysiology are not used by accident

        # correct line shift between even and odd lines and add offset
        corrected_files = list()
        for path in local_paths:
            # apply raster and offset correction and save as new file
            new_path = motion_correction.create_raster_and_offset_corrected_file(
                        local_file=path, line_shift=line_shift, offset=offset,
                        crop_left=0, crop_right=0, channel=channel )
            corrected_files.append( new_path )

        motion_correction.delete_cache_files(local_paths)

        # apply motion correction file by file (to save hard-disk storage, only 100GB available on ScienceCloud)
        import tifffile as tif
        # get number of frames without loading whole stack
        nr_frames_per_file = len( tif.TiffFile(corrected_files[0]).pages )
        scan_size = (ScanInfo & key).fetch1('pixel_per_line')

        shift_parts = list()
        for i in range(0, xy_shift.shape[1], nr_frames_per_file):
            shift_parts.append( xy_shift[:,i:i+nr_frames_per_file].T)

        temp_mmap_files = list()
        for i, file in enumerate(corrected_files):
            part_file = cm.save_memmap( [file], xy_shifts=shift_parts[i], base_name='tmp{:02d}_'.format(i+1), order='C',
                                      slices = (slice(0,100000),
                                                slice(max_shift_allowed,scan_size-max_shift_allowed),
                                                slice(max_shift_allowed,scan_size-max_shift_allowed)))
            temp_mmap_files.append(part_file)
            motion_correction.delete_cache_files([file])   # save

        # combine parts of stack to one single file
        mmap_file = cm.save_memmap(temp_mmap_files, base_name='mmap_', order='C')

        # delete temporary files
        motion_correction.delete_cache_files(temp_mmap_files)

        # create new entry in database

        # make sure no key attributes are too much or missing
        area_key = (MotionCorrection.Area & key).fetch1('KEY')

        new_entry = dict(**area_key,
                        mmap_path = mmap_file)

        self.insert1(new_entry)
        log('Finished creating memory mapped file.')

    def delete_mmap_file(self):
        """ Delete memory-mapped file from cache and remove entry
        Adrian 2020-07-22 """

        mmap_file = self.fetch1('mmap_path')

        from .utils import motion_correction
        motion_correction.delete_cache_files( [mmap_file] )

        self.delete_quick()  # delete without user confirmation


    def export_tif(self, nr_frames=100000, target_folder=None, dtype='tif', prefix=''):
        """ Export a motion corrected memory mapped file to an ImageJ readable .tif stack or .h5
        Parameters
        ----------
        nr_frames : int  (default 100000 means all)
            Number of frames to export starting from the beginning
        target_folder: str (default None)
            Destination folder of the exported .tif. If None, use folder of the .mmep file
        dtype : str (default tif)
            Data type to store results in, possible values: 'tif' or 'h5'
        prefix : str (default ''):
            Optional prefix to identify the exported file more easily
        Adrian 2019-03-21
        """

        mmap_file = self.fetch1('mmap_path')   # only one file at a time allowed
        key = self.fetch1(dj.key)

        if dtype not in ['tif', 'h5']:
            raise Exception('Only "tif" or "h5" allowed as dtype, not "{}"'.format(dtype))

        file = 'motionCorrected_mouse_{name}_day_{day}_trial_{trial}_area_{area}'.format( **key )
        file = str(prefix) + file + '.' + dtype

        if target_folder is None:
            # use the directory of the session in the datajoint directory
            base_directory = login.get_neurophys_directory()
            folder = (exp.Session() & self).fetch1('id')
            target_folder = os.path.join( base_directory, folder)

        path = os.path.join( target_folder, file )

        # load memory mapped file and transform it to 16bit and C order
        import caiman as cm

        corrected = cm.load( mmap_file )    # frames x height x width

        corrected_int = np.array( corrected[:nr_frames,:,:] , dtype='int16' )
        corrected = None   # save memory

        toSave_cOrder = corrected_int.copy(order='C')
        corrected_int = None   # save memory

        if dtype == 'tif':
            # if this throws an error, the tifffile version might be too old
            # print('Tifffile version: {}'.format(tif.__version__) )
            # upgrade it with: !pip install --upgrade tifffile
            import tifffile as tif
            tif.imwrite( path, data=toSave_cOrder)

        elif dtype == 'h5':
            import h5py
            with h5py.File(path, 'w') as h5file:
                h5file.create_dataset('scan', data=toSave_cOrder, dtype=np.int16)

        print('Done!')

@schema
class QualityControl(dj.Computed):
    definition = """ # Images and metrics of the motion corrected stack for quality control
    -> MotionCorrection.Area
    ----
    avg_image : longblob        # 2d array: Average intensity image
    cor_image : longblob        # 2d array: Correlation with 8-neighbors image
    std_image : longblob        # 2d array: Standard deviation of each pixel image
    min_image : longblob        # 2d array: Minimum value of each pixel image
    max_image : longblob        # 2d array: Maximum value of each pixel image
    percentile_999_image : longblob   # 2d array: 99.9 percentile of each pixel image
    mean_time : longblob        # 1d array: Average intensity over time
    """

    def make(self, key):
        """ Automatically compute qualtiy control metrics for scan
        Adrian 2020-07-22 """

        log('Populating QualityControl for key: {}.'.format(key))

        import caiman as cm
        from .utils import motion_correction

        if len( MemoryMappedFile() & key ) == 0:
            # In case of multiple channels, deinterleave and return channel 0 (GCaMP signal)
            channel = Scan().select_channel_if_necessary(key, 0)
            MemoryMappedFile().create(key, channel=channel)

        mmap_file = (MemoryMappedFile & key).fetch1('mmap_path')  # locally cached file

        stack = cm.load( mmap_file )

        new_entry = dict( **key,
                        avg_image = np.mean( stack, axis=0 ),
                        std_image = np.std( stack, axis=0 ),
                        min_image = np.min( stack, axis=0 ),
                        max_image = np.max( stack, axis=0 ),
                        percentile_999_image = np.percentile( stack, 99.9, axis=0 ),
                        mean_time = np.mean( stack, axis=(1,2) ),
                        )

        # calculate correlation with 8 neighboring pixels in parallel
        new_entry['cor_image'] = motion_correction.parallel_all_neighbor_correlations(stack)

        self.insert1( new_entry)
        log('Finished populating QualityControl for key: {}.'.format(key))

    def plot_avg_on_blood_vessels(self, axes=None,
                                with_vessels=True, with_scale=False):
        """ Plot 2p neuron scan on top of picture of window with blood vessels
        Adrian 2020-07-27 """

        image = self.fetch1('avg_image')

        # reuse the code to plot the 2p vessel pattern
        axes = (alg.VesselScan & self).plot_scaled(axes=axes, image=image, with_scale=with_scale)

        return axes

@schema
class CaimanParameter(dj.Lookup):
    definition = """ # Table which stores all CaImAn Parameters
    caiman_id : int           # index for parameters, base 0
    ----
    max_shift = 60 : int            # maximum allowed rigid shifts (in pixels) need to be transformed to (x,y) tuple for caiman
    pw_rigid = 0 : tinyint          # flag for performing non-rigid motion correction (0:False, 1:True)
    p = 1 : int                    # order of the autoregressive system
    gnb = 2 : int                  # number of global background components
    merge_thr = 0.85 : float       # merging threshold, max correlation allowed
    rf = 100 : int                 # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 10 : int         # amount of overlap between the patches in pixels
    k = 3 : int                    # number of components per patch, need to rename to K for caiman
    g_sig = 15 : int               # expected half size of neurons in pixels, need to rename to gSig for caiman and (x,y) tuple
    method_init = 'greedy_roi' : varchar(128) # initialization method (if analyzing dendritic data using 'sparse_nmf')
    ssub = 1 : int                 # spatial subsampling during initialization
    tsub = 1 : int                 # temporal subsampling during intialization
    # parameters for component evaluation
    min_snr = 2.0 : float          # signal to noise ratio for accepting a component, need to rename to min_SNR for caiman
    rval_thr = 0.85 : float        # space correlation threshold for accepting a component
    cnn_thr = 0.99 : float         # threshold for CNN based classifier
    cnn_lowest = 0.1 : float       # neurons with cnn probability lower than this value are rejected
    """

    def get_parameter_obj(self, scan_key):
        """ Get object of type params.CNMFParams for CaImAn
        """
        frame_rate = (ScanInfo & scan_key).fetch1('fs')
        # TODO: use decay time depending on used calcium indicator
        decay_time = 0.4

        opts_dict = { # 'fnames': fnames,
            'fr': frame_rate,
            'decay_time': decay_time,
            'max_shifts': (self.fetch1('max_shift'),self.fetch1('max_shift') ),
            'pw_rigid': self.fetch1('pw_rigid') == 1,

            'p': self.fetch1('p'),
            'nb': self.fetch1('gnb'),
            'rf': self.fetch1('rf'),
            'K': self.fetch1('k'),
            'gSig': ( self.fetch1('g_sig'), self.fetch1('g_sig') ),
            'stride': self.fetch1('stride_cnmf'),
            'method_init': self.fetch1('method_init'),
            'rolling_sum': True,
            'only_init': True,
            'ssub': self.fetch1('ssub'),
            'tsub': self.fetch1('tsub'),
            'merge_thr': self.fetch1('merge_thr'),
            'min_SNR': self.fetch1('min_snr'),
            'rval_thr': self.fetch1('rval_thr'),
            'use_cnn': True,
            'min_cnn_thr': self.fetch1('cnn_thr'),
            'cnn_lowest': self.fetch1('cnn_lowest')
            }

        from caiman.source_extraction.cnmf import params as params

        # fill in None for -1
        if opts_dict['rf'] == -1:
            opts_dict['rf'] = None

        # fix pw_rigid error in save hdf5 file below, save as true bool, not numpy.bool_
        if opts_dict['pw_rigid'] == False:
            opts_dict['pw_rigid'] = False
        else:
            opts_dict['pw_rigid'] = True


        opts = params.CNMFParams(params_dict=opts_dict)

        return opts



@schema
class AreaSegmentation(dj.Computed):
    definition = """ # Table to store results of Caiman segmentation per area into ROIs
    -> MotionCorrection.Area
    -> CaimanParameter
    ------
    nr_masks : int            # Number of total detected masks in this area (includes rejected masks)
    target_dim : longblob     # Tuple (dim_y, dim_x) to reconstruct mask from linearized index
    time_seg = CURRENT_TIMESTAMP : timestamp   # automatic timestamp
    """

    class ROI(dj.Part):
        definition = """ # Data from mask created by Caiman
        -> AreaSegmentation
        mask_id : int      #  Mask index, per area (base 0)
        -----
        pixels   : longblob     # Linearized indicies of non-zero values
        weights  : longblob     # Corresponding values at the index position
        trace    : longblob     # Extracted raw fluorescence signal for this ROI
        dff      : longblob     # Normalized deltaF/F fluorescence change
        accepted : tinyint      # 0: False, 1: True
        """
## functions of part table
        def get_roi(self):
            """Returns the ROI mask as dense 2d array of the shape of the imaging field
            TODO: add support for multiple ROIs at a time
            Adrian 2019-09-05
            """
            if len(self) != 1:
                raise Exception('Only length one allowed (not {})'.format(len(self)))

            from scipy.sparse import csc_matrix
            # create sparse matrix (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html)
            weights = self.fetch1('weights')
            pixels = self.fetch1('pixels')
            dims = (AreaSegmentation() & self).fetch1('target_dim')

            sparse_matrix = csc_matrix( (weights, (pixels, np.zeros(len(pixels)) )), shape=(dims[0]*dims[1], 1) )

            # transform to dense matrix
            return np.reshape( sparse_matrix.toarray(), dims, order='F' )

        def get_roi_center(self):
            """ Returns center of mass of a single ROI as int array of length 2
            Adrian 2019-09-05
            """
            if len(self) != 1:
                raise Exception('Only length one allowed (not {})'.format(len(self)))

            roi_mask = self.get_roi()

            # calculate center of mass of mask by center of two projections
            proj1 = np.sum( roi_mask, axis=0)
            index1 = np.arange( proj1.shape[0] )
            center1 = np.inner( proj1, index1 ) / np.sum(proj1)  # weighted average index

            proj2 = np.sum( roi_mask, axis=1)
            index2 = np.arange( proj2.shape[0] )
            center2 = np.inner( proj2, index2 ) / np.sum(proj2)  # weighted average index

            return np.array( [np.round(center1), np.round(center2)], dtype=int )



## make of main table AreaSegmentation
    def make(self, key):
        """Automatically populate the segmentation for one area of the scan
        Adrian 2019-08-21
        """
        log('Populating AreaSegmentation for {}.'.format(key))

        import caiman as cm
        from caiman.source_extraction.cnmf import cnmf as cnmf

        if len( MemoryMappedFile() & key ) == 0:
            # In case of multiple channels, deinterleave and return channel 0 (GCaMP signal)
            channel = Scan().select_channel_if_necessary(key, 0)
            MemoryMappedFile().create(key, channel)

        mmap_file = (MemoryMappedFile & key).fetch1('mmap_path')  # locally cached file

        # get parameters
        opts = (CaimanParameter() & key).get_parameter_obj(key)
        log('Using the following parameters: {}'.format( opts.to_dict() ))
        p = opts.get('temporal','p')   # save for later

        # load memory mapped file
        Yr, dims, T = cm.load_memmap(mmap_file)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
                #load frames in python format (T x X x Y)

        # start new cluster
        n_processes = 1
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=n_processes, single_thread=True)

        # disable the most common warnings in the caiman code...
        import warnings
        from scipy.sparse import SparseEfficiencyWarning
        warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)

        # First extract spatial and temporal components on patches and combine them
        # for this step deconvolution is turned off (p=0)
        opts.change_params({'p': 0})
        cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
        log('Starting CaImAn on patches...')
        cnm = cnm.fit(images)
        log('Done.')

        #%% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
        cnm.params.set('temporal', {'p': p})
        log('Starting CaImAn on the whole recording...')
        cnm2 = cnm.refit(images, dview=None)
        log('Done')

        # evaluate components
        cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)

        #%% Extract DF/F values
        cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=500)

        save_results = True
        if save_results:
            log('Saving results also to file.')
            folder = (exp.Session() & key).get_folder()
            file = 'tmp_segmentation_caiman_id_{}.hdf5'.format(key['caiman_id'])
            cnm2.save( os.path.join( folder, file) )

        # stop cluster
        cm.stop_server(dview=dview)

        ## reset warnings to normal:
        # warnings.filterwarnings('default', category=FutureWarning)
        warnings.filterwarnings('default', category=SparseEfficiencyWarning)

        # save caiman results in easy to read datajoint variables

        masks = cnm2.estimates.A   # (flattened_index, nr_masks)
        nr_masks = masks.shape[1]

        accepted = np.zeros( nr_masks )
        accepted[ cnm2.estimates.idx_components ] = 1

        traces = cnm2.estimates.C    # (nr_masks, nr_frames)
        dff = cnm2.estimates.F_dff   # (nr_masks, nr_frames)


        #### insert results in master table first
        new_master_entry = dict( **key,
                                nr_masks = nr_masks,
                                target_dim = np.array(dims))
        self.insert1( new_master_entry )

        #### insert the masks and traces in the part table
        for i in range( nr_masks ):
            new_part = dict( **key,
                             mask_id = i,
                             pixels = masks[:,i].indices,
                             weights = masks[:,i].data,
                             trace = traces[i,:],
                             dff = dff[i,:],
                             accepted = accepted[i])
            AreaSegmentation.ROI().insert1( new_part )

        # delete MemoryMappedFile to save storage
        (MemoryMappedFile & key).delete_mmap_file()

        log('Finished populating AreaSegmentation for {}.'.format(key))


##### More functions for AreaSegmentation
    def print_info(self):
        """ Helper function to print some information about selected entries
        Adrian 2020-03-16
        """
        roi_table = AreaSegmentation.ROI() & self
        total_units = len( roi_table )
        accepted_units = len( roi_table & 'accepted=1' )
        print('Total units:', total_units)
        print('Accepted units: ', accepted_units)


    def get_traces(self, only_accepted = True, trace_type = 'dff', include_id = False, decon_id=None):
        """ Main function to get fluorescent traces in format (nr_traces, timepoints)
        Parameters
        ----------
        only_accepted : bool  (default True)
            If True, only return traces which have the property AreaSegmentation.ROI accepted==1
        trace_type : str (default dff)
            Type of the trace, either 'dff' for delta F over F, or 'trace' for absolute signal values
            or 'decon' for spike rates
        include_id : bool (default False)
            If True, this functions returns a second argument with the mask ID's of the returned signals
        decon_id : int (default None)
            Additional restriction, in case trace_type 'decon' is selected and multiple deconvolution
            models have been run. In case of only one model, function selects this one.
        Returns
        -------
        2D numpy array (nr_traces, timepoints)
            Fluorescent traces
        optional: 1D numpy array (nr_traces)
            Second argument returned only if include_id==True, contains mask ID's of the rows i
        Adrian 2020-03-16
        """

        # some checks to catch errors in the input arguments
        if not trace_type in ['dff', 'trace', 'decon']:
            raise Exception('The trace_type "%s" is not allowed as input argument!' % trace_type)

        # check if multiple caiman_ids are selected with self
        caiman_ids = self.fetch('caiman_id')
        if len( set(caiman_ids) ) != 1:   # set returns unique entries in list
            raise Exception('You requested traces from more the following caiman_ids: {}\n'.format(set(caiman_ids)) + \
                            'Choose only one of them with & "caiman_id = ID"!')

        # return only accepted units if requested
        if only_accepted:
            selected_rois = AreaSegmentation.ROI() & self & 'accepted = 1'
        else:
            selected_rois = AreaSegmentation.ROI() & self

        if trace_type in ['dff', 'trace']:
            traces_list = selected_rois.fetch(trace_type, order_by=('area', 'mask_id'))
        else: # decon
            if only_accepted == False:
                raise Exception('For deconvolved traces, only accepted=True is populated. Set only_accepted=True.')

            # if no decon_id is given, check if there is a single correct one, otherwise error
            if decon_id is None:
                decon_ids = (Deconvolution & self).fetch('decon_id')
                if len(decon_ids) == 1:
                    decon_id = decon_ids[0]
                else:
                    raise Exception('The following decon_ids were found: {}. Please specify using parameter decon_id.'.format(decon_ids))

            table = Deconvolution.ROI() & selected_rois & {'decon_id':decon_id}
            traces_list = table.fetch('decon', order_by=('area', 'mask_id'))

        # some more sanity checks to catch common errors
        if len(traces_list) == 0:
            print('Warning: The query img.AreaSegmentation().get_traces() resulted in no traces!')
            return None
        # check if all traces have the same length and can be transformed into 2D array
        if not all( len(trace) == len(traces_list[0]) for trace in traces_list):
            raise Exception('Error: The traces in traces_list had different lengths (probably from different recordings)!')

        traces = np.array( [trace for trace in traces_list])   # (nr_traces, timepoints) array

        if not include_id:
            return traces

        else: # include mask_id as well
            # TODO: return area as well if this is requested
            mask_ids = selected_rois.fetch('mask_id', order_by=('area', 'mask_id'))
            return (traces, mask_ids)


@schema
class DeconvolutionModel(dj.Lookup):
    definition = """ # Table for different deconvolution methods
    decon_id      : int            # index for methods, base 0
    ----
    model_name    : varchar(128)   # Name of the model
    sampling_rate : int            # Sampling rate [Hz]
    smoothing     : float          # Std of gaussian to smooth ground truth spike rate
    causal        : int            # 0: symmetric smoothing, 1: causal kernel
    nr_datasets   : int            # Number of datasets used for training the model
    threshold     : int            # 0: threshold at zero, 1: threshold at height of one spike
    """
    contents = [
        [0, 'Universal_15Hz_smoothing100ms_causalkernel', 15, 0.1, 1, 18, 0],
        [1, 'Universal_15Hz_smoothing100ms_causalkernel', 15, 0.1, 1, 18, 1],
        [2, 'Universal_30Hz_smoothing50ms_causalkernel', 30, 0.1, 1, 18, 1],
                ]
@schema
class Deconvolution(dj.Computed):
    definition = """ # Table to store deconvolved traces (only for accepted units)
    -> AreaSegmentation
    -> DeconvolutionModel
    ------
    time_decon = CURRENT_TIMESTAMP : timestamp   # automatic timestamp
    """

    class ROI(dj.Part):
        definition = """ # Data from mask created by Caiman
        -> Deconvolution
        mask_id : int        #  Mask index (as in AreaSegmentation.ROI), per area (base 0)
        -----
        decon   : longblob   # 1d array with deconvolved activity
        """

    def make(self, key):
        """ Automatically populate deconvolution for all accepted traces of AreaSegementation.ROI
        Adrian 2020-04-23
        """

        log('Populating Deconvolution for {}'.format(key))

        from .utils.cascade2p import checks, cascade
        # To run deconvolution, tensorflow, keras and ruaml.yaml must be installed
        checks.check_packages()

        model_name = (DeconvolutionModel & key).fetch1('model_name')
        sampling_rate = (DeconvolutionModel & key).fetch1('sampling_rate')
        threshold = (DeconvolutionModel & key).fetch1('threshold')
        fs = (ScanInfo & key).fetch1('fs')

        if np.abs( sampling_rate - fs) > 1:
            raise Warning( ('The model sampling rate {}Hz is too different from the '.format(sampling_rate) +
                            'recording rate of {}Hz.'.format(fs)) )

        # get dff traces only for accepted units! If changing accepted, table has to be populated again
        traces, unit_ids = (AreaSegmentation & key).get_traces(only_accepted=True, include_id=True)

        # model is saved in subdirectory models of cascade2p
        import inspect
        cascade_path = os.path.dirname(inspect.getfile(cascade) )
        model_folder = os.path.join( cascade_path, 'models')

        decon_traces = cascade.predict( model_name, traces, model_folder=model_folder,
                                        threshold=threshold, padding=0)

        # enter results into database
        self.insert1( key )  # master entry

        part_entries = list()
        for i, unit_id in enumerate(unit_ids):
            new_part = dict(**key,
                            mask_id = unit_id,
                            decon = decon_traces[i,:])
            part_entries.append( new_part )

        self.ROI.insert( part_entries )


    def get_traces(self, only_accepted = True, include_id = False):
        """ Wrapper function for AreaSegmentation.get_traces """

        return (AreaSegmentation & self).get_traces(only_accepted=only_accepted, trace_type='decon',
                                            include_id=include_id, decon_id = self.fetch1('decon_id'))


@schema
class ActivityStatistics(dj.Computed):
    definition = """ # Table to store summed, average activity and number of events
    -> Deconvolution
    ------
    """

    class ROI(dj.Part):
        definition = """ # Part table for entries grouped by session
        -> ActivityStatistics
        mask_id : int        #  Mask index (as in AreaSegmentation.ROI), per area (base 0)
        -----
        sum_spikes   : float    # Sum of deconvolved activity trace (number of spikes)
        rate_spikes  : float    # sum_spikes normalized to spikes / second
        nr_events    : int      # Number of threshold crossings
        """

    def make(self, key):
        """ Automatically populate for all accepted traces of Deconvolution.ROI
        Adrian 2021-04-15
        """
        log('Populating ActivityStatistics for {}'.format(key))
        THRES = 0.05  # hardcoded parameter

        # traces is (nr_neurons, time) array
        traces, unit_ids = (Deconvolution & key).get_traces(only_accepted=True, include_id=True)
        fps = (ScanInfo & key).fetch1('fs')
        nr_frames = traces.shape[1]

        new_part_entries = list()
        for i, unit_id in enumerate(unit_ids):
            trace = traces[i]

            # calculate the number of threshold crossings
            thres_cross = (trace[:-1] <= THRES) & (trace[1:] > THRES)
            nr_cross = np.sum(thres_cross)

            new_entry = dict( **key,
                              mask_id = unit_id,
                              sum_spikes = np.sum(trace),
                              rate_spikes = np.sum(trace) / nr_frames * fps,
                              nr_events = nr_cross  )
            new_part_entries.append( new_entry )

        # insert into database
        ActivityStatistics.insert1(key)
        ActivityStatistics.ROI.insert(new_part_entries)