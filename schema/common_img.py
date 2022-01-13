#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 25/06/2021 10:21
@author: hheise

Adapted from Adrian: https://github.com/HelmchenLabSoftware/adrian_pipeline/blob/master/schema/img.py

Schemas for the CaImAn 2-photon image analysis pipeline
"""

# imports
import datajoint as dj
import login

# Only import Caiman-specific modules if code is run inside a caiman environment
try:
    from util import scanimage, motion_correction
    import caiman as cm
    from caiman.motion_correction import MotionCorrect
    from caiman.source_extraction.cnmf import params as params
    from caiman.source_extraction.cnmf import cnmf as cnmf

except ModuleNotFoundError:
    print('Skipping caiman-specific imports because code is not run in CaImAn environment.\nYou can view the tables, '
          'but CaImAn-specific functions will not work.')

from util import helper
from schema import common_exp, common_mice
import os
import numpy as np
from typing import Optional, List, Union, Tuple
import tifffile as tif
import yaml
from glob import glob
import matplotlib.pyplot as plt

schema = dj.schema('common_img', locals(), create_tables=True)


# CURRENT_VERSION = 0   # identifier for management of different versions
# VERBOSE = 5           # control level of debug messages

# Commented for now because we do not need it for now, the caiman pipeline should be set up.
# @schema
# class Version(dj.Lookup):
#     definition = """ # Version control for the imaging pipeline
#     version      : int    # version index (base 0)
#     ---
#     description  : varchar(256)      # Notes on this version
#     """
#     contents = [ [0, 'Setting up pipeline'] ]


@schema
class Microscope(dj.Lookup):
    definition = """ # Used microscope for scanning
    microscope         : varchar(128)     # Microsope short name
    ---
    microscope_details : varchar(1048)    # Additional details
    """

    contents = [
        ['Scientifica', 'Scientifica microscope in H37 (right)'],
    ]


@schema
class Laser(dj.Lookup):
    definition = """ # Laser used for scanning
    laser    : varchar(128)         # Laser short name
    ---
    laser_details : varchar(1024)   # Additional details
    """

    contents = [
        ['MaiTai', 'MaiTai laser, tunable, power depending on wavelength'],
        ['1040', 'add details...']
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
        ['CA1'],
        ['CA3'],
        ['DG'],
    ]


@schema
class BrainRegion(dj.Lookup):
    definition = """ # Possible brain regions for scanning
    brain_region    : varchar(32)       # Short name for brain region
    ----
    region_details   : varchar(128)     # More detailed description of area
    """
    contents = [
        ['HPC', '(dorsal) hippocampus'],
        ['Other', 'Other brain area, to be defined if necessary'],
        ['None', 'Not assigned because between boundaries']
    ]


@schema
class FieldOfViewSize(dj.Lookup):
    definition = """ # Size of FOV (in um) of H37R Scientifica microscope at different zooms
    zoom    : tinyint       # zoom strength (higher = more magnification)
    ----
    x       : float         # Width
    y       : float         # Height            
    """
    contents = [
        [1, 850, 780],
        [2, 425, 390],
        [3, 288, 275],
        [4, 203, 205],
        [5, 167, 165]
    ]


@schema
class CaIndicator(dj.Lookup):
    definition = """ # Calcium indicators and their properties
    ca_name    : varchar(32)    # Short name of the indicator
    ----
    decay      : float          # Mean decay time in s           
    """
    contents = [
        ['GCaMP6f', 0.4]
    ]


@schema
class Scan(dj.Manual):
    definition = """ # Basic info about the recorded scan and the equipment/hardware used
    -> common_exp.Session
    ---
    -> Microscope
    -> Laser
    -> Layer
    -> CaIndicator
    objective = '16x' : varchar(64)  # Used objective for the scan
    nr_channels = 1   : tinyint      # Number of recorded channels (1 or 2)
    network_id = 1    : tinyint      # Network ID if several networks were recorded in the same day. This would count as separate exp.Session entries (incrementing session_num)
    """

    @staticmethod
    def select_channel_if_necessary(key: dict, return_channel: int) -> Optional[int]:
        """
        Return None if the Scan only has 1 channel, otherwise the given channel is returned.
        Adrian 2021-05-21

        Args:
            key:                Primary keys of the queried Scan() entry.
            return_channel:     Index (base 0) of requested channel.

        Returns:

        """
        if (Scan & key).fetch1('nr_channels') == 1:
            return None
        else:
            return return_channel


@schema
class RawImagingFile(dj.Imported):
    definition = """ # File names and stack size of the acquired .tif raw imaging files
    -> Scan
    part        : smallint                  # Counter for part files of the same scan (base 0)
    ---
    file_name   : varchar(512)              # File name with relative path compared to session directory
    nr_frames   : int                       # Number of frames in this file
    file_size   : int                       # Approximate file size in KB (for memory demand estimation)
    """

    def make(self, key: dict) -> None:
        """
        Automatically looks up file names for .tif files of a single Scan() entry.

        Args:
            key: Primary keys of the queried Scan() entry.
        """

        print('Finding raw imaging files for entry {}'.format(key))

        # Get session path
        path = (common_exp.Session() & key).get_absolute_path()

        # Load default parameters
        default_params = login.get_default_parameters()

        # Iterate through the session directory and subdirectories and find files with the matching naming pattern
        file_list = []
        for step in os.walk(path):
            for file_pattern in default_params['imaging']['scientifica_file']:
                file_list.extend(glob(step[0] + f'\\{file_pattern}'))

        if len(file_list) == 0:
            raise ImportError("No files found in {} that fit the patterns {}!".format(path, default_params['imaging'][
                'scientifica_file']))

        # Sort list by postfix number
        file_list_sort = helper.alphanumerical_sort(file_list)

        # Insert files sequentially to the table
        for idx, file in enumerate(file_list_sort):
            # Get number of frames in the TIFF stack
            nr_frames = len(tif.TiffFile(file).pages)

            file_size = int(np.round(os.stat(file).st_size / 1024))

            # get relative file path compared to session directory
            base_directory = login.get_working_directory()
            sess_folder = (common_exp.Session() & key).fetch1('session_path')
            rel_filename = os.path.relpath(file, os.path.join(base_directory, sess_folder))

            self.insert1(dict(**key, part=idx, file_name=rel_filename, nr_frames=nr_frames, file_size=file_size))

    def get_path(self) -> str:
        """
        Returns a string with the absolute file path for a single file on given system.
        This function uses the file paths defined in login.py to find the location
        of files for the current machine on which the code is executed.
        Adrian 2019-08-12

        Returns:
            Absolute file path for a single raw .tif file
        """
        # Quick check if only one file is selected
        if len(self) != 1:
            raise Exception('Only length one allowed (not {})'.format(len(self)))

        # Return file at remote location
        base_directory = login.get_working_directory()
        folder = (common_exp.Session() & self).fetch1('session_path')
        file = self.fetch1('file_name')

        return os.path.join(base_directory, folder, file)

    def get_paths(self) -> List[str]:
        """
        Return multiple files (in case of scan split up in multiple files).

        Returns:
            List of absolute file paths for all raw .tif files of the queried scan.
        """
        keys = self.fetch(dj.key)
        path_list = []
        for key in keys:
            path_list.append((self & key).get_path())
        return path_list


@schema
class ScanInfo(dj.Computed):
    definition = """ # Scan specific information and software settings common to all planes and channels
    -> Scan
    ---
    fr             : float                          # Sampling rate in Hz
    zoom           : tinyint                        # Selected zoom setting (higher: more magnification)
    nr_lines       : smallint                       # Number of lines in the scan field (Height)
    pixel_per_line : smallint                       # Number of pixel per line (Width)
    scanning       : enum("Resonant", "Galvo")      # Scanning type (Resonant or Galvo)
    pockels        : tinyint                        # Setting on pockels cell to modulate power
    gain           : int                            # PMT gain to modulate sensitivity
    x_motor = -1.  : float                          # X Motor position of microscope (relative to zeroed position)
    y_motor = -1.  : float                          # Y Motor position of microscope (relative to zeroed position)
    z_motor = -1.  : float                          # Z Motor position of microscope (relative to zeroed position)
    nr_frames = -1 : int                            # Total number of frames in this recording
    """

    def make(self, key: dict) -> None:
        """
        Automatically populate the ScanInfo table. RawImagingFile has to be populated beforehand.
        # TODO: Save locations of scan fields in part tables
                Create part tables for channel information
        Adrian 2019-08-21

        Args:
            key: Primary keys of the current Scan() entry.
        """

        print('Populating ScanInfo for key: {}'.format(key))

        if (Scan & key).fetch1('microscope') == 'Scientifica':
            # Extract meta-information from imaging .tif file
            path = (RawImagingFile & key & 'part=0').get_path()  # Extract only from first file
            info = scanimage.get_meta_info_as_dict(path)
            info['pockels'] = info['powers'][0]  # TODO: remove hardcoding of MaiTai laser
            info['gain'] = info['gains'][0]

        else:
            raise Exception('Only Scientifica H37R supported so far.')

        new_entry = dict(**key,
                         fr=info['fs'],
                         zoom=info['zoom'],
                         nr_lines=info['nr_lines'],
                         pixel_per_line=info['pixel_per_line'],
                         scanning=info['scanning'][1:-1],  # Scanning string includes two apostrophes, remove one set
                         pockels=info['pockels'],
                         gain=info['gain'],
                         x_motor=info['motor_pos'][0],
                         y_motor=info['motor_pos'][1],
                         z_motor=info['motor_pos'][2],
                         )

        # Calculate total number of frames in this scan session
        new_entry['nr_frames'] = np.sum((RawImagingFile & key).fetch('nr_frames'))

        self.insert1(new_entry)
        # log('Finished populating ScanInfo for key: {}'.format(key))


@schema
class MotionParameter(dj.Manual):
    definition = """ # Storage of sets of CaImAn motion correction parameters plus some custom parameters
    motion_id           : smallint  # index for unique parameter set, base 0
    ----
    motion_shortname    : varchar(256)      # Short meaningful name of parameter set
    motion_description  : varchar(1024)     # Longer description of conditions for which this parameter set fits
    # Custom parameters related to preprocessing and cropping
    crop_left   = 12    : smallint  # Pixels to crop on the left to remove scanning artifacts before MC.
    crop_right  = 12    : smallint  # See crop_left. The actual movie is not cropped here, but in MemoryMappedFile(), 
                                    # where it is used to remove border artifacts.
    offset = 220        : int       # Fixed value that is added to all pixels to make mean pixel values positive.
                                    # Default value of 220 is ~95th percentile of 900 randomly checked raw tif files.
    # CaImAn motion correction parameters
    max_shift = 50      : smallint  # maximum allowed rigid shifts (in um)
    stride_mc = 150     : smallint  # stride size for non-rigid correction (in um), patch size is stride+overlap)
    overlap_mc = 30     : smallint  # Overlap between patches, in pixels
    pw_rigid = 1        : tinyint   # flag for performing rigid  or piecewise (patch-wise) rigid mc (0: rigid, 1: pw)
    max_dev_rigid = 3   : smallint  # maximum deviation allowed for patches with respect to rigid shift
    border_nan = 0      : tinyint   # flag for allowing NaN in the boundaries. If False, value of the nearest data point
    n_iter_rig = 2      : tinyint   # Number of iterations for rigid motion correction (not used for pw-rigid)
    nonneg_movie = 1    : tinyint   # flag for producing a non-negative movie
    """

    # TODO: CRUCIAL!! CHECK HOW THE CHANGED PMT SETTING THAT AFFECTS dF/F CAN BE CORRECTED TO MAKE SESSIONS COMPARABLE
    # Todo: check if n_iter is only important for rigid, or also for pw-rigid correction

    def helper_insert1(self, entry: dict) -> None:
        """
        Extended insert1() method that also creates a backup YAML file for every parameter set.

        Args:
            entry: Content of the new MotionParameter() entry.
        """

        self.insert1(entry)

        full_entry = (self & entry).fetch1()  # Query full entry in case some default attributes were not set

        # TODO: remove hard-coding of folder location
        REL_BACKUP_PATH = "Datajoint/manual_submissions"

        identifier = f"motion_{full_entry['motion_id']}"

        # save dictionary in a backup YAML file for faster re-population
        filename = os.path.join(login.get_neurophys_wahl_directory(), REL_BACKUP_PATH, identifier + '.yaml')
        with open(filename, 'w') as outfile:
            yaml.dump(full_entry, outfile, default_flow_style=False)

    def get_parameter_obj(self, scan_key: dict):
        """
        Exports parameters as a params.CNMFParams type dictionary for CaImAn.
        Args:
            scan_key: Primary keys of ScanInfo() entry that is being processed

        Returns:
            CNMFParams-type dictionary that CaImAn uses for its pipeline
        """
        frame_rate = (ScanInfo & scan_key).fetch1('fr')
        decay_time = (CaIndicator & scan_key).fetch1('decay')

        # Caiman wants border_nan = False to be 'copy'
        border_nan = 'copy' if not self.fetch1('border_nan') else True

        # Calculate X/Y resolution from FOV size and zoom setting
        zoom = {'zoom': (ScanInfo & scan_key).fetch1('zoom')}
        fov = ((FieldOfViewSize & zoom).fetch1('x'), (FieldOfViewSize & zoom).fetch1('y'))

        dxy = (fov[0] / (ScanInfo & scan_key).fetch1('pixel_per_line'),
               fov[1] / (ScanInfo & scan_key).fetch1('nr_lines'))

        # Transform distance-based patch metrics to pixels
        max_shifts = [int(a / b) for a, b in zip((self.fetch1('max_shift'), self.fetch1('max_shift')), dxy)]
        strides = tuple([int(a / b) for a, b in zip((self.fetch1('stride_mc'), self.fetch1('stride_mc')), dxy)])

        opts_dict = {'fr': frame_rate,
                     'decay_time': decay_time,
                     'dxy': dxy,
                     'max_shifts': max_shifts,
                     'strides': strides,
                     'overlaps': (self.fetch1('overlap_mc'), self.fetch1('overlap_mc')),
                     'max_deviation_rigid': self.fetch1('max_dev_rigid'),
                     'pw_rigid': bool(self.fetch1('pw_rigid')),
                     'border_nan': border_nan,
                     'niter_rig': self.fetch1('n_iter_rig'),
                     'nonneg_movie': bool(self.fetch1('nonneg_movie'))
                     }

        opts = params.CNMFParams(params_dict=opts_dict)

        return opts


@schema
class MotionCorrection(dj.Computed):
    definition = """ # Motion correction of the network scan. Default attribute values are valid for Scientifica H37R.
    -> ScanInfo
    -> MotionParameter
    ------    
    shifts               : longblob     # 2d array (xy, nr_frames) of shift
    x_std                : float        # Standard deviation of shifts in x (left/right)
    y_std                : float        # Standard deviation of shifts in y (top/bottom)
    x_max                : smallint     # Maximal shift in x
    y_max                : smallint     # Maximal shift in y
    template             : longblob     # 2d image of used template
    template_correlation : longblob     # 1d array (nr_frames) with correlations with the template
    outlier_frames       : longblob     # 1d array with detected outlier in motion correction
    line_shift           : smallint     # Detected shift between even and odd lines.    
    align_time=CURRENT_TIMESTAMP : timestamp     # Automatic timestamp of alignment
    """

    def make(self, key: dict) -> None:
        """
        Automatically populate the MotionCorrection for all networks of this scan
        TODO:   - include motion correction of the second channel with the same parameters as primary channel
                - for multiple planes, remove the black stripe in the middle
        Adrian 2019-08-21

        Args:
            key: Primary keys of the current NetworkScan() entry.
        """
        print('Populating MotionCorrection for key: {}'.format(key))

        # start the cluster (if a cluster already exists terminate it)
        if 'dview' in locals():
            cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)

        ## get the parameters for the motion correction
        motion_params = (MotionParameter & key).fetch1()  # custom, non-Caiman params for preprocessing
        # Select second channel in case there are 2 (base 0 index)
        CHANNEL_SELECT = Scan().select_channel_if_necessary(key, 1)

        # Get Caiman Param object from the parameter table
        opts_dict = (MotionParameter & key).get_parameter_obj(key)

        # perform motion correction area by area
        new_part_entries = []  # save new entries for part tables
        part_mmap_files = []

        # get path to file for this network scan and locally cache files
        paths = (RawImagingFile & key).get_paths()
        local_cache = login.get_cache_directory()

        try:
            local_paths = motion_correction.cache_files(paths, local_cache)
        except MemoryError as ex:
            # Catch memory error and stop cluster
            cm.stop_server(dview=dview)
            raise ex

        paths = None  # make sure that the files at neurophysiology are not used by accident

        corrected_files = []

        print('Calculating shift between even and odd lines...')
        # For multiple channels, take images from both (offset should be the same)
        line_shift = motion_correction.find_shift_multiple_stacks(local_paths)
        print('Detected line shift of {} pixel'.format(line_shift))

        print("Correcting file with offset {}".format(motion_params['offset']))

        for path in local_paths:
            # apply raster and offset correction and save as new file
            new_path = motion_correction.create_raster_and_offset_corrected_file(
                local_file=path, line_shift=line_shift, offset=motion_params['offset'],
                crop_left=motion_params['crop_left'], crop_right=motion_params['crop_right'],
                channel=CHANNEL_SELECT)
            corrected_files.append(new_path)

        # delete not corrected files from cache to save storage
        print('Deleting raw cached files...')
        motion_correction.delete_cache_files(local_paths)

        # perform actual motion correction
        mc = MotionCorrect(corrected_files, **opts_dict.get_group('motion'), dview=dview)

        print('Starting CaImAn motion correction...')
        # log('Used parameters: {}'.format(opts.get_group('motion')))
        mc.motion_correct(save_movie=True)
        # log('Finished CaImAn motion correction for area {}.'.format(area))

        # log('Remove temporary created files (H45: warped, Scientifica: raster+edge correction)')
        for file in corrected_files:
            os.remove(file)

        # the result of the motion correction is saved in a memory mapped file
        mmap_files = mc.mmap_file  # list of files

        # extract and calculate information about the motion correction
        shifts = np.array(mc.shifts_rig).T  # caiman output: list with x,y shift tuples, shape (2, nr_frames)
        template = mc.total_template_rig  # 2D np array, mean intensity image

        print('Calculate correlation between template and frames...')
        template_correlations = []
        for mmap_file in mmap_files:
            template_correlations.append(motion_correction.calculate_correlation_with_template(
                mmap_file, template, sigma=2))
        template_correlation = np.concatenate(template_correlations)

        # delete memory mapped files
        for file in mmap_files:
            os.remove(file)

        new_part_entries.append(
            dict(**key,
                 # motion_id=0, Todo: check if this attribute is needed or automatically filled
                 shifts=shifts,
                 x_std=np.std(shifts[0, :]),
                 y_std=np.std(shifts[1, :]),
                 x_max=int(np.max(np.abs(shifts[0, :]))),
                 y_max=int(np.max(np.abs(shifts[1, :]))),
                 template=template,
                 template_correlation=template_correlation)
        )
        # stop cluster
        cm.stop_server(dview=dview)

        # After all areas have been motion corrected, calculate overview stats

        # TODO: Implement both functions (currently just placeholders)
        outlier_frames = motion_correction.find_outliers(new_part_entries)
        avg_shifts = motion_correction.calculate_average_shift(new_part_entries, outlier_frames)

        # insert MotionCorrection main table
        new_entry = dict(**key,
                         # motion_id=0, Todo: check if this attribute is needed or automatically filled
                         shifts=shifts,
                         x_std=np.std(shifts[0, :]),
                         y_std=np.std(shifts[1, :]),
                         x_max=int(np.max(np.abs(shifts[0, :]))),
                         y_max=int(np.max(np.abs(shifts[1, :]))),
                         template=template,
                         template_correlation=template_correlation,
                         line_shift=line_shift,
                         outlier_frames=outlier_frames)
        self.insert1(new_entry)

        print('Finished populating MotionCorrection for key: {}'.format(key))


@schema
class MemoryMappedFile(dj.Imported):
    definition = """ # Table to store path of motion corrected memory mapped file (C-order) used for ROI detection.
    -> MotionCorrection
    -----
    mmap_path : varchar(256)    # path to the cached motion corrected memory mapped file
    """

    ## entries are inserted during population of the motion correction table
    def make(self, key: dict, channel: Optional[int] = None) -> None:
        """
        Creates a memory mapped file with raster and motion correction and cropping from parameters computed and
        stored in the queried MotionCorrection() entry.
        Created if demanded by other functions in the pipeline (e.g. Caiman Segmentation), but the file and table entry
        are immediately deleted afterwards to save storage space.

        Adrian 2020-07-22

        Args:
            key:        Primary keys of the queried MotionCorrection() entry.
            channel:    If value is given (0 or 1), the stack is deinterleaved before corrections. Default is None.
        """

        print('Creating memory mapped file...')

        if len(MemoryMappedFile() & key) != 0:
            raise Exception('The memory mapped file already exists!')

        # get parameter from motion correction
        line_shift = (MotionCorrection() & key).fetch1('line_shift')
        offset = (MotionParameter() & key).fetch1('offset')
        crop = (MotionParameter() & key).fetch1('crop_left')
        xy_shift = (MotionCorrection & key).fetch1('shifts')  # (2 x nr_frames)

        # save raw recordings locally in cache
        paths = (RawImagingFile & key).get_paths()
        local_cache = login.get_cache_directory()

        local_paths = motion_correction.cache_files(paths, local_cache)
        paths = None  # make sure that the files at neurophysiology are not used by accident

        # correct line shift between even and odd lines and add offset
        corrected_files = []
        for path in local_paths:
            # apply raster and offset correction and save as new file
            new_path = motion_correction.create_raster_and_offset_corrected_file(
                local_file=path, line_shift=line_shift, offset=offset,
                crop_left=0, crop_right=0, channel=channel)
            corrected_files.append(new_path)

        # Delete uncorrected files now that the corrected files are cached
        motion_correction.delete_cache_files(local_paths)

        # apply motion correction file by file (to save hard-disk storage, only 100GB available on ScienceCloud)
        # get number of frames without loading whole stack
        nr_frames_per_file = np.cumsum([0, *[len(tif.TiffFile(x).pages) for x in corrected_files]])
        scan_size = (ScanInfo & key).fetch1('pixel_per_line')

        shift_parts = []
        for i in range(len(nr_frames_per_file) - 1):
            shift_parts.append(xy_shift[:, nr_frames_per_file[i]:nr_frames_per_file[i + 1]].T)

        temp_mmap_files = list()
        for i, file in enumerate(corrected_files):
            part_file = cm.save_memmap([file], xy_shifts=shift_parts[i], base_name='tmp{:02d}_'.format(i + 1),
                                       order='C',
                                       slices=(slice(0, 100000),
                                               slice(crop, scan_size - crop),
                                               slice(crop, scan_size - crop)))
            temp_mmap_files.append(part_file)
            motion_correction.delete_cache_files([file])  # save

        # combine parts of stack to one single file
        mmap_file = cm.save_memmap(temp_mmap_files, base_name='mmap_', order='C')

        # delete temporary files
        motion_correction.delete_cache_files(temp_mmap_files)

        # make new entry in database
        # make sure no key attributes are too much or missing
        motion_key = (MotionCorrection & key).fetch1('KEY')

        new_entry = dict(**motion_key, mmap_path=mmap_file)

        self.insert1(new_entry, allow_direct_insert=True)
        # log('Finished creating memory mapped file.')

    def delete_mmap_file(self) -> None:
        """
        Delete single-queried memory-mapped file from cache and remove entry.
        Adrian 2020-07-22
        """

        mmap_file = self.fetch1('mmap_path')

        motion_correction.delete_cache_files([mmap_file])

        self.delete_quick()  # delete without user confirmation

    def export_tif(self, nr_frames: int = 100000, target_folder: Optional[str] = None, dtype: str = 'tif',
                   prefix: str = '') -> None:
        """
        Export a motion corrected memory mapped file to an ImageJ readable .tif stack or .h5.
        Adrian 2019-03-21

        Args:
            nr_frames:      Number of frames to export, counting from the beginning. Default 100000 means all.
            target_folder:  Destination folder of the exported file. If None, use session folder on Neurophys.
            dtype:          Data type to store results in, possible values: 'tif' or 'h5'.
            prefix:         Optional prefix to identify the exported file more easily.
        """

        mmap_file = self.fetch1('mmap_path')  # only one file at a time allowed
        key = self.fetch1(dj.key)

        if dtype not in ['tif', 'h5']:
            raise Exception('Only "tif" or "h5" allowed as dtype, not "{}"'.format(dtype))

        if target_folder is None:
            # use the directory of the session on Neurophys
            base_directory = login.get_working_directory()
            folder = (common_exp.Session() & self).fetch1('session_path')
            target_folder = os.path.join(base_directory, folder)

        # load memory mapped file and transform it to 16bit and C order
        corrected = cm.load(mmap_file)  # frames x height x width

        corrected_int = np.array(corrected[:nr_frames, :, :], dtype='int16')

        file = 'motionCorrected_mouse_{mouse_id}_day_{day}_session_{session_num}_frames_{frames}'.format(
            **dict(**key, frames=len(corrected)))

        corrected = None  # save memory

        toSave_cOrder = corrected_int.copy(order='C')
        corrected_int = None  # save memory

        file = str(prefix) + file + '.' + dtype
        path = os.path.join(target_folder, file)

        if dtype.lower() in ['tif', 'tiff']:
            # if this throws an error, the tifffile version might be too old
            # print('Tifffile version: {}'.format(tif.__version__) )
            # upgrade it with: !pip install --upgrade tifffile
            import tifffile as tif
            tif.imwrite(path, data=toSave_cOrder)

        elif dtype.lower() == 'h5':
            import h5py
            with h5py.File(path, 'w') as h5file:
                h5file.create_dataset('scan', data=toSave_cOrder, dtype=np.int16)

        print('Saved file at {}'.format(path))


@schema
class QualityControl(dj.Computed):
    definition = """ # Images and metrics of the motion corrected stack for quality control
    -> MotionCorrection
    ----
    avg_image:              longblob        # 2d array: Average intensity image
    cor_image:              longblob        # 2d array: Correlation with 8-neighbors image
    std_image:              longblob        # 2d array: Standard deviation of each pixel image
    min_image:              longblob        # 2d array: Minimum value of each pixel image
    max_image:              longblob        # 2d array: Maximum value of each pixel image
    percentile_999_image:   longblob        # 2d array: 99.9 percentile of each pixel image
    mean_time:              longblob        # 1d array: Average intensity over time
    """

    def make(self, key: dict) -> None:
        """
        Automatically compute quality control metrics for a motion corrected mmap file.
        Adrian 2020-07-22

        Args:
            key: Primary keys of the current MotionCorrection() entry.
        """

        # log('Populating QualityControl for key: {}.'.format(key))

        if len(MemoryMappedFile() & key) == 0:
            # In case of multiple channels, deinterleave and return channel 0 (GCaMP signal)
            channel = Scan().select_channel_if_necessary(key, 0)
            MemoryMappedFile().make(key, channel=channel)

        mmap_file = (MemoryMappedFile & key).fetch1('mmap_path')  # locally cached file

        stack = cm.load(mmap_file)

        new_entry = dict(**key,
                         avg_image=np.mean(stack, axis=0, dtype=np.float16),
                         std_image=np.std(stack, axis=0, dtype=np.float16),
                         min_image=np.min(stack, axis=0, dtype=np.float16),
                         max_image=np.max(stack, axis=0, dtype=np.float16),
                         percentile_999_image=np.percentile(stack, 99.9, axis=0, dtype=np.float16),
                         mean_time=np.mean(stack, axis=(1, 2), dtype=np.float16),
                         )

        # calculate correlation with 8 neighboring pixels in parallel
        new_entry['cor_image'] = np.array(motion_correction.parallel_all_neighbor_correlations(stack), dtype=np.float16)

        self.insert1(new_entry)

        # delete MemoryMappedFile to save storage
        try:
            # Clean up movie variables to close mmap file for deletion
            stack = None
            (MemoryMappedFile & key).delete_mmap_file()
        except PermissionError:
            print("Deleting mmap file failed, file is being used: {}".format(mmap_file))

        # log('Finished populating QualityControl for key: {}.'.format(key))

    # Commented out for now until we implement blood vessel alignments
    # def plot_avg_on_blood_vessels(self, axes=None,
    #                               with_vessels=True, with_scale=False):
    #     """ Plot 2p neuron scan on top of picture of window with blood vessels
    #     Adrian 2020-07-27 """
    #
    #     image = self.fetch1('avg_image')
    #
    #     # reuse the code to plot the 2p vessel pattern
    #     axes = (alg.VesselScan & self).plot_scaled(axes=axes, image=image, with_scale=with_scale)
    #
    #     return axes


@schema
class CaimanParameter(dj.Manual):
    definition = """ # Table which stores sets of CaImAn Parameters. Defaults fit for dense CA1 1x FOV with good SNR.
    -> common_mice.Mouse
    caiman_id:          smallint        # index for unique parameter set, base 0
    ----
    # Parameters for CNMF and component extraction
    p = 1:                  tinyint     # order of the autoregressive system (should always be 1)
    nb = 2:                 tinyint     # number of global background components
    merge_thr = 0.75:       float       # merging threshold, max correlation of components allowed before merged
    rf = 25:                smallint    # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50. -1 if no patches.
    stride_cnmf = 6 :       smallint    # amount of overlap between the patches in pixels
    k = 18:                 smallint    # number of components per patch
    g_sig = 4:              smallint    # expected half size of neurons in pixels
    method_init = 'greedy_roi' : varchar(128)   # initialization method (if analyzing dendritic data using 'sparse_nmf')
    ssub = 2:               tinyint     # spatial subsampling during initialization
    tsub = 2:               tinyint     # temporal subsampling during initialization
    # parameters for component evaluation
    snr_lowest = 5.0:       float       # rejection threshold of signal-to-noise ratio
    snr_thr = 9.0:          float       # upper threshold of signal-to-noise ratio
    rval_lowest = -1.0:     float       # rejection threshold of spatial correlation of ROI mask vs trace
    rval_thr = 0.85:        float       # upper threshold of spatial correlation of ROI mask vs trace
    cnn_lowest = 0.1:       float       # rejection threshold of CNN-based classifier
    cnn_thr = 0.9:          float       # upper threshold for CNN based classifier
    # Parameters for deltaF/F computation
    flag_auto = 1:          tinyint     # flag for using provided or computed percentile as baseline fluorescence. If 
                                        # True, Caiman estimates best percentile as the cumulative distribution function 
                                        # of a kernel density estimation.
    quantile_min = 8:       tinyint     # Quantile to use as baseline fluorescence. Only used if flag_auto is False.
    frame_window = 2000:    int         # Sliding window size of fluorescence normalization
    # Custom parameters that are not directly part of Caimans CNMFParams object
    weight_thresh = 0.005:  float       # Threshold of giving an ROI ownership over a pixel
    """

    def helper_insert1(self, entry: dict) -> None:
        """
        Extended insert1() method that also creates a backup YAML file for every parameter set.

        Args:
            entry: Content of the new CaimanParameter() entry.
        """

        self.insert1(entry)

        full_entry = (self & entry).fetch1()  # Query full entry in case some default attributes were not set

        # TODO: remove hard-coding of folder location
        REL_BACKUP_PATH = "Datajoint/manual_submissions"

        identifier = f"caiman_{full_entry['caiman_id']}_{full_entry['username']}_M{full_entry['mouse_id']}"

        # save dictionary in a backup YAML file for faster re-population
        filename = os.path.join(login.get_neurophys_wahl_directory(), REL_BACKUP_PATH, identifier + '.yaml')
        with open(filename, 'w') as outfile:
            yaml.dump(full_entry, outfile, default_flow_style=False)

    def get_parameter_obj(self, scan_key: dict, return_dict: bool = True):
        """
        Exports parameters as a params.CNMFParams type dictionary for CaImAn.
        Args:
            scan_key: Primary keys of Scan entry that is being processed
            return_dict: Bool flag whether a a dict instead of a CNMFParams object should be returned.
                            This is necessary to integrate the Caiman parameters into an existing CNMFParams object.

        Returns:
            Dict of CaImAn parameters (if return_dict == True)
            CNMFParams-type dictionary that CaImAn uses for its pipeline (if return_dict == False)
        """
        frame_rate = (ScanInfo & scan_key).fetch1('fr')
        decay_time = (CaIndicator & scan_key).fetch1('decay')

        # Calculate X/Y resolution from FOV size and zoom setting
        zoom = {'zoom': (ScanInfo & scan_key).fetch1('zoom')}
        fov = ((FieldOfViewSize & zoom).fetch1('x'), (FieldOfViewSize & zoom).fetch1('y'))

        dxy = (fov[0] / (ScanInfo & scan_key).fetch1('pixel_per_line'),
               fov[1] / (ScanInfo & scan_key).fetch1('nr_lines'))

        opts_dict = {  # 'fnames': fnames,
            'fr': frame_rate,
            'decay_time': decay_time,
            'dxy': dxy,
            'p': self.fetch1('p'),
            'nb': self.fetch1('nb'),
            'rf': self.fetch1('rf'),
            'K': self.fetch1('k'),
            'gSig': [self.fetch1('g_sig'), self.fetch1('g_sig')],
            'stride': self.fetch1('stride_cnmf'),
            'method_init': self.fetch1('method_init'),
            'rolling_sum': True,
            'only_init': True,
            'ssub': self.fetch1('ssub'),
            'tsub': self.fetch1('tsub'),
            'merge_thr': self.fetch1('merge_thr'),
            'SNR_lowest': self.fetch1('snr_lowest'),
            'min_SNR': self.fetch1('snr_thr'),
            'rval_lowest': self.fetch1('rval_lowest'),
            'rval_thr': self.fetch1('rval_thr'),
            'use_cnn': True,
            'cnn_lowest': self.fetch1('cnn_lowest'),
            'min_cnn_thr': self.fetch1('cnn_thr')
        }

        # fill in None for -1
        if opts_dict['rf'] == -1:
            opts_dict['rf'] = None

        if return_dict:
            return opts_dict
        else:
            opts = params.CNMFParams(params_dict=opts_dict)
            return opts


@schema
class Segmentation(dj.Computed):
    definition = """ # Table to store results of Caiman segmentation into ROIs
    -> MotionCorrection
    -> CaimanParameter
    ------
    nr_masks                        : smallint      # Number of total detected masks in this FOV (includes rejected masks)
    target_dim                      : longblob      # Tuple (dim_y, dim_x) to reconstruct mask from linearized index
    s_background                    : longblob      # Spatial background component(s) weight mask (dim_y, dim_x, nb) 
    f_background                    : longblob      # Background fluorescence (nb, nr_frames)
    roi_map                         : longblob      # FOV with pixels labelled with their (primary) ROI occupant. -1 if no ROI occupies this pixel.
    traces                          : varchar(128)  # Relative path (from session folder) to "raw" trace of each ROI (estimates.C + estimates.YrA)
    residuals                       : varchar(128)  # Relative path (from session folder) to residual fluorescence of each ROI (estimates.YrA)
    time_seg = CURRENT_TIMESTAMP    : timestamp     # automatic timestamp
    """

    class ROI(dj.Part):
        definition = """ # Data from mask created by Caiman
        -> Segmentation
        mask_id  : smallint     #  Mask index (base 0)
        -----
        pixels   : longblob     # Linearized indices of non-zero values
        weights  : longblob     # Corresponding values at the index position
        com      : longblob     # Center of Mass (x/row, y/column)
        dff      : longblob     # Normalized deltaF/F fluorescence change
        perc     : float        # Percentile used for deltaF/F computation 
        snr      : float        # Signal-to-noise ratio of this ROI (evaluation criterion)
        r        : float        # Spatial correlation of fluorescence and mask (evaluation criterion)
        cnn      : float        # CNN estimation of neuron-like shape (evaluation criterion)
        """

        def get_rois(self) -> np.ndarray:
            """
            Returns the ROI mask as a dense 2d array of the shape of the imaging field
            Adrian 2019-09-05

            Returns:
                Dense 2d array of the shape of the imaging field, shape (n_rois, x, y)
            """

            from scipy import sparse

            # Query data
            weights = self.fetch('weights')
            pixels = self.fetch('pixels')
            dims = (Segmentation() & self).fetch1('target_dim')

            # make sparse matrices for all queried ROIs
            sparse_matrices = []
            for i in range(len(weights)):
                sparse_matrices.append(sparse.csc_matrix((weights[i], (pixels[i], np.zeros(len(pixels[i])))),
                                                         shape=(dims[0] * dims[1], 1)))

            # stack sparse matrices
            sparse_matrix = sparse.hstack(sparse_matrices)
            # transform to dense matrix
            dense = np.reshape(sparse_matrix.toarray(), (dims[0], dims[1], len(weights)), order='F')

            # swap axes to keep n_rois as first axis and return
            return np.moveaxis(dense, 2, 0)

        # Obsolete function, CoM is calculated during make() and stored in DB
        # def get_roi_center(self) -> np.ndarray:
        #     """
        #     Returns center of mass of a single ROI as int array of length 2.
        #     Adrian 2019-09-05
        #
        #     Returns:
        #         Center of mass coordinates as np.array(x, y) or np.array(row, col)
        #     """
        #     if len(self) != 1:
        #         raise Exception('Only length one allowed (not {})'.format(len(self)))
        #
        #     roi_mask = self.get_rois()[0]
        #
        #     # calculate center of mass of mask by center of two projections
        #     proj1 = np.sum(roi_mask, axis=0)
        #     index1 = np.arange(proj1.shape[0])
        #     center1 = np.inner(proj1, index1) / np.sum(proj1)  # weighted average index
        #
        #     proj2 = np.sum(roi_mask, axis=1)
        #     index2 = np.arange(proj2.shape[0])
        #     center2 = np.inner(proj2, index2) / np.sum(proj2)  # weighted average index
        #
        #     return np.array([np.round(center1), np.round(center2)], dtype=int)

    # make of main table Segmentation
    def make(self, key: dict, save_results: bool = False, save_overviews: bool = False) -> None:
        """
        Automatically populate the segmentation for the scan.
        Adrian 2019-08-21

        Args:
            key:            Primary keys of the current MotionCorrection() entry.
            save_results:   Flag to save Caiman results in the session's folder in an HDF5 file in addition to storing
                            data in Segmentation() and ROI().
            save_overviews: Flag to plot overview graphs during Segmentation (pre- and post-evaluation components)
                            and save them in the session folder.
        """

        print('Populating Segmentation for {}.'.format(key))

        # Create the memory mapped file if it does not exist yet

        # Look for mmap files in the temp folder that might be leftover from a previously cancelled make() call
        mmap_path = glob(login.get_cache_directory() + "\\*.mmap")
        mmap_frames = [int(mmap.split("_")[-2]) for mmap in mmap_path]
        try:
            # N_frames of the found mmap files are matched with the session's nr_frames to find correct mmap file
            mmap = mmap_path[mmap_frames.index((ScanInfo & key).fetch1('nr_frames'))]
            # If a mmap file with the correct nr of frames exists, force-insert it into the table for later deletion
            MemoryMappedFile().insert1(dict(**{i: key[i] for i in key if i != 'caiman_id'}, mmap_path=mmap),
                                       allow_direct_insert=True)
            print("\tFound existing mmap file, skipping motion correction.")
        except ValueError:
            # If no fitting mmap file has been found in the temp folder, check if an entry already exists somewhere else
            # If no entry exists, re-create the mmap file.
            if len(MemoryMappedFile() & key) == 0:
                # In case of multiple channels, deinterleave and return channel 0 (GCaMP signal)
                channel = Scan().select_channel_if_necessary(key, 0)
                MemoryMappedFile().make(key, channel)

        mmap_file = (MemoryMappedFile & key).fetch1('mmap_path')  # locally cached file
        folder = (common_exp.Session() & key).get_absolute_path()  # Session folder on the Neurophys server

        # Get Caiman parameters and include it into the parameters of the motion correction
        opts = (MotionParameter() & key).get_parameter_obj(key)
        opts_dict = (CaimanParameter() & key).get_parameter_obj(key, return_dict=True)
        opts = opts.change_params(opts_dict)

        # log('Using the following parameters: {}'.format(opts.to_dict()))
        p = opts.get('temporal', 'p')  # save for later
        cn = (QualityControl() & key).fetch1('cor_image')  # Local correlation image for FOV background

        # load memory mapped file
        Yr, dims, T = cm.load_memmap(mmap_file)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        # load frames in python format (T x X x Y)

        # start new cluster
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=True)

        # disable the most common warnings in the caiman code...
        import warnings
        from scipy.sparse import SparseEfficiencyWarning
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        print("\tStart running CNMF...")
        # First extract spatial and temporal components on patches and combine them
        # for this step deconvolution is turned off (p=0)
        opts.change_params({'p': 0})
        cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
        # print('Starting CaImAn on patches...')
        cnm = cnm.fit(images)
        # log('Done.')

        # %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
        cnm.params.set('temporal', {'p': p})
        # log('Starting CaImAn on the whole recording...')
        cnm2 = cnm.refit(images, dview=dview)

        if save_overviews:
            print("Saving plots at: {}".format(folder))
            cnm2.estimates.plot_contours(img=cn, display_numbers=False)
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches((10, 10))
            plt.savefig(os.path.join(folder, 'pre_sel_components.png'))
            plt.close()

        # evaluate components
        cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)

        cnm2.estimates.plot_contours(img=cn, idx=cnm2.estimates.idx_components, display_numbers=False)
        plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches((10, 10))

        # discard rejected components
        cnm2.estimates.select_components(use_object=True)

        if save_overviews:
            cnm2.estimates.plot_contours(img=cn, display_numbers=False)
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches((10, 10))
            plt.savefig(os.path.join(folder, 'components.png'))
            plt.close()

        # Extract DF/F values
        # Caiman's source code has to be edited to return the computed percentiles (which are always used instead of
        # quantileMin if flag_auto is not actively set to False). It is a numpy array with shape (#components) with the
        # percentile used as fluorescence baseline for each neuron.
        print("\tStart computing dF/F...")
        flag_auto = bool((CaimanParameter() & key).fetch1('flag_auto'))
        frames_window = (CaimanParameter() & key).fetch1('frame_window')
        quantileMin = (CaimanParameter() & key).fetch1('quantile_min')
        _, perc = cnm2.estimates.detrend_df_f(flag_auto=flag_auto, quantileMin=quantileMin, frames_window=frames_window)

        if not flag_auto:
            # If percentiles are not computed, they are not returned but have to be filled manually as quantileMin
            perc = np.array([quantileMin] * len(cnm2.estimates.F_dff))

        # Save_results is a custom argument that is provided during the populate() call and passed to all subsequent
        # make() calls. For populate() to accept additional kwargs, the source code under datajoint/autopopulate.py
        # had to be adapted.
        if save_results:
            # log('Saving results also to file.')
            file = 'tmp_segmentation_caiman_id_{}.hdf5'.format(key['caiman_id'])
            path = os.path.join(folder, file)
            print("Saving results at {}".format(path))
            cnm2.save(path)

        # stop cluster
        cm.stop_server(dview=dview)

        # reset warnings to normal:
        warnings.filterwarnings('default', category=FutureWarning)
        warnings.filterwarnings('default', category=SparseEfficiencyWarning)
        warnings.filterwarnings('default', category=ConvergenceWarning)

        #################################################################
        #### SAVE CAIMAN RESULTS IN EASY TO READ DATAJOINT VARIABLES ####
        #################################################################
        print("\tStart saving data in DataJoint format...")

        s_background = np.reshape(cnm2.estimates.b, cnm2.dims + (opts.get('init', 'nb'),), order='F')
        f_background = cnm2.estimates.f

        masks = cnm2.estimates.A  # (flattened_index, nr_masks)
        nr_masks = masks.shape[1]

        from scipy.ndimage.measurements import center_of_mass

        # Transform sparse mask into a dense array for CoM and neuron_map calculation
        dense_masks = np.reshape(masks.toarray(), (cnm2.dims[0], cnm2.dims[1], len(cnm2.estimates.C)), order='F')
        # Move axis of ROIs to position 0
        dense_masks = np.moveaxis(dense_masks, 2, 0)

        # Use scipy to calculate center of mass
        coms = []
        for footprint in dense_masks:
            coms.append(center_of_mass(footprint))

        # Create ROI location map
        # Each pixel belongs to the ROI with the largest weight, with a default weight threshold of 0.005
        max_vals = np.amax(dense_masks, axis=0)
        pixel_owners = np.argmax(dense_masks, axis=0)
        pixel_owners = np.where(max_vals > (CaimanParameter & key).fetch1('weight_thresh'), pixel_owners, -1)

        traces = cnm2.estimates.C + cnm2.estimates.YrA
        residual = cnm2.estimates.YrA
        dff = np.array(cnm2.estimates.F_dff, dtype=np.float32)  # save traces as float32 to save disk space

        # Inf SNR is possible, has to be replaced by a large number (100 is arbitrary, but should work
        snr = np.nan_to_num(cnm2.estimates.SNR_comp, posinf=100)
        r = cnm2.estimates.r_values
        cnn = cnm2.estimates.cnn_preds

        # Save non-normalized traces (estimates.C) and residuals (estimates.YrA) externally to save server disk space
        traces_filename = "traces.npy"
        residuals_filename = "residuals.npy"
        np.save(os.path.join(folder, traces_filename), np.array(traces, dtype=np.float32))
        np.save(os.path.join(folder, residuals_filename), np.array(residual, dtype=np.float32))

        #### insert results in master table first
        new_master_entry = dict(**key,
                                nr_masks=nr_masks,
                                target_dim=np.array(dims),
                                s_background=np.array(s_background, dtype=np.float32),
                                f_background=np.array(f_background, dtype=np.float32),
                                roi_map=pixel_owners,
                                traces=traces_filename,
                                residuals=residuals_filename)
        self.insert1(new_master_entry)

        #### insert the masks and traces in the part table
        for i in range(nr_masks):
            new_part = dict(**key,
                            mask_id=i,
                            pixels=masks[:, i].indices,
                            weights=np.array(masks[:, i].data, dtype=np.float32),
                            com=coms[i],
                            dff=dff[i, :],
                            perc=perc[i],
                            snr=snr[i],
                            r=r[i],
                            cnn=cnn[i])
            self.ROI().insert1(new_part)

        # delete MemoryMappedFile to save storage
        try:
            # Clean up movie variables to close mmap file for deletion
            Yr = None
            images = None
            (MemoryMappedFile & key).delete_mmap_file()
        except PermissionError:
            print("Deleting mmap file failed, file is being used: {}".format(mmap_file))

        print('Finished populating Segmentation for {}.'.format(key))

    ##### More functions for Segmentation
    # def print_info(self) -> None:
    #     """ Helper function to print some information about selected entries
    #     Adrian 2020-03-16
    #     """
    #     roi_table = Segmentation.ROI() & self
    #     total_units = len(roi_table)
    #     accepted_units = len(roi_table & 'accepted=1')
    #     print('Total units:', total_units)
    #     print('Accepted units: ', accepted_units)

    def get_traces(self, trace_type: str = 'dff', include_id: bool = False,
                   decon_id: bool = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], None]:
        """
        Main function to get fluorescent traces in format (nr_traces, timepoints)
        Adrian 2020-03-16

        Args:
            trace_type      : Type of the trace: 'dff', 'trace' (absolute signal values), 'decon' (Cascade spike rates)
            include_id      : Flag to return a second argument with the ROI ID's of the returned signals
            decon_id        : Additional restriction, in case trace_type 'decon' is selected and multiple deconvolution
                                models have been run. In case of only one model, function selects this one.

        Returns:
            2D numpy array (nr_traces, timepoints): Fluorescent traces
            optional: 1D numpy array (nr_traces): Only if include_id==True, contains mask ID's of the rows i
        """

        # some checks to catch errors in the input arguments
        if not trace_type in ['dff', 'trace', 'decon']:
            raise Exception('The trace_type "%s" is not allowed as input argument!' % trace_type)

        # check if multiple caiman_ids are selected with self
        caiman_ids = self.fetch('caiman_id')
        if len(set(caiman_ids)) != 1:  # set returns unique entries in list
            raise Exception('You requested traces from more the following caiman_ids: {}\n'.format(set(caiman_ids)) + \
                            'Choose only one of them with & "caiman_id = ID"!')

        selected_rois = Segmentation.ROI() & self

        if trace_type == 'dff':
            traces_list = selected_rois.fetch(trace_type, order_by='mask_id')
        elif trace_type == 'trace':
            # Todo: Implement fetch of raw traces stored externally in npy files
            raise NotImplementedError("Fetching of raw traces not implemented yet.")
        else:  # decon
            # if no decon_id is given, check if there is a single correct one, otherwise error
            if decon_id is None:
                decon_ids = (Deconvolution & self).fetch('decon_id')
                if len(decon_ids) == 1:
                    decon_id = decon_ids[0]
                else:
                    raise Exception(
                        'The following decon_ids were found: {}. Please specify using parameter decon_id.'.format(
                            decon_ids))

            table = Deconvolution.ROI() & selected_rois & {'decon_id': decon_id}
            traces_list = table.fetch('decon', order_by='mask_id')

        # some more sanity checks to catch common errors
        if len(traces_list) == 0:
            print('Warning: The query img.Segmentation().get_traces() resulted in no traces!')
            return None
        # check if all traces have the same length and can be transformed into 2D array
        if not all(len(trace) == len(traces_list[0]) for trace in traces_list):
            raise Exception(
                'Error: The traces in traces_list had different lengths (probably from different recordings)!')

        traces = np.array([trace for trace in traces_list])  # (nr_traces, timepoints) array

        if not include_id:
            return traces

        else:  # include mask_id as well
            mask_ids = selected_rois.fetch('mask_id', order_by='mask_id')
            return traces, mask_ids


@schema
class DeconvolutionModel(dj.Lookup):
    definition = """ # Table for different deconvolution models
    decon_id      : int            # index for methods, base 0
    ----
    model_name    : varchar(128)   # Name of the model
    sampling_rate : int            # Sampling rate [Hz]
    smoothing     : float          # Std of gaussian to smooth ground truth spike rate [in sec]
    causal        : int            # 0: symmetric smoothing, 1: causal kernel
    nr_datasets   : int            # Number of datasets used for training the model
    threshold     : int            # 0: threshold at zero, 1: threshold at height of one spike
    """
    contents = [
        [0, 'Global_EXC_30Hz_smoothing50ms_causalkernel', 30, 0.05, 1, 18, 0],
        [1, 'Global_EXC_30Hz_smoothing50ms', 30, 0.05, 0, 18, 0],
        [2, 'Global_EXC_30Hz_smoothing100ms_causalkernel', 30, 0.2, 1, 18, 0],
        [3, 'Global_EXC_30Hz_smoothing100ms', 30, 0.2, 0, 18, 0]
    ]


@schema
class Deconvolution(dj.Computed):
    definition = """ # Table to store deconvolved traces (only for accepted units)
    -> Segmentation
    -> DeconvolutionModel
    ------
    time_decon = CURRENT_TIMESTAMP : timestamp   # automatic timestamp
    """

    class ROI(dj.Part):
        definition = """ # Data from mask created by Caiman
        -> Deconvolution
        mask_id : int        #  Mask index (as in Segmentation.ROI, base 0)
        -----
        decon   : longblob   # 1d array with deconvolved activity
        noise   : float      # noise level as computed by Cascade based on frame-rate-normalized dF/F standard deviation
        """

    def make(self, key: dict) -> None:
        """
        Automatically populate deconvolution for all accepted traces of Segmentation.ROI()
        Adrian 2020-04-23

        Args:
            key: Primary keys of the current Segmentation() entry.
        """

        print('Populating Deconvolution for {}'.format(key))

        from util.cascade2p import checks, cascade
        # To run deconvolution, tensorflow, keras and ruaml.yaml must be installed
        checks.check_packages()

        model_name = (DeconvolutionModel & key).fetch1('model_name')
        sampling_rate = (DeconvolutionModel & key).fetch1('sampling_rate')
        threshold = (DeconvolutionModel & key).fetch1('threshold')
        fs = (ScanInfo & key).fetch1('fr')

        print('Using deconvolution model {}'.format(model_name))

        if np.abs(sampling_rate - fs) > 1:
            raise Warning(('The model sampling rate {}Hz is too different from the '.format(sampling_rate) +
                           'recording rate of {}Hz.'.format(fs)))

        traces, unit_ids = (Segmentation & key).get_traces(include_id=True)

        # model is saved in subdirectory models of cascade2p
        import inspect
        cascade_path = os.path.dirname(inspect.getfile(cascade))
        model_folder = os.path.join(cascade_path, 'Pretrained_models')

        # Transform traces back to float64 to not confuse CASCADE
        decon_traces, trace_noise_levels = cascade.predict(model_name, np.array(traces, dtype=np.float64),
                                                           model_folder=model_folder, threshold=threshold, padding=0)
        # Store traces in float32 to save disk space
        decon_traces = np.array(decon_traces, dtype=np.float32)

        # enter results into database
        self.insert1(key)  # master entry

        part_entries = list()
        for i, unit_id in enumerate(unit_ids):
            new_part = dict(**key,
                            mask_id=unit_id,
                            decon=decon_traces[i, :],
                            noise=trace_noise_levels[i])
            part_entries.append(new_part)

        self.ROI.insert(part_entries)

    def get_traces(self, include_id: bool = False) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], None]:
        """ Wrapper function for Segmentation.get_traces(). See that function for documentation """

        return (Segmentation & self).get_traces(trace_type='decon', include_id=include_id,
                                                decon_id=self.fetch1('decon_id'))


@schema
class ActivityStatistics(dj.Computed):
    definition = """ # Table to store summed, average activity and number of events
    -> Deconvolution
    ------
    """

    class ROI(dj.Part):
        definition = """ # Part table for entries grouped by session
        -> ActivityStatistics
        mask_id : int        #  Mask index (as in Segmentation.ROI, base 0)
        -----
        sum_spikes   : float    # Sum of deconvolved activity trace (number of spikes)
        rate_spikes  : float    # sum_spikes normalized to spikes / second
        nr_events    : int      # Number of threshold crossings
        """

    def make(self, key: dict) -> None:
        """
        Automatically populate for all accepted traces of Deconvolution.ROI
        Adrian 2021-04-15

        Args:
            key: Primary keys of the current Deconvolution() entry.
        """
        # log('Populating ActivityStatistics for {}'.format(key))
        THRESH = 0.05  # Threshold for deconvolved events, hardcoded parameter

        # traces is (nr_neurons, time) array
        traces, unit_ids = (Deconvolution & key).get_traces(include_id=True)
        fps = (ScanInfo & key).fetch1('fr')
        nr_frames = traces.shape[1]

        new_part_entries = list()
        for i, unit_id in enumerate(unit_ids):
            trace = traces[i]

            # calculate the number of threshold crossings
            thres_cross = (trace[:-1] <= THRESH) & (trace[1:] > THRESH)
            nr_cross = np.sum(thres_cross)

            new_entry = dict(**key,
                             mask_id=unit_id,
                             sum_spikes=np.sum(trace),
                             rate_spikes=np.sum(trace) / nr_frames * fps,
                             nr_events=nr_cross)
            new_part_entries.append(new_entry)

        # insert into database
        ActivityStatistics.insert1(key)
        ActivityStatistics.ROI.insert(new_part_entries)
