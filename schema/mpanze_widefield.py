"""Schema for widefield imaging related files and information"""

import datajoint as dj
import login
import pathlib
from schema import common_mice, common_exp
import cv2
import numpy as np
import tifffile as tif

schema = dj.schema('mpanze_widefield', locals(), create_tables=True)


@schema
class LightSource(dj.Lookup):
    definition = """ # Illumination source used for imaging
    source_name           : varchar(20)   # short name of source
    ---
    source_description    : varchar(256)  # longer description of source    
    """
    contents = [
        ["Blue", "470nm GCaMP stimulation LED through objective"],
        ["UV", "405nm hemodynamics control LED through objective"],
        ["Green", "Green LED illumination from side"],
        ["Yellow", "Yellow LED illumination from side"],
        ["Red", "Red LED illumination from side"]
    ]


@schema
class ReferenceImage(dj.Manual):
    definition = """ # Reference image from the widefield microscope, used as a template for spatial alignment
    -> common_mice.Mouse
    -> LightSource
    ref_date                : date          # date the image was taken (YYYY-MM-DD)
    ---
    ref_image               : longblob      # reference image (np.uint16 array with 512x512 dimensions)
    ref_mask                : longblob      # mask with size matching the reference image (np.unint8 array)
    ref_notes               : varchar(256)  # additional notes
    """


@schema
class ImagingMethod(dj.Lookup):
    definition = """ # Lookup table for differentiating between different acquisition methods (e.g. alternating BLUE/UV)
    method_name         : varchar(20)   # short name of imaging method
    ---
    method_description  : varchar(256)  # longer description of imaging method
    """
    contents = [
        ["Single Wavelength", "Imaging at a fixed single wavelength, no alternating"],
        ["Blue/UV", "Alternate each frame between Blue and UV sources, starting from Blue"]
    ]


@schema
class RawImagingFile(dj.Manual):
    definition = """ # Paths to raw widefield imaging files in .tif format
    -> common_exp.Session
    ---
    -> ImagingMethod
    filename_img                : varchar(256)  # name of the imaging file, relative to the session folder
    """

    def get_paths(self):
        """Construct full paths to raw imaging files"""
        path_neurophys = login.get_neurophys_data_directory()  # get data directory path on local machine
        # find sessions corresponding to current files
        sessions = (self * common_exp.Session())

        # iterate over sessions
        paths = []
        for session in sessions:
            # obtain full path
            path_session = session["path"]
            path_file = session["filename_img"]
            paths.append(pathlib.Path(path_neurophys, path_session, path_file))
        return paths


@schema
class SpatialAlignmentParameters(dj.Manual):
    definition = """ # Contains transformation matrix for spatially registering different imaging sessions
    -> RawImagingFile
    ---
    TransformationMatrix        : longblob      # forward transformation matrix for aligning to reference image
    """


@schema
class RawChannelFile(dj.Computed):
    definition = """ # Contains spatially registered Raw Imaging Files, split by Channel
    --> RawImagingFile
    --> LightSource
    ---
    filename_channel            : varchar(256)  # name of the raw channel file, relative to the session folder
    """

    def get_paths(self):
        """Construct full paths to raw channel files"""
        path_neurophys = login.get_neurophys_data_directory()  # get data directory path on local machine
        # find sessions corresponding to current files
        sessions = (self * common_exp.Session())

        # iterate over sessions
        paths = []
        for session in sessions:
            # obtain full path
            path_session = session["path"]
            path_file = session["filename_channel"]
            paths.append(pathlib.Path(path_neurophys, path_session, path_file))
        return paths


@schema
class SynchronisationMethod(dj.Lookup):
    definition = """ # Information about the synchronisation methods
    sync_method                 : varchar(256) # synchronisation method (short description)
    ---
    sync_description            : varchar(1024)# longer description of snyc method
    """
    contents = [
        ["Exposure", "Sync file contains signal obtained from exposure times of the widefield camera."]
    ]


@schema
class RawSynchronisationFile(dj.Manual):
    definition = """ # Paths to raw synchronisation files. 1-1 relationship to Synchronisation
    -> RawImagingFile
    ---
    filename_sync               : varchar(256) # name of synchronisation file, relative to session folder
    -> SynchronisationMethod
    """

    def get_paths(self):
        """Return list of paths to raw synchronisation files"""
        path_neurophys = login.get_neurophys_data_directory()  # get data directory path on local machine
        # find sessions corresponding to current files
        sessions = (self * common_exp.Session())

        # iterate over sessions
        paths = []
        for session in sessions:
            # obtain full path
            path_session = session["path"]
            path_file = session["filename_sync"]
            paths.append(pathlib.Path(path_neurophys, path_session, path_file))
        return paths
