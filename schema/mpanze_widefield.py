"""Schema for widefield calcium and intrinsic imaging related files and information"""

import datajoint as dj
import login
import pathlib
from schema import common_mice, common_exp

schema = dj.schema('mpanze_widefield', locals(), create_tables=True)


@schema
class LED(dj.Lookup):
    definition = """ # Illumination source used for imaging
    led_colour                  : varchar(20)   # short description of LED
    ---
    led_wavelength              : int           # central wavelength of the LED, in nm
    led_description             : varchar(256)  # longer description of LED    
    """
    contents = [
        ["Blue", 470, "470nm GCaMP stimulation LED through objective"],
        ["UV", 405, "405nm hemodynamics control LED through objective"]
    ]


@schema
class AcquisitionMethod(dj.Lookup):
    definition = """ # Acquisition methods (e.g. alternating BLUE/UV)
    method_name         : varchar(20)   # short name of imaging method
    ---
    method_description  : varchar(256)  # longer description of imaging method
    """
    contents = [
        ["Single Wavelength", "Imaging at a fixed single wavelength"],
        ["Blue/UV", "Alternate each frame between Blue and UV sources"]
    ]


@schema
class Objective(dj.Lookup):
    definition = """ # Objectives for widefield imaging
    objective_name              : varchar(100)  # short name identifying the objective
    ---
    efl                         : float         # effective focal length of the objective in mm
    bfl = NULL                  : float         # back focal length of the objective in mm, if available
    f_stop = NULL               : float         # minimum F stop, if available
    na = NULL                   : float         # objective NA (numerical aperture), if available
    magnification = NULL        : float         # magnification of objective if applicable
    model = NULL                : varchar(256)  # manufacturer and model number
    objective_notes = ""        : varchar(256)  # any relevant notes about the objective
    """
    # contents = [
    #     {"objective_name": "Navitar 50mm F0.95", "efl": 50.0, "bfl": 25.62, "f_stop": 0.95,
    #      "model": "Navitar DO-5095"},
    #     {"objective_name": "Navitar 50mm F1.4", "efl": 50.0, "bfl": 14.80, "f_stop": 1.4,
    #      "model": "Navitar NMV-50M1"}
    # ]


@schema
class WidefieldMicroscope(dj.Lookup):
    definition = """ # Basic info about the widefield microscope
    microscope_name             : varchar(256)  # name of the microscope
    ---
    microscope_details          : varchar(1048) # additional details
    """
    contents = [
        ['J92 widefield', 'widefield microscope in the large behavioural box in J92']
    ]


@schema
class Scan(dj.Manual):
    definition = """ # Basic info about the hardware configuration of a recorded scan
    -> common_exp.Session
    ---
    -> WidefieldMicroscope
    -> AcquisitionMethod
    -> Objective.proj(top_objective = 'objective_name')
    top_f_stop                  : float         # F stop used on top obj - should default to objective's minimum F
    -> Objective.proj(bottom_objective = 'objective_name')
    bottom_f_stop               : float         # F stop used on bottom obj - should default to objective's minimum F
    """


@schema
class ScanInfo(dj.Manual):
    definition = """ # Additional scan parameters
    -> Scan
    ---
    exposure_time               : float         # exposure time in ms
    binning = 2                 : tinyint       # pixel binning used (this is already applied by the imaging camera)
    pixels_x                    : smallint      # number of pixels recorded along x axis (2nd matrix dimension) 
    pixels_y                    : smallint      # number of pixels recorded along y axis (1st matrix dimension)
                                                # n. of pixels refers already to the raw image,binning is pre-applied!!! 
    """


@schema
class RawImagingFile(dj.Manual):
    definition = """ # Paths to raw widefield imaging files in .tif format
    -> Scan
    -> LED
    part = 0                    : int           # counter for parts of the same scan
    ---
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


# @schema
# class ReferenceImage(dj.Manual):
#     definition = """ # Reference image from the widefield microscope, used as a template for spatial alignment
#     -> common_mice.Mouse
#     -> LED
#     ref_date                : date          # date the image was taken (YYYY-MM-DD)
#     ---
#     ref_image               : longblob      # reference image (np.uint16 array with 512x512 dimensions)
#     ref_mask                : longblob      # mask with size matching the reference image (np.unint8 array)
#     ref_notes               : varchar(256)  # additional notes
#     """

# @schema
# class SpatialAlignmentParameters(dj.Manual):
#     definition = """ # Contains transformation matrix for spatially registering different imaging sessions
#     -> RawImagingFile
#     ---
#     TransformationMatrix        : longblob      # forward transformation matrix for aligning to reference image
#     """
#
# @schema
# class SynchronisationMethod(dj.Lookup):
#     definition = """ # Information about the synchronisation methods
#     sync_method                 : varchar(256) # synchronisation method (short description)
#     ---
#     sync_description            : varchar(1024)# longer description of snyc method
#     """
#     contents = [
#         ["Exposure", "Sync file contains signal obtained from exposure times of the widefield camera."]
#     ]
#
#
# @schema
# class RawSynchronisationFile(dj.Manual):
#     definition = """ # Paths to raw synchronisation files. 1-1 relationship to Synchronisation
#     -> RawImagingFile
#     ---
#     filename_sync               : varchar(256) # name of synchronisation file, relative to session folder
#     -> SynchronisationMethod
#     """
#
#     def get_paths(self):
#         """Return list of paths to raw synchronisation files"""
#         path_neurophys = login.get_neurophys_data_directory()  # get data directory path on local machine
#         # find sessions corresponding to current files
#         sessions = (self * common_exp.Session())
#
#         # iterate over sessions
#         paths = []
#         for session in sessions:
#             # obtain full path
#             path_session = session["path"]
#             path_file = session["filename_sync"]
#             paths.append(pathlib.Path(path_neurophys, path_session, path_file))
#         return paths
