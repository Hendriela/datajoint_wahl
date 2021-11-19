"""Schema for widefield calcium and intrinsic imaging related files and information"""

import datajoint as dj
import login
import pathlib
from schema import common_mice, common_exp
from mpanze_scripts.widefield import utils, smoothing_functions
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
import cv2

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
        ["UV", 405, "405nm hemodynamics control LED through objective"],
        ["Green", 530, "Green LED from the side"]
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
    contents = [
        {"objective_name": "Navitar 50mm F0.95", "efl": 50.0, "bfl": 25.62, "f_stop": 0.95,
         "model": "Navitar DO-5095"},
        {"objective_name": "Navitar 50mm F1.4", "efl": 50.0, "bfl": 14.80, "f_stop": 1.4,
         "model": "Navitar NMV-50M1"},
        {"objective_name": "Thorlabs 100mm Tube Lens", "efl": 100.0, "bfl": 96.4, "magnification": 0.5,
         "model": "Thorlabs WFA4102"},
        {"objective_name": "Thorlabs 4x Apochromatic Objective", "efl": 60.0, "magnification": 4, "na": 0.2,
         "model": "Thorlabs TL4X-SAP"},
        {"objective_name": "Walimex 85mm F 1.4", "efl": 85.0, "f_stop": 1.4,
         "model": "Walimex Product no. 19624", "objective_notes": "objective from Philipp"}
    ]


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
    scan_id                     : smallint      # keep track of multiple scans per session
    ---
    -> WidefieldMicroscope
    -> Objective.proj(top_objective = 'objective_name')
    -> Objective.proj(bottom_objective = 'objective_name')
    nr_channels                 : tinyint       # number of channels recorded
    """

    def helper_insert1(self, new_entry):
        """
        Automatically computes the scan_id for the session
        Args:
            new_entry: dict containing all attributes, except scan_id

        Returns:
            str: key for the new entry
        """
        query = Scan() & new_entry
        if len(query) == 0:
            scan_id = 0
        else:
            scan_id = max(query.fetch("scan_id"))+1
        new_entry["scan_id"] = scan_id
        Scan().insert1(new_entry)
        return (Scan() & new_entry).fetch1("KEY")


@schema
class ScanInfo(dj.Manual):
    definition = """ # Additional scan parameters
    -> Scan
    ---
    exposure_time               : float         # exposure time in ms
    binning = 2                 : tinyint       # pixel binning used (this is already applied by the imaging camera)
    pixels_x                    : smallint      # number of pixels recorded along x axis (2nd matrix dimension) 
    pixels_y                    : smallint      # number of pixels recorded along y axis (1st matrix dimension)
    scan_notes                  : varchar(512)  # additional notes about the scan
    """


@schema
class RawImagingFile(dj.Manual):
    definition = """ # Paths to raw widefield imaging files in .tif format
    -> Scan
    -> LED
    ---
    filename_img                : varchar(512)  # name of the imaging file, relative to the session folder
    """

    def get_paths(self):
        """Construct full paths to raw imaging files"""
        path_neurophys = login.get_working_directory()  # get data directory path on local machine
        # find sessions corresponding to current files
        sessions = (self * common_exp.Session())

        # iterate over sessions
        paths = []
        for session in sessions:
            # obtain full path
            path_session = session["session_path"]
            path_file = session["filename_img"]
            paths.append(pathlib.Path(path_neurophys, path_session, path_file))
        return paths

    def get_path(self, check_existence=False):
        """
        Construct full path to a raw imaging file.
        Method only works for single-element query.
        """
        p = self.get_paths()[0]
        if len(self) == 1:
            if check_existence:
                if not p.exists():
                    raise Exception("The file was not found at %s" % str(p))
            return p
        else:
            raise Exception("This method only works for a single entry! For multiple entries use get_paths")

    def get_first_image(self):
        if len(self) != 1:
            raise Exception("method implemented only for single element in query")
        path = self.get_path()
        return tif.imread(path, key=0)

    def plot_first_image(self):
        if len(self) != 1:
            raise Exception("method implemented only for single element in query")
        plt.figure()
        plt.imshow(self.get_first_image(), "Greys_r")


@schema
class ReferenceImage(dj.Manual):
    definition = """ # Reference image from the widefield microscope, used as a template for spatial alignment
    -> common_mice.Mouse
    -> LED.proj(ref_led = 'led_colour')
    ref_date                : date          # date the image was taken (YYYY-MM-DD)
    ---
    ref_image               : longblob      # reference image (np.uint16 array with 512x512 dimensions)
    ref_mask = NULL         : longblob      # mask with size matching the reference image (np.unint8 array). A value of 255 indicates a masked entry.
    ref_notes = ""          : varchar(256)  # additional notes
    """

    def plot(self, mask=False):
        """
        Plots the reference image(s), with the option to apply the mask
        Args:
            mask: optional, if True, the reference image is masked using ref_mask
        """
        for row in self:
            plt.figure("M%i, %s, %s" % (row["mouse_id"], row["ref_led"], row["ref_date"]))
            img_array = row["ref_image"]
            if mask:
                img_array = np.ma.masked_array(img_array, mask=row["ref_mask"])
            plt.imshow(img_array, cmap="Greys_r")
            plt.colorbar()


@schema
class AffineRegistration(dj.Manual):
    definition = """ # Affine Registration matrix for a given reference image and imaging file
    -> RawImagingFile
    -> ReferenceImage
    ---
    affine_matrix           : longblob      # affine transformation matrix
    """

    def load_registered_stack(self, start, end):
        """
        Loads a stack delimited by start, end and performs the affine transform
        Returns: registered stack
        """
        if len(self) != 1:
            raise Exception("A single file must be selected!")

        p_file = (RawImagingFile & self).get_path()
        affine_matrix = self.fetch1()["affine_matrix"]
        # open file in memory mapped mode
        memmap_file = tif.memmap(str(p_file))
        # load chunk we need and close file
        chunk = np.copy(memmap_file[start:end])
        h, w = chunk.shape[1], chunk.shape[2]
        del memmap_file
        # spatial registration
        for i, frame in enumerate(chunk):
            chunk[i] = cv2.warpAffine(frame, affine_matrix, (h, w))
        return chunk


@schema
class Smoothing(dj.Lookup):
    definition = """ # Class that implements various image processing methods
    kernel_name             : varchar(128)  # name of the kernel in the smoothing file. must adhere to python function naming conventions
    kernel_id               : int           # integer for counting different kernel parameters
    ---
    kernel_params           : varchar(1024) # kernel parameters, should be a dict formatted as a .json style string
    kernel_description      : varchar(1024) # long description of what the smoothing function does
    """
    contents = [{"kernel_name": "gaussian_blur_2d", "kernel_id": 0, "kernel_params": '{"size_x": 5, "size_y": 5}',
                 "kernel_description": "applies gaussian blur in x and y axis frame by frame"}]

    def smooth_stack(self, stack):
        """
        Performs smoothing on a stack of the form (N_frames, pixels_x, pixels_y), as a numpy array
        Each kernel has a corresponding function in mpanze_scripts/widefield/smoothing_functions.py
        Args:
            stack: input stack to be smoothed. numpy array of dtype uin16 or float32, with shape (frames, x, y)
        Returns: smoothed stack, of the same shape and dtype as stack
        """
        # check that a single kernel is selected
        if len(self) != 1:
            raise Exception("A single smoothing kernel must be selected!")
        kernel = self.fetch1()
        if kernel["kernel_name"] == "gaussian_blur_2d":
            stack_blur = smoothing_functions.gaussian_blur_2d(stack, kernel["kernel_params"])
        return stack_blur

    def to_string(self):
        # utility function for generating filenames
        if len(self) != 1:
            raise Exception("A single smoothing kernel must be selected!")
        kernel = self.fetch1()
        return kernel["kernel_name"] + "_" + "{:02d}".format(kernel["kernel_id"])



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
