"""Schema for widefield imaging related files and information"""

import datajoint as dj
import login
import pathlib
from schema import common_mice, common_exp

schema = dj.schema('mpanze_widefield', locals(), create_tables=True)


@schema
class ReferenceImage(dj.Manual):
    definition = """ # Reference image from the widefield microscope, used as a template for spatial alignment
    -> common_mice.Mouse
    id_ref_img                  : tinyint       # id for keeping track of multiple reference images
    ---
    date_ref_img                : date          # date the image was taken (YYYY-MM-DD)
    reference_image             : longblob      # reference image (numpy array with 512x512 dimensions)
    reference_image_mask        : longblob      # mask with size matching the reference image (np.unint8 array)
    light_source                : varchar(20)   # light source used to take image (e.g. blue LED)
    """


@schema
class RawImagingFile(dj.Manual):
    definition = """ # Paths to raw widefield imaging files in .tif format
    -> common_exp.Session
    id_img_file                 : tinyint       # for keeping track of multiple files within a session (e.g. sensory mapping)
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
