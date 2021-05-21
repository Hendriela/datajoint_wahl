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
    definition = """ # Raw widefield imaging file
    -> common_exp.Session
    id_img_file                 : tinyint       # for keeping track of multiple files within a session (e.g. sensory mapping)
    ---
    filename_img                : varchar(256)  # name of the imaging file, relative to the session folder
    """