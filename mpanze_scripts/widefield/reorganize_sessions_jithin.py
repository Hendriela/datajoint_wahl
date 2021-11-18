# reorganise sessions from jithin's sensory mapping to match mapping pipeline
# this will involve dropping tables so better save the already computed AffineRegistration params

## imports
import login
login.connect()
from schema.common_exp import Session
from schema.mpanze_widefield import *

## data backup: save backups of each RawImagingFile and associated entries
login.set_working_directory("F:/Jithin/")
keys_img = RawImagingFile().fetch("KEY")
import pickle
import tqdm
for key_img in tqdm.tqdm(keys_img):
    file_base = (RawImagingFile() & key_img).get_path().with_suffix("")
    row_img = (RawImagingFile() & key_img).fetch1()
    file_img = str(file_base) + "_img.pickle"
    with open(file_img, 'wb') as f:
        pickle.dump(row_img, f, protocol=pickle.HIGHEST_PROTOCOL)
    row_scan = (Scan() & key_img).fetch1()
    file_scan = str(file_base) + "_scan.pickle"
    with open(file_scan, 'wb') as f:
        pickle.dump(row_scan, f, protocol=pickle.HIGHEST_PROTOCOL)
    row_session = (Session() & key_img).fetch1()
    file_session = str(file_base) + "_session.pickle"
    with open(file_session, 'wb') as f:
        pickle.dump(row_session, f, protocol=pickle.HIGHEST_PROTOCOL)
    row_info = (ScanInfo() & key_img).fetch1()
    file_info = str(file_base) + "_info.pickle"
    with open(file_info, 'wb') as f:
        pickle.dump(row_info, f, protocol=pickle.HIGHEST_PROTOCOL)
    if len(AffineRegistration() & key_img) > 0:
        row_affine = (AffineRegistration() & key_img).fetch1()
        file_affine = str(file_base) + "_affine.pickle"
        with open(file_affine, 'wb') as f:
            pickle.dump(row_affine, f, protocol=pickle.HIGHEST_PROTOCOL)


## delete all .yaml files in folder
import pathlib
import os
from tqdm import tqdm

p = pathlib.Path("F:/Jithin")
for x in tqdm(p.rglob("*.pickle")):
    os.remove(x)

## rename imaging files to match params file
import pathlib
import os
from tqdm import tqdm

p = pathlib.Path("F:/Jithin")
for x in tqdm(p.rglob("*.tif")):
    y = "!!"
    if ("forelimb" in str(x)) or ("Forelimb" in str(x)):
        y = pathlib.Path(x.parent, "Forelimb_1.tif")
        os.rename(x, y)
    elif ("hindlimb" in str(x)) or ("Hindlimb" in str(x)):
        y = pathlib.Path(x.parent, "Hindlimb_1.tif")
        os.rename(x, y)
    else:
        print(x)