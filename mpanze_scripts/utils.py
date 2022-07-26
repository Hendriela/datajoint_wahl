import tifffile as tif
try:
    import dcimg
except ModuleNotFoundError:
    print('Import mpanze_scripts.utils with read-only access.')
import numpy as np
from tqdm import tqdm, trange
import pathlib
import json
from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta
from tifffile import TiffWriter

import login
login.connect()


def mapping_date_from_dict(data_json):
    """
    Coverts the datetime timestamp from a mapping session to YYYY-MM-DD format for datajoint
    Args:
        data_json: dict containing mapping parameters
    Returns:
        datetime string, in the format YYYY-MM-DD
    """
    datetime_json = data_json["Date_Time"]
    datetime_object = datetime.strptime(datetime_json, '%Y%m%d_%H%M')
    return datetime_object.strftime("%Y-%m-%d")

def mapping_datetime_from_dict(data_json):
    """
    Coverts the datetime timestamp into a datetime_object
    Args:
        data_json: dict containing mapping parameters
    Returns:
        datetime string, in the format YYYY-MM-DD
    """
    datetime_json = data_json["Date_Time"]
    datetime_object = datetime.strptime(datetime_json, '%Y%m%d_%H%M')
    return datetime_object


def mapping_date_from_json(path_json):
    """
    Coverts the datetime timestamp from a mapping session to YYYY-MM-DD format for datajoint
    Args:
        path_json: path to the .json file containing the mapping session parameters
    Returns:
        datetime string, in the format YYYY-MM-DD
    """
    with open(path_json) as f:
        data_json = json.load(f)
    return mapping_date_from_dict(data_json)


def dcimg_to_tiff(path):
    """
    Converts Hamamatsu .dcimg file to .tif
    :param path: string containing path of .dcimg file
    :return: string containing path of new tif file
    adapted from code by Adrian Hoffmann
    """
    # check whether the file already exists or if the path given is already a tif
    if type("path") == str:
        path = pathlib.Path(path)
    if (path.suffix == ".tif") or (path.suffix == ".tiff"):
        print("%s is already a tif file!" % str(path))
        return str(path)
    save_path = path.with_suffix(".tif")
    if save_path.exists():
        print("%s already exists!" % str(save_path))
        return str(save_path)

    wide = dcimg.DCIMGFile(path)
    # save as .tif file
    toSave_cOrder = np.array(np.array(wide[:, :, :]).copy(order='C'), dtype=np.uint16)
    tif.imwrite(save_path, data=toSave_cOrder)
    print("Saved:", save_path)
    del wide
    del toSave_cOrder

    return str(save_path)


def dcimg_to_tiff_chunks(path, chunk_size_GB = 4):
    """
    Converts Hamamatsu .dcimg file to .tif file in chunks
    For now only supports unsigned 16-bit integer (uint16) data type.
    Supports arbitrary resolutions (tested on 512x512)
    Args:
        path: pathlib Path object containing path of .dcimg file
        chunk_size_GB: approximate chunk size in GB. defaults to 4 gigabytes
    Returns: string containing path of new tif file
    """
    # check whether the file already exists or if the path given is already a tif
    if type("path") == str:
        path = pathlib.Path(path)
    if path.suffix == ".tif":
        print("%s is already a tif file!" % str(path))
        return str(path)
    save_path = path.with_suffix(".tif")
    if save_path.exists():
        print("%s already exists!" % str(save_path))
        return str(save_path)

    #  open memory-mapped dcimg file
    wide = dcimg.DCIMGFile(path)
    n_frames = int(wide.shape[0])
    frame_size_bytes = 2*int(wide.shape[1])*int(wide.shape[2])
    chunksize = int(chunk_size_GB*1e9/frame_size_bytes)

    # compute number of chunks
    n_chunks = int(np.ceil(n_frames/chunksize))
    # create tif file writer
    tif_file_out = TiffWriter(str(save_path), bigtiff=True)

    print("Converting %s..." % str(path), flush=True)
    # loop over chunks
    for i in range(n_chunks):
        lim_1, lim_2 = i*chunksize, (i+1)*chunksize
        if lim_2 > n_frames:
            lim_2 = n_frames
        # load chunk and convert to C order
        print("loading chunk %i/%i..." % (i+1, n_chunks), end=' ', flush=True)
        chunk = np.array(np.array(wide[lim_1:lim_2, :, :]).copy(order='C'), dtype=np.uint16)
        print("done!", flush=True)
        # write to memory and clear chunk
        for frame in tqdm(chunk, desc="Writing chunk %i/%i" % (i+1, n_chunks)):
            tif_file_out.write(frame, contiguous=True)

    tif_file_out.close()
    print("file saved at %s" % str(save_path))
    return str(save_path)


def frame_timestamps_from_txt_file(path_sync):
    data_sync = np.loadtxt(path_sync, skiprows=1, delimiter='\t')
    t_sample = data_sync[:, 0]
    widefield = data_sync[:, 1]

    # convert trigger trace to nice square wave
    widefield_thresh = (widefield > 0).astype(np.float64)
    diff = np.diff(widefield_thresh)

    # identify frame timestamps, by averaging timestamps when camera has active exposure
    frames = []
    sequence = []
    for i in range(len(t_sample[1:])):
        if diff[i] == -1:
            sequence = np.array(sequence)
            frames.append(np.mean(sequence[sequence != 0]))
            sequence = []
        sequence.append(widefield_thresh[1:][i] * t_sample[1:][i])

    # discard first 2 "frames", which are due to camera starting up, not actual data acquisition
    frames_widefield = np.array(frames[2:])
    diff = np.diff(frames_widefield)
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)
    fps = 1/diff_mean

    return frames_widefield, diff_mean, diff_std, fps

def filename_from_session(key):
    # takes a primary key (dependent on common_exp.Session) and gives the filename stem
    filename = "M{:03d}".format(key["mouse_id"])
    filename += "_" + key["day"].strftime("%Y-%m-%d")
    filename += "_" + str(key["session_num"])
    return filename

def session_from_filename(filename):
    # parse filename to obtain session details
    if not isinstance(filename, pathlib.Path):
        f = pathlib.Path(filename)
    else:
        f = filename

    stem = str(f.stem)
    parts = stem.split("_")
    assert len(parts) == 4
    mouse_id = int(parts[0][1:])
    date_str = parts[1]
    session_num = int(parts[2])
    suffix = f.suffix
    file_desc = parts[3]
    return mouse_id, date_str, session_num, suffix, file_desc

from schema.common_exp import Session
def get_paths(table, attribute):
    # build relative paths to experimental session
    path_neurophys = login.get_working_directory()

    # find sessions corresponding to current files
    sessions = (table * Session())

    # iterate over sessions
    paths = []
    for session in sessions:
        path_session = session["session_path"]
        path_file = session[attribute]
        paths.append(pathlib.Path(path_neurophys, path_session, path_file))
    return paths


def get_path(table, attribute, check_existence = False):
    if len(table) == 1:
        p = get_paths(table, attribute)[0]
        if check_existence:
            if not p.exists():
                raise Exception("The file was not found at %s" % str(p))
        return p
    else:
        raise Exception("This method only works for a single entry! For multiple entries use get_paths")
