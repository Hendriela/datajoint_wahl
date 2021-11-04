import tifffile as tif
import dcimg
import numpy as np
from tqdm import tqdm, trange
import pathlib
import json
from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta
from tifffile import TiffWriter



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
