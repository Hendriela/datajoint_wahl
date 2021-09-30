import tifffile as tif
import dcimg
import numpy as np
import tqdm
import pathlib
import json
from datetime import datetime


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

def dcimg_to_tiff_chunks(path, chunksize = 8000):
    """
    Converts Hamamatsu .dcimg file to .tif file in chunks of ~4GB.
    Much slower than normal dcimg_to_tiff, but useful if the entire file cannot fit into RAM.
    Args:
        path: pathlib Path object containing path of .dcimg file
        chunksize: stack size of an individual chunk. For uint16 data type, the default value yields ~4GB chunks
    Returns: string containing path of new tif file
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

    #  open memory-mapped dcimg file
    wide = dcimg.DCIMGFile(path)
    n_frames = int(wide.shape[0])
    n_chunks = int(np.ceil(n_frames/chunksize))
    # create memory_mapped tif file
    print("reserving space on disk...")
    tif_memmap = tif.memmap(str(save_path), shape=tuple([int(j) for j in wide.shape]), dtype="uint16", bigtiff=True)

    # loop over chunks
    for i in range(n_chunks):
        lim_1, lim_2 = i*chunksize, (i+1)*chunksize
        if lim_2 > n_frames:
            lim_2 = n_frames
        # load chunk and convert to C order
        print("loading chunk %i/%i..." % (i+1, n_chunks), end = " ")
        chunk = np.array(np.array(wide[lim_1:lim_2, :, :]).copy(order='C'), dtype=np.uint16)
        # write to memory and clear chunk
        print("writing to memory...", end = " ")
        tif_memmap[lim_1:lim_2] = chunk
        tif_memmap.flush()
        del chunk
        print("done!")

    del tif_memmap
    print("file saved at %s" % str(save_path))
    return str(save_path)