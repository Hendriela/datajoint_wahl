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