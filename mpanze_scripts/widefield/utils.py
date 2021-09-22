import tifffile as tif
import dcimg
import numpy as np
import tqdm
import pathlib


def dcimg_to_tif(dcimg_path):
    """
    Converts dcimg file to tif file and saves it in same directory.
    Processes the files in blocks of roughly 4Gb size to prevent RAM from running out
    :param dcimg_path: string containing full path to dcimg file
    :return: string containing path to the new tif file
    """
    # make filepath for tif file
    p = pathlib.Path(dcimg_path)
    p_tif = str(pathlib.Path(p.parent, str(p.stem) + ".tif"))

    # get dcimg file
    with dcimg.DCIMGFile(dcimg_path) as f_dcimg:
        # get shape
        n_frames = int(f_dcimg.shape[0])
        x_pixels = int(f_dcimg.shape[1])
        y_pixels = int(f_dcimg.shape[2])
        # allocate data for tif file
        print("Writing data to %s ..." % p_tif, flush=True)
        f_tif = tif.memmap(p_tif, shape=(n_frames, x_pixels, y_pixels), dtype=np.uint16)
        for i in tqdm.tqdm(range(0, n_frames, 1000)):
            # compute block edges
            start = i
            stop = i+1000
            if stop > n_frames:
                stop = n_frames
            # copy block from dcimg file, switch to c order
            data_c_order = np.copy(f_dcimg[start:stop], order='C', dtype=np.uint16)
            # write to tif file and flush
            f_tif[start:stop] = data_c_order
            f_tif.flush()

    return p_tif
