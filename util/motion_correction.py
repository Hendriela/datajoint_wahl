#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper functions for the motion correction

Adapted from Adrian: https://github.com/HelmchenLabSoftware/adrian_pipeline/blob/master/schema/utils/motion_correction.py
"""

import os
import numpy as np
from typing import Union, List, Optional

import caiman as cm  # run in CaImAn environment
import scipy.ndimage.filters as filters
from skimage import io
import tifffile as tif
import shutil  # see https://docs.python.org/3/library/shutil.html#shutil.copy
import matplotlib.pyplot as plt


def calculate_correlation_with_template(mmap_file: str, template: np.ndarray, sigma: int = 2) -> np.ndarray:
    """
    Calculate correlation with template (smoothed) from a memory-mapped movie.

    Adrian 2019-08-21

    Args:
        mmap_file:  Full file path of the mmap file
        template:   2D template array of the FOV
        sigma:      Standard deviation of the Gaussian filter used to smooth the template

    Returns:
        2D array with the correlation values between movie and template.
    """

    # define memory mapped file (not loaded in RAM)
    aligned_scan = cm.load(mmap_file)  # format time,y,x

    # smooth template to get rid of high-frequency noise
    smooth_template = filters.gaussian_filter(template, sigma=sigma)
    raveled_template = smooth_template.ravel()

    nr_frames = aligned_scan.shape[0]
    template_correlation = np.zeros(nr_frames)

    for frame in range(nr_frames):
        smooth_frame = filters.gaussian_filter(aligned_scan[frame, :, :], sigma=sigma)
        template_correlation[frame] = np.corrcoef(smooth_frame.ravel(), raveled_template)[0, 1]

    return template_correlation


def calculate_local_correlations(mmap_file: str) -> np.ndarray:
    """
    Calculate correlation with neighboring 8 pixel of a memory-mapped movie. If move is longer than 40,000 frames,
    calculate correlation only from every other frame to get around memory error problem.
    Todo: Try out different solutions, like sequential correlation from single-trial movies and averaging.
    Adrian 2019-08-21

    Args:
        mmap_file: Full file path of the mmap file

    Returns:
        2D array with local correlation template
    """

    # define memory mapped file (not loaded in RAM)
    aligned_scan = cm.load(mmap_file)  # format time,y,x

    # throw every other frame away if the movie is too big to be loaded in memory
    if len(aligned_scan) > 40000:
        aligned_scan = aligned_scan[::2]

    # calculate correlations with 8 neighboring pixel
    correlation_map = cm.local_correlations(aligned_scan, swap_dim=False)

    return correlation_map


def find_outliers(new_part_entries: dict) -> np.ndarray:
    """
    Detect outliers in the motion correction of several areas
    TODO: implement function

    Args:
        new_part_entries: Motion correction data of several areas

    Returns:
        Location of outliers (?)
    """
    return np.zeros(new_part_entries[0]['template_correlation'].shape) == 1  # all False


def calculate_average_shift(new_part_entries: dict, outlier_frames: np.ndarray) -> np.ndarray:
    """
    Calculate average shift based on shifts of all areas.
    TODO: implement function

    Args:
        new_part_entries: Motion correction data of several areas
        outlier_frames:   Frames to exclude from calculation

    Returns:
        2D array with average pixel shifts
    """
    return new_part_entries[0]['shifts']


# =============================================================================
# Calculation of shifts between even and odd lines
# =============================================================================


def shifted_corr(even_lines: np.ndarray, odd_lines: np.ndarray, lag: int=0) -> float:
    """
    Calculate correlation between two 1D arrays with a specified lag. Used to calculate correlation of even vs odd lines
    of a flattened 2D array.
    Adrian 2020-03-10

    Args:
        even_lines: First image, flattened (1D)
        odd_lines:  Second image, flattened (1D)
        lag:        Value by which to shift both flattened images.

    Returns:
        Correlation between both flattened images
    """

    if lag > 0:
        return np.corrcoef(even_lines[lag:], odd_lines[:-lag])[0, 1]
    elif lag < 0:
        return np.corrcoef(even_lines[:lag], odd_lines[-lag:])[0, 1]
    else:
        return np.corrcoef(even_lines, odd_lines)[0, 1]


def find_shift_image(image: np.ndarray, nr_lags: int = 10, debug: bool = False, return_all: bool = False) -> Union[float, tuple]:
    """
    Get shift between even and odd lines of an image (2d array).
    Adrian 2020-03-10

    Args:
        image:          Single 2D image
        nr_lags:        Range of lag shifts between lines that should be calculated
        debug:          Flag whether correlation vs lag should be plotted. Default False.
        return_all:     Flag whether all or just the lag with the optimal correlation should be returned. Default False.

    Returns:
        Optimal lag if return_all=False, else tuple of arrays with all lags and their correlation.
    """
    lags = np.arange(-nr_lags, nr_lags + 1, 1)
    corr = np.zeros(lags.shape)

    for i, lag in enumerate(lags):
        # pass even and odd lines of the image to the function
        corr[i] = shifted_corr(image[::2, :].flatten(), image[1::2, :].flatten(), lag=lag)

    if debug:
        plt.figure()
        plt.plot(lags, corr)
        plt.title('Maximum at {}'.format(lags[np.argmax(corr)]))

    if not return_all:
        return lags[np.argmax(corr)]  # return only optimal lag
    else:
        # return lags and correlation values
        return lags, corr


def find_shift_stack(stack: np.ndarray, nr_lags: int = 10, nr_samples: int = 100, debug: bool = False,
                     return_all: bool = False) -> Union[float, tuple]:
    """
    Find optimal shift between even and odd lines in stack (nr_frames,x,y)
    Takes nr_samples images from the stack and calculates the lag for values from -nr_lags
    to nr_lags, averages them and gives back the optimal lag between even and odd lines
    Adrian 2020-03-10

    Args:
        stack:          3D Image stack (nr_frames, x, y)
        nr_lags:        Range of lag shifts between lines that should be calculated
        nr_samples:     Number of randomly chosen frames from the stack that should be used for correlation
        debug:          Flag whether correlation vs lag should be plotted. Default False.
        return_all:     Flag whether all or just the lag with the optimal correlation should be returned. Default False.

    Returns:
        Optimal lag if return_all=False, else tuple of arrays with all lags and their average correlation.
    """
    nr_frames = stack.shape[0]

    np.random.seed(123532)
    random_frames = np.random.choice(nr_frames, np.min([nr_samples, nr_frames]), replace=False)
    corrs = list()

    for frame in random_frames:
        lags, corr = find_shift_image(stack[frame, :, :], return_all=True)
        corrs.append(corr)

    avg_corr = np.mean(np.array(corrs), axis=0)  # array from corrs has shape (nr_samples, lags)

    if debug:  # plot avg correlation at various lags with SEM
        plt.figure()
        plt.plot(lags, avg_corr)

        err = np.std(np.array(corrs), axis=0) / np.sqrt(nr_samples)
        m = avg_corr
        plt.fill_between(lags, m - err, m + err, alpha=0.3)
        plt.legend(['Mean', 'SEM'])
        plt.title('Optimal correlation at lag {}'.format(lags[np.argmax(avg_corr)]))

    if return_all:
        return lags, avg_corr
    else:
        return lags[np.argmax(avg_corr)]


def find_shift_multiple_stacks(paths: List[str]) -> float:
    """
    Find the average shift between odd and even lines (raster correction) for a stack that has multiple parts.
    Adrian 2020-07-20

    Args:
        paths: List of file paths to the stack parts

    Returns:
        Optimal lag with the highest average correlation across stacks
    """

    avg_corrs = []

    for path in paths:
        stack = io.imread(path)
        lags, avg_corr = find_shift_stack(stack, nr_samples=200, return_all=True)
        avg_corrs.append(avg_corr)

    stack_avg = np.mean(np.array(avg_corrs), axis=0)  # array from corrs has shape (nr_samples, lags)

    return lags[np.argmax(stack_avg)]


def apply_shift_to_stack(stack: np.ndarray, shift: int, crop_left: int = 50, crop_right: int = 50) -> np.ndarray:
    """
    Shift the corresponding lines (even or odd) to the left to optimal value and crop stack on the left.
    Adrian 2020-03-10

    Args:
        stack:      3D Image stack (nr_frames, x, y) that needs lines to be aligned.
        shift:      Number of pixels that should be shifted to the left. If > 0, even lines, else odd lines are shifted.
        crop_left:  Number of pixels that should be cropped on the left to remove artifacts.
        crop_right: Number of pixels that should be cropped on the right to remove artifacts.

    Returns:
        3D image stack (nr_frames, x, y-crop_left-crop_right), with lines aligned and cropped.
    """

    if shift > 0:
        stack[:, ::2, :-shift] = stack[:, ::2, shift:]  # shift all even lines by "shift" to the left
    if shift < 0:
        shift = -shift
        stack[:, 1::2, :-shift] = stack[:, 1::2, shift:]  # shift all odd lines by "shift" to the left

    if crop_left > 0:
        # remove a part on the left side of the image to avoid shifting artifact (if value around 10)
        # or to remove the artifact of late onset of the blanking on Scientifica microscope
        stack = stack[:, :, crop_right:-crop_left]

    return stack


def correct_line_shift_stack(stack: np.ndarray, crop_left: int = 5, crop_right: int = 0, nr_samples: int = 100,
                             nr_lags: int = 10) -> np.ndarray:
    """
    Correct the shift between even and odd lines in an imaging stack (nr_frames, x, y) acquired on the Scientifica.
    Adrian 2020-03-10

    Args:
        stack:      3D Image stack (#frames, x, y) that needs lines to be aligned.
        crop_left:  Number of pixels that should be cropped on the left to remove artifacts.
        crop_right: Number of pixels that should be cropped on the right to remove artifacts.
        nr_samples: Number of randomly chosen frames from the stack that should be used for correlation.
        nr_lags:    Range of lag shifts between lines that should be calculated.

    Returns:
        3D image stack (nr_frames, x, y-crop_left-crop_right), with lines aligned and cropped.
    """

    line_shift = find_shift_stack(stack, nr_lags=nr_lags, nr_samples=nr_samples)
    print('Correcting a shift of', line_shift, 'pixel.')

    stack = apply_shift_to_stack(stack, line_shift, crop_left=crop_left, crop_right=crop_right)

    return stack


def create_raster_and_offset_corrected_file(local_file: str, line_shift: int, offset: int = 0, crop_left: int = 0,
                                            crop_right: int = 0, channel: Optional[int] = None) -> str:
    """
    Apply corrections to a 2p imaging .tif file that is stored locally for faster processing.
    Returns path to corrected file saved in the same directory as input file. Raises Warning if a file on the
    Neurophys storage is given.
    Adrian 2020-07-22

    Args:
        local_file: Absolute file path of the .tif file.
        line_shift: Amount of pixels that even and odd lines are supposed to be shifted.
        offset:     Fixed value that is added to all pixel values of the stack to avoid negative pixel values.
        crop_left:  Number of pixels that should be cropped on the left to remove artifacts.
        crop_right: Number of pixels that should be cropped on the right to remove artifacts.
        channel:    If not None, deinterleave the stack and choose selected channel (base 0).

    Returns:
        Absolute file path of the new corrected .tif file.
    """

    if 'neurophys' in local_file:
        raise Warning('You have passed a file on neurophys storage to mc.create_corrected_file()!')

    stack = tif.imread(local_file)

    if channel is not None:
        stack = stack[channel::2]  # take every second frame, starting from channel (0/1)

    stack = apply_shift_to_stack(stack, line_shift,
                                 crop_left=crop_left, crop_right=crop_right)

    # Make movie positive (negative values crash NON-NEGATIVE matrix factorisation)
    stack = stack + offset

    new_path = os.path.splitext(local_file)[0] + '_corrected.tif'
    tif.imwrite(new_path, data=stack)
    return new_path


# =============================================================================
# Cache raw imaging files
# =============================================================================


def cache_files(paths: Union[List[str], str], cache_directory: str) -> List[str]:
    """
    Copy files from Neurophys-Storage to a local cache directory.
    Adrian 2020-07-22

    Args:
        paths:              Absolute file path(s) on the Neurophys-Storage server.
        cache_directory:    Local cache directory where the files should be copied to.

    Returns:
        Absolute file paths in the local cache directory.
    """
    if type(paths) == str:  # catch passed string
        paths = [paths]

    # Copy files and store new file paths
    local_paths = [shutil.copy(source_path, cache_directory) for source_path in paths]

    return local_paths


def delete_cache_files(cache_paths: Union[List[str], str]) -> None:
    """
     Delete files from the cache.

    Args:
        cache_paths: Absolute file paths in the local cache directory that should be deleted.
    """

    if type(cache_paths) == str:  # catch passed string
        cache_paths = [cache_paths]

    for path in cache_paths:
        if 'neurophys' in path:
            # sanity check to avoid deleting files on the neuropysiology by accident
            raise Exception('Aborted deletion because neurophys was in path!!!')

        # force garbage collection to unmap memory-mapped files (https://stackoverflow.com/questions/39953501/)
        import gc
        gc.collect()

        os.remove(path)


# =============================================================================
# Calculate correlation with 8-neighboring pixels in parallel
# =============================================================================

def single_neighbor_correlation(stack, i, j) -> float:
    """
    Calculate correlation of a single pixel with 8 neighboring pixels in time series.
    Do not call this function for border pixels, this will result in unexpected behavior!
    Adrian 2020-07-22

    Args:
        stack: 3D array (nr_frames, x, y)
        i: X-coordinate of target pixel.
        j: Y-coordinate of target pixel.

    Returns:
        Local correlation for target pixel.
    """

    trace = stack[:, i, j]

           # left   # middle   # right
    dis = [-1, -1, -1, 0, 0, 1, 1, 1]
    djs = [-1, 0, 1, -1, 1, -1, 0, 1]

    cor = 0
    for di, dj in zip(dis, djs):
        cor += np.corrcoef(trace, stack[:, i + di, j + dj])[0, 1] / 8

    return cor


def row_cor(stack_part: np.array) -> np.ndarray:
    """
    Calculation of neighbor correlation for one row for parallelization.
    Adrian 2020-07-22

    Args:
        stack_part: Section of whole frame stack with 3 rows (middle is target row), all columns.

    Returns:
        Neighbor correlation of target row pixels.
    """
    nr_columns = stack_part.shape[1]
    row = np.zeros(nr_columns)

    for j in range(1, nr_columns - 1):
        row[j] = single_neighbor_correlation(stack_part, 1, j)

    return row


def parallel_all_neighbor_correlations(stack: np.ndarray) -> np.ndarray:
    """
    Parallelize calculation of local correlation image. Each pixel with its 8 neighbors is processed in parallel.
    Adrian 2020-07-22

    Args:
        stack: 3D array (nr_frames, x, y)

    Returns:
        2D local correlation image
    """

    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count() - 1)

    nr_frames, max_i, max_j = stack.shape

    cor_image = np.zeros((max_i, max_j))

    # pixel by pixel as reference:
    for i in range(1, max_i - 1):
        for j in range(1, max_j - 1):
            cor_image[i, j] = single_neighbor_correlation(stack, i, j)

    # row by row as reference: pass row with top and bottom row)
    # rows = [ row_cor( stack[:, (i-1):(i+2), :] ) for i in range(1,max_i-1)]

    # parallel execution with starmap
    # parallel_rows = pool.starmap(
    #      row_cor, [[stack[:, (i-1):(i+2), :]] for i in range(1,max_i-1)]
    #                             )
    # cor_image = np.array( parallel_rows )
    pool.close()

    return cor_image
