#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:33:17 2019
Functions to read in meta data from scanimage scans
@author: adhoff
"""

import numpy as np
from typing import Union, Dict

# if not installed: pip install scanimage-tiff-reader
from ScanImageTiffReader import ScanImageTiffReader as scanReader


def string_meta_info(path: str) -> str:
    """
    Returns the meta information as one long string with '\n' separations
    Adrian 2019-07-18

    Args:
        path: full file path of the .tif file

    Returns:
        String with metadata.
    """

    with scanReader(path) as file:
        meta_info = file.metadata()

    return meta_info


def parse_line(meta_info: str, tag: str) -> Union[float, str, np.ndarray]:
    """
    Get attribute value of one line in the meta_info string
    Adrian 2019-07-18

    Args:
        meta_info: Return value of read_meta_info
        tag:       Unique identifier of one line/attribute, e.g. "SI.hPmts.gains"

    Returns:
        Attribute value, type depends on the specified format of the tag
    """

    list_meta = meta_info.split('\n')
    tag = tag + ' '  # make sure no additional stuff is added after the tag

    matches = [s for s in list_meta if tag in s]

    if len(matches) != 1:
        raise Exception('The tag "%s" was found %d times (should be only 1)' % (tag, len(matches)))

    str_value = matches[0].split(' = ')[1]  # get part right of = in this line

    if str_value.startswith('[') & str_value.endswith(']'):
        # this is an array
        content = str_value[1:-1]  # remove brackets
        value = [helper_data_type(elem) for elem in content.split(' ')]
        value = np.array(value)  # turn list into numpy array

    else:
        # single number
        value = helper_data_type(str_value)

    return value


def helper_data_type(string_element: str) -> str:
    """
    Transforms a string into an integer, float or string.
    Adrian 2019-07-23

    Args:
        string_element: string to be converted

    Returns:
        Value converted to the appropriate type.
    """

    try:
        return int(string_element)
    except ValueError:
        try:
            return float(string_element)
        except ValueError:
            return string_element


def get_meta_info_as_dict(path: str) -> Dict:
    """
    Get a few interesting values of the meta-information of a scan as dictionary
    Adrian 2019-07-18

    Args:
        path: Complete file path to the scan .tif file


    Returns:
        Dictionary with some important meta information about the scan
    """
    # hardcoded value pairs:
    pairs = dict(gains='SI.hPmts.gains',
                 powers='SI.hBeams.powers',
                 saved='SI.hChannels.channelSave',
                 motor_pos='SI.hMotors.motorPosition',
                 absolute_pos='SI.hMotors.motorPositionTarget',
                 nr_lines='SI.hRoiManager.linesPerFrame',
                 pixel_per_line='SI.hRoiManager.pixelsPerLine',
                 fs='SI.hRoiManager.scanFrameRate',
                 line_period='SI.hRoiManager.linePeriod',
                 zoom='SI.hRoiManager.scanZoomFactor',
                 offsets='SI.hScan2D.channelOffsets',
                 line_phase='SI.hScan2D.linePhase',
                 pixel_time='SI.hScan2D.scanPixelTimeMean',
                 scanning='SI.hScan2D.scannerType')

    meta_string = string_meta_info(path)

    meta_dict = dict()

    # go through pairs one by one
    for key, value in pairs.items():
        meta_dict[key] = parse_line(meta_string, value)

    return meta_dict
