#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 23/06/2021 16:06
@author: hheise

Utility functions for Hendriks DataJoint section
"""
import matplotlib
if matplotlib.get_backend() != 'TkAgg': # If TkAgg is set before during import, Python crashes with a stack overflow
    matplotlib.use('TkAgg')
import os
from pathlib import Path
import re
from typing import List, Any, Type, Optional, Iterable, Union, Tuple
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

import datajoint.errors
import yaml

import datajoint as dj
import login
login.connect()
from schema import common_mice, common_exp, common_img

REL_BACKUP_PATH = "Datajoint/manual_submissions"


def alphanumerical_sort(x: List[str]) -> List[str]:
    """
    Sorts a list of strings alpha-numerically, with proper interpretation of numbers in the string. Usually used to sort
    file names to ensure files are processed in the correct order (if file name numbers are not zero-padded).
    Elements are first  sorted by characters. If characters are the same, elements are sorted by numbers in the string.
    Consecutive digits are treated as one number and sorted accordingly (e.g. "text_11" will be sorted behind "text_2").
    Leading Zeros are ignored.

    Args:
        x:  List to be sorted. Elements are usually file names.

    Returns:
        Sorted list.
    """
    x_sort = x[:]

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    x_sort.sort(key=natural_keys)
    return x_sort


def make_yaml_backup(sess_dict: dict) -> None:
    """
    Creates YAML backup file for Hendrik's common_exp.Session() entries.

    Args:
        sess_dict: Holds data of the current common_exp.Session() entry.
    """
    identifier = common_exp.Session().create_id(investigator_name=sess_dict['username'],
                                                mouse_id=sess_dict['mouse_id'],
                                                date=sess_dict['day'],
                                                session_num=sess_dict['session_num'])
    file = os.path.join(login.get_neurophys_wahl_directory(), REL_BACKUP_PATH, identifier + '.yaml')

    # Transform session path from Path to string (with universal / separator) to make it YAML-compatible
    sess_dict['session_path'] = str(sess_dict['session_path']).replace("\\", "/")
    with open(file, 'w') as outfile:
        yaml.dump(sess_dict, outfile, default_flow_style=False)


def get_autopath(info: dict) -> str:
    """
    Creates absolute path for a session (with Neurophys directory) based on a common_exp.Session() entry dict.
    Structure: 'DATA-DIR\batch\mouseID\day'. Example: '...\Data\Batch1\M15\20201224'.

    Args:
        info: Holds information about the current session. The dict has to contain the keys "mouse_id" and "day".

    Returns:
        Absolute directory path of the session.
    """
    mouse = str(info['mouse_id'])
    batch = str((common_mice.Mouse & "username = '{}'".format(login.get_user())
                 & "mouse_id = {}".format(mouse)).fetch1('batch'))
    if batch == 0:
        raise Exception('Mouse {} has no batch (batch 0). Cannot create session path.\n'.format(mouse))

    # Get possible data paths
    paths = [Path(login.get_neurophys_data_directory()), *[Path(x) for x in login.get_alternative_data_directories()]]

    for poss_path in paths:
        path = os.path.join(poss_path, "Batch" + batch, "M" + mouse, info['day'].replace('-', ''))
        # Check if the folder exists, otherwise skip
        if os.path.isdir(path):
            # print(f'Path {path} exists.')
            return path
        # else:
        #     print(f'Path {path} does not exist.')

    # If no valid folder is found, raise an error
    raise ImportError(f'Found no valid folders for session:\n{info}')


def get_microsphere_sessions(mouse_id: int, pre_sessions: int = 5, post_sessions: int = 100) -> Tuple[list, list]:
    """
    Convenience function that looks up the dates of sessions around the microsphere surgery of a single mouse for
    easier analysis querying.

    Args:
        mouse_id        : ID of the mouse
        pre_sessions    : Number of pre-injection sessions to consider
        post_sessions   : Maximum number of post-injection sessions to consider

    Returns:
        One list with the dates, one with boolean flags which of these dates are pre-stroke sessions.
    """
    # get date of microsphere injection
    surgery_day = (common_mice.Surgery() & 'surgery_type="Microsphere injection"' &
                   f'mouse_id={mouse_id}').fetch1('surgery_date')

    # Get the dates of 5 imaging sessions before the surgery, and all dates from 2 days after it
    pre_dates = (common_img.Scan & f'mouse_id={mouse_id}' &
                 f'day <= "{surgery_day.date()}"').fetch('day')[-pre_sessions:]
    post_dates = (common_img.Scan & f'mouse_id={mouse_id}' &
                  f'day > "{surgery_day.date()+timedelta(days=1)}"').fetch('day')[:post_sessions]
    dates = [*pre_dates, *post_dates]
    is_pre = [*[True]*len(pre_dates), *[False]*len(post_dates)]

    return dates, is_pre


def remove_session_path(key: dict, path: str) -> str:
    """
    Creates absolute path of a single-trial file relative to it's session directory.

    Args:
        key:    Holds primary keys of the common_exp.Session() entry to which the file belongs.
        path:   Path of the single-trial file. Has to be on the same drive as the session directory.

    Returns:
        Relative file path with the session directory as the reference directory.
    """
    sess_path = os.path.join(login.get_neurophys_data_directory(), (common_exp.Session() & key).fetch1('session_path'))
    return os.path.relpath(path, sess_path)


def add_many_sessions(date: Union[str, Iterable[str]], mice: Union[int, Iterable[int]],
                      block: Optional[Union[int, List[int]]] = None, switch: Optional[List[List[int]]] = None,
                      **attributes: Any) -> None:
    """
    Automatically adds sessions of many mice on the same day to common_exp.Session and common_img.Scan (if it was an
    imaging session) and creates backup YAML files.
    "Block" and "switch" can be left empty or have to be a list with length 'len(mice)'.

    Args:
        date:           Date of the sessions in format (YYYY-MM-DD)
        mice:           Mouse_ids of mice that had a session on that day
        block:          Block number of the session, in case sessions are grouped. Defaults to 1 if left empty.
                        If set, needs one entry for all mice, or a list of entries, one per mouse).
        switch:         The first trial index of a new condition. List of lists, outer list has 'len(mice)' entries,
                        with each entry being a list with integers of condition-switched trials (multiple switches per
                        session are possible). If no condition switch in this session, leave empty, defaults to -1.
        **attributes:   Optional values for the Session() attributes.
    """

    # Figure out if sessions should be added day-wise (many mice on one day) or mouse-wise (many sessions for one mouse)
    if type(date) == str and type(mice) != int:
        iterator = mice
        mouse_wise = False
    elif type(date) != str and type(mice) == int:
        iterator = date
        mouse_wise = True
    else:
        raise TypeError("Either date or mice has to be a string, the other an ")

    # Set default values for attributes
    session_dict = dict(username='hheise',
                        session_num=1,
                        anesthesia='Awake',
                        setup='VR',
                        task='Active',
                        experimenter='hheise',
                        session_notes='')
    scan_dict = dict(username='hheise',
                     session_num=1,
                     microscope='Scientifica',
                     laser='MaiTai',
                     layer='CA1',
                     ca_name='GCaMP6f',
                     objective='16x',
                     nr_channels=1,
                     network_id=1)

    if mouse_wise:
        session_dict['mouse_id'] = mice
        scan_dict['mouse_id'] = mice
    else:
        session_dict['day'] = date
        scan_dict['day'] = date

    if block is None:
        block = [1] * len(iterator)
    elif type(block) == int:
        block = [block] * len(iterator)
    if switch is None:
        switch = [[-1]] * len(iterator)

    # Change default values if provided in attributes
    for key in session_dict:
        if key in attributes:
            session_dict[key] = attributes[key]
    for key in scan_dict:
        if key in attributes:
            scan_dict[key] = attributes[key]

    for idx, element in enumerate(iterator):

        # Expand dict for mouse-specific values
        if mouse_wise:
            element_session_dict = dict(**session_dict, day=element)
            element_scan_dict = dict(**scan_dict, day=element)
        else:
            element_session_dict = dict(**session_dict, mouse_id=element)
            element_scan_dict = dict(**scan_dict, mouse_id=element)

        # Set values for each mouse if multiple were provided
        for key, value in session_dict.items():
            if (type(value) == list) or (type(value) == tuple):
                element_session_dict[key] = session_dict[key][idx]
        for key, value in scan_dict.items():
            if (type(value) == list) or (type(value) == tuple):
                element_scan_dict[key] = scan_dict[key][idx]
        # Enter block and switch values in session_notes
        element_session_dict['session_notes'] = "{" + "'block':'{}', " \
                                                    "'switch':'{}', 'notes':'{}'".format(block[idx], switch[idx],
                                                                                         element_session_dict[
                                                                                             'session_notes']) + "}"

        # Create session folder path
        abs_session_path = Path(get_autopath(element_session_dict))
        element_session_dict['session_path'] = abs_session_path

        # Check if the folder exists, otherwise skip
        if not os.path.isdir(abs_session_path):
            print(f'ImportError: Could not find directory {abs_session_path}, skipping insert.')
            continue

        try:
            # Insert entry into Session()
            print(common_exp.Session().helper_insert1(element_session_dict))
        except datajoint.errors.DuplicateError as ex:
            print('Entry already exists in common_exp.Session(), skipping insert:\n', ex)

        # Find TIFF files in the session folder or subfolder to determine whether this was a imaging session
        if len(glob(str(abs_session_path) + '\\*.tif') +
               glob(str(abs_session_path) + '\\*\\*.tif')) > 0:
            try:
                common_img.Scan().insert1(element_scan_dict)
            except datajoint.errors.DuplicateError as ex:
                print('Entry already exists in common_img.Scan(), skipping insert:\n', ex)
        else:
            print(f'No TIFF files found, assuming that no imaging was performed. Check this!')
        print(' ')


def validate_segmentation(mice: Optional[Iterable[int]] = None, thr: Optional[int] = 20,
                          plot_all: Optional[bool] = False) -> None:
    """
    Plot number of accepted ROIs across sessions of specified mice. Creates one figure per mouse. By default, only plots
    mice that have a difference between sessions of more than "thr" percent.

    Args:
        mice: List of mouse IDs that should be plotted. If None, all mice are queried.
        thr: Percentage threshold of difference in ROI numbers between sessions above which the data will be plotted.
        plot_all: Bool flag whether mice without a difference above "thr" should be plotted as well.
    """
    thr_up = 1 + thr/100
    thr_low = 1 - thr / 100

    entries = common_img.Segmentation & 'username="hheise"'

    if mice is None:
        mice = np.unique(entries.fetch('mouse_id'))

    # For each mouse, compare nr of masks between sessions and raise alarm if the nr differs by > 20%
    for mouse in mice:
        print(f'Validating sessions of mouse {mouse}...')

        found_diff = False

        days = (entries & f'mouse_id={mouse}').fetch('day')         # Fetch days of all sessions of this mouse
        # Fetch nr of masks from the first day which is not directly queried
        prev_mask = (entries & f'mouse_id={mouse}' & f'day="{days[0]}"').fetch1('nr_masks')

        for day in days[1:]:
            # Fetch number of current mask and get relative difference
            curr_mask = (entries & f'mouse_id={mouse}' & f'day="{day}"').fetch1('nr_masks')
            nr_masks_diff = curr_mask / prev_mask

            # Construct and print message
            msg = [f'On day {day}, {thr}%', f'ROIs than on the previous session were found. ({curr_mask} vs {prev_mask}).']
            if nr_masks_diff > thr_up:
                print('\t', msg[0], 'more', msg[1])
                found_diff = True
            elif nr_masks_diff < thr_low:
                print('\t', msg[0], 'fewer', msg[1])
                found_diff = True
            prev_mask = curr_mask

        if found_diff or plot_all:
            # If a difference was found between neighboring sessions, plot nr_masks across sessions to spot outliers

            nr_masks = (entries & f'mouse_id={mouse}').fetch('nr_masks')

            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            ax[0].plot(days, nr_masks, 'o-')
            ax[0].set_title(f'Mouse {mouse}')
            ax[0].set_ylabel('accepted masks')

            ax[1].plot(days, np.hstack((1, nr_masks[1:]/nr_masks[:-1])), 'o-', c='orange')
            ax[1].set_ylabel('relative change between sessions')
            ax[1].set_xlabel('session dates')
            ax[1].axhline(1, c='black', linestyle='--')
            ax[1].axhline(thr_up, c='r', linestyle='--')
            ax[1].axhline(thr_low, c='r', linestyle='--')


# dj.table.Table should be the master class of all Datajoint tables, and as such the proper type hinting class
def add_column(table: Type[dj.table.Table], name: str, dtype: str, default_value: Optional[str] = None,
               use_keyword_default: bool = False, comment: Optional[str] = None) -> None:
    """
    A (hacky) convenience function to add a new column into an existing table (in-place).

    Args:
        table:                  Table to add new column (attribute) to.
        name:                   Name of the new column.
        dtype:                  Data type of the new column (see
                                https://docs.datajoint.org/python/v0.11/definition/06-Datatypes.html).
        default_value:          Default value for the new column. If 'null' or None, then the attribute is considered
                                non-required. Defaults to None.
        use_keyword_default:    Set to True if you want the default_value to be treated as MySQL keyword (e.g.
                                `CURRENT_TIMESTAMP`). This is False by default.
        comment:                Comment for the new column. Defaults to None.
    """
    full_table_name = table.full_table_name
    if default_value is None or default_value.strip().lower() == 'null':
        query = 'ALTER TABLE {} ADD {} {}'.format(full_table_name, name, dtype)
        definition = '{}=NULL: {}'.format(name, dtype)
    else:
        default_string = default_value if use_keyword_default else repr(default_value)
        query = 'ALTER TABLE {} ADD {} {} NOT NULL DEFAULT {}'.format(full_table_name, name, dtype, default_string)
        definition = '{}={}: {}'.format(name, default_string, dtype)

    if comment is not None:
        query += ' COMMENT "{}"'.format(comment)
        definition += ' # {}'.format(comment)
    table.connection.query(query)
    print('Be sure to add following entry to your table definition')
    print(definition)
    table.__class__._heading = None
