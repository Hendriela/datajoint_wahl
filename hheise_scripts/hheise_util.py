#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 23/06/2021 16:06
@author: hheise

Utility functions for Hendriks DataJoint section
"""
import os
from pathlib import Path
import re
from typing import List, Any, Type, Optional, Iterable
from glob import glob

import datajoint.errors
import yaml

import datajoint as dj
from schema import common_mice, common_exp, common_img
import login

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
    path = os.path.join(login.get_neurophys_data_directory(),
                        "Batch" + batch, "M" + mouse, info['day'].replace('-', ''))
    return path


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


def add_many_sessions(date: str, mice: Iterable[str], block: Optional[List[int]] = None,
                      switch: Optional[List[List[int]]] = None, **attributes: Any) -> None:
    """
    Automatically adds sessions of many mice on the same day to common_exp.Session and common_img.Scan (if it was an
    imaging session) and creates backup YAML files.
    "Block" and "switch" can be left empty or have to be a list with length 'len(mice)'.

    Args:
        date:           Date of the sessions in format (YYYY-MM-DD)
        mice:           Mouse_ids of mice that had a session on that day
        block:          Block number of the session, in case sessions are grouped. Defaults to 1 if left empty.
                        If set, needs one entry (int) per mouse).
        switch:         The first trial index of a new condition. List of lists, outer list has 'len(mice)' entries,
                        with each entry being a list with integers of condition-switched trials (multiple switches per
                        session are possible). If no condition switch in this session, leave empty, defaults to -1.
        **attributes:   Optional values for the Session() attributes.
    """

    # Set default values for attributes
    session_dict = dict(username='hheise',
                        day=date,
                        session_num=1,
                        anesthesia='Awake',
                        setup='VR',
                        task='Active',
                        experimenter='hheise',
                        session_notes='')
    scan_dict = dict(username='hheise',
                     day=date,
                     session_num=1,
                     microscope='Scientifica',
                     laser='MaiTai',
                     layer='CA1',
                     ca_name='GCaMP6f',
                     objective='16x',
                     nr_channels=1,
                     network_id=1)

    if block is None:
        block = [1] * len(mice)
    if switch is None:
        switch = [[-1]] * len(mice)

    # Change default values if provided in attributes
    for key in session_dict:
        if key in attributes:
            session_dict[key] = attributes[key]
    for key in scan_dict:
        if key in attributes:
            scan_dict[key] = attributes[key]

    for idx, mouse in enumerate(mice):

        # Expand dict for mouse-specific values
        mouse_session_dict = dict(**session_dict, mouse_id=mouse)
        mouse_scan_dict = dict(**scan_dict, mouse_id=mouse)

        # Set values for each mouse if multiple were provided
        for key, value in session_dict.items():
            if (type(value) == list) or (type(value) == tuple):
                mouse_session_dict[key] = session_dict[key][idx]
        for key, value in scan_dict.items():
            if (type(value) == list) or (type(value) == tuple):
                mouse_scan_dict[key] = scan_dict[key][idx]
        # Enter block and switch values in session_notes
        mouse_session_dict['session_notes'] = "{" + "'block':'{}', " \
                                                    "'switch':'{}', 'notes':'{}'".format(block[idx], switch[idx],
                                                                                         mouse_session_dict[
                                                                                             'session_notes']) + "}"

        # Create session folder path
        abs_session_path = Path(get_autopath(mouse_session_dict))
        mouse_session_dict['session_path'] = abs_session_path

        try:
            # Insert entry into Session()
            print(common_exp.Session().helper_insert1(mouse_session_dict))
        except datajoint.errors.DuplicateError as ex:
            print('Entry already exists in common_exp.Session(), skipping insert:\n', ex)

        # Find TIFF files in the session folder or subfolder to determine whether this was a imaging session
        if len(glob(str(abs_session_path) + '\\file_*.tif') +
               glob(str(abs_session_path) + '\\*\\file_*.tif')) > 0:
            try:
                common_img.Scan().insert1(mouse_scan_dict)
            except datajoint.errors.DuplicateError as ex:
                print('Entry already exists in common_img.Scan(), skipping insert:\n', ex)
        else:
            print(f'No TIFF files found, assuming that no imaging was performed. Check this!')
        print(' ')
        # Save data in YAML
        make_yaml_backup(mouse_session_dict)


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
