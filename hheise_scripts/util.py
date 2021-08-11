#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 23/06/2021 16:06
@author: hheise

Utility functions for Hendriks DataJoint section
"""
import os
from pathlib import Path
from copy import deepcopy
import re
from glob import glob

import yaml
import numpy as np

from schema import common_mice, common_exp
import login

REL_BACKUP_PATH = "Datajoint/manual_submissions"


def numerical_sort(x):
    """Sort a list of strings numerically"""
    x_sort = x.copy()
    def atoi(text):
        return int(text) if text.isdigit() else text
    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]
    x_sort.sort(key=natural_keys)
    return x_sort


def make_yaml_backup(sess_dict):
    """Create YAML backup file for Hendriks Session entries"""
    identifier = common_exp.Session().create_id(investigator_name=sess_dict['username'],
                                                mouse_id=sess_dict['mouse_id'],
                                                date=sess_dict['day'],
                                                session_num=sess_dict['session_num'])
    file = os.path.join(login.get_neurophys_wahl_directory(), REL_BACKUP_PATH, identifier + '.yaml')

    # Transform session path from Path to string (with universal / separator) to make it YAML-compatible
    sess_dict['session_path'] = str(sess_dict['session_path']).replace("\\", "/")
    with open(file, 'w') as outfile:
        yaml.dump(sess_dict, outfile, default_flow_style=False)


def get_autopath(info):
    """ Create automatic ABSOLUTE (with neurophys) session folder path based on the session_dict 'info'"""
    mouse = str(info['mouse_id'])
    batch = str((common_mice.Mouse & "username = '{}'".format(login.get_user())
                                   & "mouse_id = {}".format(mouse)).fetch1('batch'))
    if batch == 0:
        raise Exception('Mouse {} has no batch (batch 0). Cannot create session path.\n'.format(mouse))
    path = os.path.join(login.get_neurophys_data_directory(),
                        "Batch" + batch, "M" + mouse, info['day'].replace('-', ''))
    return path


def remove_session_path(key, path):
    """Make absolute trial path relative to session directory"""
    sess_path = os.path.join(login.get_neurophys_data_directory(), (common_exp.Session() & key).fetch1('session_path'))
    return os.path.relpath(path, sess_path)


def add_many_sessions(date, mice, block=None, switch=None, **attributes):
    """
    Automatically adds sessions of many mice on the same day. "Block" and "switch" can be set for each mouse separately.
    :param date: str, date of the sessions in format (YYYY-MM-DD)
    :param mice: list, mouse_ids of mice that had a session on that day
    :param attributes: dict containing the values of the Session() attributes
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

    if block is None:
        block = [1]*len(mice)
    if switch is None:
        switch = [[-1]] * len(mice)

    # Change default values if provided in attributes
    for key in session_dict:
        if key in attributes:
            session_dict[key] = attributes[key]

    for idx, mouse in enumerate(mice):

        # Expand dict for mouse-specific values
        mouse_session_dict = deepcopy(session_dict)
        mouse_session_dict['mouse_id'] = mouse

        # Set values for each mouse if multiple were provided
        for key, value in session_dict.items():
            if (type(value) == list) or (type(value) == tuple):
                mouse_session_dict[key] = session_dict[key][idx]
        # Enter block and switch values in session_notes
        mouse_session_dict['session_notes'] = "{" + "'block':'{}', " \
        "'switch':'{}', 'notes':'{}'".format(block[idx], switch[idx], mouse_session_dict['session_notes']) + "}"

        # Create session folder path
        mouse_session_dict['session_path'] = Path(get_autopath(mouse_session_dict))

        # Insert entry into Session()
        print(common_exp.Session().helper_insert1(mouse_session_dict))

        # Save data in YAML
        make_yaml_backup(mouse_session_dict)


def add_column(table, name, dtype, default_value=None, use_keyword_default=False, comment=None):
    """
    A (hacky) convenience function to add a new column into an existing table.

    Args:
        table (DataJoint table class instance): table to add new column (attribute) to
        name (str): name of the new column
        dtype (str): data type of the new column
        default_value (str, optional): default value for the new column. If 'null' or None, then the attribute
            is considered non-required. Defaults to None.
        use_keyword_default: Set to True if you want the default_value to be treated as MySQL keyword (e.g. `CURRENT_TIMESTAMP`)
            This is False by default.
        comment (str, optional): comment for the new column
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