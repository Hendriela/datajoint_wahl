#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Functions to re-populate database after a drop or crash
Created on Wed Jun 2 17:33:21 2021

@author: hheise
"""

import sys
sys.path.append("..")  # Adds higher directory to python modules path.

import login
import os
import glob
import yaml
from glob import glob

# connect to datajoint database
login.connect()
from schema import common_mice, common_exp
import gui_enter_new_mouse


def restore_data(tables=None, adjust_funcs=None, verbose=False):
    """
    Restores manually entered data from backup YAML files back into the database
    :param tables: optional, give a list of tables to be re-populated. Table names HAVE to be lowercase!
                    If None, all files will be inserted.
    :param adjust_funcs: optional, dict with table names as key and the associated adjustment function that changes the
                            dict structure of the entries for this table as value
    :param verbose: bool flag whether Exception messages should be displayed
    :return:
    """

    # Get backup directory
    backup_path = login.get_neurophys_wahl_directory() + "/" + gui_enter_new_mouse.get_backup_path()

    # Get list of all YAML files there
    yaml_list = glob(backup_path + "/*.yaml")

    if len(yaml_list) == 0:
        raise FileNotFoundError("No YAML files found at '{}'!".format(gui_enter_new_mouse.get_backup_path()))
    else:
        successful = 0
        skipped = 0
        if not verbose:
            block_print()
        for file in yaml_list:
            curr_table = os.path.basename(file).split('_')[0]
            # Check if the current file belongs to a table that should be restored
            if (tables is None) or (tables is not None and curr_table in tables):
                # Check if an adjustment function for that table has been provided
                if (adjust_funcs is not None) and (curr_table in adjust_funcs):
                    success = restore_data_from_yaml(file, adjust_funcs[curr_table], verbose)
                else:
                    success = restore_data_from_yaml(file, verbose=verbose)

                if success:
                    successful += 1
                else:
                    skipped += 1
                    if verbose:
                        print('Failed Insert into table "{}" of file {}\n'.format(curr_table, file))
        if not verbose:
            enable_print()
        print('Successfully inserted {}/{} entries into the database, and skipped {} files.'.format(successful,
                                                                                                    len(yaml_list),
                                                                                                    skipped))


def restore_data_from_yaml(path, adjust_func=None, verbose=False):
    """
    Loads a single YAML dict, makes changes if necessary, and inserts it into the database.
    :param path: str, full file name of the YAML file
    :param adjust_func: optional, custom function that takes the YAML dict as input, changes its contents and returns
                        the changed dict
    :param verbose: bool flag whether Exception messages should be displayed
    :return: True if insert worked, False if exception occurred
    """
    # Separate filename into segments and primary keys
    segments = os.path.basename(path).split('_')
    table_name = segments.pop(0)
    segments[-1] = segments[-1].split('.')[0]

    if table_name == 'mouse':
        table = common_mice.Mouse
    elif table_name == 'surgery':
        table = common_mice.Surgery
    elif table_name == 'injection':
        table = common_mice.Injection
    elif table_name == 'session':
        table = common_exp.Session
    else:
        raise NameError("Cannot find table named '%s'." % table_name)

    # Load YAML file
    with open(path) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    # If necessary, adjust the content to adapt to DB structure changes and save the new YAML
    if adjust_func is not None:
        data = adjust_func(data)
        with open(path, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    # Try to enter it into the database
    try:
        table.insert1(data)
        return True
    except Exception as ex:
        if verbose:
            print('\nException manually caught:', ex)
        return False


# Disable
def block_print():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = sys.__stdout__
