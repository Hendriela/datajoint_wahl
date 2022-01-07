#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rename this file to login.py after changing your name and password.

Has datajoint login and directory information for the local machine the code is running on.
The return values of these functions have to be adjusted for each machine and the file added to .gitignore because it
should not be synched across devices.
Adrian 2019-08-16

To store the password on your computer, run the command
> keyring.set_password('datajoint_user', 'your_username', 'your_password')
once on each machine.
This has to be the same password you used when registering at the MySQL database!

Adapted by Hendrik 2021-05-04
"""
import os

import keyring
import datajoint as dj
import yaml

# define variable to keep track of working directory
__cwd__ = ""


def get_ip() -> str:
    """Return ip address of the server"""
    return '130.60.53.47'  # Ubuntu server in Anna-Sophias room


def get_user() -> str:
    """Return user name (short name) that is used in the Datajoint and MySQL credentials."""
    # Remove this line after setting a username
    raise Exception('Define a username before connecting to the database.')
    return  # Put your shortname here


def get_password() -> str:
    """Return password of the Datajoint and MySQL credentials"""
    # Before using DataJoint for the first time on a computer, store your password with
    # > keyring.set_password('datajoint_user', 'your_username', 'your_password')
    # This has to be the same password you used when registering at the MySQL database!
    return keyring.get_password('datajoint_user', get_user())


def connect() -> None:
    """ Connects to the database using the credentials in the login.py file"""
    dj.config['database.host'] = get_ip()
    dj.config['database.user'] = get_user()
    dj.config['database.password'] = get_password()
    dj.conn()

    # set working directory to neurophys by default
    global __cwd__
    __cwd__ = str(get_neurophys_data_directory())

    # TODO: Define cache and external storage here as well


def get_default_parameters() -> dict:
    """
    Load user-specific gui_params.yaml file with default parameters for various GUIs.

    Returns:
        Dictionary with default parameters, some grouped in sub-dicts. See gui_params.yaml for data.
    """

    tmp_wd = os.getcwd()  # Store the current working directory from where the function was called
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Change working directory to path of the current file

    try:
        with open(r'gui_params.yaml') as file:
            # The FullLoader parameter handles the conversion from YAML scalar values to Python's dictionary format
            default_params = yaml.load(file, Loader=yaml.FullLoader)
        os.chdir(tmp_wd)  # Reset working directory

        return default_params

    except FileNotFoundError:
        os.chdir(tmp_wd)  # Reset working directory
        raise FileNotFoundError("Parameter file gui_params.yaml could not be found. Check that it is in the same"
                                "folder as login.py")


## Functions to modify file paths depending on location of execution

def get_cache_directory() -> str:
    """Return the local directory used as cache (temporary storage of files during processing)"""

    # comment this line out after creating a cache path
    raise Exception('The cached directory path has not been set in the file '
                    '"login.py". Please modify this file.')

    return  # for example for Hendrik's PC: 'C:\\Users\\hheise\\Datajoint\\temp'


def get_neurophys_wahl_directory() -> str:
    """Return  path to the neurophys-storage1 Wahl folder (common folder for all Wahl group members) on this system"""

    # comment this line out after setting the mapped neurophys storage path
    raise Exception('The path to the mapped neurophys directory path has not been set in the file '
                    '"login.py". Please modify this file.')

    return  # for example for Hendrik's PC: 'W:\\Neurophysiology-Storage1\\Wahl'


def get_neurophys_data_directory() -> str:
    """Return the path to the neurophys-storage1 data folder (specific folder for current user) on this system"""

    # comment this line out after setting the mapped neurophys storage path
    raise Exception('The path to the mapped neurophys directory path has not been set in the file '
                    '"login.py". Please modify this file.')

    return  # for example for Hendrik's PC:  'W:\\Neurophysiology-Storage1\\Wahl\\Hendrik\\PhD\\Data'


def get_computer_name() -> str:
    """ Return the name of the local computer to check if the file is locally cached """

    # comment this line out after setting the computer name
    raise Exception('The name of the computer this code is run on has not been set in the file '
                    '"login.py". Please modify this file.')

    return  # for example for Hendrik's PC:  'Hendrik_Lab'


def get_working_directory() -> str:
    """Return the current working directory"""
    global __cwd__
    return __cwd__


def set_working_directory(new_path: str) -> None:
    """Set the current working directory to the one specified by new_path"""
    global __cwd__
    __cwd__ = str(new_path)
