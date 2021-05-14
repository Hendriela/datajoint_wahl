#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rename this file to login.py after changing your name and password.

Has datajoint login and directory information for the local machine the code is running on.
The return values of these functions have to be adjusted for each machine and the file added to .gitignore because it
should not be synched across devices.
Adrian 2019-08-16

Adapted by Hendrik 2021-05-04
"""

def get_ip():
    """Return ip address of the server"""
    return '130.60.53.47'   # Ubuntu server in Anna-Sophias room

def get_user():
    """Return user name"""
    return 'root'           # default username of the DJ database

def get_password():
    """Return password"""
    return 'simple'         # default username of the DJ database

def connect():
    """ Connects to the database using the credentials in the login.py file"""

    import datajoint as dj

    dj.config['database.host'] = get_ip()
    dj.config['database.user'] = get_user()
    dj.config['database.password'] = get_password()
    dj.conn()

    # TODO: Define cache and external storage here as well


## Functions to modify file paths depending on location of execution

def get_cache_directory():
    """Return the local directory used as cache"""

    # comment this line out after creating a cache path
    raise Exception('The cached directory path has not been set in the file '
                    '"login.py". Please modify this file.')

    return  # for example for Hendrik's PC: 'C:\\Users\\hheise\\Datajoint\\temp'


def get_neurophys_wahl_directory():
    """Return  path to the neurophys-storage1 Wahl folder (common folder for all Wahl group members) on this system"""

    # comment this line out after setting the mapped neurophys storage path
    raise Exception('The path to the mapped neurophys directory path has not been set in the file '
                    '"login.py". Please modify this file.')

    return  # for example for Hendrik's PC: 'W:\\Neurophysiology-Storage1\\Wahl'

def get_neurophys_data_directory():
    """Return the path to the neurophys-storage1 data folder (specific folder for current user) on this system"""

    # comment this line out after setting the mapped neurophys storage path
    raise Exception('The path to the mapped neurophys directory path has not been set in the file '
                    '"login.py". Please modify this file.')

    return # for example for Hendrik's PC:  'W:\\Neurophysiology-Storage1\\Wahl\\Hendrik\\PhD\\Data'

def get_computer_name():
    """ Return the name of the local computer to check if the file is locally cached """

    # comment this line out after setting the computer name
    raise Exception('The name of the computer this code is run on has not been set in the file '
                    '"login.py". Please modify this file.')

    return # for example for Hendrik's PC:  'Hendrik_Lab'
