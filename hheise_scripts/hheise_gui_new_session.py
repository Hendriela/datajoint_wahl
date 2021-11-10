#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:13:55 2019
@author: adhoff
Installation of wxpython in Ubuntu 18.04 (pip install did not work):
conda install -c anaconda wxpython
pip install datajoint

Adapted by Hendrik 2021-05-15
"""

import sys

sys.path.append("..")  # Adds higher directory to python modules path.

import argparse
import wx
import login
import datajoint as dj
from datetime import datetime
import os
import glob
import numpy as np
import yaml
from pathlib import Path
from typing import Union

# connect to datajoint database
login.connect()
from schema import common_mice, common_exp, common_img

# import user-specific schema
if login.get_user() == "hheise":
    from schema import hheise_behav as user_behav
else:
    raise ImportError("Script has to be run by user 'hheise'!")

# =============================================================================
# HARDCODED PARAMETER FOR GUI
# =============================================================================

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

WINDOW_WIDTH_L = 900

BUTTON_WIDTH = 130
BUTTON_HEIGHT = 40

# session info
S_LEFT = 30 + 20  # position of the first element in session box
S_TOP = 20 + 30
ROW = 70
COL = 200
S_HEIGHT = 330  # height of session box
S_WIDTH = WINDOW_WIDTH_L - 2 * S_LEFT

# User-specific information
U_TOP = S_TOP + 20
U_LEFT = S_LEFT + 4 * COL + 40
U_HEIGHT = S_HEIGHT
U_WIDTH = 2 * COL - 20

# imaging
I_LEFT = S_LEFT
I_TOP = S_HEIGHT + 70
I_HEIGHT = 3 * ROW - 40
I_WIDTH = S_WIDTH

# Status box
L_TOP = I_TOP + I_HEIGHT

# relative path to manual_submission backup folder inside the Neurophysiology Wahl directory (common for all users)
REL_BACKUP_PATH = "Datajoint/manual_submissions"

# Parse flag whether to display dead mice in the GUI
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sacrificed", help="show sacrificed mice",
                    action="store_true")
args = parser.parse_args()


# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

def find_default(params: dict, category: str, keyword: str) -> Union[str, dict]:
    """
    Wrapper function that looks up value in dict and displays a custom error message to the GUI.

    Args:
        params:     Dict with parameter names and their default values (from gui_params.yaml)
        category:   Name of category (1st level dict) in which "keyword" is located
        keyword:    Name of parameter

    Returns:
        Default value of given parameter

    Raises:
        KeyError with custom message that the default parameter should be defined in the YAML file.
    """

    try:
        return params[category][keyword]
    except KeyError:
        raise KeyError(f'Default value for parameter {keyword} not found.\nDefine {keyword} and its default value under'
                       f'the {category} category in {os.path.join(os.getcwd(), "gui_params.yaml")}')


# Load YAML file (has to be present in the same folder
with open('..\\gui_params.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML scalar values to Python's dictionary format
    default_params = yaml.load(file, Loader=yaml.FullLoader)

# Set current day as default day
current_day = datetime.today().strftime('%Y-%m-%d')  # YYYY-MM-DD

# Get path to the data folder on the Neurophysiology server from login.py
path_neurophys_data = login.get_neurophys_data_directory()

# Sanity checks: check if the username has been changed from the template and if the username exists in the database
investigators = list(common_mice.Investigator().fetch("username"))  # Fetch list of investigator usernames from DJ
if default_params['username'] == 'default_username':
    raise ValueError("Current username is 'default_username'. \nInsert your username in the gui_params_TEMPLATE.yaml.")
elif default_params['username'] not in investigators:
    raise ValueError("Current username '{}' does not exist in common_mice.Investigator(). \n\tInsert the new username "
                     "there before adding mice and sessions to it.".format(default_params['username']))

# Check the default values for adaptive parameters
if default_params['behavior']['default_experimenter'] == 'username':
    default_params['behavior']['default_experimenter'] = default_params['username']

for key, value in default_params['paths'].items():
    if value == 'neurophys':
        default_params['paths'][key] = path_neurophys_data

# Get all mouse_IDs from the current investigator that are still alive and sort them by descending ID
username_filter = "username = '{}'".format(default_params['username'])
if args.sacrificed:
    raw_mouse_ids = (common_mice.Mouse() & username_filter)
else:
    raw_mouse_ids = (common_mice.Mouse() & username_filter) - common_mice.Sacrificed()
mouse_ids = raw_mouse_ids.fetch('mouse_id', order_by='mouse_id DESC')
if default_params['behavior']['default_mouse'] == 'last_mouse':
    default_params['behavior']['default_mouse'] = mouse_ids[0]

# Check how many sessions this mouse has on the current day and adjust the trial counter accordingly
mouse_id_filter = "mouse_id = '{}'".format(default_params['behavior']['default_mouse'])
date_filter = "day = '{}'".format(current_day)
all_trials = (common_exp.Session() & username_filter & mouse_id_filter & date_filter).fetch('session_num')
if len(all_trials) == 0:
    next_trial = '1'
else:
    next_trial = str(max(all_trials) + 1)

# =============================================================================
# Load options for drop down menus
# =============================================================================

investigator = default_params['username']
inv_fullname = (common_mice.Investigator() & "username = '{}'".format(investigator)).fetch1('full_name')

setups = common_exp.Setup().fetch('setup')
tasks = (common_exp.Task() & "username = '{}'".format(investigator)).fetch('task')  # Restrict tasks by investigator
anesthesias = common_exp.Anesthesia().fetch('anesthesia')
experimenters = common_mice.Investigator().fetch('username')

microscopes = common_img.Microscope().fetch('microscope')
lasers = common_img.Laser().fetch('laser')
layers = common_img.Layer().fetch('layer')
indicators = common_img.CaIndicator().fetch('ca_name')


# =============================================================================
# Default parameter for dropdown menus and text boxes
# =============================================================================


class window(wx.Frame):

    def __init__(self, parent, id):

        wx.Frame.__init__(self, parent, id, '{} ({}): Enter data into pipeline'.format(inv_fullname, investigator),
                          size=(WINDOW_WIDTH, WINDOW_HEIGHT))
        panel = wx.Panel(self)

        self.job_list = list()  # save jobs in format [ [table, entry_dict, source_path, target_path], [...], ...]
        # =============================================================================
        # Upper left box: Add new session
        # =============================================================================
        wx.StaticBox(panel, label='SESSION INFORMATION',
                     pos=(S_LEFT - 20, S_TOP - 30), size=(S_WIDTH, S_HEIGHT))

        # Mouse name
        wx.StaticText(panel, label="Mouse ID:", pos=(S_LEFT, S_TOP))
        self.mouse_name = wx.ComboBox(panel, choices=np.array(mouse_ids, dtype=str), style=wx.CB_READONLY,
                                      pos=(S_LEFT, S_TOP + 20), size=(170, -1))
        self.mouse_name.Bind(wx.EVT_COMBOBOX, self.event_mouse_selected)
        item = self.mouse_name.FindString(str(default_params['behavior']['default_mouse']))
        self.mouse_name.SetSelection(item)

        # Day of experiment
        wx.StaticText(panel, label="Day (YYYY-MM-DD):", pos=(S_LEFT + COL, S_TOP))
        self.day = wx.TextCtrl(panel, pos=(S_LEFT + COL, S_TOP + 20), size=(170, -1))
        self.day.SetValue(current_day)

        # Trial
        wx.StaticText(panel, label="Session number (that day):", pos=(S_LEFT + 400, S_TOP))
        self.trial = wx.TextCtrl(panel, pos=(S_LEFT + 2 * COL, S_TOP + 20), size=(170, -1))
        self.trial.SetValue(next_trial)

        # Folder with behavioral data
        wx.StaticText(panel, label="Session folder:", pos=(S_LEFT, S_TOP + ROW))
        self.session_folder = wx.TextCtrl(panel, value=login.get_neurophys_data_directory(),
                                          pos=(S_LEFT, S_TOP + ROW + 20),
                                          size=(2 * COL - 20, 25))

        # Button to select new folder
        self.select_session_folder = wx.Button(panel, label="Select folder",
                                               pos=(S_LEFT + 2 * COL, S_TOP + ROW + 20), size=(100, 25))
        self.Bind(wx.EVT_BUTTON, self.event_select_session_folder, self.select_session_folder)

        # Checkbox if session folder should be created automatically
        # wx.StaticText(panel, label="Find session folder automatically?", pos=(S_LEFT + 2 * COL, S_TOP + ROW))
        self.autopath = wx.CheckBox(panel, label="Find session folder automatically?",
                                    pos=(S_LEFT + 2 * COL + 120, S_TOP + ROW + 23), size=(200, 20))
        self.autopath.SetValue(True)

        # Setup
        wx.StaticText(panel, label="Setup:", pos=(S_LEFT, S_TOP + 2 * ROW))
        self.setup = wx.ComboBox(panel, choices=setups, style=wx.CB_READONLY,
                                 pos=(S_LEFT, S_TOP + 2 * ROW + 20), size=(170, -1))
        item = self.setup.FindString(default_params['behavior']['default_setup'])
        self.setup.SetSelection(item)

        # Task
        wx.StaticText(panel, label="Task:", pos=(S_LEFT + COL, S_TOP + 2 * ROW))
        self.task = wx.ComboBox(panel, choices=tasks, style=wx.CB_READONLY,
                                pos=(S_LEFT + COL, S_TOP + 2 * ROW + 20), size=(170, -1))
        item = self.task.FindString(default_params['behavior']['default_task'])
        self.task.SetSelection(item)

        # Anesthesia
        wx.StaticText(panel, label="Anesthesia:", pos=(S_LEFT + 2 * COL, S_TOP + 2 * ROW))
        self.anesthesia = wx.ComboBox(panel, choices=anesthesias, style=wx.CB_READONLY,
                                      pos=(S_LEFT + 2 * COL, S_TOP + 2 * ROW + 20), size=(170, -1))
        item = self.anesthesia.FindString(default_params['behavior']['default_anesthesia'])
        self.anesthesia.SetSelection(item)

        # Experimenter
        wx.StaticText(panel, label="Experimenter:", pos=(S_LEFT + 3 * COL, S_TOP + 2 * ROW))
        self.experimenter = wx.ComboBox(panel, choices=experimenters, style=wx.CB_READONLY,
                                        pos=(S_LEFT + 3 * COL, S_TOP + 2 * ROW + 20), size=(170, -1))
        item = self.experimenter.FindString(default_params['behavior']['default_experimenter'])
        self.experimenter.SetSelection(item)

        # Notes
        wx.StaticText(panel, label="Notes:", pos=(S_LEFT, S_TOP + 3 * ROW))
        self.notes = wx.TextCtrl(panel, value="",
                                 style=wx.TE_MULTILINE,
                                 pos=(S_LEFT, S_TOP + 3 * ROW + 20),
                                 size=(WINDOW_WIDTH_L - 3 * S_LEFT - 200, 50))

        # Load session button
        self.load_session_button = wx.Button(panel, label="Load session",
                                             pos=(S_LEFT + 3 * COL, S_TOP),
                                             size=(150, 50))
        self.Bind(wx.EVT_BUTTON, self.event_load_session, self.load_session_button)

        # Submit session button
        self.submit_session_button = wx.Button(panel, label="Submit session",
                                               pos=(S_LEFT + 3 * COL, S_TOP + 3 * ROW + 20),
                                               size=(150, 50))
        self.Bind(wx.EVT_BUTTON, self.event_submit_session, self.submit_session_button)

        # =============================================================================
        # Upper left box: Add new session
        # =============================================================================
        wx.StaticBox(panel, label='USER-SPECIFIC INFORMATION',
                     pos=(U_LEFT - 40, U_TOP - 50), size=(U_WIDTH, U_HEIGHT))

        # Hendriks parameters
        wx.StaticBox(panel, label='Hendrik', pos=(U_LEFT - 20, U_TOP - 20), size=(2 * COL / 1.5, ROW))

        # Block
        wx.StaticText(panel, label="Session block:", pos=(U_LEFT, U_TOP))
        self.block = wx.TextCtrl(panel, pos=(U_LEFT, U_TOP + 20), size=(170 / 2, -1))
        self.block.SetValue('1')

        # Condition switch
        wx.StaticText(panel, label="First trials of\ncondition switch(es):", pos=(U_LEFT + COL / 2, U_TOP - 12))
        self.switch = wx.TextCtrl(panel, pos=(U_LEFT + COL / 2, U_TOP + 20), size=(170 / 2, -1))
        self.switch.SetValue('[-1]')

        # =============================================================================
        # Enter imaging data
        # =============================================================================

        wx.StaticBox(panel, label='IMAGING INFORMATION',
                     pos=(I_LEFT - 20, I_TOP - 30), size=(I_WIDTH, I_HEIGHT))

        # Dropdown menus
        # Microscope
        wx.StaticText(panel, label="Microscope:", pos=(I_LEFT, I_TOP))
        self.microscope = wx.ComboBox(panel, choices=microscopes, style=wx.CB_READONLY,
                                      pos=(I_LEFT, I_TOP + 20), size=(170, -1))
        item = self.microscope.FindString(find_default(default_params, 'imaging', 'default_microscope'))
        self.microscope.SetSelection(item)

        # Laser
        wx.StaticText(panel, label="Laser:", pos=(I_LEFT + COL, I_TOP))
        self.laser = wx.ComboBox(panel, choices=lasers, style=wx.CB_READONLY,
                                 pos=(I_LEFT + COL, I_TOP + 20), size=(170, -1))
        item = self.laser.FindString(find_default(default_params, 'imaging', 'default_laser'))
        self.laser.SetSelection(item)

        # Layer
        wx.StaticText(panel, label="Layer:", pos=(I_LEFT + 2 * COL, I_TOP))
        self.layer = wx.ComboBox(panel, choices=layers, style=wx.CB_READONLY,
                                 pos=(I_LEFT + 2 * COL, I_TOP + 20), size=(170, -1))
        item = self.layer.FindString(find_default(default_params, 'imaging', 'default_layer'))
        self.layer.SetSelection(item)

        # Calcium Indicator
        wx.StaticText(panel, label="Calcium indicator:", pos=(I_LEFT + 3 * COL, I_TOP))
        self.indicator = wx.ComboBox(panel, choices=indicators, style=wx.CB_READONLY,
                                     pos=(I_LEFT + 3 * COL, I_TOP + 20), size=(170, -1))
        item = self.indicator.FindString(find_default(default_params, 'imaging', 'default_indicator'))
        self.indicator.SetSelection(item)

        # Objective
        wx.StaticText(panel, label="Objective:", pos=(I_LEFT, I_TOP + ROW))
        self.objective = wx.TextCtrl(panel, pos=(I_LEFT, I_TOP + ROW + 20), size=(170, -1))
        self.objective.SetValue('16x')

        # Network
        wx.StaticText(panel, label="Network:", pos=(I_LEFT + COL, I_TOP + ROW))
        self.network = wx.TextCtrl(panel, pos=(I_LEFT + COL, I_TOP + ROW + 20), size=(170, -1))
        self.network.SetValue('1')

        # Channels
        wx.StaticText(panel, label="Channels:", pos=(I_LEFT + 2 * COL, I_TOP + ROW))
        self.channels = list()  # list with checkboxes
        for i in range(2):
            self.channels.append(
                wx.CheckBox(panel, label=str(i + 1),
                            pos=(I_LEFT + 2 * COL + i * 40, I_TOP + ROW + 20), size=(40, 20))
            )
        self.channels[0].SetValue(True)

        # Submit imaging button
        self.submit_img_button = wx.Button(panel, label="Submit Scan",
                                           pos=(I_LEFT + 3 * COL, I_TOP + ROW),
                                           size=(150, 50))
        self.Bind(wx.EVT_BUTTON, self.event_submit_scan, self.submit_img_button)

        # =============================================================================
        # Submit and close buttons
        # =============================================================================

        self.quit_button = wx.Button(panel, label="Quit",
                                     pos=(30, L_TOP),
                                     size=(BUTTON_WIDTH, BUTTON_HEIGHT))
        self.Bind(wx.EVT_BUTTON, self.event_quit_button, self.quit_button)

        # status text
        self.status_text = wx.TextCtrl(panel, value="Status updates will appear here:\n",
                                       style=wx.TE_MULTILINE,
                                       pos=(40 + BUTTON_WIDTH, L_TOP),
                                       size=(WINDOW_WIDTH - BUTTON_WIDTH - 100, WINDOW_HEIGHT - L_TOP - 80))

    # =============================================================================
    # Events for menus and button presses
    # =============================================================================

    def get_autopath(self, info):
        """ Create automatic ABSOLUTE (with neurophys) session folder path based on the session_dict 'info'"""

        mouse = str(info['mouse_id'])
        batch = str((common_mice.Mouse & username_filter & "mouse_id = {}".format(mouse)).fetch1('batch'))
        if batch == 0:
            self.status_text.write('Mouse {} has no batch (batch 0). Cannot create session path.\n'.format(mouse))
            raise Exception
        path = os.path.join(login.get_neurophys_data_directory(),
                            "Batch" + batch, "M" + mouse, info['day'].replace('-', ''))
        return path

    def event_mouse_selected(self, event):
        """ The user selected a mouse name in the dropdown menu """
        print('New mouse selected')

    def event_submit_session(self, event):
        """ The user clicked on the button to submit a session """

        # create session dictionary that can be entered into datajoint pipeline
        session_dict = dict(username=investigator,
                            mouse_id=self.mouse_name.GetValue(),
                            day=self.day.GetValue(),
                            session_num=int(self.trial.GetValue()),
                            session_path=Path(self.session_folder.GetValue()),
                            anesthesia=self.anesthesia.GetValue(),
                            setup=self.setup.GetValue(),
                            task=self.task.GetValue(),
                            experimenter=self.experimenter.GetValue(),
                            session_notes=self.notes.GetValue()
                            )

        # check if the session is already in the database (most common error)
        key = dict(username=investigator,
                   mouse_id=self.mouse_name.GetValue(),
                   day=self.day.GetValue(),
                   session_num=int(self.trial.GetValue()))
        if len(common_exp.Session() & key) > 0:
            message = 'The session you wanted to enter into the database already exists.\n' + \
                      'Therefore, nothing was entered into the database.'
            wx.MessageBox(message, caption="Session already in database", style=wx.OK | wx.ICON_INFORMATION)
            return

        # find path automatically if box was ticked
        if int(self.autopath.GetValue()) == 1:
            auto_path = self.get_autopath(session_dict)
            if auto_path is None:
                self.status_text.write('No automatic folder method defined yet for user: ' + str(investigator) + '\n')
            else:
                session_dict['session_path'] = Path(self.get_autopath(session_dict))

        # Store Hendriks additional information in the "notes" section as a string dict
        session_dict['session_notes'] = "{" + "'block':'{}', 'switch':'{}', 'notes':'{}'".format(
            self.block.GetValue(),
            self.switch.GetValue(),
            session_dict['session_notes']) + "}"

        # add entry to database
        try:
            common_exp.Session().helper_insert1(session_dict)
            self.status_text.write('Sucessfully entered new session: ' + str(key) + '\n')

            # save dictionary that is entered in a backup YAML file for faster re-population
            identifier = common_exp.Session().create_id(investigator_name=investigator,
                                                        mouse_id=session_dict['mouse_id'],
                                                        date=datetime.strptime(session_dict['day'], '%Y-%m-%d'),
                                                        session_num=session_dict['session_num'])
            file = os.path.join(login.get_neurophys_wahl_directory(), REL_BACKUP_PATH, identifier + '.yaml')
            # TODO show prompt if a backup file with this identifier already exists and ask the user to overwrite
            # if os.path.isfile(file):
            #     message = 'The backup file of the session you wanted to enter into the database with the unique identifier ' \
            #               '{} already exists.\nTherefore, nothing was entered into the database.'.format(identifier)
            #     wx.MessageBox(message, caption="Backup file already exists", style=wx.OK | wx.ICON_INFORMATION)
            #     return
            # else:

            # Transform session path from Path to string (with universal / separator) to make it YAML-compatible
            session_dict['session_path'] = str(session_dict['session_path']).replace("\\", "/")
            with open(file, 'w') as outfile:
                yaml.dump(session_dict, outfile, default_flow_style=False)

        # except Exception as ex:
        except ValueError as ex:
            print('Exception manually caught:', ex)
            self.status_text.write('Error: ' + str(ex) + '\n')

    def event_submit_scan(self, event):
        """ The user clicked on the button to submit a scan """

        # create session dictionary that can be entered into datajoint pipeline
        scan_dict = dict(username=investigator,
                         mouse_id=self.mouse_name.GetValue(),
                         day=self.day.GetValue(),
                         session_num=int(self.trial.GetValue()),
                         microscope=self.microscope.GetValue(),
                         laser=self.laser.GetValue(),
                         layer=self.layer.GetValue(),
                         ca_name=self.indicator.GetValue(),
                         objective=self.objective.GetValue(),
                         nr_channels=sum([checkbox.GetValue() for checkbox in self.channels]),
                         network_id=self.network.GetValue(),
                         )

        # add entry to database
        try:
            key = dict(username=investigator,
                       mouse_id=self.mouse_name.GetValue(),
                       day=self.day.GetValue(),
                       session_num=int(self.trial.GetValue()))
            common_img.Scan().insert1(scan_dict)
            self.status_text.write('Sucessfully entered new scan: ' + str(key) + '\n')

            # save dictionary that is entered in a backup YAML file for faster re-population
            identifier = "scan_{}_{}_{}_{}".format(scan_dict['username'], scan_dict['mouse_id'], scan_dict['day'],
                                                   scan_dict['session_num'])
            file = os.path.join(login.get_neurophys_wahl_directory(), REL_BACKUP_PATH, identifier + '.yaml')
            # TODO show prompt if a backup file with this identifier already exists and ask the user to overwrite
            # if os.path.isfile(file):
            #     message = 'The backup file of the session you wanted to enter into the database with the unique identifier ' \
            #               '{} already exists.\nTherefore, nothing was entered into the database.'.format(identifier)
            #     wx.MessageBox(message, caption="Backup file already exists", style=wx.OK | wx.ICON_INFORMATION)
            #     return
            # else:

            with open(file, 'w') as outfile:
                yaml.dump(scan_dict, outfile, default_flow_style=False)

        # except Exception as ex:
        except dj.errors.IntegrityError as ex:
            print('IntegrityError:', ex)
            self.status_text.write('IntegrityError: ' + str(ex) + '\nThis most likely means that you tried to enter a '
                                                                  'Scan where the Session is not in the database yet.'
                                                                  'Enter Session before the Scan.\n')

    def event_load_session(self, event):
        """ User wants to load additional information about session into GUI """

        session_dict = dict(username=investigator,
                            mouse_id=self.mouse_name.GetValue(),
                            day=self.day.GetValue(),
                            session_num=int(self.trial.GetValue()))
        entries = (common_exp.Session() & session_dict).fetch(as_dict=True)
        # check there is only one table corresponding to this
        if len(entries) > 1:
            self.status_text.write(
                'Can not load session info for {} because there are {} sessions corresponding to this'.format(
                    session_dict, len(entries)) + '\n')
            return

        entry = entries[0]

        # set the selections in the menus according to the loaded info
        item = self.setup.FindString(entry['setup'])
        self.setup.SetSelection(item)
        item = self.task.FindString(entry['task'])
        self.task.SetSelection(item)
        # self.stage.SetValue(str(entry['stage']))
        item = self.anesthesia.FindString(entry['anesthesia'])
        self.anesthesia.SetSelection(item)
        item = self.experimenter.FindString(entry['experimenter'])
        self.experimenter.SetSelection(item)
        self.notes.SetValue(entry['notes'])

    def event_select_session_folder(self, event):
        """ User clicked on select session folder button """

        # open file dialog
        folder_dialog = wx.DirDialog(parent=None, message="Choose directory of session",
                                     defaultPath=self.session_folder.GetValue(),
                                     style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        exit_flag = folder_dialog.ShowModal()
        # update the folder in the text box when the user exited with ok
        if exit_flag == wx.ID_OK:
            self.session_folder.SetValue(folder_dialog.GetPath())

    def event_quit_button(self, event):
        """ User pressed quit button """
        self.Close(True)

    def save_insert(self, table, dictionary):
        """
        Returns True, if the new_entry could be successfully entered into the database
        """

        try:
            table.insert1(dictionary)
            self.status_text.write('Sucessfully entered new entry: ' + str(dictionary) + '\n')
            return True

        except Exception as ex:
            print('Exception manually caught:', ex)
            self.status_text.write('Error while entering ' + str(dictionary) + ' : ' + str(ex) + '\n')
            return False


# run the GUI
if __name__ == '__main__':
    app = wx.App()
    frame = window(parent=None, id=-1)
    frame.Show()
    app.MainLoop()
