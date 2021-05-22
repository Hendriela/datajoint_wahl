#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" GUI to enter new mice or mouse-related data (new weights, new surgeries...)
Created on Sat May 15 22:54:12 2021

@author: hheise
Installation of wxpython in Ubuntu 18.04 (pip install did not work):
conda install -c anaconda wxpython
pip install datajoint
"""

import sys

sys.path.append("..")  # Adds higher directory to python modules path.

import wx
import login
import datajoint as dj
from datetime import datetime
import os
import glob
import numpy as np
import yaml
from pathlib import Path

# connect to datajoint database
login.connect()
from schema import common_mice, common_exp, common_behav  # , common_img, common_el

# =============================================================================
# HARDCODED PARAMETER FOR GUI
# =============================================================================

WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 1100

WINDOW_WIDTH_L = 900

BUTTON_WIDTH = 130
BUTTON_HEIGHT = 40

# New mouse box
M_LEFT = 30 + 20  # position of the first element in mouse box
M_TOP = 20 + 30
ROW = 70
COL = 200
M_HEIGHT = 330  # height of mouse box

# New surgery box
S_LEFT = M_LEFT + 600
S_TOP = M_TOP
S_HEIGHT = 260

# New injection box
I_LEFT = S_LEFT
I_TOP = S_TOP + 350
I_HEIGHT = 100

# New weight box
W_LEFT = WINDOW_WIDTH_L - 20
W_TOP = M_TOP

# Sacrificed (euthanized) box
E_LEFT = W_LEFT
E_TOP = M_TOP + 350

# Status box
B_TOP = E_TOP + 350

# relative path to manual_submission backup folder inside the Neurophysiology Wahl directory (common for all users)
REL_BACKUP_PATH_MOUSE = "Datajoint/manual_mouse_submissions"
REL_BACKUP_PATH_SURGERY = "Datajoint/manual_surgery_submissions"

# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

# Load YAML file (has to be present in the same folder
with open(r'gui_params.yaml') as file:
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
raw_mouse_ids = (common_mice.Mouse() & username_filter) - common_mice.Sacrificed()
mouse_ids = raw_mouse_ids.fetch('mouse_id', order_by='mouse_id DESC')
if default_params['behavior']['default_mouse'] == 'last_mouse':
    default_params['behavior']['default_mouse'] = mouse_ids[0]



# =============================================================================
# Load options for drop down menus
# =============================================================================

investigator = default_params['username']
inv_fullname = (common_mice.Investigator() & "username = '{}'".format(investigator)).fetch1('full_name')

# Mouse info
sexes = ['M', 'F', 'U']
strains = common_mice.Strain().fetch('strain')
genotypes = ['WT', 'n.d.', '+/+', '+/-', '-/-']
licences = common_mice.Licence().fetch('licence_id')

# Surgery info
substances = common_mice.Substance().fetch('substance_name')

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
        # Left box: Add new mouse (3x3 fields + notes)
        # =============================================================================
        wx.StaticBox(panel, label='NEW MOUSE INFORMATION',
                     pos=(M_LEFT - 20, M_TOP - 30), size=(WINDOW_WIDTH_L - 2 * M_LEFT, M_HEIGHT))

        # Date of birth (default is current day)
        wx.StaticText(panel, label="Date of birth (YYYY-MM-DD):", pos=(M_LEFT, M_TOP))
        self.dob = wx.TextCtrl(panel, pos=(M_LEFT, M_TOP + 20), size=(170, -1))
        self.dob.SetValue(current_day)

        # Sex (default is 'U', undefined)
        wx.StaticText(panel, label="Sex:", pos=(M_LEFT + COL, M_TOP))
        self.sex = wx.ComboBox(panel, choices=sexes, style=wx.CB_READONLY,
                               pos=(M_LEFT + COL, M_TOP+ 20), size=(170, -1))
        self.sex.SetSelection(sexes[2])

        # Batch (default is 0, no batch)
        wx.StaticText(panel, label="Batch:", pos=(M_LEFT + 2*COL, M_TOP))
        self.batch = wx.TextCtrl(panel, pos=(M_LEFT + 2*COL, M_TOP + 20), size=(170, -1))
        self.batch.SetValue('0')

        # Strain (default is 'WT')
        wx.StaticText(panel, label="Strain:", pos=(M_LEFT, M_TOP + ROW))
        self.strain = wx.ComboBox(panel, choices=strains, style=wx.CB_READONLY,
                                  pos=(M_LEFT, M_TOP + ROW + 20), size=(170, -1))
        self.strain.SetSelection('WT')

        # Genotype (default 'WT')
        self.genotype = wx.Button(panel, label="Genotype:", pos=(M_LEFT + COL, M_TOP + ROW), size=(100, 25))
        self.genotype = wx.ComboBox(panel, choices=strains, style=wx.CB_READONLY, pos=(M_LEFT + COL, M_TOP + ROW + 20),
                                    size=(170, -1))
        self.genotype.SetSelection('WT')

        # iRats ID (default empty)
        wx.StaticText(panel, label="iRats ID:", pos=(M_LEFT + 2*COL, M_TOP + ROW))
        self.irats = wx.TextCtrl(panel, pos=(M_LEFT + 2*COL, M_TOP + ROW + 20), size=(170, -1))
        self.irats.SetValue('')

        # Cage number (default empty)
        wx.StaticText(panel, label="Cage number:", pos=(M_LEFT, M_TOP + 2*ROW))
        self.cage = wx.TextCtrl(panel, pos=(M_LEFT, M_TOP + 2*ROW + 20), size=(170, -1))
        self.cage.SetValue('')

        # Cage number
        wx.StaticText(panel, label="Ear marks:", pos=(M_LEFT + COL, M_TOP + 2*ROW))
        self.ear = wx.TextCtrl(panel, pos=(M_LEFT + COL, M_TOP + 2*ROW + 20), size=(170, -1))
        self.ear.SetValue('')

        # Licence (default from YAML)
        self.licence = wx.Button(panel, label="Licence:", pos=(M_LEFT + 2*COL, M_TOP + 2*ROW), size=(100, 25))
        self.licence = wx.ComboBox(panel, choices=licences, style=wx.CB_READONLY,
                                    pos=(M_LEFT + 2*COL, M_TOP + 2*ROW + 20), size=(170, -1))
        self.licence.SetSelection(default_params['mice']['default_licence'])

        # Notes
        wx.StaticText(panel, label="Notes:", pos=(M_LEFT, M_TOP + 3 * ROW))
        self.mouse_notes = wx.TextCtrl(panel, value="", style=wx.TE_MULTILINE, pos=(M_LEFT, M_TOP + 3 * ROW + 20),
                                       size=(WINDOW_WIDTH_L - 3 * M_LEFT - 200, 50))

        # Load mouse button
        self.load_session_button = wx.Button(panel, label="Load session",
                                             pos=(M_LEFT + 3 * COL, M_TOP),
                                             size=(150, 50))
        self.Bind(wx.EVT_BUTTON, self.event_load_session, self.load_session_button)

        # Submit mouse button
        self.submit_surgery_button = wx.Button(panel, label="Add new mouse",
                                               pos=(M_LEFT + 3 * COL, M_TOP + 3 * ROW + 20),
                                               size=(150, 50))
        self.Bind(wx.EVT_BUTTON, self.event_submit_mouse, self.submit_surgery_button)

        # =============================================================================
        # Middle upper box: Enter new surgery
        # =============================================================================

        wx.StaticBox(panel, label='NEW SURGERY',
                     pos=(S_LEFT - 20, S_TOP - 30), size=(WINDOW_WIDTH_L - 2 * S_LEFT, S_HEIGHT))

        # Surgery number (default is max surgery num for the current mouse + 1)
        wx.StaticText(panel, label="Surgery number:", pos=(S_LEFT, S_TOP))
        self.surg_num = wx.TextCtrl(panel, pos=(S_LEFT, S_TOP + 20), size=(170, -1))
        self.surg_num.SetValue(next_surgery_num)

        # Date and time of surgery (default is current day, 12 pm)
        wx.StaticText(panel, label="Date and time of surgery (YYYY-MM-DD HH:MM):", pos=(S_LEFT + COL, S_TOP))
        self.dos = wx.TextCtrl(panel, pos=(S_LEFT + COL, S_TOP + 20), size=(170, -1))
        self.dos.SetValue(current_day + " 12:00")

        # Surgery type (default from YAML)
        wx.StaticText(panel, label="Surgery type:", pos=(S_LEFT + 2 * COL, S_TOP))
        self.type = wx.TextCtrl(panel, pos=(S_LEFT + 2 * COL, S_TOP + 20), size=(170, -1))
        self.type.SetValue(default_params['mice']['default_surgery_type'])

        # Anesthesia (default is 'WT')
        wx.StaticText(panel, label="Anesthesia:", pos=(S_LEFT, S_TOP + ROW))
        self.anesthesia = wx.TextCtrl(panel, pos=(S_LEFT, S_TOP + ROW + 20), size=(170, -1))
        self.anesthesia.SetValue(default_params['mice']['default_anesthesia'])

        # Pre-OP weight in grams (default is empty)
        wx.StaticText(panel, label="Pre-OP weight (grams):", pos=(S_LEFT + COL, S_TOP + ROW))
        self.pre_op_weight = wx.TextCtrl(panel, pos=(S_LEFT + COL, S_TOP + ROW + 20), size=(170, -1))
        self.pre_op_weight.SetValue('')

        # Duration
        wx.StaticText(panel, label="Duration of surgery (min):", pos=(S_LEFT + 2 * COL, S_TOP + ROW))
        self.duration = wx.TextCtrl(panel, pos=(S_LEFT + 2 * COL, S_TOP + ROW + 20), size=(170, -1))
        self.duration.SetValue('')

        # Notes
        wx.StaticText(panel, label="Notes:", pos=(S_LEFT, S_TOP + 2 * ROW))
        self.surgery_notes = wx.TextCtrl(panel, value="", style=wx.TE_MULTILINE, pos=(S_LEFT, S_TOP + 2 * ROW + 20),
                                         size=(WINDOW_WIDTH_L - 2 * S_LEFT - 200, 50))

        # Submit surgery button
        self.submit_surgery_button = wx.Button(panel, label="Add new surgery",
                                               pos=(S_LEFT + 3 * COL, S_TOP + 3 * ROW + 20),
                                               size=(150, 50))
        self.Bind(wx.EVT_BUTTON, self.event_submit_surgery, self.submit_surgery_button)

        # =============================================================================
        # Middle lower box: Enter new injection
        # =============================================================================

        wx.StaticBox(panel, label='NEW INJECTION',
                     pos=(I_LEFT - 20, I_TOP - 30), size=(WINDOW_WIDTH_L - 2 * I_LEFT, I_HEIGHT))

        # Injection number (default is 0)
        wx.StaticText(panel, label="Injection number:", pos=(I_LEFT, I_TOP))
        self.inj_num = wx.TextCtrl(panel, pos=(S_LEFT, S_TOP + 20), size=(170, -1))
        self.inj_num.SetValue(next_injection_num)

        # Substance (default from YAML)
        wx.StaticText(panel, label="Substance:", pos=(I_LEFT + COL, I_TOP))
        self.substance = wx.ComboBox(panel, choices=licences, style=wx.CB_READONLY,
                                     pos=(I_LEFT + COL, I_TOP + 20), size=(170, -1))
        self.substance.SetSelection(default_params['mice']['default_substance'])

        # Volume (default 0.3 uL)
        wx.StaticText(panel, label="Injected volume:", pos=(S_LEFT + 2 * COL, S_TOP))
        self.volume = wx.TextCtrl(panel, pos=(S_LEFT + 2 * COL, S_TOP + 20), size=(170, -1))
        self.volume.SetValue('0.3')

        # Site (default from YAML)
        wx.StaticText(panel, label="Injection site:", pos=(S_LEFT, S_TOP + ROW))
        self.site = wx.TextCtrl(panel, pos=(S_LEFT, S_TOP + ROW + 20), size=(170, -1))
        self.site.SetValue(default_params['mice']['default_site'])

        # Coordinates (default empty)
        wx.StaticText(panel, label="Stereotaxic coordinates:", pos=(S_LEFT + COL, S_TOP + ROW))
        self.coordinates = wx.TextCtrl(panel, pos=(S_LEFT + COL, S_TOP + ROW + 20), size=(170, -1))
        self.coordinates.SetValue('')

        # Dilution
        wx.StaticText(panel, label="Dilution:", pos=(S_LEFT + 2 * COL, S_TOP + ROW))
        self.dilution = wx.TextCtrl(panel, pos=(S_LEFT + 2 * COL, S_TOP + ROW + 20), size=(170, -1))
        self.dilution.SetValue('')

        # Notes
        wx.StaticText(panel, label="Notes:", pos=(S_LEFT, S_TOP + 2 * ROW))
        self.inj_notes = wx.TextCtrl(panel, value="", style=wx.TE_MULTILINE, pos=(S_LEFT, S_TOP + 2 * ROW + 20),
                                       size=(WINDOW_WIDTH_L - 2 * S_LEFT - 200, 50))

        # Submit surgery button
        self.submit_injection_button = wx.Button(panel, label="Add new Injection",
                                               pos=(S_LEFT + 3 * COL, S_TOP + 3 * ROW + 20),
                                               size=(150, 50))
        self.Bind(wx.EVT_BUTTON, self.event_submit_injection, self.submit_injection_button)

        # =============================================================================
        # Submit and close buttons
        # =============================================================================

        self.quit_button = wx.Button(panel, label="Quit",
                                     pos=(30, L_TOP + ROW),
                                     size=(BUTTON_WIDTH, BUTTON_HEIGHT))
        self.Bind(wx.EVT_BUTTON, self.event_quit_button, self.quit_button)

        # status text
        self.status_text = wx.TextCtrl(panel, value="Status updates will appear here:\n",
                                       style=wx.TE_MULTILINE,
                                       pos=(S_LEFT + COL, L_TOP),
                                       size=(WINDOW_WIDTH - S_LEFT - COL - 30, WINDOW_HEIGHT - L_TOP - 30))

    # =============================================================================
    # Events for menus and button presses
    # =============================================================================

    def event_submit_mouse(self, event):
        pass

    def event_submit_surgery(self, event):
        pass

    def event_submit_injection(self, event):
        pass

    def event_submit_session(self, event):
        """ The user clicked on the button to submit a session """

        # create session dictionary that can be entered into datajoint pipeline
        session_dict = dict(username=investigator,
                            mouse_id=self.mouse_name.GetValue(),
                            day=self.day.GetValue(),
                            trial=int(self.trial.GetValue()),
                            path=Path(self.session_folder.GetValue()),
                            anesthesia=self.anesthesia.GetValue(),
                            setup=self.setup.GetValue(),
                            task=self.task.GetValue(),
                            stage=int(self.stage.GetValue()),
                            experimenter=self.experimenter.GetValue(),
                            notes=self.notes.GetValue()
                            )

        # check if the session is already in the database (most common error)
        key = dict(username=investigator,
                   mouse_id=self.mouse_name.GetValue(),
                   day=self.day.GetValue(),
                   trial=int(self.trial.GetValue()))
        if len(common_exp.Session() & key) > 0:
            message = 'The session you wanted to enter into the database already exists.\n' + \
                      'Therefore, nothing was entered into the database.'
            wx.MessageBox(message, caption="Session already in database", style=wx.OK | wx.ICON_INFORMATION)
            return

        # add entry to database
        try:
            common_exp.Session().helper_insert1(session_dict)
            self.status_text.write('Sucessfully entered new session: ' + str(key) + '\n')

            # save dictionary that is entered in a backup YAML file for faster re-population
            identifier = common_exp.Session().create_id(investigator_name=investigator,
                                                        mouse_id=session_dict['mouse_id'],
                                                        date=session_dict['day'],
                                                        trial=session_dict['trial'])
            file = os.path.join(login.get_neurophys_wahl_directory(), REL_BACKUP_PATH, identifier + '.yaml')
            # TODO show prompt if a backup file with this identifier already exists and ask the user to overwrite
            # if os.path.isfile(file):
            #     message = 'The backup file of the session you wanted to enter into the database with the unique identifier ' \
            #               '{} already exists.\nTherefore, nothing was entered into the database.'.format(identifier)
            #     wx.MessageBox(message, caption="Backup file already exists", style=wx.OK | wx.ICON_INFORMATION)
            #     return
            # else:

            # Transform session path from Path to string (with universal / separator) to make it YAML-compatible
            session_dict['path'] = str(session_dict['path']).replace("\\", "/")
            with open(file, 'w') as outfile:
                yaml.dump(session_dict, outfile, default_flow_style=False)

        except Exception as ex:
            print('Exception manually caught:', ex)
            self.status_text.write('Error: ' + str(ex) + '\n')

    def event_load_session(self, event):
        """ User wants to load additional information about session into GUI """

        session_dict = dict(username=investigator,
                            mouse_id=self.mouse_name.GetValue(),
                            day=self.day.GetValue(),
                            trial=int(self.trial.GetValue()))
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
        self.stage.SetValue(str(entry['stage']))
        item = self.anesthesia.FindString(entry['anesthesia'])
        self.anesthesia.SetSelection(item)
        item = self.experimenter.FindString(entry['experimenter'])
        self.experimenter.SetSelection(item)
        self.notes.SetValue(entry['notes'])

    def event_submit_behavior(self, event):
        """ User clicked on button to submit the behavioral data """

        # go through all behavioral files and upload if checkbox is set
        session_key = dict(name=self.mouse_name.GetValue(),
                           day=self.day.GetValue(),
                           trial=int(self.trial.GetValue()))
        trial = int(self.trial.GetValue())
        # wheel
        if self.wheel_checkbox.GetValue():
            # insert the main table
            wheel_dict = dict(**session_key,
                              wheel_type=self.wheel.GetValue())
            self.save_insert(common_behav.Wheel(), wheel_dict)

            # add job to transfer file later
            file = default_params['behavior']['wheel_file'].format(trial)
            raw_wheel_dict = dict(**session_key,
                                  file_name=file)

            self.job_list.append([common_behav.RawWheelFile(), raw_wheel_dict, self.behav_folder.GetValue(), file])
        # synchronization
        if self.sync_checkbox.GetValue():
            sync_dict = dict(**session_key,
                             sync_type=self.imaging_sync.GetValue())
            self.save_insert(common_behav.Synchronization(), sync_dict)

            file = default_params['behavior']['synch_file'].format(trial)
            raw_sync_dict = dict(**session_key,
                                 file_name=file)
            self.job_list.append(
                [common_behav.RawSynchronizationFile(), raw_sync_dict, self.behav_folder.GetValue(), file])
        # video
        if self.video_checkbox.GetValue():
            video_dict = dict(**session_key,
                              camera_nr=0,  # TODO: modify for multiple cameras
                              camera_position=self.camera_pos.GetValue(),
                              frame_rate=int(self.frame_rate.GetValue()))
            self.save_insert(common_behav.Video(), video_dict)

            file = self.video_file.GetValue().format(trial)
            raw_video_dict = dict(**session_key,
                                  camera_nr=0,
                                  part=0,
                                  file_name=file)
            self.job_list.append([common_behav.RawVideoFile(), raw_video_dict, self.behav_folder.GetValue(), file])
        # whisker stimulator
        if self.whisker_checkbox.GetValue():
            whisker_dict = dict(**session_key,
                                stimulator_type=self.stimulator.GetValue())
            self.save_insert(common_behav.WhiskerStimulator(), whisker_dict)

            file = default_params['behavior']['whisker_file'].format(trial)
            raw_whisker_dict = dict(**session_key,
                                    file_name=file)
            self.job_list.append(
                [common_behav.RawWhiskerStimulatorFile(), raw_whisker_dict, self.behav_folder.GetValue(), file])
        # events
        if self.event_checkbox.GetValue():
            event_dict = dict(**session_key,
                              sensory_event_type=self.event_type.GetValue())
            self.save_insert(common_behav.SensoryEvents(), event_dict)

            file = default_params['behavior']['event_file'].format(trial)
            raw_event_dict = dict(**session_key,
                                  file_name=file)
            self.job_list.append(
                [common_behav.RawSensoryEventsFile(), raw_event_dict, self.behav_folder.GetValue(), file])


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

    def event_select_behav_folder(self, event):
        """ User clicked on select behavior folder button """

        # open file dialog
        folder_dialog = wx.DirDialog(parent=None, message="Choose directory of behavioral data",
                                     defaultPath=default_params['paths']['default_behav_folder'],
                                     style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        exit_flag = folder_dialog.ShowModal()
        # update the folder in the text box when the user exited with ok
        if exit_flag == wx.ID_OK:
            self.behav_folder.SetValue(folder_dialog.GetPath())

    def event_select_img_folder(self, event):
        """ User clicked on select imaging folder button """

        # open file dialog
        folder_dialog = wx.DirDialog(parent=None, message="Choose directory of imaging data",
                                     defaultPath=default_params['paths']['default_img_folder'],
                                     style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        exit_flag = folder_dialog.ShowModal()
        # update the folder in the text box when the user exited with ok
        if exit_flag == wx.ID_OK:
            self.img_folder.SetValue(folder_dialog.GetPath())

    def event_select_el_folder(self, event):
        """ User clicked on select electrical folder button """

        # open file dialog
        folder_dialog = wx.DirDialog(parent=None, message="Choose directory of electrical recording data",
                                     defaultPath=default_params['paths']['default_el_folder'],
                                     style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        exit_flag = folder_dialog.ShowModal()
        # update the folder in the text box when the user exited with ok
        if exit_flag == wx.ID_OK:
            self.el_folder.SetValue(folder_dialog.GetPath())

    def event_quit_button(self, event):
        """ User pressed quit button """
        self.Close(True)

    def event_transfer_data(self, event):
        """ Transfer data to the neurophysiology server and add Raw...File to database """

        jobs = self.job_list
        print('Transfering the files for {} entries in the job list.'.format(len(jobs)))

        while jobs:  # do as long as the list is not empty
            job = jobs.pop(0)  # get first job (FIFO)

            # extract variables for better code readablility
            table = job[0]
            new_entry = job[1]
            source_folder = job[2]
            file_name = job[3]

            # enter Raw...File into database
            successful = self.save_insert(table, new_entry)

            if not successful:
                continue  # jump to next job and do not transfer the data for this

            # transfer file to neurophysiology folder
            source_path = os.path.join(source_folder, file_name)

            folder_id = common_exp.Session().create_id(new_entry['name'], new_entry['day'], new_entry['trial'])
            target_folder = os.path.join(path_neurophys_data, folder_id)

            target_path = os.path.join(target_folder, file_name)

            # create directory if it does not exists
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            # print the size of the file if it is large and will take a while
            size = os.path.getsize(source_path)
            if size > 1e7:  # 10 MB
                print('Copying large file of size {:.2f}GB...'.format(size / 1e9) + '\n')
                print('File:', source_path)

            # copy the file to the neurophysiology directory
            if os.name == 'nt':  # windows pc
                os.system('copy "{}" "{}" '.format(source_path, target_path))
            else:  # assuming linux
                os.system('cp "{}" "{}" '.format(source_path, target_path))

            print('Copied file ' + source_path + '\n')

        print('Transfer ended.')

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