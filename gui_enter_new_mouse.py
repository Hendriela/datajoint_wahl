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

BOX_WIDTH = 170
BOX_HEIGHT = -1                         # -1 makes text boxes automatically one line high
BOX_TITLE_COLOR = (0, 0, 0)             # in RGB

BUTTON_WIDTH = BOX_WIDTH
BUTTON_HEIGHT = 50

# New mouse box
M_LEFT = 30 + 20                        # position of the first element in mouse box
M_TOP = 20 + 30
ROW = 70
COL = 200
M_HEIGHT = 4 * ROW + BUTTON_HEIGHT      # height of mouse box
M_WIDTH = 3 * COL + 20                  # width of mouse box

# New surgery box
S_LEFT = M_LEFT
S_TOP = M_HEIGHT + 70
S_HEIGHT = 3 * ROW + BUTTON_HEIGHT
S_WIDTH = M_WIDTH

# New injection box
I_LEFT = S_LEFT + S_WIDTH + 30
I_TOP = S_TOP
I_HEIGHT = S_HEIGHT
I_WIDTH = S_WIDTH

# New weight box
W_LEFT = I_LEFT
W_TOP = M_TOP
W_HEIGHT = 2*ROW
W_WIDTH = I_WIDTH

# Sacrificed (euthanized) box
E_LEFT = W_LEFT
E_TOP = M_TOP + W_HEIGHT + 10
E_HEIGHT = M_HEIGHT - (W_HEIGHT + 10)
E_WIDTH = W_WIDTH

# Load mouse box
L_LEFT = M_LEFT
L_TOP = S_TOP + 4*ROW
L_HEIGHT = 2*ROW + BUTTON_HEIGHT + 50
L_WIDTH = BUTTON_WIDTH + 50

# WINDOW SIZE
WINDOW_WIDTH = M_LEFT + M_WIDTH + I_WIDTH + 40
WINDOW_HEIGHT = 1100

# Status box
B_TOP = L_TOP

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

        wx.Frame.__init__(self, parent, id, '{} ({}): Enter mouse data into database'.format(inv_fullname, investigator),
                          size=(WINDOW_WIDTH, WINDOW_HEIGHT))
        panel = wx.Panel(self)

        self.job_list = list()  # save jobs in format [ [table, entry_dict, source_path, target_path], [...], ...]
        # =============================================================================
        # Left upper box: Add new mouse (3x3 fields + notes)
        # =============================================================================
        mouse_box = wx.StaticBox(panel, label='NEW MOUSE INFORMATION',
                                 pos=(M_LEFT - 20, M_TOP - 30), size=(M_WIDTH, M_HEIGHT))
        mouse_box.SetForegroundColour(BOX_TITLE_COLOR)

        # Mouse ID (default is highest existing mouse ID + 1)
        wx.StaticText(panel, label="Mouse ID:", pos=(M_LEFT, M_TOP))
        self.mouse_id = wx.TextCtrl(panel, pos=(M_LEFT, M_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.mouse_id.SetValue(str(int(mouse_ids[0]) + 1))

        # Date of birth (default is current day)
        wx.StaticText(panel, label="Date of birth (YYYY-MM-DD):", pos=(M_LEFT + COL, M_TOP))
        self.dob = wx.TextCtrl(panel, pos=(M_LEFT + COL, M_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.dob.SetValue(current_day)

        # Sex (default is 'U', undefined)
        wx.StaticText(panel, label="Sex:", pos=(M_LEFT + 2 * COL, M_TOP))
        self.sex = wx.ComboBox(panel, choices=sexes, style=wx.CB_READONLY,
                               pos=(M_LEFT + 2 * COL, M_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        item = self.sex.FindString(sexes[2])
        self.sex.SetSelection(item)

        # Batch (default is 0, no batch)
        wx.StaticText(panel, label="Batch:", pos=(M_LEFT, M_TOP + ROW))
        self.batch = wx.TextCtrl(panel, pos=(M_LEFT, M_TOP + ROW + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.batch.SetValue('0')

        # Strain (default is 'WT')
        wx.StaticText(panel, label="Strain:", pos=(M_LEFT + COL, M_TOP + ROW))
        self.strain = wx.ComboBox(panel, choices=strains, style=wx.CB_READONLY,
                                  pos=(M_LEFT + COL, M_TOP + ROW + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        item = self.strain.FindString('WT')
        self.strain.SetSelection(item)

        # Genotype (default 'WT')
        wx.StaticText(panel, label="Genotype:", pos=(M_LEFT + 2 * COL, M_TOP + ROW))
        self.genotype = wx.ComboBox(panel, choices=genotypes, style=wx.CB_READONLY,
                                    pos=(M_LEFT + 2 * COL, M_TOP + ROW + 20), size=(BOX_WIDTH, -1))
        item = self.genotype.FindString('WT')
        self.genotype.SetSelection(item)

        # iRats ID (default empty)
        wx.StaticText(panel, label="iRats ID:", pos=(M_LEFT, M_TOP + 2 * ROW))
        self.irats = wx.TextCtrl(panel, pos=(M_LEFT, M_TOP + 2 * ROW + 20), size=(BOX_WIDTH/2, BOX_HEIGHT))
        self.irats.SetValue('')

        # Ear marks (default empty)
        wx.StaticText(panel, label="Ear marks:", pos=(M_LEFT + COL/2, M_TOP + 2*ROW))
        self.ear = wx.TextCtrl(panel, pos=(M_LEFT + COL/2, M_TOP + 2*ROW + 20), size=(BOX_WIDTH/2 - 15, BOX_HEIGHT))
        self.ear.SetValue('')

        # Cage number (default empty)
        wx.StaticText(panel, label="Cage number:", pos=(M_LEFT + COL, M_TOP + 2*ROW))
        self.cage = wx.TextCtrl(panel, pos=(M_LEFT + COL, M_TOP + 2*ROW + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.cage.SetValue('')

        # Licence (default from YAML)
        wx.StaticText(panel, label="Licence:", pos=(M_LEFT + 2*COL, M_TOP + 2*ROW))
        self.licence = wx.ComboBox(panel, choices=licences, style=wx.CB_READONLY,
                                    pos=(M_LEFT + 2*COL, M_TOP + 2*ROW + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        item = self.licence.FindString(default_params['mice']['default_licence'])
        self.licence.SetSelection(item)

        # Notes
        wx.StaticText(panel, label="Notes:", pos=(M_LEFT, M_TOP + 3 * ROW))
        self.mouse_notes = wx.TextCtrl(panel, value="", style=wx.TE_MULTILINE, pos=(M_LEFT, M_TOP + 3 * ROW + 20),
                                       size=(COL + BOX_WIDTH, 50))

        # # Load mouse button
        # self.load_session_button = wx.Button(panel, label="Load session",
        #                                      pos=(M_LEFT + 3 * COL, M_TOP),
        #                                      size=(BUTTON_WIDTH, BUTTON_HEIGHT))
        # self.Bind(wx.EVT_BUTTON, self.event_load_session, self.load_session_button)

        # Submit mouse button
        self.submit_surgery_button = wx.Button(panel, label="Add new mouse",
                                               pos=(M_LEFT + 2 * COL, M_TOP + 3 * ROW + 20),
                                               size=(BUTTON_WIDTH, BUTTON_HEIGHT))
        self.Bind(wx.EVT_BUTTON, self.event_submit_mouse, self.submit_surgery_button)

        # =============================================================================
        # Left lower box: Enter new surgery (2x3 fields + notes)
        # =============================================================================

        surg_box = wx.StaticBox(panel, label='NEW SURGERY', pos=(S_LEFT - 20, S_TOP - 30), size=(S_WIDTH, S_HEIGHT))
        surg_box.SetForegroundColour(BOX_TITLE_COLOR)

        # Surgery number (default is empty, will be filled when a mouse is loaded)
        wx.StaticText(panel, label="Surgery number:", pos=(S_LEFT, S_TOP))
        self.surg_num = wx.TextCtrl(panel, pos=(S_LEFT, S_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.surg_num.SetValue('')

        # Date and time of surgery (default is current day, 12 pm)
        wx.StaticText(panel, label="Datetime (YYYY-MM-DD HH:MM):", pos=(S_LEFT + COL, S_TOP))
        self.dos = wx.TextCtrl(panel, pos=(S_LEFT + COL, S_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.dos.SetValue(current_day + " 12:00")

        # Surgery type (default from YAML)
        wx.StaticText(panel, label="Surgery type:", pos=(S_LEFT + 2 * COL, S_TOP))
        self.type = wx.TextCtrl(panel, pos=(S_LEFT + 2 * COL, S_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.type.SetValue(default_params['mice']['default_surgery_type'])

        # Anesthesia (default from YAML)
        wx.StaticText(panel, label="Anesthesia:", pos=(S_LEFT, S_TOP + ROW))
        self.anesthesia = wx.TextCtrl(panel, pos=(S_LEFT, S_TOP + ROW + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.anesthesia.SetValue(default_params['mice']['default_anesthesia'])

        # Pre-OP weight in grams (default is empty)
        wx.StaticText(panel, label="Pre-OP weight (grams):", pos=(S_LEFT + COL, S_TOP + ROW))
        self.pre_op_weight = wx.TextCtrl(panel, pos=(S_LEFT + COL, S_TOP + ROW + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.pre_op_weight.SetValue('')

        # Duration
        wx.StaticText(panel, label="Duration of surgery (min):", pos=(S_LEFT + 2 * COL, S_TOP + ROW))
        self.duration = wx.TextCtrl(panel, pos=(S_LEFT + 2 * COL, S_TOP + ROW + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.duration.SetValue('')

        # Notes
        wx.StaticText(panel, label="Notes:", pos=(S_LEFT, S_TOP + 2 * ROW))
        self.surgery_notes = wx.TextCtrl(panel, value="", style=wx.TE_MULTILINE, pos=(S_LEFT, S_TOP + 2 * ROW + 20),
                                         size=(COL + BOX_WIDTH, 50))

        # Submit surgery button
        self.submit_surgery_button = wx.Button(panel, label="Add new surgery",
                                               pos=(S_LEFT + 2 * COL, S_TOP + 2 * ROW + 20),
                                               size=(BUTTON_WIDTH, BUTTON_HEIGHT))
        self.Bind(wx.EVT_BUTTON, self.event_submit_surgery, self.submit_surgery_button)

        # =============================================================================
        # Right lower box: Enter new injection (2x3 fields + notes)
        # =============================================================================

        inj_box = wx.StaticBox(panel, label='NEW INJECTION', pos=(I_LEFT - 20, I_TOP - 30), size=(I_WIDTH, I_HEIGHT))
        inj_box.SetForegroundColour(BOX_TITLE_COLOR)

        # Injection number (default is empty, will be filled after mouse is loaded)
        wx.StaticText(panel, label="Injection number:", pos=(I_LEFT, I_TOP))
        self.inj_num = wx.TextCtrl(panel, pos=(I_LEFT, I_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.inj_num.SetValue('')

        # Substance (default from YAML)
        wx.StaticText(panel, label="Substance:", pos=(I_LEFT + COL, I_TOP))
        self.substance = wx.ComboBox(panel, choices=substances, style=wx.CB_READONLY,
                                     pos=(I_LEFT + COL, I_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        item = self.substance.FindString(default_params['mice']['default_substance'])
        self.substance.SetSelection(item)

        # Volume (default 0.3 uL)
        wx.StaticText(panel, label="Injected volume (\u03BCL):", pos=(I_LEFT + 2 * COL, I_TOP))
        self.volume = wx.TextCtrl(panel, pos=(I_LEFT + 2 * COL, I_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.volume.SetValue(default_params['mice']['default_volume'])

        # Site (default from YAML)
        wx.StaticText(panel, label="Injection site:", pos=(I_LEFT, I_TOP + ROW))
        self.site = wx.TextCtrl(panel, pos=(I_LEFT, I_TOP + ROW + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.site.SetValue(default_params['mice']['default_site'])

        # Coordinates (default empty)
        wx.StaticText(panel, label="Stereotaxic coordinates:", pos=(I_LEFT + COL, I_TOP + ROW))
        self.coordinates = wx.TextCtrl(panel, pos=(I_LEFT + COL, I_TOP + ROW + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.coordinates.SetValue(default_params['mice']['default_coords'])

        # Dilution
        wx.StaticText(panel, label="Dilution:", pos=(I_LEFT + 2 * COL, I_TOP + ROW))
        self.dilution = wx.TextCtrl(panel, pos=(I_LEFT + 2 * COL, I_TOP + ROW + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.dilution.SetValue(default_params['mice']['default_dilution'])

        # Notes
        wx.StaticText(panel, label="Notes:", pos=(I_LEFT, I_TOP + 2 * ROW))
        self.inj_notes = wx.TextCtrl(panel, value="", style=wx.TE_MULTILINE, pos=(I_LEFT, I_TOP + 2 * ROW + 20),
                                     size=(COL + BOX_WIDTH, 50))

        # Submit injection button
        self.submit_injection_button = wx.Button(panel, label="Add new Injection",
                                               pos=(I_LEFT + 2 * COL, I_TOP + 2 * ROW + 20),
                                               size=(BUTTON_WIDTH, BUTTON_HEIGHT))
        self.Bind(wx.EVT_BUTTON, self.event_submit_injection, self.submit_injection_button)

        # =============================================================================
        # Right upper box: Enter new weight (1x2 fields)
        # =============================================================================

        weight_box = wx.StaticBox(panel, label='NEW WEIGHT', pos=(W_LEFT - 20, W_TOP - 30), size=(W_WIDTH, W_HEIGHT))
        weight_box.SetForegroundColour(BOX_TITLE_COLOR)

        # Date of weight (default is current day)
        wx.StaticText(panel, label="Date (YYYY-MM-DD):", pos=(W_LEFT, W_TOP))
        self.dow = wx.TextCtrl(panel, pos=(W_LEFT, W_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.dow.SetValue(current_day)

        # Weight in grams (default is empty)
        wx.StaticText(panel, label="Weight (grams):", pos=(W_LEFT + COL, W_TOP))
        self.weight = wx.TextCtrl(panel, pos=(W_LEFT + COL, W_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.weight.SetValue('')

        # Submit weight button
        self.submit_weight_button = wx.Button(panel, label="Add new weight",
                                                 pos=(W_LEFT + 2 * COL, W_TOP),
                                                 size=(BUTTON_WIDTH, BUTTON_HEIGHT-5))
        self.Bind(wx.EVT_BUTTON, self.event_submit_weight, self.submit_weight_button)

        # =============================================================================
        # Right middle box: Enter new sacrificed mouse (1x2 fields + notes)
        # =============================================================================

        sac_box = wx.StaticBox(panel, label='MOUSE EUTHANIZED', pos=(E_LEFT - 20, E_TOP - 30), size=(E_WIDTH, E_HEIGHT))
        sac_box.SetForegroundColour(BOX_TITLE_COLOR)

        # Date of weight (default is current day)
        wx.StaticText(panel, label="Date (YYYY-MM-DD):", pos=(E_LEFT, E_TOP))
        self.doe = wx.TextCtrl(panel, pos=(E_LEFT, E_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.doe.SetValue(current_day)

        # Perfused check box (default is unchecked)
        wx.StaticText(panel, label="Brain kept for storage?", pos=(E_LEFT + COL, E_TOP))
        self.perfused = wx.CheckBox(panel, label='Perfused', pos=(E_LEFT + COL, E_TOP + 20), size=(130, 20))

        # Reason
        wx.StaticText(panel, label="Reason:", pos=(E_LEFT, E_TOP + ROW - 10))
        self.reason = wx.TextCtrl(panel, value="End of experiment", style=wx.TE_MULTILINE,
                                  pos=(E_LEFT, E_TOP + ROW + 10), size=(COL + BOX_WIDTH, 50))

        # Submit euthanasia button
        self.submit_euthanasia_button = wx.Button(panel, label="Add euthanasia",
                                                 pos=(E_LEFT + 2 * COL, E_TOP),
                                                 size=(BUTTON_WIDTH, BUTTON_HEIGHT-5))
        self.Bind(wx.EVT_BUTTON, self.event_submit_euthanasia, self.submit_euthanasia_button)

        # =============================================================================
        # Bottom left box: Load mouse
        # =============================================================================

        load_box = wx.StaticBox(panel, label='LOAD MOUSE', pos=(L_LEFT - 20, L_TOP - 30), size=(L_WIDTH, L_HEIGHT))
        load_box.SetForegroundColour(BOX_TITLE_COLOR)

        wx.StaticText(panel, label="Mouse ID:", pos=(L_LEFT, L_TOP))
        self.new_mouse = wx.ComboBox(panel, choices=mouse_ids, style=wx.CB_READONLY,
                                 pos=(L_LEFT, L_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        item = self.new_mouse.FindString(default_params['behavior']['default_mouse'])
        self.new_mouse.SetSelection(item)

        # Load mouse button
        self.load_mouse_button = wx.Button(panel, label="Load mouse",
                                                 pos=(L_LEFT, L_TOP + ROW),
                                                 size=(BUTTON_WIDTH, BUTTON_HEIGHT))
        self.Bind(wx.EVT_BUTTON, self.event_load_mouse, self.load_mouse_button)

        # Currently loaded mouse field
        curr_mouse_title = wx.StaticText(panel, label="CURRENTLY LOADED MOUSE:", pos=(L_LEFT, L_TOP + 2*ROW))
        curr_mouse_title.SetForegroundColour((255, 0, 0))
        self.curr_mouse = wx.TextCtrl(panel, pos=(L_LEFT, L_TOP + 2*ROW + 20), style=wx.TE_READONLY | wx.TE_CENTER,
                                      size=(BOX_WIDTH, 30))
        self.curr_mouse.SetFont(wx.Font(16, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        self.curr_mouse.SetBackgroundColour((255, 70, 70))
        self.curr_mouse.SetValue('None')

        # =============================================================================
        # Submit and close buttons
        # =============================================================================

        self.quit_button = wx.Button(panel, label="Quit",
                                     pos=(M_LEFT, B_TOP + L_HEIGHT),
                                     size=(BUTTON_WIDTH, BUTTON_HEIGHT))
        self.Bind(wx.EVT_BUTTON, self.event_quit_button, self.quit_button)

        # status text
        self.status_text = wx.TextCtrl(panel, value="Status updates will appear here:\n",
                                       style=wx.TE_MULTILINE,
                                       pos=(S_LEFT + COL + 30, B_TOP - 23),
                                       size=(WINDOW_WIDTH - S_LEFT - COL - 60, WINDOW_HEIGHT - S_TOP - S_HEIGHT - ROW))

    # =============================================================================
    # Events for menus and button presses
    # =============================================================================

    def event_submit_mouse(self, event):
        """Handle user click on "Submit Mouse" button"""

        # create database entry
        mouse_dict = dict(username=investigator,
                          mouse_id=self.mouse_id.GetValue(),
                          dob=self.dob.GetValue(),
                          sex=self.sex.GetValue(),
                          batch=self.batch.GetValue(),
                          strain=self.strain.GetValue(),
                          genotype=self.genotype.GetValue(),
                          irats_id=self.irats.GetValue(),
                          cage_num=self.cage.GetValue(),
                          ear_mark=self.ear.GetValue(),
                          licence_id=self.licence.GetValue(),
                          info=self.mouse_notes.GetValue(),
                          )
        try:
            # insert mouse into database
            common_mice.Mouse().insert1(mouse_dict)
            self.status_text.write('Successfully entered new entry: ' + str(mouse_dict) + '\n')

            # save dictionary that is entered in a backup YAML file for faster re-population
            identifier = "%s_%s_%s" % (investigator, self.mouse_id.GetValue(), current_day)
            filename = os.path.join(login.get_neurophys_wahl_directory(), "Datajoint/manual_mouse_submissions", identifier + '.yaml')
            with open(filename, 'w') as outfile:
                yaml.dump(mouse_dict, outfile, default_flow_style=False)
            self.status_text.write('Created backup file at %s' % filename + '\n')

        except Exception as ex:
            print('Exception manually caught:', ex)
            self.status_text.write('Error while entering ' + str(mouse_dict) + ' : ' + str(ex) + '\n')

    def event_submit_surgery(self, event):
        """Enter surgery data as new entry into the Surgery table"""
        pass

    def event_submit_injection(self, event):
        """Enter new injection for the currently selected surgery"""
        pass

    def event_submit_weight(self, event):
        """Enter a new weight for the currently selected mouse"""
        pass

    def event_submit_euthanasia(self, event):
        """Move the currently selected mouse to the Sacrificed table"""
        pass

    def event_load_mouse(self, event):
        """Load data of an already existing mouse to add surgeries/weights/euthanasia"""

        mouse_dict = dict(username=investigator,
                          mouse_id=self.new_mouse.GetValue())
        entries = (common_mice.Mouse() & mouse_dict).fetch(as_dict=True)
        # check there is only one table corresponding to this
        if len(entries) != 1:
            self.status_text.write(
                'Can not load mouse info for {} because there are {} mice corresponding to this'.format(
                    mouse_dict, len(entries)) + '\n')
            return

        entry = entries[0]

        # set the selections in the menus according to the loaded info
        self.mouse_id.SetValue(entry['mouse_id'])
        self.dob.SetValue(entry['dob'].strftime('%Y-%m-%d'))
        item = self.sex.FindString(entry['sex'])
        self.sex.SetSelection(item)
        self.batch.SetValue(str(entry['batch']))
        item = self.strain.FindString(entry['strain'])
        self.strain.SetSelection(item)
        item = self.genotype.FindString(entry['genotype'])
        self.genotype.SetSelection(item)
        self.irats.SetValue(entry['irats_id'])
        self.ear.SetValue(entry['ear_mark'])
        self.cage.SetValue(str(entry['cage_num']))
        item = self.licence.FindString(entry['licence_id'])
        self.licence.SetSelection(item)
        self.mouse_notes.SetValue(entry['info'])

        self.status_text.write(
            'Successfully loaded info for mouse {}. You can now add new data associated with this mouse. \n\t'
            '--> Do not change data in the MOUSE box while adding new info!'.format(mouse_dict) + '\n')


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
            file = os.path.join(login.get_neurophys_wahl_directory(), 'REL_BACKUP_PATH', identifier + '.yaml')
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