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
from schema import common_mice, common_exp  #, common_behav, common_img, common_el

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
W_HEIGHT = ROW + 10
W_WIDTH = I_WIDTH

# Post-OP care box
P_LEFT = I_LEFT
P_TOP = M_TOP + W_HEIGHT + 10
P_HEIGHT = W_HEIGHT
P_WIDTH = I_WIDTH

# Sacrificed (euthanized) box
E_LEFT = W_LEFT
E_TOP = M_TOP + W_HEIGHT + P_HEIGHT + 20
E_HEIGHT = M_HEIGHT - (W_HEIGHT + P_HEIGHT + 20)
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
REL_BACKUP_PATH = "Datajoint/manual_submissions"

# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

# Load YAML file (has to be present in the same folder)
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
if len(mouse_ids) == 0:
    default_params['behavior']['default_mouse'] = 1
elif default_params['behavior']['default_mouse'] == 'last_mouse':
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
care_substances = common_mice.CareSubstance().fetch('care_name')
types = common_mice.SurgeryType().fetch('surgery_type')

# =============================================================================
# Default parameter for dropdown menus and text boxes
# =============================================================================


class window(wx.Frame):

    def __init__(self, parent, id):

        wx.Frame.__init__(self, parent, id, '{} ({}): Enter mouse data into database'.format(inv_fullname, investigator),
                          size=(WINDOW_WIDTH, WINDOW_HEIGHT))
        panel = wx.Panel(self)
        bold_font = wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.BOLD)      # Create bold font for primary key labels

        self.job_list = list()  # save jobs in format [ [table, entry_dict, source_path, target_path], [...], ...]
        # =============================================================================
        # Left upper box: Add new MOUSE (3x3 fields + notes)
        # =============================================================================
        mouse_box = wx.StaticBox(panel, label='NEW MOUSE INFORMATION',
                                 pos=(M_LEFT - 20, M_TOP - 30), size=(M_WIDTH, M_HEIGHT))
        mouse_box.SetForegroundColour(BOX_TITLE_COLOR)

        # Mouse ID (default is highest existing mouse ID + 1)
        text = wx.StaticText(panel, label="Mouse ID:", pos=(M_LEFT, M_TOP))
        text.SetFont(bold_font)
        self.mouse_id = wx.TextCtrl(panel, pos=(M_LEFT, M_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.mouse_id.SetValue(str(default_params['behavior']['default_mouse'] + 1))

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

        # Submit mouse button
        self.submit_mouse_button = wx.Button(panel, label="Add new\nmouse",
                                               pos=(M_LEFT + 2 * COL, M_TOP + 3 * ROW + 20),
                                               size=(BUTTON_WIDTH/2 + 20, BUTTON_HEIGHT))
        self.Bind(wx.EVT_BUTTON, self.event_submit_mouse, self.submit_mouse_button)

        # Update mouse button
        self.update_mouse_button = wx.Button(panel, label="Update\nmouse",
                                             pos=(M_LEFT + 2 * COL + BUTTON_WIDTH/2 + 20, M_TOP + 3 * ROW + 20),
                                             size=(BUTTON_WIDTH/2 - 20, BUTTON_HEIGHT))
        self.Bind(wx.EVT_BUTTON, self.event_update_mouse, self.update_mouse_button)
        self.update_mouse_button.Disable()  # disabled by default, becomes clickable once a mouse is loaded

        # =============================================================================
        # Left lower box: Enter new SURGERY (2x3 fields + notes)
        # =============================================================================

        surg_box = wx.StaticBox(panel, label='NEW SURGERY', pos=(S_LEFT - 20, S_TOP - 30), size=(S_WIDTH, S_HEIGHT))
        surg_box.SetForegroundColour(BOX_TITLE_COLOR)

        # Surgery number (default is empty, will be filled when a mouse is loaded)
        text = wx.StaticText(panel, label="Surgery number:", pos=(S_LEFT, S_TOP))
        text.SetFont(bold_font)
        self.surg_num = wx.TextCtrl(panel, pos=(S_LEFT, S_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.surg_num.SetValue('')

        # Date and time of surgery (default is current day, 12 pm)
        wx.StaticText(panel, label="Datetime (YYYY-MM-DD HH:MM):", pos=(S_LEFT + COL, S_TOP))
        self.dos = wx.TextCtrl(panel, pos=(S_LEFT + COL, S_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.dos.SetValue(current_day + " 12:00")

        # Surgery type (default from YAML)
        wx.StaticText(panel, label="Surgery type:", pos=(S_LEFT + 2 * COL, S_TOP))
        self.type = wx.ComboBox(panel, choices=types, style=wx.CB_READONLY,
                                pos=(S_LEFT + 2 * COL, S_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        item = self.type.FindString(default_params['mice']['default_surgery_type'])
        self.type.SetSelection(item)

        # Anesthesia (default from YAML)
        wx.StaticText(panel, label="Anesthesia:", pos=(S_LEFT, S_TOP + ROW))
        self.anesthesia = wx.TextCtrl(panel, pos=(S_LEFT, S_TOP + ROW + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.anesthesia.SetValue(default_params['mice']['default_anesthesia'])

        # Pre-OP weight in grams (default is empty)
        wx.StaticText(panel, label="Pre-OP weight [grams]:", pos=(S_LEFT + COL, S_TOP + ROW))
        self.pre_op_weight = wx.TextCtrl(panel, pos=(S_LEFT + COL, S_TOP + ROW + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.pre_op_weight.SetValue('')

        # Duration
        wx.StaticText(panel, label="Duration of\nsurgery [min]:", pos=(S_LEFT + 2 * COL, S_TOP + ROW - 12))
        self.duration = wx.TextCtrl(panel, pos=(S_LEFT + 2 * COL, S_TOP + ROW + 20), size=(BOX_WIDTH/2 - 2, BOX_HEIGHT))
        self.duration.SetValue('')

        # Illumination time
        wx.StaticText(panel, label="Stroke illumination\ntime [min]:",
                      pos=(S_LEFT + 2 * COL + BOX_WIDTH/2 + 2, S_TOP + ROW - 12))
        self.illumination = wx.TextCtrl(panel, pos=(S_LEFT + 2 * COL + BOX_WIDTH/2 + 2, S_TOP + ROW + 20),
                                        size=(BOX_WIDTH/2, BOX_HEIGHT))
        self.illumination.SetValue('')

        # Stroke parameters
        wx.StaticText(panel, label="Stroke parameters (if applicable):", pos=(S_LEFT, S_TOP + 2 * ROW))
        self.stroke_params = wx.TextCtrl(panel, value="", style=wx.TE_MULTILINE, pos=(S_LEFT, S_TOP + 2 * ROW + 20),
                                         size=(BOX_WIDTH, 50))
        # Notes
        wx.StaticText(panel, label="Notes:", pos=(S_LEFT + COL, S_TOP + 2 * ROW))
        self.surgery_notes = wx.TextCtrl(panel, value="", style=wx.TE_MULTILINE, pos=(S_LEFT + COL, S_TOP + 2*ROW + 20),
                                         size=(BOX_WIDTH, 50))

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
        text = wx.StaticText(panel, label="Injection number:", pos=(I_LEFT, I_TOP))
        text.SetFont(bold_font)
        self.inj_num = wx.TextCtrl(panel, pos=(I_LEFT, I_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        self.inj_num.SetValue('')

        # Substance (default from YAML)
        wx.StaticText(panel, label="Substance:", pos=(I_LEFT + COL, I_TOP))
        self.substance = wx.ComboBox(panel, choices=substances, style=wx.CB_READONLY,
                                     pos=(I_LEFT + COL, I_TOP + 20), size=(BOX_WIDTH, BOX_HEIGHT))
        item = self.substance.FindString(default_params['mice']['default_substance'])
        self.substance.SetSelection(item)

        # Volume (default 0.3 uL)
        wx.StaticText(panel, label="Injected volume [\u03BCL]:", pos=(I_LEFT + 2 * COL, I_TOP))
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

        weight_box = wx.StaticBox(panel, label='NEW WEIGHT', pos=(W_LEFT - 20, W_TOP - 20), size=(W_WIDTH, W_HEIGHT))
        weight_box.SetForegroundColour(BOX_TITLE_COLOR)

        # Date of weight (default is current day)
        text = wx.StaticText(panel, label="Date (YYYY-MM-DD):", pos=(W_LEFT, W_TOP))
        text.SetFont(bold_font)
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
        # Right upper middle box: Enter new post-op care (1x2 fields)
        # =============================================================================

        weight_box = wx.StaticBox(panel, label='NEW POST-OP CARE', pos=(P_LEFT - 20, P_TOP - 20), size=(P_WIDTH, P_HEIGHT))
        weight_box.SetForegroundColour(BOX_TITLE_COLOR)

        # Date of administration (default is current day)
        text = wx.StaticText(panel, label="Date:", pos=(P_LEFT, P_TOP))
        text.SetFont(bold_font)
        self.doa = wx.TextCtrl(panel, pos=(P_LEFT, P_TOP + 20), size=(BOX_WIDTH/2, BOX_HEIGHT))
        self.doa.SetValue(current_day)

        # Type of substance
        wx.StaticText(panel, label="Substance:", pos=(P_LEFT + 0.5* COL, P_TOP))
        self.care_substance = wx.ComboBox(panel, choices=care_substances, style=wx.CB_READONLY,
                                          pos=(P_LEFT + 0.5* COL, P_TOP + 20), size=(BOX_WIDTH/2 + 30, BOX_HEIGHT))
        item = self.care_substance.FindString('Carprofen (s.c.)')
        self.care_substance.SetSelection(item)

        # Injected volume
        wx.StaticText(panel, label="Volume [\u03BCL]:", pos=(P_LEFT + COL+30, P_TOP))
        self.care_vol = wx.TextCtrl(panel, pos=(P_LEFT + COL+30, P_TOP + 20), size=(BOX_WIDTH/2-10, BOX_HEIGHT))
        self.care_vol.SetValue('80')

        # Frequency
        wx.StaticText(panel, label="Inj/day:", pos=(P_LEFT + COL+120, P_TOP))
        size = (W_LEFT + COL + BOX_WIDTH)-(P_LEFT+COL+120)
        self.freq = wx.TextCtrl(panel, pos=(P_LEFT+COL+120, P_TOP + 20), size=(size, BOX_HEIGHT))
        self.freq.SetValue('1')

        # Submit weight button
        self.submit_care_button = wx.Button(panel, label="Add new care",
                                                 pos=(P_LEFT + 2 * COL, P_TOP),
                                                 size=(BUTTON_WIDTH, BUTTON_HEIGHT-5))
        self.Bind(wx.EVT_BUTTON, self.event_submit_care, self.submit_care_button)

        # =============================================================================
        # Right middle box: Enter new sacrificed mouse (1x2 fields + notes)
        # =============================================================================

        sac_box = wx.StaticBox(panel, label='MOUSE EUTHANIZED', pos=(E_LEFT - 20, E_TOP - 20), size=(E_WIDTH, E_HEIGHT))
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
                                  pos=(E_LEFT, E_TOP + ROW + 10), size=(COL + BOX_WIDTH, 30))

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
        self.new_mouse = wx.ComboBox(panel, choices=np.array(mouse_ids, dtype=str), style=wx.CB_READONLY,
                                 pos=(L_LEFT, L_TOP + 20), size=(BOX_WIDTH - 50, BOX_HEIGHT))
        item = self.new_mouse.FindString(str(default_params['behavior']['default_mouse']))
        self.new_mouse.SetSelection(item)

        # Next mouse button
        self.next_mouse_button = wx.Button(panel, label="\u25B2",
                                                 pos=(L_LEFT + BOX_WIDTH - 45, L_TOP + 19),
                                                 size=(20, BOX_HEIGHT))
        self.Bind(wx.EVT_BUTTON, self.event_next_mouse, self.next_mouse_button)

        # Previous mouse button
        self.prev_mouse_button = wx.Button(panel, label="\u25BC",
                                                 pos=(L_LEFT + BOX_WIDTH - 20, L_TOP + 19),
                                                 size=(20, BOX_HEIGHT))
        self.Bind(wx.EVT_BUTTON, self.event_prev_mouse, self.prev_mouse_button)

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
        """Insert new mouse to Mouse()"""

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
                          info=self.mouse_notes.GetValue())

        # Insert into database and save backup YAML
        identifier = 'mouse_{}_M{:03d}'.format(investigator, int(self.mouse_id.GetValue()))
        success = self.safe_insert(common_mice.Mouse(), mouse_dict, identifier, REL_BACKUP_PATH)

        if success:
            # Update "load mouse" drop-down menu
            new_mouse_id = self.mouse_id.GetValue()
            curr_mouse_ids = self.new_mouse.GetItems()
            if new_mouse_id not in curr_mouse_ids:
                new_mouse_ids = np.array(np.append(curr_mouse_ids, self.mouse_id.GetValue()), dtype=int)    # Insert new mouse ID to the choices
                new_mouse_ids[::-1].sort()                                          # Sort the list again (descending)
                self.new_mouse.Clear()                                              # Remove old choices
                for id in new_mouse_ids:                                            # Add the new ordered list
                    self.new_mouse.Append(str(id))
                item = self.new_mouse.FindString(self.mouse_id.GetValue())
                self.new_mouse.SetSelection(item)

            # Set the new mouse as "loaded" so that surgeries can directly be added to it
            self.curr_mouse.SetValue(new_mouse_id)

    def event_update_mouse(self, event):
        """Update info for an existing mouse"""

        # Get info from fields
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
                          info=self.mouse_notes.GetValue())

        # Update entry in database
        common_mice.Mouse.update1(row=mouse_dict)

        # Overwrite YAML backup with changes
        identifier = 'mouse_{}_M{:03d}'.format(investigator, int(self.mouse_id.GetValue()))
        filename = os.path.join(login.get_neurophys_wahl_directory(), REL_BACKUP_PATH, identifier + '.yaml')
        with open(filename, 'w') as outfile:
            yaml.dump(mouse_dict, outfile, default_flow_style=False)
        self.status_text.write('Updated backup file at %s' % filename + '\n')

    def event_submit_surgery(self, event):
        """Enter surgery data as new entry into the Surgery table"""

        # Check if a mouse was loaded before submitting a surgery
        if self.curr_mouse.GetValue() == 'None':
            self.status_text.write('Load a mouse before submitting a surgery!')
            return

        # Get data from relevant fields
        surg_dict = dict(username=investigator,
                         mouse_id=self.curr_mouse.GetValue(),
                         surgery_num=self.surg_num.GetValue(),
                         surgery_date=self.dos.GetValue(),
                         surgery_type=self.type.GetValue(),
                         anesthesia=self.anesthesia.GetValue(),
                         pre_op_weight=self.pre_op_weight.GetValue(),
                         stroke_params=self.stroke_params.GetValue(),
                         duration=self.duration.GetValue(),
                         surgery_notes=self.surgery_notes.GetValue())
        # Add illumination time only if it was provided, otherwise defaults to None
        if len(self.illumination.GetValue()) > 0:
            surg_dict['illumination_time'] = self.illumination.GetValue()

        # Todo: Ask user to overwrite if a weight on that day already exists (happens if a surgery was re-inserted)
        #  - Dont overwrite YAML file if a second weight has been added
        # Insert into database and save backup YAML
        identifier = 'surgery_{}_M{:03d}_{}'.format(investigator, int(self.mouse_id.GetValue()), self.surg_num.GetValue())
        self.safe_insert(common_mice.Surgery(), surg_dict, identifier, REL_BACKUP_PATH)

    def event_submit_injection(self, event):
        """Enter new injection for the currently selected surgery"""

        # Check if a mouse was loaded before submitting an injection
        if self.curr_mouse.GetValue() == 'None':
            self.status_text.write('Load a mouse before submitting an injection!')
            return

        # Get data from relevant fields
        inj_dict = dict(username=investigator,
                        mouse_id=self.curr_mouse.GetValue(),
                        surgery_num=int(self.surg_num.GetValue()),
                        injection_num=int(self.inj_num.GetValue()),
                        substance_name=self.substance.GetValue(),
                        volume=self.volume.GetValue(),
                        dilution=self.dilution.GetValue(),
                        site=self.site.GetValue(),
                        coordinates=self.coordinates.GetValue(),
                        injection_notes=self.inj_notes.GetValue())

        # Insert into database and save backup YAML
        identifier = 'injection_{}_M{:03d}_{}_{}'.format(investigator, int(self.mouse_id.GetValue()),
                                                         self.surg_num.GetValue(), self.inj_num.GetValue())
        success = self.safe_insert(common_mice.Injection(), inj_dict, identifier, REL_BACKUP_PATH)

        if success:
            # Increase the injection_num counter by 1
            self.inj_num.SetValue(str(inj_dict['injection_num']+1))

    def event_submit_weight(self, event):
        """Enter a new weight for the currently selected mouse"""

        # Check if a mouse was loaded before submitting a weight
        if self.curr_mouse.GetValue() == 'None':
            self.status_text.write('Load a mouse before submitting a weight measurement!')
            return

        # Get data from relevant fields
        weight_dict = dict(username=investigator,
                           mouse_id=self.curr_mouse.GetValue(),
                           date_of_weight=self.dow.GetValue(),
                           weight=self.weight.GetValue())

        # Insert into database and save backup YAML
        identifier = 'weight_{}_M{:03d}_{}'.format(investigator, int(self.mouse_id.GetValue()), self.dow.GetValue())
        self.safe_insert(common_mice.Weight(), weight_dict, identifier, REL_BACKUP_PATH)

    def event_submit_care(self, event):
        """Enter a new care administration for the currently selected mouse"""

        # Check if a mouse was loaded before submitting a weight
        if self.curr_mouse.GetValue() == 'None':
            self.status_text.write('Load a mouse before submitting a care administration!')
            return

        # Get data from relevant fields
        care_dict = dict(username=investigator,
                         mouse_id=self.curr_mouse.GetValue(),
                         date_of_care=self.doa.GetValue(),
                         care_name=self.care_substance.GetValue(),
                         care_volume=int(self.care_vol.GetValue()),
                         care_frequency=int(self.freq.GetValue()))

        # Insert into database and save backup YAML
        identifier = 'care_{}_M{:03d}_{}'.format(investigator, int(self.mouse_id.GetValue()), self.doa.GetValue())
        self.safe_insert(common_mice.PainManagement(), care_dict, identifier, REL_BACKUP_PATH)

    def event_submit_euthanasia(self, event):
        """Move the currently selected mouse to the Sacrificed table"""

        # Check if a mouse was loaded before submitting a euthanasia
        if self.curr_mouse.GetValue() == 'None':
            self.status_text.write('Load a mouse before submitting a euthanasia!')
            return

        # Get data from relevant fields
        weight_dict = dict(username=investigator,
                           mouse_id=self.curr_mouse.GetValue(),
                           date_of_sacrifice=self.doe.GetValue(),
                           perfused=int(self.perfused.GetValue()),
                           reason=self.reason.GetValue())

        # Insert into database and save backup YAML
        identifier = 'sacrificed_{}_M{:03d}'.format(investigator, int(self.mouse_id.GetValue()))
        self.safe_insert(common_mice.Sacrificed(), weight_dict, identifier, REL_BACKUP_PATH)

    def load_mouse(self):
        """ Load data of an already existing mouse to add surgeries/weights/euthanasia """
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
        self.mouse_id.SetValue(str(entry['mouse_id']))
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

        self.curr_mouse.SetValue(str(entry['mouse_id']))

        # Set the surgery_num to the next integer and injection_num to 0, ready for a new surgery submission
        mouse_filter = "mouse_id = '{}'".format(entry['mouse_id'])
        try:
            self.surg_num.SetValue(str(max((common_mice.Surgery() & username_filter &
                                            mouse_filter).fetch('surgery_num'))+1))
        except ValueError:
            self.surg_num.SetValue('1')

        # Enable the "update mouse" button
        self.update_mouse_button.Enable()

        self.status_text.write(
            '\nSuccessfully loaded mouse {}. You can now add new or update existing data.'.format(mouse_dict) + '\n')

    def change_mouse(self, change):
        """ Change the current mouse ID by the given value and load the new mouse data """
        curr_id = int(self.new_mouse.GetValue())

        # Do not change mouse_id if the next ID would be out of range of existing IDs
        if (min(mouse_ids) > curr_id+change) or (curr_id+change > max(mouse_ids)):
            return

        # If mouse ID does not exist, look for the next existing one
        elif curr_id+change not in mouse_ids:
            if change < 0:
                down = -1
            else:
                down = 1
            while curr_id+change not in mouse_ids:
                change += down

        # Set the new mouse ID into the field and load mouse data
        item = self.new_mouse.FindString(str(curr_id+change))
        self.new_mouse.SetSelection(item)
        self.load_mouse()

    def event_load_mouse(self, event):
        """Load data of an already existing mouse to add surgeries/weights/euthanasia"""
        self.load_mouse()

    def event_next_mouse(self, event):
        """ Select and load the mouse with the next higher ID """
        self.change_mouse(1)

    def event_prev_mouse(self, event):
        """ Select and load the mouse with the next lower ID """
        self.change_mouse(-1)

    def event_quit_button(self, event):
        """ User pressed quit button """
        self.Close(True)

    def safe_insert(self, table, dictionary, identifier, backup):
        """ Enter a dict into a table. If successful, returns True and the dict is saved in a backup YAML file."""
        try:
            id = table.insert1(dictionary)       # some inserts (e.g. care) return an ID that is added to the identifier
            self.status_text.write('Sucessfully entered new entry in table "{}": \n\t'.format(table.table_name) +
                                   str(dictionary) + '\n')
            
            if id is not None:
                identifier += "_{}".format(id)
            
            # save dictionary in a backup YAML file for faster re-population
            filename = os.path.join(login.get_neurophys_wahl_directory(), backup, identifier + '.yaml')
            with open(filename, 'w') as outfile:
                yaml.dump(dictionary, outfile, default_flow_style=False)
            self.status_text.write('Created backup file at %s' % filename + '\n')
            return True

        except Exception as ex:
            print('Exception manually caught:', ex)
            self.status_text.write('Error while entering ' + str(dictionary) + ' : ' + str(ex) + '\n')
            return False


def get_backup_path():
    return REL_BACKUP_PATH


# run the GUI
if __name__ == '__main__':
    app = wx.App()
    frame = window(parent=None, id=-1)
    frame.Show()
    app.MainLoop()