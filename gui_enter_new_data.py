#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:13:55 2019
@author: adhoff
Installation of wxpython in Ubuntu 18.04 (pip install did not work):
conda install -c anaconda wxpython
pip install datajoint
"""

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import wx
import login
import datajoint
from datetime import datetime
import os
import glob
import numpy as np

path_neurophysiology = login.get_neurophys_directory()

# =============================================================================
# HARDCODED PARAMETER FOR GUI
# =============================================================================

WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 1100

WINDOW_WIDTH_L = 900

BUTTON_WIDTH = 130
BUTTON_HEIGHT = 40

# session info
S_LEFT = 30+20    # position of the first element in session box
S_TOP = 20+30
ROW = 70
COL = 200
S_HEIGHT = 260   # height of session box

# behavior
B_LEFT = S_LEFT
B_TOP = S_HEIGHT + 70

# imaging
I_LEFT = WINDOW_WIDTH_L-20
I_TOP = B_TOP

L_TOP = B_TOP + 500

# =============================================================================
# DETAILS FOR A PARAMETER FILE LATER
# =============================================================================

# TODO: put this in yaml parameter file, see here https://stackoverflow.com/a/5060485

# behavior
default_mouse = 'default_mouse'
default_setup = 'default_setup'
default_task = 'default_task'
default_stage = '0'
default_anesthesia = 'Awake'
default_experimenter = 'Adrian'
default_wheel = 'default_wheel'
default_sync = 'default_sync'
default_camera_pos = 'default_camera_pos'
default_video_rate = '30'
default_stimulator = 'default_stimulator'
default_event_type = 'default_event_type'

default_folder = path_neurophysiology
wheel_file = 'wheel_rec_{:02d}.bin'
sync_file = 'imaging_signal_{:02d}.bin'
video_file = '{:02d}_vid.avi'
whisker_file = 'whisker_stim_{:02d}.bin'
event_file = 'sensory_events_{:02d}.txt'

# imaging
default_microscope = 'Scientifica'
default_laser = 'MaiTai'
default_layer = 'default_layer'
default_planes = '1'

SKIP_IMG_TRANSFER = True
default_img_folder = path_neurophysiology
H45_file = 'test_A{}_Ch{}_ ????.tif'    # later used as format string to fill in {} and wildcard characters ? for glob.glob
scientifica_file = 'file_000??.tif'     # Hendriks format
parameter_file = 'parameters.xml'

# show only necessary parts of the panels (here: E-phys instead of behavior)
show_ephys = False
default_el_sync = 'Galvo_Y_Clipped'
default_el_sync_file = 'board-ADC-02.dat'

default_el_folder = path_neurophysiology
probe_files = 'amp-A-0??.dat'
info_file = 'info.rhd'

# general paths
path_backup_npy = '../data/'
default_probe = '16_Wide'
default_shutter_file = '###'


# =============================================================================
# Load options for drop down menus
# =============================================================================

# connect to datajoint database
login.connect()
from schema import common_mice, common_exp, common_behav  # , common_img, common_el

alive_mice = common_mice.Mouse() - common_mice.Sacrificed()
mouse_names = alive_mice.fetch('mouse_id', order_by='mouse_id')     # Todo: restrict mouse selection by investigator

setups = common_exp.Setup().fetch('setup')
tasks = common_exp.Task().fetch('task')
anesthesias = common_exp.Anesthesia().fetch('anesthesia')
experimenters = common_mice.Investigator().fetch('username')

wheels = common_behav.WheelType().fetch('wheel_type')
signals = common_behav.SynchronizationType().fetch('sync_type')
camera_pos = common_behav.CameraPosition().fetch('camera_position')
stimulators = common_behav.StimulatorType().fetch('stimulator_type')
event_types = common_behav.SensoryEventType().fetch('sensory_event_type')

# microscopes = img.Microscope().fetch('microscope')
# lasers = img.Laser().fetch('laser')
# layers = img.Layer().fetch('layer')
#
# probes = el.ProbeType().fetch('probe_type')
# el_sync_signals = ['Galvo_Y', 'Galvo_Y_Clipped']

current_day = datetime.today().strftime('%Y-%m-%d')  # YYYY-MM-DD

# =============================================================================
# Default parameter for dropdown menus and text boxes
# =============================================================================


class window(wx.Frame):

    def __init__(self, parent, id):

        wx.Frame.__init__(self, parent, id, 'Enter data into pipeline', size=(WINDOW_WIDTH, WINDOW_HEIGHT))
        panel = wx.Panel(self)

        self.job_list = list()    # save jobs in format [ [table, entry_dict, source_path, target_path], [...], ...]
# =============================================================================
# Upper box: Add new session
# =============================================================================
        wx.StaticBox(panel, label='SESSION INFORMATION',
                     pos=(S_LEFT-20, S_TOP-30), size=(WINDOW_WIDTH_L-2*S_LEFT, S_HEIGHT))

        # Mouse name
        wx.StaticText(panel,label="Mouse name:", pos=(S_LEFT, S_TOP))
        self.mouse_name = wx.ComboBox(panel, choices=mouse_names, style=wx.CB_READONLY,
                                      pos=(S_LEFT, S_TOP+20), size=(170, -1))
        self.mouse_name.Bind(wx.EVT_COMBOBOX, self.event_mouse_selected)
        item = self.mouse_name.FindString(default_mouse)
        self.mouse_name.SetSelection(item)

        # Day of experiment
        wx.StaticText(panel,label="Day (YYYY-MM-DD):", pos=(S_LEFT+COL,S_TOP))
        self.day = wx.TextCtrl(panel, pos=(S_LEFT+COL,S_TOP+20), size=(170,-1))
        self.day.SetValue( current_day)

        # Trial
        wx.StaticText(panel,label="Trial (base 1):", pos=(S_LEFT+400,S_TOP))
        self.trial = wx.TextCtrl(panel, pos=(S_LEFT+2*COL,S_TOP+20), size=(170,-1))
        self.trial.SetValue( '1' )

        # Setup
        wx.StaticText(panel,label="Setup:", pos=(S_LEFT,S_TOP+ROW))
        self.setup = wx.ComboBox(panel, choices = setups, style=wx.CB_READONLY,
                                      pos=(S_LEFT,S_TOP+ROW+20), size=(170,-1) )
        item = self.setup.FindString(default_setup)
        self.setup.SetSelection(item)

        # Task
        wx.StaticText(panel,label="Task:", pos=(S_LEFT+COL,S_TOP+ROW))
        self.task = wx.ComboBox(panel, choices = tasks, style=wx.CB_READONLY,
                                      pos=(S_LEFT+COL,S_TOP+ROW+20), size=(130,-1) )
        item = self.task.FindString(default_task)
        self.task.SetSelection(item)

        # Stage
        wx.StaticText(panel,label="Stage:", pos=(S_LEFT+COL+130,S_TOP+ROW))
        self.stage = wx.TextCtrl(panel, pos=(S_LEFT+COL+140,S_TOP+ROW+20), size=(30,-1))
        self.stage.SetValue( default_stage )
        # Anesthesia
        wx.StaticText(panel,label="Anesthesia:", pos=(S_LEFT+2*COL,S_TOP+ROW))
        self.anesthesia = wx.ComboBox(panel, choices = anesthesias, style=wx.CB_READONLY,
                                      pos=(S_LEFT+2*COL,S_TOP+ROW+20), size=(170,-1) )
        item = self.anesthesia.FindString(default_anesthesia)
        self.anesthesia.SetSelection(item)

        # Experimenter
        wx.StaticText(panel,label="Experimenter:", pos=(S_LEFT+3*COL,S_TOP+ROW))
        self.experimenter = wx.ComboBox(panel, choices = experimenters, style=wx.CB_READONLY,
                                      pos=(S_LEFT+3*COL,S_TOP+ROW+20), size=(170,-1) )
        item = self.experimenter.FindString(default_experimenter)
        self.experimenter.SetSelection(item)

        # Notes
        wx.StaticText(panel,label="Notes:", pos=(S_LEFT,S_TOP+2*ROW))
        self.notes = wx.TextCtrl(panel, value="",
                                       style=wx.TE_MULTILINE,
                                       pos=(S_LEFT,S_TOP+2*ROW+20),
                                       size=(WINDOW_WIDTH_L-3*S_LEFT-200,50 ) )

        # Load session button
        self.load_session_button = wx.Button(panel,label="Load session",
                                       pos=(S_LEFT+3*COL, S_TOP),
                                       size=(150, 50) )
        self.Bind( wx.EVT_BUTTON, self.event_load_session, self.load_session_button)

        # Submit session button
        self.submit_session_button = wx.Button(panel,label="Submit session",
                                       pos=(S_LEFT+3*COL, S_TOP+2*ROW+20),
                                       size=(150, 50) )
        self.Bind( wx.EVT_BUTTON, self.event_submit_session, self.submit_session_button)

# =============================================================================
# Enter behavioral data from wheel, video, ...
# =============================================================================
        if show_ephys == False:
            wx.StaticBox(panel, label='BEHAVIOR RECORDINGS',
                        pos=(B_LEFT-20, B_TOP-30), size=(WINDOW_WIDTH_L-2*S_LEFT, 500))

            # Folder with behavioral data
            wx.StaticText(panel,label="Data folder:", pos=(B_LEFT,B_TOP))
            self.folder = wx.TextCtrl(panel, value=default_folder,
                                           pos=(B_LEFT,B_TOP+20),
                                           size=(2*COL-20,25 ) )

            # Button to select new folder
            self.select_folder = wx.Button(panel,label="Select folder",
                                           pos=(B_LEFT+2*COL, B_TOP+20),
                                           size=(100, 25) )
            self.Bind( wx.EVT_BUTTON, self.event_select_folder, self.select_folder)



        # Wheel information
            self.wheel_checkbox = wx.CheckBox(panel, label='Wheel',
                                         pos = (B_LEFT, B_TOP+ROW+20), size=(130,20) )

            # dropdown menu wheel
            wx.StaticText(panel,label="Wheel type:", pos=(B_LEFT+COL,B_TOP+ROW))
            self.wheel = wx.ComboBox(panel, choices = wheels, style=wx.CB_READONLY,
                                          pos=(B_LEFT+COL,B_TOP+ROW+20), size=(170,-1) )
            item = self.wheel.FindString(default_wheel)
            self.wheel.SetSelection(item)


        # Imaging sync
            self.sync_checkbox = wx.CheckBox(panel, label='Imaging Sync',
                                         pos = (B_LEFT, B_TOP+2*ROW+20), size=(130,20) )

            # dropdown menu wheel
            wx.StaticText(panel,label="Signal type:", pos=(B_LEFT+COL,B_TOP+2*ROW))
            self.imaging_sync = wx.ComboBox(panel, choices = signals, style=wx.CB_READONLY,
                                          pos=(B_LEFT+COL,B_TOP+2*ROW+20), size=(170,-1) )
            item = self.imaging_sync.FindString(default_sync)
            self.imaging_sync.SetSelection(item)


        # Video
            self.video_checkbox = wx.CheckBox(panel, label='Video',
                                         pos = (B_LEFT, B_TOP+3*ROW+20), size=(130,20) )

            # dropdown menu wheel
            wx.StaticText(panel,label="Camera position:", pos=(B_LEFT+COL,B_TOP+3*ROW))
            self.camera_pos = wx.ComboBox(panel, choices = camera_pos, style=wx.CB_READONLY,
                                          pos=(B_LEFT+COL,B_TOP+3*ROW+20), size=(170,-1) )
            item = self.camera_pos.FindString(default_camera_pos)
            self.camera_pos.SetSelection(item)

            # file type
            wx.StaticText(panel,label="File format:", pos=(B_LEFT+2*COL,B_TOP+3*ROW))
            self.video_file = wx.TextCtrl(panel, pos=(B_LEFT+2*COL,B_TOP+3*ROW+20), size=(190,-1))
            self.video_file.SetValue( video_file )

            # frame rate
            wx.StaticText(panel,label="Frame rate [Hz]:", pos=(B_LEFT+3*COL,B_TOP+3*ROW))
            self.frame_rate = wx.TextCtrl(panel, pos=(B_LEFT+3*COL,B_TOP+3*ROW+20), size=(150,-1))
            self.frame_rate.SetValue( default_video_rate )

        # Whisker stim & event times
            self.whisker_checkbox = wx.CheckBox(panel, label='Whisker stim',
                                         pos = (B_LEFT, B_TOP+4*ROW+20), size=(130,20) )

            # dropdown menu whisker stimulator
            wx.StaticText(panel,label="Whisker stimulator:", pos=(B_LEFT+COL,B_TOP+4*ROW))
            self.stimulator = wx.ComboBox(panel, choices = stimulators, style=wx.CB_READONLY,
                                          pos=(B_LEFT+COL,B_TOP+4*ROW+20), size=(170,-1) )
            item = self.stimulator.FindString(default_stimulator)
            self.stimulator.SetSelection(item)


            self.event_checkbox = wx.CheckBox(panel, label='Events',
                                         pos = (B_LEFT, B_TOP+5*ROW+20), size=(130,20) )
            # dropdown menu events
            wx.StaticText(panel,label="Event file type:", pos=(B_LEFT+COL,B_TOP+5*ROW))
            self.event_type = wx.ComboBox(panel, choices = event_types, style=wx.CB_READONLY,
                                          pos=(B_LEFT+COL,B_TOP+5*ROW+20), size=(170,-1) )
            item = self.event_type.FindString(default_event_type)
            self.event_type.SetSelection(item)

            # Submit behavior button
            self.submit_behav_button = wx.Button(panel,label="Submit behavior",
                                           pos=(B_LEFT+3*COL, B_TOP+5*ROW),
                                           size=(150, 50) )
            self.Bind( wx.EVT_BUTTON, self.event_submit_behavior, self.submit_behav_button)

# =============================================================================
# Enter E-phys data
# =============================================================================
        """
        COMMENTED OUT AND IGNORED FOR NOW TO TEST OUT GUI
        
        if show_ephys == True:
            wx.StaticBox(panel, label='ELECTRICAL RECORDINGS',
                        pos=(B_LEFT-20, B_TOP-30), size=(WINDOW_WIDTH_L-2*S_LEFT, 500))

            # Folder with behavioral data
            wx.StaticText(panel,label="Data folder:", pos=(B_LEFT,B_TOP))
            self.el_folder = wx.TextCtrl(panel, value=default_el_folder,
                                           pos=(B_LEFT,B_TOP+20),
                                           size=(2*COL-20,25 ) )

            # Button to select new folder
            self.select_el_folder = wx.Button(panel,label="Select folder",
                                           pos=(B_LEFT+2*COL, B_TOP+20),
                                           size=(100, 25) )
            self.Bind( wx.EVT_BUTTON, self.event_select_el_folder, self.select_el_folder)


        # Probe channels
            self.probe_checkbox = wx.CheckBox(panel, label='Probe',
                                         pos = (B_LEFT, B_TOP+ROW+20), size=(130,20) )

            # dropdown menu wheel
            wx.StaticText(panel,label="Probe type:", pos=(B_LEFT+COL,B_TOP+ROW))
            self.probe = wx.ComboBox(panel, choices = probes, style=wx.CB_READONLY,
                                          pos=(B_LEFT+COL,B_TOP+ROW+20), size=(170,-1) )
            item = self.probe.FindString(default_probe)
            self.probe.SetSelection(item)


        # Imaging sync
            self.sync_el_checkbox = wx.CheckBox(panel, label='Sync Imaging',
                                         pos = (B_LEFT, B_TOP+2*ROW+20), size=(130,20) )

            # dropdown menu
            wx.StaticText(panel,label="Signal type:", pos=(B_LEFT+COL,B_TOP+2*ROW))
            self.el_sync = wx.ComboBox(panel, choices = el_sync_signals, style=wx.CB_READONLY,
                                          pos=(B_LEFT+COL,B_TOP+2*ROW+20), size=(170,-1) )
            item = self.el_sync.FindString(default_el_sync)
            self.el_sync.SetSelection(item)


        # Shutter
            self.shutter_checkbox = wx.CheckBox(panel, label='Imaging Shutter',
                                         pos = (B_LEFT, B_TOP+3*ROW+20), size=(130,20) )

            # file type
            wx.StaticText(panel,label="Shutter file:", pos=(B_LEFT+COL,B_TOP+3*ROW))
            self.shutter_file = wx.TextCtrl(panel, pos=(B_LEFT+COL,B_TOP+3*ROW+20), size=(190,-1))
            self.shutter_file.SetValue( default_shutter_file )


        # Whisker stim & event times
            self.info_checkbox = wx.CheckBox(panel, label='Info File',
                                         pos = (B_LEFT, B_TOP+4*ROW+20), size=(130,20) )


            # Submit behavior button
            self.submit_el_button = wx.Button(panel,label="Submit E-phys",
                                           pos=(B_LEFT+3*COL, B_TOP+5*ROW),
                                           size=(150, 50) )
            self.Bind( wx.EVT_BUTTON, self.event_submit_el, self.submit_el_button)

# =============================================================================
# Enter imaging data
# =============================================================================
        wx.StaticBox(panel, label='IMAGING',
                    pos=(I_LEFT-20, I_TOP-30), size=(WINDOW_WIDTH-I_LEFT-20, 500))

        # Folder with imaging data
        wx.StaticText(panel,label="Data folder:", pos=(I_LEFT,I_TOP))
        self.img_folder = wx.TextCtrl(panel, value=default_img_folder,
                                       pos=(I_LEFT,I_TOP+20),
                                       size=(2*COL-20,25 ) )

        # Button to select new folder
        self.select_img_folder = wx.Button(panel,label="Select folder",
                                       pos=(I_LEFT+2*COL, I_TOP+20),
                                       size=(100, 25) )
        self.Bind( wx.EVT_BUTTON, self.event_select_img_folder, self.select_img_folder)

    # Dropdown menus
        # Microscope
        wx.StaticText(panel,label="Microscope:", pos=(I_LEFT,I_TOP+ROW))
        self.microscope = wx.ComboBox(panel, choices = microscopes, style=wx.CB_READONLY,
                                      pos=(I_LEFT,I_TOP+ROW+20), size=(170,-1) )
        item = self.microscope.FindString(default_microscope)
        self.microscope.SetSelection(item)

        # Laser
        wx.StaticText(panel,label="Laser:", pos=(I_LEFT+COL,I_TOP+ROW))
        self.laser = wx.ComboBox(panel, choices = lasers, style=wx.CB_READONLY,
                                      pos=(I_LEFT+COL,I_TOP+ROW+20), size=(170,-1) )
        item = self.laser.FindString(default_laser)
        self.laser.SetSelection(item)

        # Layer
        wx.StaticText(panel,label="Layer:", pos=(I_LEFT+2*COL,I_TOP+ROW))
        self.layer = wx.ComboBox(panel, choices = layers, style=wx.CB_READONLY,
                                      pos=(I_LEFT+2*COL,I_TOP+ROW+20), size=(170,-1) )
        item = self.layer.FindString(default_layer)
        self.layer.SetSelection(item)

     # Checkboxes

        # Areas
        wx.StaticText(panel,label="Areas:", pos=(I_LEFT,I_TOP+2*ROW))
        self.areas = list()    # list with checkboxes
        for i in range(4):
            self.areas.append(
                        wx.CheckBox(panel, label=str(i+1),
                                     pos = (I_LEFT + i*40, I_TOP+2*ROW+30), size=(40,20) )
                    )

        # Channels
        wx.StaticText(panel,label="Channels:", pos=(I_LEFT+COL,I_TOP+2*ROW))
        self.channels = list()    # list with checkboxes
        for i in range(2):
            self.channels.append(
                        wx.CheckBox(panel, label=str(i+1),
                                     pos = (I_LEFT+COL + i*40, I_TOP+2*ROW+30), size=(40,20) )
                    )

        # Number of planes
        wx.StaticText(panel,label="Number of planes:", pos=(I_LEFT+2*COL,I_TOP+2*ROW))
        self.planes = wx.TextCtrl(panel, pos=(I_LEFT+2*COL,I_TOP+2*ROW+20), size=(170,-1))
        self.planes.SetValue( default_planes )


        # Parameter file
        self.img_parameter = wx.CheckBox(panel, label='Parameter File',
                                     pos = (I_LEFT, I_TOP+4*ROW+20), size=(130,20) )

        # Submit imaging button
        self.submit_img_button = wx.Button(panel,label="Submit imaging",
                                       pos=(I_LEFT+2*COL, I_TOP+5*ROW),
                                       size=(150, 50) )
        self.Bind( wx.EVT_BUTTON, self.event_submit_img, self.submit_img_button)
        """
# =============================================================================
# Submit and close buttons
# =============================================================================

        self.transfer_button = wx.Button(panel,label="Transfer data",
                                       pos=(30, L_TOP),
                                       size=(BUTTON_WIDTH, BUTTON_HEIGHT) )
        self.Bind( wx.EVT_BUTTON, self.event_transfer_data, self.transfer_button)

        self.quit_button = wx.Button(panel,label="Quit",
                                     pos=(30, L_TOP+ROW),
                                     size=(BUTTON_WIDTH, BUTTON_HEIGHT) )
        self.Bind( wx.EVT_BUTTON, self.event_quit_button, self.quit_button)

        # status text
        self.status_text = wx.TextCtrl(panel, value="Status updates will appear here:\n",
                                       style=wx.TE_MULTILINE,
                                       pos=(S_LEFT+COL, L_TOP),
                                       size=(WINDOW_WIDTH-S_LEFT-COL-30, WINDOW_HEIGHT-L_TOP-30 ) )


# =============================================================================
# Events for menus and button presses
# =============================================================================

    def event_mouse_selected(self, event):
        """ The user selected a mouse name in the dropdown menu """
        print('New mouse selected')

    def event_submit_session(self, event):
        """ The user clicked on the button to submit a session """

        # create session dictionary that can be entered into datajoint pipeline
        session_dict = dict(mouse_id = self.mouse_name.GetValue(),
                            day = self.day.GetValue(),
                            trial = int( self.trial.GetValue() ),
                            anesthesia = self.anesthesia.GetValue(),
                            setup = self.setup.GetValue(),
                            task = self.task.GetValue(),
                            stage = int( self.stage.GetValue() ),
                            username = self.experimenter.GetValue(),
                            notes = self.notes.GetValue()
                            )

        # save dictionary that is entered
        identifier = common_exp.Session().create_id(investigator_name=session_dict['username'],
                                                    mouse_id=session_dict['mouse_id'],
                                                    date=session_dict['day'],
                                                    trial=session_dict['trial'])
        file = os.path.join(path_backup_npy, identifier+'.npy')
        # np.save(file, session_dict)

        # check if the session is already in the database (most common error)
        key = dict(name=self.mouse_name.GetValue(),
                   day=self.day.GetValue(),
                   trial=int( self.trial.GetValue() ) )
        if len( common_exp.Session() & key) > 0:
            message = 'The session you wanted to enter into the database already exists.\n' + \
                      'Therefore, nothing was entered into the databse.'
            wx.MessageBox(message, caption="Session already in database" ,style=wx.OK | wx.ICON_INFORMATION)
            return

        # add entry to database
        try:
            common_exp.Session().helper_insert1( session_dict )
            self.status_text.write('Sucessfully entered new session: ' + str(key) + '\n')
        except Exception as ex:
            print('Exception manually caught:', ex)
            self.status_text.write('Error: '+ str(ex) +'\n')

    def event_load_session(self, event):
        """ User wants to load additional information about session into GUI """

        session_dict = dict( name = self.mouse_name.GetValue(),
                            day = self.day.GetValue(),
                            trial = int( self.trial.GetValue() ) )
        entries = (common_exp.Session() & session_dict ).fetch(as_dict=True)
        # check there is only one table corresponding to this
        if len(entries) != 1:
            self.status_text.write('Can not load session info for {} because there are {} sessions corresponing to this'.format(session_dict, len(entries))+'\n')
            return

        entry = entries[0]

        # set the selections in the menus according to the loaded info
        item = self.setup.FindString(entry['setup'])
        self.setup.SetSelection(item)
        item = self.task.FindString(entry['task'])
        self.task.SetSelection(item)
        self.stage.SetValue( str(entry['stage']) )
        item = self.anesthesia.FindString(entry['anesthesia'])
        self.anesthesia.SetSelection(item)
        item = self.experimenter.FindString(entry['experimenter'])
        self.experimenter.SetSelection(item)
        self.notes.SetValue( entry['notes'] )



    def event_submit_behavior(self, event):
        """ User clicked on buttion to submit the behavioral data """

        # go through all behavioral files and upload if checkbox is set
        session_key = dict( name = self.mouse_name.GetValue(),
                            day = self.day.GetValue(),
                            trial = int( self.trial.GetValue() ) )
        trial = int( self.trial.GetValue() )
    # wheel
        if self.wheel_checkbox.GetValue():
            # insert the main table
            wheel_dict = dict( **session_key,
                              wheel_type = self.wheel.GetValue())
            self.save_insert( common_behav.Wheel(), wheel_dict)

            # add job to transfer file later
            file = wheel_file.format(trial)
            raw_wheel_dict = dict( **session_key,
                                  file_name = file)

            self.job_list.append( [common_behav.RawWheelFile(), raw_wheel_dict, self.folder.GetValue(), file])
    # synchronization
        if self.sync_checkbox.GetValue():
            sync_dict = dict(**session_key,
                             sync_type=self.imaging_sync.GetValue())
            self.save_insert( common_behav.Synchronization(), sync_dict)

            file = sync_file.format(trial)
            raw_sync_dict = dict(**session_key,
                                 file_name=file)
            self.job_list.append( [common_behav.RawSynchronizationFile(), raw_sync_dict, self.folder.GetValue(), file])
    # video
        if self.video_checkbox.GetValue():
            video_dict = dict(**session_key,
                              camera_nr = 0,   # TODO: modify for multiple cameras
                              camera_position=self.camera_pos.GetValue(),
                              frame_rate = int( self.frame_rate.GetValue() ) )
            self.save_insert( common_behav.Video(), video_dict)

            file = self.video_file.GetValue().format(trial)
            raw_video_dict = dict( **session_key,
                                 camera_nr = 0,
                                 part = 0,
                                 file_name = file)
            self.job_list.append( [common_behav.RawVideoFile(), raw_video_dict, self.folder.GetValue(), file])
    # whisker stimulator
        if self.whisker_checkbox.GetValue():
            whisker_dict = dict(**session_key,
                             stimulator_type=self.stimulator.GetValue())
            self.save_insert( common_behav.WhiskerStimulator(), whisker_dict)

            file = whisker_file.format(trial)
            raw_whisker_dict = dict(**session_key,
                                 file_name=file)
            self.job_list.append( [common_behav.RawWhiskerStimulatorFile(), raw_whisker_dict, self.folder.GetValue(), file])
    # events
        if self.event_checkbox.GetValue():
            event_dict = dict(**session_key,
                              sensory_event_type = self.event_type.GetValue() )
            self.save_insert( common_behav.SensoryEvents(), event_dict)

            file = event_file.format(trial)
            raw_event_dict = dict(**session_key,
                                 file_name=file)
            self.job_list.append( [common_behav.RawSensoryEventsFile(), raw_event_dict, self.folder.GetValue(), file])

    """ COMMENTED OUT BECAUSE WE DONT HAVE THE IMG() OR EL() SCHEMA YET
    def event_submit_img(self, event):
        """ 'User clicked on submit imaging info button' """

        # create session key
        trial = int( self.trial.GetValue() )
        session_key = dict( name = self.mouse_name.GetValue(),
                            day = self.day.GetValue(),
                            trial = trial )

        # get active areas and channels in easier data format
        areas = [ checkbox.GetValue() for checkbox in self.areas]
        channels = [checkbox.GetValue() for checkbox in self.channels]

        # insert new entry in the database with details
        scan_dict = dict( **session_key,
                         microscope = self.microscope.GetValue(),
                         laser = self.laser.GetValue(),
                         layer = self.layer.GetValue(),
                         nr_channels = sum( channels ),
                         nr_areas = sum( areas ),
                         nr_planes = int( self.planes.GetValue() )
                         )
        self.save_insert( img.Scan(), scan_dict)

        if SKIP_IMG_TRANSFER:
            return  # stop after populating main table

        if scan_dict['microscope'] == 'Scientifica' and ( sum(areas) != 1 or sum(channels) != 1):
            raise Exception('Multiple areas or channels not supported at Scientifica')

        # copy files from multi-area imaging
        for area, copy_area in enumerate(areas):
            for channel, copy_channel in enumerate(channels):

                # only copy the file if both are selected
                if copy_area and copy_channel:
                    if scan_dict['microscope'] == 'H45':
                        file_type = H45_file.format(area,channel)  # e.g. H45_file = 'test_A{}_Ch{}_ ????.tif'
                    else:
                        file_type = scientifica_file.format(trial)  # e.g. '{:02d}_img_00001_000??.tif'

                    folder = self.img_folder.GetValue()
                    matching_paths = sorted( glob.glob( os.path.join(folder, file_type) ) )

                    # fix problem on multi-area microscope, that one empty file is generated between true files
                    if (scan_dict['microscope'] == 'H45'):
                        matching_paths = matching_paths[::2]  # only every second file

                    print('Found the following files matching the imaging file type:\n', matching_paths)

                    for i, matching_path in enumerate(matching_paths):
                        # now we can finally create a job for this...
                        file = os.path.basename( matching_path )

                        raw_imaging_dict = dict(**session_key,
                                           area = area,
                                           channel=channel,
                                           part=i,
                                           file_name = file)
                        self.job_list.append( [img.RawImagingFile(), raw_imaging_dict, folder, file] )

                else:
                    # TODO: include optional delete of not copied files
                    pass

        # copy parameter file
        if self.img_parameter.GetValue():

            folder = self.img_folder.GetValue()
            file = parameter_file

            raw_parameter_dict = dict( **session_key,
                                      file_name = file)

            self.job_list.append( [img.RawParameterFile(), raw_parameter_dict, folder, file] )


    def event_submit_el(self, event):
        """ 'User clicked on buttion to submit the E-phys data' """

        # go through all behavioral files and upload if checkbox is set
        session_key = dict( name = self.mouse_name.GetValue(),
                            day = self.day.GetValue(),
                            trial = int( self.trial.GetValue() ) )
        # insert the main table
        ephys_dict = dict( **session_key,
                          probe_type = self.probe.GetValue())
        self.save_insert( el.Ephys(), ephys_dict)

    # probe data
        if self.probe_checkbox.GetValue():

            # insert parent table for 16 channels
            master_entry = dict( **session_key,
                                  file_type = 'Probe',
                                  file_name = 'see_part_table')
            self.save_insert( el.RawEphysFiles(), master_entry )

            folder = self.el_folder.GetValue()
            matching_paths = sorted( glob.glob( os.path.join(folder, probe_files) ) )

            if len(matching_paths) != 16:  # quick check if the hardcoding worked...
                raise Exception('The mask resulted in {} instead of 16 results:\n{}'.format( len(matching_paths), matching_paths ))

            file_list = [os.path.basename(path) for path in matching_paths] # extract file name from path
            print(file_list)

            for file in file_list:
                # hardcoded extraction of the intan channel number
                try:
                    intan_nr = int( file[6:9] )
                except:
                    raise Exception('Hardcoded extraction of intan channel number did not work.')

                # map intan number onto channel number (see schema/utils/mapping_intan_16chanProbe.png)
                from schema.utils import intan
                channel_nr = intan.intan_to_probe( intan_nr )

                part_entry = dict( **session_key,
                                    file_type = 'Probe',
                                    channel_nr = channel_nr,
                                    file_name = file)

                self.job_list.append( [el.RawEphysFiles.Channel(), part_entry, folder, file])

    # sync data with imaging (e.g. galvo recording)
        if self.sync_el_checkbox.GetValue():
            file = default_el_sync_file     # TODO: make more general
            sync_dict = dict(**session_key,
                             file_type = self.el_sync.GetValue(),
                             file_name = file)
            self.job_list.append( [el.RawEphysFiles(), sync_dict, self.el_folder.GetValue(), file])

    # shutter from imaging
        if self.shutter_checkbox.GetValue():
            file = self.shutter_file.GetValue()
            shutter_dict = dict(**session_key,
                             file_type = 'Shutter',
                             file_name = file)
            self.job_list.append( [el.RawEphysFiles(), shutter_dict, self.el_folder.GetValue(), file])

    # Info file with additional information about recording
        if self.info_checkbox.GetValue():
            file = info_file     # TODO: make more general
            info_dict = dict(**session_key,
                             file_type = 'Info',
                             file_name = file)
            self.job_list.append( [el.RawEphysFiles(), info_dict, self.el_folder.GetValue(), file])
    """


    def event_select_folder(self, event):
        """ User clicked on select folder button """

        # open file dialog
        folder_dialog = wx.DirDialog (parent=None, message="Choose directory of behavioral data",
                                      defaultPath=default_folder, style = wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        exit_flag = folder_dialog.ShowModal()
        # update the folder in the text box when the user exited with ok
        if exit_flag == wx.ID_OK:
            self.folder.SetValue( folder_dialog.GetPath() )


    def event_select_img_folder(self, event):
        """ User clicked on select imaging folder button """

        # open file dialog
        folder_dialog = wx.DirDialog (parent=None, message="Choose directory of imaging data",
                                      defaultPath=default_img_folder, style = wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        exit_flag = folder_dialog.ShowModal()
        # update the folder in the text box when the user exited with ok
        if exit_flag == wx.ID_OK:
            self.img_folder.SetValue( folder_dialog.GetPath() )

    def event_select_el_folder(self, event):
        """ User clicked on select electrical folder button """

        # open file dialog
        folder_dialog = wx.DirDialog (parent=None, message="Choose directory of electrical recording data",
                                      defaultPath=default_el_folder, style = wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        exit_flag = folder_dialog.ShowModal()
        # update the folder in the text box when the user exited with ok
        if exit_flag == wx.ID_OK:
            self.el_folder.SetValue( folder_dialog.GetPath() )

    def event_quit_button(self, event):
        """ User pressed quit button """
        self.Close(True)

    def event_transfer_data(self, event):
        """ Transfer data to the neurophysiology server and add Raw...File to database """

        jobs = self.job_list
        print('Transfering the files for {} entries in the job list.'.format( len(jobs) ))

        while jobs:  # do as long as the list is not empty
            job = jobs.pop(0)     # get first job (FIFO)

            # extract variables for better code readablility
            table = job[0]
            new_entry = job[1]
            source_folder = job[2]
            file_name = job[3]

            # enter Raw...File into database
            successful = self.save_insert( table, new_entry )

            if not successful:
                continue   # jump to next job and do not transfer the data for this

            # transfer file to neurophysiology folder
            source_path = os.path.join( source_folder, file_name)

            folder_id = common_exp.Session().create_id(new_entry['name'], new_entry['day'], new_entry['trial'])
            target_folder = os.path.join( path_neurophysiology, folder_id )

            target_path = os.path.join( target_folder, file_name)

            # create directory if it does not exists
            if not os.path.exists( target_folder ):
                os.makedirs( target_folder )

            # print the size of the file if it is large and will take a while
            size = os.path.getsize( source_path )
            if size > 1e7:  # 10 MB
                print('Copying large file of size {:.2f}GB...'.format(size/1e9) +'\n')
                print('File:', source_path)

            # copy the file to the neurophysiology directory
            if os.name == 'nt':   # windows pc
                os.system( 'copy "{}" "{}" '.format( source_path, target_path))
            else:  # assuming linux
                os.system( 'cp "{}" "{}" '.format( source_path, target_path))

            print('Copied file ' + source_path +'\n')

        print('Transfer ended.')


    def save_insert(self, table, dictionary):
        """
        Returns True, if the new_entry could be successfully entered into the database
        """

        try:
            table.insert1( dictionary )
            self.status_text.write('Sucessfully entered new entry: ' + str(dictionary) +'\n')
            return True

        except Exception as ex:
            print('Exception manually caught:', ex)
            self.status_text.write('Error while entering ' + str(dictionary) + ' : '+ str(ex) +'\n')
            return False



# run the GUI
if __name__ == '__main__':
    app = wx.App()
    frame = window(parent=None,id=-1)
    frame.Show()
    app.MainLoop()