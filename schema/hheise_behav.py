#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 01/06/2021 13:25
@author: hheise

Schema to store behavioral data of Hendriks VR task
"""

import login
login.connect()
import datajoint as dj
from schema import common_exp as exp
from schema import common_mice as mice
from schema import hheise_img as img

from datetime import datetime
import ast
import os
from glob import glob
import numpy as np
import pandas as pd

schema = dj.schema('hheise_behav', locals(), create_tables=True)

@schema
class BatchData(dj.Manual):
    definition = """ # Filename of the unprocessed LOG file for each session
    batch_id            : tinyint           # ID of the batch, corresponds to "batch" attribute in common_mice.Mouse
    ---
    behav_excel         : varchar(128)      # relative filepath (from neurophys data) of the behavior_evaluation Excel
    """

@schema
class CorridorPattern(dj.Lookup):
    definition = """ # Different types of VR corridor patterns and RZ locations
    pattern             : varchar(128)      # Description of pattern
    ---
    positions           : longblob          # Start and end of RZs in VR coordinates
    """
    contents = [
        ['none', [[-6, 4], [26, 36], [58, 68], [90, 100]]],
        ['none_shifted', [[-6, 4], [34, 44], [66, 76], [90, 100]]],
        ['training', [[-6, 4], [26, 36], [58, 68], [90, 100]]],
        ['training_shifted', [[-6, 4], [34, 44], [66, 76], [90, 100]]],
        ['novel', [[9, 19], [34, 44], [59, 69], [84, 94]]]
    ]

@schema
class RawTCPFile(dj.Manual):
    definition = """ # Filename of the unprocessed TCP (VR position) file for each trial
    -> VRSession.VRTrial
    ---
    tcp_filename        : varchar(128)      # filename of the TCP file
    """

@schema
class RawTDTFile(dj.Manual):
    definition = """ # Filename of the unprocessed TDT (licking and frame trigger) file for each trial
    -> VRSession.VRTrial
    ---
    tdt_filename        : varchar(128)      # filename of the TDT file
    """

@schema
class RawEncFile(dj.Manual):
    definition = """ # Filename of the unprocessed Encoder (running speed) file for each trial
    -> VRSession.VRTrial
    ---
    enc_filename        : varchar(128)      # filename of the Enc file
    """

@schema
class VRSession(dj.Imported):
    definition = """ # Info about the VR Session
    -> exp.Session
    ---
    imaging_session     : tinyint           # bool flag whether imaging was performed during this session
    condition_switch    : longblob          # List of ints indicating trial numbers at which the condition switched
    valve_duration      : smallint          # Duration of valve opening during reward in ms
    length              : smallint          # Track length in cm
    running             : enum('none', 'very bad', 'bad', 'medium', 'okay', 'good', 'very good') 
    licking             : enum('none', 'very bad', 'bad', 'medium', 'okay', 'good', 'very good') 
    deprivation         : varchar(256)      # Water deprivation status before the session
    block               : tinyint           # sessions of one batch can be sub-grouped in blocks (e.g. multiple strokes)
    vr_notes            : varchar(1024)     # Notes about the session
    """

    class VRTrial(dj.Part):
        definition = """ # Single trial of VR data
        -> VRSession
        trial_id            : tinyint           # Counter of the trial in 
        ---
        -> CorridorPattern
        tone                : tinyint           # bool flag whether the RZ tone during the trial was on (1) or off (0)
        pos                 : longblob          # 1d array with VR position sampled every 0.5 ms
        lick                : longblob          # 1d array with licks sampled every 0.5 ms
        frame               : longblob          # 1d array with frame triggers sampled every 0.5 ms
        enc                 : longblob          # 1d array with encoder data sampled every 0.5 ms
        valve               : longblob          # 1d array with valve openings (reward) sampled every 0.5 ms
        """

    def make(self, key):

        # Safety check that only my sessions are processed (should be restricted during the populate() call)
        if key['username'] != login.get_user():
            return

        # First, insert session information into the VRSession table
        self.insert_vr_info(key)

        self.populate_trials()



    def insert_vr_info(self, key):
        """Fills VRSession table with basic info about the session, mainly from Excel file"""
        # Save original key
        orig_key = key.copy()

        # Get current mouse
        mouse = (mice.Mouse & key).fetch1()

        # Load info from the Excel file
        excel_path = os.path.join(login.get_neurophys_data_directory(),
                                  (BatchData & {"batch_id": mouse['batch']}).fetch1('behav_excel'))
        excel = pd.read_excel(excel_path, sheet_name="M{}".format(mouse['mouse_id']))
        # Day is returned as date, has to be cast as datetime for pandas comparison
        sess_entry = excel.loc[excel['Date'] == datetime(key['day'].year, key['day'].month, key['day'].day)]

        # Fill in info from Excel entry
        key['valve_duration'] = sess_entry['Water'].values[0].split()[1][:3]
        key['length'] = sess_entry['Track length'].values[0]
        key['running'] = sess_entry['Running'].values[0]
        key['licking'] = sess_entry['Licking'].values[0]
        key['deprivation'] = sess_entry['Deprivation'].values[0]
        key['vr_notes'] = sess_entry['Notes'].values[0]

        # Enter weight if given
        if not pd.isna(sess_entry['weight [g]'].values[0]):
            mice.Weight().insert1({'username': key['username'], 'mouse_id': key['mouse_id'],
                                   'date_of_weight': key['day'], 'weight': sess_entry['weight [g]'].values[0]})

        # Get block and condition switch from session_notes string
        note_dict = ast.literal_eval((exp.Session & key).fetch1('session_notes'))
        key['block'] = note_dict['block']
        key['condition_switch'] = note_dict['switch']

        # Check if this is an imaging session (session has to be inserted into hheise_img.Scan() first)
        if len(img.Scan & key) == 1:
            key['imaging_session'] = 1
        else:
            key['imaging_session'] = 0

        # Get filename of this session's LOG file (path is relative to the session directory)
        log_name = glob(os.path.join(login.get_neurophys_data_directory(),
                                     (exp.Session & key).fetch1('session_path'),
                                     'TDT LOG_*'))

        if len(log_name) == 0:
            raise Warning('No LOG file found for M{} session {}!'.format(mouse['mouse_id'], key['day']))
        elif len(log_name) > 1:
            raise Warning('{} LOG files found for M{} session {}!'.format(len(log_name), mouse['mouse_id'], key['day']))
        else:
            orig_key['log_filename'] = os.path.basename(log_name[0])
            VRLogFile.insert1(orig_key)         # Insert the filename (without path) into the responsible table
            VRLog.populate()                    # Load the LOG file into the other table

        # Insert final dict into the table
        self.insert1(key)

    def populate_trials(self):
        """
        Find raw behavior files, insert filenames into respective tables, load data, align it and insert it into the
        VRTrial table.
        :param root:            str, path to the session folder (includes behavioral .txt files)
        :param imaging:         boolean flag whether this was an imaging trial (.tif files exist)
        :param verbose:         boolean flag whether unnecessary status updates should be printed to the console
        """

        def find_file(tstamp, file_list):
            """
            Finds a file with the same timestamp from a list of files.
            """
            time_format = '%H%M%S'
            time_stamp = datetime.strptime(str(tstamp), time_format)
            matched_file = []
            for filename in file_list:
                curr_stamp = datetime.strptime(filename.split('_')[-1][:-4], time_format)
                diff = time_stamp - curr_stamp
                if abs(diff.total_seconds()) < 3:
                    matched_file.append(filename)
            if len(matched_file) == 0:
                print(f'No files with timestamp {tstamp} found in {root}!')
                return
            elif len(matched_file) > 1:
                print(f'More than one file with timestamp {tstamp} found in {root}!')
                return
            else:
                return matched_file[0]
    def


@schema
class VRLogFile(dj.Manual):
    definition = """ # Filename of the unprocessed LOG file for each session
    -> VRSession
    ---
    log_filename        : varchar(128)      # filename of the LOG file (should be inside the session directory)
    """


@schema
class VRLog(dj.Imported):
    definition = """ # Processed LOG data
    -> VRLogFile
    ---
    log                 : longblob        # np.array of imported LOG data
    """

    def make(self, key):
        # Load LOG file
        log = pd.read_csv(os.path.join(login.get_neurophys_data_directory(),
                                       (exp.Session & key).fetch1('session_path'), key['log_filename']))

        # Validate mouse and track length info
        line = log['Event'].loc[log['Event'].str.contains("VR Task start, Animal:")].values[0]
        log_length = int(line.split('_')[1])
        log_mouse = int(line.split('_')[0].split()[-1][1:])
        tab_length = (VRSession & key).fetch1('length')
        if log_length != tab_length:
            raise Warning('Session {}:\nTrack length {} in LOG file does not correspond to length {} in '
                          'database.'.format(key, log_length, tab_length))
        if log_mouse != key['mouse']:
            raise Warning('Session {}: Mouse ID M{} in LOG file does not correspond to ID in '
                          'database M{}'.format(key['day'], log_mouse, key['mouse']))

        key['log'] = log
        self.insert1(key)


@schema
class Behavior(dj.Imported):
    definition = """ # Processed and merged TCP, TDT and Encoder data
    -> RawTCPFile
    -> RawTDTFile
    -> RawEncFile
    ---
    pos                 : longblob          # 1d array with VR position sampled every 0.5 ms
    lick                : longblob          # 1d array with licks sampled every 0.5 ms
    frame               : longblob          # 1d array with frame triggers sampled every 0.5 ms
    enc                 : longblob          # 1d array with encoder data sampled every 0.5 ms
    valve               : longblob          # 1d array with valve openings (reward) sampled every 0.5 ms
    """
    #todo: Try to reduce sampling rate to reduce file sizes?