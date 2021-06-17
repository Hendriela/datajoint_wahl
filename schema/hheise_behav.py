""" Schema to store behavioral data of Hendriks VR task"""
import login
login.connect()
import datajoint as dj
from schema import common_exp as exp
from schema import common_mice as mice

# from .utils.common import log    # standardized logging
# from .utils import analysis
from datetime import datetime

import os
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
class VRSession(dj.Imported):
    definition = """ # Info about the VR Session
    -> exp.Session
    ---
    condition_switch    : longblob          # List of ints indicating trial numbers at which the condition switched
    valve_duration      : smallint          # Duration of valve opening during reward in ms
    length              : smallint          # Track length in cm
    running             : enum('none', 'very bad', 'bad', 'medium', 'okay', 'good', 'very good') 
    licking             : enum('none', 'very bad', 'bad', 'medium', 'okay', 'good', 'very good') 
    deprivation         : varchar(256)      # Water deprivation status before the session
    block               : tinyint           # sessions of one batch can be sub-grouped in blocks (e.g. multiple strokes)
    vr_notes            : varchar(1024)     # Notes about the session
    """

    key = {'username': 'hheise', 'mouse_id': 0, 'day': datetime.strptime('1900-1-1', '%Y-%m-%d'), 'trial': 1}

    def make(self, key):
        # Get current mouse
        mouse = (mice.Mouse & key).fetch1()

        mouse['batch'] = 7          ##### DELETE LATER

        # Load info from the Excel file
        excel_path = os.path.join(login.get_neurophys_data_directory(),
                                  (BatchData & {"batch_id": mouse['batch']}).fetch1('behav_excel'))
        excel = pd.read_excel(excel_path, sheet_name="M{}".format(mouse['mouse_id']))

        key['day'] = datetime.strptime('2021-06-17', '%Y-%m-%d') ##### DELETE LATER

        sess_entry = excel.loc[excel['Date'] == key['day']]

        # Fill in info from Excel entry
        key['valve_duration'] = sess_entry['Water'][0].split()[1][:3]
        key['length'] = sess_entry['Track length'][0]
        key['running'] = sess_entry['Running'][0]
        key['licking'] = sess_entry['Licking'][0]
        key['deprivation'] = sess_entry['Deprivation'][0]
        key['vr_notes'] = sess_entry['Notes'][0]

        # Enter weight if given
        if not pd.isna(sess_entry['weight [g]'][0]):
            mice.Weight().insert1({'username': key['username'], 'mouse_id': key['mouse_id'],
                                   'date_of_weight': key['day'], 'weight': sess_entry['weight [g]'][0]})

        # TODO: get condition_switch and block from GUI


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
class VRTrial(dj.Computed):
    definition = """ # Single trial of VR data
    -> VRSession
    trial_id            : tinyint           # Counter of the trial in 
    ---
    -> CorridorPattern
    condition_switch    : longblob          # List of ints indicating trial numbers at which the condition switched
    reward_duration     : smallint          # Duration of valve opening during reward
    length              : smallint          # Track length in cm
    running             : enum('none', 'very bad', 'bad', 'medium', 'okay', 'good', 'very good') 
    licking             : enum('none', 'very bad', 'bad', 'medium', 'okay', 'good', 'very good') 
    deprivation         : varchar(256)      # Water deprivation status before the session
    vr_notes            : varchar(1024)     # Notes about the session
    """


@schema
class RawTCPFile(dj.Manual):
    definition = """ # Filename of the unprocessed TCP (VR position) file for each trial
    -> VRTrial
    ---
    tcp_filename        : varchar(128)      # filename of the TCP file
    """


@schema
class RawTDTFile(dj.Manual):
    definition = """ # Filename of the unprocessed TDT (licking and frame trigger) file for each trial
    -> VRTrial
    ---
    tdt_filename        : varchar(128)      # filename of the TDT file
    """


@schema
class RawEncFile(dj.Manual):
    definition = """ # Filename of the unprocessed Encoder (running speed) file for each session
    -> VRTrial
    ---
    enc_filename        : varchar(128)      # filename of the TCP file
    """


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