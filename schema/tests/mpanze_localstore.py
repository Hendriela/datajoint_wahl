# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 19:18:30 2022

@author: mpanze
"""

import datajoint as dj
import login
login.connect()

import numpy as np

schema = dj.schema('mpanze_localstore', locals(), create_tables=True)

@schema
class TestTable(dj.Manual):
    definition = """ # Testing local store
    row_id              : int  # primary_key
    ---
    data                : blob@external         # save data in test storage
    """

@schema
class TestTableExt(dj.Manual):
    definition = """ # Testing local store
    row_id              : int  # primary_key
    ---
    data                : blob@data         # save data in test storage
    """


@schema
class TestTableRaw(dj.Manual):
    definition = """ # Testing local store
    row_id              : int  # primary_key
    ---
    data                : blob@teststore         # save data in test storage
    """
