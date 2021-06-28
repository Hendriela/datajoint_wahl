#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 24/06/2021 16:18
@author: hheise

Schema to store 2-photon imaging of Hendriks VR task
"""

import login
login.connect()
import datajoint as dj
from schema import common_exp as exp
from schema import common_mice as mice

schema = dj.schema('hheise_img', locals(), create_tables=True)

@schema
class Scan(dj.Manual):
    definition = """ # Info about the imaging scan session
    -> exp.Session
    ---

    """