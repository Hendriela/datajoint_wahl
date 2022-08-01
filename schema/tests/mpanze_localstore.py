# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 19:18:30 2022

@author: mpanze
"""

import datajoint as dj
import login
login.connect()

schema = dj.schema('mpanze_localstore', locals(), create_tables=True)


@schema
class TestTable(dj.Manual):
    definition = """ # Example table for storing data in 'datastore' external storage
    row_id              : int  # primary_key
    ---
    data                : blob@datastore         # blobs and longblobs inserted here will be stored externally
    """

# # example code
# import numpy as np
# data = np.random.randint(42, size=(4,3,100))
# TestTable().insert1([0, data])
# print((TestTable()&"row_id=0").fetch1())