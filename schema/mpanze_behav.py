"""
Schema for processing behavioural data from reach-to-grasp experiments
"""

import datajoint as dj
import login
login.connect()

schema = dj.schema('mpanze_behav', locals(), create_tables=True)