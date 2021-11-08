"""Schema for processing sensory mapping data"""

import datajoint as dj
import login
login.connect()

schema = dj.schema('mpanze_mapping', locals(), create_tables=True)