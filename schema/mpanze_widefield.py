"""Schema for widefield imaging related files and information"""

import datajoint as dj
import login
from schema import common_mice, common_exp

schema = dj.schema('mpanze_widefield', locals(), create_tables=True)