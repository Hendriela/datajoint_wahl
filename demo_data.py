import datajoint as dj

# connect to the database through machine-specific info
import login
login.connect()

mouse_schema = dj.schema('common_mice')
exp_schema = dj.schema('common_exp')

mouse_schema.spawn_missing_classes()
exp_schema.spawn_missing_classes()

# Add mice
Mouse().insert1({"username": "hheise", "mouse_id": 0, "dob": "1900-07-28", "sex": "U", "batch": 0,
                 "strain": "WT", "genotype": "n.d.", "irats_id": "BJ1234", "cage_num": 1100589, "ear_mark": "RLL",
                 "licence_id": "241/2018-B", "info": "This is a dummy test mouse."})

Mouse().insert1({"username": "hheise", "mouse_id": 1, "dob": "2021-01-24", "sex": "M", "batch": 1,
                 "strain": "Snap25-GCaMP6f", "genotype": "+/+", "irats_id": "BJ1235", "cage_num": 1100589, "ear_mark": "RRLL",
                 "licence_id": "241/2018-A", "info": "This is a second dummy test mouse."})

Mouse().insert1({"username": "mpanze", "mouse_id": 0, "dob": "2021-01-01", "sex": "M", "batch": 1,
                 "strain": "L2/3-TIGRE2.0-GCaMP6f", "genotype": "+/+", "irats_id": "BJ1111", "cage_num": 11002312, "ear_mark": "L",
                 "licence_id": "241/2018-A", "info": "Third dummy test mouse"})

# Add weights
Weight().insert1({"username": "hheise", "mouse_id": 1, "date_of_weight": "2021-04-18", "weight": 20.1})
Weight().insert1({"username": "hheise", "mouse_id": 1, "date_of_weight": "2021-04-19", "weight": 20.4})
Weight().insert1({"username": "hheise", "mouse_id": 1, "date_of_weight": "2021-04-20", "weight": 20.9})

# Add surgeries
# #TODO: It might be useful to automatically add a weight entered in a different table (e.g. Surgery) to the Weight() table
Surgery().insert1({"username": "hheise", "mouse_id": 1, "surgery_num": 2, "surgery_date": "2021-04-20",
                   "surgery_type": "microsphere injection", "anesthesia": "triple shot", "weight": 20.9,
                   "stroke_params": 'n.a.', "duration": 30, "surgery_notes": "This is another dummy surgery"})

Injection.insert1({"username": "hheise", "mouse_id": 1, "surgery_num": 2, "injection_num": 1, "substance_name": "endothelin",
                   "volume": 30, "dilution": "10mg in 0.1ml saline", "site": "HPC",
                   "coordinates": "0mm A/P, -1mm M/L, 1.5 mm D/V","injection_notes":"first mock injection"})

Injection.insert1({"username": "hheise", "mouse_id": 1, "surgery_num": 2, "injection_num": 2, "substance_name": "AAV9-hSyn-GCaMP6f",
                   "volume": 0.2, "dilution": "very diluted", "site": "neocortex",
                   "coordinates": "-1mm A/P, -1mm M/L, 0.5 mm D/V","injection_notes":"second mock injection"})