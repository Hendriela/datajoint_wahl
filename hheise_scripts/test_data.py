from schema import common_exp, hheise_behav
from schema import common_mice
import yaml
from datetime import datetime

# Enter demo session
with open(r'W:\Neurophysiology-Storage1\Wahl\Datajoint\manual_submissions\session_hheise_M092_2021-06-22_01.yaml') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
common_exp.Session.insert1(data)

# manual key
key = {'username': 'hheise', 'mouse_id': 92, 'session_num': 1, 'day': '2021-06-22'}

# Matteos session that should be ignored by populate()
common_exp.Session.insert1({"username": "mpanze", "mouse_id": 1, "day": '1900-01-01', "session_num": 1, "session_id": 'hheise_M001_1900-01-01_00',
                 "session_path": '\\test_path\\hheise_M001_1900-01-01_00', "session_counter": 1, "anesthesia": "Awake", "setup": "VR",
                 "task": "Passive", "experimenter": "mpanze", "session_notes": "A test session"})

common_exp.Session()
hheise_behav.VRSession().populate('username="hheise"')

test = [1,2,3,4,5,6]


test_dict = dict(name='Hendrik', age=100)
extend_dict = dict(test_dict, new_field='test')

def change_dict(dic):
    dic['new_entry'] = 'new_value'

change_dict(test_dict)

mouse = (common_mice.Mouse() & 'username="hheise"' & "mouse_id=81").fetch()
mice = (common_mice.Mouse() & 'username="hheise"' & "batch=7" & "sex='F'").fetch('KEY', as_dict=True)
(common_mice.Mouse() & 'username="hheise"' & "batch=7").get_weight_threshold()

key = {'username': 'hheise', 'mouse_id': 81, 'day': '2021-06-17', 'session_num': 1, 'log_filename': 'TDT LOG_20210617_141323.txt'}
insert_dict = dict(key)
insert_dict.pop('log_filename')
insert_dict['log_time'] = np.array(log['Date_Time'], dtype=str)
insert_dict['log_trial'] = np.array(log['Trial'])
insert_dict['log_event'] = np.array(log['Event'])
hheise_behav.VRLog.insert1(insert_dict, allow_direct_insert=True)

hheise_behav.VRSession() & key

log = pd.read_csv(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch7\M81\20210617\TDT LOG_20210617_141323.txt', sep='\t', parse_dates=[[0, 1]])
time1= log['Date_Time'][0]
hheise_behav.DateTimeTest.insert1({'bla':1, 'tyme':time1})

time2 = (hheise_behav.DateTimeTest & 'bla=1').fetch1('tyme')

