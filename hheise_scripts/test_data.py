from schema import common_exp, hheise_behav
from schema import common_mice
import yaml
from datetime import datetime

# Enter demo session
with open(r'W:\Neurophysiology-Storage1\Wahl\Datajoint\manual_submissions\session_hheise_M092_2021-06-22_01.yaml') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
common_exp.Session.insert1(data)

# manual key
key = {'username': 'hheise', 'mouse_id': 81, 'session_num': 1, 'day': datetime.strptime('2021-06-25', '%Y-%m-%d')}
hheise_behav.VRSession().make(key)

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




enc_path = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch7\M90\20210628\Encoder data20210628_113413.txt'
pos_path = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch7\M90\20210628\TCP read data20210628_113414.txt'
trig_path = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch7\M90\20210628\TDT TASK DIG-IN_20210628_113413.txt'
key = dict(username='hheise', mouse_id=90, day='2021-06-28', session_num=1)
self = (hheise_behav.VRLog & trial_key)
log = pd.DataFrame({'log_time': self.fetch1('log_time'),
              'log_trial': self.fetch1('log_trial'),
               'log_event': self.fetch1('log_event')})

