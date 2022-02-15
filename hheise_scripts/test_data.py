from schema import common_exp, hheise_behav
from schema import common_mice
import yaml
from datetime import datetime

# Enter demo session
with open(r'W:\Neurophysiology-Storage1\Wahl\Datajoint\manual_submissions\session_hheise_M092_2021-06-22_01.yaml') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
common_exp.Session.insert1(data)

# manual key
key = {'username': 'hheise', 'mouse_id': 81, 'session_num': 1, 'day': datetime.strptime('2021-06-17', '%Y-%m-%d'), 'perf_param_id':0}
hheise_behav.VRSessionInfo().make(key)

# Matteos session that should be ignored by populate()
common_exp.Session.insert1({"username": "mpanze", "mouse_id": 1, "day": '1900-01-01', "session_num": 1, "session_id": 'hheise_M001_1900-01-01_00',
                 "session_path": '\\test_path\\hheise_M001_1900-01-01_00', "session_counter": 1, "anesthesia": "Awake", "setup": "VR",
                 "task": "Passive", "experimenter": "mpanze", "session_notes": "A test session"})

common_exp.Session()
hheise_behav.VRSessionInfo().populate('username="hheise"')

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

#%% Test VRSessionInfo validation
key = {'username': 'hheise', 'mouse_id': 81, 'session_num': 1, 'day': datetime.strptime('2021-06-17', '%Y-%m-%d')}
sess_trials = hheise_behav.VRSessionInfo.VRTrial() & key

lys = []
for i in range(1,7):
    lys.append(dict(pos=(sess_trials & f"trial_id={i}").fetch1('pos'),
                    lick=(sess_trials & f"trial_id={i}").fetch1('lick'),
                    frame=(sess_trials & f"trial_id={i}").fetch1('frame'),
                    enc=(sess_trials & f"trial_id={i}").fetch1('enc'),
                    valve=(sess_trials & f"trial_id={i}").fetch1('valve')))
data_dicts = dict(trial=lys)
#%%

# Test performance calculation (dummy session: Batch 6 M80 20210426
zone_borders = np.array([[-6, 4], [26, 36], [58, 68], [90, 100]])
zone_borders[:, 0] -= 2
zone_borders[:, 1] += 2

file = r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch6\M80\20210426\merged_behavior_154351.txt'
merge = np.loadtxt(file)
lick = merge[:, 2]
pos = merge[:, 1]
enc = merge[:, 4]
valve = merge[:, 6]


results_pickle = pickle.dumps(results)

hheise_behav.CorridorPattern.insert1(dict(pattern='test', positions=results_pickle))
results_pick = (hheise_behav.CorridorPattern & 'pattern="test"').fetch1('positions')
restored = pickle.loads(results_pick)



# Test performance plotting
self = (hheise_behav.VRPerformance * common_mice.Mouse) & 'batch=7'
self = hheise_behav.VRSessionInfo.VRTrial & 'mouse_id=81' & 'day="2021-07-05"'
position = (hheise_behav.VRSessionInfo.VRTrial & 'mouse_id=81' & 'day="2021-07-05"' & 'trial_id=1').fetch1('pos')
enc_path = r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch7\M81\20210705\Encoder data20210705_143133.txt"
pos_path = r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch7\M81\20210705\TCP read data20210705_143134.txt"
trig_path = r"W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch7\M81\20210705\TDT TASK DIG-IN_20210705_143134.txt"

attr = ['frame', 'lick', 'enc', 'pos', 'valve']
# Fetch behavioral data of the current trial, add time scale and merge into np.array
data = (hheise_behav.VRSessionInfo.VRTrial & 'mouse_id=81' & 'day="2021-07-05"' & 'trial_id=1').fetch1(*attr)
# To avoid floating point rounding errors, first create steps in ms (*1000), then divide by 1000 for seconds
time = np.array(range(0, len(data[0]) * int(SAMPLE * 1000), int(SAMPLE * 1000))) / 1000
array = np.vstack((time, *data)).T
position = (hheise_behav.VRSessionInfo.VRTrial & 'mouse_id=81' & 'day="2021-07-05"' & 'trial_id=1').fetch1('pos')



### TEST CELL_MATCH PLOTTING BECAUSE PLOTTING IN CONSOLDE DOES NOT WORK IN THAT PROJECT
import sys
sys.path.append('..\\cell_matching\\')
import cell_match_helper as helper
import matplotlib.pyplot as plt

# Load data of example sessions
ref_key = {'username': 'hheise', 'mouse_id': 93, 'day': '2021-07-08', 'session_num': 1, 'motion_id': 0, 'caiman_id': 0}
tar_key = {'username': 'hheise', 'mouse_id': 93, 'day': '2021-07-09', 'session_num': 1, 'motion_id': 0, 'caiman_id': 0}
session_keys = [None] * 6   # List of Dicts of the primary keys for each session, used to query DataJoint
cor_image = [None] * 6      # List of local correlation images for each session, used as background

for idx, dic in enumerate([ref_key, tar_key]):
    session_keys[idx] = dic
    new_session = helper.fetch_new_session(dic)
    cor_image[idx] = new_session['cor_im']

test = cor_image[0]

fig = px.imshow(matrix, color_continuous_scale=
    [[0.0, '#0d0887'],
     [0.0333333333333333, '#46039f'],
     [0.0444444444444444, '#7201a8'],
     [0.0555555555555555, '#9c179e'],
     [0.0666666666666666, '#bd3786'],
     [0.0888888888888888, '#d8576b'],
     [0.1111111111111111, '#ed7953'],
     [0.1333333333333333, '#fb9f3a'],
     [0.1444444444444444, '#fdca26'],
     [0.1777777777777777, '#f0f921'],
     [0.25, "white"], [0.4, "white"], [0.4, "grey"], [0.5, "grey"],
     [0.5, "red"], [0.6, "red"], [0.6, "green"], [0.7, "green"], [0.7, "pink"], [0.8, "pink"], [0.8, "black"],
     [1, "black"]], range_color=[0, 5])

plt.imshow(test)

cmap = plt.get_cmap('jet')
rgba_img = cmap(cor_image[0])
