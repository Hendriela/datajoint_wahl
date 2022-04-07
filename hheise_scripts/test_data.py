import login
from schema import common_exp, hheise_behav
from schema import common_mice
import yaml
from datetime import datetime

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


### HYPERPARAMETER-TUNING: DETRENDING FRAME WINDOW
from schema import common_img
import numpy as np
# Find sessions with negative transients (< -0.02)

sessions = common_img.Segmentation().fetch('KEY')

neg_trace = []

for key in sessions:
    traces = np.vstack((common_img.Segmentation.ROI() & key).fetch('dff'))
    neg_trace.append(len(np.where(np.min(traces, axis=1) < -0.35)[0])/len(traces))

traces = np.vstack((common_img.Segmentation.ROI() & sessions[28]).fetch('dff'))
mins = np.argmin(np.min(traces, axis=1))


c = np.load(os.path.join(login.get_neurophys_data_directory(),
                         (common_exp.Session & sessions[28]).fetch1('session_path'),
                         (common_img.Segmentation & sessions[28]).fetch1('traces')))

bad_key = sessions[28].copy()

import matplotlib.pyplot as plt
plt.plot(traces[20])

plt.plot(c[20])

std_params = (common_img.CaimanParameter & bad_key).fetch1()

windows = [200, 500, 1500, 2000, 3000]

for i, w in enumerate(windows):
    std_params['caiman_id'] = i+2
    std_params['frame_window'] = w
    common_img.CaimanParameter().insert1(std_params)

del bad_key['caiman_id']


for id in range(2,7):
    bad_key['caiman_id'] = id
    common_img.Segmentation().populate(bad_key)

# Load different dFF traces:

windows = [500, 1000, 1500, 2000, 3000, 5000, 10000]

dff = []

trace = np.load(f'C:\\Users\\hheise\\Datajoint\\temp\\traces.npy')

for w in windows:
    dff.append(np.load(f'C:\\Users\\hheise\\Datajoint\\temp\\{w}.npy'))


for d, w in zip(dff, windows):
    plt.plot(d[150], label=w)
plt.legend()

data = {'username': 'hheise', 'mouse_id': 85, 'day': '2021-07-08', 'motion_id':0, 'caiman_id':0,
        'trace':trace, '500':dff[0], '1000':dff[0],'1500':dff[0],'2000':dff[0],'3000':dff[0],'5000':dff[0],'10000':dff[0]}

import pickle
with open(r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Tests\dff_window_test.pickle', 'wb') as f:
    pickle.dump(data, f)

### RE-COMPUTE DFF WITH NEW FRAME WINDOW
from schema import common_img, common_exp
from caiman.source_extraction.cnmf import utilities as util
import os
import numpy as np
import login
import matplotlib.pyplot as plt
from scipy import sparse

sessions = (common_img.Segmentation & 'username="hheise"').fetch('KEY')

last_mouse = sessions[0]['mouse_id']

for sess in sessions:

    if sess['mouse_id'] != last_mouse:
        # Update Caiman Parameter set
        print(f'Updating CaimanParameter for mouse {last_mouse}')
        c_ids = (common_img.CaimanParameter() & dict(username=sess['username'], mouse_id=last_mouse)).fetch('caiman_id')
        for c_id in c_ids:
            common_img.CaimanParameter().update1(dict(username=sess['username'],
                                                      caiman_id=c_id, mouse_id=last_mouse,
                                                      frame_window=2000))

    print(f'Processing session {sess}')
    frames_window = (common_img.CaimanParameter & sess).fetch1('frame_window')
    if frames_window != 2000:

        ## Reconstruct data for dF/F
        # Load deconvolved and residual traces
        YrA = np.load(os.path.join(login.get_neurophys_data_directory(),
                                   (common_exp.Session & sess).fetch1('session_path'),
                                   (common_img.Segmentation & sess).fetch1('residuals')))
        C = np.load(os.path.join(login.get_neurophys_data_directory(),
                                   (common_exp.Session & sess).fetch1('session_path'),
                                   (common_img.Segmentation & sess).fetch1('traces'))) - YrA

        # Flatten spatial background components and load temporal ones (stored in database unmodified)
        s_b = (common_img.Segmentation & sess).fetch1('s_background')
        b = np.vstack([s_b[:, :, i].flatten(order='F') for i in range((common_img.CaimanParameter & sess).fetch1('nb'))]).T
        f = (common_img.Segmentation & sess).fetch1('f_background')

        # Transform spatial masks into sparse matrix
        pixels, weights = (common_img.Segmentation.ROI & sess).fetch('pixels', 'weights')
        dims = (common_img.Segmentation & sess).fetch1('target_dim')
        sparse_matrices = []
        for i in range(len(weights)):
            sparse_matrices.append(sparse.csc_matrix((weights[i], (pixels[i], np.zeros(len(pixels[i])))),
                                                     shape=(dims[0] * dims[1], 1)))
        A = sparse.hstack(sparse_matrices)

        dffs, percs = util.detrend_df_f(A, b, C, f, YrA, frames_window=2000)

        print('\tUpdating entries...')
        # Update dFF for the ROIs
        for id, (dff, perc) in enumerate(zip(dffs, percs)):
            common_img.Segmentation.ROI().update1(dict(**sess, mask_id=id, dff=dff, perc=perc))
        print('\tDone!')
    else:
        print('\tdFF frame window is already 2000, skipping session.')
    last_mouse = sess['mouse_id']

