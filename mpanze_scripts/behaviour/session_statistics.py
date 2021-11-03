import numpy as np
import matplotlib.pyplot as plt


def load_widefield_sync(exp_stem):
    path = exp_stem + "_widefield.bin"
    data = np.fromfile(path)
    t = data[::3]
    ch_1 = data[1::3]
    ch_2 = data[2::3]
    return t, ch_1, ch_2


def get_frame_timestamps(t, channel, plot=False):
    # convert trigger trace to nice square wave
    channel_thresh = (channel > 0.5).astype(np.float64)
    diff = np.diff(channel_thresh)

    # identify frame timestamps, by averaging timestamps when camera has active exposure
    frames = []
    sequence = []
    for i in range(len(t[1:])):
        if diff[i] == -1:
            sequence = np.array(sequence)
            frames.append(np.mean(sequence[sequence != 0]))
            sequence = []
        sequence.append(channel_thresh[1:][i] * t[1:][i])

    # discard first 2 "frames", which are due to camera starting up, not actual data acquisition
    frames_widefield = np.array(frames[2:])
    exposure_mean, exposure_std = np.mean(np.diff(frames_widefield)), np.std(np.diff(frames_widefield))

    if plot:
        plt.figure()
        plt.plot(t, channel_thresh, 'r-')
        plt.plot(frames_widefield, np.ones(len(frames_widefield)), 'k.')
        plt.show()

    print("%i frame timestamps from widefield identified!" % len(frames_widefield), end=' ')
    print("exposure time is %f +/- %f ms equivalent to %f fps" % (exposure_mean*1000, exposure_std*1000,
                                                                  1 / exposure_mean))

    return frames_widefield, exposure_mean, exposure_std

def load_joystick(exp_stem, averaging=1):
    path = exp_stem + "_joystick.bin"
    data = np.fromfile(path)
    t = data[::3]
    x = data[1::3]
    y = data[2::3]

    if averaging > 1:
        new_lim = int(averaging/2)
        t = t[new_lim:-new_lim]
        x = np.convolve(x, np.ones(averaging)/averaging, mode="same")[new_lim:-new_lim]
        y = np.convolve(y, np.ones(averaging)/averaging, mode="same")[new_lim:-new_lim]

    return t, x, y

# TODO: joystick calibration
#def calibrate_joystick()
#

def load_events(exp_stem):
    path = exp_stem + "_events.txt"
    events = np.genfromtxt(path, skip_header=1, delimiter="\t", dtype="f8,U15", names="timestamps,strings")
    return events

def get_trials(events):
    # return trial timestamps as follows (start of pretrial (2s) - start of trial - end of trial - end of intertrial (+3s), flag (whether or not trial was completed))
    # skip 1st as there is no baseline,and cue doesn't work properly on 1st trial due to Labview bug
    # count number of trials
    trial_start_idx = np.where(events["strings"] == "Trial start")[0]
    trial_completed_idx = np.where(events["strings"] == "Trial completed")[0]
    trial_failed_idx = np.where(events["strings"] == "Trial failed")[0]
    trial_end_idx = np.sort(list(trial_completed_idx) + list(trial_failed_idx))

    n_trials = len(trial_start_idx)

    # cut off 1st trial (bugged audio cue) & last trial (may not be cut-off early)
    trial_timestamps = []
    for idx in trial_start_idx[1:-1]:
        timestamp = 5*[0]
        timestamp[1] = events["timestamps"][idx]
        timestamp[0] = timestamp[1]-2       # 2 seconds of baseline
        # find 1st trial end after idx
        idx_end = trial_end_idx[trial_end_idx > idx][0]
        timestamp[2] = events["timestamps"][idx_end]
        timestamp[3] = timestamp[2] + 3
        if idx_end in trial_completed_idx:
            timestamp[4] = 1
        else:
            timestamp[4] = 0

        trial_timestamps.append(timestamp)

    return np.array(trial_timestamps)


def plot_performance(mouse_dir):
    import pathlib
    import json
    mouse_dir = pathlib.Path(mouse_dir)
    paths = mouse_dir.rglob("*params.json")

    plt.figure()
    i = 0
    ticks = []
    labels = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        # get task
        if data["Task"] != "Pull with cue":
            continue
        date = p.stem.split("_")[1]
        path_stem = str(p)[:-12]
        events = load_events(path_stem)
        timestamps = get_trials(events)
        timestamps = timestamps[timestamps[:,4] == 1]
        #plt.hist(timestamps[:, 2] - timestamps[:, 1], 20)
        #median_response_time = np.median(timestamps[:,2]-timestamps[:,1])
        #ticks.append(i)
        plt.subplot(211)
        plt.plot(i,np.median(timestamps[:,2]-timestamps[:,1]), 'k.')
        plt.subplot(212)
        plt.plot(i, np.std(timestamps[:,2]-timestamps[:,1]), 'r.')
        #plt.plot(i, np.sum(timestamps[:,4])/timestamps.shape[0], 'k.')
        i += 1
        labels.append(date)
    #plt.xticks(ticks, labels, rotation = 45)
    #plt.xlabel("date")
    #plt.ylabel("ratio correct")
    #plt.ylabel("median response time")
    #plt.tight_layout()
    return