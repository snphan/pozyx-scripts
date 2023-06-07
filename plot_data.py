import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from collections import deque
from PIL import Image
import time
import sys
from config import constants
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.path as mpltPath
import json
import math
import utils
from sklearn.preprocessing import OneHotEncoder
from scipy.signal import find_peaks


def what_location(x, y, regions):
    for k, v in regions.items():
        path = mpltPath.Path(v)
        if path.contains_point([x,y]): return k
    return "undefined"


"""
Plot the data in realtime
"""
####################################################################################################
#                                       CONFIGURATION
####################################################################################################
SAMPLE_RATE = 16 # Hz
SECONDS_SHOW = 3 # s
MAV_WINDOW = 20
NUM_POINTS = SAMPLE_RATE * SECONDS_SHOW * len(constants.REMOTE_IDS) + MAV_WINDOW
REGIONS = json.load(open('2023-03-14 12:15:31.794149.json'))
MODEL_FOLDER = '06_07_Model'
CLF = joblib.load(Path().joinpath('models', MODEL_FOLDER, 'output_model.joblib'))
LOCATION_ENCODER = joblib.load(Path().joinpath('models', MODEL_FOLDER, 'location_encoder.joblib'))
LABEL_ENCODER = joblib.load(Path().joinpath('models', MODEL_FOLDER, 'label_encoder.joblib'))
DATA_TYPES = ['POS', 'LINACC', 'ACC', 'GYRO', 'PRESSURE', 'ORIENT'] # For orientation x = heading, y = roll, z = pitch
COLORS = [
    [255, 154, 162],
    [255, 218, 193],
    [226, 240, 203],
    [255, 183, 178],
    [181, 234, 215],
    [199, 206, 234],
]
bg_options = {
    "ILS": {"path": "ISL HQ Screenshot-rotated.png", "multiplier": 6.3, },
    "GR": {"path": "Glenrose Research First Floor Cropped.png", "multiplier": 14.75}
}
####################################################################################################
data_path = Path(__file__).resolve().parents[0].joinpath("data")

if len(sys.argv) < 2:
    data_file = data_path.joinpath(input("type the name of data file: ") + ".csv")
else: 
    data_file = data_path.joinpath(f"{sys.argv[1]}.csv")

location = input(f'Select location {("/").join(list(bg_options.keys()))}: ')

if location not in bg_options:
    print("Location not found")
    quit()

bg_path = bg_options[location]["path"]
bg_multiplier= bg_options[location]["multiplier"]
mydata = deque([1,2,3,4,5])
img = Image.open(bg_path).convert("L")
img = np.asarray(img)

remote_id = ["0x%0.4x" % id for id in constants.REMOTE_IDS]  # remote device network ID
buffer = {}
names = {}
values = {}
activities_duration = {}
for tag_id in remote_id:
    buffer[tag_id] = {}
    for data_type in DATA_TYPES:
        buffer[tag_id][data_type] = {}
        buffer[tag_id][data_type]["timestamp"] = []
        buffer[tag_id][data_type]["x"] = []
        buffer[tag_id][data_type]["y"] = []
        buffer[tag_id][data_type]["z"] = []


# Create figure for plotting
fig = plt.figure()
gs = GridSpec(8, 8, figure=fig)
gs.update(wspace=0.3,hspace=0.3)
axzpos = fig.add_subplot(gs[:2, :2])
ax = fig.add_subplot(gs[3:, :8])
ax_accel = fig.add_subplot(gs[:2, 3:5])
ax_gyro = fig.add_subplot(gs[:2, 6:8])

# (axzpos, ax) = fig.subplots(2,1, height_ratios=[1, 3])
axzpos.set_ylim(-1000, 2500)
axzpos.set_title("Z Position")
axzpos.set_ylabel("z (mm)")

ax.set_title('Real Time Positioning')
ax.set_xlabel('X (mm)')
ax.set_ylabel('y (mm)')

ax_accel.set_title("ACC of Tag (mg)")
ax_accel.set_ylim(-2000, 2000)

ax_gyro.set_title("GYRO of Tag (dps)")
ax_gyro.set_ylim(-2000, 2000)

# Create a blank line. We will update the line in animate
lines = {}
for k in buffer:
    # print(buffer)
    # Plot POS_XY Overlay on Image
    lines[k] = {}
    lines[k]['POS'] = {}
    line,  = ax.plot(
        buffer[k]['POS']["x"],
        buffer[k]['POS']["y"],
        'o-',
        markerfacecolor='none',
        color=(*(np.array(COLORS[remote_id.index(k)])/255).tolist(),
        0.50),
        markeredgecolor=[*(np.array(COLORS[remote_id.index(k)])/255).tolist(),1],
        linewidth=1,
        markersize=3,
        markevery=[-1],
        label=k
    )
    lines[k]['POS']["xy"] = line
    lines[k]['POS']['direction'] = ax.arrow(0, 0, 1, 1, width=3, alpha=0.5, linewidth=5)

    # Plot POS_Z
    line, = axzpos.plot([], buffer[k]['POS']["z"], 'o-', color=(np.array(COLORS[remote_id.index(k)])/255).tolist(), markersize=1, linewidth=1, label=k)
    lines[k]['POS']["z"] = line

    # Plot ACC_XYZ
    lines[k]['ACC'] = {}
    for ind, axis in enumerate(["x", "y", "z"]):
        line, = ax_accel.plot([], buffer[k]['ACC'][axis], 'o-', color=(np.array(COLORS[ind])/255).tolist(), markersize=1, linewidth=1) #label=f"ACC_{['X','Y','Z']}" Legend wont display?
        lines[k]['ACC'][axis] = line

    # Plot GYRO_XYZ
    lines[k]['GYRO'] = {}
    for ind, axis in enumerate(["x", "y", "z"]):
        line, = ax_gyro.plot([], buffer[k]['GYRO'][axis], 'o-', color=(np.array(COLORS[ind])/255).tolist(), markersize=1, linewidth=1) #label=f"GYRO_{axis}" Legend wont display?
        lines[k]['GYRO'][axis] = line


ax.imshow(img, extent=[0,img.shape[1]*bg_multiplier,0,img.shape[0]*bg_multiplier], cmap='gray', vmin=0, vmax=255)


currentlocation = ''
currentactivity = ''
location_start_time = activity_start_time = time.time()

# This function is called periodically from FuncAnimation
def animate(i, buffer,):
    global currentlocation, currentactivity, location_start_time, activity_start_time
    # Clear the buffer
    
    for k in buffer:
        for data_type in DATA_TYPES:
            buffer[k][data_type]["timestamp"] = []
            buffer[k][data_type]["x"] = []
            buffer[k][data_type]["y"] = []
            buffer[k][data_type]["z"] = []

    with open(data_file, 'r') as f:
        all_data = f.readlines()
        data = all_data[-NUM_POINTS:]

        for one_data in data:
            splitted = one_data.split(",")
            tag_id = splitted[-1].replace("\n", "") # tagId
            timestamp = float(splitted[0]) 
            pos_x = float(splitted[1])
            pos_y = float(splitted[2])
            pos_z = float(splitted[3])
            heading = float(splitted[4])
            roll = float(splitted[5])
            pitch = float(splitted[6])
            acc_x = float(splitted[7])
            acc_y = float(splitted[8])
            acc_z = float(splitted[9])
            lin_acc_x = float(splitted[10])
            lin_acc_y = float(splitted[11])
            lin_acc_z = float(splitted[12])
            gyro_x = float(splitted[13])
            gyro_y = float(splitted[14])
            gyro_z = float(splitted[15])
            pressure = float(splitted[-2])

            #Filtering Extraneous Z positions
            if pos_z > 2000 : 
                pos_z = 2000
            elif pos_z < 0:
                pos_z = 0 

            buffer[tag_id]['ORIENT']["timestamp"] = timestamp
            buffer[tag_id]['ORIENT']["x"].append(heading)
            buffer[tag_id]['ORIENT']["y"].append(roll)
            buffer[tag_id]['ORIENT']["z"].append(pitch)

            buffer[tag_id]['POS']["timestamp"].append(timestamp)
            buffer[tag_id]['POS']["x"].append(pos_x)
            buffer[tag_id]['POS']["y"].append(pos_y)
            buffer[tag_id]['POS']["z"].append(pos_z)

            buffer[tag_id]['LINACC']["timestamp"].append(timestamp)
            buffer[tag_id]['LINACC']["x"].append(lin_acc_x)
            buffer[tag_id]['LINACC']["y"].append(lin_acc_y)
            buffer[tag_id]['LINACC']["z"].append(lin_acc_z)

            buffer[tag_id]['ACC']["timestamp"].append(timestamp)
            buffer[tag_id]['ACC']["x"].append(acc_x)
            buffer[tag_id]['ACC']["y"].append(acc_y)
            buffer[tag_id]['ACC']["z"].append(acc_z)

            buffer[tag_id]['GYRO']["timestamp"].append(timestamp)
            buffer[tag_id]['GYRO']["x"].append(gyro_x)
            buffer[tag_id]['GYRO']["y"].append(gyro_y)
            buffer[tag_id]['GYRO']["z"].append(gyro_z)

            buffer[tag_id]['PRESSURE']["timestamp"].append(timestamp)
            buffer[tag_id]['PRESSURE']["x"].append(pressure)

    ##################################################
    # ML PREPROCESSING
    
    for tag_id in buffer:
        data = []
        for item in ['POS', 'ORIENT', 'ACC', 'LINACC', 'GYRO']:
            for axis in ['x', 'y', 'z']:
                data.append(buffer[tag_id][item][axis])
        data.append(buffer[tag_id]['PRESSURE']['x'])
        data = np.array(data)

        # Cleaning
        df = pd.DataFrame(data.T, columns=['POS_X', 'POS_Y', 'POS_Z', 'Heading', 'Roll', 'Pitch', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'Pressure'])
        cleaned_df = (df
                .loc[:, ['POS_X', 'POS_Y', 'POS_Z', 'ACC_X', 'ACC_Y', 'ACC_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'Heading', 'Roll', 'Pitch', 'Pressure']]
                .pipe(utils.handle_spikes, ['POS_Z', 'POS_Y'], [1500.0, 800.0]) 
                .pipe(utils.MAV_cols, ['POS_X', 'POS_Y', 'POS_Z'], MAV_WINDOW)
                .pipe(utils.drop_columns_that_contain, 'Pressure')
                .dropna().reset_index(drop=True)
            )

        # Feature Extraction
        mean = cleaned_df.mean()
        mean.index = ['MEAN_' + ind for ind in mean.index]

        median = cleaned_df.median()
        median.index = ['MEDIAN_' + ind for ind in median.index]

        std = cleaned_df.std()
        std.index = ['STD_' + ind for ind in std.index]

        mode = cleaned_df.copy().pipe(utils.round_cols, ['POS_X', 'POS_Y', 'POS_Z'], 50).mode().iloc[0, :] # 50 mm = 5 cm, mode may output 2 rows
        mode.index = ['MODE_' + ind for ind in mode.index]

        max_value = cleaned_df.max()
        max_value.index = ['MAX_' + ind for ind in max_value.index]

        min_value = cleaned_df.min()
        min_value.index = ['MIN_' + ind for ind in min_value.index]

        xpeaks = find_peaks(abs(cleaned_df.iloc[:,3] - cleaned_df.iloc[:,3].mean()), height = 500)
        ypeaks = find_peaks(abs(cleaned_df.iloc[:,4] - cleaned_df.iloc[:,4].mean()), height = 500)
        zpeaks = find_peaks(abs(cleaned_df.iloc[:,5] - cleaned_df.iloc[:,5].mean()), height = 500)
        peaks = [len(xpeaks[1]['peak_heights']), len(ypeaks[1]['peak_heights']), len(zpeaks[1]['peak_heights'])]
        
        column_name = ['Peaks_Acc_X','Peaks_Acc_Y','Peaks_Acc_Z']

        mode_location = pd.Series(cleaned_df.copy().pipe(utils.determine_location, REGIONS).loc[:, 'Location'].mode()[0], index=["LOCATION"])

        accel_peak = pd.Series(peaks, index = column_name)

        feature_vector = pd.concat([mean, median, std, mode, max_value, min_value, mode_location, accel_peak]).to_frame().T

        # Feature selection
        feature_list = ['LOCATION', 'Peaks_Acc_X', 'Peaks_Acc_Y', 'Peaks_Acc_Z', ]
        feature_list += feature_vector.columns[feature_vector.columns.str.contains('POS')].tolist()
        feature_list += feature_vector.columns[feature_vector.columns.str.contains('_ACC')].tolist()
  
        feature_vector = feature_vector.loc[:, feature_list ]   
        
        feature_vector = (feature_vector.pipe(utils.one_hot_encode_col, 'LOCATION', LOCATION_ENCODER))

        y_pred_label = LABEL_ENCODER.classes_[CLF.predict(feature_vector.values)[0]]

        
        #Creating CSVs for Analytics 
        if mode_location.values[0] != currentlocation : 
            location_final_time  = timestamp
            location_duration = location_final_time - location_start_time
            location_start_time = location_final_time
            currentlocation = mode_location.values[0]
            df = pd.DataFrame([[currentlocation, location_duration]], columns=['Location', 'Duration'])
            filename_1 = 'RoomAnalytics.csv'
            with open(filename_1, 'a') as f:
                df.to_csv(f, mode='a', header = f.tell()==0, index = False)
            # df.to_csv('RoomAnalytics.csv', mode='a', index=False, header = None)
        
        if y_pred_label != currentactivity : 
            activity_final_time = timestamp 
            activity_duration = activity_final_time - activity_start_time
            activity_start_time = activity_final_time
            currentactivity = y_pred_label
            df = pd.DataFrame([[currentactivity, activity_duration]], columns=['Activity', 'Duration'])
            filename_2 = 'ActivityAnalytics.csv'
            with open(filename_2, 'a') as f:
                df.to_csv(f, mode='a', header = f.tell()==0, index = False)
            # df.to_csv('RoomAnalytics.csv', mode='a', index=False, header = None)

        df_activity = pd.DataFrame([[timestamp, currentactivity]], columns = ['Time', 'Activity'])
        filename_3 = 'RawActivityData.csv'
        with open(filename_3, 'a') as f:
             df_activity.to_csv(f, mode='a', header = f.tell()==0, index = False)
                
        print(f"Patient is in {currentlocation} doing '{currentactivity}' activity. Peaks : {accel_peak.values}")


    ##################################################

    # Update line with new Y values
    max_time = 0                # To dynamically update the xlim of the z-graph
    min_time = float("inf")     # To dynamically update the xlim of the z-graph
    for k in lines:

        x_pos = pd.Series(buffer[k]['POS']["x"]).rolling(20).mean().to_numpy()
        y_pos = pd.Series(buffer[k]['POS']["y"]).rolling(20).mean().to_numpy()
        # lines[k]['POS']["xy"].set_xdata(buffer[k]['POS']["x"])
        # lines[k]['POS']["xy"].set_ydata(buffer[k]['POS']["y"])
        lines[k]['POS']["xy"].set_xdata(x_pos)
        lines[k]['POS']["xy"].set_ydata(y_pos)
        lines[k]['POS']["z"].set_ydata(buffer[k]['POS']["z"])
        lines[k]['POS']["z"].set_xdata(buffer[k]['POS']["timestamp"])
        lines[k]['POS']['direction'].set_data(x=x_pos[-1], y=y_pos[-1], dx=500*np.sin(buffer[k]['ORIENT']['x'][-1] * np.pi / 180), dy=500*np.cos(buffer[k]['ORIENT']['x'][-1] * np.pi / 180))


        lines[k]['ACC']["x"].set_ydata(buffer[k]['ACC']["x"])
        lines[k]['ACC']["x"].set_xdata(buffer[k]['ACC']["timestamp"])
        lines[k]['ACC']["y"].set_ydata(buffer[k]['ACC']["y"])
        lines[k]['ACC']["y"].set_xdata(buffer[k]['ACC']["timestamp"])
        lines[k]['ACC']["z"].set_ydata(buffer[k]['ACC']["z"])
        lines[k]['ACC']["z"].set_xdata(buffer[k]['ACC']["timestamp"])

        lines[k]['GYRO']["x"].set_ydata(buffer[k]['GYRO']["x"])
        lines[k]['GYRO']["x"].set_xdata(buffer[k]['GYRO']["timestamp"])
        lines[k]['GYRO']["y"].set_ydata(buffer[k]['GYRO']["y"])
        lines[k]['GYRO']["y"].set_xdata(buffer[k]['GYRO']["timestamp"])
        lines[k]['GYRO']["z"].set_ydata(buffer[k]['GYRO']["z"])
        lines[k]['GYRO']["z"].set_xdata(buffer[k]['GYRO']["timestamp"])

        # Dynamically scale the z-pos graph with a min and max time 
        if buffer[k]['POS']["timestamp"][-1] > max_time: max_time = buffer[k]['POS']["timestamp"][-1] + 0.2
        if buffer[k]['POS']["timestamp"][0] < min_time: min_time = buffer[k]['POS']["timestamp"][0] - 0.2

    ax_accel.set_xlim(min_time, max_time)
    ax_gyro.set_xlim(min_time, max_time)
    axzpos.set_xlim(min_time, max_time)
    lines_array = []
    for tagid in lines:
        for data_type in lines[tag_id]:
            for plot_config in lines[tag_id][data_type]:
                lines_array.append(lines[tagid][data_type][plot_config])

    return lines_array


# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig,
    animate,
    fargs=(buffer, ),
    interval=1,
    blit=True)

plt.legend()
plt.show()