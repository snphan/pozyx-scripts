import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from collections import deque
from PIL import Image
import time
import sys
from config import constants
from pathlib import Path
import pandas as pd
import utils
import matplotlib.path as mpltPath
import joblib
import matplotlib.path as mpltPath
import json
from scipy.signal import find_peaks
########################## ML Cleaning/Classification
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

#Set Directory and Input File
data_path = Path(__file__).resolve().parents[0].joinpath("data")

if len(sys.argv) < 2:
    data_file = data_path.joinpath(input("type the name of data file: ") + ".csv")
else: 
    data_file = data_path.joinpath(f"{sys.argv[1]}.csv")


remote_id = ["0x%0.4x" % id for id in constants.REMOTE_IDS]  # remote device network ID
buffer = values = names = {}

#Create dictionary for each tag 
for tag_id in remote_id:
    buffer[tag_id] = {}
    for data_type in DATA_TYPES:
        buffer[tag_id][data_type] = {}
        buffer[tag_id][data_type]["timestamp"] = []
        buffer[tag_id][data_type]["x"] = []
        buffer[tag_id][data_type]["y"] = []
        buffer[tag_id][data_type]["z"] = []

global currentlocation, currentactivity, location_start_time, activity_start_time, activity_duration
currentactivity = currentlocation = ''
location_start_time = activity_start_time = time.time()


with open(data_file, 'r') as f:
    all_data = f.readlines(-1) #reads all lines of input csv 

    for i in range(0,len(all_data)-20): #Increments through each line of selected csv
        data = all_data[i:NUM_POINTS+i:]

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

        
        ################################################### ML PREPROCESSING

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
            
            new_df = cleaned_df[-SECONDS_SHOW*SAMPLE_RATE:] #Holds 3 seconds of data 
        
            # Feature Extraction
            mean = new_df.mean()
            mean.index = ['MEAN_' + ind for ind in mean.index]
            median = new_df.median()
            median.index = ['MEDIAN_' + ind for ind in median.index]

            std = new_df.std()
            std.index = ['STD_' + ind for ind in std.index]

            mode = new_df.copy().pipe(utils.round_cols, ['POS_X', 'POS_Y', 'POS_Z'], 50).mode().iloc[0, :] # 50 mm = 5 cm, mode may output 2 rows
            mode.index = ['MODE_' + ind for ind in mode.index]

            max_value = new_df.max()
            max_value.index = ['MAX_' + ind for ind in max_value.index]

            min_value = new_df.min()
            min_value.index = ['MIN_' + ind for ind in min_value.index]

            xpeaks = find_peaks(abs(new_df.iloc[:,3] - new_df.iloc[:,3].mean()), height = 500)
            ypeaks = find_peaks(abs(new_df.iloc[:,4] - new_df.iloc[:,4].mean()), height = 500)
            zpeaks = find_peaks(abs(new_df.iloc[:,5] - new_df.iloc[:,5].mean()), height = 500)
            peaks = [len(xpeaks[1]['peak_heights']), len(ypeaks[1]['peak_heights']), len(zpeaks[1]['peak_heights'])]
            
            column_name = ['Peaks_Acc_X','Peaks_Acc_Y','Peaks_Acc_Z']

            mode_location = pd.Series(new_df.copy().pipe(utils.determine_location, REGIONS).loc[:, 'Location'].mode()[0], index=["LOCATION"])

            accel_peak = pd.Series(peaks, index = column_name)

            feature_vector = pd.concat([mean, median, std, mode, max_value, min_value, mode_location, accel_peak]).to_frame().T

            # Feature selection
            feature_list = ['LOCATION', 'Peaks_Acc_X', 'Peaks_Acc_Y', 'Peaks_Acc_Z', ]
            feature_list += feature_vector.columns[feature_vector.columns.str.contains('POS')].tolist()
            feature_list += feature_vector.columns[feature_vector.columns.str.contains('_ACC')].tolist()

            feature_vector = feature_vector.loc[:, feature_list ]   
           
            feature_vector = (feature_vector.pipe(utils.one_hot_encode_col, 'LOCATION', LOCATION_ENCODER))

            y_pred_label = LABEL_ENCODER.classes_[CLF.predict(feature_vector.values)[0]]

            #Attaches activity label with corresponding timestamp from CSV selected
            if y_pred_label != currentactivity: 
                old_activity = currentactivity
                activity_final_time = timestamp 
                activity_duration = activity_final_time - activity_start_time
                activity_start_time = activity_final_time
                currentactivity = y_pred_label
                df = pd.DataFrame([[old_activity, activity_duration]], columns=['Activity', 'Duration'])
                filename_2 = 'ActivityAnalytics.csv'
                with open(filename_2, 'a') as f:
                    df.to_csv(f, mode='a', header = f.tell()==0, index = False)

            if i == 0:
                initial_time = timestamp
                current_time = timestamp
            currentactivity = y_pred_label
            df_activity = pd.DataFrame([[currentactivity, timestamp, timestamp - initial_time]], columns = ['Activity', 'Time (UTC)', 'Time'])
            filename_3 = 'RawActivity.csv' 
            with open(filename_3, 'a') as f:
                df_activity.to_csv(f, mode='a', header = f.tell()==0, index = False)


        #Clears data in buffer before looping
        for k in buffer:
            for data_type in DATA_TYPES:
                buffer[k][data_type]["timestamp"] = []
                buffer[k][data_type]["x"] = []
                buffer[k][data_type]["y"] = []
                buffer[k][data_type]["z"] = []


######################################### Classifier Analytics

classifier_file = pd.read_csv('ActivityAnalytics.csv')

# activity_sessions = {activity: activity - start_time for activity in unique_post_activities}
# Group the data by 'Activity' and sum the 'Time' for each activity
activity_totals = classifier_file.groupby('Activity')['Duration'].sum()

# Print the total time for each activity
for activity, total_time in activity_totals.items():
    total_time = total_time/60
    print(f"Activity: {activity}, Total Time: {total_time}")

quit()
for activity in activity_sessions: 
    rows_activity = classifier_file[classifier_file['Activity'].str.contains(activity)]
    print(rows_activity)
    quit()
    duration_activity = rows_activity['Time'].sum()
    activity_sessions[activity] = duration_activity
    print(activity_sessions)

## Need to gety all unique values and then sum them and plot with percentages
## SHould have a inputs to type in the real durations fo each activites and then spit out accuracies of model 
## Need a dataframe or list to cycle through different feature combos ** Difficult

######################################### Room and Activity Analytics
# Using gridspec and seaborn plot all analytic graphs on same plot 
# Heatmap over floorplan 


# room_data = pd.read_csv('RoomAnalytics.csv')
# activity_data = pd.read_csv('ActivityAnalytics.csv')
# # adl_list = {'ADL': ['cooking', 'hygiene', 'cleaning', 'changing', 'toileting', 'leisure', 'eating', 'sleep'] }
# unique_activities = pd.unique(activity_data.iloc[:,0]) 
# unique_rooms = pd.unique(room_data.iloc[:,0])

# #Assigns all activity/room labels to dictionary
# activities_duration = {act: 0 for act in unique_activities} 
# room_duration = {room: 0 for room in unique_rooms}

# #Assigns all activity/room durations to their respective labels
# for act in activities_duration:
#     rows_act = activity_data[activity_data['Activity'].str.contains(act)]
#     dur_act = rows_act['Duration'].sum()/60 # Converts time duration to minutes
#     activities_duration[act] = dur_act

# for room in room_duration:
#     rows_room = room_data[room_data['Location'].str.contains(room)]
#     dur_room = rows_room['Duration'].sum()/60 
#     room_duration[room] = dur_room


# #Obtains room/activity labels and durations
# names_act = list(activities_duration.keys()) 
# values_act = list(activities_duration.values()) 
# names_room = list(room_duration.keys())
# values_room = list(room_duration.values())

# num_activities = range(len(activities_duration))

# #Create multiplot figure for room, activity, walking analytics
# fig = plt.figure()
# gs = GridSpec(9, 9, figure=fig)
# gs.update(wspace=0.3,hspace=0.3)
# activity = fig.add_subplot(gs[:9,:4])
# room = fig.add_subplot(gs[:9,5:9])

# #Activity Plot
# activity.set_title("User Activity")
# activity.set_ylabel("Duration (Mins)")
# activity.set_xlabel("Activities")
# activity.bar(names_act,values_act)
# activity.set_xticks(range(len(activities_duration)), names_act, rotation = 'vertical')
# activity.tick_params(axis='x', which='major', labelsize = 8)
# activity.tick_params(axis='y', which='major', labelsize = 10)

# #Room Plot
# room.set_title("Room Activity")
# room.set_ylabel("Duration (Mins)")
# room.set_xlabel("Rooms")
# room.bar(names_room, values_room)
# room.set_xticks(range(len(room_duration)), names_room, rotation = 'vertical')
# room.tick_params(axis='x', which='major', labelsize = 8)
# room.tick_params(axis='y', which='major', labelsize = 10)
# print(names_room)

#####################################################
