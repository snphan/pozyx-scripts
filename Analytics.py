import pandas as pd 
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
import matplotlib.path as mpltPath

room_data = pd.read_csv('RoomAnalytics.csv')
activity_data = pd.read_csv('ActivityAnalytics.csv')
# adl_list = {'ADL': ['cooking', 'hygiene', 'cleaning', 'changing', 'toileting', 'leisure', 'eating', 'sleep'] }
unique_activities = pd.unique(activity_data.iloc[:,0]) 
unique_rooms = pd.unique(room_data.iloc[:,0])

#Assigns all activity/room labels to dictionary
activities_duration = {act: 0 for act in unique_activities} 
room_duration = {room: 0 for room in unique_rooms}

#Assigns all activity/room durations to their respective labels
for act in activities_duration:
    rows_act = activity_data[activity_data['Activity'].str.contains(act)]
    dur_act = rows_act['Duration'].sum()/60 # Converts time duration to minutes
    activities_duration[act] = dur_act

for room in room_duration:
    rows_room = room_data[room_data['Location'].str.contains(room)]
    dur_room = rows_room['Duration'].sum()/60 
    room_duration[room] = dur_room


#Obtains room/activity labels and durations
names_act = list(activities_duration.keys()) 
values_act = list(activities_duration.values()) 
names_room = list(room_duration.keys())
values_room = list(room_duration.values())

num_activities = range(len(activities_duration))

#Create multiplot figure for room, activity, walking analytics
fig = plt.figure()
gs = GridSpec(9, 9, figure=fig)
gs.update(wspace=0.3,hspace=0.3)
activity = fig.add_subplot(gs[:9,:4])
room = fig.add_subplot(gs[:9,5:9])

#Activity Plot
activity.set_title("User Activity")
activity.set_ylabel("Duration (Mins)")
activity.set_xlabel("Activities")
activity.bar(names_act,values_act)
activity.set_xticks(range(len(activities_duration)), names_act, rotation = 'vertical')
activity.tick_params(axis='x', which='major', labelsize = 8)
activity.tick_params(axis='y', which='major', labelsize = 10)

#Room Plot
room.set_title("Room Activity")
room.set_ylabel("Duration (Mins)")
room.set_xlabel("Rooms")
room.bar(names_room, values_room)
room.set_xticks(range(len(room_duration)), names_room, rotation = 'vertical')
room.tick_params(axis='x', which='major', labelsize = 8)
room.tick_params(axis='y', which='major', labelsize = 10)
print(names_room)

#Walking/Stationary Plot
# data_file = (input("type the name of data file: ") + ".csv")
# with open(data_file, 'r') as f:
#     all_data = f.readlines()
#     data = all_data[-NUM_POINTS:]


#####################################################

plt.show()

# def animate(frame):
#     global bar
#     bar[frame].set_height(values_act)

#     return

# ani = animation.FuncAnimation(fig, animate, frames = len(values_act))
# plt.show()

# openfreezer = activity_data[activity_data['Activity'].str.contains('FREEZER')]
# time_freezer = openfreezer['Duration'].sum()
