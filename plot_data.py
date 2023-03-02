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

"""
Plot the data in realtime
"""
####################################################################################################
#                                       CONFIGURATION
####################################################################################################
NUM_POINTS = 40 * len(constants.REMOTE_IDS)
# DATA_TYPES = ['POS', 'ACC', 'GYRO', 'LINACC', 'ORIENT'] # For orientation x = heading, y = roll, z = pitch
DATA_TYPES = ['POS', 'ACC'] # For orientation x = heading, y = roll, z = pitch
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
gs = GridSpec(4, 4, figure=fig)
gs.update(wspace=0.3,hspace=0.3)
axzpos = fig.add_subplot(gs[0, :])
ax = fig.add_subplot(gs[1:, :2])
ax_accel = fig.add_subplot(gs[1, 2:])
# (axzpos, ax) = fig.subplots(2,1, height_ratios=[1, 3])
axzpos.set_ylim(-100, 5000)
axzpos.set_title("Z Position")
axzpos.set_ylabel("z (mm)")

ax.set_title('Real Time Positioning')
ax.set_xlabel('X (mm)')
ax.set_ylabel('y (mm)')

ax_accel.set_title("ACC of Tag")
ax_accel.set_ylim(-2000, 2000)

# Create a blank line. We will update the line in animate
lines = {}
for k in buffer:
    print(buffer)
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
    # Plot POS_Z
    line, = axzpos.plot([], buffer[k]['POS']["z"], 'o-', color=(np.array(COLORS[remote_id.index(k)])/255).tolist(), markersize=1, linewidth=1, label=k)
    lines[k]['POS']["z"] = line

    # Plot ACC_XYZ
    lines[k]['ACC'] = {}
    for ind, axis in enumerate(["x", "y", "z"]):
        line, = ax_accel.plot([], buffer[k]['ACC'][axis], 'o-', color=(np.array(COLORS[ind])/255).tolist(), markersize=1, linewidth=1, label=f"ACC_{axis}")
        lines[k]['ACC'][axis] = line


print(lines)


ax.imshow(img, extent=[0,img.shape[1]*bg_multiplier,0,img.shape[0]*bg_multiplier], cmap='gray', vmin=0, vmax=255)

# This function is called periodically from FuncAnimation
def animate(i, buffer):

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
            key = splitted[-1].replace("\n", "")
            timestamp = float(splitted[0]) 
            pos_x = float(splitted[1])
            pos_y = float(splitted[2])
            pos_z = float(splitted[3])
            acc_x = float(splitted[7])
            acc_y = float(splitted[8])
            acc_z = float(splitted[9])

            buffer[key]['POS']["timestamp"].append(timestamp)
            buffer[key]['POS']["x"].append(pos_x)
            buffer[key]['POS']["y"].append(pos_y)
            buffer[key]['POS']["z"].append(pos_z)

            buffer[key]['ACC']["timestamp"].append(timestamp)
            buffer[key]['ACC']["x"].append(acc_x)
            buffer[key]['ACC']["y"].append(acc_y)
            buffer[key]['ACC']["z"].append(acc_z)

    # Update line with new Y values
    max_time = 0                # To dynamically update the xlim of the z-graph
    min_time = float("inf")     # To dynamically update the xlim of the z-graph
    for k in lines:
        lines[k]['POS']["xy"].set_xdata(buffer[k]['POS']["x"])
        lines[k]['POS']["xy"].set_ydata(buffer[k]['POS']["y"])
        lines[k]['POS']["z"].set_ydata(buffer[k]['POS']["z"])
        lines[k]['POS']["z"].set_xdata(buffer[k]['POS']["timestamp"])
        lines[k]['ACC']["x"].set_ydata(buffer[k]['ACC']["x"])
        lines[k]['ACC']["x"].set_xdata(buffer[k]['ACC']["timestamp"])
        lines[k]['ACC']["y"].set_ydata(buffer[k]['ACC']["y"])
        lines[k]['ACC']["y"].set_xdata(buffer[k]['ACC']["timestamp"])
        lines[k]['ACC']["z"].set_ydata(buffer[k]['ACC']["z"])
        lines[k]['ACC']["z"].set_xdata(buffer[k]['ACC']["timestamp"])

        # Dynamically scale the z-pos graph with a min and max time 
        if buffer[k]['POS']["timestamp"][-1] > max_time: max_time = buffer[k]['POS']["timestamp"][-1] + 0.2
        if buffer[k]['POS']["timestamp"][0] < min_time: min_time = buffer[k]['POS']["timestamp"][0] - 0.2

    ax_accel.set_xlim(min_time, max_time)
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
    fargs=(buffer,),
    interval=1,
    blit=True)

plt.legend()
plt.show()