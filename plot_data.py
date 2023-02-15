import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

data_path = Path(__file__).resolve().parents[0].joinpath("data")

if len(sys.argv) < 2:
    data_file = data_path.joinpath(input("type the name of data file: ") + ".csv")
else: 
    data_file = data_path.joinpath(f"{sys.argv[1]}.csv")

bg_options = {
    "ILS": {"path": "ISL HQ Screenshot-rotated.png", "multiplier": 6.3, },
    "GR": {"path": "Glenrose Research First Floor Cropped.png", "multiplier": 14.75}
}

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
    buffer[tag_id]["x"] = []
    buffer[tag_id]["y"] = []
    buffer[tag_id]["z"] = []

# Create figure for plotting
fig = plt.figure()
(ax, axzpos) = fig.subplots(1,2, width_ratios=[3, 1])
axzpos.set_ylim(-100, 5000)
axzpos.set_xlim(-1, 4)

# Create a blank line. We will update the line in animate
lines = {}
for k in buffer:
    # Plot xy
    lines[k] = {}
    line, = ax.plot(buffer[k]["x"], buffer[k]["y"], label=k)
    lines[k]["xy"] = line

    # Plot z
    line, = axzpos.plot([remote_id.index(k)]*(len(buffer[k]["z"])), buffer[k]["z"], 'o-', label=k)
    lines[k]["z"] = line



ax.imshow(img, extent=[0,img.shape[1]*bg_multiplier,0,img.shape[0]*bg_multiplier], cmap='gray', vmin=0, vmax=255)
# Add labels
plt.title('Real Time Positioning')
plt.xlabel('X (mm)')
plt.ylabel('y (mm)')

# This function is called periodically from FuncAnimation
def animate(i, buffer):

    # Clear the buffer
    for k in buffer:
        buffer[k]["x"] = []
        buffer[k]["y"] = []
        buffer[k]["z"] = []

    with open(data_file, 'r') as f:
        all_data = f.readlines()
        data = all_data[-20:]
        for one_data in data:
            splitted = one_data.split(",")
            key = splitted[-1].replace("\n", "") 
            x = float(splitted[1])
            y = float(splitted[2])
            z = float(splitted[3])
            buffer[key]["x"].append(x)
            buffer[key]["y"].append(y)
            buffer[key]["z"].append(z)

    # Limit z to 3 points
    for key in buffer:
        buffer[key]["z"] = buffer[key]["z"][-3:]

    # Update line with new Y values
    for k in lines:
        lines[k]["xy"].set_xdata(buffer[k]["x"])
        lines[k]["xy"].set_ydata(buffer[k]["y"])
        lines[k]["z"].set_ydata(buffer[k]["z"])
        lines[k]["z"].set_xdata([remote_id.index(k)]*len(buffer[k]["z"]))

    return [lines[tagid][data_type] for tagid in lines for data_type in lines[tagid]]

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig,
    animate,
    fargs=(buffer,),
    interval=1,
    blit=True)

plt.legend()
plt.show()