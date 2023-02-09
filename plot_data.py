import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
from PIL import Image
import time
import sys

if len(sys.argv) < 2:
    data_file = input("type the name of data file: ") + ".csv"
else: 
    data_file = f"{sys.argv[1]}.csv"

bg_path = "ISL HQ Screenshot-rotated.png"
bg_multiplier=6.3
mydata = deque([1,2,3,4,5])
img = Image.open(bg_path).convert("L")
img = np.asarray(img)

remote_id = [0x684A, 0x683F]  # remote device network ID
buffer = {}

for id in remote_id:
    tag_id =  "0x%0.4x" % id
    buffer[tag_id] = {}
    buffer[tag_id]["x"] = []
    buffer[tag_id]["y"] = []



# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Create a blank line. We will update the line in animate
lines = {}
for k in buffer:
    line, = ax.plot(buffer[k]["x"], buffer[k]["y"], label=k)
    lines[k] = line

ax.imshow(img, extent=[0,1803*bg_multiplier,0,1683*bg_multiplier], cmap='gray', vmin=0, vmax=255)

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

    with open(data_file, 'r') as f:
        all_data = f.readlines()
        data = all_data[-20:]
        for one_data in data:
            splitted = one_data.split(",")
            k = splitted[-1].replace("\n", "") 
            x = float(splitted[1])
            y = float(splitted[2])
            buffer[k]["x"].append(x)
            buffer[k]["y"].append(y)

    # Update line with new Y values
    for k in lines:
        lines[k].set_xdata(buffer[k]["x"])
        lines[k].set_ydata(buffer[k]["y"])

    return lines.values()

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig,
    animate,
    fargs=(buffer,),
    interval=1,
    blit=True)

plt.legend()
plt.show()