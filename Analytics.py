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
import json
import utils

room_data = pd.read_csv('RoomAnalytics.csv')
activity_data = pd.read_csv('ActivityAnalytics.csv')
undefined = activity_data[activity_data['Activity'].str.contains('UND')]

time_undefined = undefined['Duration'].sum()
print(undefined)
print(time_undefined)
# activity_data.plot(kind = 'hist', x = 'Activity', y = 'Duration')
# plt.show()