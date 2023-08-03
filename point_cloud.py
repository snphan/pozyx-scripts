import numpy as np
import pandas as pd
from pathlib import Path

#############################################################################################################
# Using this script to generate point clouds in desmos for UWB paper to visualize furniture accuracy results
#############################################################################################################

data_path = Path(__file__).resolve().parents[0].joinpath("data")
data_file = data_path.joinpath(input("type the name of data file: ") + ".csv")
data_csv = pd.read_csv(data_file)

start_index = int(input('Start row of point cloud:'))
end_index = int(input('End row of point cloud:'))

position_data = data_csv.iloc[start_index:end_index, 1:3]
position_data.columns = ['x', 'y']

for index, row in position_data.iterrows():
    x_val = row['x']/1000
    y_val = row['y']/1000
    print(f"({x_val},{y_val}),", end='') #Prints all coordinates in same line so it all can be copy and pasted into desmos in one variable 

