"""
This script unpacks the zipped datafile into a pickled dataframe.
"""
import pandas as pd
from os.path import isfile
from config import *

# Unpack compressed data file if csv file is not found
if not isfile(data_file_pkl):
    print(f"Extracing {data_file_zip}...")
    data = pd.read_csv(network_traffic_data)
    time_format = "%Y-%m-%d %H:%M:%S"
    if "time_from" in data.columns:
        # data["time_from"] = pd.to_datetime(data['time_from'], unit='s').dt.strftime("%Y-%m-%d %H:%M:00")
        data["time_from"] = pd.to_datetime(data['time_from'], unit='s').dt.floor('1S')
    print(f"Saving processed DataFrame to {data_file_pkl}...")
    print(data)
    data.to_pickle(data_file_pkl)

# Load the .pkl file
df = pd.read_pickle(network_graph_file)

# Save it as a CSV
df.to_csv("weighted_adj_mtx.csv", index=True)

