import pandas as pd
from os.path import isfile
from tqdm import tqdm
from config import *

if __name__ == "__main__":
    assert isfile(data_file_pkl), "Error: Pickled data not found. Please run unpack_data.py first."

    traffic_data = pd.read_pickle(data_file_pkl)
    # station_df = pd.read_csv(stations_data_file)
    unique_stations = traffic_data["SourceIP"].unique()
    first_timestamp = traffic_data["time_from"].min()
    last_timestamp = traffic_data["time_from"].max()

    print(f"First timestamp: {first_timestamp}")
    print(f"Last timestamp: {last_timestamp}")
    print("Building time series dataframe... Please grab a coffee!")

    time_series_data = pd.DataFrame(index=pd.date_range(first_timestamp, last_timestamp, freq="1S"), columns=unique_stations)
    print("-----------> time_series before")
    print(time_series_data)
    for station_id in tqdm(unique_stations):
        df = traffic_data.loc[traffic_data["SourceIP"]==station_id, ["TotalFwdPackets", "time_from"]]
        df = df.groupby("time_from").agg({"TotalFwdPackets": "sum"})
        df = df.reindex(time_series_data.index, method=None)
        # df.fillna(0, inplace=True) 
        time_series_data[station_id] = df["TotalFwdPackets"]
        
        # print(time_series_data[station_id])
        # print(df["TotalFwdPackets"])
        # print(df.index[:5].tolist())
        # print(time_series_data.index[:5].tolist())

    # Drop stations with too many NaNs / too few observations
    time_series_data.interpolate(method='linear', inplace=True)
    time_series_data.fillna(0, inplace=True)
    print(f"Dropping stations with too few observations (<{min_number_of_observations})...")
    time_series_data.dropna(thresh=min_number_of_observations, axis=1, inplace=True)
    

    # All stations are missing values at 22:00 every day. 
    # Replace these all-NaN rows by the mean of the row before and the row after.
    print("Filling rows with all NaN...")
    time_series_data.loc[time_series_data.isnull().all(axis=1), :] = (time_series_data.ffill(limit=1) + time_series_data.bfill(limit=1)) / 2
# Linear interpolation to fill missing values in time-series data
    

    # Split the dataset into training, validation and testing data
    n_total = len(time_series_data)
    print("-------------------------------------------->time_series_data")
    print(time_series_data)
    val_size = int(val_fraction * n_total)
    test_size = int(test_fraction * n_total)
    train_size = n_total - val_size - test_size
    train_df = time_series_data.iloc[0 : train_size]
    val_df = time_series_data.iloc[train_size : train_size + val_size]
    test_df = time_series_data.iloc[train_size + val_size : n_total]

    if normalize_data: 
        print("Normalizing data...") 
        if normalize_data == "minmax":
            # Scale to [0,1]
            min_val, max_val = train_df.min(), train_df.max() 
            mean, std = min_val, max_val - min_val
        elif normalize_data == "normal":
            # Compute z-scores
            mean, std = train_df.mean(), train_df.std()
        else:
            print("Invalid normalization method: {normalize_data}.")
            mean, std = 0, 1

        train_df = (train_df - mean) / std
        val_df = (val_df - mean) / std
        test_df = (test_df - mean) / std
    print("------train df")
    print(train_df)
    train_df.to_pickle(train_data_file)
    train_df.to_csv(network_train_csv)
    val_df.to_pickle(val_data_file)
    test_df.to_pickle(test_data_file)
    test_df.to_csv(network_test_csv)

    print(f"Time series contain {n_total} hours of data from {len(time_series_data.columns)} stations. (Missing observations have value NaN)")
    print(f"Split: {len(train_df)} (train), {len(val_df)} (val) and {len(test_df)} (test) samples")
    print(f"Time series data saved to \"{train_data_file}\", \"{val_data_file}\" and \"{test_data_file}\"")

    stations_included = train_df.columns
    print("-----> si")
    print(stations_included.size)
    pd.Series(stations_included).to_csv(stations_included_file)
    print(f"IDs of stations included in pre-processed data saved to {stations_included_file}.")
