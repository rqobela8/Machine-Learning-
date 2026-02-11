import pandas as pd
import numpy as np
from Data.data import download_dataset



def print_data_info(data):
    print(data.head(1))
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns}")
    print(f"Data info: {data.describe()}")

def transform_data(data):
    data["Remote_Work"] = data["Remote_Work"].map({"Yes":1,"No":0})
    data = data.drop(columns=["Code_Complexity", "Experience_Years"])

    return data

def read_data():
    data_path = download_dataset("mabubakrsiddiq/developer-stress-simulation-dataset")
    df = pd.read_csv(data_path)
    return df

def np_data():
    df = read_data()
    data = transform_data(df)
    X = data.drop(columns=["Stress_Level"]).values
    Y = data["Stress_Level"].values

    return X,Y




