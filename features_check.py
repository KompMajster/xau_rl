import pandas as pd

df = pd.read_csv("data/XAUUSD_M5_features.csv", parse_dates=["time"])
print("features rows:", len(df), "from:", df["time"].min(), "to:", df["time"].max())
print(df.tail(3))