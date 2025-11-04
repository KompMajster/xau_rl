import pandas as pd
df = pd.read_csv("data/XAUUSD_M5.csv", parse_dates=["time"])
print("bars rows:", len(df), "from:", df["time"].min(), "to:", df["time"].max())
print(df.tail(3))