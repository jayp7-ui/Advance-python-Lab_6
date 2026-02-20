import pandas as pd

# Sample dataset
data = {
    "date":["2024-01-01","2024-01-02","2024-01-03","2024-01-05","2024-01-08"],
    "stock_symbol":["TECH"]*5,
    "open_price":[150.5,152.0,None,149.8,151.5],
    "close_price":[152.3,None,150.5,151.2,153.0],
    "volume":[1000000,950000,None,1100000,1050000],
    "high":[153.0,152.8,151.0,151.5,154.2],
    "low":[149.8,151.2,149.5,149.0,151.0]
}
df = pd.DataFrame(data)

# i. Convert date to datetime and set index
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# ii. Handle missing dates vs missing values
# Reindex to include all business days
all_days = pd.date_range(start="2024-01-01", end="2024-01-08", freq="B")
df = df.reindex(all_days)

# iii. Imputation for stock prices
df["close_price"].ffill(inplace=True)   # forward fill for prices
df["open_price"].ffill(inplace=True)
df["volume"].fillna(df["volume"].median(), inplace=True)  # median for volume

print(df)

# a. Rolling statistics
df["close_7d_ma"] = df["close_price"].rolling(window=7).mean()
df["close_30d_ma"] = df["close_price"].rolling(window=30).mean()

# b. Lag features
df["close_lag1"] = df["close_price"].shift(1)   # previous day
df["close_lag5"] = df["close_price"].shift(5)   # previous week (5 business days)

# c. Percentage change and returns
df["pct_change"] = df["close_price"].pct_change()
df["returns"] = df["close_price"].pct_change().fillna(0)

# d. Exponentially weighted moving average (EWMA)
df["ewma_7"] = df["close_price"].ewm(span=7, adjust=False).mean()

print(df[["close_price","close_7d_ma","close_lag1","pct_change","ewma_7"]])

# a. Resample to weekly and monthly
weekly = df["close_price"].resample("W").mean()
monthly = df["close_price"].resample("MS").mean()

print("Weekly:\n", weekly)
print("Monthly:\n", monthly)

# c. Handle timezone-aware datetime
df.index = df.index.tz_localize("UTC")   # make index timezone-aware
df.index = df.index.tz_convert("Asia/Kolkata")  # convert to IST

#Apply One-Hot Encoding
#One-Hot Encoding converts categorical columns into "binary column"
import pandas as pd
data={
    'Color':['Red','Blue','Green','Red','Blue']
}
df = pd.DataFrame(data)
#Apply One-Hot Encoding
df_encoded=pd.get_dummies(df,columns=['Color'])
print(df_encoded)