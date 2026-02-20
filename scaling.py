import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

# Sample dataset
data = {
    "customer_id":["C001","C002","C003","C004","C005"],
    "age":[25,45,35,55,28],
    "annual_income":[35000,95000,60000,120000,42000],
    "credit_score":[650,750,700,800,680],
    "years_employed":[2,15,8,25,3],
    "num_purchases":[5,50,25,100,8]
}
df = pd.DataFrame(data)

features = ["age","annual_income","credit_score","years_employed","num_purchases"]

# i. StandardScaler (Z-score normalization)
scaler_std = StandardScaler()
df_std = scaler_std.fit_transform(df[features])

# ii. MinMaxScaler
scaler_mm = MinMaxScaler()
df_mm = scaler_mm.fit_transform(df[features])

# iii. RobustScaler
scaler_rb = RobustScaler()
df_rb = scaler_rb.fit_transform(df[features])

# iv. MaxAbsScaler
scaler_ma = MaxAbsScaler()
df_ma = scaler_ma.fit_transform(df[features])

print("StandardScaler:\n", df_std)
print("MinMaxScaler:\n", df_mm)
print("RobustScaler:\n", df_rb)
print("MaxAbsScaler:\n", df_ma)


# Add synthetic outlier
df.loc[len(df)] = ["C006", 120, 500000, 900, 40, 500]  # extreme values

# Apply scalers again
df_std_out = scaler_std.fit_transform(df[features])
df_mm_out = scaler_mm.fit_transform(df[features])
df_rb_out = scaler_rb.fit_transform(df[features])
df_ma_out = scaler_ma.fit_transform(df[features])
df_std_out
df_mm_out
df_rb_out
df_ma_out