import pandas as pd

# Example dataset
data = {
    "customer_id":["C001","C002","C003"],
    "signup_date":["2023-01-01","2022-06-15","2021-12-01"],
    "last_login":["2024-01-10","2024-01-05","2024-01-08"],
    "age":[25,45,60],
    "state":["NY","CA","TX"],
    "plan_type":["Basic","Premium","Standard"],
    "monthly_charges":[50,80,60],
    "total_charges":[600,2000,1500],
    "num_support_calls":[2,10,5],
    "contract_length":[12,24,12]
}
df = pd.DataFrame(data)

df["signup_date"] = pd.to_datetime(df["signup_date"])
df["last_login"] = pd.to_datetime(df["last_login"])

# i. Temporal features
df["account_age_days"] = (pd.to_datetime("2024-01-15") - df["signup_date"]).dt.days
df["days_since_login"] = (pd.to_datetime("2024-01-15") - df["last_login"]).dt.days
df["signup_month"] = df["signup_date"].dt.month
df["signup_quarter"] = df["signup_date"].dt.quarter
df["signup_dayofweek"] = df["signup_date"].dt.dayofweek

# ii. Ratio features
df["avg_monthly_spend"] = df["total_charges"] / df["contract_length"]
df["support_calls_per_month"] = df["num_support_calls"] / df["contract_length"]

# iii. Binning
df["age_group"] = pd.cut(df["age"], bins=[0,30,50,100],
                         labels=["Young","Middle-aged","Senior"])
df["charge_category"] = pd.cut(df["monthly_charges"], bins=[0,60,80,200],
                               labels=["Low","Medium","High"])

# iv. Interaction
df["plan_contract"] = df["plan_type"] + "_" + df["contract_length"].astype(str)
print(df)

import numpy as np
from sklearn.preprocessing import PowerTransformer

# a. Log transform skewed features
df["log_total_charges"] = np.log1p(df["total_charges"])

# b. Box-Cox (only positive values)
pt_boxcox = PowerTransformer(method="box-cox")
df["boxcox_income"] = pt_boxcox.fit_transform(df[["monthly_charges"]])

# c. Yeo-Johnson (works with negatives too)
pt_yj = PowerTransformer(method="yeo-johnson")
df["yj_support_calls"] = pt_yj.fit_transform(df[["num_support_calls"]])
print(df)

import matplotlib.pyplot as plt

# Before transformation
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
df["total_charges"].hist(bins=20)
plt.title("Original Distribution: total_charges")

# After log transformation
plt.subplot(1,2,2)
df["log_total_charges"].hist(bins=20)
plt.title("Log-Transformed Distribution: total_charges")

plt.show()