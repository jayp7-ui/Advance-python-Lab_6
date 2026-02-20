import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier

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

# a. Polynomial features up to degree 2
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[["monthly_charges","num_support_calls"]])

# b. Trade-off
# More features = richer representation, but risk of overfitting and higher computation.

# c. Feature selection
# Variance threshold
selector = VarianceThreshold(threshold=0.01)
reduced = selector.fit_transform(poly_features)

# Correlation-based selection
corr = df.corr(numeric_only=True)
high_corr = corr[(corr.abs() > 0.8) & (corr.abs() < 1)]

# Feature importance from tree models
rf = RandomForestClassifier()
X = df[["monthly_charges","total_charges","num_support_calls","contract_length"]]
y = [0,1,0]  # example churn labels
rf.fit(X,y)
importances = rf.feature_importances_
print(poly_features)
corr