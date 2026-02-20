import pandas as pd
import numpy as np

# Simulate a smaller version of your dataset for demonstration
df = pd.DataFrame({
    "transaction_id": np.arange(1,11_0001),   # int64
    "user_id": np.random.randint(1,1000, size=11_0000),   # int64
    "product_id": np.random.randint(1,5000, size=11_0000), # int64
    "timestamp": pd.date_range("2024-01-01", periods=11_0000, freq="min").astype(str), # object
    "category": np.random.choice(["Electronics","Clothing","Books"], size=11_0000), # object
    "subcategory": np.random.choice(["Mobile","Laptop","Shirt","Novel"], size=11_0000), # object
    "price": np.random.uniform(10,500, size=11_0000), # float64
    "quantity": np.random.randint(1,10, size=11_0000), # int64
    "discount": np.random.uniform(0,0.5, size=11_0000), # float64
    "status": np.random.choice(["Completed","Pending","Cancelled"], size=11_0000), # object
    "payment_method": np.random.choice(["Card","Cash","UPI"], size=11_0000), # object
    "rating": np.random.uniform(1,5, size=11_0000) # float64
})

# i. Current memory usage
df.info(memory_usage="deep")
print("Memory usage before optimization (bytes):", df.memory_usage(deep=True).sum())

# Downcast numerics
df["transaction_id"] = pd.to_numeric(df["transaction_id"], downcast="integer")
df["user_id"] = pd.to_numeric(df["user_id"], downcast="integer")
df["product_id"] = pd.to_numeric(df["product_id"], downcast="integer")
df["quantity"] = pd.to_numeric(df["quantity"], downcast="integer")
df["price"] = pd.to_numeric(df["price"], downcast="float")
df["discount"] = pd.to_numeric(df["discount"], downcast="float")
df["rating"] = pd.to_numeric(df["rating"], downcast="float")

# Convert object to category
for col in ["category","subcategory","status","payment_method"]:
    df[col] = df[col].astype("category")

# Optimize datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(df)

print("Memory usage after optimization (bytes):", df.memory_usage(deep=True).sum())

before = 23_500_000
after = df.memory_usage(deep=True).sum()
reduction = (before - after) / before * 100
print(f"Percentage memory reduction: {reduction:.2f}%")