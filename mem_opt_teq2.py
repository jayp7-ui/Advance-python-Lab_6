import numpy as np
import pandas as pd

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

# Suppose df is your Part A DataFrame
chunk_size = 10000
num_chunks = int(np.ceil(len(df) / chunk_size))

for i in range(num_chunks):
    chunk = df.iloc[i*chunk_size : (i+1)*chunk_size]
    print("Chunk shape:", chunk.shape)
    # process chunk here

df_small = df[["user_id","price"]]
print(df_small.head())
print("Memory usage selective:", df_small.memory_usage(deep=True).sum())

df_opt = df.copy()
df_opt["user_id"] = df_opt["user_id"].astype("int32")
df_opt["category"] = df_opt["category"].astype("category")

print(df_opt.dtypes)
print("Memory usage optimized:", df_opt.memory_usage(deep=True).sum())

print("Original memory:", df.memory_usage(deep=True).sum())
print("Selective columns memory:", df_small.memory_usage(deep=True).sum())
print("Optimized dtypes memory:", df_opt.memory_usage(deep=True).sum())

df["category_astype"] = df["category"].astype("category")
df["category_pdcat"] = pd.Categorical(df["category"])

print("astype dtype:", df["category_astype"].dtype)
print("pd.Categorical dtype:", df["category_pdcat"].dtype)

print("Memory before:", df["status"].memory_usage(deep=True))
df["status"] = df["status"].astype("category")
print("Memory after:", df["status"].memory_usage(deep=True))

sparse_col = pd.arrays.SparseArray([0,0,0,1,0,0,2,0])
print("Sparse column:", sparse_col)
print("Memory usage:", sparse_col.nbytes)

print("Alternatives: Dask (parallel, out-of-core), Vaex (lazy evaluation, fast), Polars (Rust-based, very fast), Spark (distributed big data).")