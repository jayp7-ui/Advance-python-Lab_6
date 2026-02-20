import pandas as pd

# Sample dataset
data = {
    "date": ["2024-01-01","2024-01-01","2024-01-02","2024-01-02"],
    "region": ["North","South","North","East"],
    "category": ["Electronics","Furniture","Clothing","Clothing"],
    "sales": [15000,5000,18000,7000],
    "units": [25,10,30,95]
}
df = pd.DataFrame(data)

# i. Create a multi-index DataFrame
df_multi = df.set_index(["date","region","category"])
print("Multi-index DataFrame:\n", df_multi)

# ii. Groupby operations on different levels
# Example: sum sales grouped by region (level=1)
region_sales = df_multi.groupby(level="region")["sales"].sum()
print("\nSales by region:\n", region_sales)

# iii. Use xs() to slice data at specific index levels
north_data = df_multi.xs("North", level="region")
print("\nCross-section for region=North:\n", north_data)

# iv. Reset and manipulate multi-index
reset_df = df_multi.reset_index()
print("\nReset multi-index:\n", reset_df)

# Groupby with multiple aggregations
agg_df = df.groupby(["region","category"]).agg({
    "sales": ["sum","mean","std"],
    "units": ["sum","count"]
})
print("\nAggregated Data:\n", agg_df)

# a. Custom aggregation functions
def range_func(x):
    return x.max() - x.min()

custom_agg = df.groupby("region").agg({
    "sales": ["sum", range_func],
    "units": ["mean"]
})
print("\nCustom aggregation:\n", custom_agg)

# b. Different functions per column already shown above

# c. transform() vs apply()
# transform: returns a Series aligned with original DataFrame (same shape)
df["sales_mean_by_region"] = df.groupby("region")["sales"].transform("mean")

# apply: returns aggregated values (reduced shape)
region_apply = df.groupby("region")["sales"].apply(lambda x: x.mean())
print("\nTransform example:\n", df[["region","sales","sales_mean_by_region"]])
print("\nApply example:\n", region_apply)

# d. Filter groups based on aggregate conditions
# Example: keep regions where total sales > 20000
filtered = df.groupby("region").filter(lambda g: g["sales"].sum() > 20000)
print("\nFiltered groups (sales > 20000):\n", filtered)