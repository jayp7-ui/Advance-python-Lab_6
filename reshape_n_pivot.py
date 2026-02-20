import pandas as pd

# Sample dataset
data = {
    "date": ["2024-01-01","2024-01-01","2024-01-01","2024-01-01",
             "2024-01-02","2024-01-02","2024-01-02","2024-01-02"],
    "store_id": ["S01","S01","S02","S02","S01","S01","S02","S03"],
    "region": ["North","North","South","South","North","North","South","East"],
    "product_category": ["Electronics","Clothing","Electronics","Furniture",
                         "Electronics","Clothing","Electronics","Clothing"],
    "sales_amount": [15000,8000,12000,5000,18000,9500,13500,7000],
    "units_sold": [25,120,20,10,30,135,22,95]
}
df = pd.DataFrame(data)

# i. Pivot table: total sales by region and category
pivot1 = pd.pivot_table(df, values="sales_amount",
                        index="region", columns="product_category",
                        aggfunc="sum", fill_value=0)

# ii. Multi-level pivot: date + region as rows, sales and units as values
pivot2 = pd.pivot_table(df, values=["sales_amount","units_sold"],
                        index=["date","region"], aggfunc="sum")

print("Pivot 1:\n", pivot1)
print("\nPivot 2:\n", pivot2)

# a. Melt sales_amount and units_sold into long format
melted = pd.melt(df,
                 id_vars=["date","store_id","region","product_category"],
                 value_vars=["sales_amount","units_sold"],
                 var_name="metric", value_name="value")

print("\nMelted Data:\n", melted.head())

# a. Stack and unstack
stacked = df.set_index(["date","region","product_category"]).stack()
print("\nStacked:\n", stacked.head())

unstacked = stacked.unstack()
print("\nUnstacked:\n", unstacked.head())

# b. Crosstab: count of transactions by region and category
crosstab = pd.crosstab(df["region"], df["product_category"])
print("\nCrosstab:\n", crosstab)