import pandas as pd

# --- Create sample data directly (no CSV needed) ---
sales_data = {
    "transaction_id": ["TX001","TX002","TX003","TX004","TX005"],
    "customer_id": ["CUST101","CUST102","CUST103","CUST101","CUST104"],
    "product_id": ["PROD201","PROD202","PROD201","PROD203","PROD204"],
    "store_id": ["ST01","ST02","ST01","ST03","ST02"],
    "quantity": [3,1,2,5,1],
    "sale_date": ["2024-01-15","2024-01-15","2024-01-16","2024-01-16","2024-01-17"]
}
sales = pd.DataFrame(sales_data)

customers_data = {
    "customer_id":["CUST101","CUST102","CUST105"],
    "name":["Alice Brown","Bob Smith","Carol White"],
    "email":["alice@email.com","bob@email.com","carol@email.com"],
    "city":["New York","London","Tokyo"],
    "country":["USA","UK","Japan"],
    "join_date":["2023-03-15","2023-05-20","2023-07-10"],
    "loyalty_tier":["Gold","Silver","Gold"]
}
customers = pd.DataFrame(customers_data)

products_data = {
    "product_id":["PROD201","PROD202","PROD203","PROD205"],
    "product_name":["Laptop","Coffee Maker","Office Chair","Headphones"],
    "category":["Electronics","Appliances","Furniture","Electronics"],
    "unit_price":[1200.00,89.99,250.00,150.00],
    "supplier_id":["SUP01","SUP02","SUP01","SUP03"],
    "stock_qty":[50,100,75,200]
}
products = pd.DataFrame(products_data)

# --- Merge operations ---
sales_customers = sales.merge(customers, on="customer_id", how="left")
full_data = sales_customers.merge(products, on="product_id", how="left")

# Calculate revenue
full_data["revenue"] = full_data["quantity"] * full_data["unit_price"]

# Print merged table
print("Merged Transactions:\n", full_data)

# Print revenue by loyalty tier
revenue_by_tier = full_data.groupby("loyalty_tier")["revenue"].sum()
print("\nRevenue per loyalty tier:\n", revenue_by_tier)

# Identify duplicate transaction IDs
duplicates = sales[sales.duplicated(subset="transaction_id", keep=False)]
print("Duplicate transactions:\n", duplicates)

# Check if duplicates are exact copies
exact_dupes = duplicates[duplicates.duplicated(keep=False)]
print("\nExact duplicate rows:\n", exact_dupes)

# Strategy: drop exact duplicates, keep unique ones
cleaned_sales = sales.drop_duplicates()
print("\nCleaned sales data:\n", cleaned_sales)