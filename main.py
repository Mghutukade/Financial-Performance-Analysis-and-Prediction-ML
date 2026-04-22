# 📊 data loading + cleaning + EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 🔹 Load data
data = pd.read_csv("data/Financials.csv")

print("🔹 Raw Data Preview:")
print(data.head())

print("\n🔹 Data Info:")
print(data.info())

print("\n🔹 Statistical Summary:")
print(data.describe())

# ---------------------------
# 🔹 Data Cleaning
# ---------------------------

# Fix column names
data.columns = data.columns.str.strip()

# Columns to clean
cols = ['Units Sold','Manufacturing Price','Sale Price','Gross Sales',
        'Discounts','Sales','COGS','Profit']

# Clean values
for col in cols:
    data[col] = (
        data[col]
        .replace('[\$,]', '', regex=True)
        .replace(' -', np.nan)
    )
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop missing values
data = data.dropna()

print("\n✅ Cleaned Data Types:")
print(data.dtypes)

# ---------------------------
# 📊 EDA (Visualizations)
# ---------------------------

# Profit vs Sales
plt.scatter(data['Sales'], data['Profit'])
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.title("Sales vs Profit")
plt.show()

# Trend
data[['Sales', 'Profit']].plot(title="Sales & Profit Trend")
plt.show()