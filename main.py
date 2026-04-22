import pandas as pd 

# Load&understand the data 
data = pd.read_csv("data/Financials.csv")

print(data.head())
print(data.info())
print(data.describe())

# Data cleaning -------------
  
# Remove $ and commas, converts to folat 


data.columns = data.columns.str.strip()
print(data.columns)
     
