import pandas as pd 

# Load&understand the data 
data = pd.read_csv("data/Financials.csv")

print(data.head())
print(data.info())
print(data.describe())


