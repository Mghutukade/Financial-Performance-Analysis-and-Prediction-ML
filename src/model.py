# 🤖 model training file

import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ---------------- LOAD DATA ----------------
data = pd.read_csv("data/Financials.csv")

# ---------------- CLEAN DATA ----------------
data.columns = data.columns.str.strip()

cols = ['Units Sold','Manufacturing Price','Sale Price','Gross Sales',
        'Discounts','Sales','COGS','Profit']

for col in cols:
    data[col] = (
        data[col]
        .replace('[\$,]', '', regex=True)
        .replace(' -', np.nan)
    )
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna()

# ---------------- FEATURES ----------------
X = data[['Units Sold', 'COGS', 'Sales']]
y = data['Profit']

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODELS ----------------
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(n_estimators=100, random_state=42)

# ---------------- TRAIN ----------------
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
lr_score = lr.score(X_test, y_test)
rf_score = rf.score(X_test, y_test)
xgb_score = xgb.score(X_test, y_test)

print("\n📊 Model Performance:")
print("Linear Regression:", lr_score)
print("Random Forest:", rf_score)
print("XGBoost:", xgb_score)

# ---------------- SAVE MODEL ----------------
joblib.dump(rf, "model.pkl")
print("\n✅ Model saved as model.pkl")

# ---------------- SAVE SCORES ----------------
scores = {
    "Linear": lr_score,
    "Random Forest": rf_score,
    "XGBoost": xgb_score
}

with open("scores.json", "w") as f:
    json.dump(scores, f)

print("✅ Scores saved to scores.json")

# ---------------- FEATURE IMPORTANCE ----------------
print("\n🔥 Feature Importance (Random Forest):")
for name, importance in zip(X.columns, rf.feature_importances_):
    print(f"{name}: {importance:.4f}")