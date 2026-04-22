# 📊 Financial Performance Analysis & Prediction

> 🚀 A Machine Learning powered dashboard for analyzing financial data and predicting profit in real-time.

---

## 🖼️ Dashboard Preview

<p align="center">
  <img src="images/dashboard.png" width="800"/>
</p>

---

## ✨ Key Highlights

✔ End-to-End Machine Learning Pipeline  
✔ Interactive Streamlit Dashboard  
✔ Real-time Profit Prediction  
✔ Business Insights & KPIs  
✔ Model Comparison & Feature Importance  

---

## 📊 Features

🔮 **Prediction System**
- Predict profit based on input values  
- Real-time ML model inference  

📈 **Data Insights**
- Sales vs Profit analysis  
- Top performing products  
- Country-wise profit breakdown  

🤖 **Model Analysis**
- Linear Regression, Random Forest, XGBoost  
- Model performance comparison  
- Feature importance visualization  

---

## 🧠 Models Used

| Model | Purpose |
|------|--------|
| Linear Regression | Baseline model |
| Random Forest | Final model (best performance) |
| XGBoost | Advanced boosting model |

---

## 📂 Project Structure

```bash
Financial_Project/
│
├── src/
│   ├── main.py        # Data preprocessing
│   ├── model.py       # Model training & evaluation
│   ├── app.py         # Streamlit dashboard
│
├── data/
│   └── Financials.csv
│
├── model.pkl          # Trained model
├── scores.json        # Model performance
├── images/            # Screenshots
└── README.md