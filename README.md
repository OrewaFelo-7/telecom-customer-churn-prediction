# 📱 Telecom Customer Churn Prediction

Predicting customer churn for a telecom provider using 100,000 customer records 
and ~100 features. Achieved **ROC-AUC 0.693** with LightGBM and identified 
**$34.9M in annual revenue at risk**.

---

## 📊 Project Overview

| Aspect | Detail |
|--------|--------|
| **Dataset** | 100,000 customers × ~100 features |
| **Target** | Churn within 31-60 days |
| **Churn Rate** | 49.6% (2.3× industry average of 21.5%) |
| **Best Model** | LightGBM |
| **Primary Metric** | ROC-AUC = 0.693 |
| **Business Impact** | 15-25% projected churn reduction |

---

## 🔑 Key Findings

- **Danger Zone**: Customers at 13-24 months tenure show **51.8% churn**
- **Device Age**: 2+ year old devices → **57.9% churn rate**
- **Usage Signal**: `change_mou` drop &lt; -30% → strongest early warning
- **Revenue at Risk**: $34.9M annually (conservative $7.3M recoverable)

---

## 🏗️ Repository Structure
