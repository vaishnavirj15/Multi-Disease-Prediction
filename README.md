# Multi-Disease Prediction Platform 🚑

An **interactive web-based platform** built with **Streamlit** that empowers users to assess their health risks for **Diabetes**, **Stroke**, and **Framingham Heart Disease** using personalized health metrics. This platform combines **data-driven insights** with **machine learning models** to provide reliable risk predictions and actionable health feedback.  

> Achieves up to **90% prediction accuracy** for stroke risk using advanced ML techniques and addresses class imbalance via **SMOTE**.

---

## 🔑 Key Features
- **Multi-Disease Prediction**: Risk assessment for **Diabetes**, **Stroke**, and **Heart Disease** in one platform.  
- **Interactive Data Input**: User-friendly forms for real-time, personalized predictions.  
- **Accurate & Explainable Models**: Implements **Random Forest** and **Logistic Regression** for robust, interpretable predictions.  
- **Comprehensive Data Analysis**: Exploratory Data Analysis (EDA), feature importance, and performance visualizations to validate model effectiveness.  
- **Evaluation Metrics**: Confusion matrices, classification reports, and accuracy scores for transparent model evaluation.

---

## 🛠 Technologies Used
- **Programming**: Python  
- **Frameworks & Libraries**: Streamlit, Scikit-learn, Pandas, Matplotlib, Seaborn, Imbalanced-Learn  
- **Development Environment**: Google Colab / Local Python Environment

---

## ⚡ Data Analysis & Insights
- Conducted **EDA** to understand key health indicators impacting each disease.  
- Identified correlations and trends to improve model feature selection.  
- Handled **class imbalance** in datasets using **SMOTE**, enhancing prediction reliability.  
- Visualized metrics and insights for easier interpretation and user trust.

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.7+ OR Google Colab for cloud-based development.
  
---
### Installation
1. **Clone the repository**:
```bash
git clone https://github.com/vaishnavirj15/Multi-Disease-Prediction.git
cd Multi-Disease-Prediction
```

2.**Install Dependencies**:

```bash
pip install streamlit pandas matplotlib seaborn scikit-learn imbalanced-learn
```
3. **Prepare datasets**:

*Upload datasets to Google Drive (if using Colab) or local folder*.

*Update dataset paths in the code accordingly*.

🚀 Running the Application
Ensure app.py is in your working directory.

Locally:
```bash
streamlit run app.py
```
Colab:
```bash

!streamlit run app.py & npx localtunnel --port 8501
```
---
📊 Project Impact:

-**Demonstrates end-to-end machine learning workflow: data collection → preprocessing → model training → evaluation → deployment.**

-**Provides real-time health risk predictions, supporting proactive health management.**

-**A portfolio-worthy project showcasing ML modeling, data analysis, and interactive web deployment for recruiters in data science, AI, and healthcare analytics roles.**

---
📂 Repository Contents
-app.py – Main Streamlit application

-diabetes.csv, framingham.csv, healthcare-dataset-stroke-data.csv – Source datasets

-healthmodel.ipynb – ML modeling, EDA, and preprocessing notebook
