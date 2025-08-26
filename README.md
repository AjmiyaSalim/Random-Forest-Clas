# 💧 Water Potability Prediction

## 🎯 Objective
The main objective of this project is to **predict the potability of water** (safe to drink or not) using its chemical and physical characteristics.  
By leveraging machine learning, the project aims to provide a quick and reliable tool for assessing water quality, which can help in **early detection of unsafe water sources**.

---

## About the Model

- **Random Forest Classifier**  
  - An ensemble learning method that combines multiple decision trees.  
  - Handles non-linear relationships and imbalanced data effectively.  
  - Parameters used: `n_estimators=250`, `max_depth=13`, `class_weight='balanced'`.  

The trained model achieved a strong accuracy on the test dataset, ensuring reliable predictions for potability classification.

---

## Dataset
  
- **Features (Inputs):**  
  - pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity  
- **Target (Output):**  
  - Potability → `1 = Safe to Drink`, `0 = Not Safe to Drink`
---

## 🌐 About Streamlit
**Streamlit** is an open-source Python library designed to build **interactive web applications** directly from Python scripts.  
It is especially popular among data scientists and machine learning practitioners for:  
- **Deploying ML models** quickly as shareable web apps.  
- **Creating dashboards** for data visualization and analysis.  
- **Rapid prototyping** without requiring web development skills.  


