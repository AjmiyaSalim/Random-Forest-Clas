import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    return pd.read_csv('C:\\Users\\ajuaj\\Downloads\\vs code\\Random Forest Clas\\water_quality_potability.csv')

@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=250,
        criterion='gini',
        max_depth=13,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

data = load_data()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = train_model(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

st.set_page_config(page_title="Water Potability Classifier", layout="wide")

st.title(" Water Potability Prediction App")
st.markdown(f"###  Model Accuracy: **{accuracy:.2f}%**")

st.header(" Enter Water Quality Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
    solids = st.number_input("Solids (ppm)", min_value=0.0)
    sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0)

with col2:
    hardness = st.number_input("Hardness (mg/L)", min_value=0.0)
    chloramines = st.number_input("Chloramines (ppm)", min_value=0.0)
    conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0)

with col3:
    organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0)
    trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0)


if st.button(" Predict Potability"):
    input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate,
                                 conductivity, organic_carbon, trihalomethanes, turbidity]],
                               columns=X.columns)
    prediction = model.predict(input_data)[0]
    result = " Safe to Drink" if prediction == 1 else " Not Safe to Drink"
    
    st.subheader(" Prediction Result:")
    st.success(result)
