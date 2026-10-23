import streamlit as st
import pandas as pd
import numpy as np
import pickle

model=pickle.load(open("rf_model.pkl", 'rb'))
scaler=pickle.load(open("scaler.pkl", 'rb'))

st.title("Insurance Premium Prediction")
st.write("predict your estimated insurance charges based on personal and lifecycle factors")

age=st.number_input("enter age", min_value=0, max_value=100, step=1)
sex=st.selectbox("select gender", ['male', 'female'])
bmi=st.number_input("enter bmi", min_value=10.0, max_value=60.0, step=0.1)
children=st.number_input("number of children", min_value=0, max_value=10, step=1)
smoker=st.selectbox("do you smoke", ['yes','no'])
region=st.selectbox("select region", ['northeast','northwest','southeast','southwest'])

if st.button("Predict"):
    input_df=pd.DataFrame({
        "age":[age],
        "sex":[sex],
        "bmi":[bmi],
        "children":[children],
        "smoker":[smoker],
        "region":[region]
    })

    input_df['sex']=input_df['sex'].map({"male":1, "female":0})
    input_df['smoker']=input_df['smoker'].map({"yes":1, "no":0})
    input_df['region']=input_df['region'].map({"southeast":0, "southwest":1,"northwest":2, "northeast":3})

    num_cols = ['age','bmi','children']
    input_df[num_cols]= scaler.transform(input_df[num_cols])

    prediction= model.predict(input_df)

    st.success(f"Estimated Annual Insurance Charge: INR {prediction[0]:,.2f}")
