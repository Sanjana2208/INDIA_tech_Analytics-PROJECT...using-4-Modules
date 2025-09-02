import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Smart India Dashboard", layout="wide")

#📊 Module 1: Technological Growth Analysis Across Indian States
def tech_growth_analysis():
    st.header("📊 Technological Growth Analysis Across Indian States")
    data = {
        'State': ['Karnataka', 'Maharashtra', 'Tamil Nadu', 'Telangana', 'Kerala',
                  'Gujarat', 'Delhi', 'Punjab', 'Rajasthan', 'Uttar Pradesh',
                  'West Bengal', 'Bihar', 'Odisha', 'Assam', 'Jharkhand'],
        'Startups': [1200, 1100, 950, 1000, 800, 700, 850, 400, 300, 250, 500, 150, 200, 180, 160],
        'TechParks': [15, 14, 12, 13, 10, 9, 11, 5, 4, 3, 6, 2, 3, 2, 2],
        'R&D_Centers': [50, 48, 45, 47, 40, 35, 38, 20, 18, 15, 25, 10, 12, 11, 10],
        'Label': [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)

    X = df[['Startups', 'TechParks', 'R&D_Centers']]
    y = df['Label']
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
    st.pyplot(fig)

# Sidebar Navigation
st.sidebar.title("📌 Select a Module : Technological Growth Analysis")
module = st.sidebar.radio("Choose :",
    ["Technological Growth Analysis"])


if module == "Technological Growth Analysis":
    tech_growth_analysis()
    
else: "Module Closed!"

