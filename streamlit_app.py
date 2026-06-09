import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

st.sidebar.title("Student Performance Insights Dashboard")
st.sidebar.text("Presented by: Abhimanyu, Eric, Jessie, and Joe")
st.sidebar.text("Predicting how students will place on exams is a difficult issue as there are multiple factors that can contribute to passing vs. failing. In order to best assist a student, it is important to understand what they are currently doing and what adjustments will allow for the most impactful results")

df = pd.read_csv("student_dataset.csv")


page = st.sidebar.selectbox("Select Page",["Introduction","Data Viz","Prediction"])

if page == "Introduction":

    st.subheader("Data Preview")
    st.text("We use a Kaggle dataset with 10,000 student records. This project uses study hours, attendance, sleep, internet usage, assignments completed, previous scores, exam scores, and placement status as variables. ")
    st.dataframe(df.head())

    st.subheader("00 - Show Dataset")

    data = [
        ("Study Hours", "Number of hours per day the student spends studying outside of class"),
        ("Attendance", "Percentage of classes attended; consistent attendance is linked to better academic outcomes"),
        ("Sleep Hours", "Average hours of sleep per night; sleep quality affects memory consolidation and focus"),
        ("Internet Usage", "Daily hours spent on the internet, which may reflect both research and distraction time"),
        ("Assignments Completed", "Number of assignments submitted; a proxy for student engagement and work ethic"),
        ("Previous Score", "Score from the student's most recent prior exam, capturing historical academic baseline"),
        ("Placement Status", "Whether the student was placed after graduation — Placed or Not Placed"),
        ("Exam Score", "The final exam score (0-100); the variable our model aims to predict"),
    ]

    cols = st.columns(8)
    for col, (name, desc) in zip(cols, data):
        col.markdown(f"**{name}**")
        col.caption(desc)

    st.text("Number of rows and columns helps us to determine how large the dataset is.")
    st.text("(Rows, Columns)")
    st.code(str(df.shape), language=None)

    st.subheader("01 - Description")
    st.dataframe(df.describe())

    st.subheader("02 - Missing Values")
    st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")

    missing = df.isnull().sum()
    pct = round((missing.sum() / (df.shape[0] * df.shape[1])) * 100, 2)

    st.markdown(f"**Percentage of total missing values:** `{pct}`")
    st.dataframe(missing.to_frame())

    st.subheader("03 - Completeness")
    st.markdown("Completeness is defined as the ratio of non-missing values to total records in dataset.")

    completeness = df.count()
    ratio = round(completeness.sum() / (df.shape[0] * df.shape[1]), 2)

    st.markdown(f"**Completeness ratio:** `{ratio}`")
    st.dataframe(completeness.to_frame())

elif page == "Data Viz":
    st.subheader("Data Viz")








elif page == "Prediction":
    st.subheader("Data Preview")
