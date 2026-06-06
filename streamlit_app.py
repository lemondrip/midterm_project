import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

df = pd.read_csv("student_dataset.csv")

#Design
st.title("Student Performance Insights Dashboard")
st.text("Presented by: Abimanyu, Eric, Jessie, and Joe")
st.text("Predicting how students will place on exams is a difficult issue as there are multiple factors that can contribute to passing vs. failing. In order to best assist a student, it is important to understand what they are currently doing and what adjustments will allow for the most impactful results")

st.subheader("Data Preview")
st.text("We use a Kaggle dataset with 10,000 student records. This project uses study hours, attendance, sleep, internet usage, assignments completed, previous scores, exam scores, and placement status as variables. ")
st.dataframe(df.head())

