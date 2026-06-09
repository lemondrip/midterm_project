import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import metrics as mt


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

    df["pass_fail"] = df["placement_status"].map({
        "Placed": "Passed",
        "Not Placed": "Failed"
    })

    # -----------------------------
    # Helpers
    # -----------------------------
    def add_pass_line(ax):
        ax.axhline(70, linestyle="--", linewidth=1.5)

    def scatter_chart(df, x_col, x_label, title):
        """Scatterplot of a behavior vs exam score, colored by pass/fail."""
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.scatterplot(
            data=df, x=x_col, y="exam_score",
            hue="pass_fail", alpha=0.65, ax=ax
        )
        add_pass_line(ax)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Exam Score")
        ax.set_title(title)
        ax.legend(title="Outcome")
        st.pyplot(fig)          # <- was plt.show()

        st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        fmt=".2f",        
        cmap="coolwarm",  
        ax=ax
    )
    ax.set_title("Correlation Between Numeric Variables")
    st.pyplot(fig)


    # -----------------------------
    # 1. Exam score distribution
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x="exam_score", bins=25, kde=True, ax=ax)
    ax.axvline(70, linestyle="--", linewidth=1.5)
    ax.text(71, ax.get_ylim()[1] * 0.9, "Passing score: 70", fontsize=10)
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Exam Scores")
    st.pyplot(fig)              # <- was plt.show()

    # -----------------------------
    # 2. Behavioral factors vs exam score
    # -----------------------------
    scatter_chart(df, "study_hours",    "Study Hours",    "Study Hours vs Exam Score")
    scatter_chart(df, "attendance",     "Attendance (%)", "Attendance vs Exam Score")
    scatter_chart(df, "sleep_hours",    "Sleep Hours",    "Sleep Hours vs Exam Score")
    scatter_chart(df, "internet_usage", "Internet Usage", "Internet Usage vs Exam Score")

    # -----------------------------
    # 3. Academic preparation factors
    # -----------------------------
    scatter_chart(df, "assignments_completed", "Assignments Completed",
                  "Assignments Completed vs Exam Score")
    scatter_chart(df, "previous_score", "Previous Score",
                  "Previous Score vs Exam Score")

    # -----------------------------
    # 4. Feature importance (Linear Regression)
    # -----------------------------
    feature_cols = [
        "study_hours", "attendance", "sleep_hours",
        "internet_usage", "assignments_completed", "previous_score"
    ]

    X = df[feature_cols]
    y = df["exam_score"]

    model = LinearRegression()
    model.fit(X, y)

    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_
    })
    coef_df["Absolute Impact"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Absolute Impact", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(coef_df["Feature"], coef_df["Coefficient"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Regression Coefficient")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance for Predicting Exam Score")
    st.pyplot(fig)              # <- was plt.show()


   
    st.subheader("Most Common Weak Spots Among Failing Students")
    st.caption("Among students who failed, how often each factor falls below a healthy threshold. "
               "Students can have more than one weak spot, so bars don't sum to 100%.")

    failed = df[df["pass_fail"] == "Failed"].copy()

    # Thresholds: below these counts as a 'weak spot' (higher internet = worse)
    deficiencies = {
        "Low study hours":        failed["study_hours"] < 5,
        "Low attendance":         failed["attendance"] < 70,
        "Few assignments done":   failed["assignments_completed"] < 8,
        "Low previous score":     failed["previous_score"] < 60,
        "Short sleep":            failed["sleep_hours"] < 6,
        "High internet usage":    failed["internet_usage"] > 6,
    }

    counts = {k: v.sum() for k, v in deficiencies.items()}
    defi_df = (pd.DataFrame({"Weak Spot": counts.keys(), "Students": counts.values()})
               .sort_values("Students", ascending=True))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(defi_df["Weak Spot"], defi_df["Students"])
    ax.set_xlabel("Number of Failing Students")
    ax.set_title("Common Weak Spots Among Failing Students")
    st.pyplot(fig)


elif page == "Prediction":
    st.subheader("Data Preview")

    # ---------------- Load ----------------

    TARGET = "exam_score"      # variable to predict
    TRAIN_SIZE = 0.70          # fraction used for training
    SEED = 42

    num_df = df.drop(columns=["placement_status"])   # numeric features only

    features = [c for c in num_df.columns if c != TARGET]
    x = num_df[features]
    y = df[TARGET]

    print("=" * 60)
    print(f"STUDENT REGRESSION  |  predicting: {TARGET}")
    print("=" * 60)
    print(f"Rows: {len(df)}   Features: {features}\n")

    # ---------------- Fit ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=1 - TRAIN_SIZE, random_state=SEED
    )
    lm = LinearRegression().fit(X_train, y_train)
    pred = lm.predict(X_test)

    # ---------------- Results ----------------
    print("RESULTS")
    print(f"  Explained variance : {mt.explained_variance_score(y_test, pred) * 100:6.2f} %")
    print(f"  MAE                : {mt.mean_absolute_error(y_test, pred):6.2f}")
    print(f"  MSE                : {mt.mean_squared_error(y_test, pred):6.2f}")
    print(f"  R-Square           : {mt.r2_score(y_test, pred):6.3f}")

    # ---------------- Raw coefficients ----------------
    coef = pd.DataFrame({"feature": features, "coef": lm.coef_})
    coef = coef.reindex(coef.coef.abs().sort_values(ascending=False).index)
    print("\nRAW COEFFICIENTS (effect of +1 unit on " + TARGET + ")")
    for _, r in coef.iterrows():
        print(f"  {r.feature:24s} {r.coef:+8.3f}")
    print(f"  {'intercept':24s} {lm.intercept_:+8.3f}")

    # ---------------- Standardized betas (comparable scale) ----------------
    xz = (x - x.mean()) / x.std()
    yz = (y - y.mean()) / y.std()
    lmz = LinearRegression().fit(xz, yz)
    beta = pd.DataFrame({"feature": features, "std_beta": lmz.coef_})
    beta = beta.reindex(beta.std_beta.abs().sort_values(ascending=False).index)
    print("\nSTANDARDIZED BETAS (ranked levers, apples-to-apples)")
    for _, r in beta.iterrows():
        bar = "#" * int(abs(r.std_beta) * 40)
        print(f"  {r.feature:24s} {r.std_beta:+6.3f}  {bar}")

    print("\n" + "=" * 60)
