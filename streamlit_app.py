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
        ("Placement Status", "Whether the student has passed or failed — 70 is pass mark"),
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

    # Sidebar
# -----------------------------
st.sidebar.title("🎓 Dashboard Controls")
st.sidebar.write(
    "Use these filters to explore how different student groups perform."
)

min_score = float(df["exam_score"].min())
max_score = float(df["exam_score"].max())

score_range = st.sidebar.slider(
    "Filter by exam score",
    min_value=float(np.floor(min_score)),
    max_value=float(np.ceil(max_score)),
    value=(float(np.floor(min_score)), float(np.ceil(max_score)))
)

selected_outcomes = st.sidebar.multiselect(
    "Filter by outcome",
    options=sorted(df["pass_fail"].unique()),
    default=sorted(df["pass_fail"].unique())
)

filtered_df = df[
    (df["exam_score"] >= score_range[0]) &
    (df["exam_score"] <= score_range[1]) &
    (df["pass_fail"].isin(selected_outcomes))
]

# -----------------------------
# Header
# -----------------------------
st.title("🎓 Student Exam Success Dashboard")

st.markdown(
    """
    This dashboard explores the factors that influence whether students pass or fail an exam.
    Since passing is determined by scoring **70 or higher**, the goal is not only to show who passed,
    but to understand which student behaviors are associated with stronger exam performance.
    """
)

st.markdown("---")

# -----------------------------
# Section 1: Overall performance
# -----------------------------
st.header("1. Overall Student Performance")

st.markdown(
    """
    We begin with a high-level view of the class. These metrics summarize the general academic
    performance and give context before analyzing individual factors.
    """
)

total_students = len(filtered_df)
avg_score = filtered_df["exam_score"].mean()
pass_rate = (filtered_df["pass_fail"] == "Passed").mean() * 100
avg_study = filtered_df["study_hours"].mean()
avg_attendance = filtered_df["attendance"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Students", f"{total_students:,}")
col2.metric("Average Exam Score", f"{avg_score:.1f}")
col3.metric("Pass Rate", f"{pass_rate:.1f}%")
col4.metric("Average Attendance", f"{avg_attendance:.1f}%")

st.write("")



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
    st.subheader("Exam Score Prediction Model")
    st.text("A Linear Regression model trained to predict exam score from student habits.")

    # ---------------- Setup ----------------
    TARGET = "exam_score"      # variable to predict
    TRAIN_SIZE = 0.70          # fraction used for training
    SEED = 42

    num_df = df.drop(columns=["placement_status"])   # numeric features only
    features = [c for c in num_df.columns if c != TARGET]
    x = num_df[features]
    y = df[TARGET]

    st.markdown(f"**Rows:** {len(df)}  |  **Features:** {', '.join(features)}")

    # ---------------- Fit ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=1 - TRAIN_SIZE, random_state=SEED
    )
    lm = LinearRegression().fit(X_train, y_train)
    pred = lm.predict(X_test)

    # ---------------- Results ----------------
    st.subheader("Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Explained Variance", f"{mt.explained_variance_score(y_test, pred) * 100:.1f}%")
    c2.metric("MAE", f"{mt.mean_absolute_error(y_test, pred):.2f}")
    c3.metric("MSE", f"{mt.mean_squared_error(y_test, pred):.2f}")
    c4.metric("R-Square", f"{mt.r2_score(y_test, pred):.3f}")

    # ---------------- Raw coefficients ----------------
    st.subheader("Raw Coefficients")
    st.caption(f"Effect of a +1 unit change in each feature on {TARGET}.")
    coef = pd.DataFrame({"feature": features, "coef": lm.coef_})
    coef = coef.reindex(coef.coef.abs().sort_values(ascending=False).index)
    st.dataframe(coef.reset_index(drop=True))
    st.markdown(f"**Intercept:** `{lm.intercept_:.3f}`")

    # ---------------- Standardized betas ----------------
    st.subheader("Standardized Betas (Ranked Levers)")
    st.caption("Features rescaled to the same units so their influence is directly comparable.")
    xz = (x - x.mean()) / x.std()
    yz = (y - y.mean()) / y.std()
    lmz = LinearRegression().fit(xz, yz)
    beta = pd.DataFrame({"feature": features, "std_beta": lmz.coef_})
    beta = beta.reindex(beta.std_beta.abs().sort_values(ascending=True).index)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(beta["feature"], beta["std_beta"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Standardized Coefficient")
    ax.set_title("Ranked Influence on Exam Score")
    st.pyplot(fig)

 
    # ==========================================================
    # Interactive predictor — full input, builds on the lm model above
    # ==========================================================
    st.subheader("Try It: Predict Placement")
    st.caption("Enter the student's details. The model predicts their exam score "
               "and whether they place.")

    # friendly labels for the existing feature columns
    labels = {
        "study_hours": "Study Hours",
        "attendance": "Attendance (%)",
        "sleep_hours": "Sleep Hours",
        "internet_usage": "Internet Usage (hrs)",
        "assignments_completed": "Assignments Completed",
        "previous_score": "Previous Score",
    }

    inputs = {}
    cols = st.columns(3)
    for i, feat in enumerate(features):                 # reuse 'features' from above
        inputs[feat] = cols[i % 3].number_input(
            labels.get(feat, feat),
            value=float(round(x[feat].mean(), 1))       # reuse 'x' from above
        )

    if st.button("Predict Placement"):
        row = pd.DataFrame([inputs])[features]          # match the model's column order
        predicted_score = float(lm.predict(row)[0])     # reuse 'lm' from above
        placed = predicted_score >= 70

        st.metric("Predicted Exam Score", f"{predicted_score:.1f}")
        if placed:
            st.success("Result: **Placed**  (predicted score is 70 or above)")
        else:
            st.error("Result: **Not Placed**  (predicted score is below 70)")
