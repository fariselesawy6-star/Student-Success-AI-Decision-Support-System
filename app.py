import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Student Success AI | DSS",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# FILE PATHS
# ==================================================
DATA_PATH = Path("data/student_data.csv")
MODEL_PATH = Path("student_model.joblib")
RESULTS_PATH = Path("data/model_results.txt")

# ==================================================
# EMERALD GREEN COLOR PALETTE
# ==================================================
GREEN = "#168568"
DARK_GREEN = "#105C49"
WARM_WHITE = "#F3F7F2"
CHARCOAL = "#26332F"
MINT = "#DCEDE5"
MUTED = "#68766F"
BORDER = "#D8E2DB"
WHITE = "#FFFFFF"
RED = "#C74D4D"
GOLD = "#C58B2A"

CHART_COLORS = [GREEN, "#75B69B", "#BCDCCB"]

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F3F7F2;
    color: #26332F;
}

[data-testid="stSidebar"] {
    background: #26332F;
    border-right: 1px solid #D8E2DB;
}

[data-testid="stSidebar"] * {
    color: #F3F7F2;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #D5E3DA;
}

h1, h2, h3 {
    color: #26332F;
    letter-spacing: -0.4px;
}

.hero {
    background: linear-gradient(120deg, #26332F 0%, #105C49 100%);
    padding: 30px 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 24px;
    border: 1px solid #315B4D;
}

.hero h1 {
    color: white;
    font-size: 32px;
    margin: 0 0 10px 0;
    font-weight: 800;
}

.hero p {
    color: #D9EAE0;
    font-size: 15px;
    margin: 0;
    line-height: 1.7;
}

.eyebrow {
    color: #A8DCC5;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #D8E2DB;
    border-radius: 14px;
    padding: 20px;
    min-height: 125px;
    box-shadow: 0 3px 12px rgba(38, 51, 47, 0.04);
}

.metric-label {
    color: #68766F;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 12px;
}

.metric-value {
    color: #26332F;
    font-size: 28px;
    font-weight: 800;
    line-height: 1.2;
}

.metric-note {
    color: #168568;
    font-size: 12px;
    margin-top: 8px;
    font-weight: 600;
}

.section-heading {
    font-size: 19px;
    font-weight: 800;
    color: #26332F;
    margin: 20px 0 12px 0;
}

.info-card {
    background: #FFFFFF;
    border: 1px solid #D8E2DB;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 12px;
}

.info-card h3 {
    margin-top: 0;
    font-size: 16px;
}

.info-card p {
    color: #68766F;
    line-height: 1.7;
    font-size: 14px;
}

.pill {
    display: inline-block;
    background: #DCEDE5;
    color: #105C49;
    border-radius: 20px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #D8E2DB;
    padding: 16px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #68766F;
}

div[data-testid="stMetricValue"] {
    color: #168568;
}

.stButton > button {
    background: #168568;
    color: #FFFFFF;
    border: 1px solid #168568;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 700;
    min-height: 42px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #105C49;
    border-color: #105C49;
    color: #FFFFFF;
}

div[data-baseweb="select"] > div {
    background: #FFFFFF;
    border-color: #D8E2DB;
    border-radius: 8px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #D8E2DB;
    border-radius: 10px;
    overflow: hidden;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #D8E2DB;
}

a {
    color: #168568;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.footer {
    color: #68766F;
    text-align: center;
    padding: 22px 0 10px 0;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA AND MODEL
# ==================================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not DATA_PATH.exists():
    st.error("Dataset not found: data/student_data.csv")
    st.info("Make sure your dataset file is inside the data folder.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Trained model not found: student_model.joblib")
    st.info("Run train_model.py first to train and save the model.")
    st.stop()

try:
    df = load_data()
    model = load_model()
except Exception as error:
    st.error(f"Could not load the dataset or model: {error}")
    st.stop()

TARGET = "Target"

if TARGET not in df.columns:
    st.error("The required Target column was not found in the dataset.")
    st.stop()

if df.empty:
    st.error("The dataset contains no records.")
    st.stop()

features = df.drop(columns=[TARGET])
target = df[TARGET]
classes = list(model.classes_)

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow, title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(ax):
    ax.set_facecolor(WHITE)
    ax.figure.set_facecolor(WHITE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)


def get_recommendation(prediction):
    if prediction == "Dropout":
        return {
            "priority": "High priority for review",
            "action": (
                "Arrange an academic advising meeting, review course "
                "performance and attendance where available, and ask "
                "whether the student needs tutoring or other support."
            ),
            "color": RED,
        }

    if prediction == "Enrolled":
        return {
            "priority": "Monitor progress",
            "action": (
                "Review the student's academic progress and consider "
                "regular check-ins, tutoring, or guidance for challenging courses."
            ),
            "color": GOLD,
        }

    return {
        "priority": "Continue positive support",
        "action": (
            "Continue appropriate academic guidance and encourage "
            "participation in learning and career-development opportunities."
        ),
        "color": GREEN,
    }


def predict_record(record):
    prediction = model.predict(record)[0]
    probabilities = model.predict_proba(record)[0]
    return prediction, probabilities


# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 22px 0;">
            <div style="font-size: 34px;">🎓</div>
            <div style="font-size: 21px; font-weight: 800; color: white;">
                Student Success AI
            </div>
            <div style="font-size: 11px; color: #A8DCC5; letter-spacing: 1.3px;">
                DECISION SUPPORT SYSTEM
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**NAVIGATION**")

    page = st.radio(
        "Choose a page",
        [
            "Overview",
            "AI Prediction",
            "Decision Center",
            "Model Evaluation",
            "About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**PROJECT STATUS**")
    st.markdown("🟢 Dataset loaded")
    st.markdown("🟢 Model loaded")
    st.markdown("🟢 Prediction interface ready")

    st.markdown("---")
    st.caption("Academic prototype")
    st.caption("Human review is essential.")

# ==================================================
# PAGE 1: OVERVIEW
# ==================================================
if page == "Overview":
    page_header(
        "STUDENT ANALYTICS PLATFORM",
        "Student Success Dashboard",
        "Explore student outcome patterns and discover how AI can support "
        "early academic guidance and informed decision-making.",
    )

    total_students = len(df)
    total_features = len(features.columns)
    class_count = target.nunique()
    dropout_count = int((target == "Dropout").sum())
    dropout_rate = (
        dropout_count / total_students * 100 if total_students else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "TOTAL RECORDS",
            f"{total_students:,}",
            "Records in the dataset",
        )

    with c2:
        metric_card(
            "INPUT FEATURES",
            str(total_features),
            "Predictor variables",
        )

    with c3:
        metric_card(
            "OUTCOME CLASSES",
            str(class_count),
            "Recorded outcome categories",
        )

    with c4:
        metric_card(
            "DROPOUT SHARE",
            f"{dropout_rate:.1f}%",
            "Observed dataset proportion",
        )

    st.markdown(
        '<div class="section-heading">Student Outcome Analysis</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 1])

    counts = target.value_counts().reindex(classes, fill_value=0)

    with left:
        st.markdown("#### Outcome distribution")

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [
            CHART_COLORS[i % len(CHART_COLORS)]
            for i in range(len(counts))
        ]

        bars = ax.bar(
            counts.index.astype(str),
            counts.values,
            color=colors,
            width=0.62,
        )

        style_chart(ax)
        ax.set_ylabel("Number of records")
        ax.bar_label(
            bars,
            padding=3,
            color=CHARCOAL,
            fontsize=10,
        )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("#### Share by outcome")

        fig, ax = plt.subplots(figsize=(5, 4))
        pie_colors = [
            CHART_COLORS[i % len(CHART_COLORS)]
            for i in range(len(counts))
        ]

        ax.pie(
            counts.values,
            labels=counts.index.astype(str),
            autopct="%1.1f%%",
            startangle=90,
            colors=pie_colors,
            textprops={"color": CHARCOAL, "fontsize": 9},
            wedgeprops={
                "width": 0.42,
                "edgecolor": WHITE,
                "linewidth": 2,
            },
        )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(
        '<div class="section-heading">Dataset Preview</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These charts describe the dataset, not a forecast of an individual student's future."
    )

# ==================================================
# PAGE 2: AI PREDICTION
# ==================================================
elif page == "AI Prediction":
    page_header(
        "MACHINE LEARNING",
        "AI Student Outcome Prediction",
        "Select an existing dataset record and explore the predicted outcome "
        "and the model's estimated class probabilities.",
    )

    st.info(
        "Demo mode: this page predicts outcomes for existing dataset records. "
        "It does not create a new student profile."
    )

    selected_index = st.number_input(
        "Select dataset record index",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
    )

    record = features.iloc[[int(selected_index)]]

    with st.expander("View selected record inputs"):
        st.dataframe(record, use_container_width=True)

    if st.button("Run AI Prediction", type="primary"):
        prediction, probabilities = predict_record(record)
        recommendation = get_recommendation(prediction)

        st.markdown("---")
        st.markdown("### Prediction Result")

        result_col, chart_col = st.columns([1, 1.5])

        with result_col:
            st.markdown(
                f"""
                <div class="info-card" style="border-top: 5px solid {recommendation['color']};">
                    <div style="color: {MUTED}; font-size: 12px; font-weight: 700;">
                        PREDICTED OUTCOME
                    </div>
                    <div style="font-size: 28px; font-weight: 800;
                                color: {recommendation['color']}; margin: 12px 0;">
                        {prediction}
                    </div>
                    <span class="pill">{recommendation['priority']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with chart_col:
            st.markdown("#### Class probability estimates")

            prob_df = pd.DataFrame({
                "Outcome": classes,
                "Probability (%)": probabilities * 100,
            }).sort_values("Probability (%)", ascending=True)

            fig, ax = plt.subplots(figsize=(7, 3.5))

            bar_colors = [
                GREEN if label == prediction else "#BCDCCB"
                for label in prob_df["Outcome"]
            ]

            bars = ax.barh(
                prob_df["Outcome"],
                prob_df["Probability (%)"],
                color=bar_colors,
            )

            ax.set_xlim(0, 110)
            ax.set_xlabel("Estimated probability (%)")
            ax.set_facecolor(WHITE)
            fig.patch.set_facecolor(WHITE)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(BORDER)
            ax.spines["bottom"].set_color(BORDER)
            ax.tick_params(colors=MUTED)
            ax.grid(axis="x", alpha=0.15)
            ax.set_axisbelow(True)

            ax.bar_label(
                bars,
                fmt="%.1f%%",
                padding=3,
                color=CHARCOAL,
            )

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("#### Suggested support action")
        st.write(recommendation["action"])

        st.warning(
            "Model probabilities are estimates, not guarantees. "
            "A qualified staff member should review relevant context "
            "before deciding whether support is needed."
        )

# ==================================================
# PAGE 3: DECISION CENTER
# ==================================================
elif page == "Decision Center":
    page_header(
        "ACTIONABLE INSIGHTS",
        "Academic Decision Center",
        "Use model output as a starting point for supportive, "
        "human-led academic review.",
    )

    st.markdown(
        """
        <div class="info-card">
            <h3>Decision-support workflow</h3>
            <p>
            1. Select a dataset record.<br>
            2. Review the predicted outcome.<br>
            3. Consider the suggested support action.<br>
            4. Let qualified staff review the student's individual circumstances.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    idx = st.selectbox(
        "Choose a dataset record",
        options=list(range(len(df))),
        format_func=lambda i: f"Record #{i}",
    )

    chosen = features.iloc[[idx]]

    st.markdown("#### Record details")
    st.dataframe(chosen, use_container_width=True)

    if st.button("Generate Support Recommendation", type="primary"):
        prediction, probabilities = predict_record(chosen)
        rec = get_recommendation(prediction)

        st.markdown("---")
        st.markdown("### Recommended Review Priority")

        st.markdown(
            f"""
            <div class="info-card" style="border-left: 6px solid {rec['color']};">
                <div style="font-size: 12px; color: {MUTED}; font-weight: 700;">
                    MODEL-PREDICTED OUTCOME
                </div>
                <div style="font-size: 26px; color: {rec['color']};
                            font-weight: 800; margin: 8px 0;">
                    {prediction}
                </div>
                <span class="pill">{rec['priority']}</span>
                <hr>
                <h3>Suggested support action</h3>
                <p>{rec['action']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        probability_df = pd.DataFrame({
            "Outcome": classes,
            "Probability": probabilities,
        }).sort_values("Probability", ascending=False)

        probability_df["Probability"] = probability_df[
            "Probability"
        ].map(lambda value: f"{value:.1%}")

        st.markdown("#### Prediction probability breakdown")

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True,
        )

        st.warning(
            "This is an academic prototype, not an automated decision-maker. "
            "Do not use the prediction alone to deny admission, financial aid, "
            "or educational opportunities."
        )

# ==================================================
# PAGE 4: MODEL EVALUATION
# ==================================================
elif page == "Model Evaluation":
    page_header(
        "MODEL TRANSPARENCY",
        "Model Evaluation & Performance",
        "Review saved test results and explore the relative importance "
        "of features used by the trained Random Forest model.",
    )

    if RESULTS_PATH.exists():
        st.markdown("### Classification Report")
        results_text = RESULTS_PATH.read_text(encoding="utf-8")
        st.code(results_text, language="text")
    else:
        st.warning(
            "Evaluation report not found at data/model_results.txt."
        )

    st.markdown("### Feature Importance")

    try:
        classifier = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]

        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        })

        importance_df = importance_df.nlargest(
            15, "Importance"
        ).sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"],
            color=GREEN,
        )

        ax.set_facecolor(WHITE)
        fig.patch.set_facecolor(WHITE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BORDER)
        ax.spines["bottom"].set_color(BORDER)
        ax.tick_params(colors=MUTED)
        ax.set_xlabel("Relative feature importance")
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Feature importance describes how the trained model used its input features. "
            "It does not establish causation."
        )

    except Exception as error:
        st.info(
            "Feature importance could not be displayed. "
            f"Details: {error}"
        )

    st.markdown("### How to interpret the evaluation")

    st.markdown(
        """
        - **Accuracy:** the proportion of correct predictions on the held-out test set.
        - **Precision:** how often predictions for a class were correct.
        - **Recall:** how many actual cases in a class the model identified.
        - **F1-score:** a combined measure of precision and recall.
        - **Class differences:** results should be inspected separately for each outcome.
        """
    )

    st.warning(
        "Historical test results may not generalize to other institutions or future cohorts. "
        "Accuracy alone is not enough to validate a real-world intervention system."
    )

# ==================================================
# PAGE 5: ABOUT PROJECT
# ==================================================
elif page == "About Project":
    page_header(
        "PROJECT OVERVIEW",
        "About the Student Success DSS",
        "An academic prototype combining machine learning, "
        "analytics, and human-centered decision support.",
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="info-card">
                <h3>🎯 Project Objective</h3>
                <p>
                Explore how student-record data and machine learning can help
                identify students who may benefit from timely academic guidance.
                </p>
            </div>
            <div class="info-card">
                <h3>🧠 AI Component</h3>
                <p>
                A Random Forest classifier predicts one of three dataset outcomes:
                Dropout, Enrolled, or Graduate.
                </p>
            </div>
            <div class="info-card">
                <h3>📊 Decision Support</h3>
                <p>
                Dashboard analytics, model predictions, probability estimates,
                evaluation results, and suggested human-reviewed support actions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="info-card">
                <h3>🛠️ Technology Stack</h3>
                <p>
                Python · Streamlit · Pandas · NumPy · Scikit-learn ·
                Matplotlib · Joblib
                </p>
            </div>
            <div class="info-card">
                <h3>⚖️ Responsible Use</h3>
                <p>
                Model outputs may reflect historical patterns or data bias.
                They should not be the sole basis for decisions affecting students.
                </p>
            </div>
            <div class="info-card">
                <h3>🔍 Current Limitation</h3>
                <p>
                This interface demonstrates predictions on existing records.
                It is not a validated deployment system or a replacement for academic staff.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Project Workflow")

    st.markdown(
        """
        <div class="info-card">
            <div style="font-weight: 700; color: #105C49; line-height: 2;">
                Data Collection → Preprocessing → Model Training → Evaluation
                → Prediction → Human Review → Support Action
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Dataset Outcome Classes")

    outcome_table = pd.DataFrame({
        "Class": ["Dropout", "Enrolled", "Graduate"],
        "Meaning": [
            "The dataset records the student as having dropped out.",
            "The dataset records the student as still enrolled.",
            "The dataset records the student as having graduated.",
        ],
    })

    st.dataframe(
        outcome_table,
        use_container_width=True,
        hide_index=True,
    )

# ==================================================
# FOOTER
# ==================================================
st.markdown(
    """
    <div class="footer">
        <strong>Student Success AI</strong> · Decision Support System Prototype<br>
        Academic demonstration · Human judgment remains essential
    </div>
    """,
    unsafe_allow_html=True,
)
