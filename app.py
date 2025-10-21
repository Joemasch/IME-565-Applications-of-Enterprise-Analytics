import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from mapie.metrics import regression_coverage_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path("Admission_Predict.csv")
MODEL_PATH = Path("reg_admission.pickle")
HERO_IMAGE_PATH = Path("admission.jpg")
ALPHA = 0.1  # 90% confidence interval
CONFIDENCE_LEVEL = int((1 - ALPHA) * 100)


@st.cache_resource
def load_trained_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


@st.cache_data
def load_training_artifacts():
    """Return dataset pieces used throughout the app."""
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Chance of Admit"])
    y = df["Chance of Admit"]
    X_encoded = pd.get_dummies(X)
    split = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42,
    )
    train_X, test_X, train_y, test_y = split
    artifacts = {
        "raw_df": df,
        "feature_columns": X_encoded.columns.tolist(),
        "train_X": train_X,
        "test_X": test_X,
        "train_y": train_y,
        "test_y": test_y,
    }
    return artifacts


def build_feature_frame(values: dict, feature_columns: list[str]) -> pd.DataFrame:
    """Convert sidebar inputs into the encoded feature frame expected by the model."""
    base_df = pd.DataFrame(
        {
            "GRE Score": [values["gre_score"]],
            "TOEFL Score": [values["toefl_score"]],
            "University Rating": [values["university_rating"]],
            "SOP": [values["sop_score"]],
            "LOR": [values["lor_score"]],
            "CGPA": [values["cgpa"]],
            "Research": ["Yes" if values["research_experience"] else "No"],
        }
    )
    encoded_df = pd.get_dummies(base_df)
    encoded_df = encoded_df.reindex(columns=feature_columns, fill_value=0)
    return encoded_df


def run_point_and_interval_prediction(model, features: pd.DataFrame):
    """Return point prediction and the matching prediction interval for a single sample."""
    pred, intervals = model.predict(features, alpha=ALPHA)
    point_prediction = float(pred[0])
    # MAPIE returns shape (n_samples, 2, n_alpha); collapse the trailing axis when present.
    if intervals.ndim == 3:
        lower_bound, upper_bound = intervals[0, :, 0]
    else:
        lower_bound, upper_bound = intervals[0]
    return point_prediction, float(lower_bound), float(upper_bound)


def generate_diagnostics(model, artifacts):
    """Pre-compute diagnostics needed for the insights tab."""
    test_X = artifacts["test_X"]
    test_y = artifacts["test_y"]
    train_X = artifacts["train_X"]

    preds, intervals = model.predict(test_X, alpha=ALPHA)
    if intervals.ndim == 3:
        intervals = intervals[:, :, 0]
    residuals = test_y - preds
    coverage = regression_coverage_score(test_y, intervals[:, 0], intervals[:, 1])

    feature_importances = model.estimator_.single_estimator_.feature_importances_

    diagnostics = {
        "predictions": preds,
        "intervals": intervals,
        "residuals": residuals,
        "coverage": coverage,
        "feature_importances": feature_importances,
        "feature_names": train_X.columns.tolist(),
        "test_y": test_y,
    }
    return diagnostics


def plot_feature_importance(names, importances):
    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_idx = np.argsort(importances)
    ax.barh(np.array(names)[sorted_idx], importances[sorted_idx], color="#2ca02c")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    return fig


def plot_residual_histogram(residuals):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(residuals, bins=25, color="lime", edgecolor="black")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    return fig


def plot_predicted_vs_actual(actual, predicted):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.scatter(actual, predicted, color="blue", alpha=0.6, edgecolor="black", s=40)
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs. Actual")
    return fig


def plot_coverage(actual, intervals, coverage):
    sorted_idx = np.argsort(actual.to_numpy())
    sorted_actual = actual.iloc[sorted_idx].reset_index(drop=True)
    sorted_intervals = intervals[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sorted_actual, "go", markersize=3, label="Actual")
    ax.fill_between(
        np.arange(len(sorted_actual)),
        sorted_intervals[:, 0],
        sorted_intervals[:, 1],
        alpha=0.2,
        color="green",
        label="Prediction Interval",
    )
    ax.set_xlim([0, len(sorted_actual)])
    ax.set_xlabel("Samples")
    ax.set_ylabel("Chance of Admit")
    ax.set_title(f"Prediction Interval Coverage: {coverage * 100:.2f}%")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Graduate Admission Predictor", layout="wide")
    st.title("Graduate Admission Prediction")
    st.markdown(
        """
        This app estimates graduate admission chances using a random forest with MAPIE prediction intervals.
        **How to use it:** adjust the applicant profile in the sidebar, press **Predict**, then review the
        point estimate and confidence interval that appears below. Explore the *Model Insights* section to
        see how the model behaves on the holdout validation data.
        """
    )
    st.markdown("**What each input controls**")
    st.markdown(
        """
        - **TOEFL Score**: English proficiency score scaled 0–120.
        - **GRE Score**: Quantitative aptitude score (acts as the GPA analogue in this version of the model).
        - **CGPA**: Cumulative GPA on a 0–10 scale.
        - **Research Experience**: Check if the applicant has prior research exposure.
        - **University Rating**: Institution prestige rating from 1 (low) to 5 (high).
        - **Statement of Purpose (SOP)**: Strength of the SOP from 1.0 to 5.0.
        - **Letter of Recommendation (LOR)**: Average strength of recommendation letters from 1.0 to 5.0.
        """
    )
    if HERO_IMAGE_PATH.exists():
        st.image(
            str(HERO_IMAGE_PATH),
            caption="Prediction interval example from the original analysis.",
            use_container_width=True,
        )

    model = load_trained_model()
    artifacts = load_training_artifacts()

    st.sidebar.header("Applicant Profile")
    toefl_score = st.sidebar.number_input("TOEFL Score", min_value=0, max_value=120, value=100, step=1)
    gre_score = st.sidebar.number_input(
        "GRE Score (enter GPA equivalent if preferred)",
        min_value=250,
        max_value=340,
        value=320,
        step=1,
        help="The model was trained with GRE scores; treated as the requested GPA input.",
    )
    cgpa = st.sidebar.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.5, step=0.01)
    research_experience = st.sidebar.checkbox("Research Experience", value=True)
    st.sidebar.subheader("Profile Strength")
    university_rating = st.sidebar.slider("University Rating", min_value=1, max_value=5, value=3)
    sop_score = st.sidebar.slider("Statement of Purpose (SOP)", min_value=1.0, max_value=5.0, value=3.5, step=0.5)
    lor_score = st.sidebar.slider("Letter of Recommendation (LOR)", min_value=1.0, max_value=5.0, value=3.5, step=0.5)
    predict_triggered = st.sidebar.button("Predict")

    feature_columns = artifacts["feature_columns"]
    user_values = {
        "toefl_score": toefl_score,
        "gre_score": gre_score,
        "cgpa": cgpa,
        "research_experience": research_experience,
        "university_rating": university_rating,
        "sop_score": sop_score,
        "lor_score": lor_score,
    }

    with st.spinner("Preparing model insights..."):
        diagnostics = generate_diagnostics(model, artifacts)

    st.subheader("Admission Chance")
    if predict_triggered:
        features = build_feature_frame(user_values, feature_columns)
        point_pred, lower_bound, upper_bound = run_point_and_interval_prediction(model, features)
        clipped_lower = max(0.0, lower_bound)
        clipped_upper = min(1.0, upper_bound)
        st.metric("Point Prediction", f"{point_pred * 100:.1f}%")
        st.caption(
            f"{CONFIDENCE_LEVEL}% confidence that the admission chance is between "
            f"{clipped_lower * 100:.1f}% and {clipped_upper * 100:.1f}%."
        )
    else:
        st.info("Adjust the sidebar inputs and click **Predict** to generate an admission estimate.")

    st.divider()

    st.subheader("Model Insights")
    insights_tabs = st.tabs(
        [
            "Feature Importance",
            "Residual Distribution",
            "Predicted vs Actual",
            "Coverage",
        ]
    )

    with insights_tabs[0]:
        st.pyplot(plot_feature_importance(diagnostics["feature_names"], diagnostics["feature_importances"]))
    with insights_tabs[1]:
        st.pyplot(plot_residual_histogram(diagnostics["residuals"]))
    with insights_tabs[2]:
        st.pyplot(plot_predicted_vs_actual(diagnostics["test_y"], diagnostics["predictions"]))
    with insights_tabs[3]:
        st.pyplot(
            plot_coverage(
                diagnostics["test_y"],
                diagnostics["intervals"],
                diagnostics["coverage"],
            )
        )

    st.caption(
        "Insights are based on the original holdout set used during training with the MAPIE interval estimator."
    )


if __name__ == "__main__":
    main()
