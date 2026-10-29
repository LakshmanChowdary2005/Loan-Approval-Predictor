"""Loan approval model review dashboard.

Run with: streamlit run streamlit_app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import CATEGORICAL_FEATURES, MODEL_FILES, NUMERICAL_FEATURES
from src.prediction_service import get_binary_class_probabilities
from src.validators import validate_applicant_data

st.set_page_config(
    page_title="CreditPulse | Loan review",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(BASE_DIR, "dataset", "loan_data.csv")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


@st.cache_resource
def load_models():
    loaded_models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)
    return loaded_models


@st.cache_data(ttl="15m", max_entries=1)
def load_metadata():
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(ttl="15m", max_entries=1)
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


def build_input_frame(profile):
    row = dict(profile)
    row["Dependents"] = str(row["Dependents"]).replace("+", "")
    return pd.DataFrame([row])


def probability_label(value):
    return f"{value:.1f}%" if value is not None else "Not available"


models = load_models()
metadata = load_metadata()
df = load_dataset()

if not models:
    st.error("No trained model files were found in the model folder. Train the models before launching this dashboard.")
    st.stop()

with st.sidebar:
    st.title("CreditPulse")
    st.caption("Loan application review")
    selected_model_name = st.selectbox("Scoring model", list(models))
    st.divider()
    st.caption(
        "Predictions are estimates from historical data, not credit decisions, "
        "financial advice, or a guarantee of future outcomes."
    )

model_metrics = metadata.get("benchmark_metrics", {}).get(selected_model_name, {})
dataset_summary = metadata.get("dataset_summary", {})
cv_accuracy = model_metrics.get("cv_accuracy_mean")
cv_std = model_metrics.get("cv_accuracy_std")
test_accuracy = model_metrics.get("test_accuracy")
approval_rate = dataset_summary.get("approval_rate")

st.title("Loan application review", icon=":material/assured_workload:")
st.write(
    "Explore a model estimate, compare it with other available classifiers, "
    "and review the assumptions behind the result."
)
st.warning(
    "This academic model was trained on a small historical dataset. Its output is not "
    "a lending decision and should not be used as the sole basis for one.",
    icon=":material/info:",
)

metric_columns = st.columns(4)
metric_columns[0].metric("Training records", f"{dataset_summary.get('total_records', len(df)):,}")
metric_columns[1].metric(
    "5-fold CV accuracy",
    probability_label(cv_accuracy),
    delta=f"± {cv_std:.2f} percentage points" if cv_std is not None else None,
)
metric_columns[2].metric(
    "Held-out accuracy",
    probability_label(test_accuracy),
    help="Measured on the stratified test split; separate from cross-validation.",
)
metric_columns[3].metric(
    "Historical approval share",
    probability_label(approval_rate),
    help="Share of approved applications in the source dataset.",
)
st.caption(
    "Reported scores describe historical data only. The held-out test set contains "
    "123 applications and is not a guarantee of real-world performance."
)

st.header("Applicant profile", divider="gray")
with st.form("applicant_profile"):
    col_identity, col_income, col_loan = st.columns(3)

    with col_identity:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital status", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self-employed", ["No", "Yes"])

    with col_income:
        applicant_income = st.number_input(
            "Applicant monthly income (dataset units)",
            min_value=0.0,
            value=5849.0,
            step=100.0,
        )
        coapplicant_income = st.number_input(
            "Co-applicant monthly income (dataset units)",
            min_value=0.0,
            value=1500.0,
            step=100.0,
        )
        property_area = st.selectbox("Property area", ["Semiurban", "Urban", "Rural"])
        credit_history = st.selectbox(
            "Credit history",
            [1.0, 0.0],
            format_func=lambda value: (
                "Meets historical guidelines" if value == 1.0 else "Does not meet historical guidelines"
            ),
        )

    with col_loan:
        loan_amount = st.number_input(
            "Requested loan (thousands of dataset units)",
            min_value=1.0,
            value=140.0,
            step=5.0,
        )
        loan_term = st.selectbox("Repayment term (months)", [360, 240, 180, 120, 84, 60, 36, 12])
        annual_rate = st.number_input(
            "Illustrative annual interest rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=8.5,
            step=0.25,
        )
        st.caption("Income and loan amounts use the source dataset's units.")

    submitted = st.form_submit_button(
        "Estimate outcome",
        type="primary",
        icon=":material/query_stats:",
        width="stretch",
    )

if submitted:
    st.session_state.pop("last_prediction", None)
    st.session_state.pop("last_profile", None)
    st.session_state.pop("last_model", None)
    applicant_data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }
    is_valid, errors, cleaned_profile = validate_applicant_data(applicant_data)
    if not is_valid:
        st.error("Please correct the applicant profile before estimating.")
        for error in errors:
            st.write(f"- {error}")
    else:
        input_frame = build_input_frame(cleaned_profile)
        model = models[selected_model_name]
        predicted_class = int(model.predict(input_frame)[0])
        class_probabilities = get_binary_class_probabilities(model, input_frame)
        st.session_state["last_profile"] = cleaned_profile
        st.session_state["last_model"] = selected_model_name
        st.session_state["last_prediction"] = {
            "class": predicted_class,
            "probabilities": class_probabilities,
            "annual_rate": annual_rate,
        }

if "last_prediction" in st.session_state:
    last_result = st.session_state["last_prediction"]
    last_profile = st.session_state["last_profile"]
    approval_probability = last_result["probabilities"][1]
    predicted_approval = last_result["class"] == 1
    monthly_rate = last_result["annual_rate"] / 1200
    payment_count = int(last_profile["Loan_Amount_Term"])
    principal = float(last_profile["LoanAmount"]) * 1000
    emi = (
        principal * monthly_rate * (1 + monthly_rate) ** payment_count
        / ((1 + monthly_rate) ** payment_count - 1)
        if monthly_rate > 0
        else principal / payment_count
    )
    household_income = (
        float(last_profile["ApplicantIncome"]) + float(last_profile["CoapplicantIncome"])
    )
    dti = emi / household_income * 100 if household_income > 0 else None

    st.header("Estimate", divider="gray")
    if predicted_approval:
        st.success("The selected model predicts the approved class.", icon=":material/check_circle:")
    else:
        st.error("The selected model predicts the rejected class.", icon=":material/cancel:")
    st.caption(
        f"Model: {st.session_state['last_model']} · "
        "Probabilities are model estimates, not calibrated lending odds."
    )

    result_columns = st.columns(4)
    result_columns[0].metric("Approval probability", f"{approval_probability * 100:.1f}%")
    result_columns[1].metric("Rejection probability", f"{last_result['probabilities'][0] * 100:.1f}%")
    result_columns[2].metric("Illustrative monthly payment", f"{emi:,.2f}")
    result_columns[3].metric("Payment / household income", f"{dti:.1f}%" if dti is not None else "Not defined")
    st.progress(float(approval_probability), text="Estimated approval probability")
    st.caption(
        "Payment estimate assumes a fixed interest rate entered above, monthly compounding, "
        "and the dataset's loan amount scale. It is an illustration, not a quoted payment."
    )

tab_predict, tab_simulator, tab_consensus, tab_batch, tab_benchmarks, tab_data = st.tabs(
    ["Prediction", "What-if", "Model consensus", "Batch scoring", "Model performance", "Dataset"]
)

with tab_predict:
    st.subheader("What the model can and cannot tell you")
    st.write(
        "The estimate reflects patterns in the training data. It does not explain an "
        "individual lender's policy, verify applicant information, or establish eligibility."
    )
    st.caption("Use the model as a learning and review aid only.")

with tab_simulator:
    st.subheader("What-if analysis")
    st.caption("Change one input to see how this model's estimated probability responds.")
    if "last_profile" not in st.session_state:
        st.info("Enter an applicant profile and select **Estimate outcome** first.")
    else:
        profile = st.session_state["last_profile"]
        simulation_loan = st.number_input(
            "Alternative loan amount (thousands of dataset units)",
            min_value=1.0,
            value=float(profile["LoanAmount"]),
            step=5.0,
            key="simulation_loan",
        )
        simulation_profile = dict(profile, LoanAmount=simulation_loan)
        simulation_frame = build_input_frame(simulation_profile)
        simulation_model = models[selected_model_name]
        simulation_probabilities = get_binary_class_probabilities(simulation_model, simulation_frame)
        st.metric("Estimated approval probability", f"{simulation_probabilities[1] * 100:.1f}%")

        maximum_loan = max(450.0, float(profile["LoanAmount"]) * 2)
        loan_values = np.linspace(1.0, maximum_loan, 20)
        curve = []
        for amount in loan_values:
            scenario = dict(profile, LoanAmount=float(amount))
            scenario_probabilities = get_binary_class_probabilities(
                simulation_model,
                build_input_frame(scenario),
            )
            curve.append(
                {
                    "Loan amount (thousands of dataset units)": float(amount),
                    "Approval probability (%)": scenario_probabilities[1] * 100,
                }
            )
        st.line_chart(
            pd.DataFrame(curve).set_index("Loan amount (thousands of dataset units)"),
            y="Approval probability (%)",
            x_label="Loan amount (thousands of dataset units)",
            y_label="Approval probability (%)",
        )

with tab_consensus:
    st.subheader("Compare available models")
    st.caption("Every model evaluates the same applicant profile saved from the prediction form.")
    if "last_profile" not in st.session_state:
        st.info("Enter an applicant profile and select **Estimate outcome** first.")
    else:
        profile_frame = build_input_frame(st.session_state["last_profile"])
        votes = []
        for name, model in models.items():
            predicted_class = int(model.predict(profile_frame)[0])
            probabilities = get_binary_class_probabilities(model, profile_frame)
            votes.append(
                {
                    "Model": name,
                    "Predicted class": "Approved" if predicted_class == 1 else "Rejected",
                    "Approval probability (%)": round(probabilities[1] * 100, 2),
                    "Rejection probability (%)": round(probabilities[0] * 100, 2),
                }
            )
        votes_df = pd.DataFrame(votes)
        st.dataframe(votes_df, hide_index=True, width="stretch")
        approvals = sum(row["Predicted class"] == "Approved" for row in votes)
        st.metric("Model agreement", f"{approvals} of {len(votes)} predict the approved class")
        st.caption("A majority vote is not a lending decision and does not resolve disagreement between models.")

with tab_batch:
    st.subheader("Batch scoring")
    st.write("Upload a CSV with all applicant features. A `Loan_ID` column is optional.")
    required_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    uploaded_file = st.file_uploader("Applicant CSV", type=["csv"], key="batch_upload")
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not read the uploaded CSV: {error}")
        else:
            missing_columns = [column for column in required_columns if column not in batch_df.columns]
            if missing_columns:
                st.error(f"Missing required feature columns: {', '.join(missing_columns)}")
            elif st.button("Score batch", type="primary", icon=":material/upload_file:"):
                valid_profiles = []
                invalid_rows = []
                for row_index, row in batch_df.iterrows():
                    row_data = row.to_dict()
                    if "Dependents" in row_data and isinstance(row_data["Dependents"], (int, float)):
                        if float(row_data["Dependents"]).is_integer():
                            row_data["Dependents"] = str(int(row_data["Dependents"]))
                    is_valid, errors, cleaned = validate_applicant_data(row_data)
                    if is_valid:
                        valid_profiles.append((row_index, row_data.get("Loan_ID", row_index + 1), cleaned))
                    else:
                        invalid_rows.append(
                            {
                                "CSV row": row_index + 2,
                                "Loan_ID": row_data.get("Loan_ID", ""),
                                "Validation errors": "; ".join(errors),
                            }
                        )

                if valid_profiles:
                    model = models[selected_model_name]
                    scored_rows = []
                    for _, loan_id, applicant in valid_profiles:
                        frame = build_input_frame(applicant)
                        prediction = int(model.predict(frame)[0])
                        probabilities = get_binary_class_probabilities(model, frame)
                        scored_rows.append(
                            {
                                "Loan_ID": loan_id,
                                "Predicted class": "Approved" if prediction == 1 else "Rejected",
                                "Approval probability (%)": round(probabilities[1] * 100, 2),
                                "Rejection probability (%)": round(probabilities[0] * 100, 2),
                            }
                        )
                    scored_df = pd.DataFrame(scored_rows)
                    st.dataframe(scored_df, hide_index=True, width="stretch")
                    st.download_button(
                        "Download scored CSV",
                        data=scored_df.to_csv(index=False).encode("utf-8"),
                        file_name="loan_predictions.csv",
                        mime="text/csv",
                        icon=":material/download:",
                    )
                else:
                    st.warning("No valid applicant rows were found; no predictions were generated.")

                if invalid_rows:
                    st.warning(f"{len(invalid_rows)} row(s) were skipped because their values were invalid.")
                    st.dataframe(pd.DataFrame(invalid_rows), hide_index=True, width="stretch")

with tab_benchmarks:
    st.subheader("Model performance")
    st.caption(
        "Cross-validation summarizes five training folds. Held-out scores come from one "
        "stratified test split and can vary with the split."
    )
    benchmark_metrics = metadata.get("benchmark_metrics", {})
    if benchmark_metrics:
        benchmark_rows = []
        for name, metrics in benchmark_metrics.items():
            benchmark_rows.append(
                {
                    "Model": name,
                    "CV accuracy (%)": metrics.get("cv_accuracy_mean"),
                    "CV accuracy SD (pp)": metrics.get("cv_accuracy_std"),
                    "CV balanced accuracy (%)": metrics.get("cv_balanced_accuracy_mean"),
                    "CV macro F1 (%)": metrics.get("cv_macro_f1_mean"),
                    "Held-out accuracy (%)": metrics.get("test_accuracy"),
                    "Held-out balanced accuracy (%)": metrics.get("test_balanced_accuracy"),
                    "Held-out macro F1 (%)": metrics.get("macro_f1"),
                    "Held-out ROC-AUC (%)": metrics.get("roc_auc"),
                }
            )
        benchmark_df = pd.DataFrame(benchmark_rows).sort_values(
            by="CV accuracy (%)",
            ascending=False,
            na_position="last",
        )
        st.dataframe(benchmark_df, hide_index=True, width="stretch")
        if "cv_balanced_accuracy_mean" not in next(iter(benchmark_metrics.values())):
            st.info("Retrain models to add cross-validated balanced accuracy and macro F1 metrics.")
    else:
        st.info("No benchmark metadata is available. Train the models to generate evaluation metrics.")

with tab_data:
    st.subheader("Training dataset")
    if df.empty:
        st.info("The source dataset was not found.")
    else:
        st.caption(f"{len(df):,} source records · {len(df.columns)} columns")
        if "Loan_Status" in df.columns:
            status_counts = df["Loan_Status"].value_counts(dropna=False).rename_axis("Loan status").to_frame("Applications")
            st.bar_chart(status_counts, y="Applications", x_label="Loan status", y_label="Applications")
        eda_dir = os.path.join(BASE_DIR, "static", "images", "eda")
        image_names = [
            "loan_status_distribution.png",
            "credit_history_vs_approval.png",
            "property_area_vs_approval.png",
            "education_vs_approval.png",
        ]
        existing_images = [
            (name, os.path.join(eda_dir, name))
            for name in image_names
            if os.path.exists(os.path.join(eda_dir, name))
        ]
        if existing_images:
            with st.expander("Exploratory charts"):
                for name, image_path in existing_images:
                    st.image(image_path, caption=name.replace("_", " ").removesuffix(".png"), alt=name)
        st.dataframe(df, hide_index=True, height=440, width="stretch")
