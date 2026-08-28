import streamlit as st
import pandas as pd
import joblib
import shap
from pathlib import Path
import numpy as np
from ml.frequency_encoder import FrequencyEncoder
# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RazorShield",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

MODEL_DIR = Path("ml/models")
DATA_FILE = Path("ml/data/raw/train_transaction.csv")
TEST_TRANSACTIONS_FILE = (
    MODEL_DIR / "razorshield_test_transactions.pkl"
)
TEST_PREDICTIONS_FILE = (
    MODEL_DIR / "razorshield_test_predictions.pkl"
)
model = joblib.load(
    MODEL_DIR / "razorshield_model.pkl"
)

@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_shap_explainer(model)

preprocessor = joblib.load(
    MODEL_DIR / "razorshield_preprocessor.pkl"
)
selected_feature_names = joblib.load(
    MODEL_DIR / "razorshield_features.pkl"
)
test_transactions = joblib.load(
    TEST_TRANSACTIONS_FILE
)
test_probability = joblib.load(
    TEST_PREDICTIONS_FILE
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)
transactions = load_data()

# ============================================================
# HEADER
# ============================================================

st.title("🛡️ RazorShield")
st.caption("AI Risk Manager — Transaction Fraud Detection")
st.divider()

# ============================================================
# RISK OVERVIEW
# ============================================================

RISK_THRESHOLD = 0.70

total_test_transactions = len(test_probability)

high_risk_count = int(
    (test_probability >= RISK_THRESHOLD).sum()
)

fraud_detected_count = 1543

high_risk_percentage = (
    high_risk_count / total_test_transactions * 100
)

st.subheader("Risk Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📋 Transactions",
        f"{total_test_transactions:,}"
    )

with col2:
    st.metric(
        "🔴 High Risk",
        f"{high_risk_count:,}"
    )

with col3:
    st.metric(
        "🚨 Fraud Detected",
        f"{fraud_detected_count:,}"
    )

with col4:
    st.metric(
        "⚠️ High-Risk Rate",
        f"{high_risk_percentage:.2f}%"
    )

st.caption(
    f"Risk threshold: {RISK_THRESHOLD:.2f}"
)

st.divider()
# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("Model Performance — Held-out Test Set")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("PR-AUC", "0.4939")

with col2:
    st.metric("ROC-AUC", "0.8998")

with col3:
    st.metric("Precision", "44.16%")

with col4:
    st.metric("Recall", "50.05%")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("F1 Score", "46.92%")

with col2:
    st.metric("False Positives", "1,951")

with col3:
    st.metric("False Negatives", "1,540")

with col4:
    st.metric("Total Cost", "9,651")

st.caption(
    "Evaluation performed on unseen test transactions at threshold 0.70."
)

st.divider()

# ============================================================
# TRANSACTION SELECTION
# ============================================================

st.subheader("Transaction Assessment")

transaction_index = st.number_input(
    "Select Transaction",
    min_value=0,
    max_value=len(transactions) - 1,
    value=0,
    step=1
)


transaction = transactions.iloc[[transaction_index]]


# ============================================================
# SHOW TRANSACTION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Transaction ID",
        int(transaction["TransactionID"].iloc[0])
    )

with col2:
    st.metric(
        "Transaction Amount",
        f"₹{transaction['TransactionAmt'].iloc[0]:,.2f}"
    )

with col3:
    st.metric(
        "Product",
        transaction["ProductCD"].iloc[0]
    )


# ============================================================
# ASSESS TRANSACTION
# ============================================================

if st.button(
    "Assess Transaction",
    type="primary"
):

    # Remove target
    X_transaction = transaction.drop(
        columns=["isFraud"],
        errors="ignore"
    )
    X_transaction["amount_log"] = np.log1p(
        X_transaction["TransactionAmt"]
    )
    
    X_transaction["transaction_hour"] = (
            (X_transaction["TransactionDT"] // 3600) % 24
        )
    
    X_transaction["transaction_day"] = (
            X_transaction["TransactionDT"] // 86400
        )

    # Preprocess
    X_processed = preprocessor.transform(
        X_transaction
    )

    # Convert to dataframe so we can select
    # the exact 106 features used by final model
    processed_feature_names = (
        preprocessor.get_feature_names_out()
    )

    X_processed_df = pd.DataFrame(
        X_processed.toarray()
        if hasattr(X_processed, "toarray")
        else X_processed,
        columns=processed_feature_names
    )
    
    X_selected = X_processed_df[
        selected_feature_names
    ]

    # Prediction
    fraud_probability = model.predict_proba(
        X_selected
    )[0, 1]

    risk_score = fraud_probability * 100


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if fraud_probability >= 0.70:

        risk_level = "HIGH"
        action = "REVIEW"

    elif fraud_probability >= 0.30:

        risk_level = "MEDIUM"
        action = "VERIFY"

    else:

        risk_level = "LOW"
        action = "APPROVE"


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.subheader("Risk Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Risk Score",
            f"{risk_score:.2f} / 100"
        )

    with col3:
        st.metric(
            "Decision",
            action
        )


    if risk_level == "HIGH":

        st.error(
            f"🔴 HIGH RISK — {action}"
        )

    elif risk_level == "MEDIUM":

        st.warning(
            f"🟡 MEDIUM RISK — {action}"
        )

    else:

        st.success(
            f"🟢 LOW RISK — {action}"
        )

# ============================================================
# RISK DASHBOARD
# ============================================================

st.divider()

st.subheader("📊 Risk Dashboard")

RISK_THRESHOLD = 0.70

high_risk_mask = test_probability >= RISK_THRESHOLD

high_risk_count = int(high_risk_mask.sum())

total_transactions = len(test_probability)

high_risk_rate = (
    high_risk_count / total_transactions * 100
)


# ============================================================
# RISK SUMMARY
# ============================================================

high_risk_count = int(
    (test_probability >= 0.70).sum()
)

medium_risk_count = int(
    (
        (test_probability >= 0.30)
        & (test_probability < 0.70)
    ).sum()
)

low_risk_count = int(
    (test_probability < 0.30).sum()
)

total_transactions = len(test_probability)

high_risk_rate = (
    high_risk_count / total_transactions * 100
)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🔴 High Risk",
        f"{high_risk_count:,}"
    )

with col2:
    st.metric(
        "🟡 Medium Risk",
        f"{medium_risk_count:,}"
    )

with col3:
    st.metric(
        "🟢 Low Risk",
        f"{low_risk_count:,}"
    )

with col4:
    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader("📈 Risk Distribution")

risk_distribution = pd.DataFrame({
    "Risk Level": [
        "High Risk",
        "Medium Risk",
        "Low Risk"
    ],
    "Transactions": [
        high_risk_count,
        medium_risk_count,
        low_risk_count
    ]
})

risk_distribution["Percentage"] = (
    risk_distribution["Transactions"]
    / total_transactions
    * 100
)

col1, col2 = st.columns(2)

with col1:

    st.markdown("#### Transactions by Risk Level")

    st.bar_chart(
        risk_distribution.set_index("Risk Level")[
            "Transactions"
        ]
    )

with col2:

    st.markdown("#### Risk Percentage")

    percentage_display = risk_distribution.copy()

    percentage_display["Percentage"] = (
        percentage_display["Percentage"]
        .round(2)
        .astype(str)
        + "%"
    )

    st.dataframe(
        percentage_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# HIGH-RISK TRANSACTIONS
# ============================================================

st.subheader("🚨 High-Risk Transactions")

high_risk_transactions = test_transactions.loc[
    high_risk_mask
].copy()

high_risk_transactions["Fraud Probability"] = (
    test_probability[high_risk_mask]
)

high_risk_transactions["Risk Score"] = (
    high_risk_transactions["Fraud Probability"] * 100
)

high_risk_transactions = (
    high_risk_transactions
    .sort_values(
        "Fraud Probability",
        ascending=False
    )
)

display_columns = [
    "TransactionID",
    "TransactionAmt",
    "ProductCD",
    "Fraud Probability",
    "Risk Score"
]

st.dataframe(
    high_risk_transactions[display_columns],
    use_container_width=True,
    hide_index=True
)

# ============================================================
# TRANSACTION EXPOSURE
# ============================================================

st.divider()

st.subheader("💰 Transaction Exposure")

high_risk_amount = high_risk_transactions["TransactionAmt"].sum()

average_high_risk_amount = (
    high_risk_transactions["TransactionAmt"].mean()
    if len(high_risk_transactions) > 0
    else 0
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "💰 High-Risk Transaction Value",
        f"₹{high_risk_amount:,.2f}"
    )

with col2:
    st.metric(
        "💵 Average High-Risk Amount",
        f"₹{average_high_risk_amount:,.2f}"
    )

with col3:
    st.metric(
        "🚨 High-Risk Transactions",
        f"{high_risk_count:,}"
    )

st.caption(
    "Total transaction value associated with transactions "
    "classified as high risk at the 0.70 threshold."
)


# ============================================================
# SELECT HIGH-RISK TRANSACTION
# ============================================================

st.divider()

st.subheader("🔍 Inspect High-Risk Transaction")

if len(high_risk_transactions) > 0:

    selected_transaction_id = st.selectbox(
        "Select a high-risk Transaction ID",
        high_risk_transactions["TransactionID"].tolist()
    )

    selected_transaction = high_risk_transactions[
        high_risk_transactions["TransactionID"]
        == selected_transaction_id
    ].iloc[0]

    # --------------------------------------------------------
    # TRANSACTION DETAILS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Transaction ID",
            int(selected_transaction["TransactionID"])
        )

    with col2:
        st.metric(
            "Amount",
            f"₹{selected_transaction['TransactionAmt']:,.2f}"
        )

    with col3:
        st.metric(
            "Fraud Probability",
            f"{selected_transaction['Fraud Probability'] * 100:.2f}%"
        )

    with col4:
        st.metric(
            "Risk Score",
            f"{selected_transaction['Risk Score']:.2f} / 100"
        )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    st.error(
        "🔴 HIGH RISK — REVIEW TRANSACTION"
    )

else:

    st.success(
        "No high-risk transactions found."
    )

# ============================================================
# SHAP EXPLANATION FOR SELECTED HIGH-RISK TRANSACTION
# ============================================================

if len(high_risk_transactions) > 0:

    st.divider()

    st.subheader(
        "🧠 Why is this transaction high risk?"
    )

    # --------------------------------------------------------
    # Get original transaction
    # --------------------------------------------------------

    selected_original = transactions[
        transactions["TransactionID"]
        == selected_transaction_id
    ].copy()

    if len(selected_original) > 0:

        # Remove target
        X_selected_transaction = selected_original.drop(
            columns=["isFraud"],
            errors="ignore"
        )

        # ----------------------------------------------------
        # Feature engineering
        # ----------------------------------------------------

        X_selected_transaction["amount_log"] = np.log1p(
            X_selected_transaction["TransactionAmt"]
        )

        X_selected_transaction["transaction_hour"] = (
            (
                X_selected_transaction["TransactionDT"]
                // 3600
            ) % 24
        )

        X_selected_transaction["transaction_day"] = (
            X_selected_transaction["TransactionDT"]
            // 86400
        )

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        X_selected_processed = preprocessor.transform(
            X_selected_transaction
        )

        processed_feature_names = (
            preprocessor.get_feature_names_out()
        )

        X_selected_processed_df = pd.DataFrame(
            X_selected_processed.toarray()
            if hasattr(
                X_selected_processed,
                "toarray"
            )
            else X_selected_processed,
            columns=processed_feature_names
        )

        # ----------------------------------------------------
        # Select the exact 106 model features
        # ----------------------------------------------------

        X_selected_model = (
            X_selected_processed_df[
                selected_feature_names
            ]
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        selected_shap_values = explainer.shap_values(
            X_selected_model
        )

        if isinstance(selected_shap_values, list):
            selected_shap_values = selected_shap_values[1]

        selected_shap_values = np.asarray(
            selected_shap_values
        )

        if selected_shap_values.ndim == 2:
            selected_shap_values = (
                selected_shap_values[0]
            )

        # ----------------------------------------------------
        # Create explanation dataframe
        # ----------------------------------------------------

        selected_explanation = pd.DataFrame({
            "Feature": selected_feature_names,
            "SHAP Value": selected_shap_values
        })

        selected_explanation["Absolute Impact"] = (
            selected_explanation["SHAP Value"].abs()
        )

        selected_explanation = (
            selected_explanation
            .sort_values(
                "Absolute Impact",
                ascending=False
            )
        )

        # ----------------------------------------------------
        # Human-readable names
        # ----------------------------------------------------

        def readable_selected_feature(feature):

            if "TransactionAmt" in feature:
                return "Transaction amount"

            if "TransactionDT" in feature:
                return "Transaction time"

            if "card1" in feature or "card2" in feature:
                return "Payment card signal"

            if "card5" in feature:
                return "Payment card verification"

            if "card6" in feature:
                return "Card category"

            if "ProductCD" in feature:
                return "Product category"

            if "__M" in feature or feature.startswith("M"):
                return "Transaction verification signal"

            if "__V" in feature or feature.startswith("V"):
                return "Transaction verification signal"

            if "__C" in feature or feature.startswith("C"):
                return "Transaction count / activity signal"

            if "__D" in feature or feature.startswith("D"):
                return "Transaction timing signal"

            if "addr" in feature:
                return "Address signal"

            if "email" in feature.lower():
                return "Email signal"

            if "dist" in feature.lower():
                return "Transaction distance signal"

            return "Transaction risk signal"

        selected_explanation["Signal"] = (
            selected_explanation["Feature"]
            .apply(readable_selected_feature)
        )

        # ----------------------------------------------------
        # Separate positive and negative SHAP values
        # ----------------------------------------------------

        selected_increasing = (
            selected_explanation[
                selected_explanation["SHAP Value"] > 0
            ]
            .head(5)
            .copy()
        )

        selected_reducing = (
            selected_explanation[
                selected_explanation["SHAP Value"] < 0
            ]
            .sort_values(
                "SHAP Value",
                ascending=True
            )
            .head(5)
            .copy()
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 🔴 Risk-Increasing Signals"
            )

            if len(selected_increasing) > 0:

                display_increasing = (
                    selected_increasing[
                        ["Signal", "SHAP Value"]
                    ].copy()
                )

                display_increasing["SHAP Value"] = (
                    display_increasing["SHAP Value"]
                    .round(4)
                )

                st.dataframe(
                    display_increasing,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No risk-increasing signals found."
                )

        with col2:

            st.markdown(
                "### 🟢 Risk-Reducing Signals"
            )

            if len(selected_reducing) > 0:

                display_reducing = (
                    selected_reducing[
                        ["Signal", "SHAP Value"]
                    ].copy()
                )

                display_reducing["SHAP Value"] = (
                    display_reducing["SHAP Value"]
                    .round(4)
                )

                st.dataframe(
                    display_reducing,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No risk-reducing signals found."
                )

# ============================================================
# NEW TRANSACTION PREDICTION
# ============================================================

st.divider()

st.subheader("🧪 Test a New Transaction")

st.write(
    "Enter transaction details below to estimate fraud risk."
)

with st.form("new_transaction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        new_amount = st.number_input(
            "Transaction Amount (₹)",
            min_value=0.01,
            value=1000.00,
            step=100.00
        )

    with col2:
        product_options = sorted(
            transactions["ProductCD"]
            .dropna()
            .unique()
            .tolist()
        )

        new_product = st.selectbox(
            "Product",
            product_options
        )

    with col3:
        card4_options = sorted(
            transactions["card4"]
            .dropna()
            .unique()
            .tolist()
        )

        new_card4 = st.selectbox(
            "Card Type",
            card4_options
        )

    col1, col2 = st.columns(2)

    with col1:

        card6_options = sorted(
            transactions["card6"]
            .dropna()
            .unique()
            .tolist()
        )

        new_card6 = st.selectbox(
            "Card Category",
            card6_options
        )

    with col2:

        st.write("")
        st.write("")
        predict_button = st.form_submit_button(
            "🔍 Predict Transaction Risk",
            type="primary"
        )


# ============================================================
# PREDICT NEW TRANSACTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Use an existing transaction as a template
    # --------------------------------------------------------

    new_transaction = transactions.iloc[[0]].copy()

    # --------------------------------------------------------
    # Replace user-controlled values
    # --------------------------------------------------------

    new_transaction["TransactionAmt"] = new_amount

    new_transaction["ProductCD"] = new_product

    new_transaction["card4"] = new_card4

    new_transaction["card6"] = new_card6

    # --------------------------------------------------------
    # Remove target
    # --------------------------------------------------------

    X_new = new_transaction.drop(
        columns=["isFraud"],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Feature engineering
    # MUST match training
    # --------------------------------------------------------

    X_new["amount_log"] = np.log1p(
        X_new["TransactionAmt"]
    )

    X_new["transaction_hour"] = (
        (X_new["TransactionDT"] // 3600) % 24
    )

    X_new["transaction_day"] = (
        X_new["TransactionDT"] // 86400
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    X_new_processed = preprocessor.transform(
        X_new
    )

    # --------------------------------------------------------
    # Convert processed data to DataFrame
    # --------------------------------------------------------

    processed_feature_names = (
        preprocessor.get_feature_names_out()
    )

    X_new_processed_df = pd.DataFrame(
        X_new_processed.toarray()
        if hasattr(
            X_new_processed,
            "toarray"
        )
        else X_new_processed,
        columns=processed_feature_names
    )

    # --------------------------------------------------------
    # Select exactly the 106 model features
    # --------------------------------------------------------

    X_new_selected = (
        X_new_processed_df[
            selected_feature_names
        ]
    )

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    new_probability = model.predict_proba(
        X_new_selected
    )[0, 1]

    new_risk_score = new_probability * 100

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if new_probability >= 0.70:

        new_risk_level = "HIGH"
        new_action = "REVIEW"

    elif new_probability >= 0.30:

        new_risk_level = "MEDIUM"
        new_action = "VERIFY"

    else:

        new_risk_level = "LOW"
        new_action = "APPROVE"

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🔎 Prediction Result"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Fraud Probability",
            f"{new_probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Risk Score",
            f"{new_risk_score:.2f} / 100"
        )

    with col3:

        st.metric(
            "Decision",
            new_action
        )

    # --------------------------------------------------------
    # RISK MESSAGE
    # --------------------------------------------------------

    if new_risk_level == "HIGH":

        st.error(
            "🔴 HIGH RISK — Transaction should be reviewed."
        )

    elif new_risk_level == "MEDIUM":

        st.warning(
            "🟡 MEDIUM RISK — Additional verification recommended."
        )

    else:

        st.success(
            "🟢 LOW RISK — Transaction can be approved."
        )

    # ============================================================
    # SHAP RISK EXPLANATION
    # ============================================================

    st.divider()

    st.subheader("🧠 Why did the model make this decision?")

    st.caption(
        "These are the features that contributed most to this prediction."
    )

    # Calculate SHAP values
    shap_values = explainer.shap_values(
        X_new_selected
    )

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)

    # Get SHAP values for this transaction
    if shap_values.ndim == 2:
        transaction_shap = shap_values[0]
    else:
        transaction_shap = shap_values


    # Create explanation dataframe
    explanation_df = pd.DataFrame({
        "Feature": selected_feature_names,
        "SHAP Value": transaction_shap
    })

    # ========================================================
    # HUMAN-READABLE FEATURE NAMES
    # ========================================================

    def readable_feature(feature):

        if "TransactionAmt" in feature:
            return "Transaction amount"

        if "TransactionDT" in feature:
            return "Transaction time"

        if "card1" in feature or "card2" in feature:
            return "Payment card signal"

        if "card5" in feature:
            return "Payment card verification"

        if "card6" in feature:
            return "Card category"

        if "ProductCD" in feature:
            return "Product category"

        if "__M" in feature or feature.startswith("M"):
            return "Transaction verification signal"

        if "__V" in feature or feature.startswith("V"):
            return "Transaction verification signal"

        if "__C" in feature or feature.startswith("C"):
            return "Transaction count / activity signal"

        if "__D" in feature or feature.startswith("D"):
            return "Transaction timing signal"

        if "addr" in feature:
            return "Address signal"

        if "email" in feature.lower():
            return "Email signal"

        if "dist" in feature.lower():
            return "Transaction distance signal"

        return "Transaction risk signal"


    explanation_df["Signal"] = (
        explanation_df["Feature"]
        .apply(readable_feature)
    )

    # Sort by absolute importance
    explanation_df["Absolute Impact"] = (
        explanation_df["SHAP Value"].abs()
    )

    explanation_df = explanation_df.sort_values(
        "Absolute Impact",
        ascending=False
    )


    # ============================================================
    # RISK-INCREASING SIGNALS
    # ============================================================

    risk_increasing = (
        explanation_df[
            explanation_df["SHAP Value"] > 0
        ]
        .head(5)
        .copy()
    )


    # ============================================================
    # RISK-REDUCING SIGNALS
    # ============================================================

    risk_reducing = (
        explanation_df[
            explanation_df["SHAP Value"] < 0
        ]
        .head(5)
        .copy()
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown("### 🔴 Risk-Increasing Signals")

        if len(risk_increasing) > 0:

            display_increasing = risk_increasing[
                ["Signal", "SHAP Value"]
            ].copy()

            display_increasing["SHAP Value"] = (
                display_increasing["SHAP Value"]
                .round(4)
            )

            st.dataframe(
                display_increasing,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No strong risk-increasing signals."
            )


    with col2:

        st.markdown("### 🟢 Risk-Reducing Signals")

        if len(risk_reducing) > 0:

            display_reducing = risk_reducing[
                ["Signal", "SHAP Value"]
            ].copy()

            display_reducing["SHAP Value"] = (
                display_reducing["SHAP Value"]
                .round(4)
            )

            st.dataframe(
                display_reducing,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No strong risk-reducing signals."
            )
