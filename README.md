# 🛡️ RazorShield

## AI-Powered Transaction Fraud Detection & Risk Management

RazorShield is a machine learning-based fraud detection application that analyzes financial transactions and predicts their fraud risk.

The application uses a trained machine learning model together with a Streamlit dashboard to provide fraud probability, risk score, risk level, and recommended action.

---

## 🚀 Features

- Transaction fraud prediction
- Manual transaction entry
- Transaction amount analysis
- Fraud probability prediction
- Risk score calculation
- Low, Medium, and High risk classification
- Interactive risk dashboard
- High-risk transaction detection
- Risk distribution visualization
- Model performance dashboard
- SHAP-based model explainability
- Machine learning preprocessing pipeline

---

## 🧠 Machine Learning

RazorShield uses supervised machine learning for binary fraud classification.

Target variable:

isFraud

- 0 = Legitimate transaction
- 1 = Fraudulent transaction

The project includes feature engineering, preprocessing, categorical encoding, frequency encoding, and feature selection.

---

## 📊 Model Performance

The model was evaluated on an unseen test dataset.

| Metric | Result |
|---|---:|
| PR-AUC | 0.4939 |
| ROC-AUC | 0.8998 |
| Precision | 44.16% |
| Recall | 50.05% |
| F1 Score | 46.92% |
| False Positives | 1,951 |
| False Negatives | 1,540 |
| Total Cost | 9,651 |

Risk threshold:

0.70

---

## 🚦 Risk Classification

| Fraud Probability | Risk Level | Action |
|---|---|---|
| Below 0.30 | 🟢 LOW | APPROVE |
| 0.30 to below 0.70 | 🟡 MEDIUM | VERIFY |
| 0.70 or above | 🔴 HIGH | REVIEW |

For example, if the model predicts a fraud probability of 82%:

Risk Score: 82 / 100

Risk Level: HIGH

Decision: REVIEW

---

## 🖥️ Application

RazorShield provides an interactive Streamlit dashboard.

### Risk Overview

Displays:

- Total transactions
- High-risk transactions
- Fraud detected
- Risk threshold

### Transaction Assessment

Users can select an existing transaction and assess its fraud risk.

The application displays:

- Transaction ID
- Transaction amount
- Product
- Fraud probability
- Risk score
- Decision

### Manual Transaction Assessment

Users can enter transaction information manually and receive a fraud-risk prediction.

The prediction includes:

- Fraud Probability
- Risk Score
- Risk Level
- Decision

### Risk Dashboard

The dashboard displays:

- High-risk transactions
- Medium-risk transactions
- Low-risk transactions
- Total transactions
- Risk distribution
- High-risk transaction details

---

## 🔬 Explainability

RazorShield supports SHAP-based model explainability.

SHAP helps identify which features contribute to increasing or decreasing the predicted fraud risk.

---

## 📁 Project Structure

razorshield-ai/

├── app.py
├── frequency_encoder.py
├── .gitignore
├── README.md
│
└── ml/
    ├── explore.ipynb
    ├── preprocess.ipynb
    ├── split_data.ipynb
    ├── frequency_encoder.py
    │
    └── models/
        ├── razorshield_model.pkl
        ├── razorshield_preprocessor.pkl
        └── razorshield_features.pkl

Large test transaction files are excluded from GitHub because they exceed GitHub's file-size limit.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- SHAP
- Git
- GitHub

---

## ⚙️ Installation

Clone the repository:

git clone https://github.com/Prajwalssajp/razorshield-ai.git

Open the project folder:

cd razorshield-ai

Create a virtual environment:

python -m venv venv

Activate the environment on Windows:

venv\Scripts\activate

Install the required packages:

pip install streamlit pandas numpy scikit-learn joblib shap

---

## ▶️ Run the Application

Run:

streamlit run app.py

The RazorShield application will open in your web browser.

---

## 🔄 Machine Learning Workflow

Raw Transaction Data

↓

Data Exploration

↓

Data Cleaning

↓

Train / Validation / Test Split

↓

Feature Engineering

↓

Categorical Encoding

↓

Frequency Encoding

↓

Preprocessing

↓

Model Training

↓

Model Evaluation

↓

Feature Selection

↓

Saved Model

↓

Streamlit Application

↓

Transaction Input

↓

Fraud Probability

↓

Risk Score

↓

Risk Classification

↓

Decision

---

## 🤖 Model Artifacts

The trained model uses these files:

ml/models/razorshield_model.pkl

ml/models/razorshield_preprocessor.pkl

ml/models/razorshield_features.pkl

These saved artifacts allow the application to make predictions without retraining the model every time.

---

## 🎯 Project Objective

The objective of RazorShield is to demonstrate how machine learning can be used for transaction fraud detection and risk management.

Instead of only returning a fraud or non-fraud result, RazorShield converts the model prediction into an understandable risk score and recommended action.

---

## 📈 Future Improvements

- Real-time transaction processing
- Database integration
- User authentication
- Real-time fraud alerts
- Automated model retraining
- Model monitoring
- Cloud deployment
- Prediction API
- Transaction history
- Advanced SHAP visualizations
- Model drift detection

---

## ⚠️ Disclaimer

RazorShield is an educational and machine-learning demonstration project.

The predictions generated by this application should not be treated as a definitive determination of financial fraud.

Production fraud detection systems require additional validation, security controls, business rules, monitoring, and human review.

---

## 👨‍💻 Author

Prajwal SS

GitHub:

https://github.com/Prajwalssajp

---

## 📄 License

This project is intended for educational and demonstration purposes.
