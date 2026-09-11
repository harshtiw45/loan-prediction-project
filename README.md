# 💰 Loan Approval Prediction System

An AI-powered Loan Approval Prediction System that uses Machine Learning to predict whether a loan application is likely to be Approved or Rejected based on an applicant's personal, financial, and credit-related information.

The project uses a Random Forest Classifier trained on historical loan application data. The trained model is integrated with an interactive Streamlit web application, allowing users to enter applicant details and receive an instant prediction.

## 🚀 Features

- 🤖 Machine Learning-based loan prediction
- 🌳 Random Forest Classification algorithm
- 📊 Data preprocessing and feature engineering
- 💰 Applicant and co-applicant income analysis
- 🏦 Credit history-based prediction
- ⚡ Instant loan approval/rejection prediction
- 📈 Prediction confidence score
- 🖥️ Interactive Streamlit web application
- 💾 Trained model saved using Joblib

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Joblib
- Streamlit

## 🔍 How It Works

1. Historical loan application data is loaded from the dataset.
2. Missing values are handled during preprocessing.
3. Categorical variables are converted into numerical values.
4. Additional features such as Total Income and Loan-Income Ratio are created.
5. The dataset is divided into training and testing sets.
6. A Random Forest Classifier is trained on the training data.
7. The model is evaluated using accuracy.
8. The trained model is saved as `loan_model.pkl`.
9. The Streamlit application loads the trained model.
10. Users enter applicant details and receive a real-time loan prediction.

## 📁 Project Structure

```text
loan-prediction-project/
│
├── dataset/
│   └── loan.csv
│
├── app.py
├── train_model.py
├── loan_model.pkl
└── README.md
