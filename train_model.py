import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================================
# LOAD DATASET
# =========================================

print("Loading dataset...")

df = pd.read_csv("dataset/loan.csv")

print(df.head())


# =========================================
# HANDLE MISSING VALUES
# =========================================

for column in df.columns:

    try:
        # Try numeric median filling
        df[column] = df[column].fillna(df[column].median())

    except:
        # If column is text
        df[column] = df[column].fillna(df[column].mode()[0])
# =========================================
# DROP UNNECESSARY COLUMN
# =========================================

if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)


# =========================================
# ENCODE CATEGORICAL DATA
# =========================================

label_encoders = {}

for column in df.columns:

    if df[column].dtype == object:

        le = LabelEncoder()

        df[column] = le.fit_transform(df[column])

        label_encoders[column] = le

 # =========================================
# FEATURE ENGINEERING
# =========================================

# Total Income
df["TotalIncome"] = (
    df["ApplicantIncome"] +
    df["CoapplicantIncome"]
)

# Loan vs Income Ratio
df["Loan_Income_Ratio"] = (
    df["LoanAmount"] /
    df["TotalIncome"]
)       


# =========================================
# SPLIT FEATURES AND TARGET
# =========================================

X = df.drop("Loan_Status", axis=1)

y = df["Loan_Status"]


# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================
# TRAIN MODEL
# =========================================

print("Training model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# =========================================
# MODEL EVALUATION
# =========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")


# =========================================
# SAVE MODEL
# =========================================

joblib.dump(model, "loan_model.pkl")

print("Model saved successfully")