 """
FUTURE_DS_01 - Customer Churn Analysis
Intern: Gonela Tharun
CIN: FIT/AUG26/DS22911
Internship: Future Interns - Data Science & Analytics
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Create required folders
os.makedirs("data", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ---------------------------------------------------------
# 1. Create a synthetic customer-churn dataset
# ---------------------------------------------------------

np.random.seed(42)

num_customers = 5000

customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, num_customers + 1)]

tenure = np.random.randint(1, 73, num_customers)

monthly_charges = np.round(np.random.uniform(20, 120, num_customers), 2)

total_charges = np.round(monthly_charges * tenure, 2)

contract_types = np.random.choice(
    ["Month-to-month", "One year", "Two year"],
    num_customers,
    p=[0.55, 0.25, 0.20]
)

payment_methods = np.random.choice(
    ["Electronic cheque", "Mailed cheque", "Bank transfer", "Credit card"],
    num_customers,
    p=[0.35, 0.20, 0.25, 0.20]
)

internet_service = np.random.choice(
    ["Fibre optic", "DSL", "No internet"],
    num_customers,
    p=[0.45, 0.40, 0.15]
)

tech_support = np.random.choice(
    ["Yes", "No"],
    num_customers,
    p=[0.35, 0.65]
)

online_security = np.random.choice(
    ["Yes", "No"],
    num_customers,
    p=[0.40, 0.60]
)

churn_probability = []

for i in range(num_customers):
    probability = 0.10

    if contract_types[i] == "Month-to-month":
        probability += 0.30
    elif contract_types[i] == "One year":
        probability += 0.10

    if tenure[i] < 12:
        probability += 0.20

    if monthly_charges[i] > 80:
        probability += 0.15

    if payment_methods[i] == "Electronic cheque":
        probability += 0.10

    if tech_support[i] == "No":
        probability += 0.08

    if online_security[i] == "No":
        probability += 0.05

    churn_probability.append(min(probability, 0.95))

churn = np.random.binomial(1, churn_probability)

df = pd.DataFrame({
    "customerID": customer_ids,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract_types,
    "PaymentMethod": payment_methods,
    "InternetService": internet_service,
    "TechSupport": tech_support,
    "OnlineSecurity": online_security,
    "Churn": churn
})

df.to_csv("data/customer_churn.csv", index=False)

print("Dataset created successfully.")
print("Dataset shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# 2. Data cleaning and preparation
# ---------------------------------------------------------

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

df.drop("customerID", axis=1, inplace=True)

df["Churn"] = df["Churn"].map({1: "Yes", 0: "No"})

# ---------------------------------------------------------
# 3. Exploratory Data Analysis
# ---------------------------------------------------------

plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("images/churn_distribution.png")
plt.close()

plt.figure(figsize=(8, 4))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("images/churn_by_contract.png")
plt.close()

plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30)
plt.title("Churn by Customer Tenure")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("images/churn_by_tenure.png")
plt.close()

plt.figure(figsize=(8, 4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("images/monthly_charges_by_churn.png")
plt.close()

plt.figure(figsize=(9, 4))
sns.countplot(x="PaymentMethod", hue="Churn", data=df)
plt.title("Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("images/churn_by_payment_method.png")
plt.close()

# ---------------------------------------------------------
# 4. Machine-learning model
# ---------------------------------------------------------

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

categorical_columns = df.select_dtypes(include="object").columns

df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png")
plt.close()

# ---------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

feature_importance["Absolute Importance"] = feature_importance["Coefficient"].abs()
feature_importance = feature_importance.sort_values(
    by="Absolute Importance",
    ascending=False
)

print("\nTop 15 Important Features")
print(feature_importance.head(15))

feature_importance.head(15).to_csv("images/top_features.csv", index=False)

plt.figure(figsize=(8, 6))
top_features = feature_importance.head(15)
sns.barplot(x="Coefficient", y="Feature", data=top_features)
plt.title("Top Features Influencing Customer Churn")
plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("images/top_features.png")
plt.close()

print("\nAnalysis complete.")
print("Dataset saved as: data/customer_churn.csv")
print("Charts saved in the images folder.")
