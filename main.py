# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Ensure reproducibility
np.random.seed(42)

# Load dataset
DATA_PATH = "data/AmesHousing.csv"  # Update to actual dataset path
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")

data = pd.read_csv(DATA_PATH)

# Initial Exploration
print("Dataset Shape:", data.shape)
print("Columns:", data.columns)

# Data Cleaning and Preprocessing
print("\nMissing Values (Before):\n", data.isnull().sum().sort_values(ascending=False))
# Separate numeric and non-numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns

# Impute missing values
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())  # Median for numeric columns
data[non_numeric_cols] = data[non_numeric_cols].fillna("Unknown")  # Fill categorical columns with 'Unknown'

print("\nMissing Values (After):\n", data.isnull().sum().sort_values(ascending=False))

# Select only numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Compute correlation with SalePrice and drop irrelevant features
correlation_with_target = numeric_data.corr()['SalePrice'].sort_values(ascending=False)
print("Correlation with SalePrice:\n", correlation_with_target)

# Select only features with significant correlation (e.g., > 0.3 or < -0.3)
significant_features = correlation_with_target[abs(correlation_with_target) > 0.3].index.tolist()
if 'SalePrice' in significant_features:
    significant_features.remove('SalePrice')  # Exclude target variable

# Prepare X and y
X = numeric_data[significant_features]
y = numeric_data['SalePrice']

# Ensure X and y are not empty
if X.empty or y.empty:
    raise ValueError("Feature set (X) or target (y) is empty after preprocessing. Check your data and feature selection.")

# Debugging: Print shapes and sample data
print("Selected Features for Training:\n", significant_features)
print("Feature set shape:", X.shape)
print("Target shape:", y.shape)

# Remove rows with missing values
X = X.dropna()
y = y[X.index]  # Align target with cleaned features

# Verify alignment
assert len(X) == len(y), "Mismatch between feature set (X) and target (y)."

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models and Training
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)

    # Prediction and Evaluation
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Store Results
    results[name] = {
        "RMSE (CV)": cv_rmse.mean(),
        "RMSE (Test)": rmse,
        "R² (Test)": r2
    }

    print(f"{name} - Test RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Results Summary
results_df = pd.DataFrame(results).T
print("\nModel Performance:\n", results_df)

# Visualization
os.makedirs("results", exist_ok=True)
results_df.plot(kind='bar', figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("results/model_performance.png")
plt.show()

# Save the Trained Models
os.makedirs("models", exist_ok=True)
for name, model in models.items():
    joblib.dump(model, f"models/{name.replace(' ', '_').lower()}_model.pkl")
print("Models saved in 'models/' directory.")
