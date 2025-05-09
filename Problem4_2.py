from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("EX4/players_with_values.csv")
# Rename 'Value' column to 'market_value' to align with the code
data.rename(columns={"Value": "market_value"}, inplace=True)

# Inspect data type of market_value column
print("Data types in market_value column:")
print(data["market_value"].apply(type).value_counts())
print("First few values of market_value column:")
print(data["market_value"].head(10))

# Function to convert market value
def parse_value(value):
    # Handle missing or invalid values
    if pd.isna(value) or value == "N/A":
        return np.nan
    # Return directly if already a number
    if isinstance(value, (int, float)):
        return float(value)
    # Process string values with currency symbols
    if isinstance(value, str):
        value = value.replace("€", "").replace("£", "").strip()
        try:
            if "M" in value:
                return float(value.replace("M", "")) * 1e6
            elif "K" in value:
                return float(value.replace("K", "")) * 1e3
            return float(value)  # Handle plain numeric strings
        except ValueError:
            return np.nan
    return np.nan

# Apply value parsing function
data["market_value"] = data["market_value"].apply(parse_value)
# Remove rows with missing market_value
data.dropna(subset=["market_value"], inplace=True)

# Identify categorical and numerical columns
cat_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
num_columns = data.select_dtypes(include=np.number).columns.tolist()

# Exclude unnecessary columns
exclude_cols = ["Player", "Nation", "Team", "Position", "market_value"]
# Select numerical columns not in exclude_cols
selected_features = [col for col in num_columns if col not in exclude_cols and col in data.columns]

# Encode categorical columns
label_encoders = {}
for col in ["Nation", "Team", "Position"]:
    if col in data.columns:
        encoder = LabelEncoder()
        data[col] = data[col].astype(str)
        data[col] = encoder.fit_transform(data[col])
        label_encoders[col] = encoder
        if col not in selected_features:
            selected_features.append(col)

# Filter final features (only numeric or encoded columns)
final_attributes = []
for col in selected_features:
    if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
        final_attributes.append(col)
    elif col in label_encoders.keys():
        final_attributes.append(col)

# Prepare features and target
features = data[final_attributes]
target = data["market_value"]

# Handle missing values in features
features = features.fillna(features.median())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=3,
)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate
predictions = rf_model.predict(X_test)
mse_value = mean_squared_error(y_test, predictions)
r2_value = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse_value:.2f}")
print(f"R² Score: {r2_value:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.scatter(y_test / 1e6, predictions / 1e6, alpha=0.6, edgecolors="k", s=50)
plt.plot(
    [min(y_test / 1e6), max(y_test / 1e6)],
    [min(y_test / 1e6), max(y_test / 1e6)],
    "--",
    color="red",
    lw=2,
)
plt.title(
    f"Actual vs Predicted Values (Random Forest)\nR² Score: {r2_value:.4f} | MSE: {mse_value / 1e12:.2f} ($M^2€^2$)"
)
plt.xlabel("Actual Value (€ Millions)")
plt.ylabel("Predicted Value (€ Millions)")
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300, format="png", bbox_inches="tight")
plt.show()

# Feature importance
feature_weights = rf_model.feature_importances_
importance_df = pd.DataFrame({"Attribute": final_attributes, "Weight": feature_weights})
importance_df = importance_df.sort_values(by="Weight", ascending=False)
print("\nFeature Importance:")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x="Weight", y="Attribute", data=importance_df)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, format="png", bbox_inches="tight")
plt.show()