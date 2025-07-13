# Import required libraries
import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('../../Data/1.heal_prot+met+pop+diea+age.csv')

# Extract protein data
protein_data = data.iloc[:, 1:2924]

# Prepare features (X) and target variable (y)
X = data.iloc[:, 1:2924]  # Select protein features
y = data.iloc[:, -1]  # Age is the target variable

# Split the data into training (80%) and testing (20%) sets with fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a LightGBM regression model
model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Initialize SHAP TreeExplainer to interpret the LightGBM model
explainer = shap.TreeExplainer(model)
# Calculate SHAP values which show the impact of each feature on model predictions
shap_values = explainer.shap_values(X_train)

# Create a DataFrame of feature importance using mean absolute SHAP values
shap_importance = pd.DataFrame(list(zip(X_train.columns, np.abs(shap_values).mean(axis=0))),
                               columns=['Feature', 'Importance'])

# Select the top 300 most important features based on SHAP importance，n=50, 100, 200, ..., 800
top_300_features = shap_importance.nlargest(300, 'Importance')

# Filter the dataset to only include the top 300 important features
X_selected = X[top_300_features['Feature']]

# Add participant ID (eid) column as the first column
X_selected.insert(0, 'eid', data['eid'])

# Add age column as the last column
X_selected['Age'] = data['Age']

# Save the selected features with IDs and age to a CSV file
X_selected.to_csv('../../Data/SHAP/Prot/SHAP_300_data.csv', index=False)

# Count how many protein features were selected in the top 300
protein_features = [col for col in top_300_features['Feature'] if col in protein_data.columns]
protein_count = len(protein_features)

# Print the count of selected protein features
print(f'Number of selected protein features: {protein_count}')
