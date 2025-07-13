# Import required libraries
import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('../../Data/1.heal_prot+met+pop+diea+age.csv')

# Extract protein and metabolite data
protein_data = data.iloc[:, 1:2924]  # Protein data
metabolite_data = data.iloc[:, 2924:3175]  # Metabolite data

# Prepare features (X) and target (y)
X = data.iloc[:, 1:3175]  # All features (proteins + metabolites)
y = data.iloc[:, -1]  # Target variable (age, last column)

# Split data into training (80%) and test (20%) sets with fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train LightGBM regression model
model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Initialize SHAP TreeExplainer for the LightGBM model
explainer = shap.TreeExplainer(model)
# Calculate SHAP values for training data (feature contributions to predictions)
shap_values = explainer.shap_values(X_train)

# Create DataFrame of feature importance (mean absolute SHAP values)
shap_importance = pd.DataFrame(list(zip(X_train.columns, np.abs(shap_values).mean(axis=0))),
                               columns=['Feature', 'Importance'])

# Select top 300 most important features based on SHAP values，n=50, 100, 200, ..., 800
top_300_features = shap_importance.nlargest(300, 'Importance')

# Filter original dataset to only include top 300 features
X_selected = X[top_300_features['Feature']]

# Add subject ID (eid) as first column
X_selected.insert(0, 'eid', data['eid'])

# Add age as last column
X_selected['Age'] = data['Age']

# Identify which features are from protein vs metabolite data
protein_features = [col for col in top_300_features['Feature'] if col in protein_data.columns]
metabolite_features = [col for col in top_300_features['Feature'] if col in metabolite_data.columns]

# Reorder features: proteins first, then metabolites
sorted_features = protein_features + metabolite_features
X_selected_sorted = X_selected[['eid'] + sorted_features + ['Age']]

# Save selected features to CSV file
X_selected_sorted.to_csv('../../Data/SHAP/Prot+Met/SHAP_300_data.csv', index=False)

# Print counts of selected protein and metabolite features
protein_count = len(protein_features)
metabolite_count = len(metabolite_features)

print(f'Number of protein features selected: {protein_count}')
print(f'Number of metabolite features selected: {metabolite_count}')
