# Import required libraries
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# Load the pre-trained best model from saved file
best_model_path = '../Model/Best_Model/Best_Model.pkl'
print(f"Loading best model from {best_model_path}...")
best_model = joblib.load(best_model_path)

# Read new unseen data for prediction
new_data_path = '.././Data/2.unheal_prot+met+pop+diea+age.csv'
data = pd.read_csv(new_data_path)

# Extract feature columns
features = data.iloc[:, 1:301]

# Initialize DataFrame to store prediction results
result_df = pd.DataFrame(columns=['eid', 'aging_rate'])

# Make predictions using the loaded model
print("Predicting aging rates using the loaded model...")
y_pred = best_model.predict(features)

# Calculate aging rate (predicted age / actual age)
target = data.iloc[:, -1]
aging_rate = y_pred / target

# Create results DataFrame with participant IDs and calculated aging rates
result_df['eid'] = data['eid']
result_df['aging_rate'] = aging_rate

# Add participant IDs
result_df['eid'] = data['eid']
# Add calculated aging rates
result_df['aging_rate'] = aging_rate

# Calculate model performance metrics:
# - MSE/RMSE: Measures of prediction error
# - R²: Explained variance
# - MAE: Mean absolute error
# - Pearson: Linear correlation coefficient
mse = mean_squared_error(target, y_pred)
rmse = mse ** 0.5
r2 = r2_score(target, y_pred)
mae = mean_absolute_error(target, y_pred)
pearson_corr, _ = pearsonr(target, y_pred)

# Print evaluation metrics
print("Evaluation Metrics:")
print(f"R-squared: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Pearson Correlation: {pearson_corr:.4f}")

# Save results to CSV file
output_path = '../Result/2.unheal_AR.csv'
result_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}!")
