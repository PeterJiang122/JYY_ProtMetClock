# Import required libraries
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np

# Load the dataset containing protein and metabolite features with SHAP values
data = pd.read_csv('../../Data/SHAP/Prot+Met/SHAP_300_data.csv')

# Extract features (all columns except first and last) and target (last column)
features = data.iloc[:, 1:-1]
target = data.iloc[:, -1]

# Standardize features using z-score normalization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Initialize a DataFrame to store results with columns for sample ID and aging rate
result_df = pd.DataFrame(columns=['eid', 'aging_rate'])

# Initialize Lasso regression model with L1 regularization
model = Lasso(alpha=0.01, random_state=42)  # alpha controls regularization strength

# Configure Bootstrap parameters
n_bootstrap = 5  # Number of bootstrap iterations
sample_size = int(0.8 * len(features_scaled))  # Training set size for each bootstrap sample

print("Starting Bootstrap sampling with Lasso regression!")

for bootstrap_iter in range(n_bootstrap):
    print(f'Processing Bootstrap Iteration {bootstrap_iter + 1}')

    # Generate bootstrap samples (with replacement) and validation set (out-of-bag samples)
    bootstrap_indices = np.random.choice(len(features_scaled), size=sample_size, replace=True)
    val_indices = [i for i in range(len(features_scaled)) if i not in bootstrap_indices]

    # Split data into bootstrap training set and validation set
    X_train, X_val = features_scaled[bootstrap_indices], features_scaled[val_indices]
    y_train, y_val = target.iloc[bootstrap_indices], target.iloc[val_indices]

    # Train the Lasso model on bootstrap sample
    model.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred = model.predict(X_val)

    # Calculate aging rate (actual age / predicted age)
    aging_rate = y_val.values / y_pred

    # Store results for current bootstrap iteration
    bootstrap_result_df = pd.DataFrame({
        'eid': data.iloc[val_indices]['eid'].values,  # Sample IDs
        'aging_rate': aging_rate                     # Calculated aging rates
    })
    result_df = pd.concat([result_df, bootstrap_result_df], ignore_index=True)

    # Calculate performance metrics
    mse = mean_squared_error(y_val, y_pred)               # Mean Squared Error
    rmse = np.sqrt(mse)                                   # Root Mean Squared Error
    r2 = r2_score(y_val, y_pred)                          # R-squared score
    mae = mean_absolute_error(y_val, y_pred)              # Mean Absolute Error
    # Mean Absolute Percentage Error (with division by zero check)
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100 if np.all(y_val != 0) else np.nan
    pearson_corr, _ = pearsonr(y_val, y_pred)             # Pearson correlation coefficient

    # Print performance metrics for current iteration
    print(f'Bootstrap Iteration {bootstrap_iter + 1} Results:')
    print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')
    print(f'MAE: {mae:.4f}, MAPE: {mape:.4f}%, Pearson: {pearson_corr:.4f}')
    print('-' * 60)

print("Bootstrap analysis completed successfully!")
