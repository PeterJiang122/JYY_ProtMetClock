# Import required libraries
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import pearsonr

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

# Configure LightGBM regression model with specified parameters
model = LGBMRegressor(random_state=42,
                      metric='r2',
                      boosting_type='gbdt',
                      objective='regression',
                      learning_rate=0.01,
                      n_estimators=1000)

# Initialize 5-fold cross-validation with shuffling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Starting 5-fold cross-validation with LightGBM!")

# Perform 5-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(features_scaled)):
    print(f'Processing Fold {fold + 1}')

    # Split data into training and validation sets
    X_train, X_val = features_scaled[train_idx], features_scaled[val_idx]
    y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

    # Train the model on training data
    model.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred = model.predict(X_val)

    # Calculate aging rate (actual age / predicted age)
    aging_rate = y_val.values / y_pred

    # Store results for current fold
    fold_result_df = pd.DataFrame({
        'eid': data.iloc[val_idx]['eid'].values,  # Sample IDs
        'aging_rate': aging_rate                  # Calculated aging rates
    })

    # Append current fold results to main results DataFrame
    result_df = pd.concat([result_df, fold_result_df], ignore_index=True)

    # Calculate various performance metrics
    mse = mean_squared_error(y_val, y_pred)               # Mean Squared Error
    rmse = np.sqrt(mse)                                   # Root Mean Squared Error
    r2 = r2_score(y_val, y_pred)                          # R-squared score
    mae = mean_absolute_error(y_val, y_pred)              # Mean Absolute Error
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100  # Mean Absolute Percentage Error
    pearson_corr, _ = pearsonr(y_val, y_pred)             # Pearson correlation coefficient

    # Print performance metrics for current fold
    print(f'Fold {fold + 1} Results:')
    print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')
    print(f'MAE: {mae:.4f}, MAPE: {mape:.4f}%, Pearson: {pearson_corr:.4f}')
    print('-' * 50)

print("Cross-validation completed successfully!")
