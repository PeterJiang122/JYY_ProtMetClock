# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import StackingRegressor
from scipy.stats import pearsonr
from xgboost import XGBRegressor
import joblib  # For saving models

# Read data from CSV file
data = pd.read_csv('.././Data/SHAP/Prot+Met/SHAP_300_data.csv')

# Extract features (all columns except first and last) and target (last column)
features = data.iloc[:, 1:-1]  # Exclude first column (eid) and last column (age)
target = data.iloc[:, -1]      # Last column is the target (age)

# Initialize DataFrame to store results (eid and calculated aging rate)
result_df = pd.DataFrame(columns=['eid', 'aging_rate'])

# Define base learners for stacking ensemble:
# 1. LightGBM with specific hyperparameters
# 2. ElasticNet with L1/L2 regularization
# 3. Linear Regression as simple baseline
base_learners = [
    ('lgbm', LGBMRegressor(boosting_type='gbdt', objective='regression',
                          learning_rate=0.01, n_estimators=1000)),
    ('elasticnet', ElasticNet(alpha=0.01, l1_ratio=0.99)),  # Strong L1 regularization
    ('lr', LinearRegression())  # Simple linear model
]

# Create StackingRegressor model:
# - Uses the 3 base learners for first-level predictions
# - Uses XGBoost as meta-learner (final estimator) to combine predictions
# - 5-fold cross-validation for stacking
stacking_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.01,
        n_estimators=1000,
        max_depth=6,
        subsample=0.8,        # Subsample ratio of training instances
        colsample_bytree=0.8, # Subsample ratio of features
        random_state=42      # For reproducibility
    ),
    cv=5  # 5-fold CV for the stacking process
)

# Initialize 5-fold cross-validation
# shuffle=True for better distribution, random_state for reproducibility
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Start StackingRegressor training with LightGBM, ElasticNet, and XGBoost as final estimator!")

# Initialize variables to track best model
best_model = None
best_pearson = -np.inf  # Initialize with negative infinity

# Perform 5-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
    print(f'Fold {fold + 1}')

    # Split data into training and validation sets
    X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
    y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

    # Train stacking model on current fold
    stacking_model.fit(X_train, y_train)

    # Predict ages on validation set
    y_pred = stacking_model.predict(X_val)

    # Calculate aging rate (predicted age / actual age)
    # Higher than 1 means aging faster than chronological age
    aging_rate = y_pred / y_val.values

    # Store results for current fold
    fold_result_df = pd.DataFrame({
        'eid': data.iloc[val_idx]['eid'].values,  # Subject IDs
        'aging_rate': aging_rate                  # Calculated aging rates
    })

    # Append to main results DataFrame
    result_df = pd.concat([result_df, fold_result_df], ignore_index=True)

    # Calculate evaluation metrics:
    # MSE, RMSE - Measure prediction error
    # R² - Explained variance
    # MAE, MAPE - Absolute errors
    # Pearson - Linear correlation
    mse = mean_squared_error(y_val, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100 if np.all(y_val != 0) else np.nan
    pearson_corr, _ = pearsonr(y_val, y_pred)

    # Print metrics for current fold
    print(f'Fold {fold + 1} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, '
          f'R²: {r2:.4f}, MAE: {mae:.4f}, '
          f'MAPE: {mape:.2f}%, Pearson: {pearson_corr:.4f}')

    # Update best model if current fold has higher Pearson correlation
    if pearson_corr > best_pearson:
        best_pearson = pearson_corr
        best_model = stacking_model

# Save the best performing model (highest Pearson correlation)
if best_model is not None:
    joblib.dump(best_model, '../Model/Best_Model/Best_Model.pkl')
    print(f'Best model saved with Pearson correlation: {best_pearson:.4f}')

print("Done!")

# Sort the results DataFrame by 'eid' in ascending order
result_df.sort_values(by='eid', inplace=True)

# Define the output file path for saving the aging rate results
output_path = '../Result/1.heal_AR.csv'
result_df.to_csv(output_path, index=False)

print(f'Results saved to {output_path}')
