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

# Load dataset
data = pd.read_csv('.././Data/1.heal_prot+met+pop+diea+age.csv')

# Extract features
features = data.iloc[:, 2924:3175]
target = data.iloc[:, -1]   # Age as target variable

# Initialize DataFrame to store results (participant IDs and aging rates)
result_df = pd.DataFrame(columns=['eid', 'aging_rate'])

# Define base learners for stacking ensemble
base_learners = [
    ('lgbm', LGBMRegressor(boosting_type='gbdt', objective='regression', learning_rate=0.01, n_estimators=1000)),
    ('elasticnet', ElasticNet(alpha=0.01, l1_ratio=0.99)),
    ('lr', LinearRegression())
]

# Configure StackingRegressor:
# - Uses the 3 base learners for first-level predictions
# - Uses XGBoost as meta-learner (final estimator)
# - 5-fold cross-validation for stacking
stacking_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.01,
        n_estimators=1000,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    cv=5
)

# Initialize 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Start StackingRegressor training with LightGBM, ElasticNet, and XGBoost as final estimator!")

# Initialize variables to track best model
best_model = None
best_pearson = -np.inf

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
    aging_rate = y_pred / y_val.values

    # Store results for current fold
    fold_result_df = pd.DataFrame({
        'eid': data.iloc[val_idx]['eid'].values,
        'aging_rate': aging_rate
    })

    # Append to main results DataFrame
    result_df = pd.concat([result_df, fold_result_df], ignore_index=True)

    # Calculate evaluation metrics:
    # MSE, RMSE - Measure prediction error
    # R² - Explained variance
    # MAE, MAPE - Absolute errors
    # Pearson - Linear correlation between predicted and actual ages
    mse = mean_squared_error(y_val, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100 if np.all(y_val != 0) else np.nan
    pearson_corr, _ = pearsonr(y_val, y_pred)

    # Print metrics for current fold
    print(f'Fold {fold + 1} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, Pearson: {pearson_corr:.4f}')

print("Done!")
