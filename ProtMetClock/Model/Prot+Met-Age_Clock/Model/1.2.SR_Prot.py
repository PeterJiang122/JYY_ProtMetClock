# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from catboost import CatBoostRegressor

# Load dataset
data = pd.read_csv('.././Data/SHAP/Prot/SHAP_300_data.csv')

# Extract features (all columns except first and last) and target (last column)
features = data.iloc[:, 1:-1]
target = data.iloc[:, -1]

# Initialize DataFrame to store results
result_df = pd.DataFrame(columns=['eid', 'aging_rate'])

# Define base learners for stacking ensemble:
base_learners = [
    ('gbdt', GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=42)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(2, 2), max_iter=1000, random_state=42)),
    ('gp', GaussianProcessRegressor(n_restarts_optimizer=2, random_state=42)),
    ('et', ExtraTreesRegressor(n_estimators=100, max_depth=2, random_state=42))
]

# Create StackingRegressor model configuration:
stacking_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=CatBoostRegressor(
        iterations=10,
        depth=2,
        learning_rate=0.1,
        loss_function='RMSE',
        random_state=42,
        verbose=0
    ),
    cv=5
)

# Add random noise features to reduce potential overfitting
noise = np.random.rand(features.shape[0], 50)
# Combine original features with noise features
features_with_noise = pd.concat([features, pd.DataFrame(noise)], axis=1)

# Initialize 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Start StackingRegressor training with LightGBM, ElasticNet, and XGBoost as final estimator!")

# Initialize variables to track best model performance
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
