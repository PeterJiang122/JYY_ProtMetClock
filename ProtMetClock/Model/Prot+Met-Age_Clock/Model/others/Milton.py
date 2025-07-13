# Import required libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


# Data augmentation function using interpolation
def interpolation_augmentation(features, targets, num_augmentations=5):
    """
    Generate augmented data by linear interpolation between random pairs of samples

    Args:
        features: Original feature matrix
        targets: Original target values
        num_augmentations: Number of augmented samples to generate

    Returns:
        Tuple of augmented features and corresponding targets
    """
    augmented_features = []
    augmented_targets = []
    for _ in range(num_augmentations):
        # Randomly select two different samples
        idx = np.random.choice(len(features), 2, replace=False)
        # Generate random interpolation coefficient
        alpha = np.random.rand()
        # Create interpolated features and targets
        interpolated_features = alpha * features[idx[0]] + (1 - alpha) * features[idx[1]]
        interpolated_targets = alpha * targets[idx[0]] + (1 - alpha) * targets[idx[1]]
        augmented_features.append(interpolated_features)
        augmented_targets.append(interpolated_targets)
    return np.array(augmented_features), np.array(augmented_targets)


# Load data from CSV file
data = pd.read_csv('../../Data/SHAP/Prot+Met/SHAP_300_data.csv')

# Extract features (all columns except first and last) and targets (last column)
features = data.iloc[:, 1:-1].values
targets = data.iloc[:, -1].values  # Assuming target is in the last column

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Apply data augmentation to training set
aug_features, aug_targets = interpolation_augmentation(X_train, y_train, num_augmentations=100)
# Combine original and augmented data
X_train = np.vstack([X_train, aug_features])
y_train = np.hstack([y_train, aug_targets])

# Initialize XGBoost model
print("Training Model...")
# Convert data to XGBoost's DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'objective': 'reg:squarederror',  # Regression task
    'eval_metric': 'rmse',  # Root Mean Squared Error as evaluation metric
    'eta': 0.1,  # Learning rate
    'max_depth': 6,  # Maximum tree depth
    'subsample': 0.8,  # Subsample ratio of training instances
    'colsample_bytree': 0.8,  # Subsample ratio of features
    'seed': 42  # Random seed for reproducibility
}

# Train the model with early stopping
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,  # Maximum number of boosting iterations
    evals=[(dtrain, 'train'), (dtest, 'eval')],  # Evaluation sets
    early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
    verbose_eval=False  # Don't print evaluation progress
)

# Prepare test data for prediction
dtest = xgb.DMatrix(X_test)
# Make predictions on test set
y_pred = model.predict(dtest)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared score
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
pearson_corr, _ = pearsonr(y_test, y_pred)  # Pearson correlation coefficient

# Print evaluation results
print("Model Metrics:")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²: {r2:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  MAPE: {mape:.2f}%")
print(f"  Pearson Correlation: {pearson_corr:.4f}")

print("Done!")
