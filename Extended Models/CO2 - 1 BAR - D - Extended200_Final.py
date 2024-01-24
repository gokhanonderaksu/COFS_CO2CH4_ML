import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('C:/Users/WS01/Desktop/Github/Extended Excels/Group D - 200/CO2 - 1 BAR - D - Extended200.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('NCO2 at 1 bar (mol/kg)', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['NCO2 at 1 bar (mol/kg)'], random_state=42)
            
# Average CV score on the training set was: -0.010939522662844864
exported_pipeline = make_pipeline(
    RobustScaler(),
    StackingEstimator(estimator=RidgeCV()),
    XGBRegressor(learning_rate=0.1, max_depth=4, min_child_weight=19, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.8, verbosity=0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
y_pred_train=exported_pipeline.predict(training_features)
preds = exported_pipeline.predict(testing_features)

X_unseen = pd.read_csv('C:/Users/WS01/Desktop/Github/Unseen Test Sheets/Unseen Label Sheets - Part2/hypoCOFUnseen-CO2 - D - 298 K - Part2 - 42K.csv', sep=',')
y_pred_unseen = exported_pipeline.predict(X_unseen)
df5 = pd.DataFrame(y_pred_unseen)
df5.to_csv('hypoCOFUnseen-CO2 - D - 298 K - Part2 - 42K.csv', index=False)

# Create a list to store the metrics
metrics = []

# Calculate metrics
metrics.append({'Metric': 'R2', 'Train': r2_score(training_target, y_pred_train), 'Test': r2_score(testing_target, preds)})
metrics.append({'Metric': 'MSE', 'Train': mean_squared_error(training_target, y_pred_train), 'Test': mean_squared_error(testing_target, preds)})
metrics.append({'Metric': 'MAE', 'Train': mean_absolute_error(training_target, y_pred_train), 'Test': mean_absolute_error(testing_target, preds)})
metrics.append({'Metric': 'RMSE', 'Train': math.sqrt(mean_squared_error(training_target, y_pred_train)), 'Test': math.sqrt(mean_squared_error(testing_target, preds))})
metrics.append({'Metric': 'SRCC', 'Train': spearmanr(training_target, y_pred_train)[0], 'Test': spearmanr(testing_target, preds)[0]})

# Create a DataFrame from the list of metrics
metrics_df = pd.DataFrame(metrics)

# Print the DataFrame
print(metrics_df)

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Scatter plot for training data
training_scatter = ax.scatter(training_target, y_pred_train, color="blue", label='Training Data')
ax.set_xlabel('True Values')
ax.set_ylabel('Predicted Values')

# Scatter plot for testing data
testing_scatter = ax.scatter(testing_target, preds, color="red", label='Testing Data')
ax.set_xlabel('Simulated')
ax.set_ylabel('ML-predicted')

# Plot the x=y line
x = np.linspace(min(min(training_target), min(testing_target)), max(max(training_target), max(testing_target)), 100)
ax.plot(x, x, color='black', linestyle='--', label='_nolegend_')

# Add legend with only "Training Data" and "Testing Data"
handles = [training_scatter, testing_scatter]
labels = [handle.get_label() for handle in handles]
ax.legend(handles=handles, labels=labels)

# Show the plot
plt.show()
