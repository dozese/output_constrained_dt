import numpy as np
import pandas as pd

series_length = 1000
dataset = np.zeros(series_length)

#### Constraints
# Each value in the series should be less than 100
dataset[0:2] = np.random.randint(0, 100, size=2)

# If two consecutive values sum up to 70 or more, the next value should be less than 50
if (dataset[1] + dataset[0] >= 70):
    dataset[2] = np.random.randint(0, 50)

# If three consecutive values sum up to 120 or more, the next value should be less than 10
for i in range(3, series_length):
    dataset[i] = np.random.randint(0, 100)
    if (dataset[i-1] + dataset[i-2] >= 70):
        dataset[i] = np.random.randint(0, 50)
    if (dataset[i-1] + dataset[i-2] + dataset[i-3] >= 120):
        dataset[i] = np.random.randint(0, 10)
####

# Function to prepare dataset for multi-target prediction
def prepare_dataset(data, n_features, n_targets):
    X, y = [], []
    endi = len(data) - (n_features + n_targets) + 1
    for i in range(endi):
        features = data[i:i+n_features]
        targets = data[i+n_features:i+n_features+n_targets]
        X.append(features)
        y.append(targets)

    return np.array(X), np.array(y)

# Define number of features and number of steps ahead for prediction
n_features = 30
n_targets = 6

# Prepare dataset
X, y = prepare_dataset(dataset, n_features, n_targets)

# Reshape X to have n_features per sample
X = X.reshape((X.shape[0], n_features))

# Print the input-output pairs
for i in range(len(X)):
    print(X[i], y[i])

feature_cols = [f'Feature {id+1}' for id in range(X.shape[1])]
target_cols = [f'Target {id + 1}' for id in range(y.shape[1])]

feature_df = pd.DataFrame(X, columns=feature_cols)
target_df = pd.DataFrame(y, columns=target_cols)

full_df = pd.concat([feature_df, target_df], axis=1)
full_df.to_csv('data/forecasting.csv', index=False)