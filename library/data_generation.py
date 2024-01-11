import os
import numpy as np
import pandas as pd

np.random.seed(0)

data_size = 300
target_size = 3
feature_size = 2

base_folder = os.path.dirname(os.getcwd())

feature_cols = [f'Feature{id+1}' for id in range(feature_size)]
target_cols = [f'Course{id+1}' for id in range(target_size)]

feature_1 = np.append(np.append(np.random.normal(loc=20, scale=10, size=int(data_size/3)),
                                np.random.normal(loc=50, scale=15, size=int(data_size/3))),
                      np.random.normal(loc=80, scale=15, size=int(data_size/3)))
feature_1 = np.where(feature_1 > 100, 100, feature_1)
feature_1 = np.where(feature_1 < 0, 0, feature_1)

feature_2 = np.append(np.append(np.random.choice([1, 2, 3], size=int(data_size/3), p=[0.8, 0.1, 0.1]),
                                np.random.choice([1, 2, 3], size=int(data_size/3), p=[0.1, 0.8, 0.1])),
                      np.random.choice([1, 2, 3], size=int(data_size/3), p=[0.1, 0.1, 0.8]))

all_features = np.array([feature_1, feature_2]).swapaxes(0, 1)
features_df = pd.DataFrame(all_features, columns=feature_cols)

binary_targets = np.append(np.append(np.repeat(0, data_size/target_size), np.repeat(1, data_size/target_size)), np.repeat(0, data_size/target_size))
binary_targets = np.vstack([binary_targets, np.append(np.repeat(1, data_size/target_size), np.repeat(0, 2*data_size/target_size))])
binary_targets = np.vstack([binary_targets, np.append(np.repeat(0, 2*data_size/target_size), np.repeat(1, data_size/target_size))])
binary_targets_df = pd.DataFrame(np.transpose(binary_targets), columns=target_cols)

numeric_targets = np.random.normal(loc=70, scale=15, size=(data_size, target_size))
numeric_targets = np.where(numeric_targets > 100, 100, numeric_targets)
numeric_targets = np.where(numeric_targets < 0, 0, numeric_targets)
numeric_targets_df = pd.DataFrame(numeric_targets, columns=target_cols)
targets_df = binary_targets_df * numeric_targets_df

full_df = pd.concat([features_df, targets_df], axis=1)
full_df = full_df.sample(frac=1, random_state=0).reset_index(drop=True)
full_df.to_csv(f'{base_folder}/data/full_df_size_{data_size}_targets_{target_size}.csv', index=False)
