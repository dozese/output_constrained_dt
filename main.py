import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data_size = 1000

feature_cols = ['EnrolledElectiveBefore', 'GradeAvgFromPreviousElective',
                            'Grade', 'Major', 'Class', 'GradePerm']
target_cols = [f'Course{id+1}' for id in range(5)]

feature_0 = np.random.randint(low=2, size=data_size) # EnrolledElectiveBefore

feature_1 = np.random.normal(loc=60, scale=15, size=data_size) # GradeAvgFromPreviousElective
feature_1 = np.where(feature_1 > 100, 100, feature_1)
feature_1 = np.where(feature_1 < 0, 0, feature_1)
feature_1 = np.where(feature_0 == 0, 0, feature_1)

feature_2 = np.random.normal(loc=70, scale=10, size=data_size) # Grade
feature_2 = np.where(feature_2 > 100, 100, feature_2)
feature_2 = np.where(feature_2 < 0, 0, feature_2)

feature_3 = np.random.choice([1, 2, 3], size=data_size) # Major
feature_4 = np.random.choice([1, 2, 3, 4], size=data_size) # Class
feature_5 = np.random.permutation(feature_2) # GradePerm

all_features = np.array([feature_0, feature_1, feature_2, feature_3,
                         feature_4, feature_5]).swapaxes(0, 1)

features_df = pd.DataFrame(all_features, columns=feature_cols)

targets_list = np.empty((0, 5))
for _, row in features_df.iterrows():
    target_row = np.zeros(5)
    if row['GradeAvgFromPreviousElective'] > 0:
        course_grades = (0.5 * row['GradeAvgFromPreviousElective'] + 0.5 * row['Grade'] +
                        np.random.uniform(low=-10, high=10, size=5))
    else:
        course_grades = row['Grade'] + np.random.normal(loc=10, scale=2, size=5)
    course_grades = np.where(course_grades < 0, 0, course_grades)
    course_grades = np.where(course_grades > 100, 100, course_grades)
    if row['EnrolledElectiveBefore'] == 0:
        selected_course = np.random.choice([2, 3, 4])
        target_row[selected_course] = course_grades[selected_course]
        targets_list = np.vstack([targets_list, target_row])
        continue
    if row['Class'] == 0:
        selected_course = np.random.choice([0, 1])
        target_row[selected_course] = course_grades[selected_course]
        targets_list = np.vstack([targets_list, target_row])
        continue
    if (row['Major'] > 2) & (row['Class'] > 2):
        selected_course = np.random.choice([0, 2, 3, 4], size=2)
        target_row[selected_course] = course_grades[selected_course]
        targets_list = np.vstack([targets_list, target_row])
        continue
    if row['Grade'] < 70:
        selected_course = np.random.choice([1, 2, 3], size=2)
        target_row[selected_course] = course_grades[selected_course]
        targets_list = np.vstack([targets_list, target_row])
        continue
    selected_course = np.random.choice([0, 1, 2, 3, 4], size=2)
    target_row[selected_course] = course_grades[selected_course]
    targets_list = np.vstack([targets_list, target_row])

targets_list = np.empty((0, 5))
for _, row in features_df.iterrows():
    if row['GradeAvgFromPreviousElective'] > 0:
        course_grades = (0.5 * row['GradeAvgFromPreviousElective'] +
                         0.5 * row['Grade'] + np.random.uniform(-10, 10, 5))
    else:
        course_grades = row['Grade'] + np.random.normal(10, 2, 5)
    course_grades = np.clip(course_grades, 0, 100)

    if row['EnrolledElectiveBefore'] == 0:
        selected_course = np.random.choice([2, 3, 4])
    elif row['Class'] == 0:
        selected_course = np.random.choice([0, 1])
    elif row['Major'] > 2 and row['Class'] > 2:
        selected_course = np.random.choice([0, 2, 3, 4], size=2)
    elif row['Grade'] < 70:
        selected_course = np.random.choice([1, 2, 3], size=2)
    else:
        selected_course = np.random.choice([0, 1, 2, 3, 4], size=2)

    target_row = np.zeros(5)
    target_row[selected_course] = course_grades[selected_course]
    targets_list = np.vstack([targets_list, target_row])

targets_df = pd.DataFrame(targets_list, columns=target_cols)

features_df
targets_df

full_df = pd.concat([features_df, targets_df], axis=1)

x_train, x_test, y_train, y_test = train_test_split(features_df, targets_df, test_size = 0.3, random_state=0)

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print('\nRF Accuracy: ', mean_squared_error(y_test, y_pred))
cumsums = np.array([sum(y_pred[i] > 0.0001) for i in range(len(y_pred))])
print('Number of infeasible predictions for RF: ', np.sum(cumsums >= 3))

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print('\nDT Accuracy: ', mean_squared_error(y_test, y_pred))
cumsums = np.array([sum(y_pred[i] > 0.0001) for i in range(len(y_pred))])
print('Number of infeasible predictions for DT: ', np.sum(cumsums >= 3))

regressor = DecisionTreeRegressor(random_state=0, max_depth=3)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print('\nDT(max_depth=3) Accuracy: ', mean_squared_error(y_test, y_pred))
cumsums = np.array([sum(y_pred[i] > 0.0001) for i in range(len(y_pred))])
print('Number of infeasible predictions for DT(max_depth=3): ', np.sum(cumsums >= 3))