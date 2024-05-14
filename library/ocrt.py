# Based on the work by baydoganm/mtTrees

import os
import time
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import KFold

import gurobipy as gp
from gurobipy import GRB

def return_medoid(y):
    return y[np.argmin(euclidean_distances(y).mean(axis=1))]

def return_mean(y):
    return np.asarray(y.mean(axis=0))

def calculate_mse(y, predictions):
    return ((y - predictions) ** 2).mean()

def calculate_mad(y, predictions):
    return np.mean(np.abs(y - predictions))

def calculate_poisson_deviance(y, predictions):
    return 2 * np.sum(predictions - y - y * np.log(predictions / y))

def formulate_and_solve_lp_cars_data(y, x, lagrangian_multiplier=0, verbose=False, bigM=100000):
    # Create a new Gurobi model
    model = gp.Model("Binary and Continuous Variables")
    if not verbose:
        model.Params.LogToConsole = 0

    # Define variables
    predictions = model.addVars(1, lb=0, name="y")  # Continuous variables
    binary_vars = model.addVars(1, vtype=GRB.BINARY, name="z")  # Binary indicators

    # Objective: Minimize Sum of Squared Errors (SSE)
    sse = gp.quicksum((predictions[0] - y[i][1]) * (predictions[0] - y[i][1]) for i in range(y.shape[0]))

    if lagrangian_multiplier > 0:
        # Constraint penalization terms
        constraint_terms = gp.QuadExpr(0)

        # Constraint 1: predictions[0] >= binary_vars[0]
        constraint_terms.add(predictions[0] - binary_vars[0])

        # Constraint 2: predictions[0] <= bigM * binary_vars[0]
        constraint_terms.add(bigM * binary_vars[0] - predictions[0])

        # Add Lagrangian relaxation term to the objective function
        sse += lagrangian_multiplier * constraint_terms
    else:
        # Constraints
        model.addConstr(predictions[0] >= binary_vars[0], "y_constraint")
        model.addConstr(predictions[0] <= bigM * binary_vars[0], "y_upper_bound_constraint")

    model.setObjective(sse, GRB.MINIMIZE)

    # Optimize the model
    model.optimize()
    # model.write('/home/user/Desktop/Research/OCRT/model.lp')

    preds = np.array([binary_vars[0].X, predictions[0].X])

    # Display the results
    if verbose:
        print(f"Optimal Solution: {preds}")
        print(f"Objective (Sum of Squared Errors): {model.objVal}")

    return preds

def formulate_and_solve_lp_courses_data(y, x, lagrangian_multiplier=0, verbose=False):
    num_instances = y.shape[0]
    num_targets = y.shape[1]

    # Create model
    model = gp.Model("Minimize SSE")
    if not verbose:
        model.Params.LogToConsole = 0

    # Create decision variables
    predictions = model.addVars(num_targets, lb=0, ub=100, name="y")
    binary_vars = model.addVars(num_targets, vtype=GRB.BINARY, name="z")

    # Create objective function
    sse = gp.quicksum((predictions[j] - y[i][j]) * (predictions[j] - y[i][j]) for i in range(num_instances) for j in range(num_targets))

    if lagrangian_multiplier > 0:
        # Constraints penalization terms
        constraint_terms = gp.QuadExpr(0)

        # One prediction constraint
        constraint_terms.add(binary_vars.sum() - 1)

        # Constraint 1: predictions[i] <= 110 * binary_vars[i]
        for i in range(num_targets):
            constraint_terms.add(predictions[i] - 110 * binary_vars[i])

        # Add Lagrangian relaxation term to the objective function
        sse += lagrangian_multiplier * constraint_terms * num_instances
    else:
        # Create constraints
        model.addConstr(binary_vars.sum() <= 1, "one_prediction_constraint")
        for i in range(num_targets):
            model.addConstr(predictions[i] <= 110 * binary_vars[i], f"z_relationship_{i}")

    # Solve the problem
    model.setObjective(sse, GRB.MINIMIZE)
    model.optimize()
    # model.write('/home/user/Desktop/Research/OCRT/model.lp')

    if verbose:
        print("Optimal Solution:")
        for i in range(num_targets):
            print(f"Target {i + 1}: Prediction = {predictions[i].X}, Indicator = {binary_vars[i].X}")
        print("Objective (Sum of Squared Errors):", model.objVal)

    preds = np.array([predictions[i].X for i in range(num_targets)])

    return preds


def formulate_and_solve_lp_forecasting_data(y, x, lagrangian_multiplier = 0, verbose = False):
    num_instances = y.shape[0]
    num_targets = y.shape[1]

    # Create a new model
    model = gp.Model("Minimize SSE")
    if not verbose:
        model.Params.LogToConsole = 0

    # Decision Variables
    predictions = model.addVars(num_targets, lb=0, ub=100, name="predictions")
    indicator_70 = model.addVars(num_targets-1, vtype=GRB.BINARY, name=f"indicator_70")
    indicator_120 = model.addVars(num_targets-1, vtype=GRB.BINARY, name=f"indicator_120")

    # Objective
    # Minimize the sum of squared errors (SSE) between predictions and actual values
    sse = gp.quicksum((predictions[j] - y[i][j]) * (predictions[j] - y[i][j]) for i in range(num_instances) for j in range(num_targets))
    model.setObjective(sse, GRB.MINIMIZE)

    # Constraints
    X_df = pd.DataFrame(x[:, -3:])
    if len(X_df.loc[X_df.sum(axis=1) > 120]):
        model.addConstr(predictions[0] <= 10)
    if len(X_df[X_df.loc[:, [1, 2]].sum(axis=1) > 70]):
        model.addConstr(predictions[0] <= 50)

    # Constraint: If sum >= 70, then next prediction should be less than 50
    model.addConstr(predictions[0] + max(X_df[2]) >= 70 * indicator_70[0])
    model.addConstr(predictions[0] <= 50 + 50 * (1 - indicator_70[0]))

    # Constraint: If sum >= 120, then next prediction should be less than 10
    model.addConstr(predictions[2] + predictions[1] + max(X_df[2]) >= 120 * indicator_120[0])
    model.addConstr(predictions[2] <= 10 + 90 * (1 - indicator_120[0]))

    model.addConstr(predictions[1] + X_df.loc[:, 1:2].sum(axis=1).max() >= 120 * indicator_120[1])
    model.addConstr(predictions[1] <= 10 + 90 * (1 - indicator_120[1]))

    for i in range(2, num_targets):
        # Constraint: If sum >= 70, then next prediction should be less than 50
        model.addConstr(predictions[i - 1] + predictions[i - 2] >= 70 * indicator_70[i-1])
        model.addConstr(predictions[i] <= 50 + 50 * (1 - indicator_70[i-1]))

        if i > 2:
            # Constraint: If sum >= 120, then next prediction should be less than 10
            model.addConstr(predictions[i - 1] + predictions[i - 2] + predictions[i - 3] >= 120 * indicator_120[i-1])
            model.addConstr(predictions[i] <= 10 + 90 * (1 - indicator_120[i-1]))

    # Optimize the model
    model.optimize()

    if verbose:
        print("Optimal Solution:")
        for i in range(num_targets):
            print(f"Target {i + 1}: Prediction = {predictions[i].X}")
        print("Objective (Sum of Squared Errors):", model.objVal)

    preds = np.array([predictions[i].X for i in range(num_targets)])

    return preds


def formulate_and_solve_lp_new_class_data(y, x, lagrangian_multiplier = 0, verbose = False):
    num_instances = y.shape[0]
    num_targets = y.shape[1]

    # Create model
    model = gp.Model("Minimize SSE")
    if not verbose:
        model.Params.LogToConsole = 0

    # Create decision variables
    predictions = model.addVars(num_targets, lb=0, ub=100, name="y")
    binary_vars = model.addVars(num_targets, vtype=GRB.BINARY, name="z")

    # Create objective function
    sse = gp.quicksum((predictions[j] - y[i][j]) * (predictions[j] - y[i][j]) for i in range(num_instances) for j in range(num_targets))

    if lagrangian_multiplier > 0:
        # Constraints penalization terms
        constraint_terms = gp.QuadExpr(0)

        # Constraint 1: predictions[i] <= 110 * binary_vars[i]
        for i in range(num_targets):
            constraint_terms.add(predictions[i] - 110 * binary_vars[i])

        # Constraint 2: predictions[1] >= 50 * binary_vars[2]
        constraint_terms.add(50 * binary_vars[2] - predictions[1])

        # Constraint 3: predictions[1] + predictions[2] >= 110 * binary_vars[0]
        constraint_terms.add(110 * binary_vars[0] - predictions[1] - predictions[2])

        # Add Lagrangian relaxation term to the objective function
        sse += lagrangian_multiplier * constraint_terms * num_instances
    else:
        # Create constraints
        for i in range(num_targets):
            model.addConstr(predictions[i] <= 110 * binary_vars[i], f"z_relationship_{i}")

        model.addConstr(predictions[1] >= 50 * binary_vars[2], "y_constraint_1")
        model.addConstr(predictions[1] + predictions[2] >= 110 * binary_vars[0], "y_constraint_2")

    # Solve the problem
    model.setObjective(sse, GRB.MINIMIZE)
    model.optimize()
    # model.write('/home/user/Desktop/Research/OCRT/model.lp')

    if verbose:
        print("Optimal Solution:")
        for i in range(num_targets):
            print(f"Target {i + 1}: Prediction = {predictions[i].X}, Indicator = {binary_vars[i].X}")
        print("Objective (Sum of Squared Errors):", model.objVal)

    preds = np.array([predictions[i].X for i in range(num_targets)])

    return preds

def calculate_number_of_infeasibilities(y_pred, X_test, dataset, model, ocrt_depth, target_cols, verbose=True):
    if dataset == 'class':
        cumsums = np.array([sum(y_pred[i] > 0.0001) for i in range(len(y_pred))])
        nof_infeasibilities = np.sum(cumsums >= 3)
    elif dataset == 'newclass':
        nof_infeasibilities = 0
        for i in range(len(y_pred)):
            if (y_pred[i][1] < 50) and (y_pred[i][2] > 0.0001):
                nof_infeasibilities += 1
            elif (round(y_pred[i][1] + y_pred[i][2], 4) < 110) and (y_pred[i][0] > 0.0001):
                nof_infeasibilities += 1
    elif dataset == 'forecasting':
        nof_infeasibilities = 0
        nof_infeasibilities += y_pred[y_pred > 100].shape[0]
        y_pred_df = pd.DataFrame(y_pred, index=X_test.index)
        y_pred_df = pd.concat([X_test.iloc[:, -3:], y_pred_df], axis=1)
        y_pred_df.columns = range(y_pred_df.shape[1])
        for i in range(1, y_pred_df.shape[1] - 2):
            nof_infeasibilities += y_pred_df[(y_pred_df.loc[:, [i, i+1]].sum(axis=1) > 70) &
                                             (y_pred_df.loc[:, i+2] > 50)].shape[0]
        for i in range(y_pred_df.shape[1] - 3):
            nof_infeasibilities += y_pred_df[(y_pred_df.loc[:, i: i + 2].sum(axis=1) > 120) &
                                             (y_pred_df.loc[:, i + 3] > 10)].shape[0]
    else:
        y_pred_df = pd.DataFrame(y_pred, columns=target_cols)
        nof_infeasibilities = y_pred_df[(y_pred_df['TARGET_FLAG'] == 0) & (y_pred_df['TARGET_AMT'] > 0)].shape[0]

    if verbose:
        print(f'Number of infeasible predictions for {model} (Depth {ocrt_depth}): {nof_infeasibilities}')

    return nof_infeasibilities

def split_criteria_with_methods(y, x, nof_infeasibilities_method, lagrangian_multiplier, prediction_method,
                                evaluation_method, optimization_problem, verbose=False):
    if prediction_method == 'medoid':
        predictions = return_medoid(y)
    elif prediction_method == 'optimal':
        predictions = optimization_problem(y, x, lagrangian_multiplier, verbose)
    else:
        predictions = return_mean(y)

    if evaluation_method == 'mse':
        split_evaluation = calculate_mse(y, predictions)
    elif evaluation_method == 'mad':
        split_evaluation = calculate_mad(y, predictions)
    else:
        split_evaluation = calculate_poisson_deviance(y, predictions)

    predictions_all = (predictions * np.ones((y.shape[0], y.shape[1])))
    nof_infeasibilities = nof_infeasibilities_method(predictions_all, x)
    # split_evaluation += nof_infeasibilities * lagrangian_multiplier

    return predictions, split_evaluation

class Node:
    """
    Node class for a Decision Tree.

    Attributes:
        right (Node): Right child node.
        left (Node): Left child node.
        column (int): Index of the feature used for splitting.
        column_name (str): Name of the feature.
        threshold (float): Threshold for the feature split.
        id (int): Identifier for the node.
        depth (int): Depth of the node in the tree.
        is_terminal (bool): Indicates if the node is a terminal node.
        prediction (numpy.ndarray): Predicted values for the node.
        count (int): Number of samples in the node.

    Methods:
        No specific methods are defined in this class.

        It will be mainly used in the construction of the tree.
    """
    def __init__(self):
        self.right = None
        self.left = None
        self.column = None
        self.column_name = None
        self.threshold = None
        self.id = None
        self.depth = None
        self.is_terminal = False
        self.prediction = None
        self.count = None

class OCDT:
    """
    Predictive Clustering Tree.

    Args:
        max_depth (int): Maximum depth of the tree.
        min_samples_leaf (int): Minimum number of samples in a leaf node.
        min_samples_split (int): Minimum number of samples to split a node.
        split_style (str): Splitting style.
        verbose (bool): Whether to print verbose information.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_samples_leaf (int): Minimum number of samples in a leaf node.
        min_samples_split (int): Minimum number of samples to split a node.
        split_style (str): Splitting style (e.g. 'custom')
        verbose (bool): Whether to print verbose information.
        Tree (Node): Root node of the predictive clustering tree.

    Methods:
        buildDT(features, labels, node):
            Build the predictive clustering tree.

        fit(features, labels):
            Fit the predictive clustering tree to the data.

        nodePredictions(y):
            Calculate predictions for a node.

        applySample(features, depth, node):
            Passes one object through the decision tree and returns the prediction.

        apply(features, depth):
            Returns the node id for each X.

        get_rules(features, depth, node, rules):
            Returns the decision rules for feature selection.

        calcBestSplit(features, labels, current_label):
            Calculates the best split based on features and labels.

        calcBestSplitCustom(features, labels):
            Calculates the best custom split for features and labels.
    """
    def __init__(self, max_depth = 5,
                 min_samples_leaf = 5,
                 min_samples_split = 10,
                 split_criteria = None,
                 leaf_prediction_method = None,
                 ocrt_solve_only_leaves = False,
                 nof_infeasibilities_method = None,
                 verbose = False):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.split_criteria = split_criteria
        self.ocrt_solve_only_leaves = ocrt_solve_only_leaves
        self.nof_infeasibilities_method = nof_infeasibilities_method
        self.leaf_prediction_method = leaf_prediction_method
        self.verbose = verbose
        self.Tree = None

    def buildDT(self, features, labels, node):
        """
        Build the predictive clustering tree.

        Args:
            features (pandas.DataFrame): The input features used for building the tree.
            labels (pandas.DataFrame): The labels or target variables corresponding to the features.
            node (Node): The current node in the tree being built.
        """
        node.prediction, _ = self.split_criteria(labels.to_numpy(), features.to_numpy(), self.nof_infeasibilities_method)

        node.count = labels.shape[0]
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return

        if features.shape[0] < self.min_samples_split:
            node.is_terminal = True
            return

        current_label = range(labels.shape[1])
        target = labels

        split_info, split_gain, n_cuts = self.calcBestSplitCustom(features, target)

        if n_cuts == 0:
            node.is_terminal = True
            return

        min_max_scaler = preprocessing.MinMaxScaler()
        split_gain_scaled_total = min_max_scaler.fit_transform(split_gain)[:, 0]
        mean_rank_sort = np.argsort(split_gain_scaled_total)

        splitCol = int(split_info[mean_rank_sort[0], 0])
        thresh = split_info[mean_rank_sort[0], 1]

        node.column = splitCol
        node.column_name = features.columns[splitCol]
        node.threshold = thresh

        labels_left = labels.loc[features.iloc[:,splitCol] <= thresh, :]
        labels_right = labels.loc[features.iloc[:,splitCol] > thresh, :]

        features_left = features.loc[features.iloc[:,splitCol] <= thresh]
        features_right = features.loc[features.iloc[:,splitCol] > thresh]

        # creating left and right child nodes
        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.id = 2 * node.id

        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.id = 2 * node.id + 1

        # splitting recursively
        self.buildDT(features_left, labels_left, node.left)
        self.buildDT(features_right, labels_right, node.right)


    def fit(self, features, labels):
        """
        Fit the predictive clustering tree to the data.

        Args:
            features (pandas.DataFrame): The input features used for building the tree.
            labels (pandas.DataFrame): The labels or target variables corresponding to the features.
        """
        start = time.time()
        self.Tree = Node()
        self.Tree.depth = 0
        self.Tree.id = 1
        self.buildDT(features, labels, self.Tree)
        leaves = self.apply(features)
        leaf_predictions = {}
        for leaf_id in np.unique(leaves):
            leaf_indices = np.where(leaves == leaf_id)[0]
            leaf_labels = labels.iloc[leaf_indices].to_numpy()
            leaf_features = features.iloc[leaf_indices].to_numpy()
            leaf_predictions[leaf_id], _ = self.leaf_prediction_method(leaf_labels, leaf_features, self.nof_infeasibilities_method)
        self.leaf_predictions_df = pd.DataFrame(leaf_predictions)
        end = time.time()
        self.training_duration = end-start

    def predict(self, features):
        '''
        Returns the labels for each X
        '''
        leaves = self.apply(features)
        predictions = self.leaf_predictions_df[leaves].T

        return np.asarray(predictions)

    def predictSample(self, features, depth, node):
        '''
        Passes one object through decision tree and return the probability of it to belong to each class
        '''

        # if we have reached the terminal node of the tree
        if node.is_terminal:
            return node.prediction

        # if we have reached the provided depth
        if node.depth == depth:
            return node.prediction

        if features.iloc[node.column] > node.threshold:
            predicted = self.predictSample(features, depth, node.right)
        else:
            predicted = self.predictSample(features, depth, node.left)

        return predicted

    def applySample(self, features, depth, node):
        """
        Passes one object through the predictive clustering tree and returns the leaf ID.

        Args:
            features (pandas.Series): The input features for a single object.
            depth (int): The depth at which to stop traversing the tree.
            node (Node): The current node in the tree being traversed.

        Returns:
            predicted (int): The predicted node ID.
        """

        # if we have reached the terminal node of the tree
        if node.is_terminal:
            return node.id

        # if we have reached the provided depth
        if node.depth == depth:
            return node.id

        if features.iloc[node.column] > node.threshold:
            predicted = self.applySample(features, depth, node.right)
        else:
            predicted = self.applySample(features, depth, node.left)

        return predicted

    def apply(self, features):
        """
        Returns the node ID for each input object.

        Args:
            features (pandas.DataFrame): The input features for multiple objects.

        Returns:
            predicted_ids (numpy.ndarray): The predicted node IDs for each input object.
        """
        predicted_ids = [self.applySample(features.loc[i], self.max_depth, self.Tree) for i in features.index]
        predicted_ids = np.asarray(predicted_ids)
        return predicted_ids

    def get_rules(self, features, depth, node, rules):
        """
        Returns the decision rules for leaf node assignment.

        Args:
            features (pandas.Series): The input features for a single object.
            depth (int): The depth at which to stop traversing the tree.
            node (Node): The current node in the tree being traversed.
            rules (list): A list to store the decision rules.

        Returns:
            rules (list): The updated list of decision rules.
        """
        # if we have reached the terminal node of the tree
        if node.is_terminal:
            msg = f'Ended at terminal node with ID: {node.id}'
            print(msg)
            return rules

        # if we have reached the provided depth
        if node.depth == depth:
            msg = f'Ended at depth' + str(node.depth)
            print(msg)
            return rules

        if features.iloc[:,node.column].values[0] > node.threshold:
            msg = f'Going right: Node ID: {node.id}, Rule: {features.columns[node.column]} > {node.threshold}'
            print(msg)
            rules.append({features.columns[node.column]: {'min': node.threshold}})
            rules = self.get_rules(features, depth, node.right, rules)
        else:
            msg = f'Going left: Node ID: {node.id}, Rule: {features.columns[node.column]} <= {node.threshold}'
            print(msg)
            rules.append({features.columns[node.column]: {'max': node.threshold}})
            rules = self.get_rules(features, depth, node.left, rules)

        return rules

    def calcBestSplitCustom(self, features, labels):
        n = features.shape[0]
        cut_id = 0
        n_obj = 1
        split_perf = np.zeros((n * features.shape[1], n_obj))
        split_info = np.zeros((n * features.shape[1], 2))
        for k in range(features.shape[1]):
            if self.verbose:
                print(f'Feature Index: {k}')
            x = features.iloc[:, k].to_numpy()
            y = labels.to_numpy()
            sort_idx = np.argsort(x)
            sort_x = x[sort_idx]
            sort_y = y[sort_idx, :]

            for i in range(self.min_samples_leaf, n - self.min_samples_leaf - 1):
                xi = sort_x[i]

                left_yi = sort_y[:i, :]
                right_yi = sort_y[i:, :]

                left_xi = features.to_numpy()[sort_idx][:i]
                right_xi = features.to_numpy()[sort_idx][i:]

                left_instance_count = left_yi.shape[0]
                right_instance_count = right_yi.shape[0]

                left_prediction, left_perf = self.split_criteria(left_yi, left_xi, self.nof_infeasibilities_method)
                right_prediction, right_perf = self.split_criteria(right_yi, right_xi, self.nof_infeasibilities_method)
                curr_score = (left_perf * left_instance_count + right_perf * right_instance_count) / n

                split_perf[cut_id, 0] = curr_score
                split_info[cut_id, 0] = k
                split_info[cut_id, 1] = xi

                if i < self.min_samples_leaf or xi == sort_x[i + 1]:
                    continue

                cut_id += 1

        split_info = split_info[range(cut_id), :]
        split_gain = split_perf[range(cut_id), :]
        n_cuts = cut_id

        split_info = split_info[~np.isnan(split_gain).any(axis=1),:]
        split_gain = split_gain[~np.isnan(split_gain).any(axis=1),:]

        return split_info, split_gain, n_cuts


if __name__ == '__main__':
    base_folder = os.getcwd()
    ocrt_min_samples_split = 10
    ocrt_min_samples_leaf = 5
    number_of_folds = 5

    # dataset = 'class' # class, cars, newclass, forecasting
    dataset = 'newclass'
    verbose = False
    class_target_size = 5
    prediction_method = 'mean' # mean, medoid, optimal
    prediction_method_leaf = 'medoid'  # mean, medoid, optimal
    evaluation_method = 'mse' # mse, mad, poisson

    ocrt_depth_list = [6, 9, 12, 15]
    dataset_list = ['newclass', 'class', 'cars']
    evaluation_method_list = ['mse']
    prediction_method_leaf_list = ['medoid', 'optimal']
    prediction_method_list = ['optimal', 'mean', 'medoid']

    for ocrt_depth in ocrt_depth_list:
        for dataset in dataset_list:
            perf_df = pd.DataFrame()

            if dataset == 'class':
                optimization_problem = formulate_and_solve_lp_courses_data
                target_cols = [f'Course{id + 1}' for id in range(class_target_size)]
                lagrangian_multiplier = 1500
                if class_target_size == 3:
                    feature_size = 2
                    feature_cols = [f'Feature{id + 1}' for id in range(feature_size)]
                    full_df = pd.read_csv(f'{base_folder}/data/full_df_size_300_targets_3.csv')

                    features_df = full_df[feature_cols]
                    targets_df = full_df[target_cols]
                else:
                    feature_cols = ['EnrolledElectiveBefore', 'GradeAvgFromPreviousElective',
                                    'Grade', 'Major', 'Class', 'GradePerm']

                    full_df = pd.read_csv(f'{base_folder}/data/full_df_size_1000_targets_5.csv')

                    features_df = full_df[feature_cols]
                    targets_df = full_df[target_cols]
            elif dataset == 'forecasting':
                optimization_problem = formulate_and_solve_lp_forecasting_data
                full_df = pd.read_csv(f'{base_folder}/data/forecasting.csv')
                feature_cols = [f'Feature {id + 1}' for id in range(30)]
                target_cols = [f'Target {id + 1}' for id in range(6)]
                lagrangian_multiplier = 1000
                features_df = full_df[feature_cols]
                targets_df = full_df[target_cols]
            elif dataset == 'newclass':
                optimization_problem = formulate_and_solve_lp_new_class_data
                feature_cols = ['gender', 'race/ethnicity', 'parental level of education',
                                'lunch', 'test preparation course']
                target_cols = ['math score', 'reading score', 'writing score']
                lagrangian_multiplier = 500
                full_df = pd.read_csv(f'{base_folder}/data/constrained_exams.csv')

                features_df = full_df[feature_cols]
                targets_df = full_df[target_cols]
            else:
                optimization_problem = formulate_and_solve_lp_cars_data
                full_df = pd.read_csv(f'{base_folder}/data/insurance_evaluation_data.csv').drop(columns=['INDEX']).dropna()[:500]

                target_cols = ['TARGET_FLAG', 'TARGET_AMT']
                targets_df = full_df[target_cols]
                lagrangian_multiplier = 20000000
                feature_cols = ['KIDSDRIV', 'AGE', 'HOMEKIDS', 'YOJ', 'INCOME', 'HOME_VAL', 'TRAVTIME',
                                'BLUEBOOK', 'TIF', 'OLDCLAIM', 'CLM_FREQ', 'MVR_PTS', 'CAR_AGE']
                features_df = full_df[feature_cols]
                currency_cols = features_df.select_dtypes('object').columns
                features_df.loc[:, currency_cols] = features_df[currency_cols].replace('[\$,]', '', regex=True).astype(float)

            kf = KFold(n_splits=number_of_folds, shuffle=True, random_state=0)
            for cv_fold, (tr_idx, te_idx) in enumerate(kf.split(features_df)):
                print(f'Fold: {cv_fold}')

                # One-hot encoding for categorical features
                if dataset in ['newclass']:
                    features_df = pd.get_dummies(features_df, columns=features_df.columns, drop_first=True, dtype=int)

                X_train, y_train = features_df.iloc[tr_idx], targets_df.iloc[tr_idx]
                X_test, y_test = features_df.iloc[te_idx], targets_df.iloc[te_idx]

                regressor = DecisionTreeRegressor(random_state=20, min_samples_leaf=ocrt_min_samples_leaf,
                                                  min_samples_split=ocrt_min_samples_split, max_depth=ocrt_depth)
                start = time.time()
                regressor.fit(X_train, y_train)
                end = time.time()
                y_pred_sklearn = regressor.predict(X_test)
                dt_mse = mean_squared_error(y_test, y_pred_sklearn)
                print(f'DT MSE: {dt_mse}')
                dt_nof_infeasibilities = calculate_number_of_infeasibilities(y_pred_sklearn, X_test, dataset, 'DT',
                                                                             regressor.get_depth(), target_cols)

                perf_df = pd.concat([perf_df, pd.DataFrame({'data': dataset, 'fold': cv_fold, 'depth': ocrt_depth,
                                                            'min_samples_leaf': ocrt_min_samples_leaf,
                                                            'min_samples_split': ocrt_min_samples_split,
                                                            'prediction_method': 'sklearn',
                                                            'prediction_method_leaf': 'sklearn',
                                                            'evaluation_method': 'sklearn',
                                                            'mse': dt_mse, 'nof_infeasibilities': dt_nof_infeasibilities,
                                                            'training_duration': end - start}, index=[0])])

                for evaluation_method in evaluation_method_list:
                    for prediction_method in prediction_method_list:
                        for prediction_method_leaf in prediction_method_leaf_list:
                            print("==============")
                            print(f'Evaluation: {evaluation_method}')
                            print(f'Split Prediction: {prediction_method}')
                            print(f'Leaf Prediction: {prediction_method_leaf}')

                            nof_infeasibilities_method = lambda y, x: calculate_number_of_infeasibilities(y, x, dataset,
                                                                                      'OCRT', ocrt_depth, target_cols, verbose)
                            split_criteria = lambda y, x, nof_infeasibilities_method: split_criteria_with_methods(y, x, nof_infeasibilities_method,
                                    lagrangian_multiplier, prediction_method, evaluation_method, optimization_problem, verbose)
                            leaf_prediction_method = lambda y, x, nof_infeasibilities_method: split_criteria_with_methods(y, x, nof_infeasibilities_method,
                                    lagrangian_multiplier, prediction_method_leaf, evaluation_method, optimization_problem, verbose)

                            tree = OCDT(max_depth=ocrt_depth, min_samples_leaf=ocrt_min_samples_leaf, min_samples_split=ocrt_min_samples_split,
                                         split_criteria=split_criteria, leaf_prediction_method=leaf_prediction_method,
                                         nof_infeasibilities_method=nof_infeasibilities_method, verbose=verbose)
                            tree.fit(X_train, y_train)
                            y_pred = tree.predict(X_test)
                            ocrt_mse = mean_squared_error(y_test, y_pred)
                            print(f'OCRT MSE: {ocrt_mse}')
                            ocrt_nof_infeasibilities = calculate_number_of_infeasibilities(y_pred, X_test, dataset, 'OCRT', ocrt_depth, target_cols)

                            perf_df = pd.concat([perf_df, pd.DataFrame({'data': dataset, 'fold': cv_fold, 'depth': ocrt_depth,
                                                                        'min_samples_leaf': ocrt_min_samples_leaf,
                                                                        'min_samples_split': ocrt_min_samples_split,
                                                                        'prediction_method': prediction_method,
                                                                        'prediction_method_leaf': prediction_method_leaf,
                                                                        'evaluation_method': evaluation_method,
                                                                        'mse': ocrt_mse, 'nof_infeasibilities': ocrt_nof_infeasibilities,
                                                                        'training_duration': tree.training_duration}, index=[0])])
                            perf_df.to_csv(f'data/perf_df_{dataset}_split_{prediction_method}_leaf_{prediction_method_leaf}_depth_{ocrt_depth}.csv', index=False)

    plot_results = False
    if plot_results:
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib import pyplot as plt

        # perf_df = pd.read_csv('data/latest_results/perf_df_all.csv')

        report_metric = 'mse'
        dataset = 'newclass'

        report_cols = [report_metric, 'training_duration']
        perf_df_dataset = perf_df[perf_df['data'] == dataset]

        perf_df_plot = perf_df_dataset.groupby(['prediction_method', 'prediction_method_leaf'])[report_cols].mean().reset_index()
        perf_df_plot.loc[perf_df_plot['prediction_method'] == 'mean', 'prediction_method'] = 'average'
        perf_df_plot.loc[perf_df_plot['prediction_method'] == 'lagrangian', 'prediction_method'] = 'relaxation'
        perf_df_plot.loc[perf_df_plot['prediction_method_leaf'] == 'lagrangian', 'prediction_method_leaf'] = 'relaxation'
        perf_df_plot = perf_df_plot.set_index(['prediction_method', 'prediction_method_leaf'])
        perf_df_plot['run'] = [f'S({x[0][0].upper()}) - L({x[1][0].upper()})' for x in perf_df_plot.index]

        perf_df_plot['mse'] = perf_df_plot['mse'] / perf_df_plot['mse'].max()
        perf_df_plot['training_duration'] = perf_df_plot['training_duration'] / perf_df_plot['training_duration'].max()

        fig, ax1 = plt.subplots(figsize=(16, 8))

        ax1 = perf_df_plot[report_metric].plot.bar()
        
        # ax.set_title(f"Dataset: {dataset.upper()}", weight='bold', fontsize=20) # \n Metric: {report_metric.upper()}")
        ax1.set_ylim((0, perf_df_plot[report_metric].max() * 1.1))
        ax1.set_xlabel("Method", weight='bold', labelpad=15, fontsize=22)
        ax1.set_ylabel("(Scaled) Mean Squared Error", weight='bold', labelpad=15, fontsize=22)
        ax1.set_xticklabels(perf_df_plot.run, color='black', fontsize=16)
        ax1.set_yticklabels([x/10 for x in range(0, 11, 2)], fontsize=16)
        ax1.tick_params(axis='x', labelrotation=0)

        # Axis formatting.
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_color('#DDDDDD')
        ax1.tick_params(bottom=False, left=False, length=0)
        ax1.set_axisbelow(True)
        ax1.yaxis.grid(True, color='#EEEEEE')
        ax1.xaxis.grid(False)

        rects = ax1.patches
        perf_df_plot_nof_inf = perf_df_dataset.groupby(['prediction_method', 'prediction_method_leaf'])[
            'nof_infeasibilities'].mean()

        labels = ['MSE: ' + f'{round(perf_df_plot.mse.values[i], 3):.3f}' + '\n N. of Inf: ' + 
                  f'{round(perf_df_plot_nof_inf.values[i], 1):.1f}' for i in
                    range(len(perf_df_plot.mse.values))]

        bar_color = ax1.patches[0].get_facecolor()
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label, 
                    ha="center", va="bottom", color='blue', fontsize=15)
            
        ax2 = ax1.twinx()
        #Plot a line
        
        # ax1 = perf_df_plot[report_metric].plot.bar()
        # ax2.plot()
        # t = np.arange(0.01, 10.0, 0.01)
        # ax2.plot(t, np.sin(0.25*np.pi*t), 'r-')
        ax2.plot(perf_df_plot['training_duration'].values, 'r-', linewidth=4)

        # Make the y-axis label and tick labels match the line color.
        ax2.set_ylabel('(Scaled) Training Duration', color='black', weight='bold', labelpad=15, fontsize=22)

        # Axis formatting.
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_color('#DDDDDD')
        ax2.tick_params(bottom=False, left=False, length=0)
        ax2.yaxis.set_tick_params(labelsize=18)
        ax2.set_axisbelow(True)
        ax2.yaxis.grid(True, color='#EEEEEE')
        ax2.xaxis.grid(False)

        fig.tight_layout()

        plt.show()
