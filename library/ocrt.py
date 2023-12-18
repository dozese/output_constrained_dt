# Based on the work by baydoganm/mtTrees

import os
import pandas as pd
import numpy as np
import random

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

from gurobipy import Model, GRB

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

class TreeForecast:
    """
    Predictive Clustering Tree.

    Args:
        target_type (str): Type of target variable ('multi', 'single', 'pca', etc.).
        max_features (int): Maximum number of features to consider.
        max_target (int): Maximum number of target variables.
        max_depth (int): Maximum depth of the tree.
        min_samples_leaf (int): Minimum number of samples in a leaf node.
        min_samples_split (int): Minimum number of samples to split a node.
        split_style (str): Splitting style.
        target_diff (bool): Whether to use target differences.
        lambda_decay (float): Lambda decay parameter.
        obj_weights (numpy.ndarray): Object weights.
        verbose (bool): Whether to print verbose information.

    Attributes:
        max_features (int): Maximum number of features to consider.
        max_target (int): Maximum number of target variables.
        max_depth (int): Maximum depth of the tree.
        min_samples_leaf (int): Minimum number of samples in a leaf node.
        min_samples_split (int): Minimum number of samples to split a node.
        target_type (str): Type of target variable ('multi', 'single', 'pca', etc.).
        split_style (str): Splitting style (e.g. 'custom')
        target_diff (bool): Whether to use target differences.
        lambda_decay (float): Lambda decay parameter.
        is_weighted (bool): Indicates if weights are used.
        obj_weights (numpy.ndarray): Object weights.
        verbose (bool): Whether to print verbose information.
        Tree (Node): Root node of the predictive clustering tree.

    Methods:
        buildDT(features, labels, node):
            Build the predictive clustering tree.

        fit(features, labels):
            Fit the predictive clustering tree to the data.

        nodePredictions(y):
            Calculate predictions for a node.

        selectTarget(target_type, labels):
            Select the target variable.

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
    def __init__(self,
                 target_type = 'multi',
                 max_features = None,
                 max_target = 1,
                 max_depth = 5,
                 min_samples_leaf = 1,
                 min_samples_split = 2,
                 obj_weights = None,
                 split_style = None,
                 target_diff = False,
                 lambda_decay = None,
                 verbose = False
                 ):
        self.max_features = max_features
        self.max_target = max_target
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.obj_weights = obj_weights
        self.target_type = target_type
        self.split_style = split_style
        self.target_diff = target_diff
        self.lambda_decay = lambda_decay
        self.is_weighted = False
        if lambda_decay is not None:
            self.is_weighted = True
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
        node.prediction = self.nodePredictions(labels.to_numpy(), 'medoid')
        node.count = labels.shape[0]
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return

        if features.shape[0] < self.min_samples_split:
            node.is_terminal = True
            return

        if self.target_type == "multi":
            current_label = range(labels.shape[1])
            target = labels
        elif self.target_type == "single":
            current_label = 0
            target = labels
        else:
            current_label = self.selectTarget(self.target_type, labels)

            if self.target_type == "pca":
                target = pd.DataFrame(current_label)
                current_label = 0
            elif self.target_type == "pca-random":
                target = pd.DataFrame(current_label)
                current_label = random.randint(0, target.shape[1]-1)
            elif self.target_type == "mean":
                target = pd.DataFrame(current_label)
                current_label = 0
            else:
                target = labels

        if self.split_style == 'custom':
            split_info, split_gain, n_cuts = self.calcBestSplitCustom(features, target)
        elif self.split_style == 'custom_lp':
            split_info, split_gain, n_cuts = self.calcBestSplitCustomLP(features, target)
        else:
            splitCol, thresh = self.calcBestSplit(features, target, current_label)

        if self.split_style in ['custom', 'custom_lp']:
            if n_cuts == 0:
                node.is_terminal = True
                return

            min_max_scaler = preprocessing.MinMaxScaler()
            split_gain_scaled = min_max_scaler.fit_transform(split_gain)
            split_gain_scaled_total = np.dot(split_gain_scaled, self.obj_weights)
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
        self.Tree = Node()
        self.Tree.depth = 0
        self.Tree.id = 1
        self.buildDT(features, labels, self.Tree)

    def predict(self, features, depth):
        '''
        Returns the labels for each X
        '''
        predictions = [self.predictSample(features.loc[i], depth, self.Tree) for i in features.index]
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

        if features[node.column] > node.threshold:
            predicted = self.predictSample(features, depth, node.right)
        else:
            predicted = self.predictSample(features, depth, node.left)

        return predicted

    def nodePredictions(self, y, type='medoid'):
        """
        Calculate predictions for a node as the mean.

        Args:
            y (numpy.ndarray): The labels or target variables for a node.

        Returns:
            predictions (numpy.ndarray): Predictions for the node, which represent the mean of target variables.
        """
        if type == 'medoid':
            predictions = y[self.find_medoids(y)]
        elif type == 'lp':
            predictions = self.formulate_and_solve_lp(y)
        else:
            predictions = np.asarray(y.mean(axis=0))

        return predictions


    def find_medoids(self, yi):
        """
        Find the index of medoid.

        Args:
            yi (numpy.ndarray): Matrix containing data.

        Returns:
            int: Index of the medoid
        """

        return np.argmin(yi.mean(axis=1))

    def selectTarget(self, target_type, labels):
        """
        Select the target variable.

        Args:
            target_type (str): Type of target variable ('random', 'max_var', 'max_cor', 'max_inv_cor', 'pca', 'pca-random', 'mean').
            labels (pandas.DataFrame): The labels or target variables corresponding to the features.

        Returns:
            int or numpy.ndarray: The selected target variable or index.
        """
        if target_type == 'random':
            return(random.randint(0, labels.shape[1]-1))
        elif target_type == 'max_var':
            target_var = labels.var(axis=0)
            max_index = np.where(target_var.index == target_var.idxmax())
            return max_index[0]
        elif target_type == 'max_cor':
            cor_mat = labels.corr()
            avg = cor_mat.mean(axis=1)
            max_index = np.where(avg.index == avg.idxmax())
            return max_index[0]
        elif target_type == 'max_inv_cor':
            cor_mat = labels.corr().to_numpy()
            inv_cor = np.abs(np.linalg.inv(cor_mat))
            avg = inv_cor.mean(axis=1)
            max_index = np.where(avg == max(avg))
            return max_index[0]
        elif target_type == 'pca':
            pca = PCA(n_components=1).fit_transform(labels)
            return pca
        elif target_type == 'pca-random':
            if labels.shape[1] < labels.shape[0]:
                pca = PCA(n_components=labels.shape[1]).fit_transform(labels)
            else:
                pca = PCA(n_components=labels.shape[0]).fit_transform(labels)
            return pca
        elif target_type == 'mean':
            return labels.mean(axis=1)

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

        if features[node.column] > node.threshold:
            predicted = self.applySample(features, depth, node.right)
        else:
            predicted = self.applySample(features, depth, node.left)

        return predicted

    def apply(self, features, depth):
        """
        Returns the node ID for each input object.

        Args:
            features (pandas.DataFrame): The input features for multiple objects.
            depth (int): The depth at which to stop traversing the tree.

        Returns:
            predicted_ids (numpy.ndarray): The predicted node IDs for each input object.
        """
        predicted_ids = [self.applySample(features.loc[i], depth, self.Tree) for i in features.index]
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

    def calcBestSplit(self, features, labels, current_label):
        """
        Calculates the best split based on features and labels.

        Args:
            features (pandas.DataFrame): The input features for a single object.
            labels (pandas.DataFrame): The labels or target variables corresponding to the features.
            current_label (int): The index of the current target variable.

        Returns:
            split_col (int): The column index for the best split.
            threshold (float): The threshold for the best split.
        """
        bdc = DecisionTreeRegressor(
            random_state=0,
            criterion="squared_error",
            max_features=self.max_features,
            max_depth=1,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )
        bdc.fit(features, labels.iloc[:, current_label])

        threshold = bdc.tree_.threshold[0]
        split_col = bdc.tree_.feature[0]

        return split_col, threshold

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

                left_instance_count = left_yi.shape[0]
                right_instance_count = right_yi.shape[0]

                left_prediction = self.nodePredictions(left_yi, type='medoid')
                right_prediction = self.nodePredictions(right_yi, type='medoid')

                left_perf = ((left_yi - left_prediction) ** 2).mean()
                right_perf = ((right_yi - right_prediction) ** 2).mean()
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

    def calcBestSplitCustomLP(self, features, labels):
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

                left_instance_count = left_yi.shape[0]
                right_instance_count = right_yi.shape[0]

                left_prediction = self.nodePredictions(left_yi, type='lp')
                right_prediction = self.nodePredictions(right_yi, type='lp')

                left_perf = ((left_yi - left_prediction) ** 2).mean()
                right_perf = ((right_yi - right_prediction) ** 2).mean()
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

    def formulate_and_solve_lp(self, y):
        print(y)

        num_targets = y.shape[1]

        import gurobipy as gp
        from gurobipy import GRB

        # Create model
        model = gp.Model("Minimize SSE")

        # Create decision variables
        predictions = model.addVars(num_targets, lb=0, ub=100, name="y")
        binary_vars = model.addVars(num_targets, vtype=GRB.BINARY, name="z")

        # Create objective function
        sse = gp.quicksum((predictions[i] - y[i][j]) * (predictions[i] - y[i][j]) for i in range(num_targets) for j in range(len(y[i])))
        model.setObjective(sse, GRB.MINIMIZE)

        # Create constraints
        model.addConstr(binary_vars.sum() <= 1, "one_prediction_constraint")
        for i in range(num_targets):
            model.addConstr(predictions[i] <= 100 * binary_vars[i], f"z_relationship_{i}")

        # Solve the problem
        model.optimize()

        if self.verbose:
            print("Optimal Solution:")
            for i in range(num_targets):
                print(f"Target {i + 1}: Prediction = {predictions[i].X}, Indicator = {binary_vars[i].X}")
            print("Objective (Sum of Squared Errors):", model.objVal)

        preds = np.array([predictions[i].X for i in range(num_targets)])

        return preds




if __name__ == '__main__':
    custom_dt_depth = 15
    custom_dt_min_samples_split = 20
    custom_dt_min_samples_leaf = 10
    custom_dt_obj_weights = [1.0]
    test_set_ratio = 0.2
    target_size = 3
    feature_size = 2

    base_folder = os.getcwd()
    target_cols = [f'Course{id + 1}' for id in range(target_size)]
    feature_cols = [f'Feature{id + 1}' for id in range(feature_size)]

    if target_size == 3:
        full_df = pd.read_csv(f'{base_folder}/data/full_df_size_300_targets_3.csv')

        features_df = full_df[feature_cols]
        targets_df = full_df[target_cols]
    else:
        feature_cols = ['EnrolledElectiveBefore', 'GradeAvgFromPreviousElective',
                        'Grade', 'Major', 'Class', 'GradePerm']

        full_df = pd.read_csv(f'{base_folder}/data/full_df.csv')

        features_df = full_df[feature_cols]
        targets_df = full_df[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(features_df, targets_df,
                                                        test_size=test_set_ratio, shuffle=True)

    tree = TreeForecast(target_type='multi', max_depth=custom_dt_depth, max_features=None,
                        min_samples_leaf=custom_dt_min_samples_leaf, split_style='custom_lp',
                        min_samples_split=custom_dt_min_samples_split, target_diff=False,
                        obj_weights=custom_dt_obj_weights, lambda_decay=0.5, verbose=False)

    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test, custom_dt_depth)
    print('\nOCRT Accuracy: ', mean_squared_error(y_test, y_pred))
    cumsums = np.array([sum(y_pred[i] > 0.0001) for i in range(len(y_pred))])
    print('Number of infeasible predictions for OCRT: ', np.sum(cumsums >= 3))

    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)
    y_pred_sklearn = regressor.predict(X_test)
    print('\nDT Accuracy: ', mean_squared_error(y_test, y_pred_sklearn))
    cumsums_sklearn = np.array([sum(y_pred_sklearn[i] > 0.0001) for i in range(len(y_pred_sklearn))])
    print('Number of infeasible predictions for DT: ', np.sum(cumsums_sklearn >= 3))

    test_leaves = tree.apply(X_test, custom_dt_depth)
    y_test_df = pd.DataFrame.from_dict({'instance_id': X_test.index.values, 'leaf_id': test_leaves}).set_index('instance_id')
    y_test_df = pd.merge(y_test_df, y_test, left_index=True, right_index=True).sort_values('leaf_id')