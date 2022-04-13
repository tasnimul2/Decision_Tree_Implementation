

import pandas as pd
import numpy as np
import seaborn as sns
#https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb

from sklearn import datasets
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
    

class DecisionTreeModel:

    def __init__(self, max_depth=100, criterion = 'gini', min_samples_split=2, impurity_stopping_threshold = 1):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None

    '''Implement the code here, this is a overloaded method that will handle if a dataframe or a pd series 
    is passed in. It converts it into a numpy array so the orginal _fit method functions properly.'''
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # call the _fit method
        new_X = X.to_numpy()
        new_y = y.to_numpy()
        self._fit(new_X,new_y)
        print("Done fitting")

    '''Implement the code here'''
    def predict(self, X: pd.DataFrame):
        ''' 
        this is an overloaded method of the original _predict() 
        the purpose of this method is to handle the edge case of when a pd Dataframe is passed in instead of a 
        numpy array.
        '''
        new_X = X.to_numpy()
        predictions = [self._traverse_tree(x, self.root) for x in new_X]
        return np.array(predictions)    
       
           
    def _fit(self, X, y):
        self.root = self._build_tree(X, y)
        
    def _predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)    
        
    def _is_finished(self, depth):
        # TODO: for graduate students only, add another stopping criteria
        # modify the signature of the method if needed
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        # end TODO
        return False
    
    def _is_homogenous_enough(self):
        # TODO: for graduate students only
        result = False
        # end TODO
        return result
                              
    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)
    
    '''
    Implement the code here
    if y will either be a numpy array or pandas series. 
    if it is a pandas series, then convert it into a np array (pd.values). Then do the computation
    gini = 1 - (prob of yes)^2 - (prob of no)^2
    gint = 1 - (yes/yes+no)^2 - (no/yes+no)^2
    '''
    def _gini(self, y):
        if isinstance(y,np.ndarray):
            y_as_arr = y
        else :
            y_as_arr = y.values #y is a pd series, use pd.values to convert it to an np array

        num_yes = 0
        num_no = 0
        for i in y_as_arr:
            if(i == 1):
                num_yes = num_yes + 1
            else:
                num_no = num_no + 1
                
        gini = 1 - ((num_yes/(num_yes+ num_no))**2) - ((num_no/(num_yes+ num_no))**2)
        return gini
    
    '''
    Implement the code here
    problem : 
    the following won't work if y is not integer.
    Need make it work for the cases where y is a categorical variable 
    Solution:
    first I needed to check if y was a np array. If it was, then I need to convert it into a pandas series
    because .apply() method is only available in series not array.
    Once we know y is a series, take all of the elements in y, and check to see if any element is not an int (ie. categorical)
    if catigorical values are found, then i simply convert the M's to 0 and B's to 1. If we use a data set
    that is not breast_cancer.csv, then this logic would break since the check for 'M' is hard coded. 
    '''
    def _entropy(self, y):
        
        if isinstance(y,np.ndarray):
            y = pd.Series(y)
            
        does_have_catigorical_values_list = [type(x) == int for x in y]
        if(not any(does_have_catigorical_values_list)):
            y = y.apply(lambda x: 0 if x == 'M' else 1)
        # ^ end of my added code ^ 
        
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
        
    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    '''Implement the code here
    needed fix the code so it can switch between the two criterion: gini and entropy 
    if criterion is entropy then run given code
    if criterion is gini then use gini to find parent_loss and child loss'''
    def _information_gain(self, X, y, thresh):
        
        if self.criterion == 'gini':
            parent_loss = self._gini(y)
            left_idx, right_idx = self._create_split(X, thresh)
            n, n_left, n_right = len(y), len(left_idx), len(right_idx)

            if n_left == 0 or n_right == 0: 
                return 0
            
            child_loss = (n_left / n) * self._gini(y[left_idx]) + (n_right / n) * self._gini(y[right_idx])
            return parent_loss - child_loss
        else :
            parent_loss = self._entropy(y)
            left_idx, right_idx = self._create_split(X, thresh)
            n, n_left, n_right = len(y), len(left_idx), len(right_idx)

            if n_left == 0 or n_right == 0: 
                return 0
            
            child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
            return parent_loss - child_loss
            
    
       
    def _best_split(self, X, y, features):
        '''TODO: add comments here

        '''
        split = {'score':- 1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']
    
    def _traverse_tree(self, x, node):
        '''TODO: add some comments here
        '''
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestModel(object):

    def __init__(self, n_estimators):
        # TODO:
        pass
        # end TODO

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO:
        pass
        # end TODO
        pass


    def predict(self, X: pd.DataFrame):
        # TODO:
        pass
        # end TODO

    

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

'''The classfication_report method takes in the actual test results and our models predictions.
It utilizes the confusion matrix toderive the precision, recall, f1-score and stores and 
returns the data via a dictionary'''
def classification_report(y_test, y_pred):
    
    # calculate precision, recall, f1-score
    # returns a dictionary with the precision, recall, f1-score data.
    matrix = confusion_matrix(y_test,y_pred)
    TP = matrix[0][0]
    FP = matrix[0][1]
    FN = matrix[1][0]
    TN = matrix[1][1]
    
    precision = TP/(FP + TP)
    recall = TP/(FN + TP)
    f1_score = (2 * precision * recall)/(precision + recall)
    result = {"precision":precision,"recall":recall,"f1_score":f1_score}
    return(result)

'''confusion_matrix() method takes in the actual test results and our models prediction.
It uses this data to find out the TP,FP,FN,TN values and store it inside a 2D numpy array.
returns 2D numpy array'''
def confusion_matrix(y_test, y_pred):
    # return the 2x2 matrix
    #[TP,FP],[FN,TN]
    #y_test is pandas series
    #y_pred is numpy.ndarray
    
    TP,FP,FN,TN = 0,0,0,0
    y_test_arr = y_test.to_numpy()
    for i in range(len(y_test)):
        if y_test_arr[i] == 1 and y_pred[i] == 1:
            TP = TP + 1
        elif y_test_arr[i] == 0 and y_pred[i] == 1:
            FP = FP + 1
        elif y_test_arr[i] == 1 and y_pred[i] == 0:
            FN = FN + 1
        else:
            TN = TN + 1
    
    result = np.array([[TP, FP], [FN, TN]])
    return(result)


def _test():
    
    df = pd.read_csv('breast_cancer.csv')
    
    '''these are numpy arrays'''
    #X = df.drop(['diagnosis'], axis=1).to_numpy()
    #y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1).to_numpy()
    
    '''X is a pandas dataframe, y is pandas series. 
    Need to make DecisionTreeModel class work with this X and y '''
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1)
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTreeModel(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"CLASSIFICATION REPORT: {classification_report(y_test,y_pred)}")
    print("ACCURACY:", acc)

if __name__ == "__main__":
    _test()
