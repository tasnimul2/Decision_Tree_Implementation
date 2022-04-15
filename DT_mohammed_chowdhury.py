

from dataclasses import replace
from ensurepip import bootstrap
import pandas as pd
import numpy as np
import seaborn as sns
#https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb

from sklearn import datasets
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split
from collections import Counter

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

    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        '''Implement the code here, this is a overloaded method that will handle if a dataframe or a pd series 
        is passed in. It converts it into a numpy array so the orginal _fit method functions properly.
        Before converting  y into a numpy array, it checks to see if it is has categorical values. If it does, then 
        change the values to 0's and 1's.
        '''
        # call the _fit method
        new_X = X.to_numpy()
        does_have_catigorical_values_in_list = [type(x) == int for x in y]
        if(not any(does_have_catigorical_values_in_list)):
            y = y.apply(lambda x: 0 if x == 'M' else 1)
        
        
        if not isinstance(y,np.ndarray):
            new_y = y.to_numpy()
        else:
            new_y = y
        self._fit(new_X,new_y)
        print("Done fitting Decision Tree")

    
    def predict(self, X: pd.DataFrame):
        ''' 
        Implement the code here
        this is an overloaded method of the original _predict() 
        the purpose of this method is to handle the edge case of when a pd Dataframe is passed in instead of a 
        numpy array.
        
        X.to_numpy() creates a list of lists where each inner list represents a row in the original data frame, and each index in that list is a column.
        Ex. [[col1,col2,col3],[col1,col2,col3],[col1,col2,col3]] is the dataframe as a numpy array 
        where [col1,col2,col3] is a row in the dataframe and colN is a column in the df
        '''
        new_X = X.to_numpy()
        predictions = [self._traverse_tree(x, self.root) for x in new_X]
        return np.array(predictions)    
       
           
    def _fit(self, X, y):
        '''Derives the root of the tree by calling the build_tree method. The root has children nodes that the predict method will utilize'''
        self.root = self._build_tree(X, y)
        
    def _predict(self, X):
        '''For every row in the dataframe X, pass each row inside the traverse_tree method'''
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)    
        
    def _is_finished(self, depth):
        ''' This method checks if the tree's depth did not exceed the maximum. This is to about infinite depth first traversal'''
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
        '''
        X.shape gets the number of rows and columns in the dataframe, which represents samples and features respectively. 
        n_class_labels checks the number of unique labels there are. In this case there are 2 labels, "M and B" or "0 and 1". 
        Then there is a check to see if the tree has hit maximum depth using is_finished() method. 
        We then return the node with the value of the most common label. (answers : is M or B the most common?)
        If it is not finished building then:
        rnd_feats is a list of column numbers in the dataframe that represent different features.
        best_split method is then called that returns the feature and threshold value of the split with the highest gini impurity / entorpy score.
        Then recursively, pre-order traversal is done to build the children nodes. 
        Finaly return the root node.
        
        '''
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
    

    def _gini(self, y):
        '''
        Implement the code here
        if y will either be a numpy array or pandas series. 
        if it is a pandas series, then convert it into a np array (pd.values). Then do the computation
        gini = 1 - (prob of yes)^2 - (prob of no)^2
        gint = 1 - (yes/yes+no)^2 - (no/yes+no)^2
        '''
        
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

    def _entropy(self, y):
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
        if isinstance(y,np.ndarray):
            y = pd.Series(y)
            
        does_have_catigorical_values_in_list = [type(x) == int for x in y]
        if(not any(does_have_catigorical_values_in_list)):
            y = y.apply(lambda x: 0 if x == 'M' else 1)
        # ^ end of my added code ^ 
        
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
        
        
    def _create_split(self, X, thresh):
        ''' Creates a left chid and right child node index for buid_tree method to utilze'''
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    
    def _information_gain(self, X, y, thresh):
        '''
        Implement the code here
        needed fix the code so it can switch between the two criterion: gini and entropy 
        if criterion is entropy then run given code
        if criterion is gini then use gini to find parent_loss and child loss
        '''
        
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
        1) the build_tree method passes in X train and y train values along with features (ie. a list of column numbers)
        into this method. 
        2) Then use a loop to iterate through the column index and use X[:,feat] syntax to denote "get all elements in column 'feat' "
        this returns a list with all column data located in column number 'feat' and stores it inside X_feat variable. 
        3)then use np.unique to get all the unique values in X_feat
        4) iterate though the list of unique values inside the columns  and get a score from the information gain method, based on the gini impurity.
        5) keep running step 4 until all unique values in the list are tested. As this happens, the highest score is stored for the current split.
        6) return the feature and threshold value of the split with the highest score.
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
        called upon by the predict method that passes in a list of integer value that represent a row in the dataframe. 
        len(x) = n where n is number of columns in the dataframe.
        
        this method traverses throug the tree that was built by the fit() method. 
        1) if we are at a lead node, return the value of the current node. 
        2) check if the current nodes feature (ie. column in the dataframe) is less than the threshold. If it is, traverse to the left side of the tree 
        3) otherwise go to the rightside of the tree. 
        '''
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestModel(object):

    def __init__(self, n_estimators):
        # TODO:
        self.n_estimators = n_estimators
        self.d_trees = []
        # end TODO

    def fit(self, X: pd.DataFrame, y: pd.Series):
        '''
        depending on what the value of n_estimators is, a for loop will run to fit "n_estimators" number of decision trees and 
        store that tree in the d_trees = [] list. But, the trees are fitted with bootstrapped data (which means the original data was randomized)
        '''
        for i in range(self.n_estimators):
            currTree = DecisionTreeModel(max_depth=10)
            X_train,y_train = self.bootstrap_data(X,y)
            currTree.fit(X_train,y_train)
            self.d_trees.append(currTree)
        
        print("Random Forrest Fitted")
        
    
    def predict(self, X: pd.DataFrame):
        '''
        predictions_of_all_d_trees is a list of lists that holds the predictions for each decision tree.
        each index is a list of predictions for a single tree
        how swapaxes() works:
        assume you have a list of lists [[1,2,3],[4,5,6],[7,8,9]]
        swapaxes() will take the ith index of each sub list and put it in a single list. It will continue doing this 
        until no more elements are left. 
        result :
        [[1,4,7],[2,5,8],[3,6,9]]
        
        Then we loop through each linner list (tree_prediction) from the swapped_predictions_of_all_d_trees list derived from swapaxis()
        We utilize the counter object to count the values inside the tree_prediction list (ie. count how me 1's are in the list and how many 0's in the list)
        then we use the most_common() method to see which value, 0 or 1, occured the most. This method returns a list of tuple with this information
        denoted: [(value,number to times it occured in the list)]. Ex. [(1,3)] would mean the value 1 occured 3 times in the list.
        we use the notation counter.most_common(1)[0][0] to denote : 
        - most_common(1) means get the 1st most common value. it is was most_common(2) it would mean 2nd most common. Result [(1,3)]
        - most_common(1)[0] means get the 0th index inside the list of tuples. Result (1,3)
        - counter.most_common(1)[0][0] means get the 0th index in that tuble. Result : 1. 
        
        this resulting value is then appended to the predictions list , which is then subsequently returned after the loop finishes running through all inner lists
    
        '''
        predictions_of_all_d_trees = np.array([currTreeInList.predict(X) for currTreeInList in self.d_trees])
        swapped_predictions_of_all_d_trees = np.swapaxes(predictions_of_all_d_trees,0,1)
        
        predictions = []
        for tree_prediction in swapped_predictions_of_all_d_trees:
            counter = Counter(tree_prediction)
            most_common_value = counter.most_common(1)[0][0]
            predictions.append(most_common_value)
        
        return predictions
        
    
    def bootstrap_data(self,X,y):
        '''
        The sample() method selects "num_data_samples" rows randomly.  
        Once randomly selected data is created and stored in the bootstrapped_X and bootstrapped_y variables 
        return bootstrapped_X and bootstrapped_y
        '''
        num_data_samples = X.shape[0] #tells you number of rows in data frame
        bootstrapped_X = X.sample(num_data_samples)
        bootstrapped_y = y.sample(num_data_samples)
        return bootstrapped_X,bootstrapped_y

    

def accuracy_score(y_true, y_pred):
    '''
    checks to see if y_pred has any categorical values. If it does, it corrects it by mapping M and B to 0 and 1 respectively.
    calculates the accuracy of the model '''
    does_have_catigorical_values_in_list = [type(x) == int for x in y_true]
    if(not any(does_have_catigorical_values_in_list)):
        y_true = y_true.apply(lambda x: 0 if x == 'M' else 1)
        
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
    scores = {"precision":precision,"recall":recall,"f1_score":f1_score}
    result = pd.DataFrame([scores])
    return(result)

'''confusion_matrix() method takes in the actual test results and our models prediction.
It uses this data to find out the TP,FP,FN,TN values and store it inside a 2D numpy array.
returns 2D numpy array'''
def confusion_matrix(y_test, y_pred):
    # return the 2x2 matrix
    #[TP,FP],[FN,TN]
    #y_test is pandas series
    #y_pred is numpy.ndarray
    
    does_have_catigorical_values_in_list = [type(x) == int for x in y_test]
    if(not any(does_have_catigorical_values_in_list)):
        y_test = y_test.apply(lambda x: 0 if x == 'M' else 1)
        
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
    print(f"CLASSIFICATION REPORT for Decision Tree: \n {classification_report(y_test,y_pred)}")
    print("ACCURACY:", acc)
    
    # ------------------------
    rf = RandomForestModel(n_estimators=3)
    rf.fit(X_train,y_train)
    rf_y_pred = rf.predict(X_test)
    acc_rf = accuracy_score(y_test,rf_y_pred)
    print(f"CLASSIFICATION REPORT for Random Forrest: \n {classification_report(y_test,rf_y_pred)}")
    print("ACCURACY:", acc_rf)

if __name__ == "__main__":
    _test()
