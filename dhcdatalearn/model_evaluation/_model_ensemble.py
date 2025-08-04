# -*- coding: utf-8 -*-
# Author: wufatian

import pandas as pd
from sklearn.model_selection import StratifiedKFold

class ModelEnsemble:

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. 

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def cal_predict(self, clf, X_test, y_test):
        """
        计算预测概率
        """
        clf.fit(self.X_train, self.y_train)
        # y_test_preds = clf.predict(X_test)
        y_test_predprob = clf.predict_proba(X_test)[:,1]
        return y_test_predprob
    
    def model_blending(self, names, sampling_methods, X_test, y_test):
        """
        模型混合打分
        """
        res = pd.DataFrame()
        for (name, method) in zip(names, sampling_methods):
            res[name] = self.cal_predict(method, X_test, y_test)
        res['blend'] = res.mean(axis=1)
        return res['blend']

    def model_stacking(self, clfs, X_test, y_test):
        """
        模型堆叠打分
        """
        X_train, X_test, y_train, y_test = self.X_train.values, X_test.values, self.y_train.values, y_test.values
        X_train_stack  = np.zeros((X_train.shape[0], len(clfs)))
        X_test_stack = np.zeros((X_test.shape[0], len(clfs)))
        n_folds = 5
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
        for i,clf in enumerate(clfs):
            X_stack_test_n = np.zeros((X_test.shape[0], n_folds))
            for j, (train_index, test_index) in enumerate(skf.split(X_train, self.y_train)):
                        tr_x = X_train[train_index]
                        tr_y = y_train[train_index]
                        clf.fit(tr_x, tr_y)
                        #生成stacking训练数据集
                        X_train_stack[test_index, i] = clf.predict_proba(X_train[test_index])[:,1]
                        X_stack_test_n[:,j] = clf.predict_proba(X_test)[:,1]
            #生成stacking测试数据集
            X_test_stack[:,i] = X_stack_test_n.mean(axis=1)
            clf_second = LogisticRegression(solver="lbfgs")
            clf_second.fit(X_train_stack, y_train)
            pred = clf_second.predict_proba(X_test_stack)[:,1]
        return pred

