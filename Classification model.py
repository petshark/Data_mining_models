# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 22:25:28 2021

@author: carlo
"""

from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from math import sqrt
import numpy as np

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train, Y_train)
print('Accuracy of Logistic Regression classifier on training set: {:.2f}'
     .format(roc_auc_score(Y_train, logreg.predict_proba(X_train)[:, 1])))
print('Accuracy of Logistic Regression classifier on test set: {:.2f}'
     .format(roc_auc_score(Y_test, logreg.predict_proba(X_test)[:, 1])))


