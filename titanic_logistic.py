#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 11:25:29 2018

@author: meg116

Let's look at the variables:
    
    Pclass: categorical (1, 2, 3). Unclear whether ordinal or not.
    Sex: categorical (male, female)
    Age: numerical
    SibSp: categorical (0, 1, 2, 3, 4, 5, 8)
    Parch: categorical (0, 1, 2, 3, 4, 5, 6)
    Fare: numerical
    Cabin: categorical, too many to be useful. Will try just using first letter
    Embarked: categorical (S, C, Q, nan)
    
Labels:
    0    549
    1    342
    
Pclass 1 seems to have a much higher proportion of survivers than 2, which in
turn has more survivers than 3.

Dividing by Parch also shows very different surivival rates (e.g. classes 4
and 6 has no survivers, while class 3 had a 60% survival rate).

74% females survived, compared to 19% of men.

Cabin initial also does a somewhat good job at stratisfying the survivors.

SibSp also divides well (e.g., 5 and 8 have no suvivors, compared with 50% in
1).

Most passengers are aged between 20 and 30.             

"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np

# Import, set index.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature engineering.
train = train.set_index(['PassengerId'])
test = test.set_index(['PassengerId'])
train['Cabin'] = [str(w)[0] for w in train['Cabin'].values]
test['Cabin'] = [str(w)[0] for w in test['Cabin'].values]
train = train.drop(['Ticket', 'Name'], axis=1)
test = test.drop(['Ticket', 'Name'], axis=1)

# Explore the data.
train.groupby('Survived').mean()
train.groupby('Pclass').mean()
train.groupby('Parch').mean()
train.groupby('Sex').mean()
train.groupby('Cabin').mean()
train.groupby('SibSp').mean()

# Visualise the data.
#pd.crosstab(train['Sex'], train['Survived']).plot(kind='bar')
#train['Age'].hist()

# Make dummy variables for train set.
cat_vars = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked']
for var in cat_vars:
    cat_list = pd.get_dummies(train[var], prefix=var)
    train = train.join(cat_list)
data_vars = train.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]
train = train.loc[:, to_keep]

# Make dummy variables for test set.
for var in cat_vars:
    cat_list = pd.get_dummies(test[var], prefix=var)
    test = test.join(cat_list)
data_vars = test.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]
test = test.loc[:, to_keep]

# Add columns not present in both sets.
train['Parch_9']=0
test['Cabin_T'] = 0

# Make lists of final variables/targets.
final_vars = train.columns.values.tolist()
y = ['Survived']
X= [i for i in final_vars if i not in y]

# Remove missing values from train set
train = train[~train.isnull().any(axis=1)]

# Choose best features (just for practice - probably best to keep them all).
# clf = LogisticRegression()
# rfe = RFE(clf, 5)
# rfe.fit(train[X], train[y])
# top_vars = rfe.support_

# Find the best regularisation parameter
param_grid = {'clf__C': [.01, .1, 1, 10, 100, 1000]}
pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(train[X], np.ravel(train[y]))
best_C = grid.best_params_['clf__C']

# Estimate performance on unseen data.
print(grid.best_score_)

# Impute mean for missing values in test set.
test['Age'] = test['Age'].fillna(test['Age'].mean(skipna=True))
test['Fare'] = test['Fare'].fillna(test['Fare'].mean(skipna=True))

# Train on the whole train set. Use this set to fit the scaler, use same scaler
# on test set.
scaler = StandardScaler()
scaler.fit(train[X])
train[X] = scaler.transform(train[X])
test[X] = scaler.transform(test[X])

# Final predictions.
clf = LogisticRegression(C=best_C)
clf.fit(train[X], np.ravel(train[y]))
test['Predictions']=clf.predict(test[X])
#print(test['Predictions'])
#test['Predictions'].to_csv('//Users//meg116//Documents//Kaggle//preds.csv',
#    header=['Survived'])