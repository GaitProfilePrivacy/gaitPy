#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:23:50 2023

@author: ozgesurer
"""
import pyreadr
import pandas as pd
import numpy as np
from pyts.approximation import PiecewiseAggregateApproximation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

result = pyreadr.read_r('../data/gp.rds')
df = result[None]
sample_subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '14', '15']

X = []
y = []
index = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51]
for s_id, s in enumerate(sample_subjects):
    df_select = df.loc[df['subject_num'].isin([s])]
    unique_cyc = np.unique(df_select['gait_cycle'])
    for cyc in unique_cyc:
        df_cyc = df_select.loc[df_select['gait_cycle'].isin([cyc])]
        if len(df_cyc) > 51:
            y.append(s_id)
            X.append(df_cyc.iloc[index]['acc_magnitude'])
        #paa = PiecewiseAggregateApproximation(window_size=5) 
        #paa.transform(np.array(df_cyc['acc_magnitude']).reshape(-1, 1).T)

X = np.array(X)
y = np.array(y)

# Split the data into test and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Fit a logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

model.fit(X_train, y_train)
model.score(X_test, y_test)
cf_mat = confusion_matrix(y_test, model.predict(X_test))
print(classification_report(y_test, model.predict(X_test)))

# Fit a support vector machine
model_svm = svm.SVC().fit(X_train, y_train)

predict_svm = model_svm.predict(X_test)
print(classification_report(y_test, predict_svm))