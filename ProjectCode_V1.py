#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:42:37 2018

@author: siddhantamidwar
"""

import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


dataFrame = pd.read_csv("/Users/siddhantamidwar/Downloads/Project BE Codes/testdata.csv", names = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])

X = np.array(dataFrame.drop(['Date','Close'],1))
y = np.array(dataFrame['Close'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
reg = RandomForestRegressor(n_estimators = 1000, random_state = 42)
reg.fit(X_train, y_train)