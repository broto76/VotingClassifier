# -*- coding: utf-8 -*-
"""
Created on Sat May 16 12:49:38 2020

@author: Anirban and Anibrata
"""
from VotingClassifier1 import VotingClassifier
 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


classifier = VotingClassifier()
path = "D:/Pyhton ML/VotingClassifier/Data/car_evaluation_processed.csv"
dataset = pd.read_csv(path)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 6].values
# Y = Y/2 -1
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2)
classifier.fit_func(X_Train, Y_Train)
# result = classifier.predict_on_weight(X_Test, 0.5)
result = classifier.predict_on_weighted_votes(X_Test)
classifier.showInternalAccuracy(Y_Test)
accuracy_Voting = accuracy_score(Y_Test, result, normalize=False)
