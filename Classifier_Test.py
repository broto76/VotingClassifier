# -*- coding: utf-8 -*-
"""
Created on Sat May 16 12:49:38 2020

@author: KIIT
"""
from VotingClassifier1 import VotingClassifier
 
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

dataset=pd.read_csv(r'C:\Users\KIIT\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Anaconda3 (64-bit)\BreastCancer.csv')


X=dataset.iloc[:,0:9].values
Y=dataset.iloc[:,10].values
Y = Y/2 -1
from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=.4,random_state=0)

classi1=VotingClassifier();
classi1.fit_func(X_Train, Y_Train)
result = classi1.predict_func(X_Test)
accuracy = accuracy_score(Y_Test, result, normalize=False)
