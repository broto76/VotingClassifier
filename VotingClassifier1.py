# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:03:35 2020

@author: Anibrata and Anirban
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class VotingClassifier:
    
    X_Train = 0
    Y_Train = 0
    X_Test = 0
    Y_Test = 0
    
    _testSize = 0.2
    
    classifier_KNN = None
    classifier_SVM = None
    classifier_RFC = None
    
    Y_Predict_KNN = 0
    Y_Predict_SVM = 0
    Y_Predict_RFC = 0
    
    accuracy_KNN = 0
    accuracy_SVM = 0
    accuracy_RFC = 0
    
    Weight_KNN = 0
    Weight_SVM = 0
    Weight_RFC = 0
    
    KNN_Data = 0
    SVM_Data = 0
    RFC_Data = 0
    
    def __init__(self):
        print("__init__")
        # self._testSize = test_size
        
        self.classifier_KNN = KNeighborsClassifier()
        # TODO
        self.classifier_SVM = SVC(kernel="linear")
        self.classifier_RFC = RandomForestClassifier(criterion='entropy')
        
    
    def pre_prepossing(self, features):
        scaler = MinMaxScaler(feature_range=(0,1))
        features = scaler.fit_transform(features)
        return features
    
    def splitData(self, features, result):
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = train_test_split(features, result, test_size=self._testSize)
        
    def internalClassifierFit(self):
        self.classifier_KNN.fit(self.X_Train, self.Y_Train)
        self.classifier_SVM.fit(self.X_Train, self.Y_Train)
        self.classifier_RFC.fit(self.X_Train, self.Y_Train)
        
    def internalClassifierPredict(self):
        self.Y_Predict_KNN = self.classifier_KNN.predict(self.X_Test)
        self.Y_Predict_SVM = self.classifier_SVM.predict(self.X_Test)
        self.Y_Predict_RFC = self.classifier_RFC.predict(self.X_Test)
        
    def calculateClassifierWeights(self):
        self.accuracy_KNN = accuracy_score(self.Y_Test, self.Y_Predict_KNN, normalize=False)
        self.accuracy_SVM = accuracy_score(self.Y_Test, self.Y_Predict_SVM, normalize=False)
        self.accuracy_RFC = accuracy_score(self.Y_Test, self.Y_Predict_RFC, normalize=False)
        
        self.Weight_KNN = (self.accuracy_KNN) / (self.accuracy_KNN + self.accuracy_SVM + self.accuracy_RFC)
        self.Weight_SVM = (self.accuracy_SVM) / (self.accuracy_KNN + self.accuracy_SVM + self.accuracy_RFC)
        self.Weight_RFC = (self.accuracy_RFC) / (self.accuracy_KNN + self.accuracy_SVM + self.accuracy_RFC)
    
    def fit_func(self, features, result):
        
        # Preprocess the data between 0 and 1
        features = self.pre_prepossing(features)
        
        # Split the data to training and test set
        self.splitData(features, result)
        self.internalClassifierFit()
        self.internalClassifierPredict()
        self.calculateClassifierWeights()
        
        
    def predict_func(self, test_features):
        test_features = self.pre_prepossing(test_features)
        self.KNN_Data = self.classifier_KNN.predict(test_features)
        self.SVM_Data = self.classifier_SVM.predict(test_features)
        self.RFC_Data = self.classifier_RFC.predict(test_features)
        
        self.result_internal = (self.Weight_KNN*self.KNN_Data) + (self.Weight_SVM*self.SVM_Data) + (self.Weight_RFC * self.RFC_Data)
        
        for i in range (0, self.result_internal.size):
            if (self.result_internal[i] == 0 or self.result_internal[i] == 1):
                continue
            if self.result_internal[i] > 0.5:
                self.result_internal[i] = 1
                print (i)
            else:
                self.result_internal[i] = 0
                print (i)
                
        return self.result_internal        
        


classifier = VotingClassifier()
dataset = pd.read_csv("BreastCancer.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 10].values
Y = Y/2 -1
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2)
classifier.fit_func(X_Train, Y_Train)
result = classifier.predict_func(X_Test)
accuracy = accuracy_score(Y_Test, result, normalize=False)




