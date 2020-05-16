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
    
    
    # Not Used
    def internalClassifierPredict(self):
        self.Y_Predict_KNN = self.classifier_KNN.predict(self.X_Test)
        self.Y_Predict_SVM = self.classifier_SVM.predict(self.X_Test)
        self.Y_Predict_RFC = self.classifier_RFC.predict(self.X_Test)
    
        
    # Not Used
    def calculateClassifierWeights(self):
        self.accuracy_KNN = accuracy_score(self.Y_Test, self.Y_Predict_KNN, normalize=False)
        self.accuracy_SVM = accuracy_score(self.Y_Test, self.Y_Predict_SVM, normalize=False)
        self.accuracy_RFC = accuracy_score(self.Y_Test, self.Y_Predict_RFC, normalize=False)
        
        self.Weight_KNN = (self.accuracy_KNN) / (self.accuracy_KNN + self.accuracy_SVM + self.accuracy_RFC)
        self.Weight_SVM = (self.accuracy_SVM) / (self.accuracy_KNN + self.accuracy_SVM + self.accuracy_RFC)
        self.Weight_RFC = (self.accuracy_RFC) / (self.accuracy_KNN + self.accuracy_SVM + self.accuracy_RFC)
        
    def internalAssignAvgWeight(self):
        Y_Predict_KNN = self.classifier_KNN.predict(self.X_Test)
        Y_Predict_SVM = self.classifier_SVM.predict(self.X_Test)
        Y_Predict_RFC = self.classifier_RFC.predict(self.X_Test)
        
        accuracy_KNN = accuracy_score(self.Y_Test, Y_Predict_KNN, normalize=False)
        accuracy_SVM = accuracy_score(self.Y_Test, Y_Predict_SVM, normalize=False)
        accuracy_RFC = accuracy_score(self.Y_Test, Y_Predict_RFC, normalize=False)
        
        self.Weight_KNN = (accuracy_KNN) / (accuracy_KNN + accuracy_SVM + accuracy_RFC)
        self.Weight_SVM = (accuracy_SVM) / (accuracy_KNN + accuracy_SVM + accuracy_RFC)
        self.Weight_RFC = (accuracy_RFC) / (accuracy_KNN + accuracy_SVM + accuracy_RFC)
    
    def fit_func(self, features, result):
        
        # Preprocess the data between 0 and 1
        features = self.pre_prepossing(features)
        
        # Split the data to training and test set
        self.splitData(features, result)
        self.internalClassifierFit()
        # self.internalClassifierPredict()
        # self.calculateClassifierWeights()
        self.internalAssignAvgWeight()    
        
        
    def showInternalAccuracy(self,expected):
        self.accuracy_KNN_Test = accuracy_score(expected, self.KNN_Data, normalize=True)
        self.accuracy_SVM_Test = accuracy_score(expected, self.SVM_Data, normalize=True)
        self.accuracy_RFC_Test = accuracy_score(expected, self.RFC_Data, normalize=True)
        
        
    def predict_on_weight(self, test_features, threshold):
        test_features = self.pre_prepossing(test_features)
        self.KNN_Data = self.classifier_KNN.predict(test_features)
        self.SVM_Data = self.classifier_SVM.predict(test_features)
        self.RFC_Data = self.classifier_RFC.predict(test_features)
        
        self.result_internal = (self.Weight_KNN*self.KNN_Data) + (self.Weight_SVM*self.SVM_Data) + (self.Weight_RFC * self.RFC_Data)
        
        for i in range (0, self.result_internal.size):
            diff = self.result_internal[i] - int(self.result_internal[i])
            if (diff == 0):
                continue
            print (i)
            if (diff >= threshold):
                self.result_internal[i] = int(self.result_internal[i]) + 1
            else:
                self.result_internal[i] = int(self.result_internal[i])
                
        return self.result_internal  

        
    def predict_on_weighted_votes(self, test_features, threshold)
        


classifier = VotingClassifier()
path = "D:/Pyhton ML/VotingClassifier/Data/car_evaluation_processed.csv"
dataset = pd.read_csv(path)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 6].values
# Y = Y/2 -1
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2)
classifier.fit_func(X_Train, Y_Train)
result = classifier.predict_on_weight(X_Test, 0.5)
classifier.showInternalAccuracy(Y_Test)
accuracy_Voting = accuracy_score(Y_Test, result, normalize=True)


# classifier_KNN = KNeighborsClassifier()
# classifier_KNN.fit(X_Train, Y_Train)
# output = classifier_KNN.predict(X_Test)
# accuracy = accuracy_score(Y_Test, output, normalize=True)


