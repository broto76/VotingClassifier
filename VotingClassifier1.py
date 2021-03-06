# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:03:35 2020

@author: Anibrata and Anirban
"""

import numpy as np

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
    
    voting_dict = {}
    
    result_internal = 0
    
    def __init__(self):
        print("__init__")
        # self._testSize = test_size
        
        # TODO
        self.classifier_KNN = KNeighborsClassifier()
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

    # Not Used
    def buildVoteDictionary(self, data):
        candidates = np.unique(data)
        for i in range (0,candidates.size):
            self.voting_dict[candidates[i]] = 0
            
    # Not Used
    def resetVoteDictionary(self):
        for x in self.voting_dict:
            self.voting_dict[x] = 0

    # Not Used
    def putVote(self, candidate, voteWeight):
        self.voting_dict[candidate] = self.voting_dict[candidate] + voteWeight
    
    # Not Used
    def getVoteResult(self):
        maxVoteCount = 0
        maxVoteCandidate = 0
        
        for x in self.voting_dict:
            if (self.voting_dict[x] > maxVoteCount):
                maxVoteCount = self.voting_dict[x]
                maxVoteCandidate = x

        return maxVoteCandidate
        
    def fit_func(self, features, result):
        
        # Preprocess the data between 0 and 1
        features = self.pre_prepossing(features)
        
        # Split the data to training and test set
        self.splitData(features, result)
        self.internalClassifierFit()

        self.internalAssignAvgWeight()
        self.buildVoteDictionary(result)
        
    def showInternalAccuracy(self,expected):
        self.accuracy_KNN_Test = accuracy_score(expected, self.KNN_Data, normalize=False)
        self.accuracy_SVM_Test = accuracy_score(expected, self.SVM_Data, normalize=False)
        self.accuracy_RFC_Test = accuracy_score(expected, self.RFC_Data, normalize=False)
        
        
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

        
    def predict_on_weighted_votes(self, test_features):
        
        test_features = self.pre_prepossing(test_features)
        self.KNN_Data = self.classifier_KNN.predict(test_features)
        self.SVM_Data = self.classifier_SVM.predict(test_features)
        self.RFC_Data = self.classifier_RFC.predict(test_features)
        
        self.result_internal = np.zeros(self.KNN_Data.size)
        self.resetVoteDictionary()
        
        for i in range (0, self.result_internal.size):
            self.putVote(self.KNN_Data[i], self.Weight_KNN)
            self.putVote(self.SVM_Data[i], self.Weight_SVM)
            self.putVote(self.RFC_Data[i], self.Weight_RFC)

            self.result_internal[i] = self.getVoteResult()
            self.resetVoteDictionary()
        
        return self.result_internal
        
