# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:58:08 2020

@author: Anibrata
"""

import numpy as np
import copy

class VotingManager:
    
    votingDictionary = {}
    candidates = 0
    candidates_size = 0
    
    votingDictionaryList = []
    list_size = 0
    isListInitialzed = False
    
    def __init__ (self, candidates):
        self.candidates = np.unique(candidates)
        self.candidates_size = self.candidates.size
        self.buildVotingDictionary()
        self.isListInitialzed = False
    
    def buildVotingDictionary(self):
        self.votingDictionary = {}
        for i in range (0,self.candidates_size):
            self.votingDictionary[self.candidates[i]] = 0

    def resetVotingDictionary(self):
        for i in range (0,self.candidates_size):
            self.votingDictionary[self.candidates[i]] = 0

    def buildVotingList(self, size):
        self.list_size = size
        self.truncateVotingList()
        for i in range (0,self.list_size):
            tmp = copy.deepcopy(self.votingDictionary)
            self.votingDictionaryList.append(tmp)
        self.isListInitialzed = True    

    def truncateVotingList(self):
        del self.votingDictionaryList[:]
        self.list_size = 0
        self.isListInitialzed = False

    def putVote(self, index, candidate, voteWeight):
        if self.isListInitialzed == False:
            print("List Not Initialized")
        else:
            self.votingDictionaryList[index][candidate] = self.votingDictionaryList[index][candidate] + voteWeight

    def getVoteResult(self, index):
        if self.isListInitialzed == False:
            print("List Not Initialized")
            return -1
        else:
            maxVoteCount = 0
            maxVoteCandidate = 0
            
            for x in self.votingDictionaryList[index]:
                if (self.votingDictionaryList[index][x] > maxVoteCount):
                    maxVoteCount = self.votingDictionaryList[index][x]
                    maxVoteCandidate = x
                    
        return maxVoteCandidate          





