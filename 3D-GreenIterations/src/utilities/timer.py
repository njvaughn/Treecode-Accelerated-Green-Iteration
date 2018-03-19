'''
Created on Feb 21, 2018

@author: nathanvaughn
'''
import time

class Timer(object):
    '''
    A timer with simplified commands.
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.startTime = None
        self.endTime = None
    
    def start(self):
        self.startTime = time.time()
    
    def stop(self):
        self.stopTime = time.time()
        self.elapsedTime = self.stopTime - self.startTime
    
    def runningTime(self):
        return time.time() - self.startTime
    
    def getElapsedTime(self):
        return self.elapsedTime
    
    
    