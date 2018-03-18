'''
Created on Feb 24, 2018
@author: nathanvaughn
'''
import numpy as np

def getNeighbors1D(identifier):
    print('Self Identifier: ', "".join(identifier))
    
    
    def recursiveDigitFlipForGreaterNeighbor(identifierList, targetIndex):
        neighborID = np.copy(identifierList)
        if ( (targetIndex == 0) and (neighborID[targetIndex] == '2') ):
            """ We reached the first digit, and it's still a 2, so this cell has no greater neighbor. """
            return list('This cell has no greater neighbor')
        
        if neighborID[targetIndex] == '0':
            """ Skip the padded zeros, recursively call the function on the next significant digit """
            neighborID = recursiveDigitFlipForGreaterNeighbor(neighborID, targetIndex-1)
            return neighborID
        
        if neighborID[targetIndex] == '1':
            """ Flip the digit, return the neighbor ID """
            neighborID[targetIndex] = '2'
            return neighborID
        
        if neighborID[targetIndex] == '2':
            """ Flip the digit, recursively call the function on the next significant digit """
            neighborID[targetIndex] = '1'
            neighborID = recursiveDigitFlipForGreaterNeighbor(neighborID, targetIndex-1)
            return neighborID
        
    def recursiveDigitFlipForLesserNeighbor(identifierList, targetIndex):
        neighborID = np.copy(identifierList)
        if ( (targetIndex == 0) and (neighborID[targetIndex] == '1') ):
            """ We reached the first digit, and it's still a 2, so this cell has no greater neighbor. """
            return list('This cell has no lesser neighbor')

        if neighborID[targetIndex] == '0':
            """ Skip the padded zeros, recursively call the function on the next significant digit """
            neighborID = recursiveDigitFlipForLesserNeighbor(neighborID, targetIndex-1)
            return neighborID
        
        if neighborID[targetIndex] == '2':
            """ Flip the digit, return the neighbor ID """
            neighborID[targetIndex] = '1'
            return neighborID
        
        if neighborID[targetIndex] == '1':
            """ Flip the digit, recursively call the function on the next significant digit """
            neighborID[targetIndex] = '2'
            neighborID = recursiveDigitFlipForLesserNeighbor(neighborID, targetIndex-1)
            return neighborID

    
    """ Call the recursive functions above, starting with the LEAST significant digit """   
    rightNeighbor = recursiveDigitFlipForGreaterNeighbor(identifier, len(identifier)-1)
    leftNeighbor = recursiveDigitFlipForLesserNeighbor(identifier, len(identifier)-1)
    print('Left:   ', "".join(leftNeighbor))
    print('Right:  ', "".join(rightNeighbor))
    
    
if __name__ == '__main__':

    """ A few example test cases. """
    getNeighbors1D(list('111'))
    getNeighbors1D(list('121'))
    getNeighbors1D(list('122000'))
    getNeighbors1D(list('211'))
    getNeighbors1D(list('222'))