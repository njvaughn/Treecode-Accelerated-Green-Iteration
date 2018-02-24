'''
Created on Feb 24, 2018

@author: nathanvaughn
'''
import numpy as np

def getNeighbors1D(identifier):
    print('Self Identifier: ', "".join(identifier))
    
    
    def recursiveDigitFlipForGreaterNeighbor(identifierList, targetIndex):
        neighborID = np.copy(identifierList)
        if (targetIndex == 0 and neighborID[targetIndex]) == '2':
            return list('This cell has no greater neighbor')
#         print('target = ',targetIndex)
        if neighborID[targetIndex] == '0':
#             print('Target ',targetIndex,' digit is a 0, calling Digit flip on target ', targetIndex-1)
            neighborID = recursiveDigitFlipForGreaterNeighbor(neighborID, targetIndex-1)
            return neighborID
        
        if neighborID[targetIndex] == '1':
#             print('Target ',targetIndex,' digit is a 1, flipping it to 2')
            neighborID[targetIndex] = '2'
            return neighborID
        
        if neighborID[targetIndex] == '2':
#             print('Target ',targetIndex,' digit is a 2.  Swapping to 1 and calling Digit flip on target ', targetIndex-1)
            neighborID[targetIndex] = '1'
            neighborID = recursiveDigitFlipForGreaterNeighbor(neighborID, targetIndex-1)
            return neighborID
        
    def recursiveDigitFlipForLesserNeighbor(identifierList, targetIndex):
        neighborID = np.copy(identifierList)
        if (targetIndex == 0 and neighborID[targetIndex]) == '1':
            return list('This cell has no lesser neighbor')
#         print('target = ',targetIndex)
        if neighborID[targetIndex] == '0':
#             print('Target ',targetIndex,' digit is a 0, calling Digit flip on target ', targetIndex-1)
            neighborID = recursiveDigitFlipForLesserNeighbor(neighborID, targetIndex-1)
            return neighborID
        
        if neighborID[targetIndex] == '2':
#             print('Target ',targetIndex,' digit is a 1, flipping it to 2')
            neighborID[targetIndex] = '1'
            return neighborID
        
        if neighborID[targetIndex] == '1':
#             print('Target ',targetIndex,' digit is a 2.  Swapping to 1 and calling Digit flip on target ', targetIndex-1)
            neighborID[targetIndex] = '2'
            neighborID = recursiveDigitFlipForLesserNeighbor(neighborID, targetIndex-1)
            return neighborID

        
    
    
    rightNeighbor = recursiveDigitFlipForGreaterNeighbor(identifier, len(identifier)-1)
    leftNeighbor = recursiveDigitFlipForLesserNeighbor(identifier, len(identifier)-1)
    print('Left:   ', "".join(leftNeighbor))
    print('Middle: ', "".join(identifier))
    print('Right:  ', "".join(rightNeighbor))



        
    
#     for digit in identifier:
#         print(digit)

def getNeighbors2D(identifier):
    print('Self Identifier: ', "".join(identifier))
    topBottomID = identifier[::2]
    leftRightID = identifier[1::2]
    print('top/bottom ID: ', topBottomID)
    print('left/right ID: ', leftRightID)
    print('types: ', type(topBottomID))
    
    
    def recursiveDigitFlipForGreaterNeighbor(identifierList, targetIndex, noNeighborFlag):
        neighborID = np.copy(identifierList)
        if (targetIndex == 0 and neighborID[targetIndex]) == '2':
            noNeighborFlag = True
            return (list('This cell has no greater neighbor'), noNeighborFlag)
#         print('target = ',targetIndex)
        if neighborID[targetIndex] == '0':
#             print('Target ',targetIndex,' digit is a 0, calling Digit flip on target ', targetIndex-1)
            neighborID, noNeighborFlag = recursiveDigitFlipForGreaterNeighbor(neighborID, targetIndex-1, noNeighborFlag)
            return (neighborID, noNeighborFlag)
        
        if neighborID[targetIndex] == '1':
#             print('Target ',targetIndex,' digit is a 1, flipping it to 2')
            neighborID[targetIndex] = '2'
            return (neighborID, noNeighborFlag)
        
        if neighborID[targetIndex] == '2':
#             print('Target ',targetIndex,' digit is a 2.  Swapping to 1 and calling Digit flip on target ', targetIndex-1)
            neighborID[targetIndex] = '1'
            neighborID, noNeighborFlag = recursiveDigitFlipForGreaterNeighbor(neighborID, targetIndex-1, noNeighborFlag)
            return (neighborID, noNeighborFlag)
        
    def recursiveDigitFlipForLesserNeighbor(identifierList, targetIndex, noNeighborFlag):
        neighborID = np.copy(identifierList)
        if (targetIndex == 0 and neighborID[targetIndex]) == '1':
            noNeighborFlag = True
            return (list('This cell has no lesser neighbor'), noNeighborFlag)
        if neighborID[targetIndex] == '0':
            neighborID, noNeighborFlag = recursiveDigitFlipForLesserNeighbor(neighborID, targetIndex-1, noNeighborFlag)
            return (neighborID, noNeighborFlag)
        
        if neighborID[targetIndex] == '2':
            neighborID[targetIndex] = '1'
            return (neighborID, noNeighborFlag)
        
        if neighborID[targetIndex] == '1':
            neighborID[targetIndex] = '2'
            neighborID, noNeighborFlag = recursiveDigitFlipForLesserNeighbor(neighborID, targetIndex-1, noNeighborFlag)
            return (neighborID, noNeighborFlag)

        
    
    
    rightID, rightIDNoNeighborFlag = recursiveDigitFlipForGreaterNeighbor(leftRightID, len(leftRightID)-1, noNeighborFlag=False)
    leftID, leftIDNoNeighborFlag = recursiveDigitFlipForLesserNeighbor(leftRightID, len(leftRightID)-1, noNeighborFlag=False)
    topID, topIDNoNeighborFlag = recursiveDigitFlipForGreaterNeighbor(topBottomID, len(topBottomID)-1, noNeighborFlag=False)
    bottomID, bottomIDNoNeighborFlag = recursiveDigitFlipForLesserNeighbor(topBottomID, len(topBottomID)-1, noNeighborFlag=False)
    
    rightNeighbor = []
    leftNeighbor = []
    topNeighbor = []
    bottomNeighbor = []

    for i in range(len(rightID)):
        rightNeighbor.append(topBottomID[i])
        rightNeighbor.append(rightID[i])
        leftNeighbor.append(topBottomID[i])
        leftNeighbor.append(leftID[i])
        
        topNeighbor.append(topID[i])
        topNeighbor.append(leftRightID[i])
        bottomNeighbor.append(bottomID[i])
        bottomNeighbor.append(leftRightID[i])
    
    print('\nTarget Cell ID: ', "".join(identifier),'\n')
#     print('\nleftID: ', leftID)
#     print('\nrightID: ', rightID)
#     print('\nrightNoNeighborFlag ', rightIDNoNeighborFlag)
#     print('\ntopID: ', topID)
#     print('\nbottomID: ', bottomID)
#     print('\nrightNeighbor: ',rightNeighbor)

    if leftIDNoNeighborFlag == True: print('Left:    ', "".join(leftID)) 
    else: print('Left:    ', "".join(leftNeighbor))
    if rightIDNoNeighborFlag == True: print('Right:   ', "".join(rightID)) 
    else: print('Right:   ', "".join(rightNeighbor))
    if bottomIDNoNeighborFlag == True: print('Bottom:  ', "".join(bottomID)) 
    else: print('Bottom:  ', "".join(bottomNeighbor))
    if topIDNoNeighborFlag == True: print('Top:     ', "".join(topID)) 
    else: print('Top:     ', "".join(topNeighbor))

    
    ### NOTICE: top and bottom nomenclature is swapped.  This is 
    # because in the diagrams, the direction of increasing tob/bottom index is downward.  e.g., C is BELOW A,
    # but has a GREATER second index.  

if __name__ == '__main__':
#     getNeighbors1D(list('11'))
    getNeighbors2D(list('112211'))
    
    
    
    