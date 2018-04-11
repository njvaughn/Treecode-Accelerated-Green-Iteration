'''
Created on Mar 7, 2018

@author: nathanvaughn
'''
import numpy as np
import bisect
from TreeStruct import Tree
from CellStruct import Cell

xmin = ymin = zmin = -12
xmax = ymax = zmax = -xmin
tree = Tree(xmin,xmax,ymin,ymax,zmin,zmax)
tree.buildTree( minLevels=2, maxLevels=2, divideTolerance=0.07,printTreeProperties=True)

unsortedList = np.copy(tree.masterList)
sortedList = tree.masterList

# for element in List: print(element[0])

# data.sort(key=lambda tup: tup[1])
# sortedList.sort(key=lambda tup: tup[0])

for element in sortedList: print(element[0])
# for element in unsortedList: print(element[0])

testCell = Cell()
testElement = ['2121212', testCell]
# print('testElement :', testElement)
bisect.insort(sortedList, testElement)

# for element in sortedList: print(element[0])

def find(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, [x,])
    if i != len(a) and a[i][0] == x:
        return i
    raise ValueError

try: 
    index = find(sortedList, '21212121')
    print(sortedList[index])
except ValueError:
    print("Element doesn't exist")


# print(sortedList[:][0])