
from pyevtk.hl import unstructuredGridToVTK, pointsToVTK
from pyevtk.vtk import VtkTriangle, VtkQuad, VtkPolygon, VtkVoxel, VtkHexahedron
import numpy as np

import sys
import os
dir=os.getcwd()+'/../src/utilities'
print(dir)
sys.path.append(dir)
from meshUtilities import ChebyshevPoints, weights


## 3D
# Define vertices
x = np.zeros(8)
y = np.zeros(8)
z = np.zeros(8)

xmax=1
xmin=-1
ymax=1
ymin=-1
zmax=1
zmin=-1


x[0], y[0], z[0] = xmin, ymin, zmin
x[1], y[1], z[1] = xmax, ymin, zmin
x[2], y[2], z[2] = xmin, ymax, zmin
x[3], y[3], z[3] = xmax, ymax, zmin
x[4], y[4], z[4] = xmin, ymin, zmax
x[5], y[5], z[5] = xmax, ymin, zmax
x[6], y[6], z[6] = xmin, ymax, zmax
x[7], y[7], z[7] = xmax, ymax, zmax



# Define connectivity or vertices that belongs to each element
conn = np.zeros(8)
for i in range(8):
    conn[i] = i

# conn[0] = [1,2,3]
# conn[1] = [0,5,6]
# conn[2] = [1,4,6]
# conn[3] = [0,4,5]
# conn[4] = [2,3,7]
# conn[5] = [1,3,7]
# conn[6] = [1,2,7]
# conn[7] = [4,5,6]
# conn[0], conn[1], conn[2] = 0, 1, 3              
# conn[3], conn[4], conn[5] = 1, 4, 3             
# conn[6], conn[7], conn[8], conn[9] = 1, 2, 5, 4  # rectangle

# Define offset of last vertex of each element
offset = np.zeros(1)
offset[0] = 8
# offset[1] = 16

# Define cell types

ctype = np.zeros(1)
ctype[0] = VtkVoxel.tid
# ctype[1] = VtkVoxel.tid
# ctype = np.zeros(3)
# ctype[0], ctype[1] = VtkTriangle.tid, VtkTriangle.tid
# ctype[2] = VtkQuad.tid

cell_data = {'density':np.array([1]), 'volume':np.array([4])}

point_data = {'p_density':np.linspace(1,8)}
# point_data = {'p_density':np.append(np.ones(8),2*np.ones(8))}
 
savefile="/Users/nathanvaughn/Desktop/meshTests/visit-refinement/root"
unstructuredGridToVTK(savefile+'_vertices', 
                      x, y, z, connectivity = conn, offsets = offset, cell_types = ctype, 
                      cellData = cell_data, pointData = point_data)

print('Done.  Saved to ', savefile)

x = np.zeros(order**3)
y = np.zeros(order**3)
z = np.zeros(order**3)
d = {'zeroes':np.zeros(order**3)}

idx=0
for i in range(order):
    for j in range(order):
        for k in range(order):
            x[idx] = xvec[i]
            y[idx] = yvec[j]
            z[idx] = zvec[k]
            idx+=1
pointsToVTK(savefile+'_quadrature',x, y, z, d)

print('Done.  Saved to ', savefile)





## Repeat for 2D slice

# Define vertices
x = np.zeros(4)
y = np.zeros(4)
z = np.zeros(4)

xmax=1
xmin=0
ymax=1
ymin=0

order = 5

xvec = ChebyshevPoints(xmin,xmax,order)
yvec = ChebyshevPoints(ymin,ymax,order)

x[0], y[0], z[0] = xmin, ymin, 0
x[1], y[1], z[1] = xmax, ymin, 0
x[2], y[2], z[2] = xmax, ymax, 0
x[3], y[3], z[3] = xmin, ymax, 0


# Define connectivity or vertices that belongs to each element
conn = np.zeros(4)
for i in range(4):
    conn[i] = (i+1)%4


# Define offset of last vertex of each element
offset = np.zeros(1)
offset[0] = 4
# offset[1] = 16

# Define cell types

ctype = np.zeros(1)
ctype[0] = VtkQuad.tid

cell_data = {'density':np.array([1]), 'volume':np.array([4])}

point_data = {'p_density':np.linspace(1,4)}
# point_data = {'p_density':np.append(np.ones(8),2*np.ones(8))}
 
savefile="/Users/nathanvaughn/Desktop/meshTests/forVisitTesting/singleCell_2d"
unstructuredGridToVTK(savefile+'_vertices', 
                      x, y,z, connectivity = conn, offsets = offset, cell_types = ctype, 
                      cellData = cell_data, pointData = point_data)

print('Done.  Saved to ', savefile)

x = np.zeros(order**2)
y = np.zeros(order**2)
z = np.zeros(order**2)
w = {'weights':np.zeros(order**2)}

idx=0
for i in range(order):
    for j in range(order):
        x[idx] = xvec[i]
        y[idx] = yvec[j]
        w['weights'][idx] = xw[i]*yw[j]
        idx+=1
pointsToVTK(savefile+'_quadrature',x, y, z, w)

print('Done.  Saved to ', savefile)



