#! /usr/bin/env python

# ***********************************************************************************
# * Copyright 2010 - 2017 Paulo A. Herrera. All rights reserved.                    * 
# *                                                                                 *
# * Redistribution and use in source and binary forms, with or without              *
# * modification, are permitted provided that the following conditions are met:     *
# *                                                                                 *
# *  1. Redistributions of source code must retain the above copyright notice,      *
# *  this list of conditions and the following disclaimer.                          *
# *                                                                                 *
# *  2. Redistributions in binary form must reproduce the above copyright notice,   *
# *  this list of conditions and the following disclaimer in the documentation      *
# *  and/or other materials provided with the distribution.                         *
# *                                                                                 *
# * THIS SOFTWARE IS PROVIDED BY PAULO A. HERRERA ``AS IS'' AND ANY EXPRESS OR      *
# * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    *
# * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO      *
# * EVENT SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,        *
# * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
# * BUT NOT LIMITED TO, PROCUREMEN OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    *
# * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           *
# * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  *
# * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS              *
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                    *
# ***********************************************************************************

# ************************************************************************
# * Example of how to use the high level unstructuredGridToVTK function. *
# * This example shows how to export a unstructured grid give its        *
# * nodes and topology through a connectivity and offset lists.          *
# * Check the VTK file format for details of the unstructured grid.      *
# ************************************************************************

from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle, VtkQuad, VtkPolygon, VtkVoxel, VtkHexahedron
import numpy as np

# Define vertices
x = np.zeros(16)
y = np.zeros(16)
z = np.zeros(16)

x[0], y[0], z[0] = 0.0, 0.0, 0.0
x[1], y[1], z[1] = 1.0, 0.0, 0.0
x[2], y[2], z[2] = 0.0, 1.0, 0.0
x[3], y[3], z[3] = 1.0, 1.0, 0.0
x[4], y[4], z[4] = 0.0, 0.0, 1.0
x[5], y[5], z[5] = 1.0, 0.0, 1.0
x[6], y[6], z[6] = 0.0, 1.0, 1.0
x[7], y[7], z[7] = 1.0, 1.0, 1.0

x[8], y[8], z[8]    = 1.0, 0.0, 0.0
x[9], y[9], z[9]    = 2.0, 0.0, 0.0
x[10], y[10], z[10] = 1.0, 1.0, 0.0
x[11], y[11], z[11] = 2.0, 1.0, 0.0
x[12], y[12], z[12] = 1.0, 0.0, 2.0
x[13], y[13], z[13] = 2.0, 0.0, 2.0
x[14], y[14], z[14] = 1.0, 1.0, 2.0
x[15], y[15], z[15] = 2.0, 1.0, 2.0


# Define connectivity or vertices that belongs to each element
conn = np.zeros(16)
for i in range(16):
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
offset = np.zeros(2)
offset[0] = 8
offset[1] = 16

# Define cell types

ctype = np.zeros(2)
ctype[0] = VtkVoxel.tid
ctype[1] = VtkVoxel.tid
# ctype = np.zeros(3)
# ctype[0], ctype[1] = VtkTriangle.tid, VtkTriangle.tid
# ctype[2] = VtkQuad.tid

cell_data = {'density':np.array([1,2]), 'volume':np.array([4,16])}

point_data = {'p_density':np.linspace(1,16)}
# point_data = {'p_density':np.append(np.ones(8),2*np.ones(8))}
 
savefile="/Users/nathanvaughn/Desktop/meshTests/forVisitTesting/unstructured"
unstructuredGridToVTK(savefile, 
                      x, y, z, connectivity = conn, offsets = offset, cell_types = ctype, 
                      cellData = cell_data, pointData = point_data)

print('Done.  Saved to ', savefile)




