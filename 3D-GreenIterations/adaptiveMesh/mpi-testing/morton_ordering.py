import mpi4py as MPI
import pymorton as pm
import numpy as np

N=100
X = np.random.randint(1,10,N,dtype=int)
Y = np.random.randint(1,10,N,dtype=int)
Z = np.random.randint(1,10,N,dtype=int)


# mortoncode = pm.interleave(100, 200, 50)  # 5162080
# mortoncode = pm.interleave3(100, 200, 50) # 5162080
# 
# pm.deinterleave3(mortoncode)              # (100, 200, 50)

hashedValues = np.zeros(N,dtype=int)
for i in range(N):
    print(X[i], Y[i],Z[i])
#     print(type(X[i]))
#     hashedValues[i] = int( pm.interleave3(1,2,3) )
    hashedValues[i] = pm.interleave3(int(X[i]),int(Y[i]),int(Z[i]))
#     print(hashedValues[i])
#     print(pm.deinterleave3(int(hashedValues[i])))
#     print()
    

print(hashedValues)
hashedValues = np.sort(hashedValues)
print(hashedValues)

for i in range(N):
    print(pm.deinterleave3(int(hashedValues[i])))