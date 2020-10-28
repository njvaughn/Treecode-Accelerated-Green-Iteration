import numpy as np
import ctypes


numberOfElements = 10

x = np.linspace(-1,1,numberOfElements)
print(x)
print(*x)

array_type = ctypes.c_double * numberOfElements

print(array_type(*x))
print(array_type(*x)[1])