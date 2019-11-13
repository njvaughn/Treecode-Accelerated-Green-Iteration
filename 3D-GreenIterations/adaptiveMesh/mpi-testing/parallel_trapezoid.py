from mpi4py import MPI
from trapezoid_rule import f, Trap
import time

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

if my_rank==0:
    startTime = MPI.Wtime()

a=0.0
b=1.0
n=2400*10000
dest=0
total=-1.0

h = (b-a)/n
local_n = n/p

local_a = a + my_rank*local_n*h
local_b = local_a + local_n*h
integral = Trap(local_a,local_b,local_n,h)



# print("Local piece: ", integral)

if my_rank != 0:
    comm.send(integral, dest=dest)

else:
    for procid in range(1,p):
        local_integral = comm.recv(source=procid)
        integral += local_integral
    
    print('Total Integral = ', integral)
    
if my_rank==0:
    endTime = MPI.Wtime()
    totalTime = endTime-startTime
    print('Total time: ', totalTime)
    
MPI.Finalize