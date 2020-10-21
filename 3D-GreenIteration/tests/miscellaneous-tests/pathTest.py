print("[", __file__, "] ","blah blah blah")

import sys
sys.path.append('../../src/utilities')


from mpiUtilities import rprint
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



def dummy():

    rprint(rank,"testing")

def DUMMY():
    dummy()

DUMMY()