import sys
import mpi4py.MPI as MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
import numpy as np
import inspect





def global_dot(u,v,comm):
    local_dot = np.dot(u,v)
    global_dot = comm.allreduce(local_dot)
    return global_dot
    
    
def scatterArrays(x,y,z,w,comm, verbose=0):
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    nPoints = len(x)
    if verbose>0:
        if rank==0: print(x)
    
    
    
    nPoints = comm.bcast(nPoints, root=0)
    
    if rank==0:
        ptsPerProc = nPoints - (nPoints // size)*(size-1)
    else:
        ptsPerProc = nPoints // size
#     print(ptsPerProc)
        
    xloc = np.empty(ptsPerProc)
    yloc = np.empty(ptsPerProc)
    zloc = np.empty(ptsPerProc)
    wloc = np.empty(ptsPerProc)
    
    counts = (nPoints // size)*np.ones(size, dtype=np.int)
    counts[0] = ptsPerProc
#     offsets = comm.Gatherv(ptsPerProc, 0)
    
    cum=0
    offsets = np.empty(size, dtype=np.int)
    for i in range(size):
        offsets[i] = cum
        cum += counts[i]
    
    if verbose>0:
        if rank==0:
            print(counts)
            print(offsets)
    
#     comm.Scatterv(sendbuf=x, recvbuf=xloc, offsets, root=0)
#     comm.Scatterv([test,split_sizes_input, displacements_input,MPI.DOUBLE],output_chunk,root=0)
    comm.Scatterv([x,counts, offsets,MPI.DOUBLE],xloc,root=0)
    comm.Scatterv([y,counts, offsets,MPI.DOUBLE],yloc,root=0)
    comm.Scatterv([z,counts, offsets,MPI.DOUBLE],zloc,root=0)
    comm.Scatterv([w,counts, offsets,MPI.DOUBLE],wloc,root=0)

    
#     print("proc %i: "%rank, xloc)

    print("Proc %i now how %i points." %(rank,len(xloc)))
    
    return xloc, yloc, zloc, wloc
    
    
def rprint(rank, message, message2=None, message3=None):
    '''
    Stands for root-print, meaning only the root proc will print the message.
    :param message:
    '''
    if message2 is not None:
        if message3 is not None:
            if rank==0: print("[", inspect.stack()[1][3], "] ", message, message2, message3)
        else:
            if rank==0: print("[", inspect.stack()[1][3], "] ", message, message2)
    else:
#         if rank==0: print("[", __file__, "] ", message)
        if rank==0: print("[", inspect.stack()[1][3], "] ", message)
        
                
        





if __name__=="__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    
    n=15
     
     
     
    if rank==0:
        x = np.random.random(n)
        y = np.random.random(n)
        z = np.random.random(n)
        w = np.random.random(n)
    else:
        x = np.empty(0)
        y = np.empty(0)
        z = np.empty(0)
        w = np.empty(0)
     
    scatterArrays(x,y,z,w,comm)
    