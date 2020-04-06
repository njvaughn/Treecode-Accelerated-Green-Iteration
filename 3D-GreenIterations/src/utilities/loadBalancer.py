import sys
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# from cyarray.api import UIntArray, DoubleArray
# from pyzoltan.core import zoltan
# from pyzoltan.core import zoltan_comm

import numpy as np


colors = ['r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood',
          'r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood',
          'r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood',
          'r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood',
          'r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood',
          'r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood']


def plot_points_single_proc(x, y, z, rank, title):
    if '--plot' not in sys.argv:
        return

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    s1 = fig.add_subplot(111, projection='3d')
    s1.axes.plot3D(
        x, y, z,
        c=colors[rank], marker='o', markersize=1, linestyle='None', alpha=0.5
    )

    s1.axes.set_xlabel( 'X' )
    s1.axes.set_ylabel( 'Y' )
    s1.axes.set_zlabel( 'Z' )
    
    s1.axes.set_xlim([-1,1])
    s1.axes.set_ylim([-1,1])
    s1.axes.set_zlim([-2,2])

    plt.title(title)
    plt.show() 
#     plt.savefig(savedir+filename)



def loadBalance_manual(x,y,z):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    assert ((size==1) or (size%2==0)), "manual load balancer needs size%2==0 or size=1"
    
    xmax=np.max(x)
    xmin=np.min(x)
    
    ymax=np.max(y)
    ymin=np.min(y)
    
    zmax=np.max(z)
    zmin=np.min(z)
    
#     subdomains=1
#     cycle=0
#     while subdomains<size:
#         subdomains*=
#         if cycle%3==0:   # cut the x
#             pass
#         elif cycle%3==1: # cut in y
#             pass
#         elif cycle%3==2: # cut in z
#             pass

    if size==1:
        bounds=[xmin, xmax, ymin, ymax, zmin, zmax] # 02/02/02
    elif size==2:
        xmid = (xmax+xmin)/2
        if rank==0:
            bounds=[xmin, xmid, ymin, ymax, zmin, zmax] # 01/02/02
        elif rank==1:
            bounds=[xmid, xmax, ymin, ymax, zmin, zmax] # 12/02/02
    
    elif size==4:
        xmid = (xmax+xmin)/2
        ymid = (ymax+ymin)/2
        if rank==0:
            bounds=[xmin, xmid, ymin, ymid, zmin, zmax] # 01/01/02
        elif rank==1:
            bounds=[xmin, xmid, ymid, ymax, zmin, zmax] # 01/12/02
        elif rank==2:
            bounds=[xmid, xmax, ymin, ymid, zmin, zmax] # 12/01/02
        elif rank==3:
            bounds=[xmid, xmax, ymid, ymax, zmin, zmax] # 12/12/02
            
            
    elif size==8:
        xmid = (xmax+xmin)/2
        ymid = (ymax+ymin)/2
        zmid = (zmax+zmin)/2
        if rank==0:
            bounds=[xmin, xmid, ymin, ymid, zmin, zmid] # 01/01/01
        elif rank==1:
            bounds=[xmin, xmid, ymid, ymax, zmin, zmid] # 01/12/01
        elif rank==2:
            bounds=[xmid, xmax, ymin, ymid, zmin, zmid] # 12/01/01
        elif rank==3:
            bounds=[xmid, xmax, ymid, ymax, zmin, zmid] # 12/12/01
        elif rank==4:
            bounds=[xmin, xmid, ymin, ymid, zmid, zmax] # 01/01/12
        elif rank==5:
            bounds=[xmin, xmid, ymid, ymax, zmid, zmax] # 01/12/12
        elif rank==6:
            bounds=[xmid, xmax, ymin, ymid, zmid, zmax] # 12/01/12
        elif rank==7:
            bounds=[xmid, xmax, ymid, ymax, zmid, zmax] # 12/12/12
    
    elif size==16:
        xmid = (xmax+xmin)/2
        xmidL = (xmin+xmid)/2
        xmidR = (xmax+xmid)/2
        ymid = (ymax+ymin)/2
        zmid = (zmax+zmin)/2
        if rank==0:
            bounds=[xmin, xmidL, ymin, ymid, zmin, zmid] # 01/01/01
        elif rank==1:
            bounds=[xmin, xmidL, ymid, ymax, zmin, zmid] # 01/12/01
        elif rank==2:
            bounds=[xmidL, xmid, ymin, ymid, zmin, zmid] # 12/01/01
        elif rank==3:
            bounds=[xmidL, xmid, ymid, ymax, zmin, zmid] # 12/12/01
        elif rank==4:
            bounds=[xmin, xmidL, ymin, ymid, zmid, zmax] # 01/01/12
        elif rank==5:
            bounds=[xmin, xmidL, ymid, ymax, zmid, zmax] # 01/12/12
        elif rank==6:
            bounds=[xmidL, xmid, ymin, ymid, zmid, zmax] # 12/01/12
        elif rank==7:
            bounds=[xmidL, xmid, ymid, ymax, zmid, zmax] # 12/12/12
        
        elif rank==8:
            bounds=[xmid, xmidR, ymin, ymid, zmin, zmid] # 01/01/01
        elif rank==9:
            bounds=[xmid, xmidR, ymid, ymax, zmin, zmid] # 01/12/01
        elif rank==10:
            bounds=[xmidR, xmax, ymin, ymid, zmin, zmid] # 12/01/01
        elif rank==11:
            bounds=[xmidR, xmax, ymid, ymax, zmin, zmid] # 12/12/01
        elif rank==12:
            bounds=[xmid, xmidR, ymin, ymid, zmid, zmax] # 01/01/12
        elif rank==13:
            bounds=[xmid, xmidR, ymid, ymax, zmid, zmax] # 01/12/12
        elif rank==14:
            bounds=[xmidR, xmax, ymin, ymid, zmid, zmax] # 12/01/12
        elif rank==15:
            bounds=[xmidR, xmax, ymid, ymax, zmid, zmax] # 12/12/12
            
    elif size==32:
        
        fraction=4
        xmid = (xmax+xmin)/2
        xmidL = (xmin+fraction*xmid)/(fraction+1)
        xmidR = (xmax+fraction*xmid)/(fraction+1)
        ymid = (ymax+ymin)/2
        ymidL = (ymin+fraction*ymid)/(fraction+1)
        ymidR = (ymax+fraction*ymid)/(fraction+1)
        zmid = (zmax+zmin)/2
        if rank==0:
            bounds=[xmin, xmidL, ymin, ymidL, zmin, zmid] # 01/01/01
        elif rank==1:
            bounds=[xmin, xmidL, ymidL, ymid, zmin, zmid] # 01/12/01
        elif rank==2:
            bounds=[xmidL, xmid, ymin, ymidL, zmin, zmid] # 12/01/01
        elif rank==3:
            bounds=[xmidL, xmid, ymidL, ymid, zmin, zmid] # 12/12/01
        elif rank==4:
            bounds=[xmin, xmidL, ymin, ymidL, zmid, zmax] # 01/01/12
        elif rank==5:
            bounds=[xmin, xmidL, ymidL, ymid, zmid, zmax] # 01/12/12
        elif rank==6:
            bounds=[xmidL, xmid, ymin, ymidL, zmid, zmax] # 12/01/12
        elif rank==7:
            bounds=[xmidL, xmid, ymidL, ymid, zmid, zmax] # 12/12/12
        
        elif rank==8:
            bounds=[xmid, xmidR, ymin, ymidL, zmin, zmid] # 01/01/01
        elif rank==9:
            bounds=[xmid, xmidR, ymidL, ymid, zmin, zmid] # 01/12/01
        elif rank==10:
            bounds=[xmidR, xmax, ymin, ymidL, zmin, zmid] # 12/01/01
        elif rank==11:
            bounds=[xmidR, xmax, ymidL, ymid, zmin, zmid] # 12/12/01
        elif rank==12:
            bounds=[xmid, xmidR, ymin, ymidL, zmid, zmax] # 01/01/12
        elif rank==13:
            bounds=[xmid, xmidR, ymidL, ymid, zmid, zmax] # 01/12/12
        elif rank==14:
            bounds=[xmidR, xmax, ymin, ymidL, zmid, zmax] # 12/01/12
        elif rank==15:
            bounds=[xmidR, xmax, ymidL, ymid, zmid, zmax] # 12/12/12
        
        elif rank==16:
            bounds=[xmin, xmidL, ymid, ymidR, zmin, zmid] # 01/01/01
        elif rank==17:
            bounds=[xmin, xmidL, ymidR, ymax, zmin, zmid] # 01/12/01
        elif rank==18:
            bounds=[xmidL, xmid, ymid, ymidR, zmin, zmid] # 12/01/01
        elif rank==19:
            bounds=[xmidL, xmid, ymidR, ymax, zmin, zmid] # 12/12/01
        elif rank==20:
            bounds=[xmin, xmidL, ymid, ymidR, zmid, zmax] # 01/01/12
        elif rank==21:
            bounds=[xmin, xmidL, ymidR, ymax, zmid, zmax] # 01/12/12
        elif rank==22:
            bounds=[xmidL, xmid, ymid, ymidR, zmid, zmax] # 12/01/12
        elif rank==23:
            bounds=[xmidL, xmid, ymidR, ymax, zmid, zmax] # 12/12/12
        
        elif rank==24:
            bounds=[xmid, xmidR, ymid, ymidR, zmin, zmid] # 01/01/01
        elif rank==25:
            bounds=[xmid, xmidR, ymidR, ymax, zmin, zmid] # 01/12/01
        elif rank==26:
            bounds=[xmidR, xmax, ymid, ymidR, zmin, zmid] # 12/01/01
        elif rank==27:
            bounds=[xmidR, xmax, ymidR, ymax, zmin, zmid] # 12/12/01
        elif rank==28:
            bounds=[xmid, xmidR, ymid, ymidR, zmid, zmax] # 01/01/12
        elif rank==29:
            bounds=[xmid, xmidR, ymidR, ymax, zmid, zmax] # 01/12/12
        elif rank==30:
            bounds=[xmidR, xmax, ymid, ymidR, zmid, zmax] # 12/01/12
        elif rank==31:
            bounds=[xmidR, xmax, ymidR, ymax, zmid, zmax] # 12/12/12
    
    else:
        print("Not set up for domain decomosition of size %i" %size)
    
#     elif size==16:
#         pass
#     elif size==32:
#         pass

    print("Rank %i owns " %rank, bounds)
    comm.barrier()
    
    
    cells = []
    cellsX = []
    cellsY = []
    cellsZ = []
    
    for i in range(len(x)):
        if ( (x[i]>=bounds[0]) and (x[i]<=bounds[1]) ):
            if ( (y[i]>=bounds[2]) and (y[i]<=bounds[3]) ):
                if ( (z[i]>=bounds[4]) and (z[i]<=bounds[5]) ):
                    cellsX.append(x[i])
                    cellsY.append(y[i])
                    cellsZ.append(z[i])
#     exit(-1)
    return cellsX, cellsY, cellsZ
        


def loadBalance(x,y,z,data=None,LBMETHOD='RCB',verbosity=0):
    '''
    Each processor calls loadBalance.  Using zoltan, the particles are balanced and redistributed as necessary.  Returns the balanced arrays.
    Does not require each processor to have started with the same number of particles.
    x, y, and z are arrays of length number-of-local-particles.
    data is a dictionary of data arrays
    '''
    
    ## Check if trying to communicate a data field, or just the positions.
    ## If more than 1 data field is needed, can either try to do a ndarray of data, a dictionary, or just hard-code in a second array.
    if data is not None:
        dataExists=True
        if ( (verbosity>0) and (rank==0) ): print('dataExists=True')
    else:
        dataExists=False
        if ( (verbosity>0) and (rank==0) ): print('dataExists=False')
    
    initialNumPoints = len(x)
    pointsOnEachProc = comm.allgather(initialNumPoints)
    
    # Compute local offset
    localOffset = np.int( np.sum( pointsOnEachProc[:rank]) )   
    
    
    if verbosity>0:
        if rank==0: print("Points on each proc: ", pointsOnEachProc)
        print("Rank %i offset: %i" %(rank,localOffset))
        print("Rank %i initialNumPoints: %i" %(rank,initialNumPoints))
        
    
    # Begin load balancing
    globalNumPoints = np.sum(pointsOnEachProc)
    if ( (verbosity>0) and (rank==0) ): print("global number of points: ", globalNumPoints)
    gid = np.arange( globalNumPoints, dtype=np.uint32 )[localOffset:localOffset+initialNumPoints]
    
    if verbosity>0: print("rank %i, gid: " %(rank), gid)

    ## DoubleArrays are from cyarray.  They act more like c++ vectors.  Can be resized, can get the pointer to the array, etc.
    xa = DoubleArray(initialNumPoints); xa.set_data(x)
    ya = DoubleArray(initialNumPoints); ya.set_data(y)
    za = DoubleArray(initialNumPoints); za.set_data(z)
    gida = UIntArray(initialNumPoints); gida.set_data(gid)

    # create the geometric partitioner
    if ( (verbosity>0) and (rank==0) ): print("Setting up geometric partitioner.")
    pz = zoltan.ZoltanGeometricPartitioner(
        dim=3, comm=comm, x=xa, y=ya, z=za, gid=gida)
    if ( (verbosity>0) and (rank==0) ): print("Completed geometric partitioner setup.")

    if ( (verbosity>0) and (rank==0) ): print("Calling the load balancer.")
    pz.set_lb_method(LBMETHOD)
    pz.Zoltan_Set_Param('DEBUG_LEVEL', '0')
    pz.Zoltan_Set_Param('IMBALANCE_TOL','1.1')
    pz.Zoltan_LB_Balance()
    if ( (verbosity>0) and (rank==0) ): print("Zoltan_LB_Balance complete.")
    
    # get the new assignments
    my_global_ids = list( gid )
    original_my_global_ids = np.copy(my_global_ids)
    if ( (verbosity>0) and (rank==0) ): print("new assignments set.")   
    
    # remove points to be exported
#     for i in range(pz.numExport):
#         if ( (verbosity>0) and (rank==0) ): print("removing: ",pz.exportGlobalids[i])
#         my_global_ids.remove( pz.exportGlobalids[i] )
    if ( (verbosity>0)): print("num export = %i, len my_global_ids = %i." %(pz.numExport,len(my_global_ids)))   
    if ( (verbosity>0)): print("type of pz.exportGlobalids:", type(pz.exportGlobalids))   
    if ( (verbosity>0)): print("type of my_global_ids:", type(my_global_ids))   
    
    my_global_ids = [x for x in my_global_ids if x not in pz.exportGlobalids]
    afterExport_my_global_ids = np.copy(my_global_ids)
    
    if ( (verbosity>0)): print("removed points to be exported from rank %i." %rank)   
    comm.barrier() 
    ## Communicate the changes
    
    # create the ZComm object
    nsend=len(pz.exportProcs)
    tag = np.int32(0)
    zcomm = zoltan_comm.ZComm(comm, tag=tag, nsend=nsend, proclist=pz.exportProcs.get_npy_array())
    if ( (verbosity>0) and (rank==0) ): print("zcomm object set.")   

    # the data to send and receive
    send_x=np.zeros(nsend)
    send_y=np.zeros(nsend)
    send_z=np.zeros(nsend)
    for i in range(nsend):
        send_x[i] = x[ pz.exportGlobalids[i] - localOffset ]
        send_y[i] = y[ pz.exportGlobalids[i] - localOffset ]
        send_z[i] = z[ pz.exportGlobalids[i] - localOffset ]
        
    if dataExists==True: 
        send_data=np.zeros(nsend)
        for i in range(nsend):
            send_data[i] = data[ pz.exportGlobalids[i] - localOffset]
        
    recv_x = np.ones( zcomm.nreturn )
    recv_y = np.ones( zcomm.nreturn )
    recv_z = np.ones( zcomm.nreturn )
    if dataExists==True: recv_data = np.ones( zcomm.nreturn )
    
    if ( (verbosity>0) and (rank==0) ): print("send and receive buffers set.")   

    # use zoltan to exchange data
    comm.barrier() 
    zcomm.Comm_Do(send_x, recv_x)
    zcomm.Comm_Do(send_y, recv_y)
    zcomm.Comm_Do(send_z, recv_z)
    if dataExists==True: zcomm.Comm_Do(send_data, recv_data)

    if ( (verbosity>0) and (rank==0) ): print("Comm_Do sends and received complete.")
    
    # Grab particles that remain on this processor.
    original_x = np.zeros(len(afterExport_my_global_ids))
    original_y = np.zeros(len(afterExport_my_global_ids))
    original_z = np.zeros(len(afterExport_my_global_ids))
    for i in range(len(afterExport_my_global_ids)):
        original_x[i] = x[ afterExport_my_global_ids[i] - localOffset ]
        original_y[i] = y[ afterExport_my_global_ids[i] - localOffset ]
        original_z[i] = z[ afterExport_my_global_ids[i] - localOffset ]
    
    if dataExists==True: 
        original_data = np.zeros(len(afterExport_my_global_ids))
        for i in range(len(afterExport_my_global_ids)):
            original_data[i] = data[ afterExport_my_global_ids[i] - localOffset]
    
    if ( (verbosity>0) and (rank==0) ): print("Grabbed original particles that remained local.")
    
    # Append the received particles
    balanced_x = np.append( original_x, np.copy(recv_x))
    balanced_y = np.append( original_y, np.copy(recv_y))
    balanced_z = np.append( original_z, np.copy(recv_z))
    if dataExists==True: 
        balanced_data = np.append( original_data, np.copy(recv_data))
    
    if ( (verbosity>0) and (rank==0) ): print("balanced arrays set.  Returning.")
    comm.barrier() 
#     print("Rank %i started with %i points.  After load balancing it has %i points." %(rank,initialNumPoints,len(balanced_x)))  
    
    if dataExists==True: 
        return balanced_x,balanced_y,balanced_z, balanced_data
    else:
        return balanced_x,balanced_y,balanced_z 


if __name__=="__main__":
    from numpy.random import random
    from numpy import pi, sin, cos, arccos
    
    
    n = 50*(rank+1)**2
#     if rank%2==0: n=0

#     n=1800*8
#     if rank==0:
#         n=57600
#     else:
#         n=0
    data = np.random.random( n )
    
    ## Unit cube, uniformly distributed
    x = 2*(np.random.random( n )-1/2)
    y = 2*(np.random.random( n )-1/2)
    z = 2*(np.random.random( n )-1/2)
    
#     ## Unit circle, uniformly distributed in radius
#     phi = 2*pi*random(n)
#     costheta = 2*random(n)-1
#     u = random(n)
#     r = u**3
#     
#     theta = arccos( costheta )
#     x = r * sin( theta) * cos( phi )
#     y = r * sin( theta) * sin( phi )
#     z = r * cos( theta )
#         
#     x = np.append(x, x)
#     y = np.append(y, y)
#     z = np.append(z-np.ones(n), z+np.ones(n))
#     data = np.append(data,data)
    
    initSumX = comm.allreduce(np.sum(x))
    initSumY = comm.allreduce(np.sum(y))
    initSumZ = comm.allreduce(np.sum(z))
    initSumData = comm.allreduce(np.sum(data))
    


    x2 = np.copy(x)
    y2 = np.copy(y)
    z2 = np.copy(z)
    plot_points_single_proc(x,y,z,rank,'Initial points for rank %i'%rank)
    print("calling loadBalance")
    start = MPI.Wtime()
    x,y,z = loadBalance(x,y,z,LBMETHOD='RCB')
#     x,y,z = loadBalance(x,y,z,LBMETHOD='HSFC')
#     x,y,z,data = loadBalance(x,y,z,data,LBMETHOD='RCB')
#     x,y,z,data = loadBalance(x,y,z,data,LBMETHOD='HSFC')
    plot_points_single_proc(x,y,z,rank,'Final points for rank %i'%rank)
    end = MPI.Wtime()
    
    print("Time to load balance particles: %f (s)" %(end-start))

    finalSumX = comm.allreduce(np.sum(x))
    finalSumY = comm.allreduce(np.sum(y))
    finalSumZ = comm.allreduce(np.sum(z))
    finalSumData = comm.allreduce(np.sum(data))
    
    assert abs( (initSumX-finalSumX)/initSumX )<1e-12, "Sum over x positions not preserved."
    assert abs( (initSumY-finalSumY)/initSumY )<1e-12, "Sum over y positions not preserved."
    assert abs( (initSumZ-finalSumZ)/initSumZ )<1e-12, "Sum over z positions not preserved."
    assert abs( (initSumData-finalSumData)/initSumData )<1e-12, "Sum over data values not preserved."
    
    
#     x2,y2,z2 = loadBalance(x2,y2,z2,LBMETHOD='RCB')
#     
#     assert np.max( np.abs( x-x2 ) )<1e-12, "Two calls to load balancer didn't give same decomposition."
#     assert np.max( np.abs( y-y2 ) )<1e-12, "Two calls to load balancer didn't give same decomposition."
#     assert np.max( np.abs( z-z2 ) )<1e-12, "Two calls to load balancer didn't give same decomposition."
    
    
    

    