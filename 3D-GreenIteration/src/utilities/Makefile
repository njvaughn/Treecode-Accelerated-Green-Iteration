

TwoMeshInterpolation_cpu: TwoMeshInterpolation.c TwoMeshInterpolation.c
	mpicc -O3 -lm -fPIC -shared -o libInterpolation_cpu.so TwoMeshInterpolation.c -I.
#	
#	 
cpuOrthogonalization: testParallelOrthogonalization.c Orthogonalization.c moveData.c
	mpicc -O3 -lm -o cpu_testParallelOrthogonalization testParallelOrthogonalization.c Orthogonalization.c moveData.c -I.


cpuLib: Orthogonalization.c 	
	mpicc -O3 -lm -fPIC -shared -o libOrthogonalization_cpu.so Orthogonalization.c -I.

#CC=pgcc
#CFLAGS=-I. -acc -ta=tesla:ccall,cc60 -Minfo=accel -DOPENACC_ENABLED 
#LDFLAGS = -shared -Mcuda
#
#TwoMeshInterpolation_GPU: TwoMeshInterpolation.c TwoMeshInterpolation.c
#	$(CC) $(CFLAGS) -shared -o libInterpolation_gpu.so TwoMeshInterpolation.c -I.
	
CC=mpicc
#CFLAGS=-I. -acc -ta=tesla:ccall,cc60,time -Minfo=accel -DOPENACC_ENABLED 
CFLAGS=-I. -acc -ta=tesla:cc70 -Minfo=accel -DOPENACC_ENABLED 
LDFLAGS = -Mcuda
gpuOrthogonalization: testParallelOrthogonalization.c Orthogonalization.c moveData.c
	$(CC) $(CFLAGS) -o gpu_testParallelOrthogonalization testParallelOrthogonalization.c Orthogonalization.c moveData.c -I.
	mpicc -I. -acc -ta=tesla:ccall,cc60 -Minfo=accel -DOPENACC_ENABLED  -o gpu_testParallelOrthogonalization testParallelOrthogonalization.c Orthogonalization.c moveData.c -I.


gpuLib: testParallelOrthogonalization.c Orthogonalization.c moveData.c	
	mpicc -I. -acc -ta=tesla:ccall,cc60 -Minfo=accel -DOPENACC_ENABLED -shared -o libOrthogonalization_gpu.so Orthogonalization.c moveData.c -I.	
	
#$(LDFLAGS)

#CC=pgcc
#CFLAGS=-I.
#LDFLAGS = -shared 
#
#TwoMeshInterpolation_GPU: TwoMeshInterpolation.c TwoMeshInterpolation.c
#	$(CC) $(CFLAGS) $(LDFLAGS) -o libInterpolation_cpu.so TwoMeshInterpolation.c -I.

gpuMoveData: moveData.c
	mpicc -I. -acc -ta=tesla:ccall,cc60 -Minfo=accel -DOPENACC_ENABLED -shared -o libMoveData.so moveData.c -I.
	
gpuMoveData_cpu: moveData.c
	mpicc -O3 -lm -fPIC -shared -o libMoveData.so moveData.c -I. 
	
	  
	
LB_local: testLoadBalancing.c zoltan_fns.c ZoltanRCB.c
	mpicc -O3 -lm -L/Users/nathanvaughn/Zoltan_py/lib -I/Users/nathanvaughn/Zoltan_py/include -lzoltan -o testLoadBalance testLoadBalancing.c ZoltanRCB.c zoltan_fns.c -I.

LB_lib: zoltan_fns.c ZoltanRCB.c
	mpicc -O3 -lm -fPIC -shared -L/Users/nathanvaughn/Zoltan_py/lib -I/Users/nathanvaughn/Zoltan_py/include -lzoltan -o ZoltanRCB.so ZoltanRCB.c zoltan_fns.c -I.


LB_gl: zoltan_fns.c ZoltanRCB.c
	mpicc -g -lm -fPIC -shared -L/home/njvaughn/gccZoltan/lib/ -I/home/njvaughn/gccZoltan/include -lzoltan -o ZoltanRCB.so ZoltanRCB.c zoltan_fns.c -I.


LB_comet: zoltan_fns.c ZoltanRCB.c
	mpicc -O3 -lm -fPIC -shared -L/home/njvaughn/mvapich2_ib_Zoltan_Shared/lib -I/home/njvaughn/mvapich2_ib_Zoltan_Shared/include -lzoltan -o ZoltanRCB.so ZoltanRCB.c zoltan_fns.c -I.

