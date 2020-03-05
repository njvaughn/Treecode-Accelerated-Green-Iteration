CC=gcc
CFLAGS=-I. -o3 -lm -fPIC
LDFLAGS = -shared 

TwoMeshInterpolation: TwoMeshInterpolation.c TwoMeshInterpolation.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o libInterpolation_gpu.so TwoMeshInterpolation.c -I.
	
	 
	

#CC=pgcc
#CFLAGS=-I. -acc -ta=tesla:ccall,cc60 -Minfo=accel -DOPENACC_ENABLED 
#LDFLAGS = -shared -Mcuda
#
#TwoMeshInterpolation_GPU: TwoMeshInterpolation.c TwoMeshInterpolation.c
#	$(CC) $(CFLAGS) -shared -o libInterpolation_gpu.so TwoMeshInterpolation.c -I.
	
	
	
#$(LDFLAGS)

#CC=pgcc
#CFLAGS=-I.
#LDFLAGS = -shared 
#
#TwoMeshInterpolation_GPU: TwoMeshInterpolation.c TwoMeshInterpolation.c
#	$(CC) $(CFLAGS) $(LDFLAGS) -o libInterpolation_cpu.so TwoMeshInterpolation.c -I.
	