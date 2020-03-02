CC=gcc
CFLAGS=-I. -o3 -lm -fPIC
LDFLAGS = -shared 

TwoMeshInterpolation: TwoMeshInterpolation.c TwoMeshInterpolation.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o libInterpolation.so TwoMeshInterpolation.c -I.
	
	
	

#CC=pgcc
#CFLAGS=-I. -acc -ta=nvidia -Minfo=accel -DOPENACC_ENABLED 
#LDFLAGS = -shared -Mcuda
#
#TwoMeshInterpolation_GPU: TwoMeshInterpolation.c TwoMeshInterpolation.c
#	$(CC) $(CFLAGS) $(LDFLAGS) -o libInterpolation_gpu.so TwoMeshInterpolation.c -I.
	