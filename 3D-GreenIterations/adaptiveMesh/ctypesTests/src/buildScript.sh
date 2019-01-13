gcc-8 -fopenmp -fPIC -shared -o ../lib/libconvolutionRoutines.so convolutionRoutines.c
gcc -fPIC -shared -o ../lib/libconvolutionRoutines_noOpenMP.so convolutionRoutines.c
