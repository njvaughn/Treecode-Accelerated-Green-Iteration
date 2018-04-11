#!/bin/bash
mpirun -np 2 ../bin/tree.exe /scratch/krasny_fluxg/lwwilson/S1000000.bin /scratch/krasny_fluxg/lwwilson/T10000.bin /scratch/krasny_fluxg/lwwilson/ex_s6_t4_0_p1.bin out.tsv 1000000 10000 0.5 5 1 100000 0 0 0 1 0
