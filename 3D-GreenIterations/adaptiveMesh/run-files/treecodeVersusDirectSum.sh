TREETYPE=1

SFLAG=1
PFLAG=0
DFLAG=0

N=184750
BATCHSIZE=100
MAXPARNODE=5000
SOURCES=/scratch/krasny_fluxg/njvaughn/CO_meshes/S$N.bin    
TARGETS=/scratch/krasny_fluxg/njvaughn/CO_meshes/T$N.bin
#SOURCES=/Users/nathanvaughn/Desktop/CO_meshes/S$N.bin    
#TARGETS=/Users/nathanvaughn/Desktop/CO_meshes/T$N.bin 

NUMDEVICES=0
NUMTHREADS=6

## COULOMB 
KAPPA=1.0
POTENTIALTYPE=2
DIRECTSUM=/scratch/krasny_fluxg/njvaughn/CO_meshes/ex_st_coulomb_$N.bin
#DIRECTSUM=/Users/nathanvaughn/Desktop/CO_meshes/ex_st_coulomb_$N.bin
OUTFILE=/home/njvaughn/synchronizedDataFiles/testing/coulomb.csv 
#OUTFILE=/Users/nathanvaughn/Desktop/CO_meshes/coulomb.csv 


KAPPA=1.0  #Gaussian SS parameter


#for N in 141000 
#for N in 184750 249500 459500 928500 2224375
for N in 2224375
do

	echo $N

	SOURCES=/Users/nathanvaughn/Desktop/CO_meshes/S$N.bin    
	TARGETS=/Users/nathanvaughn/Desktop/CO_meshes/T$N.bin
	DIRECTSUM=/Users/nathanvaughn/Desktop/CO_meshes/ex_st_coulomb_$N.bin
	OUTFILE=/Users/nathanvaughn/Desktop/CO_meshes/coulomb_tc.csv
	#SOURCES=/scratch/krasny_fluxg/njvaughn/CO_meshes/S$N.bin    
	#TARGETS=/scratch/krasny_fluxg/njvaughn/CO_meshes/T$N.bin
	#OUTFILE=/home/njvaughn/synchronizedDataFiles/testing/coulomb.csv 
	#DIRECTSUM=/scratch/krasny_fluxg/njvaughn/CO_meshes/ex_st_coulomb_$N.bin

	#direct-cpu   $SOURCES $TARGETS $DIRECTSUM $OUTFILE $N $N $KAPPA $POTENTIALTYPE $NUMDEVICES $NUMTHREADS
	
	for ORDER in 8
	do   
	     for THETA in 0.7   
	     do
	 		tree-cpu   	$SOURCES $TARGETS $DIRECTSUM $OUTFILE $N $N $THETA $ORDER \
	 					$TREETYPE $MAXPARNODE $KAPPA $POTENTIALTYPE $PFLAG $SFLAG $DFLAG $BATCHSIZE \
	 					$NUMDEVICES $NUMTHREADS
	     done
	done

done
