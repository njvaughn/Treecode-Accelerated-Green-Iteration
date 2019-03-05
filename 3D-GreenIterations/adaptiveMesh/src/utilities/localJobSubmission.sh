#!/usr/bin/env bash
export OMP_NUM_THREADS=6


cd /Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/KohnShamTests


DOMAIN=20
MINDEPTH=3
SINGSUBT=1

#INPUTFILE='../src/utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv'
INPUTFILE='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv'
#INPUTFILE='../src/utilities/molecularConfigurations/carbonAtomAuxiliary.csv'
#INPUTFILE='../src/utilities/molecularConfigurations/berylliumAuxiliary.csv'
#INPUTFILE='../src/utilities/molecularConfigurations/lithiumAuxiliary.csv'
#INPUTFILE='../src/utilities/molecularConfigurations/hydrogenMoleculeAuxiliary.csv'

VTKFILEDIR='/home/njvaughn/O_with_anderson/LW5_1500o5_plots_eigRes'


GRADIENTFREE='True'
GPUPRESENT='False'

TREECODE='False'
TREECODEORDER=0
THETA=0.0
MAXPARNODE=8000
BATCHSIZE=8000


GAUSSIANALPHA=1.0

MESHTYPE='Krasny'


MIXINGPARAMETER=0.5   
 
INTRASCFTOLERANCE=1e-7  
INTERSCFTOLERANCE=5e-6 
   
MIXING='Anderson'   
CELLORDER=3

DEPTHATATOMS=2 
	

for SMOOTHINGEPS in 0.0001 
do
for MAXDEPTH in 13 
do								  					 
for MESHPARAM4 in 100
do
	for MESHPARAM2 in 1.0 
	do   
		for MESHPARAM3 in 0.2
		do 
			for MESHPARAM1 in 1.0
			    
			do
			OUTPUTFILE="/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/localOxygen_VextRegularized/ds_cpu_cellOrder${CELLORDER}_smoothingEps${SMOOTHINGEPS}_maxDepth${MAXDEPTH}_minDepth${MINDEPTH}_${MESHPARAM1}_${MESHPARAM2}_${MESHPARAM3}_${MESHPARAM4}.csv"
			python -u testBatchGreenIterations_KS.py 	$DOMAIN $MINDEPTH $MAXDEPTH $DEPTHATATOMS $CELLORDER $SINGSUBT \
																		$SMOOTHINGEPS $GAUSSIANALPHA $MESHTYPE $MESHPARAM1 $MESHPARAM2 $INTERSCFTOLERANCE $INTRASCFTOLERANCE \
																		$OUTPUTFILE $INPUTFILE $VTKFILEDIR $GRADIENTFREE $MIXING $MIXINGPARAMETER \
																		$GPUPRESENT $TREECODE $TREECODEORDER $THETA $MAXPARNODE $BATCHSIZE $MESHPARAM3 $MESHPARAM4  
			done
		done
	done
done
done
done