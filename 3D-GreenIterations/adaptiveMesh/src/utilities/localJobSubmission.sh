#!/usr/bin/env bash
export OMP_NUM_THREADS=6


cd /Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/KohnShamTests


DOMAIN=20
MINDEPTH=3
SINGSUBT=1

#INPUTFILE='../src/utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv'
#INPUTFILE='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv'
#INPUTFILE='../src/utilities/molecularConfigurations/carbonAtomAuxiliary.csv'
INPUTFILE='../src/utilities/molecularConfigurations/berylliumAuxiliary.csv'
#INPUTFILE='../src/utilities/molecularConfigurations/lithiumAuxiliary.csv'
#INPUTFILE='../src/utilities/molecularConfigurations/hydrogenMoleculeAuxiliary.csv'

VTKFILEDIR='/home/njvaughn/O_with_anderson/LW5_1500o5_plots_eigRes'


GRADIENTFREE='True'
GPUPRESENT='False'

TREECODE='False'
TREECODEORDER=6
THETA=0.8
MAXPARNODE=8000
BATCHSIZE=8000


GAUSSIANALPHA=1.0

MESHTYPE='LW5'

RESTART='False'

MIXINGPARAMETER=0.75 
MIXINGHISTORY=10  
 
INTRASCFTOLERANCE=1e-6 
INTERSCFTOLERANCE=5e-5 
   
MIXING='Anderson'    
SMOOTHINGEPS=0.0
for ADDITIONALDEPTHATATOMS in 0
do
for CELLORDER in 4
do	 
for BASE in 1
do
for MAXDEPTH in 20   
do								  					 
for MESHPARAM4 in 999 
do
	for MESHPARAM3 in 999
	do    
		for MESHPARAM2 in 999
		do 
			for MESHPARAM1 in 200			    
			do
			OUTPUTFILE="/Users/nathanvaughn/Documents/synchronizedDataFiles/krasnyMeshTests/Slice_Testing/Be_LW5_${MESHPARAM1}.csv"

			python -u testBatchGreenIterations_KS.py 	$DOMAIN $MINDEPTH $MAXDEPTH $ADDITIONALDEPTHATATOMS $CELLORDER $SINGSUBT \
																		$SMOOTHINGEPS $GAUSSIANALPHA $MESHTYPE $MESHPARAM1 $MESHPARAM2 $INTERSCFTOLERANCE $INTRASCFTOLERANCE \
																		$OUTPUTFILE $INPUTFILE $VTKFILEDIR $GRADIENTFREE $MIXING $MIXINGPARAMETER $MIXINGHISTORY \
																		$GPUPRESENT $TREECODE $TREECODEORDER $THETA $MAXPARNODE $BATCHSIZE $MESHPARAM3 $MESHPARAM4 $BASE $RESTART
			done  
		done
	done 
	
done
done
done
done
done


