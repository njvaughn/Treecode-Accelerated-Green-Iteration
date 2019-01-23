#!/usr/bin/env bash

cd /Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/KohnShamTests
DOMAIN=20
MAXDEPTH=14
MINDEPTH=3
SINGSUBT=1
SMOOTHINGN=0
SMOOTHINGEPS=0

#INPUTFILE='../src/utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv'
#INPUTFILE='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv'
#INPUTFILE='../src/utilities/molecularConfigurations/carbonAtomAuxiliary.csv'
INPUTFILE='../src/utilities/molecularConfigurations/berylliumAuxiliary.csv'
#INPUTFILE='../src/utilities/molecularConfigurations/lithiumAuxiliary.csv'
#INPUTFILE='../src/utilities/molecularConfigurations/hydrogenMoleculeAuxiliary.csv'

GRADIENTFREE='True'
GPUPRESENT='False'

ORDER=5
MESHTYPE='Biros'
MESHPARAM=5e-5

MIXINGPARAMETER=0.5

INTRASCFTOLERANCE=1e-6
INTERSCFTOLERANCE=1e-6

MIXING='Anderson'	


OUTPUTFILE='/Users/nathanvaughn/Documents/GreenIterationOutputData/LW5o3_500_H2.csv'
VTKFILEDIR='/home/njvaughn/O_with_anderson/LW5_1500o5_plots_eigRes'


python -u testBatchGreenIterations_KS.py $DOMAIN $MINDEPTH $MAXDEPTH $ORDER $SINGSUBT \
	$SMOOTHINGN $SMOOTHINGEPS $MESHTYPE $MESHPARAM $INTERSCFTOLERANCE $INTRASCFTOLERANCE \
	$OUTPUTFILE $INPUTFILE $VTKFILEDIR $GRADIENTFREE $MIXING $MIXINGPARAMETER $GPUPRESENT
	
#python -u -m yep -v testBatchGreenIterations_KS.py $DOMAIN $MINDEPTH $MAXDEPTH $ORDER $SINGSUBT \
#	$SMOOTHINGN $SMOOTHINGEPS $MESHTYPE $MESHPARAM $INTERSCFTOLERANCE $INTRASCFTOLERANCE \
#	$OUTPUTFILE $INPUTFILE $VTKFILEDIR $GRADIENTFREE $MIXING $MIXINGPARAMETER $GPUPRESENT
	
	
