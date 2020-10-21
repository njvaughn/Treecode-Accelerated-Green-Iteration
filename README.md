# Treecode-Accelerated Green Iteration

This repository contains the code used to investigate treecode-accelerated Green iteration for Kohn-Sham Density Functional Theory.
The method uses Green's fucntions to invert differential operators in the Kohn-Sham problem.
Rather than solve eigenvalue problems for differential operators, this approach solves fixed-point problems for integral operators.
It is a real-space method using adaptive mesh refinement and Clenshaw-Curtis quadrature to discretize the integrals.
It uses the BaryTree library to compute fast approximations to the discrete convolutions on GPUs.

Details of the method and results can be found in the paper and the thesis.


# File Structure
The main directory contains two sub-directories, 1D-GreenIt-EigOne and 3D-GreenIterations.
1D-GreenIt-EigOne contains routines that compare Green Iteration to an unpublished algorithm called eigenvalue-one for several 1-dimensional test problems for the Schrodinger equation.
3D-GreenIterations contains the routines that perform 3D Kohn-Sham DFT calculations using Green Iteration.  


# Building




# Running
