# Treecode-Accelerated Green Iteration

This repository contains the code used to investigate treecode-accelerated Green iteration (**TAGI**) for Kohn-Sham Density Functional Theory.
The method uses Green's fucntions to invert differential operators in the Kohn-Sham problem.
Rather than solve eigenvalue problems for differential operators, this approach solves fixed-point problems for integral operators.
It is a real-space method using adaptive mesh refinement and Clenshaw-Curtis quadrature to discretize the integrals.
It uses the **BaryTree** library to compute fast approximations to the discrete convolutions on GPUs.

Details of the method and results can be found in the paper and the thesis.


# File Structure
The main directory contains two sub-directories, **1D-GreenIt-EigOne** and **3D-GreenIterations**.
1D-GreenIt-EigOne contains routines that compare Green Iteration to an unpublished algorithm called eigenvalue-one for several 1-dimensional test problems for the Schrodinger equation.
3D-GreenIterations contains the routines that perform 3D Kohn-Sham DFT calculations using Green Iteration.  


# Building

TAGI has several dependencies.
First, there are several python modules that must be installed such as pyLibXC and mpi4py (which requries first installing MPI).
Second, there are several external libraries that must be installed (**BaryTree**, **Zoltan**).
Third, there are several internal C libraries that must be installed (see TAGI/3D-GreenIterations/src/utilities/Makefile).
To run on CPUS, these C routines can be compiled with any C compiler, however to run on GPUs they must be compiled with PGI.  


# Running

Running TAGI requires providing an input file containing atomic positions and a set of run parameters.  Example atomic input files can be found in TAGI/3D-GreenIterations/src/molecularConfigurations, and example run submission files can be found in TAGI/3D-GreenIterations/run-files.
TAGI can be run in serial or in parallel using MPI.
All of the most expensive steps of TAGI are GPU accelerated, provided they have been compiled with PGI, using one GPU per MPI rank.  Calculations have been tested and performed on up to 32 GPUs.
