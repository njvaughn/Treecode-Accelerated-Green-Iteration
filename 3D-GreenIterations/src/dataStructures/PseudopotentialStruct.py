'''
@author: nathanvaughn
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, CubicSpline, UnivariateSpline
import os
import upf_to_json 
import time

import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ONCV_PSP(object):
    '''
    The ONCV Pseudopotential object.  Will contain the interpolators needed for the nonlocal potential,
    the density initialization, etc.
    '''
    def __init__(self,atomicNumber):
        '''
        PSP Constructor
        '''
        self.atomicNumber=atomicNumber
        self.atomicNumberToAtomicSymbol()
#         pspFile_local = "/Users/nathanvaughn/Desktop/ONCV_PSPs_Z/"+self.atomicSymbol+"_ONCV_PBE-1.0.upf"
        pspFile_local = "/Users/nathanvaughn/Desktop/ONCV_PSPs_LDA/"+self.atomicSymbol+"_ONCV_LDA.upf"
#         pspFile_remote = "/home/njvaughn/ONCV_PSPs_Z/"+self.atomicSymbol+"_ONCV_PBE-1.0.upf"
        pspFile_remote = "/home/njvaughn/ONCV_PSPs_LDA/"+self.atomicSymbol+"_ONCV_LDA.upf"
        try:
            upf_str = open(pspFile_local, 'r').read()
            pspFile = pspFile_local
        except FileNotFoundError:
            upf_str = open(pspFile_remote, 'r').read()
            pspFile = pspFile_remote
                
        psp_temp = upf_to_json.upf_to_json(upf_str, pspFile)
        self.psp = psp_temp['pseudo_potential']
        self.setProjectorInterpolators()
        self.setDensityInterpolator()
        self.setLocalPotentialInterpolator()
        
    def atomicNumberToAtomicSymbol(self):
        if self.atomicNumber==1:
            self.atomicSymbol="H"
        elif self.atomicNumber==4:
            self.atomicSymbol="Be"
        elif self.atomicNumber==6:
            self.atomicSymbol="C"
        elif self.atomicNumber==8:
            self.atomicSymbol="O"
        elif self.atomicNumber==14:
            self.atomicSymbol="Si"
        elif self.atomicNumber==22:
            self.atomicSymbol="Ti"
        else:
            rprint(rank,"Need to add atomic number %i to ONCV_PSP.atomicNumberToAtomicSymbol()" %self.atomicNumber)
            exit(-1)
        
    def setDensityInterpolator(self,verbose=0):
        r = np.array(self.psp['radial_grid'])
        self.maxRadialGrid = r[-1]
        density = np.array(self.psp['total_charge_density'])
        ## Is it okay to set the boundary condition to zero?  
        self.densityInterpolator = InterpolatedUnivariateSpline(r[:],density[:],k=3,ext='zeros')
        
        # Setup decaying exponential for extrapolation beyond rcutoff.
        a = r[-3]
        b = r[-1]
        
        da = self.densityInterpolator(a)
        db = self.densityInterpolator(b)
        
        logslope = (np.log(db)-np.log(da)) / (b-a)
        self.densityFarFieldExponentialCoefficient = db
        self.densityFarFieldExponentialDecayRate = logslope
        
        # Setup linear function for extrapolation beyond rcutoff.
        a = r[1]
        b = r[3]
        
        da = self.densityInterpolator(a)/(4*np.pi*a*a)
        db = self.densityInterpolator(b)/(4*np.pi*b*b)
        
        slope = (db-da) / (b-a)
        
        self.densityNearFieldLinearSlope=slope
        self.densityNearFieldHeight = da
        
        
#         self.firstNonzeroDensityTimes4pirr = density[1]/(4*np.pi*r[1]*r[1])
        self.radialCutoff = r[-1]
        self.innerCutoff  = r[1]
      
    def densityFarFieldExtrapolationFunction(self,r):
        return self.densityFarFieldExponentialCoefficient * np.exp( self.densityFarFieldExponentialDecayRate * (r-self.maxRadialGrid))
        
    def densityNearFieldExtrapolation(self,r):
        return self.densityNearFieldLinearSlope * (r-self.innerCutoff) + self.densityNearFieldHeight
        
    def evaluateDensityInterpolator(self,r):
        
        Rho = np.where( r<self.radialCutoff, self.densityInterpolator(r) / (4*np.pi*r*r) ,self.densityFarFieldExtrapolationFunction(r) / (4*np.pi*r*r) )

#         nr=len(r)
#         Rho = np.zeros(nr) 
#         for i in range(nr):
#             try:
#                 Rho[i] = self.densityInterpolator(r[i]) / (4*np.pi*r[i]*r[i])
#             except ValueError:
#                 if r[i]>self.radialCutoff:
#                     Rho[i] = self.densityFarFieldExtrapolationFunction(r[i]) / (4*np.pi*r[i]*r[i])
#                 elif r[i]<self.innerCutoff:
#                     rprint(rank,"DENSITY INTERPOLATOR FAILED EVALUATING")
#                     exit(-1)
# #                     Rho[i] = self.densityNearFieldExtrapolation(r[i]) # already has 4pirr taken care of
        return Rho
        
    def OLD_setLocalPotentialInterpolator(self,verbose=0):
        r = np.array(self.psp['radial_grid'])
        local_potential = np.array(self.psp['local_potential'])   # upf_to_json has already done the Rydberg-to-Hartree conversion by dividing Vloc by 2
        self.localPotentialInterpolator = InterpolatedUnivariateSpline(r,local_potential,k=3,ext='raise')
#         self.localPotentialInterpolator = UnivariateSpline(r,local_potential,k=1,s=0.01,ext='raise')
      
    def OLD_evaluateLocalPotentialInterpolator(self,r):
        nr=len(r)
        Vloc = np.zeros(nr)
        for i in range(nr):
            try:
                Vloc[i] =  self.localPotentialInterpolator(r[i])
            except ValueError:
                Vloc[i] = -self.psp['header']['z_valence']/r[i]
        return Vloc
    
    def setLocalPotentialInterpolator(self,verbose=0):
        r = np.array(self.psp['radial_grid'])
        local_potential = np.array(self.psp['local_potential'])   # upf_to_json has already done the Rydberg-to-Hartree conversion by dividing Vloc by 2
        
        # left boundary slope (compute slope across last two points, specify this as the left slope)
        a=r[0]
        b=r[1]
        fa=local_potential[0]
        fb=local_potential[1]
        slopeL = (fb-fa)/(b-a)
        
        # right boundary slope  (should be the same as slope of -Z/r, Z/r^2
#         slopeR = self.psp['header']['z_valence']/r[-1]**2
        slopeR = -local_potential[-1]/r[-1]
        
        self.localPotentialInterpolator = CubicSpline(r,local_potential,bc_type=((1,slopeL),(1,slopeR)),extrapolate=True)
#         self.localPotentialInterpolator = CubicSpline(r,local_potential,bc_type=((2,0),(1,slopeR)),extrapolate=True)
        
    
    def evaluateLocalPotentialInterpolator(self,r,timer=False):
        # use np.where to convert this to two vectorized calls, one to the interpolator for r<self.maxRadialGrid, one for r>
        # this requires changing the 'raise' to 'zeros' or something else.  The far field is handled with the where, not by the intepolator.
        # For cubic spline, use Extrapolate=True.  This way the call won't raise an exception.  The extrapolation will actually be done with np.where.
        if timer==True: start=time.time()
        Vloc = np.where(r<self.maxRadialGrid, self.localPotentialInterpolator(r), -self.psp['header']['z_valence']/r)
        if timer==True: end=time.time()
        if timer==True: rprint(rank,"Evaluating Vloc interpolator took %f seconds." %(end-start))

#         nr=len(r)
#         Vloc = np.zeros(nr)
#         for i in range(nr):
#             try:
#                 Vloc[i] =  self.localPotentialInterpolator(r[i])
#                 if np.isnan(Vloc[i]):
#                     raise ValueError
#             except ValueError:
#                 if r[i]>self.radialCutoff:
#                     Vloc[i] = -self.psp['header']['z_valence']/r[i]
#                 else:
#                     rprint(rank,"Warning: local PSP interpolator threw an error, not due to extrapolation beyond cutoff radius.")
        return Vloc
    
    
        
    def setProjectorInterpolators(self,verbose=0):
        
        self.projectorInterpolators = {}
        r = np.array(self.psp['radial_grid'])
        if verbose>0: 
            rprint(rank,"Number of porjectors: ", self.psp['header']['number_of_proj'])
            
#         self.projectorNearFieldSlopes = np.zeros(self.psp['header']['number_of_proj'])
#         self.projectorNearFieldOffsets = np.zeros(self.psp['header']['number_of_proj'])
        
        for i in range(self.psp['header']['number_of_proj']):
            
            if verbose>0: 
                rprint(rank,"Creating interpolator for projector %i with angular momentum %i" %(i,self.psp['beta_projectors'][i]['angular_momentum']))
            
            proj = np.array(self.psp['beta_projectors'][i]['radial_function'])
            length_of_projector_data = len(proj)
            
#             rprint(rank,"Last r value in projector interpolator: ", r[length_of_projector_data-1])
            
            self.projectorCutoffRadius=r[length_of_projector_data-1] 
            
#             self.projectorInterpolators[str(i)] = InterpolatedUnivariateSpline(r[:length_of_projector_data],proj,k=3,ext='zeros') # is ext='zeros' okay?  Could do some decay instead
            self.projectorInterpolators[str(i)] = CubicSpline(r[:length_of_projector_data],proj,bc_type=((2,0),(2,0)),extrapolate=True)
#             self.projectorInterpolators[str(i)] = CubicSpline(r[1:length_of_projector_data],proj[1:length_of_projector_data]/r[1:length_of_projector_data],bc_type=((2,0),(2,0)),extrapolate=True)
#             self.projectorInterpolators[str(i)] = InterpolatedUnivariateSpline(r[1:length_of_projector_data],proj[1:]/r[1:length_of_projector_data],k=3,ext='zeros') # is ext='zeros' okay?  Could do some decay instead
        return
    
    def evaluateProjectorInterpolator(self,idx,r,timer=False):
        # if zeroes are okay for the extrapolation, then this is okay.  If not, then need to wrap in try/except.
        # However, for 1 million points, the try/except method takes 8.5 seconds, the vectorized method takes 0.034 seconds.  Big difference.
#         return self.projectorInterpolators[str(idx)](r)/r ## dividing by r because it UPF manual says the provided fields are really r*Beta
        if timer==True: start=time.time()
        
#         output =  self.projectorInterpolators[str(idx)](r)/r
        output = np.where(r<self.projectorCutoffRadius, self.projectorInterpolators[str(idx)](r)/r, 0.0) 
#         output = np.where(r<self.projectorCutoffRadius, self.projectorInterpolators[str(idx)](r), 0.0) 
        
#         nr=len(r)
#         output = np.zeros(nr)
#         for i in range(nr):
#             try:
#                 output[i] =  self.projectorInterpolators[str(idx)](r[i])/r[i]
# #                 output[i] =  self.projectorInterpolators[str(idx)](r[i])
#             except ValueError as VE:
#                 if r[i] > self.projectorCutoffRadius:
#                     output[i] = 0.0
# #                     output[i] =  self.projectorInterpolators[str(idx)](self.projectorCutoffRadius)/self.projectorCutoffRadius
#                 elif r[i] < self.innerCutoff:
#                     rprint(rank,"Projector interpolator out of range: r was below the inner cutoff radius.")
#                     exit(-1)
#                 else:
#                     rprint(rank,"Something went wrong in projector interpolator for r=", r[i])
#                     rprint(rank,"Inner and outer cutoffs: ", self.innerCutoff, self.projectorCutoffRadius)
#                     rprint(rank,"ValueError: ", VE)
#                     exit(-1)
        if timer==True: end=time.time()
        if timer==True: rprint(rank,"Evaluating projector interpolator took %f seconds." %(end-start))
        return output
   
    def plotProjectors(self):
        
        r = np.array(self.psp['radial_grid'])
        plt.figure()
        for i in range(self.psp['header']['number_of_proj']):
            proj = np.array(self.psp['beta_projectors'][i]['radial_function'])
            length_of_projector_data = len(proj)
            plt.plot(r[:length_of_projector_data], proj,'.',label="projector data %i" %i)
            plt.title("ONCV Projector Data")
            
            
        r = np.array(self.psp['radial_grid'])
        plt.figure()
        for i in range(self.psp['header']['number_of_proj']):
            proj = np.array(self.psp['beta_projectors'][i]['radial_function'])
            length_of_projector_data = len(proj)
            rmax = r[length_of_projector_data]
            
            rInterp = np.linspace(0,rmax,1000)
            plt.plot(rInterp,self.evaluateProjectorInterpolator(i,rInterp),label="projector %i" %i)
        plt.title("Projector Cubic Spline Interpolators")
#         plt.legend()
        
        plt.show()
        
    def plotDensity(self):
        r = np.array(self.psp['radial_grid'])
        density = np.array(self.psp['total_charge_density'])
        rInterp = np.linspace(0,2*r[-1],1000)
        plt.figure()
        plt.plot(r,density,'r.',label="ONCV Data")
        plt.plot(rInterp, self.evaluateDensityInterpolator(rInterp), 'b-', label="Interpolation")
        plt.legend()
        plt.title("Density")
        plt.show()
        
    def plotLocalPotential(self,ylims=[-1.2, -10, -100]):
        r = np.array(self.psp['radial_grid'])
        local_potential = np.array(self.psp['local_potential'])
        rInterp = np.linspace(1e-2,1.5*r[-1],5000)
        Vfarfield = -self.psp['header']['z_valence']/rInterp
        V_all_electron = -self.atomicNumber/rInterp
        VlocInterp = self.evaluateLocalPotentialInterpolator(rInterp)
        
        for ylimfactor in ylims:
            plt.figure()
            plt.plot(rInterp, V_all_electron, 'r-', label="-Z/r")
            plt.plot(rInterp, Vfarfield, 'g-', label="-Zvalence/r")
            plt.plot(r,local_potential,'c.',label="ONCV Data")
            plt.plot(rInterp, VlocInterp, 'b--', label="Interpolation")
            plt.legend()
            plt.title("Local Potential")
            plt.ylim([-ylimfactor*np.min(local_potential),2])
            plt.show()