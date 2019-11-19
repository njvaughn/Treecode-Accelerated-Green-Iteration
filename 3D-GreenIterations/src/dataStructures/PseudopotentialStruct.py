'''
@author: nathanvaughn
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import os
import upf_to_json 



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
        pspFile = "/Users/nathanvaughn/Desktop/ONCV_PSPs_Z/"+str(atomicNumber)+"_ONCV_PBE-1.0.upf"
        upf_str = open(pspFile, 'r').read()
        psp_temp = upf_to_json.upf_to_json(upf_str, pspFile)
        self.psp = psp_temp['pseudo_potential']
        self.setProjectorInterpolators()
        self.setDensityInterpolator()
        self.setLocalPotentialInterpolator()
        
    def setDensityInterpolator(self,verbose=0):
        r = np.array(self.psp['radial_grid'])
        self.maxRadialGrid = r[-1]
        density = np.array(self.psp['total_charge_density'])
        ## Is it okay to set the boundary condition to zero?  
        self.densityInterpolator = InterpolatedUnivariateSpline(r,density,k=3,ext='raise')
        
        # Setup decaying exponential for extrapolation.
        a = r[-5]
        b = r[-1]
        
        da = self.densityInterpolator(a)
        db = self.densityInterpolator(b)
        
        logslope = (np.log(db)-np.log(da)) / (b-a)
        
        self.densityFarFieldExponentialCoefficient = db
        self.densityFarFieldExponentialDecayRate = logslope
      
    def densityExtrapolationFunction(self,r):
        return self.densityFarFieldExponentialCoefficient * np.exp( self.densityFarFieldExponentialDecayRate * (r-self.maxRadialGrid))
        
    def evaluateDensityInterpolator(self,r):
        # if zeroes are okay for the extrapolation, then this is okay.  If not, then need to wrap in try/except.
#         return self.densityInterpolator(r)
        nr=len(r)
        Rho = np.zeros(nr)
        for i in range(nr):
            try:
                Rho[i] = self.densityInterpolator(r[i])
            except ValueError:
                Rho[i] = self.densityExtrapolationFunction(r[i])
        return Rho
        
    def setLocalPotentialInterpolator(self,verbose=0):
        r = np.array(self.psp['radial_grid'])
        local_potential = np.array(self.psp['local_potential'])
        self.localPotentialInterpolator = InterpolatedUnivariateSpline(r,local_potential,k=3,ext='raise')
    
    def evaluateLocalPotentialInterpolator(self,r):
        nr=len(r)
        Vloc = np.zeros(nr)
        for i in range(nr):
            try:
                Vloc[i] = self.localPotentialInterpolator(r[i])
            except ValueError:
                Vloc[i] = -self.psp['header']['z_valence']/r[i]
        return Vloc
        
    def setProjectorInterpolators(self,verbose=0):
        
        self.projectorInterpolators = {}
        r = np.array(self.psp['radial_grid'])
        if verbose>0: 
            print("Number of porjectors: ", self.psp['header']['number_of_proj'])
            
        for i in range(self.psp['header']['number_of_proj']):
            
            if verbose>0: 
                print("Creating interpolator for projector %i with angular momentum %i" %(i,self.psp['beta_projectors'][i]['angular_momentum']))
            
            proj = np.array(self.psp['beta_projectors'][i]['radial_function'])
            length_of_projector_data = len(proj)
            
            self.projectorInterpolators[str(i)] = InterpolatedUnivariateSpline(r[:length_of_projector_data],proj,k=3,ext='zeros')
        return
    
    def evaluateProjectorInterpolator(self,idx,r):
        # if zeroes are okay for the extrapolation, then this is okay.  If not, then need to wrap in try/except.
        return self.projectorInterpolators[str(idx)](r)
   
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