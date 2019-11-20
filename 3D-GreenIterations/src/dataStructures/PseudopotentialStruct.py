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
        pspFile_local = "/Users/nathanvaughn/Desktop/ONCV_PSPs_Z/"+str(atomicNumber)+"_ONCV_PBE-1.0.upf"
        pspFile_remote = "/home/njvaughn/ONCV_PSPs_Z/"+str(atomicNumber)+"_ONCV_PBE-1.0.upf"
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
        
    def setDensityInterpolator(self,verbose=0):
        r = np.array(self.psp['radial_grid'])
        self.maxRadialGrid = r[-1]
        density = np.array(self.psp['total_charge_density'])
        ## Is it okay to set the boundary condition to zero?  
        self.densityInterpolator = InterpolatedUnivariateSpline(r[1:],density[1:],k=3,ext='raise')
        
        # Setup decaying exponential for extrapolation beyond rcutoff.
        a = r[-5]
        b = r[-1]
        
        da = self.densityInterpolator(a)
        db = self.densityInterpolator(b)
        
        logslope = (np.log(db)-np.log(da)) / (b-a)
        self.densityFarFieldExponentialCoefficient = db
        self.densityFarFieldExponentialDecayRate = logslope
        
        # Setup linear function for extrapolation beyond rcutoff.
        a = r[1]
        b = r[2]
        
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
        nr=len(r)
        Rho = np.zeros(nr)
        for i in range(nr):
            try:
                Rho[i] = self.densityInterpolator(r[i]) / (4*np.pi*r[i]*r[i])
            except ValueError:
                if r[i]>self.radialCutoff:
                    Rho[i] = self.densityFarFieldExtrapolationFunction(r[i]) / (4*np.pi*r[i]*r[i])
                elif r[i]<self.innerCutoff:
                    Rho[i] = self.densityNearFieldExtrapolation(r[i]) # already has 4pirr taken care of
        return Rho
        
    def setLocalPotentialInterpolator(self,verbose=0):
        r = np.array(self.psp['radial_grid'])
        local_potential = np.array(self.psp['local_potential'])   # upf_to_json has already done the Rydberg-to-Hartree conversion by dividing Vloc by 2
        self.localPotentialInterpolator = InterpolatedUnivariateSpline(r,local_potential,k=3,ext='raise')
    
    def evaluateLocalPotentialInterpolator(self,r):
        nr=len(r)
        Vloc = np.zeros(nr)
        for i in range(nr):
            try:
                Vloc[i] =  self.localPotentialInterpolator(r[i])
            except ValueError:
                Vloc[i] = -self.psp['header']['z_valence']/r[i]
        return Vloc
        
    def setProjectorInterpolators(self,verbose=0):
        
        self.projectorInterpolators = {}
        r = np.array(self.psp['radial_grid'])
        if verbose>0: 
            print("Number of porjectors: ", self.psp['header']['number_of_proj'])
            
#         self.projectorNearFieldSlopes = np.zeros(self.psp['header']['number_of_proj'])
#         self.projectorNearFieldOffsets = np.zeros(self.psp['header']['number_of_proj'])
        
        for i in range(self.psp['header']['number_of_proj']):
            
            if verbose>0: 
                print("Creating interpolator for projector %i with angular momentum %i" %(i,self.psp['beta_projectors'][i]['angular_momentum']))
            
            proj = np.array(self.psp['beta_projectors'][i]['radial_function'])
            length_of_projector_data = len(proj)
            
            self.projectorInterpolators[str(i)] = InterpolatedUnivariateSpline(r[:length_of_projector_data],proj,k=3,ext='raise') # is ext='zeros' okay?  Could do some decay instead
        return
    
    def evaluateProjectorInterpolator(self,idx,r):
        # if zeroes are okay for the extrapolation, then this is okay.  If not, then need to wrap in try/except.
#         return self.projectorInterpolators[str(idx)](r)/r ## dividing by r because it UPF manual says the provided fields are really r*Beta
        nr=len(r)
        output = np.zeros(nr)
        for i in range(nr):
            try:
                output[i] =  self.projectorInterpolators[str(idx)](r[i])/r[i]
            except ValueError:
                if r[i] > self.radialCutoff:
                    output[i] = 0.0
#                 elif r[i]<self.innerCutoff:
#                     output[i] = 0.0
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