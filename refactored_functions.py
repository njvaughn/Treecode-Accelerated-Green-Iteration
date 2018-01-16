'''
Created on Jan 15, 2018

@author: nathanvaughn
'''
from models import Poschl_Teller
import numpy as np


def setup_Poschl_Teller():
    Model = Poschl_Teller()
    Model.nx = 100 # number of grid points
    Model.N = 0 # energy level
    Model.xmin = -10 # left endpoint
    Model.xmax = 10 # right endpoint
    Model.grid = np.linspace(Model.xmin,Model.xmax,Model.nx)

    return Model
