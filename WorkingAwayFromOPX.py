import numpy as np


def Calc_corr_mat(phi, g):
    "phi in degrees"
    c00 = 1; 
    c01 = -np.tan(phi* np.pi / 180)
    c10 = 0
    c11 = 1 / ((1 + g) * np.cos(phi* np.pi / 180))
    for value in [c00, c01, c10, c11]: 
        if (value >2) or (value <-2): 
            raise ValueError('overflow in correction matrix with phi, g = {},{}. {} is out of range (-2,2)'.format(phi, g, value))
    return [c00, c01, c10, c11]