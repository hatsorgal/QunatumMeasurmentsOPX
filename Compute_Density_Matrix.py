# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:13:31 2020

@author: Eliya
"""

import numpy as np
import qutip as qt
from scipy.optimize import minimize
#%%

def calculate_eigvals(data, betas):
    betaII = betas[0]; betaZI = betas[1]; betaIZ = betas[2]; betaZZ = betas[3]
    
    II = 1
    ZZ = ((-data[1] - data[2] + 2 * betaII) / 2 / betaZZ                      + (-data[16] - data[17] + 2 * betaII) / 2 / betaZZ          )/ 2
    ZI = (( data[0] + data[2] - 2 * betaII) / 2 / betaZI                      + ( data[15] + data[17] - 2 * betaII) / 2 / betaZI          )/ 2 
    IZ = (( data[0] + data[1] - 2 * betaII) / 2 / betaIZ                      + ( data[15] + data[16] - 2 * betaII) / 2 / betaIZ          )/ 2
    YI = (( data[3] + data[6] - 2 * betaII) / 2 / betaZI                      + (-data[18] - data[23] + 2 * betaII) / 2 / betaZI          )/ 2
    YZ = (( data[3] - betaZI * YI - betaIZ * IZ - betaII) / betaZZ            + (-data[18] - betaZI * YI + betaIZ * IZ + betaII) / betaZZ )/ 2
    IY = (( data[11] + data[12] - 2 * betaII) / 2 / betaIZ                    + (-data[26] - data[27] + 2 * betaII) / 2 / betaIZ          )/ 2
    IX = ((-data[13] - data[14] + 2 * betaII) / 2 / betaIZ                    + ( data[28] + data[29] - 2 * betaII) / 2 / betaIZ          )/ 2
    XI = ((-data[10] - data[7] + 2 * betaII) / 2 / betaZI                     + ( data[22] + data[23] - 2 * betaII) / 2 / betaZI          )/ 2
    YY = (( data[4] - betaIZ * IY - betaZI * YI - betaII) / betaZZ            + ( data[19] + betaIZ * IY + betaZI * YI - betaII) / betaZZ )/ 2
    XX = (( data[9] + betaIZ * IX + betaZI * XI - betaII) / betaZZ            + ( data[24] - betaIZ * IX - betaZI * XI - betaII) / betaZZ )/ 2
    XZ = (( data[10] - data[7] + 2 * betaIZ * IZ) / 2 / betaZZ                + (-data[25] + data[22] - 2 * betaIZ * IZ) / 2 / betaZZ     )/ 2
    XY = ((-data[8] - betaZI * XI + betaIZ * IY + betaII) / betaZZ            + (-data[23] + betaZI * XI - betaIZ * IY + betaII) / betaZZ )/ 2
    YX = ((-data[5] + betaZI * YI - betaIZ * IX + betaII) / betaZZ            + (-data[20] + betaZI * YI - betaIZ * IX + betaII) / betaZZ )/ 2
    ZY = (( data[11] - data[12] - 2 * betaZI * ZI) / 2 / betaZZ               + ( data[27] - data[26] + 2 * betaZI * ZI) / 2 / betaZZ     )/ 2
    ZX = (( data[14] - data[13] + 2 * betaZI * ZI) / 2 / betaZZ               + ( data[28] - data[29] - 2 * betaZI * ZI) / 2 / betaZZ     )/ 2
    
    return II, ZZ, ZI, IZ, YI, YZ, IY, IX, XI, YY, XX, XZ, XY, YX, ZY, ZX

def matrix_from_t_list(t_list):
    T = qt.Qobj([[t_list[0]                   ,  0                          ,             0             ,            0],
                 [t_list[4]  + 1j * t_list[5] , t_list[1]                   ,             0             ,            0],
                 [t_list[10] + 1j * t_list[11], t_list[6] + 1j * t_list[7]  ,           t_list[2]       ,            0],
                 [t_list[14] + 1j * t_list[15], t_list[12] + 1j * t_list[13], t_list[8] + 1j * t_list[9],   t_list[3] ]  ])
    rho = T.dag() * T / ((T.dag() * T).tr())
    rho.dims = [[2,2],[2,2]]
    return  rho

def calculate_likelihood(t_list, eigvals, opers_list):

    rho = matrix_from_t_list(t_list)
    L = 0
    for i in range(len(t_list)):
        L = L + (eigvals[i] - qt.expect(opers_list[i], rho)) ** 2
    return L
    
def MLE_density_matrix(density_mat, eigvals):
    
    t_list0 = [density_mat[0][0][0], density_mat[1][0][1],  density_mat[2][0][2],  density_mat[3][0][3],
               np.real(density_mat[1][0][0]), np.imag(density_mat[1][0][0]), np.real(density_mat[2][0][1]), np.imag(density_mat[2][0][1]),
               np.real(density_mat[3][0][2]), np.imag(density_mat[3][0][2]), np.real(density_mat[2][0][0]), np.imag(density_mat[2][0][0]),
               np.real(density_mat[3][0][1]), np.imag(density_mat[3][0][1]), np.real(density_mat[3][0][0]), np.imag(density_mat[3][0][0]),]
    
    opers_list = [qt.tensor(qt.qeye(2),qt.qeye(2)),
      qt.tensor(qt.sigmaz(), qt.sigmaz()),
      qt.tensor(qt.sigmaz(),qt.qeye(2)),
      qt.tensor(qt.qeye(2),qt.sigmaz()),
      qt.tensor(qt.sigmay(),qt.qeye(2)),
      qt.tensor(qt.sigmay(),qt.sigmaz()),
      qt.tensor(qt.qeye(2),qt.sigmay()),
      qt.tensor(qt.qeye(2),qt.sigmax()),
      qt.tensor(qt.sigmax(),qt.qeye(2)),
      qt.tensor(qt.sigmay(),qt.sigmay()),
      qt.tensor(qt.sigmax(),qt.sigmax()),
      qt.tensor(qt.sigmax(),qt.sigmaz()),
      qt.tensor(qt.sigmax(),qt.sigmay()),
      qt.tensor(qt.sigmay(),qt.sigmax()),
      qt.tensor(qt.sigmaz(),qt.sigmay()),
      qt.tensor(qt.sigmaz(),qt.sigmax())]
    
    # constraints = 
    results = minimize(calculate_likelihood, t_list0, args = tuple([eigvals, opers_list]), options={'gtol': 1e-6, 'disp': True})
    return matrix_from_t_list(results.x)

def is_entangled(density_mat):
    """Performs Asher Peres' test for entanglement"""
    pt_mat = qt.partial_transpose(density_mat, [0,1])
    for eigenval in pt_mat.eigenenergies():
        if eigenval < 0: 
            print('The state is Entangled!')
            return True
    print('The state is Separable') 
    return False

def concurrence(density_mat):
    
    exp_values = qt.expect(   [qt.tensor(qt.sigmax(),qt.sigmax()),
                               qt.tensor(qt.sigmax(),qt.sigmay()),
                               qt.tensor(qt.sigmax(),qt.sigmaz()),
                               qt.tensor(qt.sigmay(),qt.sigmax()),
                               qt.tensor(qt.sigmay(),qt.sigmay()),
                               qt.tensor(qt.sigmay(),qt.sigmaz()),
                               qt.tensor(qt.sigmaz(),qt.sigmax()),
                               qt.tensor(qt.sigmaz(),qt.sigmay()),
                               qt.tensor(qt.sigmaz(), qt.sigmaz())],
                               density_mat)
    
    T = qt.Qobj([[exp_values[0],exp_values[1],exp_values[2]],
                 [exp_values[3],exp_values[4],exp_values[5]],
                 [exp_values[6],exp_values[7],exp_values[8]]])
    
    C = np.sqrt(((T.dag() * T).tr() - 1 ) / 2 ) 
    print('The concurrence is {}'.format(C)) 

    return C