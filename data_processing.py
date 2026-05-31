# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 09:55:02 2023

@author: Shay
"""
import numpy as np
# from sklearn.cluster import KMeans

def S21_2fr_notch(Ql, Qc, phi, S21, A, f):

    term1 = (Ql / Qc) * np.exp(1j * phi*np.pi/180)
    term2 = 1 - (S21 / A) 
    term3 = 1 / (2j * Ql)
    
    return (f / ( (term1 / term2 - 1) * term3 + 1 )).real


def round_value_by_error(value, error, digit = 1):
    """Gets value and error and returns them (in an array) after rounding up or down for the first significant digit of the error.
    can also get a list of values and a list of errors"""
    if type(value) is list or type(value) is np.ndarray:
        values = np.zeros(len(value))
        errors = np.zeros(len(error))
        i=0
        for val,er in zip(value,error):
            values[i],errors[i] = round_value_by_error(val,er)
            i+=1
        return [values, errors]
    
    if value == None or error == None:
        return np.array([np.nan, np.nan])
    if error == np.inf or error == -np.inf:
        return np.array([value, np.inf])
    if error == 0 or error == np.nan:
            return np.array([value, 0])
    try:
        scale = int(-np.floor(np.log10(abs(error)))) + digit-1
        factor = 10**scale
        sgn = -1 if value < 0 else 1
        value = sgn * np.floor(abs(value) * factor) / factor if abs(value) * factor - np.floor(abs(value) * factor) <= 0.5 else sgn * np.ceil(abs(value) * factor) / factor
        error = np.floor(abs(error) * factor) / factor if (abs(error) * factor) - np.floor(abs(error)* factor)  <= 0.5 else np.ceil(abs(error) * factor) / factor
        return np.array([value, error])
    except:
        return np.array([value, 0])

def round_sig_dig(value,n):
    """ rounds value to n significant digits """
    if not value == 0:
        scale = int(-np.floor(np.log10(abs(value)))) + n-1
        factor = 10**scale
        if abs(value) * factor - np.floor(abs(value) * factor) <= 0.5:
            value = np.floor(abs(value) * factor) / factor
        else:
            value = np.ceil(abs(value) * factor) / factor
    return value

def correct_non_integer_demod(I,Q,t,T,IF,A = 1):
    """ corrects measured I,Q in case the chunk size is not a full cycle of the IF freq.
    I,Q - the measured values (single chunk).
    [t, t+T] - time range of chunk in ns (relative to time of 0 phase)
    IF - IF freq in Hz
    A - fudge factor. when A=0 no correction, when A=1 regular correction """
    
    t *= 1e-9
    T *= 1e-9
    
    omega = 2*np.pi*IF
    sinc = np.sin(omega*T)/(omega*T) * A
    cos2 = np.cos(omega*(2*t+T))
    sin2 = np.sin(omega*(2*t+T))
    
    det = 1 - (sinc*cos2)**2
    I_corr = (1/det)*( (1-sinc*cos2)*I - sinc*sin2*Q )
    Q_corr = (1/det)*( -sinc*sin2*I + (1+sinc*cos2)*Q ) 
    
    return (I_corr,Q_corr)

def scale_data_units(data):
    """Takes data and returns the data scaled closest to unity and the units prefix. Given data must not be scaled"""
    max_data = np.max(np.abs(data))
    scale = int(-np.floor(np.log10(max_data)))
    scale = np.floor(scale/3) if scale<0 else np.ceil(scale/3)
    factor = 10**(scale*3)
    if factor > 1 : 
        data = data * int(factor)
    else :
        data = data * factor
    units_prefix = ''
    if scale == 1: units_prefix +='m'
    elif scale == 2: units_prefix +=r'$\mu$'
    elif scale == 3: units_prefix +='n'
    elif scale == -1: units_prefix +='K'
    elif scale == -2: units_prefix +='M'
    elif scale == -3: units_prefix +='G'
    return data, units_prefix, factor

def histogram_fidelity(hist1, hist2, hist3 = None, hist4 = None):
    if hist3 is None:
        mean1 = np.mean(hist1)
        mean2 = np.mean(hist2)
        thresh = np.mean([mean1, mean2])
        fdlty=1
        
        for i in range(len(hist1)):
            if mean1 > mean2:
                if hist1[i] <= thresh: fdlty = fdlty - 1/(len(hist1))
                if hist2[i] >= thresh: fdlty = fdlty - 1/(len(hist2))
            else:
                if hist1[i] >= thresh: fdlty = fdlty - 1/(len(hist1))
                if hist2[i] <= thresh: fdlty = fdlty - 1/(len(hist2))
        return fdlty, thresh
    else:
        mean1 = np.mean(hist1)
        mean2 = np.mean(hist2)
        mean3 = np.mean(hist3)
        mean4 = np.mean(hist4)
        thresh = [np.mean([mean1, mean2]), np.mean([mean2, mean3]), np.mean([mean3, mean4])]
        is_ascending =  thresh[1]>thresh[0]
        fdlty=[1,1,1,1]
        
        #Sorry for the following bad code:
        ln = len(hist1)
        for i in range(ln):
            if is_ascending:
                if hist1[i] >= thresh[0]: fdlty[0] -= 1/(ln)
                if hist2[i] <= thresh[0]: fdlty[0] -= 1/(ln)
                if hist3[i] <= thresh[0]: fdlty[0] -= 1/(ln)
                if hist4[i] <= thresh[0]: fdlty[0] -= 1/(ln)
            else:
                if hist1[i] <= thresh[0]: fdlty[0] -= 1/(ln)
                if hist2[i] >= thresh[0]: fdlty[0] -= 1/(ln)
                if hist3[i] >= thresh[0]: fdlty[0] -= 1/(ln)
                if hist4[i] >= thresh[0]: fdlty[0] -= 1/(ln)
                
        for i in range(ln):
            if is_ascending:
                if hist1[i] >= thresh[0] and hist1[i]<=thresh[1]: fdlty[1] -= 1/(ln)
                if hist2[i] <= thresh[0] or hist2[i]>=thresh[1]: fdlty[1] -= 1/(ln)
                if hist3[i] >= thresh[0] and hist3[i]<=thresh[1]: fdlty[1] -= 1/(ln)
                if hist4[i] >= thresh[0] and hist4[i]<=thresh[1]: fdlty[1] -= 1/(ln)
            else:
                if hist1[i] <= thresh[0] and hist1[i]>=thresh[1]: fdlty[1] -= 1/(ln)
                if hist2[i] >= thresh[0] or hist2[i]<=thresh[1]: fdlty[1] -= 1/(ln)
                if hist3[i] <= thresh[0] and hist3[i]>=thresh[1]: fdlty[1] -= 1/(ln)
                if hist4[i] <= thresh[0] and hist4[i]>=thresh[1]: fdlty[1] -= 1/(ln)
                
        for i in range(ln):
            if is_ascending:
                if hist1[i] >= thresh[1] and hist1[i]<=thresh[2]: fdlty[2] -= 1/(ln)
                if hist3[i] <= thresh[1] or hist3[i]>=thresh[2]: fdlty[2] -= 1/(ln)
                if hist2[i] >= thresh[1] and hist2[i]<=thresh[2]: fdlty[2] -= 1/(ln)
                if hist4[i] >= thresh[1] and hist4[i]<=thresh[2]: fdlty[2] -= 1/(ln)
            else:
                if hist1[i] <= thresh[1] and hist1[i]>=thresh[2]: fdlty[2] -= 1/(ln)
                if hist3[i] >= thresh[1] or hist3[i]<=thresh[2]: fdlty[2] -= 1/(ln)
                if hist2[i] <= thresh[1] and hist2[i]>=thresh[2]: fdlty[2] -= 1/(ln)
                if hist4[i] <= thresh[1] and hist4[i]>=thresh[2]: fdlty[2] -= 1/(ln)
                
        for i in range(ln):
            if is_ascending:
                if hist1[i] >= thresh[2]: fdlty[3] -= 1/(4*ln)
                if hist2[i] >= thresh[2]: fdlty[3] -= 1/(4*ln)
                if hist3[i] >= thresh[2]: fdlty[3] -= 1/(4*ln)
                if hist4[i] <= thresh[2]: fdlty[3] -= 1/(4*ln)
            else:
                if hist1[i] <= thresh[2]: fdlty[3] -= 1/(ln)
                if hist2[i] <= thresh[2]: fdlty[3] -= 1/(ln)
                if hist3[i] <= thresh[2]: fdlty[3] -= 1/(ln)
                if hist4[i] >= thresh[2]: fdlty[3] -= 1/(ln)
                
        return np.round(fdlty, 3), thresh
        
    
def data_to_sigma_z(data, data_g, data_e, err=0, data_g_err=0, data_e_err=0, is_thresholding = False):
    if not is_thresholding:
        pop_g = (data-data_e)/(data_g-data_e)
        pop_e = (data_g-data)/(data_g-data_e)
        
        pop_g_numerator = (data-data_e)
        
        pop_e_numerator = (data_g-data)
        
        
        denominator = (data_g-data_e)
        
        sigma_z = pop_e-pop_g
        
        if isinstance(err, (np.ndarray,list,tuple, float)): 
            pop_g_numerator_err = np.sqrt(err**2+data_e_err**2)
            pop_e_numerator_err = np.sqrt(err**2+data_g_err**2)
            denominator_err = np.sqrt(data_e_err**2+data_g_err**2)
            pop_e_err = np.sqrt((pop_e_numerator_err/denominator)**2+(denominator_err*pop_e_numerator/denominator**2)**2)
            pop_g_err = np.sqrt((pop_g_numerator_err/denominator)**2+(denominator_err*pop_g_numerator/denominator**2)**2)
            sigma_z_err = np.sqrt(pop_e_err**2+pop_g_err**2)
        else:
            sigma_z_err = 0
        return sigma_z, sigma_z_err
    else:
        if data_g>data_e:
            data_g, data_e, data = data_g*-1, data_e*-1, data*-1
        thresh = np.mean([data_g, data_e])
        inds = np.where(data<thresh)
        thresh_data = np.ones(data.shape)
        thresh_data[inds] = -thresh_data[inds]
        return thresh_data, None
    

def normalize_data(data):
    factor = 1/(data.max()-data.min())
    data = data * factor
    data = (data - (data.max()+data.min())/2)*2
    data = np.clip(data, -1, 1)
    return data,factor*2





if __name__ == '__main__':
    data_g = 1
    data_e = 3
    data = np.array([0.1,0.9,2,2.5,1.1,1.3])
    
    a = data_to_sigma_z(data,data_g,data_e,is_thresholding=True)
    print(a)