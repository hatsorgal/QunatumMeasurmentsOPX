# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:51:27 2022]

@author: Shay
"""

import pathlib
import sys
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHubMeasurementControl\freqDomain') # for fridge 1
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\freqDomain') # for fridge 1
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl') #for HFSS computer
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\freqDomain') # for fridge 2 new pc

from scipy import signal, interpolate, optimize
from frequencyDomainBaseClass import FreqDo
from labrad.units import V,mV, s,ms,us,ns, Hz,kHz,MHz,GHz, Ohm, dBm, Value
from scipy import signal, fftpack
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.formatter.limits']=(-99,99)
mpl.rcParams['axes.formatter.useoffset']=False
import numpy as np
import os
import sys
import csv 
import json
import requests
from qkit.analysis.circle_fit.circle_fit_classic import circuit
import math as math
import time as time
import pickle

from pathlib import Path
import glob

#%%
# S12 means 1 is output from PNA and 2 is input to the PNA.

np.str = str # I added this
save_folder = r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\data\25_05_2026'

in_port = 1 #2 (into PNA from fridge)
out_port = 2 #1 (out of PNA into fridge - blue light)
freqdo = FreqDo('TOP',
                which_data = 'Both',
                save_folder = save_folder)

freqdo.pna.s_parameters(1,in_port,out_port)
freqdo.is_save_data = False


#%%

freqdo.one_tone_sweep(
    center = 6*1e9,
    span = 1e9,#5e6, #8e6, #was 30
    npts = 1001, # was 501 pts 
    power = -30,
    if_bw = 100, #0.001e3, #1e3
    navg = 1,
    which_data = 'phase',
    is_save_data = True,
    meas_name = 'one_tone_sweep',
    is_get_temp = False,
    is_plot = True, 
    phase_offset = 65,
    electrical_delay = 63.66e-9
    )
#%%
freqs = [ 6.44453 * GHz]
suspected_qubit_freqs = [2.4675e9]
freqdo.two_tone_sweep(
                        center = 2.468 * GHz,
                        span = 4 * MHz,
                        npts = 401,
                        mxg_power = -20 * dBm,
                        pna_freq = 6.44453 * GHz,
                        pna_power = -60 * dBm,
                        navg =1 ,
                        if_bw = 1 * Hz,
                        which_data = 'phase',
                        is_save_data=True,
                        is_fit = False,
                        n_cut = 1,
                        chi_guess = 3.0e-3,
                        alpha_guess = 0.4,
                    )

#%% create data structure for resonators params
res_params = {}



# Asaf 
"""
TODO aron, use these paramters for a power sweep and a temp sweep, 
go to -70,-60,-50 and then,-40, -30,-20 can be with double the BW and half the navg, 
we might also want a tempture sweep, will ask Shay.
"""
# res_params['alres'] = {
#                         'names': [f'Op{i+1}'for i in range(9)], # res ID, written in freq order
#                         'freq_center': np.array([4.4816195,4.462011,5.74496541,5.8596152,6.6292709,6.62611,7.44945,7.476804,7.34127])*1e9, #[Hz]
#                         'power_ro': np.array([-60]*9), #[dBm]
#                         'span': np.array([0.02,0.02,0.02,0.03,0.06,0.12,0.06,0.2,0.12])*2e6, #[Hz]
#                         'electrical_delay': 60.98e-9, # [s] # 58.93e-9
#                         'bandwidth': 0.01e3, #[Hz]  # 1e3
#                         'phase_offset': 85, #[deg]
#                         'averages': 10,
#                         'num_points': 1001
#                         }


# res_params['Gal_Res_Q2'] = {
#                         'names': [f'Op1'], # [f'Op{i+1}'for i in range(3)] res ID, written in freq order
#                         'freq_center': np.array([6289000000]), #[Hz]
#                         'power_ro': np.array([-50]), #[dBm]
#                         'span': np.array([4])*1e6, #[Hz]
#                         'electrical_delay': 63.66e-9, # [s] # 58.93e-9
#                         'bandwidth': 1, #[Hz]  # 1e3
#                         'phase_offset': 300, #[deg]
#                         'averages': 1,
#                         'num_points': 401
#                         }


res_params['Gal_Res_Q2'] = {
                        'names': [f'Op1'], # [f'Op{i+1}'for i in range(3)] res ID, written in freq order
                        'freq_center': np.array([6.2915])*1e9, #[Hz]
                        'power_ro': np.array([-30]), #[dBm]
                        'span': np.array([50])*1e6, #[Hz]
                        'electrical_delay': 63.66e-9, # [s] # 58.93e-9
                        'bandwidth': 10, #[Hz]  # 1e3
                        'phase_offset': 300, #[deg]
                        'averages': 1,
                        'num_points': 401
                        }

saveFolder = freqdo.save_folder
freqdo.pickle_save(to_save_dict=res_params, meas_name='res_params', folder_name=saveFolder, is_print=False)

#%% measure accurate one tone scan

res_params = freqdo.pickle_load(freqdo.save_folder+'\\res_params_24_05_2026___13_58_00')

# freqdo.save_folder = r'C:\Users\PHshayg-lab3\OneDrive - Technion\Omer\RADAR\fridge2_20250225\measurements\nominal'
if not os.path.exists(freqdo.save_folder):
    os.makedirs(freqdo.save_folder)
    
exp = 'Gal_Res_Q2'
freqdo.pna.electrical_delay(res_params[exp]['electrical_delay'])
freqdo.pna.phase_offset(res_params[exp]['phase_offset'])
names = res_params[exp]['names']
freq_center = res_params[exp]['freq_center']
power_ro = res_params[exp]['power_ro'] 
span = res_params[exp]['span']
for i in range(len(names)):
    print(names[i]+': \n')
    meas_name = f'one_tone_sweep {exp} {names[i]} {freq_center[i]/1e9} GHz {power_ro[i]} dBm'
    freqdo.one_tone_sweep(
                            center = freq_center[i],
                            span = span[i],
                            npts = res_params[exp]['num_points'],
                            power = power_ro[i],
                            navg = res_params[exp]['averages'],
                            if_bw = res_params[exp]['bandwidth'],
                            is_save_data = True,
                            meas_name = meas_name,
                            is_get_temp = True,
                            )
    print(freqdo.get_temperature())


#%% analyze accurate one tone scan

exp = 'Gal_Res_Q2'
names = res_params[exp]['names']

data_all = {name:{'freq':np.array([]),'freq_err':np.array([]),'kappa_i':np.array([]),'kappa_i_err':np.array([]),
              'kappa_c':np.array([]),'kappa_c_err':np.array([]),'kappa_l':np.array([]),
              'kappa_l_err':np.array([]),'theta0':np.array([]),'phi0':np.array([]),'phi0_err':np.array([]),
              'chi_square':np.array([]),'temperature':np.array([]),'ro_power':np.array([])} for name in names}

#plt.figure(figsize=(10,8))
#save_folder = Path(freqdo.save_folder) / "power_calibration"

for root, subdirs, files in os.walk(saveFolder):
    #print('folder:',root, '\n')
    
    #print("Files: ", files)
    
    for fname in files:
        
        counter = 1
        
        
        if '.pickle' not in fname:
            continue
        if 'data_all' in fname:
            continue
        
        
        #print("fname: ", fname)
        fname = fname[:-7]
        data = freqdo.pickle_load(root+'\\'+fname)
        
        print(data)
        
        f = data['frequency']
        logmag = data['logmag']
        phase = data['phase']*np.pi/180
        # temp = data['temp']
        freq = data['frequency'][data['frequency'].size//2]
        ro_pwr = data['power']
        
        sig = 10**(logmag/20) * (np.cos(phase) + 1j*np.sin(phase))
        
        sig_real_filt = signal.savgol_filter(np.real(sig), 21, 3)
        sig_imag_filt = signal.savgol_filter(np.imag(sig), 21, 3)
        sig_filt = sig_real_filt + 1j*sig_imag_filt

        port1 = circuit.notch_port()
        port1.add_data(f,sig_filt)
        port1.autofit()
        port1.plotall(fontsize=11)
        vals = port1.fitresults
        
        is_fit_failed = False
        for key in vals:
            if math.isnan(vals[key]):
                is_fit_failed = True
        if is_fit_failed:
            ax = plt.gcf()
            plt.clf()
            continue
            
        ax = plt.gcf()
        ax.tight_layout()
        
        
        textstr = '\n'.join((
           r'$fr=%.5f\pm%.5f[GHz]$' % (vals['fr']/1e9, vals['fr_err']/1e9, ), # was 7 sig figs
           r'$Qi=%.1f\pm%.1f$' % (vals['Qi_dia_corr'], vals['Qi_dia_corr_err'], ),
           r'$Ql=%.1f\pm%.1f$' % (vals['Ql'], vals['Ql_err'], ),
           r'$Qc=%.1f\pm%.1f$' % (vals['Qc_dia_corr'], vals['absQc_err'], ),
           r'$\phi_0=%.2f\pm%.2f[deg]$' % (vals['phi0']*180/np.pi, vals['phi0_err']*180/np.pi, ),
           r'$\theta_0=%.2f[deg]$' % (vals['theta0']*180/np.pi, ),
           r'$\chi^2=%.3f$' % (vals['chi_square'], )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.59, 0.47, textstr, fontsize=12,
                verticalalignment='top', bbox=props)
        plt.show()

        plt.savefig(root+'\\'+fname+'.png')
        plt.clf()
        
       # print('\n')
        #print(name)
        print('kappa_i/2*pi =',np.round(vals['fr']/vals['Qi_dia_corr']*1e-3,2),'kHz')        
        print('kappa_e/2*pi =',np.round(vals['fr']/vals['Qc_dia_corr']*1e-3,2),'kHz')       
        
        name = "Op" + str(counter)
        
       # if data_all[name]['ro_power']==[]: data_all[name]['ro_power'] = ro_pwr
        
        # data_all[name]['temperature'] = np.append(data_all[name]['temperature'],temp)
        
        data_all[name]['freq'] = np.append(data_all[name]['freq'],vals['fr'])
        data_all[name]['freq_err'] = np.append(data_all[name]['freq_err'],vals['fr_err'])
        
        kappa_i = 2*np.pi*vals['fr']/vals['Qi_dia_corr']
        kappa_c = 2*np.pi*vals['fr']/vals['Qc_dia_corr']
        kappa_l = 2*np.pi*vals['fr']/vals['Ql']
        kappa_i_err = kappa_i * np.sqrt((vals['fr_err']/vals['fr'])**2 + (vals['Qi_dia_corr_err']/vals['Qi_dia_corr'])**2)
        kappa_c_err = kappa_c * np.sqrt((vals['fr_err']/vals['fr'])**2 + (vals['absQc_err']/vals['Qc_dia_corr'])**2)
        kappa_l_err = kappa_l * np.sqrt((vals['fr_err']/vals['fr'])**2 + (vals['Ql_err']/vals['Ql'])**2)
        
        data_all[name]['kappa_i'] = np.append(data_all[name]['kappa_i'],kappa_i)
        data_all[name]['kappa_c'] = np.append(data_all[name]['kappa_c'],kappa_c)
        data_all[name]['kappa_l'] = np.append(data_all[name]['kappa_l'],kappa_l)
        data_all[name]['kappa_i_err'] = np.append(data_all[name]['kappa_i_err'],kappa_i_err)
        data_all[name]['kappa_c_err'] = np.append(data_all[name]['kappa_c_err'],kappa_c_err)
        data_all[name]['kappa_l_err'] = np.append(data_all[name]['kappa_l_err'],kappa_l_err)
        
        data_all[name]['theta0'] = np.append(data_all[name]['theta0'],vals['theta0']*180/np.pi)
        data_all[name]['phi0'] = np.append(data_all[name]['phi0'],vals['phi0']*180/np.pi)
        data_all[name]['phi0_err'] = np.append(data_all[name]['phi0_err'],vals['phi0_err']*180/np.pi)
        data_all[name]['chi_square'] = np.append(data_all[name]['chi_square'],vals['chi_square'])
        
        counter = counter + 1

freqdo.save_folder = saveFolder
freqdo.pickle_save(data_all,'data_all ')


#%% qubit I
freqs = [6.289894 * GHz]
suspected_qubit_freqs = [...]
freqdo.two_tone_sweep(
                        center = 3 * GHz,
                        span = 4* GHz,
                        npts = 4001,
                        mxg_power = 20 * dBm,
                        pna_freq = 6.290 * GHz,
                        pna_power = -20 * dBm,
                        navg =1 ,
                        if_bw = 100 * Hz,
                        which_data = 'phase',
                        is_save_data=True,
                        is_fit = False,
                        n_cut = 1,
                        chi_guess = 3.0e-3,
                        alpha_guess = 0.4,
                    )

