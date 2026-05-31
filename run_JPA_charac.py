# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 11:17:54 2021
"""

import numpy as np
import matplotlib.pyplot as plt
from labrad.units import V,mV, s,ms,us,ns, Hz,kHz,MHz,GHz, Ohm, dBm, Value, mA
import sys
from sequence_runner_base import Sequence_runner
from  qnl_ctrl.InstrumentControl import MXG5183A, AgilentPNA, AgilentVNA, B2962A #AWG5200# import GPIB code for arb (not based on Labrad, except for unit system)
from percure_data import Network_analyzer
# from Current_sequences import RunCurrentFreqSweep
from power_sequence import Power_sequence
from Current_sequence import Current_sequence
import datetime

#%% initialize instruments

pna = AgilentPNA('TCPIP0::K-N5232B-81186::inst0::INSTR')

#%% initialize classes

saveFolderName = r'F:\Dropbox (Technion Dropbox)\Gal\JPA'    # old pc dropbox
saveFolderName = r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal\JPA'    # new pc one drive

psq = Power_sequence(saveFolderName = saveFolderName)
csq = Current_sequence(saveFolderName = saveFolderName)

#%%

# ps = B2962A.B2962A()
#%% power and frequency sweep ("sexy plot")

pna_power_pts = 1001
pna_freq_pts = 1001

s_parameters = 'S21'

lowPower = -50 * dBm
highPower = -10 * dBm

lowFreq = 4 * GHz
highFreq = 4.5 * GHz

IF_BW = 100 * Hz



saveFileName = 'freqPowerSweep'



lowToHighRes = psq.Sexy_plot(pna = pna,
                            pna_power = [lowPower, highPower],
                            pna_freq = [lowFreq, highFreq],
                            pna_freq_pts = pna_freq_pts,
                            pna_power_pts= pna_power_pts,
                            s_parameters = s_parameters,
                            IF_BW = IF_BW,
                            saveFileName = saveFileName)

highToLowRes = psq.Sexy_plot(pna = pna,
                              pna_power = [highPower, lowPower],
                              pna_freq = [lowFreq, highFreq],
                              pna_freq_pts = pna_freq_pts,
                              pna_power_pts= pna_power_pts,
                              s_parameters = s_parameters,
                              IF_BW = IF_BW,
                              saveFileName = saveFileName)

# twoSidesData = lowToHighRes['data']


#psq.plot_2D_sweep(X_axis = lowToHighRes['freq'], Y_axis = lowToHighRes['power'], DATA = lowToHighRes['data'], Xlabel = 'Frequency', Ylabel = 'Power')

#%% save data
if saveFolderName is not None:
    if saveFileName is None: saveFileName = ''
    now = datetime.datetime.now()
    completeFilePath = saveFolderName+r'\\'+saveFileName+'_LtoH'+now.strftime("_%Y_%m_%d_%H-%M-%S")+'.npz'
    np.savez(completeFilePath, DATA= lowToHighRes)
    print(f'Save data to [{completeFilePath}]')
    now = datetime.datetime.now()
    completeFilePath = saveFolderName+r'\\'+saveFileName+'_HtoL'+now.strftime("_%Y_%m_%d_%H-%M-%S")+'.npz'
    np.savez(completeFilePath, DATA= highToLowRes)
    print(f'Save data to [{completeFilePath}]')
    
#%% combined plot
sexyPlot = np.zeros((pna_power_pts,pna_freq_pts*2))
pnaFreq = np.zeros(pna_freq_pts*2)
lth = lowToHighRes['data']
htl = np.flip(highToLowRes['data'],0)
fr = lowToHighRes['freq']
for i in range(pna_freq_pts):
    lth = np.insert(lth,i,htl[:,i],axis=1)
    fr = np.insert(fr,i,fr[i])
    # sexyPlot[:,i] = lth[:,i]
    # pnaFreq[i] = fr[i]
    i+=i
    # sexyPlot[:,i] = htl[:,i-1]
    # pnaFreq[i] = fr[i-1]
# psq.plot_2D_sweep(X_axis = fr, Y_axis = lowToHighRes['power'], DATA = lth, Xlabel = 'Frequency', Ylabel = 'Power')
X_axis = lowToHighRes['freq']; Y_axis = lowToHighRes['power']; DATA = lowToHighRes['data']; Xlabel = 'Frequency'; Ylabel = 'Power'
color_data = np.angle(DATA)
plt.figure()
plt.imshow(color_data, extent = (X_axis[X_axis.units].min(), X_axis[X_axis.units].max(), Y_axis[Y_axis.units].min(), Y_axis[Y_axis.units].max()), cmap='seismic', aspect = 'auto')#,edgecolors='k', linewidths=0.02)
# twoSidesData = lowToHighRes['data']-np.flip(highToLowRes['data'],0)
# psq.plot_2D_sweep(X_axis = lowToHighRes['freq'], Y_axis = lowToHighRes['power'], DATA = twoSidesData, Xlabel = 'Frequency', Ylabel = 'Power', Unwrap = True)

#%% flux tuning curve
csq.RunCurrentFreqSweep(current_range = [-2*mA,2*mA],# a lab_rad list of [start current, stop current]
                            current_pts = 1001,# number of steps in the current sweep
                            magnetization_delay=2*s, # The delay time between setting the current and performing the frequency sweep
                            pna_power = -23*dBm, # pna power
                            pna_freq=[5*GHz,4*GHz],# a lab_rad list of [frequncey center, span]
                            frq_pts = 1001, #number of MXG sweep_points
                            IF_BW = 100*Hz,
                            averging_folder_name = r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal\JPA',
                            averging_file_name = 'fluxTuningCurve'
                            )
#%%
fluxCurve = np.load(saveFolderName+r'\fluxTuningCurvecurrent_scan_2021_08_10_05-01-31.npz', allow_pickle = True)
data = fluxCurve['DATA'].item()
X_axis = data['X_axis']; Y_axis = data['Y_axis'][0:450]; DATA = np.flip(np.transpose(data['Raw'][:,0:450])); Xlabel = 'Current'; Ylabel = 'Frequency'
color_data = np.angle(DATA)
plt.figure()
plt.imshow(color_data, extent = (X_axis[X_axis.units].min(), X_axis[X_axis.units].max(), Y_axis[Y_axis.units].min(), Y_axis[Y_axis.units].max()), cmap='seismic', aspect = 'auto')
plt.title('Flux Tuning Curve',fontweight="bold")
if Ylabel is not None:
    Ylabel += '[{0}]'.format(Y_axis.units)
    plt.ylabel(Ylabel, fontsize = 15)
if Xlabel is not None: 
    Xlabel += ' [{0}]'.format(X_axis.units)
    plt.xlabel(Xlabel, fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
cb = plt.colorbar()
cb.set_label(label='Phase', fontsize = 15)
#%%
HtoL = np.load(saveFolderName+r'\freqPowerSweep_HtoL_2021_08_09_11-09-19.npz', allow_pickle = True)
LtoH = np.load(saveFolderName+r'\freqPowerSweep_LtoH_2021_08_09_11-09-19.npz', allow_pickle = True)
dataHtoL = HtoL['DATA'].item()
dataLtoH = LtoH['DATA'].item()
Raw = np.zeros(dataHtoL['data'].shape)
for i in range(len(Raw)):
    if i%2==0:
        Raw[:,i] =  dataHtoL['data'][:,i]
    if i%2==1:
        Raw[:,i] = dataLtoH['data'][:,i]
X_axis = dataHtoL['freq']; Y_axis = dataHtoL['power']; DATA = np.flip(np.transpose(Raw)); Xlabel = 'Frequency'; Ylabel = 'Power'
color_data = np.angle(DATA)
plt.figure()
plt.imshow(color_data, extent = (X_axis[X_axis.units].min(), X_axis[X_axis.units].max(), Y_axis[Y_axis.units].min(), Y_axis[Y_axis.units].max()), cmap='seismic', aspect = 'auto')
plt.title('Sexy Plot',fontweight="bold")
if Ylabel is not None:
    Ylabel += '[{0}]'.format(Y_axis.units)
    plt.ylabel(Ylabel, fontsize = 15)
if Xlabel is not None: 
    Xlabel += ' [{0}]'.format(X_axis.units)
    plt.xlabel(Xlabel, fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
cb = plt.colorbar()
cb.set_label(label='Phase', fontsize = 15)