# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 11:17:54 2021
"""

import sys
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\measurementClass')
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\freqDomain')
import numpy as np
import matplotlib.pyplot as plt
from labrad.units import V,mV, s,ms,us,ns, Hz,kHz,MHz,GHz, Ohm, dBm, Value, mA
import sys
from sequence_runner_base import Sequence_runner
from  qnl_ctrl.InstrumentControl import AWG5200, MXG5183A, AgilentPNA, AgilentVNA, B2962A # import GPIB code for arb (not based on Labrad, except for unit system)
from percure_data import Network_analyzer
# from Current_sequences import RunCurrentFreqSweep
from power_sequence import Power_sequence
from Current_sequence import Current_sequence
import datetime

#%% initialize instruments

pna = AgilentPNA('TCPIP0::K-N5232B-81186::inst0::INSTR')

#%% initialize classes

# saveFolderName = r'F:\Dropbox (Technion Dropbox)\Gal\JPA'    # old pc dropbox
# saveFolderName = r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal\JPA'    # new pc one drive
saveFolderName = r'C:\Users\Shay\OneDrive - Technion\Guy\Zeno_qubits_spectroscopy'

# psq = Power_sequence(saveFolderName = saveFolderName)
# csq = Current_sequence(saveFolderName = saveFolderName)


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
X_axis = fr; Y_axis = lowToHighRes['power']; DATA = lth; Xlabel = 'Frequency'; Ylabel = 'Power'
color_data = np.angle(DATA)
plt.figure()
plt.imshow(color_data, extent = (X_axis.min(), X_axis.max(), Y_axis[Y_axis.units].min(), Y_axis[Y_axis.units].max()), cmap='seismic', aspect = 'auto')#,edgecolors='k', linewidths=0.02)
# twoSidesData = lowToHighRes['data']-np.flip(highToLowRes['data'],0)
# psq.plot_2D_sweep(X_axis = lowToHighRes['freq'], Y_axis = lowToHighRes['power'], DATA = twoSidesData, Xlabel = 'Frequency', Ylabel = 'Power', Unwrap = True)
