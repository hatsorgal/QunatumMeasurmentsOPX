# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:21:46 2021

@author: Shay
"""
#%% Find cavity and qubit frequencies

import datetime
from labrad.units import V,mV, s,ms,us,ns, Hz,kHz,MHz,GHz, Ohm, dBm, Value
import sys
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\freqDomain')

PNA = None
MXGTop = None
MXGBottom = None

SampleName = 'Ta7h'
UserName = 'SRJ'

# Open a folder with the year and month in the name
now = datetime.datetime.now() 
FolderName = UserName+'\\'+SampleName+now.strftime("_%Y_%m")


BigWorkingDic = {}

#%% Initialize code for spectroscopy
from percure_data import *
from MXG_SWEEP import *
PNA, MXGTop,MXGBottom = initialize_system_for_spectroscopy(PNA, MXGTop,MXGBottom) ##TODO fix visa protocol
#%% PARAMP!!!


"""
# Save for Paramp background resonances and stuff
UserName = 'Asaf'
ParampSampleName = 'Paramp_epsilon'
ParampFolderName =  UserName+'\\'+ParampSampleName+now.strftime("_%Y_%m")
Quick_save(averging_folder_name = ParampFolderName ,averging_file_name = 'ParampEpsilon' ,NetAnal = PNA )
"""
        # Q Fit
CavityFolder =  FolderName + '\\cavity'
CavityName = 'Ta'
#Resonator_fit_conf(averging_folder_name=CavityFolder,averging_file_name = CavityName,PNA = PNA)

#_=Characterize_Mode(averging_folder_name=FolderName,averging_file_name = None,PNA = PNA, triger_data =False)

#runfile('F:/Dropbox (Technion Dropbox)/programing/Calibration/sequence_runner_base.py', wdir='F:/Dropbox (Technion Dropbox)/programing/Calibration')

#Quick_save(averging_folder_name = FolderName ,averging_file_name = 'Result' ,NetAnal = PNA)


#%% Data Run & Save

#PNA.electrical_delay() = 6.569e-08 [s]
#PNA.s_parameters(in_port= 1, out_port =2 )
#PNA.powerOnOff('ON')
#Elec delay - 65.7ns (19.7m)
#Number of point - 201
#BW- 10 Hz


#Freqs=[4.083,4.2336,4.7591,4.85082,5.16042,5.27354,5.5367,5.6403] #GHz
Freqs=[5.27354,5.5367,5.6403] #GHz
Spans=[200,1000,500] #KHz
Powers=[10,0,-10,-20,-30,-40,-50,-60,-70]#
Avges=[1,1,1,1,1,2,2,2,2,2,2,2,5,7,10,12,15,17]
PNA.s_parameters(in_port=2,out_port=1)
#PNA.bandwidth(10)
PNA.bandwidth(1)
PNA.num_points(601)
PNA.powerOnOff(1)
for (freq,span) in zip(Freqs,Spans):
    PNA.frequencey_span(freq*1e9,span*1e3)
    for (power,ave) in zip(Powers,Avges):       
        PNA.reset_measure()
        PNA.power(p=power)
        PNA.averages(av=ave)
        PNA.set_format_data(frm='REAL')
        sleep(ave*PNA.get_sweep_time()['s'])
        PNA.auto_scale()
        sleep(1)
        Real = PNA.getData()
        PNA.set_format_data(frm='IMAG')
        PNA.auto_scale()
        sleep(1)
        Imag = PNA.getData()
        x = PNA.get_X_axis()
        
        plt.figure()
        plt.plot(x['Hz']/1e9,Real,x['Hz']/1e9,Imag)
        plt.title('Power = {}dBm'.format(power))
        plt.pause(0.05)
         
        FileName = 'TaResonator_{}GHz_{}dBm'.format(freq,power)
        #Ufun.create_folder(averging_folder_name)
        path = FileName+'.npz' #FolderName+'\\'+FileName+'.npz'
        
        np.savez(path,x['Hz'],Real,Imag)
        

#%% Quick Data Save


power=-60

PNA.set_format_data(frm='REAL')
PNA.auto_scale()
sleep(1)
Real = PNA.getData()
PNA.set_format_data(frm='IMAG')
PNA.auto_scale()
sleep(1)
Imag = PNA.getData()
x = PNA.get_X_axis()

plt.figure()
plt.plot(x['Hz']/1e9,Real,x['Hz']/1e9,Imag)
plt.title('Power = {}dBm'.format(power))

FileName = 'TaResonator_5.77GHz_{}dBm'.format(power)
#Ufun.create_folder(averging_folder_name)
path = FileName+'.npz' #FolderName+'\\'+FileName+'_2.npz'

np.savez(path,x['Hz'],Real,Imag)

#%% Set Freq

#PNA.s_parameters(in_port= 1, out_port =2 )
#PNA.powerOnOff('ON')

PNA.frequencey_span(CenFreq = 5.2911425e9, span = 0.6e6) #5.7625e9,5.524e9,5.2911425e9 ; 0.5e6
PNA.power(p=10)
PNA.averages(av=1)
PNA.set_format_data(frm='IMAG') #MLIN
PNA.auto_scale()
PNA.set_marker_frequencey(freq=5.524e9)

#PNA.bandwidth()
#PNA.num_points()
#PNA.phase_offset(-17)
