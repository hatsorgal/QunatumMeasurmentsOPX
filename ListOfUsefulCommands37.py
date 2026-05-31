# -*- coding: utf-8 -*-

import sys
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\measurementClass')
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\freqDomain')
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\measurementClass')
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\freqDomain')
from labrad.units import V, mV, s, ms, us, ns, Hz, kHz, MHz, GHz, Ohm, dBm, Value
import datetime
from percure_data import *
from MXG_SWEEP import *
import numpy as np
#%%

PNA = None
MXGTop = None
MXGMiddle = None
MXGBottom = None


# Open a folder with the year and month in the name
now = datetime.datetime.now().strftime("_%Y_%m")
fig_num = 1

#%% Initialize code for spectroscopy

PNA, MXPTop, MXGMiddle, MXGBottom = initialize_system_for_spectroscopy(PNA) 

#%% Q Fit
CavityName='Flute4'
FolderName = r'F:\OneDrive - Technion\Eliya\Experiment Data\Flute4'
CavityFolder = FolderName
# Resonator_fit_conf(averging_folder_name=CavityFolder,averging_file_name = CavityName,PNA = PNA)
_=Characterize_Mode(averging_folder_name=FolderName, averging_file_name = None, PNA = PNA, triger_data =False, Punchout = False)
#%% Q Fit
CavityName='Berkeley'
FolderName = r'F:\OneDrive - Technion\Eliya\Experiment Data\Berkeleye'
CavityFolder = FolderName
# Resonator_fit_conf(averging_folder_name=CavityFolder,averging_file_name = CavityName,PNA = PNA)
_=Characterize_Mode(averging_folder_name=FolderName, averging_file_name = None, PNA = PNA, triger_data =False, Punchout = False)
#%% Q Fit
CavityName='4HB'
FolderName = r'C:\Users\Shay\OneDrive - Technion\Eliya\Experiment Data\Entanglement Stabilization\Spectroscopy\One Tone'
CavityFolder = FolderName
# Resonator_fit_conf(averging_folder_name=CavityFolder,averging_file_name = CavityName,PNA = PNA)
_=Characterize_Mode(averging_folder_name=FolderName, averging_file_name = None, PNA = PNA, triger_data =False, Punchout = False)
#%% Q FiCavityName='EntStab'
FolderName = r'C:\Users\Shay\OneDrive - Technion\Eliya\Experiment Data\Entanglement Stabilization\Spectroscopy\One Tone'
CavityFolder = FolderName
# Resonator_fit_conf(averging_folder_name=CavityFolder,averging_file_name = CavityName,PNA = PNA)
_=Characterize_Mode(averging_folder_name=FolderName, averging_file_name = None, PNA = PNA, triger_data =False, Punchout = False)


#%% Q Fit
CavityName='LaserTransmon'
FolderName = r'C:\Users\Shay\OneDrive - Technion\Natan\Experiment Data\LaserTransmon\Spectroscopy\One Tone'
CavityFolder = FolderName
# Resonator_fit_conf(averging_folder_name=CavityFolder,averging_file_name = CavityName,PNA = PNA)
_=Characterize_Mode(averging_folder_name=FolderName, averging_file_name = None, PNA = PNA, triger_data =False, Punchout = False)
#%% Q Fit Gal
CavityName='GalTransmon'
FolderName = r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments'
CavityFolder = FolderName
# Resonator_fit_conf(averging_folder_name=CavityFolder,averging_file_name = CavityName,PNA = PNA)
_=Characterize_Mode(averging_folder_name=FolderName, averging_file_name = None, PNA = PNA, triger_data =False, Punchout = False)
#%%
# fit resonance from file
fit_filepath = r'C:\Users\PHshayg-lab3\OneDrive - Technion\Eliya\Experiment Data\Entanglement Stabilization\Spectroscopy\One Tone\ResonatorFit_2022_08_21_10-39-40.npz'
loaded_data = np.load(fit_filepath, allow_pickle = True)['DATA']
X = loaded_data.item()['X_axis']
y = loaded_data.item()['Y_axis']
Wr =  loaded_data.item()['Wr']
Data_dic, ax =Mode_fitNplot(X_axis=X,Y_axis=y,freq_cen =Wr)
#%% 2 tone Spectroscopy
FolderName = r'C:\Users\Shay\OneDrive - Technion\Eliya\Experiment Data\Entanglement Stabilization\Spectroscopy\Two Tone'
FileName = '2tonespec'
fig_num=1
# This is power below P.O
# This is Wc or close to it for spectroscopy
pna_freq_instance =  7.1916*GHz
pna_pow_instance = -40*dBm
# This should take into consideration SNR and power broadening
sg_power_instance = 20*dBm



XCW,YCW = CW_spectroscopy(PNA , MXGTop,
                          pna_freq = pna_freq_instance,# labrad value
                          sg_power = sg_power_instance,# a lab_rad list of [Power_sweep_start, power_sweep_stop] or  a labrad value of SG power depending if we sweep for power or frequencey
                          sg_freq = [3.2*GHz, 0.8*GHz], # a lab_rad list of [sg_frequncey_center (guess of qubit frequence), sg_frequncey_span] or  a labrad value of SG single fequencey depending if we sweep for power or frequencey
                          frq_pts= 201, #number of MXG sweep_points
                          swp_typ = 'SING',#determines if to initate a single ('SING') or continues ('CONT') sweep of the MXG
                          pna_numAv = 1, #number of times the each point of the data is averged over
                          IF_BW = 50*Hz,pna_power = pna_pow_instance, save_data= True,in_flag=True,
                          folder_name = FolderName,
                          file_name = FileName, fig_num = fig_num  )
