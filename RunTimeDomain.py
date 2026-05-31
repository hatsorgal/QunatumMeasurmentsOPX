# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 11:54:19 2026

@author: Gal
"""

import sys
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\measurementClass')
sys.path.append(r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\scripts')
from TimeDomainConfigOctave import get_my_config, create_my_config
from Time_characterzation import TiDo_Chara

from MXG5183A import MXG5183A
from AgilentPNA import AgilentPNA
from B2962A import B2962A
from CXA_SA import CXA_SA

from time import sleep

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from data_processing import round_value_by_error, data_to_sigma_z


from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig

#%% Load

try:
    for sg in tido.sg_dict.values():
        if sg['type'] == 'wf':
            sg['inst'].close()
            print(f'closed windfreak')
except:
    pass

try:
    tido.rm.close()
except:
    pass

tido = TiDo_Chara(
                Config = create_my_config(),
                fridge = 2,
                main_qubit = 'qb1',
                main_readout = 'ro',
                which_data = 'I',
                wait_between_seq = 50e3,
                get_external = True,
                is_pi_pulse = True,
                is_calc_stat_error = True,
                simulation_channels = [1,3],
                text_size = 10,
                is_save_data = True,
                save_folder = r'C:\Users\PHshayg-lab3\OneDrive - Technion\Omer\Gal Qubit Measurment\TiDo\28_5_26',
                is_get_ps = False
                )

#%% octave calib

tido.octave_calib('ro')
tido.octave_calib('qb1')

#%%
tido.update_wait_between_seq=1.6e5
#%% down conversion
tido.run_down_conversion_mixer_calibration(N_avg = 10000, wait_time = 10000)

#%% Mixer Calibration qb1
qubit = 'qb1'
# tido.element_IF(qubit, -250e6, is_change_element_freq = False)
tido.pulse_amp('qb1', 'mixer_cal_pulse', 0.25, update = True)
tido.remove_leakage('qb1', I_os = 0.01, Q_os = 0.01, LO_acc = -85, is_run = True, 
                    res_bw = 10, vid_bw = 10, freq_span_ratio = 0.01)
tido.remove_side_band('qb1', SB_acc = -65, g_os = 0.05, phi_os = 20, freq_span_ratio = 0.5, res_bw = 1000, vid_bw = 1000)
tido.remove_side_band('qb1', SB_acc = -85, g_os = 0.02, phi_os = 1, freq_span_ratio = 0.01, res_bw = 10, vid_bw = 10)
tido.set_cxa_to_element(qubit)
tido.run_continuous(qubit)
tido.CXA.Amp(5)


#%% Mixer Calibration readout
tido.pulse_amp('ro', 'mixer_cal_pulse', 0.095, update = True)
tido.remove_leakage('ro', I_os = 0.01, Q_os = 0.01, LO_acc = -100, is_run = True,
                    res_bw = 10, vid_bw = 10, freq_span_ratio = 0.001)
# tido.update_mixer_by_element('ro', 0,0)
tido.CXA.Amp(-0)
tido.remove_side_band('ro', SB_acc = -80, g_os = 0.05, phi_os = 5, freq_span_ratio = 0.01, res_bw = 100, vid_bw = 100)
tido.CXA.Amp(-0)
tido.set_cxa_to_element(tido.main_readout, res_bw = 100, vid_bw = 100, center = 'LO')
tido.run_continuous(tido.main_readout)


#%% pinopi

# tido.element_freq('ro', 7.32396e9 - 0.2e6)
# runcell('ro params', r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\scripts\RunTimeDomain.py')
# runcell('qubit params', r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\scripts\RunTimeDomain.py')


tido.load_pinopi(npts = 25, N_avg = 10000, meas_type = 'sliced', is_active_reset = False,
                 is_continuous_drive = False,
                 is_sideband_cool = False,
                 # ro_pulse = 'ro_pulse',
                 wait_time = 0)
# tido.simulate_prog()
tido.run_pinopi(is_plot_fid = False, is_update_readout_analyzer = False)
#%% Set active reset threshold
tido.load_pinopi(N_avg = 10000, is_active_reset = False, meas_type = 'full')
tido.run_pinopi(is_update_readout_analyzer=True, is_kmeans = True, is_fit_circle=False)
tido.is_active_reset = True
tido.is_loop_reset = True
phase_err = tido.results['PiNoPi']['largest_distance_phase']
tido.which_data = phase_err * np.pi / 180
tido.which_data = 'I'
tido.plot_pinopi(is_update_readout_analyzer=True, is_kmeans = True, is_fit_circle=False)

#%% Test active reset
tido.load_pinopi(N_avg = 100000, is_active_reset = True, meas_type = 'full')
tido.run_pinopi(is_update_readout_analyzer=False, is_kmeans = False, is_fit_circle=False)
#%% auto pinopi
tido.which_data = 0
tido.IW_discrim_I(1, ro_element = 'ro', ro_pulse = tido.ro_pulse, update = False)
tido.IW_discrim_Q(1, ro_element = 'ro', ro_pulse = tido.ro_pulse, update = True)
phase_err = np.inf
while np.abs(phase_err) > 1:
    tido.load_pinopi(npts = 25, N_avg = 10000, meas_type = 'sliced', is_active_reset = False)
    tido.run_pinopi(is_update_readout_analyzer = False, is_plot_fid = True, is_save_discrimination = True)
    phase_err = tido.results['PiNoPi']['largest_distance_phase']
    tido.set_IQ_rot_phase(tido.set_IQ_rot_phase(ro_pulse = tido.ro_pulse)+phase_err, ro_pulse = tido.ro_pulse)
    print(f'Corrected phase to {tido.set_IQ_rot_phase(ro_pulse = tido.ro_pulse)}')
    plt.pause(0.1)
tido.IW_discrim_I(tido.discrim_I,ro_element = 'ro', ro_pulse = tido.ro_pulse, update = False)
tido.IW_discrim_Q(tido.discrim_Q,ro_element = 'ro', ro_pulse = tido.ro_pulse)
print('\n\nFound discrim parameters\n\n')
phase_err = np.inf
while np.abs(phase_err) > 1:
    tido.load_pinopi(npts = 25, N_avg = 10000, meas_type = 'sliced', is_active_reset = False)
    tido.run_pinopi(is_update_readout_analyzer = False, is_plot_fid = True, is_save_discrimination = True)
    phase_err = tido.results['PiNoPi']['largest_distance_phase']-tido.which_data
    tido.which_data += phase_err
    # tido.IW_discrim_phase(tido.IW_discrim_phase(ro_pulse = tido.ro_pulse)+phase_err, ro_pulse = tido.ro_pulse)
    # print(f'Corrected discrim phase to {tido.IW_discrim_phase(ro_pulse = tido.ro_pulse)}')
    plt.pause(0.1)
runcell('Set active reset threshold', r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\scripts\RunTimeDomain.py')

#%% ckp
tido.pulse_amp('ro', 'mixer_cal_pulse', 0.04)
# tido.pulse_amp('ro', 'ro_pulse', 0.04)
tido.load_ckp(N_avg = 200,
              ro_detuning_start = -3e6, ro_detuning_stop = 3e6, ro_detuning_npts = 61,
              qb_detuning_start = -30e6, qb_detuning_stop = 10e6, qb_detuning_npts = 61,
    is_ramp = True, ringdown_time = 800, is_active_reset = None,
    drive_pulse = 'mixer_cal_pulse')
# tido.simulate_prog(is_IF = False, element_list = ['qb1', 'ro'], channels = [1,3])
tido.run_ckp()


 #%% Pinopi spec
# runcell('ro params', 'F:/OneDrive - Technion/Natan/Experiment Scripts/Fock State Transfer/MasterScript.py')
# pump_mxg.off()
# ps.set_curr_smooth(0e-3)
tido.load_pinopi_spec(N_avg = 5000, npts = 100, start = -5e6, stop = 5e6, qubit = 'qb1', 
                      pi_pulse = 'pi_pulse',
                      is_active_reset = False)
tido.run_pinopi_spec(is_fit_mag = True)
#%% number split spec

# tido.pulse_sig('qb1', 'pi_pulse', 1000, update = False)
# tido.pulse_amp('qb1', 'pi_pulse', 0.0464*np.sqrt(1000/64), update = False)
tido.pulse_amp('ro', 'mixer_cal_pulse', 0.000)

tido.load_number_split_spec(N_avg = 50000, npts =  21, start = -50e6, stop= 50e6)
tido.run_number_split_spec()
#%% pi2 calib

# runcell('ro params', r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\scripts\RunTimeDomain.py')
runcell('qubit params', r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\scripts\RunTimeDomain.py')

# tido.pulse_amp(tido.main_qubit, 'pi2_pulse', 0.225/2, update = False)
# tido.pulse_len(tido.main_qubit, 'pi2_pulse', 64, update = True)
tido.load_pi2_calibration(npts = 12, start_at = 0, N_avg = 5000)
tido.run_pi2_calibration()

#%% auto pi2 initial

from scipy.optimize import curve_fit
def cosPi2(x, A, delta, offset):
    return A * np.cos((1+delta)*np.pi/2*x) + offset

def auto_pi2():
    diff_thresh = 0.001
    diff = np.inf
    diff_err = 0
    is_plot_auto_pi2 = True
    tido.load_pi2_calibration(npts = 12, start = 0, N_avg = 5000,  is_active_reset = None)
    while np.abs(diff) > diff_thresh:
        tido.run_pi2_calibration(is_save_data = False, is_plot = is_plot_auto_pi2)
        if is_plot_auto_pi2: plt.pause(0.1)
        results_dict = tido.results['pi2_calibration']
        start = results_dict['start']
        npts = results_dict['npts']
        data, err = tido.process_data(results_dict)
        data, _, factor = tido.autoscale_data(data)
        err = err * factor
        offset = (data.max()+data.min())/2
        guess_sign = data[0]>offset
        A_guess = (data.max()-data.min())/2 if guess_sign else -(data.max()-data.min())/2
        fit,cov = curve_fit(cosPi2, np.arange(start, start+npts), data, p0 = [A_guess, 0, (data.max()+data.min())/2],
                            sigma = err)
        x_fit = np.linspace(start, start+npts, 101)
        plt.plot(x_fit, fit[2]+fit[0]*np.cos(np.pi/2*x_fit+np.pi/2*fit[1]*x_fit))
        plt.pause(0.1)
        diff=fit[1]
        diff_err = np.sqrt(cov[1,1])
        print(f'\nError = {diff}+-{diff_err}%\n')
        tido.pulse_amp('qb1', 'pi2_pulse', tido.pulse_amp('qb1', 'pi2_pulse') / (1+diff), keep_phase = True)
        runcell('octave calib', r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\scripts\RunTimeDomain.py') # added by gal        
        print(f'Corrected to {tido.pulse_amp("qb1", "pi2_pulse")}')
    found_amp,_ = round_value_by_error(tido.pulse_amp("qb1", "pi2_pulse"), tido.pulse_amp("qb1", "pi2_pulse")*diff)
    print(f'\npi2 amp = {found_amp}\n')
auto_pi2()

#%% auto pi2
def auto_pi2():
    diff_thresh = 0.001
    diff = np.inf
    diff_err = 0
    is_plot_auto_pi2 = True
    tido.load_pi2_calibration(npts = 12, start = 20, N_avg = 8000, is_meas_g_e = True, is_active_reset = None)
    while np.abs(diff) > diff_thresh:
        tido.run_pi2_calibration(is_save_data = False, is_plot = is_plot_auto_pi2)
        if is_plot_auto_pi2: plt.pause(0.1)
        results_dict = tido.results['pi2_calibration']
        data = tido.determine_data(results_dict['I'], results_dict['Q']).mean(0)
        data_g = data[0::4].mean()
        data_g_err = data[0::4].std()/np.sqrt(len(data[0::4]))
        data_p = data[1::4].mean()
        data_p_err = data[1::4].std()/np.sqrt(len(data[1::4]))
        data_e = data[2::4].mean()
        data_e_err = data.std()/np.sqrt(len(data[2::4]))
        data_m = data[3::4].mean()
        data_m_err = data[3::4].std()/np.sqrt(len(data[3::4]))
        
        diff = (data_p-data_m)/(data_e-data_g)/(results_dict['start']+results_dict['npts']/2)
        diff = (data_p-data_m)/(data_e-data_g)/results_dict['start']
        diff_err = np.sqrt((data_p_err/(data_e-data_g))**2 \
            + (data_m_err/(data_e-data_g))**2\
                + (data_e_err*(data_p-data_m)/(data_e-data_g)**2)**2\
                    + (data_g_err*(data_p-data_m)/(data_e-data_g)**2)**2)/results_dict['start']
        diff, diff_err = round_value_by_error(diff,diff_err)
        print(f'\nError = {diff}+-{diff_err}%\n')
        tido.pulse_amp('qb1', 'pi2_pulse', tido.pulse_amp('qb1', 'pi2_pulse') * (1-diff/2))
        runcell('octave calib', r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\scripts\RunTimeDomain.py') # added by gal
        print(f'Corrected to {tido.pulse_amp("qb1", "pi2_pulse")}')
    found_amp,_ = round_value_by_error(tido.pulse_amp("qb1", "pi2_pulse"), tido.pulse_amp("qb1", "pi2_pulse")*diff)
    print(f'\npi2 amp = {found_amp}\n')
auto_pi2()

#%% pi calib
#tido.pulse_sig(tido.main_qubit, 'pi_pulse', 8, update = False)
# tido.pulse_time_multiplier(tido.main_qubit, 'pi_pulse', 4, update = False)
# tido.pulse_amp(tido.main_qubit, 'pi_pulse', 0.225, update = False)
# tido.pulse_len(tido.main_qubit, 'pi_pulse', 32, update = True)
tido.load_pi_calibration(npts = 6, start_at = 0, N_avg = 5000)
tido.run_pi_calibration()

#%% auto pi initial

from scipy.optimize import curve_fit
def cosPi(x, A, delta, offset):
    return A * np.cos((1+delta)*np.pi*np.floor(x) + np.pi*(x-np.floor(x))) + offset

def auto_pi():
    diff_thresh = 0.001
    diff = np.inf
    diff_err = 0
    is_plot_auto_pi = True
    tido.load_pi_calibration(npts = 4, start = 0, N_avg = 5000, is_active_reset = None)
    while np.abs(diff) > diff_thresh:
        tido.run_pi_calibration(is_save_data = False, is_plot = is_plot_auto_pi)
        if is_plot_auto_pi: plt.pause(0.1)
        results_dict = tido.results['pi_calibration']
        start = results_dict['start']
        npts = results_dict['npts']
        data, err = tido.process_data(results_dict)
        data, _, factor = tido.autoscale_data(data)
        err = err * factor
        offset = (data.max()+data.min())/2
        guess_sign = data[0]>offset
        A_guess = (data.max()-data.min())/2 if guess_sign else -(data.max()-data.min())/2
        guess = [A_guess, (np.diff(data[1::2])/A_guess).mean(), (data.max()+data.min())/2]
        fit,cov = curve_fit(cosPi, np.arange(start, start+npts, 0.5), data, p0 = guess,
                            sigma = err)
        x_fit = np.arange(start, start+npts, 0.001)
        plt.plot(x_fit, cosPi(x_fit, *fit))
        plt.pause(0.1)
        diff=fit[1]
        diff_err = np.sqrt(cov[1,1])
        print(f'\nError = {diff}+-{diff_err}%\n')
        # tido.pulse_amp('qb1_ss', 'pi2_pulse', tido.pulse_amp('qb1', 'pi2_pulse') * (1-diff/2), update = False)
        tido.pulse_amp('qb1', 'pi_pulse', tido.pulse_amp('qb1', 'pi_pulse') / (1+diff), keep_phase = True)
        runcell('octave calib', r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\scripts\RunTimeDomain.py') # added by gal
        print(f'Corrected to {tido.pulse_amp("qb1", "pi_pulse")}')
    found_amp,_ = round_value_by_error(tido.pulse_amp("qb1", "pi_pulse"), tido.pulse_amp("qb1", "pi_pulse")*diff)
    print(f'\npi2 amp = {found_amp}\n')
auto_pi()

#%% auto pi
def auto_pi():
    diff_thresh = 0.001
    diff = np.inf
    diff_err = 0
    tido.load_pi_calibration(npts = 6, start = 16, N_avg = 5000, is_meas_g_e = True, is_active_reset = None)
    is_plot_auto_pi = True
    while np.abs(diff) > diff_thresh:
        tido.run_pi_calibration(is_save_data = False, is_plot = is_plot_auto_pi)
        if is_plot_auto_pi: plt.pause(0.1)
        results_dict = tido.results['pi_calibration']
        data = tido.determine_data(results_dict['I'], results_dict['Q']).mean(0)
        data_g = data[0::4].mean()
        data_g_err = data[0::4].std()/np.sqrt(len(data[0::4]))
        data_p = data[1::4].mean()
        data_p_err = data[1::4].std()/np.sqrt(len(data[1::4]))
        data_e = data[2::4].mean()
        data_e_err = data[2::4].std()/np.sqrt(len(data[2::4]))
        data_m = data[3::4].mean()
        data_m_err = data[3::4].std()/np.sqrt(len(data[3::4]))
        
        diff = (data_p-data_m)/(data_e-data_g)/(results_dict['start']+results_dict['npts']/2)
        # diff = (data_p-data_m)/(data_e-data_g)/results_dict['start']
        diff_err = np.sqrt((data_p_err/(data_e-data_g))**2 \
            + (data_m_err/(data_e-data_g))**2\
                + (data_e_err*(data_p-data_m)/(data_e-data_g)**2)**2\
                    + (data_g_err*(data_p-data_m)/(data_e-data_g)**2)**2)/(results_dict['start']+results_dict['npts']/2)
        # diff_err = np.sqrt((data_p_err/(data_e-data_g))**2 \
        #     + (data_m_err/(data_e-data_g))**2\
        #         + (data_e_err*(data_p-data_m)/(data_e-data_g)**2)**2\
        #             + (data_g_err*(data_p-data_m)/(data_e-data_g)**2)**2)/results_dict['start']
        # tido.pulse_amp('qb1_ss', 'pi_pulse', tido.pulse_amp('qb1', 'pi_pulse') * (1-diff/4), update = False)
        tido.pulse_amp('qb1', 'pi_pulse', tido.pulse_amp('qb1', 'pi_pulse') * (1-diff/4))
        runcell('octave calib', r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\scripts\RunTimeDomain.py') # added by gal
        diff, diff_err = round_value_by_error(diff,diff_err)
        print(f'\nError = {diff}+-{diff_err}%\n')
        print(f'Corrected to {tido.pulse_amp("qb1", "pi_pulse")}')
    found_amp,_ = round_value_by_error(tido.pulse_amp("qb1", "pi_pulse"), tido.pulse_amp("qb1", "pi_pulse")*diff)
    print(f'\npi amp = {found_amp}\n')
auto_pi()

#%% auto ramsey
def auto_ramsey():
    detuning_auto_ramsey = -5e-3
    deltaf = 10e9
    max_seq_time = 2000
    npts_auto_ramsey = 100
    N_avg_auto_ramsey = 1000
    delta_f_threshold = 0.05e6
    is_check_negative = False
    is_plot_auto_ramsey = True
    while np.abs(deltaf)>delta_f_threshold:
        tido.load_ramsey(N_avg = N_avg_auto_ramsey,  npts = npts_auto_ramsey, max_seq_time = max_seq_time, min_seq_time = None, detuning = detuning_auto_ramsey,
                        ZZ_interaction = False,
                        is_echo = False, is_active_reset = None)
        res, err, ax = tido.run_ramsey(is_plot_fft = False, is_save_data = False, plot = is_plot_auto_ramsey)
        freq = res[1]
        deltaf = -(np.abs(detuning_auto_ramsey)*1e9-freq*1e6)
        print(f'\n deltaf = {np.round(deltaf*1e-6,4)} MHz\n')
        tido.element_freq('qb1', tido.element_freq('qb1')+deltaf)
        runcell('octave calib', r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\scripts\RunTimeDomain.py')

        # tido.element_freq('qb1_ss', tido.element_freq('qb1_ss')+deltaf)
        print(f'{(tido.element_freq("qb1"))/1e9}e9')
        
    if is_check_negative:
        tido.load_ramsey(N_avg = N_avg_auto_ramsey,  npts = npts_auto_ramsey, max_seq_time = max_seq_time, min_seq_time = None, detuning = -detuning_auto_ramsey,
                        ZZ_interaction = False,
                        is_echo = False, is_active_reset = None)
        res, err, ax = tido.run_ramsey(is_plot_fft = False, is_save_data = False)
        freq = res[1]
        deltaf_final = np.abs(detuning_auto_ramsey)*1e9-freq*1e6
        print(f"Delta F with opposite sign of detuning is {np.round(deltaf_final*1e-6,4)} MHz")
        if np.abs(deltaf_final) < np.abs(detuning_auto_ramsey): 
            raise ValueError("Check for negative or positive detuning error.")
    plt.pause(0.1)
auto_ramsey() 

#%% T1
# tido.pulse_amp(tido.main_qubit, 'pi_pulse', 0.225, update = False)
# tido.pulse_len(tido.main_qubit, 'pi_pulse', 64, update = True)
tido.load_T1(N_avg = 1000, npts = 1000, max_seq_time = 180e3)
tido.run_T1(is_save_data=True)

#%% Ramsey
tido.load_ramsey(N_avg = 1000,  npts = 1001, max_seq_time = 200016, min_seq_time = 16, detuning = 1e-3,
                is_echo = True, is_active_reset = False)
tido.run_ramsey(is_plot_fft = False, is_save_data = True)

#%% Rabi
tido.pulse_amp(tido.main_qubit, 'rabi_pulse', 0.4)

tido.load_rabi(N_avg = 5000, npts = 51, max_seq_time = 800)
result= tido.run_rabi(is_return_fit = False)


#%% qubit params
# tido.element_freq('qb1', 2.4680719203296597e9, update = False)
# tido.pulse_len('qb1', 'pi2_pulse', 128, update = False)
# tido.pulse_len('qb1', 'pi_pulse', 128, update = False)  

# tido.pulse_amp('qb1', 'pi2_pulse', 0.07161, update = False)
# tido.pulse_amp('qb1', 'pi_pulse',  0.1466, update = False)
# tido.update_config()
# runcell('octave calib', r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\scripts\RunTimeDomain.py')

tido.element_freq('qb1', 2.468e9, update = False)
tido.pulse_len('qb1', 'pi2_pulse', 64, update = False)
tido.pulse_len('qb1', 'pi_pulse', 64, update = False)  

tido.pulse_amp('qb1', 'pi2_pulse', 0.095, update = False)
tido.pulse_amp('qb1', 'pi_pulse',  0.2, update = False)
tido.update_config()
runcell('octave calib', r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\scripts\RunTimeDomain.py')

#%% ro params
# chi = -1.14e6
# # tido.element_freq('ro', 6444.72e6, update = False)
# tido.wait_between_seq = 100e3
# tido.pulse_len('ro', 'ro_pulse', 1200, update = False)
# tido.pulse_amp('ro', 'ro_pulse', 0.05, update = False)

# tido.update_config()
# # runcell('octave calib', r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\scripts\RunTimeDomain.py')

# tido.element_freq('ro', 6444.72e6, update = False)
tido.wait_between_seq = 100e3
tido.pulse_len('ro', 'ro_pulse', 10400, update = False)
tido.pulse_amp('ro', 'ro_pulse', 0.05, update = False)

tido.update_config()
runcell('octave calib', r'C:\Users\PHshayg-lab3\OneDrive - Technion\Gal_H\Measurments\scripts\RunTimeDomain.py')

#%% Check mixer qb1
# tido.element_IF('qb1', -150e6)
qubit = 'qb1'
tido.pulse_amp(qubit, 'mixer_cal_pulse', 0.4, update = True)
tido.set_cxa_to_element(qubit, res_bw = 100, vid_bw = 100, mul_span = 8, center = 'LO')
tido.run_continuous(qubit)
tido.CXA.Amp(10)