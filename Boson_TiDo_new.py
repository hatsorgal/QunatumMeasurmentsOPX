# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:38:57 2024

@author: Shay
"""

import pathlib
import sys, os
import os.path

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent /'usefulFunctions'))
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\measurementClass') # for fridge 2 new pc
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\ConfigurationClasses') # for fridge 2 new pc
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\ConfigurationClasses') # for fridge 2 new pc

sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\InstrumentControl') # for fridge 1 new pc


# common:
import matplotlib
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
# OPX Qunatum Machine:
from qm.qua import *

from smart_fit import sFit, non_TimeDomain_fit
import Time_characterzation

from plotting import plot_2D, plot_fft, next_fig_num_by_name
from data_processing import data_to_sigma_z, normalize_data, round_value_by_error, scale_data_units
from scipy.optimize import curve_fit
from scipy.special import laguerre

from tqdm import tqdm

class Boson_TiDo_Chara( Time_characterzation.TiDo_Chara):
#%% initialzation and attributes    
    def __init__(self,Config, #make sure you follow the config example with the names + nust include to sub dicitonaries, one for the opx "opx_config" and an auxillary config "aux_config"
                 main_mm      = 'mm1' ,
                 secondary_mm = 'mm2',# memory modes
                 wait_between_seq_mm = 1e6,
                 is_ECD = False,
                 echo_wait_time = 200,
                 is_geo_phase_corr = False,
                 is_loop_amp_scale = True,
                 **kwargs):
        
        #input:    
        self.main_mm      = main_mm
        self.secondary_mm = secondary_mm
        self.wait_between_seq_mm = wait_between_seq_mm
        self.is_ECD = is_ECD
        self.echo_wait_time = echo_wait_time
        self.is_geo_phase_corr = is_geo_phase_corr
        self.geo_phase_coeff = 0
        self.is_loop_amp_scale = is_loop_amp_scale
        
        super().__init__(Config, **kwargs)
        
        
    def geometric_phase_correction(self, amps, X, Y, mm = None, npulses = 1):
        if mm is None: mm = self.main_mm
        phase_func = self.results['parameters'][mm]['geometric_phase_func']
        params = self.results['parameters'][mm]['geometric_phase_params']
        geometric_phase = -phase_func(amps, *params) /180*np.pi / 2 * npulses
        new_X = np.cos(geometric_phase) * X - np.sin(geometric_phase) * Y
        new_Y = np.cos(geometric_phase) * Y + np.sin(geometric_phase) * X
        return new_X, new_Y
        
    def play_con_disp(self,  qubit = None, mm = None, pulse = None,
                      is_unipolar_pulse = True, amp_scale = None):
        if qubit is None: qubit = self.main_qubit
        if mm is None: qubit = self.main_mm
        if pulse is None:
            pulse = 'asym_pulse'
        
        if is_unipolar_pulse:
            if amp_scale is None:
                play(pulse+'11', mm)
                frame_rotation_2pi(0.5, mm)
                play(pulse+'12', mm)
                align(qubit, mm)
                play('pi_pulse', qubit)
                align(qubit, mm)
                play(pulse+'21', mm)
                frame_rotation_2pi(0.5, mm)
                play(pulse+'22', mm)
                if self.geo_phase_coeff != 0: frame_rotation_2pi(self.geo_phase_coeff, qubit)
            else:
                if type(amp_scale) is list: play((pulse+'11') * amp(amp_scale[0], amp_scale[1], amp_scale[2], amp_scale[3]), mm)
                else: play((pulse+'11') * amp(amp_scale), mm)
                frame_rotation_2pi(0.5, mm)
                if type(amp_scale) is list: play((pulse+'12') * amp(amp_scale[0], amp_scale[1], amp_scale[2], amp_scale[3]), mm)
                else: play((pulse+'12') * amp(amp_scale), mm)
                align(qubit, mm)
                play('pi_pulse', qubit)
                align(qubit, mm)
                if type(amp_scale) is list: 
                    play((pulse+'21') * amp(amp_scale[0], amp_scale[1], amp_scale[2], amp_scale[3]), mm)
                    frame_rotation_2pi(0.5, mm)
                    play((pulse+'22') * amp(amp_scale[0], amp_scale[1], amp_scale[2], amp_scale[3]), mm)
                    if self.geo_phase_coeff != 0: frame_rotation_2pi(self.geo_phase_coeff * (amp_scale[0]*amp_scale[0]+amp_scale[1]*amp_scale[1]), qubit)
                else: 
                    play((pulse+'21') * amp(amp_scale), mm)
                    frame_rotation_2pi(0.5, mm)
                    play((pulse+'22') * amp(amp_scale), mm)
                    if self.geo_phase_coeff != 0: frame_rotation_2pi(self.geo_phase_coeff * amp_scale*amp_scale, qubit)
        else:
            if amp_scale is None:
                play(pulse+'1', mm)
                align(qubit, mm)
                play('pi_pulse', qubit)
                align(qubit, mm)
                play((pulse+'2'), mm)
                if self.geo_phase_coeff != 0: frame_rotation_2pi(self.geo_phase_coeff, qubit)
            else:
                if type(amp_scale) is list: play((pulse+'1') * amp(amp_scale[0], amp_scale[1], amp_scale[2], amp_scale[3]), mm)
                else: play((pulse+'1') * amp(amp_scale), mm)
                align(qubit, mm)
                play('pi_pulse', qubit)
                align(qubit, mm)
                if type(amp_scale) is list: 
                    play((pulse+'2') * amp(amp_scale[0], amp_scale[1], amp_scale[2], amp_scale[3]), mm)
                    if self.geo_phase_coeff != 0: frame_rotation_2pi(self.geo_phase_coeff * (amp_scale[0]*amp_scale[0]+amp_scale[1]*amp_scale[1]), qubit)
                else: 
                    play((pulse+'2') * amp(amp_scale), mm)
                    if self.geo_phase_coeff != 0: frame_rotation_2pi(self.geo_phase_coeff * amp_scale*amp_scale, qubit)
                
    def load_cat_and_back(self, N_avg = 1000, npts = 100, amp_scale_start = 0, amp_scale_stop = 1,
                          mm = None, qubit = None, ro = None, is_ECD = None,
                          is_calibrate_readout = False, N_avg_calib = 1000, is_active_reset_calib = False,
                          wait_time = None,
                          is_sb_cool = False,
                          **kwargs):
        if is_ECD is None: is_ECD = self.is_ECD
        if is_ECD: pulse = 'disp_ECD'
        else: pulse = 'asym_pulse11'
        if qubit is None: qubit = self.main_qubit
        if mm is None: mm = self.main_mm
        if ro is None: ro = self.main_readout
        if wait_time is None: wait_time = self.wait_between_seq_mm
        
        if N_avg_calib < N_avg: N_avg_calib = N_avg
        
        amp_scale_list = np.linspace(amp_scale_start, amp_scale_stop, npts)
        amps = amp_scale_list * self.pulse_amp(mm, pulse)
        self.results['cat_and_back'] = {"mm": mm, "qubit": qubit, "ro": ro, "npts": npts, "amps":amps, "N_avg": N_avg, "N_avg_calib": N_avg_calib//N_avg*N_avg,
                                        'is_calibrate_readout': is_calibrate_readout, "is_active_reset_calib":is_active_reset_calib}
        
        if is_sb_cool: run_time = N_avg * npts * 6 * (self.pulse_len(mm, pulse)*4 + self.pulse_len(mm, 'constant_sideband_cooling_pulse'))
        else: run_time = N_avg * npts * 6 * (self.pulse_len(mm, pulse)*4 + wait_time)
        print('Run time is {}s'.format(round(run_time * 1e-9))) 
        
        if not self.is_loop_amp_scale:
            self._load_py_loop_cat_and_back_prog(is_sb_cool = is_sb_cool, wait_time = wait_time)
        else:
            with program() as self.cat_and_back_prog:
                
                amp_scale = declare(fixed)
                I = declare(fixed)
                Q = declare(fixed)
                n = declare(int)
                qubit_xy_phase = declare(fixed)
                
                with for_(n, 0, n<N_avg, n+1):
                    if is_calibrate_readout:
                        self.calibrate_readout(qubit, ro, I, Q, N_avg = N_avg_calib//N_avg, is_active_reset = is_active_reset_calib, is_sb_cool = is_sb_cool and not is_active_reset_calib, **kwargs)
                    with for_each_(amp_scale, amp_scale_list.tolist()):
                        with for_(qubit_xy_phase, 0.0, qubit_xy_phase<0.9, qubit_xy_phase+0.25):
                            reset_frame(qubit)
                            reset_frame(mm)
                            align(qubit, mm)
                            play('pi2_pulse', qubit)
                            align(qubit, mm)
                            self.play_con_disp(qubit, mm, amp_scale = amp_scale)
                            align(mm, qubit)
                            play('pi_pulse', qubit)
                            align(mm, qubit)
                            self.play_con_disp(qubit, mm, amp_scale = -amp_scale)
                            # self.play_con_disp(qubit, mm, amp_scale = amp_scale)
                            align(qubit, mm)
                            frame_rotation_2pi(qubit_xy_phase, qubit)
                            play('pi2_pulse', qubit)
                            align(qubit, ro, mm)
                            self.perform_full_measurement(I,Q, ro_element = ro,  is_active_reset = False, wait_time = wait_time, is_sb_cool = is_sb_cool, **kwargs)
                            
                        with for_(qubit_xy_phase, 0.0, qubit_xy_phase<0.3, qubit_xy_phase+0.25):
                            reset_frame(qubit)
                            reset_frame(mm)
                            align(qubit, mm)
                            play('pi2_pulse', qubit)
                            align(qubit, mm)
                            self.play_con_disp(qubit, mm, amp_scale = amp_scale)
                            align(mm, qubit)
                            play('pi_pulse', qubit)
                            align(mm, qubit)
                            self.play_con_disp(qubit, mm, amp_scale = -amp_scale)
                            align(qubit, mm)
                            play('pi_pulse', qubit, condition = qubit_xy_phase==0)
                            align(qubit, ro, mm)
                            self.perform_full_measurement(I,Q, ro_element = ro,  is_active_reset = False, wait_time = wait_time, is_sb_cool = is_sb_cool, **kwargs)
                        
        self.last_prog = self.cat_and_back_prog
        
        
    def run_cat_and_back(self, is_save_data = None, prog = None, save_folder = None, **kwargs):
        results_dict = self.results['cat_and_back']
        if prog is None: prog = self.cat_and_back_prog
        if not hasattr(self,'cat_and_back_prog'): raise ValueError("You must run the load function first!")
        
        if self.is_loop_amp_scale:
            results_dict['I'], results_dict['Q'] = self.run_prog(prog, shape = (results_dict['N_avg'], results_dict['npts'], 6), **kwargs)
            
            if results_dict['is_calibrate_readout']:
                results_dict['I_pi'] = self.last_job.result_handles.get('I_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['Q_pi'] = self.last_job.result_handles.get('Q_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['I_nopi'] = self.last_job.result_handles.get('I_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['Q_nopi'] = self.last_job.result_handles.get('Q_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
        else:
            if results_dict['is_calibrate_readout']:
                self.load_pinopi(N_avg = results_dict['N_avg_calib'], is_active_reset = results_dict['is_active_reset_calib'], meas_type = 'full')
                self.run_pinopi(plot=False, is_save_data = False)
                results_dict['I_pi'] = self.last_job.result_handles.get('I_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['Q_pi'] = self.last_job.result_handles.get('Q_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['I_nopi'] = self.last_job.result_handles.get('I_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['Q_nopi'] = self.last_job.result_handles.get('Q_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                
            mm = results_dict['mm']
            prog_id = self.qm.compile(prog)
            amp_scale_list = results_dict['amps']/self.pulse_amp(mm, 'asym_pulse1')
            results_dict['I'] = np.zeros((results_dict['N_avg'], results_dict['npts'], 6))
            results_dict['Q'] = np.zeros((results_dict['N_avg'], results_dict['npts'], 6))
            for i,amp_scale in tqdm(enumerate(amp_scale_list)):
                
                results_dict['I'][:,i,:], results_dict['Q'][:,i,:] = self.run_prog(prog_id, shape = (results_dict['N_avg'], 6), 
                                         overrides_dict={'waveforms': {
                                                       'wf_asym_pulse1_mm2_I': (self.config['waveforms']['wf_asym_pulse1_mm2_I']['samples']*amp_scale).tolist(),
                                                       'wf_asym_pulse2_mm2_I': (self.config['waveforms']['wf_asym_pulse2_mm2_I']['samples']*amp_scale).tolist()}})
                
                
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'cat_and_back', foldername = save_folder)
        return self.plot_cat_and_back(**kwargs)
    
    def plot_cat_and_back(self, fig_num = None, which_data = None, is_geo_phase_corr = False, is_update_geo_phase_corr = False, a = 1, 
                          is_plot_raw = False, inds_to_fit = None, is_plot_calib = True, **kwargs):
        if which_data is None: which_data = self.which_data
        results_dict = self.results['cat_and_back']
        npts = results_dict['npts']
        mm = results_dict['mm']
        data, err = self.process_data([results_dict['I'], results_dict['Q']])
        if results_dict['is_calibrate_readout']:
            data_e, err_e = self.process_data([results_dict['I_pi'], results_dict['Q_pi']])
            data_g, err_g = self.process_data([results_dict['I_nopi'], results_dict['Q_nopi']])
            _,_,means = self.find_threshold(np.append(results_dict['I_pi'],results_dict['I_nopi']), np.append(results_dict['Q_pi'],results_dict['Q_nopi']),
                                            n_means = 2, is_fit_circle = False, is_ascending = data_g.mean()<data_e.mean(),
                                            is_plot = is_plot_calib)
            data, err = data_to_sigma_z(data=data, data_g=means[0], data_e=means[1], err=err, data_g_err=0, data_e_err=0)
        else:
            data, units_prefix, scale_factor = scale_data_units(data)
            err = scale_factor * err
            
        amps = results_dict['amps']
        if is_plot_raw:
            plt.figure()
            plt.plot(results_dict['I'].mean(0), results_dict['Q'].mean(0))
            plt.figure()
            plt.plot(amps, data)
        Y = (data[:,0] - data[:,2])/2
        X = (data[:,1] - data[:,3])/2
        Z = (data[:,4] - data[:,5])/2
        if is_geo_phase_corr: X,Y = self.geometric_phase_correction(amps, X, Y, mm, npulses = 2)
        Y_err = np.sqrt(err[:,0]**2+err[:,2]**2)/2
        X_err = np.sqrt(err[:,1]**2+err[:,3]**2)/2
        Z_err = np.sqrt(err[:,4]**2+err[:,5]**2)/2
        # a = (1/np.sqrt(0.3)*800)
        # b = 200
        # c = -86
        # plt.plot(amps, amps**2*a+amps*b+c)
        
        xlabel = 'Con. disp. amp. [V]' if a == 1 else r'$\mathrm{Re}(\alpha)$'
        fig_num = next_fig_num_by_name('Cat and back')

        plt.figure(fig_num)
        plt.errorbar(amps*a, X, X_err, label = 'X')
        plt.errorbar(amps*a, Y, Y_err, label = 'Y')
        plt.errorbar(amps*a, Z, Z_err, label = 'Z')
        if not results_dict['is_calibrate_readout']:
            plt.ylabel('{} [{}V]'.format(which_data, units_prefix))
        else:
            plt.ylabel(r'$\langle\sigma_z\rangle$')
        
        plt.xlabel(xlabel)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        
        phase = np.unwrap(np.arctan2(Y,X)) * 180 / np.pi
        ind = np.where(amps == 0)[0]
        if len(ind)==1:
            phase = phase - phase[ind]
        phase_err = 180 / np.pi * np.sqrt((Y_err/X/(1+(Y/X)**2))**2+((X_err*Y/X**2/(1+(Y/X)**2))**2))
        purity = np.sqrt(X**2+Y**2+Z**2)
        purity_err = np.sqrt((X_err*2*X/2/purity)**2+(Y_err*2*Y/2/purity)**2+(Z_err**2*Z/2/purity))
        
        f,axs = plt.subplots(2,1, sharex=True)
        plt.sca(axs[0])
        plt.errorbar(amps * a, phase, phase_err, fmt = 'o')
        if inds_to_fit is not None: 
            sfit = sFit('x2', phase[npts//2-inds_to_fit//2:npts//2+inds_to_fit//2], amps[npts//2-inds_to_fit//2:npts//2+inds_to_fit//2], phase_err[npts//2-inds_to_fit//2:npts//2+inds_to_fit//2]) 
            if sfit.is_succeed: plt.plot(amps[npts//2-inds_to_fit//2:npts//2+inds_to_fit//2] * a, sfit.func(amps[npts//2-inds_to_fit//2:npts//2+inds_to_fit//2], *sfit.fit_results))
        else: 
            sfit = sFit('x2', phase, amps, phase_err,)
            if sfit.is_succeed: plt.plot(amps * a, sfit.func(amps, *sfit.fit_results))
        
        plt.ylabel('Phase')
        plt.grid()
        plt.sca(axs[1])
        plt.errorbar(amps * a, purity, purity_err, fmt = 'o')
        plt.ylabel('Purity')
        plt.xlabel(xlabel)
        plt.grid()
        plt.tight_layout()
        
        
        if is_update_geo_phase_corr:
            self.results['parameters'][mm]['geometric_phase_func'] = sfit.func
            self.results['parameters'][mm]['geometric_phase_params'] = sfit.fit_results
        
    def load_out_and_back(self, N_avg = 1000,  N_repeat = 1,
                          amp_npts = 100, 
                          amp_scale_start_pi = 0, amp_scale_stop_pi = 1, 
                          amp_scale_start_nopi = 0, amp_scale_stop_nopi = 1, 
                          phase_npts = 100, 
                          phase_start_pi = 0, phase_stop_pi = 1, 
                          phase_start_nopi = 0, phase_stop_nopi = 1, 
                          wait_time = 1e3,
                          mm = None, qubit = None, ro = None, 
                          pulse1 = 'disp', pulse2 = 'disp',
                          is_scale_first_pulse = True,
                          is_calibrate_readout = False, N_avg_calib = 1000,
                          is_M1 = False, 
                          is_active_reset = False, is_active_reset_calib = False,
                          is_preactive_reset = None,
                          is_sb_cool = False,
                          is_kappa_corr = False,
                          **kwargs):
        if qubit is None: qubit = self.main_qubit
        if mm is None: mm = self.main_mm
        if ro is None: ro = self.main_readout
        if is_preactive_reset is None: is_preactive_reset = self.is_active_reset
        if N_avg_calib < N_avg: N_avg_calib = N_avg
        
        disp_phase_pi_list = np.linspace(phase_start_pi, phase_stop_pi, phase_npts) / 360
        disp_phase_nopi_list = np.linspace(phase_start_nopi, phase_stop_nopi, phase_npts) / 360
        amp_scale_pi_list = np.linspace(amp_scale_start_pi, amp_scale_stop_pi, amp_npts)
        amp_scale_nopi_list = np.linspace(amp_scale_start_nopi, amp_scale_stop_nopi, amp_npts)
        amps_pi = amp_scale_pi_list * np.abs(self.pulse_amp(mm, pulse2))
        amps_nopi = amp_scale_nopi_list * np.abs(self.pulse_amp(mm, pulse2))
        self.results['out_and_back'] = {"mm": mm, "qubit": qubit, "amp_npts": amp_npts, "phase_npts": phase_npts, 
                                        "amps_pi":amps_pi, "disp_phase_pi": disp_phase_pi_list,
                                        "amps_nopi":amps_nopi, "disp_phase_nopi": disp_phase_nopi_list,
                                        "N_avg": N_avg, "N_avg_calib": N_avg_calib//N_avg*N_avg, "wait_time": wait_time, 
                                        "pulse1":pulse1, "pulse2":pulse2,
                                        'is_calibrate_readout': is_calibrate_readout, "is_M1": is_M1}
        
        if is_active_reset: run_time = N_avg * phase_npts * amp_npts * 2 * (self.pulse_len(mm, pulse1)*2)
        elif is_sb_cool: run_time = N_avg * phase_npts * amp_npts * 2 * (self.pulse_len(mm, pulse1)+self.pulse_len(mm, pulse2) + self.pulse_len(mm, 'constant_sideband_cooling_pulse'))
        else: run_time = N_avg * phase_npts * amp_npts * 2 * (self.pulse_len(mm, pulse1)*2 + self.wait_between_seq_mm + wait_time)
        print('Run time is {}s'.format(round(run_time * 1e-9))) 
        
        if is_kappa_corr: 
            kappa = self.results['parameters'][mm]['kappa']
            kappa_factor = np.exp(-(wait_time+self.pulse_len(mm, pulse))*kappa/2)
        
        with program() as self.out_and_back_prog:
            
            amp_scale_pi = declare(fixed)
            amp_scale_nopi = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)
            n = declare(int)
            n_repeat = declare(int)
            is_pi = declare(int)
            disp_phase_pi = declare(fixed)
            disp_phase_nopi = declare(fixed)
            
            with for_(n, 0, n<N_avg, n+1):
                if is_calibrate_readout:
                    self.calibrate_readout(qubit, ro, I, Q, N_avg = N_avg_calib//N_avg, is_active_reset = is_active_reset_calib)
                with for_each_((disp_phase_pi, disp_phase_nopi), (disp_phase_pi_list.tolist(), disp_phase_nopi_list.tolist())):
                    with for_each_((amp_scale_pi, amp_scale_nopi), (amp_scale_pi_list.tolist(), amp_scale_nopi_list.tolist())):
                        with for_(is_pi, 0, is_pi<2, is_pi+1):
                            if is_preactive_reset: 
                                self.perform_full_measurement(I,Q, ro_element = ro,  is_active_reset = True, is_save = False, wait_time = self.wait_between_seq_mm, **kwargs)
                                align(qubit, ro, mm)
                            reset_frame(qubit, mm)
                            align(qubit, mm, ro)
                            with for_(n_repeat, 0, n_repeat<N_repeat, n_repeat + 1):
                                with if_(is_pi==1):
                                    play('pi_pulse', qubit)
                                    align(qubit, mm, ro)
                                    if is_scale_first_pulse: 
                                        play(pulse1 * amp(amp_scale_pi), mm)
                                    else: 
                                        play(pulse1, mm)
                                        
                                        align(qubit, mm)
                                        play('pi_pulse', qubit)
                                        align(qubit, mm)
                                        play('asym_pulse2', mm)
                                        align(qubit, mm)
                                        play('pi_pulse', qubit)
                                        align(qubit, mm)
                                        play(pulse1  * amp(-1), mm)
                                        align(qubit, mm)
                                        play('pi_pulse', qubit)
                                        align(qubit, mm)
                                        play('asym_pulse2' * amp(-1), mm)
                                        
                                        align(qubit, mm)
                                        play('pi_pulse', qubit)
                                        
                                    if wait_time>=16: wait(int(wait_time//4), mm)
                                    frame_rotation_2pi(disp_phase_pi+0.5, mm)
                                    play(pulse2 * amp(amp_scale_pi), mm)
                                    frame_rotation_2pi(-disp_phase_pi-0.5, mm)
                                    align(qubit, mm, ro)
                                    play('pi_pulse', qubit)
                                with else_():
                                    align(qubit, mm, ro)
                                    if is_scale_first_pulse: play(pulse1 * amp(amp_scale_nopi), mm)
                                    else: 
                                        play(pulse1, mm)
                                        
                                        align(qubit, mm)
                                        play('pi_pulse', qubit)
                                        align(qubit, mm)
                                        play('asym_pulse2', mm)
                                        align(qubit, mm)
                                        
                                        play('pi_pulse', qubit)
                                        align(qubit, mm)
                                        play(pulse1 * amp(-1), mm)
                                        align(qubit, mm)
                                        play('pi_pulse', qubit)
                                        align(qubit, mm)
                                        play('asym_pulse2' * amp(-1), mm)

                                        align(qubit, mm)
                                        play('pi_pulse', qubit)
                                        
                                    if wait_time>=16: wait(int(wait_time//4), mm)
                                    frame_rotation_2pi(disp_phase_nopi+0.5, mm)
                                    play(pulse2 * amp(amp_scale_nopi), mm)
                                    frame_rotation_2pi(-disp_phase_nopi-0.5, mm)
                            align(qubit, mm, ro)
                            if is_M1: 
                                self.perform_full_measurement(I,Q, ro = ro, I_output_name = 'M1_I', Q_output_name = 'M1_Q', is_active_reset = False, is_loop_reset = False, is_wait = False, is_sb_cool = False)
                                if self.wait_after_reset >= 16: wait(self.wait_after_reset//4, ro)
                                align(qubit, mm, ro)
                            play('con_pi_pulse', qubit)
                            align(qubit, mm, ro)
                            self.perform_full_measurement(I,Q, ro_element = ro,  is_active_reset = is_active_reset, is_sb_cool = is_sb_cool, wait_time = self.wait_between_seq_mm,  **kwargs)
        self.last_prog = self.out_and_back_prog
        
        
    def run_out_and_back(self, is_save_data = None, **kwargs):
        results_dict = self.results['out_and_back']
        if not hasattr(self,'out_and_back_prog'): raise ValueError("You must run the load function first!")
        
        results_dict['I'], results_dict['Q'] = self.run_prog(self.out_and_back_prog, shape = (results_dict['N_avg'],  results_dict['phase_npts'], results_dict['amp_npts'], 2), **kwargs)
        
        if results_dict['is_calibrate_readout']:
            results_dict['I_pi'] = self.last_job.result_handles.get('I_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'], -1))
            results_dict['Q_pi'] = self.last_job.result_handles.get('Q_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'], -1))
            results_dict['I_nopi'] = self.last_job.result_handles.get('I_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'], -1))
            results_dict['Q_nopi'] = self.last_job.result_handles.get('Q_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'], -1))
            
        if results_dict['is_M1']:
            results_dict['I_M1'] = self.last_job.result_handles.get('M1_I').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['phase_npts'], results_dict['amp_npts'], 2))
            results_dict['Q_M1'] = self.last_job.result_handles.get('M1_Q').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['phase_npts'], results_dict['amp_npts'], 2))
                
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'out_and_back')
        return self.plot_out_and_back(**kwargs)
    
    def plot_out_and_back(self, fig_num = None, which_data = None, is_geo_phase_corr = False, is_M1 = None, is_calibrate_readout = None, 
                          is_fit = True,
                          amp_fit_start_pi = -np.inf, amp_fit_stop_pi = np.inf,
                          amp_fit_start_nopi = -np.inf, amp_fit_stop_nopi = np.inf,
                          is_plot_2D = True, **kwargs):
        if which_data is None: which_data = self.which_data
        results_dict = self.results['out_and_back']
        if is_M1 is None: is_M1 = results_dict['is_M1']
        if is_calibrate_readout is None: is_calibrate_readout = results_dict['is_calibrate_readout']
        mm = results_dict['mm']
        data, err = self.process_data([results_dict['I'], results_dict['Q']], is_mean = not (is_M1 and is_calibrate_readout), is_calc_stat_error = not (is_M1 and is_calibrate_readout))
        if is_calibrate_readout:
            data_e, err_e = self.process_data([results_dict['I_pi'], results_dict['Q_pi']])
            data_g, err_g = self.process_data([results_dict['I_nopi'], results_dict['Q_nopi']])
            data, err = data_to_sigma_z(data=data, data_g=data_g, data_e=data_e, err=err, data_g_err=err_g, data_e_err=err_e)
            
            if is_M1:
                data_M1, _ = self.process_data([results_dict['I_M1'], results_dict['Q_M1']], is_mean = False)
                data_M1_sigma_z, _ = data_to_sigma_z(data=data_M1, data_g=data_g, data_e=data_e, is_thresholding = True)
                
                inds_g = np.where(data_M1_sigma_z==-1)
                inds_e = np.where(data_M1_sigma_z==1)
                data_M1_g = data.copy()
                data_M1_g[inds_e] = np.nan
                data_M1_e = data.copy()
                data_M1_e[inds_e] = np.nan
                data_nopi = np.nanmean(data_M1_g, 0)[:,:,0]
                data_pi = np.nanmean(data_M1_e, 0)[:,:,1]
                
            else:
                data_nopi = data[:,:,0]
                data_pi = data[:,:,1]
        else:
            data, units_prefix, scale_factor = scale_data_units(data)
            data_nopi = data[:,:,0]
            data_pi = data[:,:,1]
            
        amps_pi = results_dict['amps_pi']
        phases_pi = results_dict['disp_phase_pi'] * 360
        amps_nopi = results_dict['amps_nopi']
        phases_nopi = results_dict['disp_phase_nopi'] * 360
        
        
        
        fig_num = next_fig_num_by_name('Out and back')
        if is_plot_2D:
            if is_calibrate_readout: vmax, vmin = 1,-1
            else: vmax, vmin = np.max(np.array([data_nopi,data_pi]).flatten()), np.min(np.array([data_nopi,data_pi]).flatten())
            
            fig,axs = plt.subplots(1,2, sharey = True, num = fig_num)
            plot_2D(data_pi, amps_pi, phases_pi, vmax = vmax, vmin = vmin, ax = axs[0], 
                    xlabel = 'Displacement amp. [V]', ylabel = 'Phase [Deg]', 
                    is_colorbar = False)
            plot_2D(data_nopi, amps_nopi, phases_nopi, vmax = vmax, vmin = vmin, ax = axs[1], 
                    xlabel = 'Displacement amp. [V]',   
                    is_colorbar = False)
            
            axs[0].set_title(r'Prepare $|e\rangle$')
            axs[1].set_title(r'Prepare $|g\rangle$')
            plt.tight_layout()
            cmap = matplotlib.cm.get_cmap('seismic')
            normalizer = matplotlib.colors.Normalize(vmin, vmax)
            im = matplotlib.cm.ScalarMappable(norm=normalizer, cmap  = cmap)
            plt.colorbar(mappable = im, ax=axs, location='bottom')  
        else:
            plt.figure(fig_num)
            for i,dat in enumerate(data_pi):
                p,=plt.plot(amps_pi, data_pi[i], '--', label = phases_pi[i])
                plt.plot(amps_nopi, data_nopi[i], '-', label = phases_nopi[i], color = p.get_color())
            plt.legend(fontsize = 8, ncol = 2)
            plt.xlabel('Displacement amp. [V]')
            plt.ylabel(r'$\langle \sigma_z\rangle$')
            plt.tight_layout()
            
            
        if is_fit:
            centers_pi = []
            centers_pi_err = []
            amps_fitted_pi = []
            for dat_pi,amp in zip(data_pi.transpose(), amps_pi):
                if amp > amp_fit_start_pi and amp<amp_fit_stop_pi:
                    sfit = non_TimeDomain_fit('Gaussian', dat_pi, phases_pi)
                    if sfit.fit_params != 0: 
                        centers_pi.append(sfit.fit_params[1])
                        centers_pi_err.append(np.sqrt(sfit.fit_cov[1][1]))
                        amps_fitted_pi.append(amp)
                
            amps_fitted_pi = np.array(amps_fitted_pi)
            
            centers_nopi = []
            centers_nopi_err = []
            amps_fitted_nopi = []
            for dat_nopi,amp in zip(data_nopi.transpose(), amps_nopi):
                if amp > amp_fit_start_nopi and amp<amp_fit_stop_nopi:
                    sfit = non_TimeDomain_fit('Gaussian', dat_nopi, phases_nopi)
                    if sfit.fit_params != 0: 
                        centers_nopi.append(sfit.fit_params[1])
                        centers_nopi_err.append(np.sqrt(sfit.fit_cov[1][1]))
                        amps_fitted_nopi.append(amp)
                        
            amps_fitted_nopi = np.array(amps_fitted_nopi)
            
            amps_squared_pi = amps_fitted_pi**2
            amps_squared_nopi = amps_fitted_nopi**2
            
            plt.figure()
            plt.errorbar(amps_squared_pi, centers_pi, centers_pi_err, fmt = 'or', label = r'$\pi$')
            plt.errorbar(amps_squared_nopi, centers_nopi, centers_nopi_err, fmt = 'ob', label = r'no-$\pi$')
            plt.legend()
            
            
            sfit_pi = non_TimeDomain_fit('Line', centers_pi, amps_squared_pi, centers_pi_err)
            sfit_nopi = non_TimeDomain_fit('Line', centers_nopi, amps_squared_nopi, centers_nopi_err)
            
            if sfit_pi.is_succeed: plt.plot(amps_squared_pi, sfit_pi.func(np.array(amps_squared_pi), *(sfit_pi.fit_params)), '-r')
            if sfit_nopi.is_succeed: plt.plot(amps_squared_nopi, sfit_nopi.func(np.array(amps_squared_nopi), *(sfit_nopi.fit_params)), '-b')
            
            plt.ylabel('Phase [Deg]')
            plt.xlabel('Displacement amp. squared [$V^2$]')
            
            plt.grid()
            plt.tight_layout()
            
            if sfit_nopi.is_succeed and sfit_pi.is_succeed:
                phase_off_pi = sfit_pi.fit_params[1]
                phase_off_pi_err = np.sqrt(sfit_pi.fit_cov[1,1])
                phase_off_nopi = sfit_nopi.fit_params[1]
                phase_off_nopi_err = np.sqrt(sfit_nopi.fit_cov[1,1])
                
                chi = (phase_off_pi-phase_off_nopi)/360/results_dict['wait_time']
                chi_err = np.sqrt(phase_off_pi_err**2+phase_off_nopi**2)/results_dict['wait_time']/360
                print('Chi/2pi = {} +- {} kHz'.format(*round_value_by_error(chi*1e6,chi_err*1e6)))
                
                freq_off = (phase_off_pi+phase_off_nopi)/360/2/results_dict['wait_time']
                freq_off_err = np.sqrt(phase_off_pi_err**2+phase_off_nopi**2)/2/results_dict['wait_time']/360
                print('Detuning = {} +- {} kHz'.format(*round_value_by_error(freq_off*1e6,freq_off_err*1e6)))
                
                SelfKerr = sfit_nopi.fit_params[0]/360/results_dict['wait_time']
                SelfKerr_err = np.sqrt(sfit_nopi.fit_cov[0,0])/360/results_dict['wait_time']
                chiPrime = sfit_pi.fit_params[0]/360/results_dict['wait_time']-SelfKerr
                chiPrime_err = np.sqrt(sfit_pi.fit_cov[0,0]/results_dict['wait_time']**2 + SelfKerr_err**2)/360
                
                print('K/2pi = {} +- {} kHz/V^2'.format(*round_value_by_error(SelfKerr*1e6,SelfKerr_err*1e6)))
                print("chiPrime/2pi = {} +- {} kHz/V^2".format(*round_value_by_error(chiPrime*1e6,chiPrime_err*1e6)))
        
        
        
        if is_M1:
            if not is_calibrate_readout:
                data_M1_sigma_z, _ = self.process_data([results_dict['I_M1'], results_dict['Q_M1']])
            else:
                data_M1_sigma_z = data_M1_sigma_z.mean(0)
                
            fig,axs = plt.subplots(1,2, sharey = True)
            
            vmax = data_M1_sigma_z.max()
            vmin = data_M1_sigma_z.min()
            
            plot_2D(data_M1_sigma_z[:,:,1], amps_pi, phases_pi, vmax = vmax, vmin = vmin, ax = axs[0], 
                    xlabel = 'Displacement amp. [V]', ylabel = 'Phase [Deg]', 
                    is_colorbar = False)
    
            plot_2D(data_M1_sigma_z[:,:,0], amps_nopi, phases_nopi, vmax = vmax, vmin = vmin, ax = axs[1], 
                    xlabel = 'Displacement amp. [V]',   
                    is_colorbar = False)
        
            axs[0].set_title(r'Prepare $|e\rangle$')
            axs[1].set_title(r'Prepare $|g\rangle$')
            plt.tight_layout()
            cmap = matplotlib.cm.get_cmap('seismic')
            normalizer = matplotlib.colors.Normalize(vmin, vmax)
            im = matplotlib.cm.ScalarMappable(norm=normalizer, cmap  = cmap)
            plt.colorbar(mappable = im, ax=axs, location='bottom')  
            plt.suptitle('M1')
            

    def load_char_func_line(self, N_avg = 1000, npts = 100, amp_scale_start = -1, amp_scale_stop = 1,
                            angle_2pi = 0,
                          mm = None, qubit = None, ro = None,
                          is_calibrate_readout = False, is_active_reset_calib = None, N_avg_calib = 1000,
                          is_preactive_reset = False, is_M1 = False, is_mid_M1 = False,
                          prepare = None,
                          is_check_sigma_z = False,
                          is_sb_cool = None,
                          prepare_dict = {},
                          **kwargs):
        if qubit is None: qubit = self.main_qubit
        if mm is None: mm = self.main_mm
        if ro is None: ro = self.main_readout
        if is_sb_cool is None: is_sb_cool = self.is_sb_cool
        if is_active_reset_calib is None: is_active_reset_calib = self.is_active_reset
        pulse = 'disp_ECD' if self.is_ECD else 'asym_pulse11'
        if N_avg_calib < N_avg: N_avg_calib = N_avg
        
        #test
        phase_list = np.linspace(0,0.75,4)
        
        amp_scale_list = np.linspace(amp_scale_start, amp_scale_stop, npts)
        amps = amp_scale_list * np.abs(self.pulse_amp(mm, pulse))
        self.results['char_func_line'] = {"mm": mm, "qubit": qubit, "ro":ro, "npts": npts, "amps":amps,
                                          "N_avg": N_avg, "N_avg_calib": N_avg_calib//N_avg*N_avg,
                                          "is_calibrate_readout": is_calibrate_readout,
                                          "is_active_reset_calib":is_active_reset_calib,
                                          "angle_2pi": angle_2pi,
                                          "is_M1": is_M1, "is_mid_M1" : is_mid_M1, "is_check_sigma_z": is_check_sigma_z,
                                          "is_preactive_reset": is_preactive_reset}
        if prepare is not None:
            self.results['char_func_line']['prepare'] =  prepare.__name__
            self.results['char_func_line']['prepare_dict'] = prepare_dict
        else:
            self.results['char_func_line']['prepare'] = None

        
        if is_sb_cool: run_time = N_avg * npts * 4 * (self.pulse_len(mm, pulse)*2 + self.pulse_len('ro', 'constant_sideband_cooling_pulse'))
        else: run_time = N_avg * npts * 4 * (self.pulse_len(mm, pulse)*2 + self.wait_between_seq_mm)
        print('Run time is {}s'.format(round(run_time * 1e-9))) 
        if not self.is_loop_amp_scale:
            self._load_py_loop_char_func_line_prog(is_sb_cool, prepare, prepare_dict)
        else:
            with program() as self.char_func_line_prog:
                
                amp_scale = declare(fixed)
                I = declare(fixed)
                Q = declare(fixed)
                n = declare(int)
                qubit_xy_phase = declare(fixed)
                
                with for_(n, 0, n<N_avg, n+1):
                    if is_calibrate_readout:
                        self.calibrate_readout(qubit, ro, I, Q, N_avg = N_avg_calib//N_avg, is_active_reset = is_active_reset_calib, is_sb_cool = is_sb_cool and not is_active_reset_calib, wait_time = self.wait_between_seq_mm, **kwargs)
                        align(qubit, ro)
                        reset_phase(ro)
                    with for_each_(amp_scale, amp_scale_list.tolist()):
                        with for_each_(qubit_xy_phase,phase_list.tolist()):
                            if is_preactive_reset: 
                                self.perform_full_measurement(I,Q, ro = ro,  is_save = False, is_active_reset = True, is_wait = False)
                                align(ro, mm, qubit)
                            reset_frame(qubit)
                            reset_frame(mm)
                            if prepare is not None:
                                if is_mid_M1:
                                    prepare(**prepare_dict, I=I, Q=Q, ro = ro, I_output_name = 'mid_M1_I', Q_output_name = 'mid_M1_Q', is_active_reset = False, is_sb_cool = False)
                                else:
                                    prepare(**prepare_dict)
                            if is_M1: 
                                align(ro, mm, qubit)
                                self.perform_full_measurement(I,Q, ro = ro, I_output_name = 'M1_I', Q_output_name = 'M1_Q', is_active_reset = False, is_sb_cool = False, is_wait = False)
                                if self.wait_after_reset >= 16:
                                    wait(self.wait_after_reset//4, ro)
                                    align(ro, qubit, mm)
                                    reset_phase(ro)
                                    reset_phase(qubit)
                                    reset_frame(qubit)
                                    reset_frame(ro)
                            play('pi2_pulse', qubit)
                            align(qubit, mm)
                            if angle_2pi!= 0: frame_rotation_2pi(angle_2pi, mm)
                            self.play_con_disp(qubit, mm, amp_scale = amp_scale)
                            align(qubit, mm)
                            frame_rotation_2pi(qubit_xy_phase, qubit)
                            if not is_check_sigma_z: play('pi2_pulse', qubit)
                            align(qubit, ro)
                            self.perform_full_measurement(I,Q, ro = ro, wait_time = self.wait_between_seq_mm, is_active_reset = False, is_sb_cool = is_sb_cool, **kwargs)
                            align(qubit, ro)
                    
        self.last_prog = self.char_func_line_prog
        
    def run_char_func_line(self, is_save_data = None, save_folder = None, **kwargs):
        results_dict = self.results['char_func_line']
        if not hasattr(self,'char_func_line_prog'): raise ValueError("You must run the load function first!")
        if self.is_loop_amp_scale:
            results_dict['I'], results_dict['Q'] = self.run_prog(self.char_func_line_prog, shape = (results_dict['N_avg'], results_dict['npts'], 4), **kwargs)
        
            if results_dict['is_calibrate_readout']:
                results_dict['I_pi'] = self.last_job.result_handles.get('I_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['Q_pi'] = self.last_job.result_handles.get('Q_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['I_nopi'] = self.last_job.result_handles.get('I_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['Q_nopi'] = self.last_job.result_handles.get('Q_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                
                if results_dict['is_M1']:
                    results_dict['I_M1'] = self.last_job.result_handles.get('M1_I').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['npts'], 4))
                    results_dict['Q_M1'] = self.last_job.result_handles.get('M1_Q').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['npts'], 4))
                    
                if results_dict['is_mid_M1']:
                    results_dict['I_mid_M1'] = self.last_job.result_handles.get('mid_M1_I').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['npts'], 4))
                    results_dict['Q_mid_M1'] = self.last_job.result_handles.get('mid_M1_Q').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['npts'], 4))
        else:   
            mm = results_dict['mm']
            prog_id = self.qm.compile(self.char_func_line_prog)
            amp_scale_list = results_dict['amps']/self.pulse_amp(mm, 'asym_pulse11')
            
            if results_dict['is_calibrate_readout']:
                self.load_pinopi(N_avg = results_dict['N_avg_calib'], is_active_reset = results_dict['is_active_reset_calib'], meas_type = 'full')
                self.run_pinopi(plot=False, is_save_data = False)
                results_dict['I_pi'] = self.last_job.result_handles.get('I_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['Q_pi'] = self.last_job.result_handles.get('Q_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['I_nopi'] = self.last_job.result_handles.get('I_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['Q_nopi'] = self.last_job.result_handles.get('Q_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                
            results_dict['I'] = np.zeros((results_dict['N_avg'], results_dict['npts'], 4))
            results_dict['Q'] = np.zeros((results_dict['N_avg'], results_dict['npts'], 4))
            if results_dict['is_M1']:
                results_dict['I_M1'] = np.zeros((results_dict['N_avg'], results_dict['npts'], 4))
                results_dict['Q_M1'] = np.zeros((results_dict['N_avg'], results_dict['npts'], 4))
            if results_dict['is_mid_M1']:
                results_dict['I_mid_M1'] = np.zeros((results_dict['N_avg'], results_dict['npts'], 4))
                results_dict['Q_mid_M1'] = np.zeros((results_dict['N_avg'], results_dict['npts'], 4))
                
            for i,amp_scale in tqdm(enumerate(amp_scale_list)):
                
                results_dict['I'][:,i,:], results_dict['Q'][:,i,:] = self.run_prog(prog_id, shape = (results_dict['N_avg'], 4), 
                                         overrides_dict={'waveforms': {
                                                       'wf_asym_pulse1_mm2_I': (self.config['waveforms']['wf_asym_pulse1_mm2_I']['samples']*amp_scale).tolist(),
                                                       'wf_asym_pulse2_mm2_I': (self.config['waveforms']['wf_asym_pulse2_mm2_I']['samples']*amp_scale).tolist()}})
                if results_dict['is_M1']:
                    results_dict['I_M1'][:,i,:] = self.last_job.result_handles.get('I_M1').fetch_all()['value'].reshape((results_dict['N_avg'], 4))
                    results_dict['Q_M1'][:,i,:] = self.last_job.result_handles.get('Q_M1').fetch_all()['value'].reshape((results_dict['N_avg'], 4))
                if results_dict['is_mid_M1']:
                    results_dict['I_mid_M1'][:,i,:] = self.last_job.result_handles.get('I_mid_M1').fetch_all()['value'].reshape((results_dict['N_avg'], 4))
                    results_dict['Q_mid_M1'][:,i,:] = self.last_job.result_handles.get('Q_mid_M1').fetch_all()['value'].reshape((results_dict['N_avg'], 4))
                    
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'char_func_line', foldername = save_folder)
        return self.plot_char_func_line(**kwargs)
    
    
    def plot_char_func_line(self, fig_num = None, which_data = None, 
                            is_geo_phase_corr = None, is_fill_symmetric = None, is_M1 = None, is_mid_M1 = None,  post_selection = None,
                            is_calibrate_readout = None, is_fit = True, fit_func = 'coherent', max_fock = 5, a_guess = 0, delta_x_fit = None,
                            is_plot_calib = True, is_plot_M1 = True,
                            ax = None,
                            is_normalize = True,
                            a = 1, a_err = 0, **kwargs):
        if which_data is None: which_data = self.which_data
        if is_geo_phase_corr is None: is_geo_phase_corr = self.is_geo_phase_corr
        results_dict = self.results['char_func_line']
        npts = results_dict['npts']
        if is_fill_symmetric is None: is_fill_symmetric = (np.abs(np.sum(results_dict['amps']))>1e-6 and results_dict['is_calibrate_readout'])
        mm = results_dict['mm']
        if is_M1 is None: is_M1 = results_dict['is_M1']
        if is_mid_M1 is None: is_mid_M1 = results_dict['is_mid_M1']
        if is_calibrate_readout is None: is_calibrate_readout = results_dict['is_calibrate_readout']
        data, err = self.process_data([results_dict['I'], results_dict['Q']], 
                                      is_mean = not (is_M1 and is_calibrate_readout), 
                                      is_calc_stat_error = not (is_M1 and is_calibrate_readout))
        
        if is_calibrate_readout:
            data_e, err_e = self.process_data([results_dict['I_pi'], results_dict['Q_pi']])
            data_g, err_g = self.process_data([results_dict['I_nopi'], results_dict['Q_nopi']])
            _,_,means = self.find_threshold(np.append(results_dict['I_pi'],results_dict['I_nopi']), np.append(results_dict['Q_pi'],results_dict['Q_nopi']), n_means = 2, is_fit_circle = False,
                                            is_ascending = data_g.mean()<data_e.mean(), is_plot = is_plot_calib)
            data, err = data_to_sigma_z(data=data, data_g=means[0], data_e=means[1], err=err, data_g_err=0, data_e_err=0, 
                                        is_thresholding = is_M1)
            data = data
        
            if is_M1:
                data_M1, _ = self.process_data([results_dict['I_M1'], results_dict['Q_M1']], is_mean = False)
                data_M1_sigma_z, _ = data_to_sigma_z(data=data_M1, data_g=data_g, data_e=data_e, is_thresholding = True)
                if is_mid_M1:
                    data_mid_M1, _ = self.process_data([results_dict['I_mid_M1'], results_dict['Q_mid_M1']], is_mean = False)
                    data_mid_M1_sigma_z, _ = data_to_sigma_z(data=data_mid_M1, data_g=data_g, data_e=data_e, is_thresholding = True)
                    inds = np.where(data_mid_M1_sigma_z==1) #remove excited
                    data = data
                    data_post_mid_M1 = data.copy()
                    data_post_mid_M1[inds] = np.nan
                    data = data_post_mid_M1.copy()
                
                if post_selection is not None:
                    if post_selection == 'g': 
                        inds = np.where(data_M1_sigma_z==1) #remove excited
                        data = data
                    elif post_selection == 'e': 
                        inds = np.where(data_M1_sigma_z==-1) #remove ground.
                    data_post_M1 = data.copy()
                    data_post_M1[inds] = np.nan
                    data = np.nanmean(data_post_M1, 0)
                    NN = np.sum(~np.isnan(data_post_M1),0)
                    err =  np.nanstd(data_post_M1, 0)/np.sqrt(NN)
                else:
                    err =  sp.stats.sem(-data * data_M1_sigma_z,0)
                    data = (-data * data_M1_sigma_z).mean(0)
        
        else:
            data, units_prefix, scale_factor = scale_data_units(data)
            err = scale_factor * err
        
        if is_fill_symmetric: 
            amps = np.append(-np.flip(results_dict['amps'][1:]),results_dict['amps'])
            if not results_dict['is_check_sigma_z']:
                Y = np.append(np.flip((data[1:,0]-data[1:,2])/2), (data[:,0]-data[:,2])/2)
                X = np.append(-np.flip((data[1:,1]-data[1:,3])/2), (data[:,1]-data[:,3])/2)
            else:
                Y = np.append(np.flip((data[1:,0]+data[1:,2])/2), (data[:,0]+data[:,2])/2)
                X = np.append(-np.flip((data[1:,1]+data[1:,3])/2), (data[:,1]+data[:,3])/2)
            Y_err = np.append(np.flip(np.sqrt(err[1:,0]**2+err[1:,2]**2)/2), np.sqrt(err[:,0]**2+err[:,2]**2)/2)
            X_err = np.append(np.flip(np.sqrt(err[1:,1]**2+err[1:,3]**2)/2), np.sqrt(err[:,1]**2+err[:,2]**3)/2)
        else: 
            amps = results_dict['amps']
            if not results_dict['is_check_sigma_z']:
                Y = (data[:,0]-data[:,2])/2
                X = (data[:,1]-data[:,3])/2
            else:
                Y = (data[:,0]+data[:,2])/2
                X = (data[:,1]+data[:,3])/2
            Y_err = np.sqrt(err[:,0]**2+err[:,2]**2)/2
            X_err = np.sqrt(err[:,1]**2+err[:,3]**2)/2
        if is_geo_phase_corr: X,Y = self.geometric_phase_correction(amps, X, Y, mm)
        
            
        # phase = np.arctan2(Y,X) * 180 / np.pi
        # purity = np.sqrt(X**2+Y**2)
        # purity = purity/purity.max()
        # purity_error = np.sqrt((X_err)**2+()**2)
        # fig,axs = plt.subplots(2,1, sharex = True, num = fig_num)
        # plt.sca(axs[0])
        # plt.plot(amps, purity)
        # plt.ylabel(r'Purity')
        # plt.grid()
        # plt.sca(axs[1])
        # plt.plot(amps, phase)
        # plt.xlabel('Con. disp. amp. [V]')
        # plt.ylabel(r'Geometric phase $[^\circ]$')
        # plt.grid()
        # plt.tight_layout()
        if is_normalize:
            norm = np.abs(Y[np.abs(amps)<=1e-5]).max()
            if Y[np.argmax(np.abs(Y))]<0:
                Y = -Y
                x = -X
            
            X = X/norm
            Y = Y/norm
            X_err = X_err/norm
            Y_err = Y_err/norm
        
        if ax is None:
            fig_num = next_fig_num_by_name('Char func line')
            plt.figure(fig_num)
        else: plt.sca(ax)
        xerr = np.abs(amps*a_err) if a_err!=0 and a!=1 else None
        plt.errorbar(amps*a, X, yerr=X_err, xerr = xerr, fmt = 'ob', label = 'X')
        plt.errorbar(amps*a, Y, yerr=Y_err, xerr = xerr, fmt = 'or', label = 'Y')
        
        ret = 0,0
        if delta_x_fit is None: delta_x_fit = (npts-1)//2
        if is_fit:
            if fit_func == 'coherent' or fit_func == 'thermal' or fit_func == 'parabula':
                sigma=0
                
                if fit_func == 'coherent':
                    abs_XY = np.sqrt(X**2+Y**2)
                    abs_XY_err = np.sqrt(((X_err*2*X)**2+(Y_err*2*Y)**2)/(X**2+Y**2)/2)
                    # plt.errorbar(amps, abs_XY, abs_XY_err, fmt = 'ok', label = 'abs')
                    sfit_abs = non_TimeDomain_fit('Gaussian', abs_XY, amps, abs_XY_err)
                    if sfit_abs.is_succeed:
                        sigma = sfit_abs.fit_params[2]
                        is_flip_sign = sigma < 0
                        if is_flip_sign: sigma = -sigma
                        sigma_err = np.sqrt(sfit_abs.fit_cov[2,2])
                        if a is None: print('\n Condisp amplitude = {}+-{} 1/Volt'.format(*round_value_by_error(1/sigma, sigma_err/sigma**2)))

                    # plt.plot(amps, abs_XY, 'o')
                    sfit_complex = non_TimeDomain_fit('GaussianCosiSin', Y, amps, itrace = X)
                    if sigma != 0: sfit_complex.guess[2] = sigma
                    fit_params, fit_cov = sfit_complex._fit()
                    if sfit_complex.is_succeed:
                        Amp, Freq, sigma, Delta, Center, Offset = fit_params
                        # Amp, Freq, sigma, Delta,  Offset = fit_params
                        freq_err = np.sqrt(fit_cov[1,1])
                        sigma_err = np.sqrt(fit_cov[2,2])
                        if a != 1: 
                            sigma = 1/a
                            sigma_err = a_err/a**2
                            
                        alpha = Freq*np.pi*sigma
                        alpha_err = np.sqrt((freq_err*sigma)**2+(sigma_err*Freq)**2)*2*np.pi
                        plt.plot(amps*a, sfit_complex.func(amps, *fit_params)[:len(amps)], '-r')
                        plt.plot(amps*a, sfit_complex.func(amps, *fit_params)[len(amps):], '-b')
                        plt.plot(amps*a, sfit_abs.func(amps, fit_params[0], fit_params[4], fit_params[2], 0), '--k')
                        if a == 1: print('\n Condisp amplitude = {}+-{} 1/Volt'.format(*round_value_by_error(1/fit_params[2], np.sqrt(fit_cov[2,2])/fit_params[2]**2)))
                        print('\n alpha = {}+-{}'.format(*round_value_by_error(alpha,alpha_err)))
                        print(f'\n alpha = {alpha:.2f}+-{alpha_err:.2f}')
                        
                elif fit_func == 'thermal':
                    # sfit_abs = non_TimeDomain_fit('Symmetric_Gaussian', Y, amps*a, Y_err)
                    sfit_abs = non_TimeDomain_fit('Symmetric_Gaussian', Y[npts//2-delta_x_fit:npts//2+delta_x_fit+1],
                                                  amps[npts//2-delta_x_fit:npts//2+delta_x_fit+1]*a,
                                                  Y_err[npts//2-delta_x_fit:npts//2+delta_x_fit+1])
                    if sfit_abs.is_succeed:
                        sigma = sfit_abs.fit_params[1]
                        sigma_err = np.sqrt(sfit_abs.fit_cov[1,1])
                        if a == 1: print('\n Condisp amplitude = {}+-{} 1/Volt'.format(*round_value_by_error(1/sigma, sigma_err/sigma**2)))

                    if sfit_abs.is_succeed:
                        # plt.plot(amps*a, sfit_abs.func(amps*a, *sfit_abs.fit_params), '--k')
                        plt.plot(amps[npts//2-delta_x_fit:npts//2+delta_x_fit+1]*a, sfit_abs.func(amps[npts//2-delta_x_fit:npts//2+delta_x_fit+1]*a, *sfit_abs.fit_params), '--k')
                        if a != 1:
                            nbar =  1/(sigma)**2/2-1/2
                            nbar_err = np.sqrt((sigma_err/sigma**3/2)**2)
                            print('\n nbar = {}+-{}'.format(*round_value_by_error(nbar, nbar_err)))
                            ret = nbar, nbar_err
                    else: ret = 0,0
                    
                elif fit_func == 'parabula':
                    # from scipy import odr
                    # def parabula(p, x):
                        # return p[0]*(x)**2 + p[1] + p[2]*x**4
                    
                    # model = odr.Model(parabula)
                    # sx = amps*a_err
                    # sx = sx+1e-10
                    # mydata = odr.RealData(amps*a, Y, sx = sx, sy = Y_err)
                    # myodr = odr.ODR(mydata, model, beta0=[0.1, 0, 1])
                    # myoutput = myodr.run()
                    # myoutput.pprint()
                    
                    # plt.plot(amps*a, parabula(myoutput.beta, amps*a))
                    
                    sfit = sFit('x61', Y[npts//2-delta_x_fit:npts//2+delta_x_fit+1], amps[npts//2-delta_x_fit:npts//2+delta_x_fit+1]*a, Y_err[npts//2-delta_x_fit:npts//2+delta_x_fit+1])
                    
                    if sfit.is_succeed:
                        fit_params, fit_cov = sfit._fit()
                        plt.plot(amps[npts//2-delta_x_fit:npts//2+delta_x_fit+1]*a, sfit.func(amps[npts//2-delta_x_fit:npts//2+delta_x_fit+1]*a, *fit_params))
                        nbar = -1/2-fit_params[0]
                        nbar_err = np.sqrt(fit_cov[0,0])
                        if a != 1: print('nbar = {}+-{}'.format(*round_value_by_error(nbar, nbar_err)))
                        if a != 1: print('nbar = {}+-{}'.format(nbar, nbar_err))
                    ret = amps*a, Y, X
                    
            elif fit_func == 'fock':
                              
                def fock_char_func(x,n):
                    return np.exp(-np.abs(x)**2/2)*laguerre(n)(np.abs(x)**2)
                
                def wrapper_fit_func(x, *args):
                    return sum_fock_char_func(x, args[0][0], args[0][1:])
                
                def sum_fock_char_func(x, a, unnorm_p):
                    # p = np.array(sqrt_p)**2
                    p = unnorm_p/np.sum(unnorm_p)
                    char_func = np.zeros(len(x))
                    for n,pn in enumerate(p):
                        char_func+= pn * fock_char_func(a*x, n)
                    return char_func
                
                
                p0 = [a_guess] + [1/max_fock]*(max_fock)
                fit, cov = curve_fit(lambda amps, *p0: wrapper_fit_func(amps, p0), amps, Y, p0=p0, 
                                     sigma = Y_err,
                                     bounds = [[-np.inf]+[0]*(max_fock), [np.inf]+[1]*(max_fock)]
                                                    )
                # fit_temp, cov_temp,info_dict, _, _ = curve_fit(fock_funcs[i-1], amps, Y, p0 = p0, full_output=True)
                corr_mat = np.zeros(cov.shape)
                for i,ci in enumerate(cov):
                    for j,cij in enumerate(ci):
                        corr_mat[i,j] = cov[i,j]/np.sqrt(cov[i,i])/np.sqrt(cov[j,j])
                self.corr_mat = corr_mat
                print('\n Condisp amplitude = {}+-{} 1/Volt'.format(*round_value_by_error(fit[0], np.sqrt(cov[0,0]))))
                print('\n Condisp amplitude = {}+-{} 1/Volt'.format(fit[0], np.sqrt(cov[0,0])))
                # plt.plot(amps, wrapper_fit_func(amps, p0)*np.max(Y))
                plt.plot(amps, wrapper_fit_func(amps, fit)*np.max(Y))
                ps = fit[1:]
                ps_err = np.sqrt(np.diag(cov)[1:])/np.sum(ps[1:])
                ps = ps/np.sum(ps)
                for i,p in enumerate(ps):
                    print('\n P{} = {}+-{}'.format(i,*round_value_by_error(p, ps_err[i])))
                for i,p in enumerate(ps):
                    print('\n P{} = {}+-{}'.format(i,p, ps_err[i]))
                
        if not is_calibrate_readout:
            plt.ylabel('{} [{}V]'.format(which_data, units_prefix))
        else:
            plt.ylabel(r'$\langle\sigma_z\rangle$')

        xlabel =  'Con. disp. amp. [V]' if a==1 else r'$|\alpha|$'
        plt.xlabel(xlabel)
        plt.grid()
        lgd = plt.legend(fontsize = 15)
        lgd.set_draggable(True)
        plt.tight_layout()
        
        if is_M1 and is_calibrate_readout and not is_fill_symmetric and is_plot_M1:
            plt.figure()
            plt.plot(amps, data_M1_sigma_z.mean(0).mean(1), 'o')
            plt.title('M1')
            plt.ylabel(r'$\langle\sigma_z\rangle$')
            plt.xlabel(r'Con. Disp. Amp. [V]')
            plt.tight_layout()
            
        return ret
        
    def load_con_disp_spec(self, N_avg = 1000, npts = 100, start = -1e6, stop = 1e6,
                          mm = None, qubit = None, ro = None, pulse = 'asym_pulse11',
                          is_calibrate_readout = False, is_active_reset_calib = None, N_avg_calib = 1000,
                          is_preactive_reset = None, is_M1 = False,
                          **kwargs):
        if qubit is None: qubit = self.main_qubit
        if mm is None: mm = self.main_mm
        if ro is None: ro = self.main_readout
        if is_active_reset_calib is None: is_active_reset_calib = self.is_active_reset
        if is_preactive_reset is None: is_preactive_reset = self.is_active_reset
        
        if N_avg_calib < N_avg: N_avg_calib = N_avg
        
        detunings = np.linspace(start, stop, npts)
        IFs = self.element_IF(mm) + detunings
        freqs = self.element_freq(mm)-self.element_IF(mm)+IFs
        self.results['con_disp_spec'] = {"mm": mm, "qubit": qubit, "npts": npts, "freqs":freqs, "detunings": detunings, "N_avg": N_avg,
                                         "N_avg_calib": N_avg_calib//N_avg*N_avg, "is_M1": is_M1,
                                        'is_calibrate_readout': is_calibrate_readout}
        
        run_time = N_avg * npts * 4 * (self.pulse_len(mm, pulse)*4 + self.wait_between_seq_mm)
        print('Run time is {}s'.format(round(run_time * 1e-9))) 
        
        with program() as self.con_disp_spec_prog:
            
            IF = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            n = declare(int)
            is_pi = declare(int)
            qubit_xy_phase = declare(fixed)
            
            with for_(n, 0, n<N_avg, n+1):
                if is_calibrate_readout:
                    self.calibrate_readout(qubit, ro, I, Q, is_active_reset = is_active_reset_calib, N_avg = int(N_avg_calib // N_avg), is_sb_cool = is_sb_cool and not is_active_reset_calib)
                with for_each_(IF, IFs.astype(int).tolist()):
                    with for_(qubit_xy_phase, 0, qubit_xy_phase<0.9, qubit_xy_phase+0.25):
                        if is_preactive_reset: 
                            self.perform_full_measurement(I,Q, ro = ro, is_save = False, is_active_reset = True)
                            align(qubit, mm, ro)
                        reset_frame(qubit, mm)
                        update_frequency(mm, IF)
                        play('pi2_pulse', qubit)
                        align(qubit, mm)
                        self.play_con_disp(qubit, mm)
                        align(qubit, mm)
                        frame_rotation_2pi(qubit_xy_phase, qubit)
                        play('pi2_pulse', qubit)
                        align(qubit, ro)
                        self.perform_full_measurement(I,Q, ro = ro, wait_time = self.wait_between_seq_mm,  **kwargs)
                        
        self.last_prog = self.con_disp_spec_prog
        
        
    def run_con_disp_spec(self, is_save_data = None, save_folder = None, **kwargs):
        results_dict = self.results['con_disp_spec']
        if not hasattr(self,'con_disp_spec_prog'): raise ValueError("You must run the load function first!")
        
        results_dict['I'], results_dict['Q'] = self.run_prog(self.con_disp_spec_prog, shape = (results_dict['N_avg'], results_dict['npts'], 4), **kwargs)
        
        if results_dict['is_calibrate_readout']:
            results_dict['I_pi'] = self.last_job.result_handles.get('I_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['Q_pi'] = self.last_job.result_handles.get('Q_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['I_nopi'] = self.last_job.result_handles.get('I_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['Q_nopi'] = self.last_job.result_handles.get('Q_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            
        if results_dict['is_M1']:
            results_dict['I_M1'] = self.last_job.result_handles.get('M1_I').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['npts'], 4))
            results_dict['Q_M1'] = self.last_job.result_handles.get('M1_Q').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['npts'], 4))
                
            
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'con_disp_spec', foldername = save_folder)
        return self.plot_con_disp_spec(**kwargs)
    
    
    def plot_con_disp_spec(self, fig_num = None, which_data = None, **kwargs):
        if which_data is None: which_data = self.which_data
        results_dict = self.results['con_disp_spec']
        data, err = self.process_data([results_dict['I'], results_dict['Q']],
                                      is_mean = not (results_dict['is_M1'] and results_dict['is_calibrate_readout']),
                                      is_calc_stat_error = not (results_dict['is_M1'] and results_dict['is_calibrate_readout']))
        
        if results_dict['is_calibrate_readout']:
            data_e, err_e = self.process_data([results_dict['I_pi'], results_dict['Q_pi']])
            data_g, err_g = self.process_data([results_dict['I_nopi'], results_dict['Q_nopi']])
            data, err = data_to_sigma_z(data=data, data_g=data_g, data_e=data_e, err=err, data_g_err=err_g, data_e_err=err_e, is_thresholding = results_dict['is_M1'])
            
            if results_dict['is_M1']:
                data_M1, _ = self.process_data([results_dict['I_M1'], results_dict['Q_M1']], is_mean = False)
                data_M1_sigma_z, _ = data_to_sigma_z(data=data_M1, data_g=data_g, data_e=data_e, is_thresholding = True)
                
                err =  sp.stats.sem(data * data_M1_sigma_z,0)
                data = -(data * data_M1_sigma_z).mean(0)
        else:
            data, units_prefix, scale_factor = scale_data_units(data)
            err = scale_factor * err
            
        detunings = results_dict['detunings']*1e-3
        
        X = (data[:,0]-data[:,2])/2
        Y = (data[:,1]-data[:,3])/2
        X_err = np.sqrt(err[:,0]**2+err[:,2]**2)/2
        Y_err = np.sqrt(err[:,1]**2+err[:,3]**2)/2
        fig_num = next_fig_num_by_name('Con. disp. spec.')
        plt.figure(fig_num)
        plt.errorbar(detunings, X, X_err, label = r'$X$')
        plt.errorbar(detunings, Y, Y_err, label = r'$Y$')
        if not results_dict['is_calibrate_readout']:
            plt.ylabel('{} [{}V]'.format(which_data, units_prefix))
        else:
            plt.ylabel(r'$\langle\sigma_z\rangle$')
        plt.xlabel('Detuning [kHz]')
        plt.grid()
        lgd = plt.legend(fontsize = 15)
        lgd.set_draggable(True)
        plt.tight_layout()
        
        
        # Y = data[:,0]
        # X = data[:,1]
        # Y_err = err[:,0]
        # X_err = -err[:,1]
        # fig_num = next_fig_num_by_name('Con. disp. spec.')
        
        # if results_dict['is_calibrate_readout']:
        #     phase = np.arctan2(Y,X) * 180 / np.pi
        #     phase_err = 180 / np.pi * np.sqrt((Y_err/X/(1+(Y/X)**2))**2+((X_err*Y/X**2/(1+(Y/X)**2))**2))
        #     purity = np.sqrt(X**2+Y**2)
        #     purity = purity/purity.max()
        #     purity_error = np.sqrt((X_err)**2+(Y_err)**2)
            
        #     fig,axs = plt.subplots(2,1, sharex=True)
        #     plt.sca(axs[0])
        #     plt.errorbar(detunings, phase, phase_err)
        #     plt.ylabel(r'Phase')
        #     plt.sca(axs[1])
            
        #     purity_err = purity*0
        #     plt.errorbar(detunings, purity, purity_err)
        #     plt.ylabel(r'Purity')
    
        # plt.figure(fig_num)
        # plt.errorbar(detunings, X, X_err, label = 'X')
        # plt.errorbar(detunings, Y, Y_err, label = 'Y')
        # if not results_dict['is_calibrate_readout']:
        #     plt.ylabel('{} [{}V]'.format(which_data, units_prefix))
        # else:
        #     plt.ylabel(r'$\langle\sigma_z\rangle$')
        # plt.xlabel('Detuning [kHz]')
        # plt.grid()
        # plt.legend()
        # plt.tight_layout()
        
        
    def load_cavity_ramsey(self, N_avg = 1000, npts = 100, 
                           max_seq_time = 30000,
                           min_seq_time = None,
                           detuning = 0, extra_qubit_phase = 45,
                           disp_amp_scale = 1, con_disp_amp_scale = 1,
                          mm = None, qubit = None, ro = None, pulse = 'disp',
                          is_calibrate_readout = False, is_active_reset_calib = False, N_avg_calib = 1000,
                          is_preactive_reset = False,
                          is_pi_pulse = False,
                          is_sideband_cool = None,
                          is_sb_cool = None,
                          is_active_reset = False,
                          is_measure_disp = False, measure_disp_pulse = 'disp',
                          is_M1 = False,
                          **kwargs):
        if qubit is None: qubit = self.main_qubit
        if mm is None: mm = self.main_mm
        if ro is None: ro = self.main_readout
        extra_qubit_phase_2pi = extra_qubit_phase/360 if not is_pi_pulse else (180 + extra_qubit_phase)/360
        if min_seq_time is None:
            if max_seq_time//npts < 16: raise ValueError(f"The minimum wait time is 16ns and your first step is {max_seq_time/npts}")
        else:
            if min_seq_time < 16: raise ValueError(f"The minimum wait time is 16ns and your first step is {max_seq_time/npts}")
            
        if min_seq_time is None:
                        times = np.arange(max_seq_time//npts, max_seq_time+1, max_seq_time//npts)
        else:
            if (max_seq_time-min_seq_time)//(npts-1) <4:
                print("Time step is too small. Seting the time step to 4ns")
                npts = (max_seq_time-min_seq_time)//4
            times = np.arange(min_seq_time, max_seq_time+1, (max_seq_time-min_seq_time)//(npts-1))
            
        if npts !=len(times): 
            npts = len(times)
            print(f'Changed <npts> to {npts} to round up the times.')
            
            
        phase_list = np.mod(times * detuning, 1)
        
        self.results['cavity_ramsey'] = {"mm": mm, "qubit": qubit, "npts": npts, "times":times, "detuning": detuning, "N_avg": N_avg, "N_avg_calib":N_avg_calib//N_avg*N_avg, 
                                        'is_calibrate_readout': is_calibrate_readout, "is_M1": is_M1, 'pulse_amp': self.pulse_amp(mm, pulse), 'is_pi_pulse': is_pi_pulse}
        
        if is_sb_cool:
            run_time = N_avg * npts * 4 * 2 * (self.pulse_len(mm, pulse)*2 + max_seq_time/2 + self.pulse_len(mm, 'constant_sideband_cooling_pulse'))
        else:
            if min_seq_time is None:
                run_time = N_avg * npts * 4 * 2 * (self.pulse_len(mm, pulse)*2 + max_seq_time/2 + self.wait_between_seq_mm)
            else:
                run_time = N_avg * npts * 4 * 2 * (self.pulse_len(mm, pulse)*2 + (max_seq_time+min_seq_time)/2 + self.wait_between_seq_mm)
        print('Run time is {}s'.format(round(run_time * 1e-9))) 
        
        with program() as self.cavity_ramsey_prog:
            
            phase = declare(fixed)
            ti = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            n = declare(int)
            qubit_xy_phase = declare(fixed)
            mm_iphase = declare(fixed)
            
            with for_(n, 0, n<N_avg, n+1):
                if is_calibrate_readout:
                    self.calibrate_readout(qubit, ro, I, Q, N_avg = N_avg_calib//N_avg, is_active_reset = is_active_reset_calib, is_sb_cool = is_sb_cool and not is_active_reset_calib, **kwargs)
                with for_each_((ti, phase), ((times//4).astype(int).tolist(), phase_list.tolist())):
                    with for_(mm_iphase, 0.0, mm_iphase<0.3, mm_iphase+0.25):
                        with for_(qubit_xy_phase, 0.0, qubit_xy_phase<0.9, qubit_xy_phase+0.25):
                            if is_preactive_reset: 
                                self.perform_full_measurement(I,Q, ro = ro, is_save = False, is_active_reset = True)
                                align(qubit, mm, ro)
                            reset_frame(mm)
                            reset_frame(qubit)
                            reset_frame(ro)
                            if is_pi_pulse:
                                play('pi_pulse', qubit)
                                align(qubit, mm, ro)

                            play('disp' * amp(disp_amp_scale), mm)
                            wait(ti, mm)
                            align(ro, mm, qubit)
                            if is_pi_pulse:
                                play('pi_pulse', qubit)
                                align(qubit, mm, ro)
                            if is_M1: 
                                align(ro, mm, qubit)
                                self.perform_full_measurement(I,Q, ro = ro, I_output_name = 'M1_I', Q_output_name = 'M1_Q', is_active_reset = False, is_loop_reset = False, is_wait = False)
                                if self.wait_after_reset >= 16: wait(self.wait_after_reset//4, ro)
                                align(qubit, mm, ro)
                            align(qubit, mm)
                            play('pi2_pulse', qubit)
                            if is_measure_disp: play(measure_disp_pulse, mm)
                            align(qubit, mm)
                            frame_rotation_2pi(phase + mm_iphase, mm)
                            self.play_con_disp(qubit, mm, amp_scale = con_disp_amp_scale)
                            align(qubit, mm)
                            frame_rotation_2pi(extra_qubit_phase_2pi + qubit_xy_phase, qubit)
                            play('pi2_pulse', qubit)
                            align(qubit, ro)
                            self.perform_full_measurement(I,Q, ro = ro, is_active_reset = is_active_reset, wait_time = self.wait_between_seq_mm, is_sb_cool = is_sb_cool,  **kwargs)
                    
        self.last_prog = self.cavity_ramsey_prog
        
        
    def run_cavity_ramsey(self, is_save_data = None, save_folder = None, **kwargs):
        results_dict = self.results['cavity_ramsey']
        if not hasattr(self,'cavity_ramsey_prog'): raise ValueError("You must run the load function first!")
        
        results_dict['I'], results_dict['Q'] = self.run_prog(self.cavity_ramsey_prog, shape = (results_dict['N_avg'], results_dict['npts'], 2,4), **kwargs)
        
        if results_dict['is_calibrate_readout']:
            results_dict['I_pi'] = self.last_job.result_handles.get('I_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['Q_pi'] = self.last_job.result_handles.get('Q_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['I_nopi'] = self.last_job.result_handles.get('I_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['Q_nopi'] = self.last_job.result_handles.get('Q_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            
        if results_dict['is_M1']:
            results_dict['I_M1'] = self.last_job.result_handles.get('M1_I').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['npts'], 2,4))
            results_dict['Q_M1'] = self.last_job.result_handles.get('M1_Q').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['npts'], 2,4))
            
            
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'cavity_ramsey', foldername = save_folder)
        return self.plot_cavity_ramsey(**kwargs)
    
    
    def plot_cavity_ramsey(self, fig_num = None, which_data = None, is_plot_raw = False, is_plot_fft = False,
                           is_geo_phase_corr = None, is_char_func_phase = True, is_calibrate_readout = None, is_M1 = None, is_unwrap_phase=True,
                           **kwargs):
        if which_data is None: which_data = self.which_data
        if is_geo_phase_corr is None: is_geo_phase_corr = self.is_geo_phase_corr
        results_dict = self.results['cavity_ramsey']
        if is_calibrate_readout is None: is_calibrate_readout = results_dict['is_calibrate_readout']
        mm = results_dict['mm']
        if is_M1 is None: is_M1 = results_dict['is_M1']
        raw_data, raw_err = self.process_data([results_dict['I'], results_dict['Q']],
                                              is_mean = not (is_M1 and is_calibrate_readout),
                                              is_calc_stat_error = not (is_M1 and is_calibrate_readout))
        if is_calibrate_readout:
            data_e, err_e = self.process_data([results_dict['I_pi'], results_dict['Q_pi']])
            data_g, err_g = self.process_data([results_dict['I_nopi'], results_dict['Q_nopi']])
            data, err = data_to_sigma_z(data=raw_data, data_g=data_g, data_e=data_e, err=raw_err, data_g_err=err_g, data_e_err=err_e, is_thresholding = is_M1)
            
            if is_M1:
                data_M1, _ = self.process_data([results_dict['I_M1'], results_dict['Q_M1']], is_mean = False)
                data_M1_sigma_z, _ = data_to_sigma_z(data=data_M1, data_g=data_g, data_e=data_e, is_thresholding = True)
                
                # err =  sp.stats.sem(data * data_M1_sigma_z,0)
                # data = -(data * data_M1_sigma_z).mean(0)
                
                    
                inds = np.where(data_M1_sigma_z==1) #remove excited. Should always be in ground.
                data_post_M1 = data.copy()
                data_post_M1[inds] = np.nan
                data = np.nanmean(data_post_M1, 0)
                NN = np.sum(~np.isnan(data_post_M1),0)
                err =  np.nanstd(data_post_M1, 0)/np.sqrt(NN)
                
            Re_C_beta = (data[:,0,0]-data[:,0,2])/2
            Im_C_beta = (data[:,0,1]-data[:,0,3])/2
            Re_C_ibeta = (data[:,1,0]-data[:,1,2])/2
            Im_C_ibeta = (data[:,1,1]-data[:,1,3])/2
            if is_geo_phase_corr: 
                Re_C_beta, Im_C_beta = self.geometric_phase_correction(results_dict['pulse_amp'], Re_C_beta, Im_C_beta, mm)
                Re_C_ibeta, Im_C_ibeta = self.geometric_phase_correction(results_dict['pulse_amp'], Re_C_ibeta, Im_C_ibeta, mm)
            
        else:
            Re_C_beta, factor = normalize_data(raw_data[:,0,0]-raw_data[:,0,2])
            raw_err[:,0,0] = raw_err[:,0,0] * factor
            raw_err[:,0,2] = raw_err[:,0,2] * factor
            Im_C_beta, factor = normalize_data(raw_data[:,0,1]-raw_data[:,0,3])
            raw_err[:,0,1] = raw_err[:,0,1] * factor
            raw_err[:,0,3] = raw_err[:,0,3] * factor
            Re_C_ibeta, factor = normalize_data(raw_data[:,1,0]-raw_data[:,1,2])
            raw_err[:,1,0] = raw_err[:,1,0] * factor
            raw_err[:,1,2] = raw_err[:,1,2] * factor
            Im_C_ibeta, factor = normalize_data(raw_data[:,1,1]-raw_data[:,1,3])
            raw_err[:,1,1] = raw_err[:,1,1] * factor
            raw_err[:,1,3] = raw_err[:,1,3] * factor
            err = raw_err
            
        Re_C_beta_err = np.sqrt(err[:,0,0]**2+err[:,0,2]**2)/2
        Im_C_beta_err = np.sqrt(err[:,0,1]**2+err[:,0,3]**2)/2
        Re_C_ibeta_err = np.sqrt(err[:,1,0]**2+err[:,1,2]**2)/2
        Im_C_ibeta_err = np.sqrt(err[:,1,1]**2+err[:,1,3]**2)/2

        times = results_dict['times']
        scaled_times, time_units_prefix, times_scale_factor = scale_data_units(times*1e-9)
        
        if is_M1 and is_calibrate_readout:
            plt.figure()
            for i in range(data_M1_sigma_z.mean(0).shape[-1]):
                plt.plot(scaled_times, data_M1_sigma_z.mean(0)[:,0,i])
            plt.xlabel(r'Time $[\mu s]$')
            plt.title('M1')
            plt.tight_layout()
                
        if is_plot_raw:
            if is_M1 and is_calibrate_readout: raw_data = raw_data.mean(0)
            plt.figure()
            # plt.plot(times, (raw_data[:,0,0]-raw_data[:,0,2])/2)
            # plt.plot(times, (raw_data[:,0,1]-raw_data[:,0,3])/2)
            # plt.plot(times, (raw_data[:,1,0]-raw_data[:,1,2])/2)
            # plt.plot(times, (raw_data[:,1,1]-raw_data[:,1,3])/2)
            
            print('borito')
            plt.plot(times, raw_data[:,0,0])
            plt.plot(times, raw_data[:,0,1])
            plt.plot(times, raw_data[:,0,2])
            plt.plot(times, raw_data[:,0,3])
                     
            plt.plot(times, raw_data[:,1,0])
            plt.plot(times, raw_data[:,1,1])
            plt.plot(times, raw_data[:,1,2])
            plt.plot(times, raw_data[:,1,3])
                     
            if is_calibrate_readout:
                xl = plt.xlim()
                plt.plot(xl, [data_g]*2, '--k')
                plt.plot(xl, [data_e]*2, '--k')
                xl = plt.xlim(xl)
                
                if is_M1: 
                    f,axs = plt.subplots(2,1)
                    plt.sca(axs[0])
                    plt.plot(data_M1.flatten())
                    xl = plt.xlim()
                    plt.plot(xl, [data_g]*2, '--k')
                    plt.plot(xl, [data_e]*2, '--k')
                    plt.plot(xl, [(data_g+data_e)/2]*2, '--k')
                    xl = plt.xlim(xl)
                    plt.sca(axs[1])
                    plt.plot(data_M1_sigma_z.flatten())
        
        if is_char_func_phase:
            if is_unwrap_phase:
                phase_beta = np.unwrap(np.arctan2(Im_C_beta, Re_C_beta), discont = np.pi)
                phase_ibeta = np.unwrap(np.arctan2(Im_C_ibeta, Re_C_ibeta), discont = np.pi)
            else:
                phase_beta = np.arctan2(Im_C_beta, Re_C_beta)
                phase_ibeta = np.arctan2(Im_C_ibeta, Re_C_ibeta)
            phase_beta = phase_beta - phase_beta.mean()
            phase_ibeta = phase_ibeta - phase_ibeta.mean()
            phase_beta_err = np.sqrt((Im_C_beta_err/Re_C_beta/(1+(Im_C_beta/Re_C_beta)**2))**2+((Re_C_beta_err*Im_C_beta/Re_C_beta**2/(1+(Im_C_beta/Re_C_beta)**2))**2))
            phase_ibeta_err = np.sqrt((Im_C_ibeta_err/Re_C_ibeta/(1+(Im_C_ibeta/Re_C_ibeta)**2))**2+((Re_C_ibeta_err*Im_C_ibeta/Re_C_ibeta**2/(1+(Im_C_ibeta/Re_C_ibeta)**2))**2))
            
            plt.figure(next_fig_num_by_name('Cavity ramsey'))
            plt.errorbar(scaled_times, phase_beta, phase_beta_err, fmt = 'ob', mfc = (0,0,0,0), mec = (0,0,1,1), capsize = 5,  label = r'$\mathrm{Arg}\mathcal{C}(\beta)$')
            plt.errorbar(scaled_times, phase_ibeta, phase_ibeta_err, fmt = 'or', mfc = (0,0,0,0), mec = (1,0,0,1), capsize = 5, label = r'$\mathrm{Arg}\mathcal{C}(\mathrm{i}\beta)$')
            
            times_for_plot = np.linspace(times[0], times[-1], 1001)
            sfit = sFit('ExpCosiSin', phase_beta, times, itrace = phase_ibeta)
            if sfit.is_succeed:
                fit, cov = sfit.get_fit_results()
                err = np.sqrt(np.diag(cov))
                Amp, Freq, Gamma, Delta, Offset = fit
                Amp_err, Freq_err, Gamma_err, Delta_err, Offset_err = err
                plt.plot(times_for_plot*times_scale_factor*1e-9, sfit.ExpCos(times_for_plot, *sfit.fit_results),'-', color = (0, 0, 1, 0.5), label = r'$\mathrm{Arg}\mathcal{C}(\beta)$')
                plt.plot(times_for_plot*times_scale_factor*1e-9, sfit.ExpCos(times_for_plot, Amp, Freq, Gamma, Delta-np.pi/2, Offset),'-', color = (1, 0, 0, 0.5), label = r'$\mathrm{Arg}\mathcal{C}(\beta)$')
            
                text = r'Frequency = ${} \pm {}$ kHz' '\n' r' Decay time = ${} \pm {}$ ms' '\n' r'Detuning = {} kHz'.format(*round_value_by_error(Freq*1e6, Freq_err*1e6),
                                                                                                                        *round_value_by_error(1/Gamma*1e-6, Gamma_err/Gamma**2*1e-6),
                                                                                                                        results_dict['detuning']*1e6)
                ann = plt.annotate(text, xy = (0.5, 0.9), xycoords = 'axes fraction', fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
                ann.draggable()
            plt.xlabel(f'Time [{time_units_prefix}s]')
            plt.ylabel(r'$\mathrm{Arg}(C)$')
            plt.title(f'Cavity ramsey ({results_dict["mm"]})')
            plt.tight_layout()
            if is_plot_fft:
                plot_fft(times, phase_beta+1j*phase_ibeta)
        else:
            plt.figure()
            plt.errorbar(scaled_times, Re_C_beta, Re_C_beta_err, fmt = 'o', label = r'$\mathrm{Re}(C(\beta))$')
            plt.errorbar(scaled_times, Im_C_beta, Im_C_beta_err, fmt = 'o', label = r'$\mathrm{Im}(C(\beta))$')
            plt.errorbar(scaled_times, Re_C_ibeta, Re_C_ibeta_err, fmt = 'o', label = r'$\mathrm{Re}(C(\mathrm{i}\beta))$')
            plt.errorbar(scaled_times, Im_C_ibeta, Im_C_ibeta_err, fmt = 'o', label = r'$\mathrm{Im}(C(\mathrm{i}\beta))$')
            lgd = plt.legend(fontsize = 10)
            lgd.set_draggable(True)
            plt.xlabel(f'Time [{time_units_prefix}s]')
            plt.ylabel(r'$\langle\sigma_z\rangle$')
            plt.tight_layout()
        
        
    def load_cavity_T1(self, N_avg = 1000, npts = 100, 
                           max_seq_time = 30000,
                           min_seq_time = None,
                           detuning = 0,
                          mm = None, ro = None, qubit = None,
                          disp_pulse = 'disp', con_pi_pulse = 'con_pi_pulse',
                          is_calibrate_readout = False,
                          is_measure_disp = False, measure_disp_pulse = 'disp',
                          is_preactive_reset = True,
                          **kwargs):
        if mm is None: mm = self.main_mm
        if ro is None: ro = self.main_readout
        if qubit is None: qubit = self.main_qubit
        
        if min_seq_time is None:
            if max_seq_time//npts < 16: raise ValueError(f"The minimum wait time is 16ns and your first step is {max_seq_time/npts}")
        else:
            if min_seq_time < 16: raise ValueError(f"The minimum wait time is 16ns and your first step is {max_seq_time/npts}")
            
        if min_seq_time is None:
            times = np.arange(max_seq_time//npts, max_seq_time+1, max_seq_time//npts)
        else:
            if (max_seq_time-min_seq_time)//(npts-1) <4:
                print("Time step is too small. Seting the time step to 4ns")
                npts = (max_seq_time-min_seq_time)//4
            times = np.arange(min_seq_time, max_seq_time+1, (max_seq_time-min_seq_time)//(npts-1))
        if npts !=len(times): 
            npts = len(times)
            print(f'Changed <npts> to {npts} to round up the times.')
            
        self.results['cavity_T1'] = {"mm": mm,  "npts": npts, "times":times, "N_avg": N_avg}
        
        run_time = N_avg * npts * (self.pulse_len(mm, disp_pulse) + max_seq_time/2 + self.wait_between_seq_mm)
        print('Run time is {}s'.format(round(run_time * 1e-9))) 
        
        with program() as self.cavity_T1_prog:
            
            ti = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            n = declare(int)
            
            with for_(n, 0, n<N_avg, n+1):
                with for_each_(ti, (times//4).astype(int).tolist()):
                    if is_preactive_reset: 
                        self.perform_full_measurement(I,Q, ro = ro, is_save = False, is_active_reset = True)
                        align(qubit, mm, ro)
                    
                    play(disp_pulse, mm)
                    wait(ti, mm)
                    if is_measure_disp: play(measure_disp_pulse, mm)
                    align(mm, qubit)
                    play(con_pi_pulse, qubit)
                    align(qubit, ro)
                    self.perform_full_measurement(I,Q, ro_element = ro, is_active_reset = False, wait_time = self.wait_between_seq_mm, )
                    
        self.last_prog = self.cavity_T1_prog
        
    def run_cavity_T1(self, is_save_data = None, save_folder = None, **kwargs):
        results_dict = self.results['cavity_T1']
        if not hasattr(self,'cavity_T1_prog'): raise ValueError("You must run the load function first!")
        
        results_dict['I'], results_dict['Q'] = self.run_prog(self.cavity_T1_prog, shape = (results_dict['N_avg'], results_dict['npts']), **kwargs)
        
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'cavity_T1', foldername = save_folder)
        return self.plot_cavity_T1(**kwargs)
    
    
    def plot_cavity_T1(self, fig_num = None, which_data = None, **kwargs):
        if which_data is None: which_data = self.which_data
        results_dict = self.results['cavity_T1']
        data, err = self.process_data([results_dict['I'], results_dict['Q']])
        times = results_dict['times']
        self.fit_and_plot( 'Exp(Exp)' , [data, err], times, title_str = "Cavity T1 ({})".format(results_dict['mm']), **kwargs )
        plt.tight_layout()
    
    def load_char_func(self, N_avg = 1000, 
                        npts_x = 100, amp_scale_start_x = 0, amp_scale_stop_x = 1,
                        npts_y = 100, amp_scale_start_y = 0, amp_scale_stop_y = 1,
                        mm = None, qubit = None, ro = None, is_ECD = None,
                        is_calibrate_readout = False, is_preactive_reset = True, N_avg_calib = 1000, is_active_reset_calib = None,
                        prepare = None, prepare_dict = {},
                        is_M1 = False, is_sb_cool = None,
                        **kwargs): 
        if qubit is None: qubit = self.main_qubit
        if mm is None: mm = self.main_mm
        if ro is None: ro = self.main_readout
        if is_sb_cool is None: is_sb_cool = self.is_sb_cool
        if is_active_reset_calib is None: is_active_reset_calib = self.is_active_reset
        if is_ECD is None: is_ECD = self.is_ECD
        if is_ECD: pulse = 'disp_ECD'
        else: pulse = 'asym_pulse11'
        
        if N_avg_calib < N_avg: N_avg_calib = N_avg
        
        amp_scale_x_list = np.linspace(amp_scale_start_x, amp_scale_stop_x, npts_x)
        amps_x = np.abs(self.pulse_amp(mm, pulse)) * amp_scale_x_list
        amp_scale_y_list = np.linspace(amp_scale_start_y, amp_scale_stop_y, npts_y)
        amps_y = np.abs(self.pulse_amp(mm, pulse)) * amp_scale_y_list
        
        # if any(np.sqrt(amps_x**2 + amps_y**2) > 0.5): raise ValueError(f"Amplitude too large amp = sqrt(amp_x^2+amp_y^2={np.max(np.sqrt(amps_x**2+amps_y**2))}). Overflow!")
        self.results['char_func'] = {"mm": mm, "qubit": qubit, "ro": ro,
                                     "npts_x": npts_x, "npts_y": npts_y, "amps_x": amps_x, "amps_y": amps_y,
                                     "N_avg": N_avg, "N_avg_calib": N_avg_calib//N_avg*N_avg,
                                     'is_calibrate_readout': is_calibrate_readout, "is_M1": is_M1,
                                     'is_preactive_reset': is_preactive_reset, "is_active_reset_calib": is_active_reset_calib}
        
        
        if is_sb_cool: run_time = N_avg * npts_x * npts_y * 4 * (self.pulse_len(mm, pulse) * 2 + self.pulse_len('ro', 'constant_sideband_cooling_pulse'))
        else: run_time = N_avg * npts_x * npts_y * 4 * (self.pulse_len(mm, pulse) * 2 + self.wait_between_seq_mm)
        print('Run time is {}s'.format(round(run_time * 1e-9))) 
        
        if not self.is_loop_amp_scale:
            self._load_py_loop_char_func_prog(prepare, prepare_dict, is_sb_cool)
        else:
            with program() as self.char_func_prog:
                
                amp_scale_x = declare(fixed)
                amp_scale_y = declare(fixed)
                I = declare(fixed)
                Q = declare(fixed)
                n = declare(int)
                qubit_xy_phase = declare(fixed)
                
                with for_(n, 0, n<N_avg, n+1):
                    if is_calibrate_readout:
                        self.calibrate_readout(qubit, ro, I, Q, N_avg = N_avg_calib//N_avg,  is_active_reset = is_active_reset_calib, is_sb_cool = is_sb_cool and not is_active_reset_calib, **kwargs)
                    with for_each_(amp_scale_y, amp_scale_y_list.tolist()):
                        with for_each_(amp_scale_x, amp_scale_x_list.tolist()):
                            with for_(qubit_xy_phase, 0, qubit_xy_phase<0.9, qubit_xy_phase+0.25):
                                if is_preactive_reset: 
                                    self.perform_full_measurement(I,Q, ro = ro, is_save = False, is_active_reset = True)
                                    align(qubit, ro, mm)
                                reset_frame(qubit)
                                reset_frame(mm)
                                if prepare is not None:
                                    prepare(**prepare_dict)
                                if is_M1: 
                                    self.perform_full_measurement(I,Q, ro = ro, 
                                                                  I_output_name = 'M1_I', Q_output_name = 'M1_Q',
                                                                  is_active_reset = False, 
                                                                  is_loop_reset = False, is_wait = False)
                                    if self.wait_after_reset >= 16: wait(self.wait_after_reset//4, ro)
                                    align(qubit, mm, ro)
                                    
                                play('pi2_pulse', qubit)
                                align(qubit, mm)
                                self.play_con_disp(qubit, mm, amp_scale = [amp_scale_x, amp_scale_x, amp_scale_y, amp_scale_y])
                                align(qubit, mm)
                                frame_rotation_2pi(qubit_xy_phase, qubit)
                                play('pi2_pulse', qubit)
                                align(qubit, ro)
                                
                                self.perform_full_measurement(I,Q, ro = ro, wait_time = self.wait_between_seq_mm, is_active_reset = False, is_sb_cool = is_sb_cool, **kwargs)
                            
        self.last_prog = self.char_func_prog
            
    def run_char_func(self, is_save_data = None, save_folder = None, **kwargs):
        results_dict = self.results['char_func']
        if not hasattr(self,'char_func_prog'): raise ValueError("You must run the load function first!")
        results_shape = (results_dict['N_avg'], results_dict['npts_y'], results_dict['npts_x'], 4)
        if self.is_loop_amp_scale:
            results_dict['I'], results_dict['Q'] = self.run_prog(self.char_func_prog, shape = (results_dict['N_avg'], results_dict['npts_y'], results_dict['npts_x'], 4), **kwargs)
            
            if results_dict['is_calibrate_readout']:
                results_dict['I_pi'] = self.last_job.result_handles.get('I_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['Q_pi'] = self.last_job.result_handles.get('Q_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['I_nopi'] = self.last_job.result_handles.get('I_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['Q_nopi'] = self.last_job.result_handles.get('Q_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                
            if results_dict['is_M1']:
                results_dict['I_M1'] = self.last_job.result_handles.get('M1_I').fetch_all()['value'].reshape(results_shape)
                results_dict['Q_M1'] = self.last_job.result_handles.get('M1_Q').fetch_all()['value'].reshape(results_shape)
        else:
            mm = results_dict['mm']
            prog_id = self.qm.compile(self.char_func_prog)
            amp_scale_x_list = results_dict['amps_x']/self.pulse_amp(mm, 'asym_pulse1')
            amp_scale_y_list = results_dict['amps_y']/self.pulse_amp(mm, 'asym_pulse1')
            
            if results_dict['is_calibrate_readout']:
                self.load_pinopi(N_avg = results_dict['N_avg_calib'], is_active_reset = results_dict['is_active_reset_calib'], meas_type = 'full')
                self.run_pinopi(plot=False, is_save_data = False)
                results_dict['I_pi'] = self.last_job.result_handles.get('I_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['Q_pi'] = self.last_job.result_handles.get('Q_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['I_nopi'] = self.last_job.result_handles.get('I_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                results_dict['Q_nopi'] = self.last_job.result_handles.get('Q_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
                
            results_dict['I'] = np.zeros(results_shape)
            results_dict['Q'] = np.zeros(results_shape)
            if results_dict['is_M1']:
                results_dict['I_M1'] = np.zeros(results_shape)
                results_dict['Q_M1'] = np.zeros(results_shape)
                
            for i,amp_scale_x in tqdm(enumerate(amp_scale_x_list)):
                for j,amp_scale_y in enumerate(amp_scale_y_list):
                
                    results_dict['I'][:,i,j,:], results_dict['Q'][:,i,j,:] = self.run_prog(prog_id, shape = (results_dict['N_avg'], 4), 
                                             overrides_dict={'waveforms': {
                                                           'wf_asym_pulse1_mm2_I': (self.config['waveforms']['wf_asym_pulse1_mm2_I']['samples']*amp_scale_x).tolist(),
                                                           'wf_asym_pulse1_mm2_Q': (self.config['waveforms']['wf_asym_pulse1_mm2_I']['samples']*amp_scale_y).tolist(),
                                                           'wf_asym_pulse2_mm2_I': (self.config['waveforms']['wf_asym_pulse2_mm2_I']['samples']*amp_scale_x).tolist(),
                                                           'wf_asym_pulse2_mm2_Q': (self.config['waveforms']['wf_asym_pulse2_mm2_I']['samples']*amp_scale_y).tolist()}})
                    if results_dict['is_M1']:
                        results_dict['I_M1'][:,i,j,:] = self.last_job.result_handles.get('M1_I').fetch_all()['value'].reshape((results_dict['N_avg'], 4))
                        results_dict['I_M1'][:,i,j,:] = self.last_job.result_handles.get('M1_I').fetch_all()['value'].reshape((results_dict['N_avg'], 4))
                
        if is_save_data is None: is_save_data = self.is_save_data
        if save_folder is None: save_folder = self.save_folder
        if is_save_data: self.pickle_save(results_dict, 'char_func', foldername = save_folder)
        return self.plot_char_func(**kwargs)
    
    
    def plot_char_func(self, fig_num = None, which_data = None, is_geo_phase_corr = None, is_M1 = None, is_fill_symmetric = None,
                       is_save_char_func = False, post_selection = None, a = 1, **kwargs):
        if which_data is None: which_data = self.which_data
        results_dict = self.results['char_func']
        if is_fill_symmetric is None: is_fill_symmetric = np.abs(np.sum(results_dict['amps_x']))>1e-6
        if is_M1 is None: is_M1 = results_dict['is_M1']
        if is_geo_phase_corr is None: is_geo_phase_corr = self.is_geo_phase_corr
        mm = results_dict['mm']
        data, err = self.process_data([results_dict['I'], results_dict['Q']], is_mean = not (is_M1 and results_dict['is_calibrate_readout']))
        is_calibrate_readout = results_dict['is_calibrate_readout']
        if is_calibrate_readout:
                
            data_e, err_e = self.process_data([results_dict['I_pi'], results_dict['Q_pi']])
            data_g, err_g = self.process_data([results_dict['I_nopi'], results_dict['Q_nopi']])
            _,_,means = self.find_threshold(np.append(results_dict['I_pi'],results_dict['I_nopi']), np.append(results_dict['Q_pi'],results_dict['Q_nopi']), n_means = 2, is_fit_circle = False, is_ascending = data_g.mean()<data_e.mean())
            data, err = data_to_sigma_z(data=data, data_g=means[0], data_e=means[1], err=err, data_g_err=0, data_e_err=0, is_thresholding = is_M1 and results_dict['is_calibrate_readout'])
            if is_M1:
                
                    
                data_M1, _ = self.process_data([results_dict['I_M1'], results_dict['Q_M1']], is_mean = False)
                data_M1_sigma_z, _ = data_to_sigma_z(data=data_M1, data_g=data_g, data_e=data_e, is_thresholding = True)
                
                if post_selection is not None:
                    if post_selection == 'g': 
                        inds = np.where(data_M1_sigma_z==1) #remove excited
                        data =  -data
                    elif post_selection == 'e': 
                        inds = np.where(data_M1_sigma_z==-1) #remove ground.
                        data =  data
                        
                    data_post_M1 = data.copy()
                    data_post_M1[inds] = np.nan
                    data = np.nanmean(data_post_M1, 0)
                    NN = np.sum(~np.isnan(data_post_M1),0)
                    err =  np.nanstd(data_post_M1, 0)/np.sqrt(NN)
                else:
                    err =  sp.stats.sem(data * data_M1_sigma_z,0)
                    data = (data * data_M1_sigma_z).mean(0)
                            
        else:
            data, units_prefix, scale_factor = scale_data_units(data)
            err = scale_factor * err
        
        
        if is_fill_symmetric:
            amps_x = np.append(-np.flip(results_dict['amps_x'][1:]),results_dict['amps_x'])
            amps_y = results_dict['amps_y']
            Y = np.append(np.flip((data[:,1:,0]-data[:,1:,2])/2), (data[:,:,0]-data[:,:,2])/2, axis = 1)
            X = np.append(-np.flip((data[:,1:,1]-data[:,1:,3])/2), (data[:,:,1]-data[:,:,3])/2, axis = 1)
            Y_err = np.append(np.flip(np.sqrt(err[:,1:,0]**2+err[:,1:,2]**2)/2), np.sqrt(err[:,:,0]**2+err[:,:,2]**2)/2, axis = 1)
            X_err = np.append(np.flip(np.sqrt(err[:,1:,1]**2+err[:,1:,3]**2)/2), np.sqrt(err[:,:,1]**2+err[:,:,2]**3)/2, axis = 1)
        else:
            amps_x = results_dict['amps_x']
            amps_y = results_dict['amps_y']
            Y = (data[:,:,0]-data[:,:,2])/2
            X = (data[:,:,1]-data[:,:,3])/2
            Y_err = np.sqrt(err[:,:,0]**2+err[:,:,2]**2)/2
            X_err = np.sqrt(err[:,:,1]**2+err[:,:,3]**2)/2
        amps_x_grid, amps_y_grid = np.meshgrid(amps_x,amps_y)
        if is_geo_phase_corr: X,Y = self.geometric_phase_correction(np.sqrt(np.abs(amps_x_grid**2+amps_y_grid**2)), X, Y, mm)
        
        if is_calibrate_readout: Y = -Y
        
        fig_num = next_fig_num_by_name('Char func')
        
        if is_calibrate_readout: vmax, vmin = 1,-1
        else: vmax, vmin = np.max(np.abs(np.array([X,Y]).flatten())), -np.max(np.abs(np.array([X,Y]).flatten()))
        
        if is_save_char_func: self.pickle_save({"Re_C": Y, "Im_C": X, "x": amps_x, "y": amps_y}, meas_name = "Char func coherent")
            
        
        xlabel = 'Re[Con. disp. amp.] [V]' if a == 1 else r'$\mathrm{Re}(\alpha)$'
        ylabel = 'Im[Con. disp. amp.] [V]' if a == 1 else r'$\mathrm{Im}(\alpha)$'
        
        fig,axs = plt.subplots(1,2, sharey = True, num = fig_num)
        plot_2D(Y, amps_x*a, amps_y*a, vmax = vmax, vmin = vmin, ax = axs[0], 
                xlabel = xlabel, ylabel = ylabel, 
                is_colorbar = False, is_equal_aspect_ratio = True)
        plot_2D(X, amps_x*a, amps_y*a, vmax = vmax, vmin = vmin, ax = axs[1], 
                xlabel = xlabel, 
                is_colorbar = False, is_equal_aspect_ratio = True)
        
        
        axs[0].set_title(r'$\mathrm{Re}(\mathcal{C}(\alpha))$')
        axs[1].set_title(r'$\mathrm{Im}(\mathcal{C}(\alpha))$')
        plt.tight_layout()
        cmap = matplotlib.cm.get_cmap('seismic')
        normalizer = matplotlib.colors.Normalize(vmin, vmax)
        im = matplotlib.cm.ScalarMappable(norm=normalizer, cmap  = cmap)
        plt.colorbar(mappable = im, ax=axs, location='bottom')  
        # plt.xlabel(r'$\mathrm{Re}\left[\mathcal{C}[\beta]\right]$', fontsize = 15)
        
        if is_M1 and is_calibrate_readout:
            fig,axs = plt.subplots(1,2, sharey = True)
            plot_2D(data_M1_sigma_z.mean(0)[:,:,0], results_dict['amps_x'], results_dict['amps_y'], vmax = vmax, vmin = vmin, ax = axs[0], 
                    xlabel = 'Re[Con. disp. amp.] [V]', ylabel = 'Im[Con. disp. amp.] [V]', 
                    is_colorbar = False)
            plot_2D(data_M1_sigma_z.mean(0)[:,:,1], results_dict['amps_x'], results_dict['amps_y'], vmax = vmax, vmin = vmin, ax = axs[1], 
                    xlabel = 'Re[Con. disp. amp.] [V]', 
                    is_colorbar = False)
            
            axs[0].set_title(r'$\mathrm{Re}[\mathcal{C}]$')
            axs[1].set_title(r'$\mathrm{Im}[\mathcal{C}]$')
            cmap = matplotlib.cm.get_cmap('seismic')
            normalizer = matplotlib.colors.Normalize(vmin, vmax)
            im = matplotlib.cm.ScalarMappable(norm=normalizer, cmap  = cmap)
            plt.colorbar(mappable = im, ax=axs, location='bottom')  
            plt.suptitle('M1')
        
        return Y, X, amps_x*a, amps_y*a, Y_err, X_err
            
    def load_mm_pinopi_spec(self, N_avg = 1000, npts = 101,
                         freq_center = None, span = None, start = None, stop = None,
                         pulse = 'disp',
                         qubit = None, mm = None, ro_element = None,
                         is_active_reset = False,
                         is_ramp = False,
                         is_scnd_pi_pulse = False,
                         **kwargs):
        if qubit is None: qubit = self.main_qubit
        if mm is None: mm = self.main_mm
        if ro_element is None: ro_element = self.main_readout
        freq0 = self.element_IF(qubit)
        if freq_center is None and span is None and start is not None and stop is not None:
            freqs = freq0 + np.linspace(start, stop, npts)
        elif freq_center is not None and span is not None and start is None and stop is None:
            freqs = freq0 + np.linspace(freq_center - span/2, freq_center + span/2, npts)
        else:
            raise ValueError("You must pass either freq_center and span or start and stop.\n Passed: freq_center={freq_center}, span = {span}, start = {start}, stop ={stop}")
            
        self.results['mm_pinopi_spec'] = {'freqs': freqs,
                                          'N_avg': N_avg,
                                          'npts': npts,
                                          'qubit': qubit,
                                          'ro_element': ro_element}
        
        run_time = 2*N_avg*npts*(self.pulse_len(ro_element, 'ro_pulse')+self.wait_between_seq_mm)
        print('Run time is {}s'.format(round(run_time * 1e-9)))
        
        with program() as self.mm_pinopi_spec_prog:
            
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            freq = declare(int)
                
            with for_(n, 0, n < N_avg, n + 1):
                with for_each_(freq , freqs.astype(int).tolist()): #IF_list.astype(np.int64).tolist()
                    
                    update_frequency(mm,freq)
                    align(qubit, ro_element)    
                    if is_ramp:
                        play('ramp_up', mm)
                        play(pulse, mm)
                        play('ramp_down', mm)
                    else:
                        play(pulse, mm)
                    align(qubit, ro_element, mm)             
                    self.perform_full_measurement(I, Q,  I_output_name = 'I', Q_output_name = 'Q', ro_element = ro_element, is_active_reset = is_active_reset, wait_time = self.wait_between_seq_mm, )
                    
                    align(qubit, ro_element, mm)             
                    
                    play('pi_pulse', qubit)
                    align(qubit, mm)    
                    if is_ramp:
                        play('ramp_up', mm)
                        play(pulse, mm)
                        play('ramp_down', mm)
                    else:
                        play(pulse, mm)
                    align(qubit, mm)    
                    if is_scnd_pi_pulse: play('pi_pulse', qubit)
                    align(qubit, ro_element, mm)             
                    self.perform_full_measurement(I, Q,  I_output_name = 'I', Q_output_name = 'Q', ro_element = ro_element, is_active_reset = is_active_reset, wait_time = self.wait_between_seq_mm, )
        
        self.last_prog = self.mm_pinopi_spec_prog
        
    def run_mm_pinopi_spec(self,**kwargs):
    
        if not hasattr(self,'mm_pinopi_spec_prog'): raise ValueError("You must run the load function first!")
        results_dict = self.results['mm_pinopi_spec']
        results_dict["I"], results_dict["Q"] = self.run_prog(self.mm_pinopi_spec_prog, (results_dict["N_avg"], results_dict["npts"],-1) , **kwargs)        
        return self.plot_mm_pinopi_spec(**kwargs)
    
    
    def plot_mm_pinopi_spec(self,prog_name = 'number split spec', fig_num = None, is_calc_stat_error = None, **kwargs):
        results_dict = self.results['mm_pinopi_spec']
        
    
        if fig_num is  None: fig_num = next_fig_num_by_name(prog_name)
        
        is_calc_stat_error = is_calc_stat_error if is_calc_stat_error is not None else self.is_calc_stat_error
        plt.figure(fig_num)
        data, error = self.process_data(data = [results_dict["I"], results_dict["Q"]], is_calc_stat_error = is_calc_stat_error)
        data, data_prefix, factor = scale_data_units(data)
        error = error*factor
        data_nopi = data[:,0]
        data_pi = data[:,1]
        error_nopi = error[:,0]
        error_pi = error[:,1]
        
        scaled_freqs, freq_prefix, _ = scale_data_units(results_dict["freqs"]-int(self.element_IF(results_dict["qubit"])))
        plt.errorbar(scaled_freqs, data_nopi, yerr = error_nopi, label = r'no $\pi$', fmt = '--ob', capsize = 5, markersize = 6, ecolor = 'k', mfc=(0,0,1,0.5), mec = (0,0,0,1),)
        plt.errorbar(scaled_freqs, data_pi, yerr = error_pi, label = r'$\pi$', fmt = '--or', capsize = 5, markersize = 6, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1),)
        lgd = plt.legend(fontsize = 15)
        lgd.set_draggable(True)
        if self.which_data == 'Phase':
            ylabel = f'{self.which_data} [{data_prefix}Rad]'
        else:
            ylabel = f'{self.which_data} [{data_prefix}V]'
        plt.xlabel("frequencey [{}Hz]".format(freq_prefix))
        plt.ylabel(ylabel)
        plt.title(prog_name + f' ({results_dict["qubit"]})')
        plt.tight_layout()
        
        
        
    def load_displacement_calibration(self, N_avg = 1000, npts = 100, amp_scale_start = 0, amp_scale_stop = 1,
                              mm = None, qubit = None, ro = None, 
                              disp_pulse = 'disp', asym_pulse = 'asym_pulse11', condisp_scale_amp = 1,
                              is_calibrate_readout = False, N_avg_calib = 1000,
                              is_active_reset = False,
                              is_sb_cool = False,
                              **kwargs):
        if qubit is None: qubit = self.main_qubit
        if mm is None: mm = self.main_mm
        if ro is None: ro = self.main_readout
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        if N_avg_calib < N_avg: N_avg_calib = N_avg
        
        amp_scale_list = np.linspace(amp_scale_start, amp_scale_stop, npts)
        amps = amp_scale_list * np.abs(self.pulse_amp(mm, disp_pulse))
        self.results['displacement calibration'] = {"mm": mm, "qubit": qubit, "npts": npts, "amps":amps, "N_avg": N_avg,"N_avg_calib": N_avg_calib // N_avg * N_avg,
                                        'is_calibrate_readout': is_calibrate_readout}
        
        if is_active_reset: run_time = N_avg * npts * 2 * (self.pulse_len(mm, asym_pulse)*4)
        elif is_sb_cool: run_time = N_avg * npts * 2 * (self.pulse_len(mm, asym_pulse)*4+self.pulse_len(mm, 'constant_sideband_cooling_pulse'))
        else: run_time = N_avg * npts * 2 * (self.pulse_len(mm, asym_pulse)*4 + self.wait_between_seq_mm)
        print('Run time is {}s'.format(round(run_time * 1e-9))) 
        
        with program() as self.displacement_calibration_prog:
            
            amp_scale = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)
            n = declare(int)
            
            with for_(n, 0, n<N_avg, n+1):
                if is_calibrate_readout:
                    self.calibrate_readout(qubit, ro, I, Q, N_avg = N_avg_calib // N_avg, **kwargs)
                with for_each_(amp_scale, amp_scale_list.tolist()):
                    reset_frame(qubit, mm)
                    play('pi2_pulse', qubit)
                    align(qubit, mm)
                    self.play_con_disp(qubit, mm, amp_scale = condisp_scale_amp)
                    # frame_rotation_2pi(0.25, mm)
                    play(disp_pulse * amp(amp_scale), mm)
                    # align(qubit, mm)
                    # play('pi_pulse', qubit)
                    # align(qubit, mm)
                    # play(disp_pulse * amp(amp_scale), mm)
                    # frame_rotation_2pi(0.25, mm)
                    self.play_con_disp(qubit, mm, amp_scale = condisp_scale_amp)
                    # frame_rotation_2pi(0.25, mm)
                    play(disp_pulse * amp(-amp_scale), mm)
                    # align(qubit, mm)
                    # play('pi_pulse', qubit)
                    # play(disp_pulse * amp(-amp_scale), mm)
                    align(qubit, mm)
                    play('pi2_pulse', qubit)
                    align(qubit, 'ro')
                    self.perform_full_measurement(I,Q, ro_element = ro,  is_active_reset = is_active_reset, is_sb_cool = is_sb_cool, wait_time = self.wait_between_seq_mm,  **kwargs)
                    
        self.last_prog = self.displacement_calibration_prog
        
        
    def run_displacement_calibration(self, is_save_data = None, save_folder = None, **kwargs):
        results_dict = self.results['displacement calibration']
        if not hasattr(self,'displacement_calibration_prog'): raise ValueError("You must run the load function first!")
        
        results_dict['I'], results_dict['Q'] = self.run_prog(self.displacement_calibration_prog, shape = (results_dict['N_avg'], results_dict['npts']), **kwargs)
        
        if results_dict['is_calibrate_readout']:
            results_dict['I_pi'] = self.last_job.result_handles.get('I_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['Q_pi'] = self.last_job.result_handles.get('Q_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['I_nopi'] = self.last_job.result_handles.get('I_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['Q_nopi'] = self.last_job.result_handles.get('Q_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'displacement calibration', foldername = save_folder)
        return self.plot_displacement_calibration(**kwargs)
    
    def plot_displacement_calibration(self, fig_num = None, which_data = None, is_geo_phase_corr = None, beta = None, beta_err=0, **kwargs):
        if which_data is None: which_data = self.which_data
        if is_geo_phase_corr is None: is_geo_phase_corr = self.is_geo_phase_corr
        results_dict = self.results['displacement calibration']
        mm = results_dict['mm']
        data, err = self.process_data([results_dict['I'], results_dict['Q']])
        if results_dict['is_calibrate_readout']:
            data_e, err_e = self.process_data([results_dict['I_pi'], results_dict['Q_pi']])
            data_g, err_g = self.process_data([results_dict['I_nopi'], results_dict['Q_nopi']])
            data, err = data_to_sigma_z(data=data, data_g=data_g, data_e=data_e, err=err, data_g_err=err_g, data_e_err=err_e)
        else:
            data, units_prefix, scale_factor = scale_data_units(data)
            err = scale_factor * err
            
        amps = results_dict['amps']
        sfit =   sFit('Cos', data, amps, err)
        plt.figure()
        plt.errorbar(amps, data, err, fmt = '-or', capsize = 5, mec = (0,0,0,1), mfc = (1,0,0,0.5))
        plt.plot(amps, sfit.func(amps, *sfit.fit_results), '-b')
        plt.xlabel('Disp. amp. [V]')
        plt.title(f'Displacement calibration {mm}')
        if beta is not None:
            # a = np.pi*sfit.fit_results[1]/beta / 2 #2 from 2 displacements
            a = np.pi*sfit.fit_results[1]/beta
            a_err = np.pi*np.sqrt((np.sqrt(sfit.cov_results[1,1])/beta)**2+(beta_err*sfit.fit_results[1]/beta**2)**2) / 2
            a, a_err = round_value_by_error(a, a_err)
            txt = r'$a = $'f'{a}'r' $\pm$'f' {a_err} 'r'$[V^{-1}]$'
            ann = plt.annotate(txt, xy = (0.5, 0.9), xycoords = 'axes fraction', fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
            ann.draggable()
            print(f'Displacement coefficient is a = {a}+-{a_err} V^-1')
        plt.tight_layout()


    def load_sideband_cooling(self, N_avg = 1000, npts = 100, 
                           max_seq_time = 30000,
                           min_seq_time = None,
                          mm = None, qubit = None, ro = None, 
                          prepare = None,
                          is_calibrate_readout = False, is_active_reset_calib = False, N_avg_calib = 1000,
                          is_preactive_reset = False,
                          is_active_reset = False,
                          is_M1 = False,
                          is_cool = True,
                          condisp_amp_scale = 1,
                          sb_cooling_dict = {},
                          **kwargs):
        if qubit is None: qubit = self.main_qubit
        if mm is None: mm = self.main_mm
        if ro is None: ro = self.main_readout
        if min_seq_time is None:
            if max_seq_time//npts < 16: raise ValueError(f"The minimum wait time is 16ns and your first step is {max_seq_time/npts}")
        else:
            if min_seq_time < 16: raise ValueError(f"The minimum wait time is 16ns and your first step is {max_seq_time/npts}")
            
        if min_seq_time is None:
            times = np.arange(max_seq_time//npts, max_seq_time+1, max_seq_time//npts)
        else:
            if (max_seq_time-min_seq_time)//(npts-1) <4:
                print("Time step is too small. Seting the time step to 4ns")
                npts = (max_seq_time-min_seq_time)//4
            times = np.arange(min_seq_time, max_seq_time+1, (max_seq_time-min_seq_time)//(npts-1))
            
        if npts !=len(times): 
            npts = len(times)
            print(f'Changed <npts> to {npts} to round up the times.')
        
            
        self.results['sideband_cooling'] = {"mm": mm, "qubit": qubit, "npts": npts, "times":times, "N_avg": N_avg, "N_avg_calib":N_avg_calib//N_avg*N_avg, 
                                        'is_calibrate_readout': is_calibrate_readout, "is_M1": is_M1}
        
        if min_seq_time is None:
            run_time = N_avg * npts * 4 * 2 * (max_seq_time/2 + self.wait_between_seq_mm)
        else:
            run_time = N_avg * npts * 4 * 2 * ((max_seq_time+min_seq_time)/2 + self.wait_between_seq_mm)
        print('Run time is {}s'.format(round(run_time * 1e-9))) 
        
        with program() as self.sideband_cooling_prog:
            
            ti = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            n = declare(int)
            qubit_xy_phase = declare(fixed)
            
            with for_(n, 0, n<N_avg, n+1):
                if is_calibrate_readout:
                    self.calibrate_readout(qubit, ro, I, Q, N_avg = N_avg_calib//N_avg, is_active_reset = is_active_reset_calib, **kwargs)
                with for_each_(ti, (times//4).astype(int).tolist()):
                    with for_(qubit_xy_phase, 0.0, qubit_xy_phase<0.9, qubit_xy_phase+0.25):
                        if is_preactive_reset: 
                            self.perform_full_measurement(I,Q, ro = ro, is_save = False, is_active_reset = True)
                            align(qubit, mm, ro)
                        reset_frame(mm)
                        reset_frame(qubit)
                        reset_frame(ro)
                        if prepare is not None: prepare()   
                        align(mm, ro, qubit)
                        if is_cool: 
                            self.sideband_cool(sb_duration = ti, **sb_cooling_dict)
                        else: 
                            wait(ti, mm)
                            align(qubit, mm)
                            play('pi2_pulse', qubit)
                        if is_M1: 
                            align(ro, mm, qubit)
                            self.perform_full_measurement(I,Q, ro = ro, I_output_name = 'M1_I', Q_output_name = 'M1_Q', is_active_reset = False, is_loop_reset = False, is_wait = False)
                            wait(1000//4, ro)
                            align(qubit, mm, ro)
                            align(qubit, mm, ro)
                            play('pi2_pulse', qubit)
                            align(qubit, mm, ro)
                        align(qubit, mm)
                        self.play_con_disp(qubit, mm, amp_scale = condisp_amp_scale)
                        align(qubit, mm)
                        frame_rotation_2pi(qubit_xy_phase, qubit)
                        play('pi2_pulse', qubit)
                        align(qubit, ro)
                        self.perform_full_measurement(I,Q, ro = ro, is_active_reset = is_active_reset, wait_time = self.wait_between_seq_mm,  **kwargs)
                    
        self.last_prog = self.sideband_cooling_prog
        
    def run_sideband_cooling(self, is_save_data = None, **kwargs):
        results_dict = self.results['sideband_cooling']
        if not hasattr(self,'sideband_cooling_prog'): raise ValueError("You must run the load function first!")
        
        results_dict['I'], results_dict['Q'] = self.run_prog(self.sideband_cooling_prog, shape = (results_dict['N_avg'], results_dict['npts'], 4), **kwargs)
        
        if results_dict['is_calibrate_readout']:
            results_dict['I_pi'] = self.last_job.result_handles.get('I_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['Q_pi'] = self.last_job.result_handles.get('Q_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['I_nopi'] = self.last_job.result_handles.get('I_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['Q_nopi'] = self.last_job.result_handles.get('Q_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            
        if results_dict['is_M1']:
            results_dict['I_M1'] = self.last_job.result_handles.get('M1_I').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['npts'], 4))
            results_dict['Q_M1'] = self.last_job.result_handles.get('M1_Q').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['npts'], 4))
            
            
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'sideband_cooling')
        return self.plot_sideband_cooling(**kwargs)
    
    
    def plot_sideband_cooling(self, fig_num = None, which_data = None, is_geo_phase_corr = None, is_calibrate_readout = None, is_M1 = None, **kwargs):
        if which_data is None: which_data = self.which_data
        if is_geo_phase_corr is None: is_geo_phase_corr = self.is_geo_phase_corr
        results_dict = self.results['sideband_cooling']
        if is_calibrate_readout is None: is_calibrate_readout = results_dict['is_calibrate_readout']
        mm = results_dict['mm']
        if is_M1 is None: is_M1 = results_dict['is_M1']
        
        raw_data, raw_err = self.process_data([results_dict['I'], results_dict['Q']],
                                              is_mean = not (is_M1 and is_calibrate_readout),
                                              is_calc_stat_error = not (is_M1 and is_calibrate_readout))
        if is_calibrate_readout:
            data_e, err_e = self.process_data([results_dict['I_pi'], results_dict['Q_pi']])
            data_g, err_g = self.process_data([results_dict['I_nopi'], results_dict['Q_nopi']])
            data, err = data_to_sigma_z(data=raw_data, data_g=data_g, data_e=data_e, err=raw_err, data_g_err=err_g, data_e_err=err_e, is_thresholding = is_M1)
            
            if is_M1:
                data_M1, _ = self.process_data([results_dict['I_M1'], results_dict['Q_M1']], is_mean = False)
                data_M1_sigma_z, _ = data_to_sigma_z(data=data_M1, data_g=data_g, data_e=data_e, is_thresholding = True)
                
                err =  sp.stats.sem(data * data_M1_sigma_z,0)
                data = -(data * data_M1_sigma_z).mean(0)
                    
            Re_C = (data[:,0]-data[:,2])/2
            Im_C = (data[:,1]-data[:,3])/2
            if is_geo_phase_corr: 
                Re_C_beta, Im_C_beta = self.geometric_phase_correction(results_dict['pulse_amp'], Re_C_beta, Im_C_beta, mm)
            
        else:
            Re_C, prefix_Re, factor_Re = scale_data_units((raw_data[:,0]-raw_data[:,2])/2)
            Im_C, prefix_Im, factor_Im = scale_data_units((raw_data[:,1]-raw_data[:,3])/2)
            if factor_Re<factor_Im: factor, prefix = factor_Re,prefix_Re 
            else:  factor, prefix = factor_Im, prefix_Im
            err = raw_err * factor
            
        Re_C_err = np.sqrt(err[:,0]**2+err[:,2]**2)/2
        Im_C_err = np.sqrt(err[:,1]**2+err[:,3]**2)/2
    
        times = results_dict['times']
        scaled_times, time_units_prefix, times_scale_factor = scale_data_units(times*1e-9)
        
        if is_M1 and is_calibrate_readout:
            plt.figure()
            plt.plot(data_M1_sigma_z.mean(0).flatten())
        
        sfit_Re = sFit('Exp(Exp)', Re_C, times, Re_C_err)
        fit_Re, cov_Re = sfit_Re.get_fit_results()
        
        sfit_Im = sFit('Exp(Exp)', Im_C, times, Im_C_err)
        fit_Im, cov_Im = sfit_Im.get_fit_results()
        
        plt.figure()
        plt.errorbar(scaled_times, Re_C, Re_C_err, fmt = 'ro', label = r'$\mathrm{Re}(C(\beta))$')
        plt.errorbar(scaled_times, Im_C, Im_C_err, fmt = 'bo', label = r'$\mathrm{Im}(C(\beta))$')
        if sfit_Re.is_succeed: plt.plot(scaled_times, sfit_Re.func(times, *fit_Re), '-r')
        if sfit_Im.is_succeed: plt.plot(scaled_times, sfit_Im.func(times, *fit_Im), '-b')
        lgd = plt.legend(fontsize = 10)
        lgd.set_draggable(True)
        plt.xlabel(f'Time [{time_units_prefix}s]')
        if is_calibrate_readout: plt.ylabel(r'$\langle\sigma_z\rangle$')
        else: plt.ylabel(f'{self.which_data} [{prefix}V]')
        plt.tight_layout()
        
        if sfit_Re.is_succeed:
            decay, decay_error = round_value_by_error(1e-3/fit_Re[1], 1e-3 * np.sqrt(cov_Re[1,1])/fit_Re[1]**2)
            ann = plt.annotate('decay time = {} +- {} us'.format(decay,decay_error),
                               xy = (np.mean(scaled_times), max(Re_C)), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
            ann.draggable()
        
        if sfit_Im.is_succeed:
            decay, decay_error = round_value_by_error(1e-3/fit_Im[1], 1e-3 * np.sqrt(cov_Im[1,1])/fit_Im[1]**2)
            ann = plt.annotate('decay time = {} +- {} us'.format(decay,decay_error),
                               xy = (np.mean(scaled_times), max(Im_C)), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
            ann.draggable()

    def load_disp_sweep(self, N_avg = 1000, npts = 100, amp_scale_start = -1, amp_scale_stop = 1,
                          mm = None, qubit = None, ro = None,
                          is_calibrate_readout = False, is_active_reset_calib = None, N_avg_calib = 1000,
                          is_preactive_reset = False, is_M1 = False,
                          prepare = None,
                          is_check_sigma_z = False,
                          is_sb_cool = None,
                          amp_scale_cd = 1,
                          **kwargs):
        if qubit is None: qubit = self.main_qubit
        if mm is None: mm = self.main_mm
        if ro is None: ro = self.main_readout
        if is_sb_cool is None: is_sb_cool = self.is_sb_cool
        if is_active_reset_calib is None: is_active_reset_calib = self.is_active_reset
        pulse = 'disp_ECD' if self.is_ECD else 'asym_pulse1'
        if N_avg_calib < N_avg: N_avg_calib = N_avg
        
        amp_scale_list = np.linspace(amp_scale_start, amp_scale_stop, npts)
        amps = amp_scale_list * np.abs(self.pulse_amp(mm, pulse))
        self.results['disp_sweep'] = {"mm": mm, "qubit": qubit, "npts": npts, "amps":amps, "N_avg": N_avg, "N_avg_calib": N_avg_calib//N_avg*N_avg,
                                          "is_calibrate_readout": is_calibrate_readout, "is_M1": is_M1, "is_check_sigma_z": is_check_sigma_z}
        
        if is_sb_cool: run_time = N_avg * npts * 4 * (self.pulse_len(mm, pulse)*2 + self.pulse_len('ro', 'constant_sideband_cooling_pulse'))
        else: run_time = N_avg * npts * 4 * (self.pulse_len(mm, pulse)*2 + self.wait_between_seq_mm)
        print('Run time is {}s'.format(round(run_time * 1e-9))) 
        
        with program() as self.disp_sweep_prog:
            
            amp_scale = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)
            n = declare(int)
            qubit_xy_phase = declare(fixed)
            
            with for_(n, 0, n<N_avg, n+1):
                if is_calibrate_readout:
                    self.calibrate_readout(qubit, ro, I, Q, N_avg = N_avg_calib//N_avg, is_active_reset = is_active_reset_calib)
                    align(qubit, ro)
                with for_each_(amp_scale, amp_scale_list.tolist()):
                    with for_(qubit_xy_phase, 0.0, qubit_xy_phase<0.9, qubit_xy_phase+0.25):
                        if is_preactive_reset: 
                            self.perform_full_measurement(I,Q, ro = ro, is_save = False, is_active_reset = True)
                            align(ro, mm, qubit)
                        reset_frame(qubit)
                        reset_frame(mm)
                        play('disp' * amp(amp_scale), element = mm)
                        if is_M1: 
                            align(ro, mm, qubit)
                            self.perform_full_measurement(I,Q, ro = ro, I_output_name = 'M1_I', Q_output_name = 'M1_Q', is_active_reset = False, is_wait = False)
                            wait(1000//4, ro)
                            align(qubit, mm, ro)
                        play('pi2_pulse', qubit)
                        align(qubit, mm)
                        self.play_con_disp(qubit, mm, amp_scale = amp_scale_cd)
                        align(qubit, mm)
                        frame_rotation_2pi(qubit_xy_phase, qubit)
                        if not is_check_sigma_z: play('pi2_pulse', qubit)
                        align(qubit, ro)
                        self.perform_full_measurement(I,Q, ro = ro, wait_time = self.wait_between_seq_mm, is_active_reset = False, is_sb_cool = is_sb_cool, **kwargs)
                        align(qubit, ro)
                    
        self.last_prog = self.disp_sweep_prog
        
    def run_disp_sweep(self, is_save_data = None, **kwargs):
        results_dict = self.results['disp_sweep']
        if not hasattr(self,'disp_sweep_prog'): raise ValueError("You must run the load function first!")
        results_dict['I'], results_dict['Q'] = self.run_prog(self.disp_sweep_prog, shape = (results_dict['N_avg'], results_dict['npts'], 4), **kwargs)
        
        if results_dict['is_calibrate_readout']:
            results_dict['I_pi'] = self.last_job.result_handles.get('I_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['Q_pi'] = self.last_job.result_handles.get('Q_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['I_nopi'] = self.last_job.result_handles.get('I_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['Q_nopi'] = self.last_job.result_handles.get('Q_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            
            if results_dict['is_M1']:
                results_dict['I_M1'] = self.last_job.result_handles.get('M1_I').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['npts'], 4))
                results_dict['Q_M1'] = self.last_job.result_handles.get('M1_Q').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['npts'], 4))
                
                
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'disp_sweep')
        return self.plot_disp_sweep(**kwargs)
        
        
    def plot_disp_sweep(self, fig_num = None, which_data = None, 
                            is_geo_phase_corr = None, is_fill_symmetric = None, is_M1 = None,  post_selection = None,
                            is_calibrate_readout = None,
                            a = None, a_err = 0, **kwargs):
        if which_data is None: which_data = self.which_data
        if is_geo_phase_corr is None: is_geo_phase_corr = self.is_geo_phase_corr
        results_dict = self.results['disp_sweep']
        if is_fill_symmetric is None: is_fill_symmetric = (np.abs(np.sum(results_dict['amps']))>1e-6 and results_dict['is_calibrate_readout'])
        mm = results_dict['mm']
        if is_M1 is None: is_M1 = results_dict['is_M1']
        if is_calibrate_readout is None: is_calibrate_readout = results_dict['is_calibrate_readout']
        data, err = self.process_data([results_dict['I'], results_dict['Q']], 
                                      is_mean = not (is_M1 and is_calibrate_readout), 
                                      is_calc_stat_error = not (is_M1 and is_calibrate_readout))
        
        if is_calibrate_readout:
            data_e, err_e = self.process_data([results_dict['I_pi'], results_dict['Q_pi']])
            data_g, err_g = self.process_data([results_dict['I_nopi'], results_dict['Q_nopi']])
            _,_,means = self.find_threshold(np.append(results_dict['I_pi'],results_dict['I_nopi']), np.append(results_dict['Q_pi'],results_dict['Q_nopi']), n_means = 2, is_fit_circle = False, is_ascending = data_g.mean()<data_e.mean())
            data, err = data_to_sigma_z(data=data, data_g=means[0], data_e=means[1], err=err, data_g_err=0, data_e_err=0, 
                                        is_thresholding = is_M1)
            data = - data
        
            if is_M1:
                data_M1, _ = self.process_data([results_dict['I_M1'], results_dict['Q_M1']], is_mean = False)
                data_M1_sigma_z, _ = data_to_sigma_z(data=data_M1, data_g=data_g, data_e=data_e, is_thresholding = True)
                
                if post_selection is not None:
                    if post_selection == 'g': 
                        inds = np.where(data_M1_sigma_z==1) #remove excited
                        data = - data
                    elif post_selection == 'e': 
                        inds = np.where(data_M1_sigma_z==-1) #remove ground.
                    data_post_M1 = data.copy()
                    data_post_M1[inds] = np.nan
                    data = np.nanmean(data_post_M1, 0)
                    NN = np.sum(~np.isnan(data_post_M1),0)
                    err =  np.nanstd(data_post_M1, 0)/np.sqrt(NN)
                else:
                    err =  sp.stats.sem(data * data_M1_sigma_z,0)
                    data = (data * data_M1_sigma_z).mean(0)
        
        else:
            data, units_prefix, scale_factor = scale_data_units(data)
            err = scale_factor * err
        
        if is_fill_symmetric: 
            amps = np.append(-np.flip(results_dict['amps'][1:]),results_dict['amps'])
            if not results_dict['is_check_sigma_z']:
                Y = np.append(np.flip((data[1:,0]-data[1:,2])/2), (data[:,0]-data[:,2])/2)
                X = np.append(-np.flip((data[1:,1]-data[1:,3])/2), (data[:,1]-data[:,3])/2)
            else:
                Y = np.append(np.flip((data[1:,0]+data[1:,2])/2), (data[:,0]+data[:,2])/2)
                X = np.append(-np.flip((data[1:,1]+data[1:,3])/2), (data[:,1]+data[:,3])/2)
            Y_err = np.append(np.flip(np.sqrt(err[1:,0]**2+err[1:,2]**2)/2), np.sqrt(err[:,0]**2+err[:,2]**2)/2)
            X_err = np.append(np.flip(np.sqrt(err[1:,1]**2+err[1:,3]**2)/2), np.sqrt(err[:,1]**2+err[:,2]**3)/2)
        else: 
            amps = results_dict['amps']
            if not results_dict['is_check_sigma_z']:
                Y = (data[:,0]-data[:,2])/2
                X = (data[:,1]-data[:,3])/2
            else:
                Y = (data[:,0]+data[:,2])/2
                X = (data[:,1]+data[:,3])/2
            Y_err = np.sqrt(err[:,0]**2+err[:,2]**2)/2
            X_err = np.sqrt(err[:,1]**2+err[:,3]**2)/2
        if is_geo_phase_corr: X,Y = self.geometric_phase_correction(amps, X, Y, mm)
        

        
        fig_num = next_fig_num_by_name('Disp Sweep')
        plt.figure(fig_num)
        plt.errorbar(amps, X, X_err, fmt = 'ob', label = 'X')
        plt.errorbar(amps, Y, Y_err, fmt = 'or', label = 'Y')
        
        sigma=0
        
        abs_XY = np.sqrt(X**2+Y**2)
        abs_XY_err = np.sqrt(((X_err*2*X)**2+(Y_err*2*Y)**2)/(X**2+Y**2)/2)
        # plt.errorbar(amps, abs_XY, abs_XY_err, fmt = 'ok', label = 'abs')
    
        sfit_abs = non_TimeDomain_fit('Gaussian', abs_XY, amps, abs_XY_err)
        if sfit_abs.is_succeed:
            sigma = sfit_abs.fit_params[2]
            sigma_err = np.sqrt(sfit_abs.fit_cov[2,2])
            # if a is None: print('\n Condisp amplitude = {}+-{} 1/Volt'.format(*round_value_by_error(1/sigma, sigma_err/sigma**2)))
            # plt.plot(amps, sfit_abs.func(amps, *sfit_abs.fit_params), '-k')
                
        sfit_complex = non_TimeDomain_fit('GaussianCosiSin', Y, amps, itrace = X)
        if sigma != 0: sfit_complex.guess[2] = sigma
        fit_params, fit_cov = sfit_complex._fit()
        if sfit_complex.is_succeed:
            Amp, Freq, sigma, Delta, Center, Offset = fit_params
            freq_err = np.sqrt(fit_cov[1,1])
            sigma_err = np.sqrt(fit_cov[2,2])
            if a is not None: 
                sigma = 1/a
                sigma_err = a_err/a**2
                
            alpha = Freq*np.pi*sigma
            alpha_err = np.sqrt((freq_err*sigma)**2+(sigma_err*Freq)**2)*2*np.pi
            plt.plot(amps, sfit_complex.func(amps, *fit_params)[:len(amps)], '-r')
            plt.plot(amps, sfit_complex.func(amps, *fit_params)[len(amps):], '-b')
            plt.plot(amps, sfit_abs.func(amps, fit_params[0], fit_params[4], fit_params[2], fit_params[5]), '--k')
            if a is None: print('\n Condisp amplitude = {}+-{} 1/Volt'.format(*round_value_by_error(1/fit_params[2], np.sqrt(fit_cov[2,2])/fit_params[2]**2)))
            print('\n alpha = {}+-{}'.format(*round_value_by_error(alpha,alpha_err)))
        if not is_calibrate_readout:
            plt.ylabel('{} [{}V]'.format(which_data, units_prefix))
        else:
            plt.ylabel(r'$\langle\sigma_z\rangle$')
        plt.xlabel('Sweep disp. amp. [V]')
        plt.grid()
        lgd = plt.legend(fontsize = 15)
        lgd.set_draggable(True)
        plt.tight_layout()
        
        if is_M1 and is_calibrate_readout and not is_fill_symmetric:
            plt.figure()
            plt.plot(amps, data_M1_sigma_z.mean(0))
            
            
#%% Roee functions

    def load_cat_and_back_over_g(self, N_avg = 1000, npts = 100,start=0, amp_scale = 1,
                          mm = None, qubit = None, ro = None, is_ECD = None,
                          is_calibrate_readout = False, N_avg_calib = 1000, is_active_reset_calib = False,
                          
                          **kwargs):
        if is_ECD is None: is_ECD = self.is_ECD
        if is_ECD: pulse = 'disp_ECD'
        else: pulse = 'asym_pulse1'
        if qubit is None: qubit = self.main_qubit
        if mm is None: mm = self.main_mm
        if ro is None: ro = self.main_readout
        
        if N_avg_calib < N_avg: N_avg_calib = N_avg
        
        # amp_scale_list = np.linspace(amp_scale_start, amp_scale_stop, npts)
        # amps = amp_scale_list * np.abs(self.pulse_amp(mm, pulse))
        
        ## this works as the number of runs over g
        amps = np.arange(start, npts+start, 1)
        self.results['cat_and_back'] = {"mm": mm, "qubit": qubit, "npts": npts, "amps":amps, "N_avg": N_avg, "N_avg_calib": N_avg_calib//N_avg*N_avg,
                                        'is_calibrate_readout': is_calibrate_readout}
        
        run_time = N_avg * npts * (self.pulse_len(mm, pulse)*4*(start+npts/2) + self.wait_between_seq_mm)
        print('Run time is {}s'.format(round(run_time * 1e-9))) 
        
        with program() as self.cat_and_back_prog:
            
            amp_scale = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)
            n = declare(int)
            qubit_xy_phase = declare(fixed)
            i = declare(int)
            
            with for_(n, 0, n<N_avg, n+1):
                if is_calibrate_readout:
                    self.calibrate_readout(qubit, ro, I, Q, N_avg = N_avg_calib//N_avg, is_active_reset = is_active_reset_calib, **kwargs)
                with for_(i, start, i < npts + start, i+1):
                    with switch_(i):
                        for j in range(start, npts+start):
                            with case_(j):
                                reset_frame(qubit)
                                reset_frame(mm)
                                
                                for k in range(j):
                                    align(qubit, mm)
                                    self.play_con_disp(qubit, mm, amp_scale=amp_scale)
                                    align(mm, qubit)
                                    play('pi_pulse', qubit)
                                    align(mm, qubit)
                                    self.play_con_disp(qubit, mm, amp_scale=-amp_scale)
                                    align(qubit, mm)
                                    
                                align(qubit, ro, mm)
                                self.perform_full_measurement(I,Q, ro_element = ro,  is_active_reset = False, wait_time = self.wait_between_seq_mm, **kwargs)
                    
        self.last_prog = self.cat_and_back_prog       
        
        
        
        
#%% loop with python progs

    def _load_py_loop_cat_and_back_prog(self, is_sb_cool, wait_time):
        results_dict = self.results['cat_and_back']
        N_avg = results_dict['N_avg']
        mm = results_dict['mm']
        ro = results_dict['ro']
        qubit = results_dict['qubit']
        
        with program() as self.cat_and_back_prog:
            
            I = declare(fixed)
            Q = declare(fixed)
            n = declare(int)
            qubit_xy_phase = declare(fixed)
            
            with for_(n, 0, n<N_avg, n+1):
                with for_(qubit_xy_phase, 0.0, qubit_xy_phase<0.9, qubit_xy_phase+0.25):
                    reset_frame(qubit)
                    reset_frame(mm)
                    align(qubit, mm)
                    play('pi2_pulse', qubit)
                    align(qubit, mm)
                    self.play_con_disp(qubit, mm)
                    align(mm, qubit)
                    play('pi_pulse', qubit)
                    align(mm, qubit)
                    self.play_con_disp(qubit, mm, amp_scale = -1)
                    # self.play_con_disp(qubit, mm, amp_scale = amp_scale)
                    align(qubit, mm)
                    frame_rotation_2pi(qubit_xy_phase, qubit)
                    play('pi2_pulse', qubit)
                    align(qubit, ro, mm)
                    self.perform_full_measurement(I,Q, ro_element = ro,  is_active_reset = False, wait_time = wait_time, is_sb_cool = is_sb_cool)
                    
                with for_(qubit_xy_phase, 0.0, qubit_xy_phase<0.3, qubit_xy_phase+0.25):
                    reset_frame(qubit)
                    reset_frame(mm)
                    align(qubit, mm)
                    play('pi2_pulse', qubit)
                    align(qubit, mm)
                    self.play_con_disp(qubit, mm)
                    align(mm, qubit)
                    play('pi_pulse', qubit)
                    align(mm, qubit)
                    self.play_con_disp(qubit, mm, amp_scale = -1)
                    align(qubit, mm)
                    play('pi_pulse', qubit, condition = qubit_xy_phase==0)
                    align(qubit, ro, mm)
                    self.perform_full_measurement(I,Q, ro_element = ro,  is_active_reset = False, wait_time = wait_time, is_sb_cool = is_sb_cool)
                    
    def _load_py_loop_char_func_line_prog(self, is_sb_cool, prepare, prepare_dict):
        results_dict = self.results['char_func_line']
        N_avg = results_dict['N_avg']
        mm = results_dict['mm']
        ro = results_dict['ro']
        qubit = results_dict['qubit']
        is_M1 = results_dict['is_M1']
        is_preactive_reset = results_dict['is_preactive_reset']
        
        with program() as self.char_func_line_prog:
            
            I = declare(fixed)
            Q = declare(fixed)
            n = declare(int)
            qubit_xy_phase = declare(fixed)
            
            with for_(n, 0, n<N_avg, n+1):
                with for_(qubit_xy_phase, 0.0, qubit_xy_phase<0.9, qubit_xy_phase+0.25):
                    if is_preactive_reset: 
                        self.perform_full_measurement(I,Q, ro = ro, is_save = False, is_active_reset = True)
                        align(ro, mm, qubit)
                    reset_frame(qubit)
                    reset_frame(mm)
                    if prepare is not None:
                        prepare(**prepare_dict)
                    if is_M1: 
                        align(ro, mm, qubit)
                        self.perform_full_measurement(I,Q, ro = ro, I_output_name = 'M1_I', Q_output_name = 'M1_Q', is_active_reset = False, is_sb_cool = False, is_wait = False)
                        if self.wait_after_reset >= 16: wait(self.wait_after_reset//4, ro)
                        align(qubit, mm, ro)
                    play('pi2_pulse', qubit)
                    align(qubit, mm)
                    self.play_con_disp(qubit, mm)
                    align(qubit, mm)
                    frame_rotation_2pi(qubit_xy_phase, qubit)
                    play('pi2_pulse', qubit)
                    align(qubit, ro)
                    self.perform_full_measurement(I,Q, ro = ro, wait_time = self.wait_between_seq_mm, is_active_reset = False, is_sb_cool = is_sb_cool)
                    
                    
        self.last_prog = self.char_func_line_prog
        
        
    def _load_py_loop_char_func_prog(self, prepare, prepare_dict, is_sb_cool):
        results_dict = self.results['char_func']
        N_avg = results_dict['N_avg']
        mm = results_dict['mm']
        ro = results_dict['ro']
        qubit = results_dict['qubit']
        is_M1 = results_dict['is_M1']
        is_preactive_reset = results_dict['is_preactive_reset']
        with program() as self.char_func_prog:
            
            I = declare(fixed)
            Q = declare(fixed)
            n = declare(int)
            qubit_xy_phase = declare(fixed)
            
            with for_(n, 0, n<N_avg, n+1):
                with for_(qubit_xy_phase, 0, qubit_xy_phase<0.9, qubit_xy_phase+0.25):
                    if is_preactive_reset: 
                        self.perform_full_measurement(I,Q, ro = ro, is_save = False, is_active_reset = True)
                        align(qubit, ro, mm)
                    reset_frame(qubit)
                    reset_frame(mm)
                    if prepare is not None:
                        prepare(**prepare_dict)
                    if is_M1: 
                        self.perform_full_measurement(I,Q, ro = ro, 
                                                      I_output_name = 'M1_I', Q_output_name = 'M1_Q',
                                                      is_active_reset = False, 
                                                      is_loop_reset = False, is_wait = False)
                        if self.wait_after_reset >= 16: wait(self.wait_after_reset//4, ro)
                        align(qubit, mm, ro)
                        
                    play('pi2_pulse', qubit)
                    align(qubit, mm)
                    self.play_con_disp(qubit, mm)
                    align(qubit, mm)
                    frame_rotation_2pi(qubit_xy_phase, qubit)
                    play('pi2_pulse', qubit)
                    align(qubit, ro)
                    
                    self.perform_full_measurement(I,Q, ro = ro, wait_time = self.wait_between_seq_mm, is_active_reset = False, is_sb_cool = is_sb_cool)
                        
        self.last_prog = self.cat_and_back_prog  
    
    def sideband_couple(self, qubit = None, ro_element = None, mm = None, mm_target = None,
                      sb_steady_time = None, sb_wait_time = None, sb_duration = None, sb_extra_duration = None,
                      rabi_sideband_cooling_pulse = 'rabi_pulse', sideband_pulse = 'constant_sideband_cooling_pulse', 
                      mm_sideband_cooling_pulse = 'constant_sideband_cooling_pulse',
                      is_ramp = None,
                      is_pi2 = None,  pi2_amp_scale = 1.0,
                      ro_detuning = None, qb_detuning = None, mm_detuning = None, mm_target_detuning = None,
                      is_align_all = True,
                      **kwargs):
        
        if ro_element is None: ro_element = self.main_readout
        if qubit is None: qubit = self.main_qubit
        if type(qubit) is not list: qubit = [qubit]
        if mm is None: mm = self.main_mm
        if mm_target is None: mm_target = self.secondary_mm
        
        for qb in qubit:
            reset_frame(qb)
             
        if ro_detuning is None: ro_detuning = self.sb_couple_kwargs['ro_detuning']
        if qb_detuning is None: qb_detuning = self.sb_couple_kwargs['qb_detuning']
        if mm_detuning is None and mm is not None: mm_detuning = self.sb_couple_kwargs['mm_detuning']
        if mm_target_detuning is None and mm_target is not None: mm_target_detuning = self.sb_couple_kwargs['mm_target_detuning']
        if is_pi2 is None: is_pi2 = self.sb_couple_kwargs['is_pi2']
        if is_ramp is None: is_ramp = self.sb_couple_kwargs['is_ramp']
        if sb_wait_time is None: sb_wait_time = self.sb_couple_kwargs['sb_wait_time']
        if sb_steady_time is None: sb_steady_time = self.sb_couple_kwargs['sb_steady_time']
        if sb_extra_duration is None: sb_extra_duration = self.sb_couple_kwargs['sb_extra_duration']
        if sb_duration is None: sb_duration = self.sb_couple_kwargs['sb_duration']
        
        if ro_detuning!=0: update_frequency(ro_element, self.element_IF(ro_element) + ro_detuning, keep_phase = True)
        if qb_detuning!=0: update_frequency(qubit[0], self.element_IF(qubit[0]) + qb_detuning, keep_phase = True)
        if mm_detuning!=0 and mm: update_frequency(mm, self.element_IF(mm) + mm_detuning, keep_phase = True)
        if mm_target_detuning!=0 and mm_target: update_frequency(mm_target, self.element_IF(mm_target) + mm_target_detuning, keep_phase = True)
        
        if sb_extra_duration == 'auto':
            sb_extra_duration = 0
            if is_pi2: sb_extra_duration+= self.pulse_len(qubit[0], 'pi2_pulse')
            if is_ramp: sb_extra_duration+= 2*self.pulse_ramp_len(qubit[0], rabi_sideband_cooling_pulse)
        qubit_wait_time = sb_steady_time
        if is_ramp: qubit_wait_time += self.pulse_ramp_len(ro_element, sideband_pulse)
            
        if type(sb_duration) != int:
            actual_sb_duration = declare(int)
            assign(actual_sb_duration, sb_duration+(sb_steady_time+sb_extra_duration)//4)
        else:
            if sb_duration is None or sb_duration == 0: 
                sb_duration = self.pulse_len(qubit[0], rabi_sideband_cooling_pulse)//4
            actual_sb_duration = sb_duration//4+(sb_steady_time+sb_extra_duration)//4
            sb_duration = sb_duration//4
        pi2_phase = 0.25 if ro_detuning > 0 else -0.25
        
        if is_align_all: 
            if mm: align(qubit[0], mm, ro_element)
            else: align(*qubit, ro_element)
            
            
        # if is_ramp: play(sideband_pulse+"_ramp_up", ro_element)
        # play(sideband_pulse, ro_element, duration = actual_sb_duration)
        # if is_ramp: play(sideband_pulse+"_ramp_down", ro_element)
        for qb in qubit:
            if qubit_wait_time//4>0:
                wait(qubit_wait_time//4, qb)
                
            if is_ramp:
                play(rabi_sideband_cooling_pulse + "_ramp_up", qb)
            play(rabi_sideband_cooling_pulse, qb, duration = sb_duration)
            if is_ramp:
                play(rabi_sideband_cooling_pulse + "_ramp_down", qb)
                
        if mm:
            if is_ramp: play(mm_sideband_cooling_pulse + "_ramp_up", mm)
            play(mm_sideband_cooling_pulse, mm, duration = actual_sb_duration)
            if is_ramp: play(mm_sideband_cooling_pulse + "_ramp_down", mm)
        if mm_target:
            if is_ramp: play(mm_sideband_cooling_pulse + "_ramp_up", mm_target)
            play(mm_sideband_cooling_pulse, mm_target, duration = actual_sb_duration)
            if is_ramp: play(mm_sideband_cooling_pulse + "_ramp_down", mm_target)
        
        
        if is_pi2:
            frame_rotation_2pi(pi2_phase, qb)
            for qb in qubit:
                if type(pi2_amp_scale) is float or type(pi2_amp_scale) is int:
                    if pi2_amp_scale == 1.0:
                        play('pi2_pulse', qb)
                    else:
                        play('pi2_pulse' * amp(pi2_amp_scale), qb)
                else:
                    play('pi2_pulse' * amp(pi2_amp_scale), qb)
                
        if ro_detuning!=0: update_frequency(ro_element, self.element_IF(ro_element), keep_phase = True)
        if qb_detuning!=0: update_frequency(qubit[0], self.element_IF(qubit[0]), keep_phase = True)
        if mm_detuning!=0 and mm: update_frequency(mm, self.element_IF(mm), keep_phase = True)
        if mm_target_detuning!=0 and mm_target: update_frequency(mm_target, self.element_IF(mm_target), keep_phase = True)
    
        
        align(*qubit, ro_element)
        
        if sb_wait_time//4  > 0: 
            wait(int(sb_wait_time//4), ro_element)
            align(*qubit, ro_element)
    
    def load_optimize_prepare(self, N_avg = 1000, 
                        mm = None, mm_sec = None, qubit = None, ro = None,
                        duration_start = 0, duration_stop = 1000, duration_npts = 100,
                        is_calibrate_readout = False, is_preactive_reset = True, N_avg_calib = 1000, is_active_reset_calib = None,
                        prepare = None, prepare_kwargs = {},
                        is_M1 = False, is_sb_cool = None,
                        **kwargs):
        if qubit is None: qubit = self.main_qubit
        if mm is None: mm = self.main_mm
        if mm_sec is None: mm_sec = self.secondary_mm
        if ro is None: ro = self.main_readout
        if is_sb_cool is None: is_sb_cool = self.is_sb_cool
        if is_active_reset_calib is None: is_active_reset_calib = self.is_active_reset
        if prepare is None: prepare = prepare_fock
        if N_avg_calib < N_avg: N_avg_calib = N_avg
        
        duration_list = np.linspace(duration_start, duration_stop, duration_npts, dtype = int)
        rabi_ramp_up_time = int(self.pulse_ramp_len(qubit, 'rabi_coupling'))
        pi2_time = int(self.pulse_len(qubit, 'pi2_pulse'))
        sb_ramp_up_time = int(self.pulse_ramp_len(mm, 'constant_sideband_cooling_pulse'))
        sb_time_list = (duration_list + 2*rabi_ramp_up_time + pi2_time*2)
        
        # amp_scale_list = np.linspace(amp_scale_start, amp_scale_stop, npts)
        # amps = amp_scale_list * np.abs(self.pulse_amp(mm, pulse))
        self.results['optimize_prepare'] = {"mm": mm,'mm_sec' : mm_sec, "qubit": qubit, 'duration_npts' : duration_npts, "duration_list": duration_list, "N_avg": N_avg, "N_avg_calib": N_avg_calib//N_avg*N_avg,
                                          "is_calibrate_readout": is_calibrate_readout, "is_M1": is_M1, 'is_sb_cool' : is_sb_cool}
        if prepare is not None:
            del prepare_kwargs['duration']
            self.results['optimize_prepare']['prepare'] =  prepare.__name__
            self.results['optimize_prepare']['prepare_dict'] = prepare_kwargs
        
        if is_sb_cool:
            run_time = N_avg * len(duration_list) * (np.mean(duration_list) + self.pulse_len(qubit, 'pi2_pulse')*2 + self.pulse_len('ro', 'constant_sideband_cooling_pulse'))
        else:
            run_time = N_avg * (np.sum(duration_list) + len(duration_list) * self.pulse_len(qubit, 'pi2_pulse')*2 + self.wait_between_seq_mm)
        print('Run time is {}s'.format(round(run_time * 1e-9))) 
        
        sb_time_list = sb_time_list//4
        duration_list = duration_list//4
        with program() as self.optimize_prepare_prog:
            
            duration = declare(int)
            sb_time = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            n = declare(int)
            
            with for_(n, 0, n<N_avg, n+1):
                if is_calibrate_readout:
                    self.calibrate_readout(qubit, ro, I, Q, N_avg = N_avg_calib//N_avg,  is_active_reset = is_active_reset_calib, is_sb_cool = is_sb_cool and not is_active_reset_calib, **kwargs)
                with for_each_((duration,sb_time), (duration_list.tolist(),sb_time_list.tolist())):
                    if is_preactive_reset: 
                        self.perform_full_measurement(I,Q, ro = ro, is_save = False, is_active_reset = True)
                        align(qubit, ro, mm)
                    reset_frame(qubit)
                    reset_frame(mm)
                    reset_frame(mm_sec)
                    reset_frame(ro)
                    if prepare is not None:
                        prepare(duration = duration,sb_time = sb_time, sb_ramp_up_time = sb_ramp_up_time, I=I, Q=Q, I_output_name = 'mid_M1_I', Q_output_name = 'mid_M1_Q', is_loop_reset = False,**prepare_kwargs)
                    self.perform_full_measurement(I,Q, ro = ro, wait_time = self.wait_between_seq_mm, is_active_reset = False, is_sb_cool = is_sb_cool, **kwargs)
                    
        self.last_prog = self.optimize_prepare_prog
        
    
            
    def run_optimize_prepare(self, is_save_data = None, **kwargs):
        results_dict = self.results['optimize_prepare']
        if not hasattr(self,'optimize_prepare_prog'): raise ValueError("You must run the load function first!")
        
        results_dict['I'], results_dict['Q'] = self.run_prog(self.optimize_prepare_prog, shape = (results_dict['N_avg'], results_dict['duration_npts']), **kwargs)
        
        if results_dict['is_calibrate_readout']:
            results_dict['I_pi'] = self.last_job.result_handles.get('I_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['Q_pi'] = self.last_job.result_handles.get('Q_calib_pi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['I_nopi'] = self.last_job.result_handles.get('I_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            results_dict['Q_nopi'] = self.last_job.result_handles.get('Q_calib_nopi').fetch_all()['value'].reshape((results_dict['N_avg_calib'],-1))
            
        results_dict['I_mid_M1'] = self.last_job.result_handles.get('mid_M1_I').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['duration_npts']))
        results_dict['Q_mid_M1'] = self.last_job.result_handles.get('mid_M1_Q').fetch_all()['value'].reshape((results_dict['N_avg'], results_dict['duration_npts']))
            
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'optimize_prepare')
        return self.plot_optimize_prepare(**kwargs)


    def plot_optimize_prepare(self, fig_num = None, which_data = None, a = 1, **kwargs):
        if which_data is None: which_data = self.which_data
        results_dict = self.results['optimize_prepare']
        mm = results_dict['mm']
        data, err = self.process_data([results_dict['I'], results_dict['Q']], is_mean = False)
        is_calibrate_readout = results_dict['is_calibrate_readout']
        if is_calibrate_readout:
            data_e, err_e = self.process_data([results_dict['I_pi'], results_dict['Q_pi']])
            data_g, err_g = self.process_data([results_dict['I_nopi'], results_dict['Q_nopi']])
            _,_,means = self.find_threshold(np.append(results_dict['I_pi'],results_dict['I_nopi']), np.append(results_dict['Q_pi'],results_dict['Q_nopi']), n_means = 2, is_fit_circle = False, is_ascending = data_g.mean()<data_e.mean())
            data, err = data_to_sigma_z(data=data, data_g=means[0], data_e=means[1], err=err, data_g_err=0, data_e_err=0, is_thresholding = results_dict['is_calibrate_readout'])                            
            data_mid_M1, _ = self.process_data([results_dict['I_mid_M1'], results_dict['Q_mid_M1']], is_mean = False)
            data_mid_M1_sigma_z, _ = data_to_sigma_z(data=data_mid_M1, data_g=data_g, data_e=data_e, is_thresholding = True)
            inds = np.where(data_mid_M1_sigma_z==1) #remove excited
            data_post_mid_M1 = data.copy()
            data_post_mid_M1[inds] = np.nan
            data = np.nanmean(data_post_mid_M1, 0)
            NN = np.sum(~np.isnan(data_post_mid_M1),0)
            err =  np.nanstd(data_post_mid_M1, 0)/np.sqrt(NN)
        else:
            data, units_prefix, scale_factor = scale_data_units(data)
            err = scale_factor * err
        
        

        
        fig_num = next_fig_num_by_name('Optimize Prepare')
        
        if is_calibrate_readout: vmax, vmin = 1,-1
        else: vmax, vmin = np.max(data), -np.max(data)
        
        xlabel = r'Prepare Duration [$\mu s$]'
        ylabel = r'M1'
        
        fig,axs = plt.subplots(1,1, num = fig_num)
        axs.plot(results_dict['duration_list']*1e-3, data,'o')
                
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        axs.set_title(r'Duration Optimization')
        plt.tight_layout()
        

            
        return data, results_dict['duration_list']
