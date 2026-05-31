# -*- coding: utf-8 -*-
"""
Created on Sun May  3 14:33:59 2020

@author: Shay
"""



import sys
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\measurementClass')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pickle
import time

# OPX Qunatum Machine:
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.qua.lib import Math as Math
from qm import SimulationConfig
from qualang_tools.plot import interrupt_on_close #pip install qualang_tools
from qualang_tools.results import fetching_tool

from Time_characterzation import TiDo_Chara
from Time_domain_base_class import round_value_by_error, histogram_fidelity
import data_processing
from plotting import plot_2D, plot_fft, next_fig_num_by_name
from datetime import datetime

from data_processing import S21_2fr_notch, round_sig_dig, correct_non_integer_demod

matplotlib.rcParams['font.family'] = ['Cambria']

class burstDetector(TiDo_Chara):
#%% initialzation and attributes 
   
    def __init__(self, 
                 Config, 
                 burst_element = 'nis',
                 burst_pulse = 'burst_pulse',
                 **kwargs):
        
        self.experiment_dict = {'parameters':{},
                                'results':{}
                                }
        
        super().__init__(Config, **kwargs)
        self.burst_element = burst_element
        self.burst_pulse = burst_pulse
        
        
        
    def load_burst_no_burst(self, N_avg = 1000, 
                       meas_type = 'sliced',
                       npts = 101,
                       burst_element = None, 
                       burst_pulse = None,
                       ro_element = None,
                       res_name = None,
                       ro_pulse = None,
                       burst_delay = 0,
                       ramp_up = 0,
                       ):
        
        if burst_element is None: burst_element = self.burst_element
        if burst_pulse is None: burst_pulse = self.burst_pulse
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        
        if ramp_up > 0:
            is_ramp_up_meas = False
            is_reset_phase_meas = False
        else:
            is_ramp_up_meas = True
            is_reset_phase_meas = True
        
        if type(ro_element)!=list:
            ro_element = [ro_element]
        
        self.burst_no_burst_results = {}
        results_dict = self.burst_no_burst_results
        results_dict['N_avg'] = N_avg
        results_dict['meas_type'] = meas_type
        results_dict['npts'] = npts
        results_dict['res_name'] = res_name
        results_dict['ro_element'] = ro_element
        results_dict['ro_pulse'] = ro_pulse
        results_dict['ro_pulse_len'] = [self.pulse_len(ro_el,ro_pulse) for ro_el in ro_element]
        results_dict['ro_pulse_amp'] = [self.pulse_amp(ro_el,ro_pulse) for ro_el in ro_element]
        results_dict['ro_pulse_freq'] = [self.element_freq(ro_el) for ro_el in ro_element]
        results_dict['ramp_up'] = ramp_up
        results_dict['burst_element'] = burst_element
        results_dict['burst_pulse'] = burst_pulse
        results_dict['burst_delay'] = burst_delay
        results_dict['burst_pulse_len'] = self.pulse_len(burst_element,burst_pulse)
        results_dict['burst_pulse_amp'] = self.pulse_amp(burst_element,burst_pulse)
        results_dict['burst_pulse_freq'] = [self.element_freq(burst_element)]
        results_dict['wait_between_seq'] = self.wait_between_seq
        
        
        if meas_type in ['sliced', 'accumulated', 'moving_window']: 
            chunk_size =  [self.config['pulses'][self.config['elements'][ro_el]['operations'][ro_pulse]]['length'] // npts // 4 for ro_el in ro_element]
            results_dict['chunk_size'] = chunk_size
        else:
            ValueError('meas_type must be either sliced, accumulated or moving_window, but is neither!')
            # results_dict['chunk_size'] = ro_element
        # if meas_type == 'moving_window':
        #     results_dict['window_size'] = window_size
            
        
        run_time = 2*N_avg*(np.sum(results_dict['ro_pulse_len'])+ramp_up+4*results_dict['wait_between_seq'])
        print('Run time is {}s'.format(round(run_time * 1e-9)))    
        
        with program() as self.burst_no_burst_prog:
            n = declare(int)
            var_dict = {}
            
            if meas_type in ['sliced', 'accumulated', 'moving_window']:
                for ro_el in ro_element:
                    var_dict[f'I_pi_{ro_el}'] = declare(fixed, size = npts)
                    var_dict[f'I_nopi_{ro_el}'] = declare(fixed, size = npts)
                    var_dict[f'Q_pi_{ro_el}'] = declare(fixed, size = npts)
                    var_dict[f'Q_nopi_{ro_el}'] = declare(fixed, size = npts)
                i = declare(int)
                
            elif meas_type == 'full':
                for ro_el in ro_element:
                    var_dict[f'I_pi_{ro_el}'] = declare(fixed)
                    var_dict[f'I_nopi_{ro_el}'] = declare(fixed)
                    var_dict[f'Q_pi_{ro_el}'] = declare(fixed)
                    var_dict[f'Q_nopi_{ro_el}'] = declare(fixed)
                
            with for_(n, 0, n < N_avg, n + 1):
            
                if burst_delay>0:
                    wait(int(burst_delay//4), burst_element)
                    play(burst_pulse, burst_element)
                else:
                    play(burst_pulse, burst_element)
                    for ro_el in ro_element:
                        align(burst_element, ro_el)           
                    
                for ro_el, chunk_s in zip(ro_element, chunk_size):
                    if ramp_up > 0:
                        reset_phase(ro_el)
                        play(self.configObject.SuperMembers[ro_el].PulseParamsDict[ro_pulse].Additions.ramp['up']['name'], ro_el)
                        play(ro_pulse, ro_el, duration = int(ramp_up//4))
                    if results_dict['meas_type'] in ['sliced','moving_window']:
                        self.perform_sliced_measurement(var_dict[f'I_pi_{ro_el}'], var_dict[f'Q_pi_{ro_el}'], i, chunk_size = chunk_s, npts = npts, I_output_name = f'I_pi_{ro_el}', Q_output_name = f'Q_pi_{ro_el}', ro_element = ro_el, is_ramp_up = is_ramp_up_meas, is_save = False, is_reset_phase = is_reset_phase_meas)
                    elif results_dict['meas_type'] == 'accumulated':
                        self.perform_accumulated_measurement(var_dict[f'I_pi_{ro_el}'], var_dict[f'Q_pi_{ro_el}'], i, chunk_size = chunk_s, npts = npts, I_output_name = f'I_pi_{ro_el}', Q_output_name = f'Q_pi_{ro_el}', ro_element = ro_el, is_ramp_up = is_ramp_up_meas, is_save = False, is_reset_phase = is_reset_phase_meas)
                    # elif results_dict['meas_type'] == 'moving_window': self.perform_moving_window_measurement(var_dict[f'I_pi_{ro_el}'], var_dict[f'Q_pi_{ro_el}'], i, chunk_size = chunk_s, chunks_per_window = window_size, npts = npts, I_output_name = f'I_pi_{ro_el}', Q_output_name = f'Q_pi_{ro_el}', ro_element = ro_el, is_ramp_up = is_ramp_up_meas, is_save = False, is_reset_phase = is_reset_phase_meas)
                
                if results_dict['meas_type'] == 'full':
                    ValueError('meas_type = full is not supported!')
                #     for ro_el in ro_element:
                #         if ramp_up > 0:
                #             play(self.configObject.SuperMembers[ro_el].PulseParamsDict[ro_pulse].Additions.ramp['up']['name'], ro_el)
                #             play(ro_pulse, ro_el, duration = int(ramp_up//4))
                #         self.perform_full_measurement(var_dict[f'I_pi_{ro_el}'], var_dict[f'Q_pi_{ro_el}'], I_output_name = f'I_pi_{ro_el}', Q_output_name = f'Q_pi_{ro_el}', ro_element = ro_el, is_ramp_up = is_ramp_up_meas, is_save = False, is_reset_phase = is_reset_phase_meas)
                
                for ro_el in ro_element:
                    with for_(i, 0, i < npts, i+1):
                        save(var_dict[f'I_pi_{ro_el}'][i], f'I_pi_{ro_el}')
                        save(var_dict[f'Q_pi_{ro_el}'][i], f'Q_pi_{ro_el}')
                    
                for ro_el in ro_element:
                    align(burst_element, ro_el) 
                for ro_el, chunk_s in zip(ro_element, chunk_size):
                    if ramp_up > 0:
                        reset_phase(ro_el)
                        play(self.configObject.SuperMembers[ro_el].PulseParamsDict[ro_pulse].Additions.ramp['up']['name'], ro_el)
                        play(ro_pulse, ro_el, duration = int(ramp_up//4))
                    if results_dict['meas_type'] in ['sliced','moving_window']: self.perform_sliced_measurement(var_dict[f'I_nopi_{ro_el}'], var_dict[f'Q_nopi_{ro_el}'], i, chunk_size = chunk_s, npts = npts, I_output_name = f'I_nopi_{ro_el}', Q_output_name = f'Q_nopi_{ro_el}', ro_element = ro_el, is_ramp_up = is_ramp_up_meas, is_save = False, is_reset_phase = is_reset_phase_meas)
                    elif results_dict['meas_type'] == 'accumulated': self.perform_accumulated_measurement(var_dict[f'I_nopi_{ro_el}'], var_dict[f'Q_nopi_{ro_el}'], i, chunk_size = chunk_s, npts = npts,I_output_name = f'I_nopi_{ro_el}', Q_output_name = f'Q_nopi_{ro_el}', ro_element = ro_el,  is_ramp_up = is_ramp_up_meas, is_save = False, is_reset_phase = is_reset_phase_meas)
                    # elif results_dict['meas_type'] == 'moving_window': self.perform_moving_window_measurement(var_dict[f'I_nopi_{ro_el}'], var_dict[f'Q_nopi_{ro_el}'], i, chunk_size = chunk_s, chunks_per_window = window_size, npts = npts,I_output_name = f'I_nopi_{ro_el}', Q_output_name = f'Q_nopi_{ro_el}', ro_element = ro_el,  is_ramp_up = is_ramp_up_meas, is_save = False, is_reset_phase = is_reset_phase_meas)
                
                if results_dict['meas_type'] == 'full':
                    ValueError('meas_type = full is not supported!')
                #     for ro_el in ro_element:
                #         if ramp_up > 0:
                #             play(self.configObject.SuperMembers[ro_el].PulseParamsDict[ro_pulse].Additions.ramp['up']['name'], ro_el)
                #             play(ro_pulse, ro_el, duration = int(ramp_up//4))
                #         self.perform_full_measurement(var_dict[f'I_nopi_{ro_el}'], var_dict[f'Q_nopi_{ro_el}'], I_output_name = f'I_nopi_{ro_el}', Q_output_name = f'Q_nopi_{ro_el}', ro_element = ro_el,  is_ramp_up = is_ramp_up_meas, is_save = False, is_reset_phase = is_reset_phase_meas)
                    
                for ro_el in ro_element:
                    with for_(i, 0, i < npts, i+1):
                        save(var_dict[f'I_nopi_{ro_el}'][i], f'I_nopi_{ro_el}')
                        save(var_dict[f'Q_nopi_{ro_el}'][i], f'Q_nopi_{ro_el}')
                        
        self.last_prog = self.burst_no_burst_prog  
        
        
    def run_burst_no_burst(self, is_save_data = None, num_of_bins = 300, 
                           is_plot = True, is_find_burst = None,
                           **kwargs): 
        
        if is_save_data is None: is_save_data = self.is_save_data
        
        results_dict = self.burst_no_burst_results
        ro_element = results_dict['ro_element']
        meas_type = results_dict['meas_type']
        npts = results_dict['npts']
        N_avg = results_dict['N_avg']
        res_name = results_dict['res_name']
        if type(res_name)!=list:
            res_name = [res_name]
        
        if not hasattr(self, 'burst_no_burst_prog'):
            raise ValueError("No burst_no_burst program defined!")

        self.qm_server.clear_all_job_results()            
        job = self.qm.execute(self.burst_no_burst_prog, duration_limit=0, data_limit=0)
        job.result_handles.wait_for_all_values()
        self.last_job = job
        
        for ro_el, chunk_size in zip(ro_element, results_dict['chunk_size']):
            I_pi = job.result_handles.get(f'I_pi_{ro_el}').fetch_all()['value'].reshape((N_avg, -1))
            I_nopi = job.result_handles.get(f'I_nopi_{ro_el}').fetch_all()['value'].reshape(( N_avg, -1))
            Q_pi = job.result_handles.get(f'Q_pi_{ro_el}').fetch_all()['value'].reshape(( N_avg, -1))
            Q_nopi = job.result_handles.get(f'Q_nopi_{ro_el}').fetch_all()['value'].reshape(( N_avg, -1))
            
            if meas_type in ['sliced','accumulated','moving_window']:
                time_since_pulse = np.linspace(chunk_size * 4, chunk_size * 4 * npts , npts)
                results_dict['time'] = time_since_pulse
                
            results_dict[f'I_pi_{ro_el}'] = I_pi
            results_dict[f'Q_pi_{ro_el}'] = Q_pi
            results_dict[f'I_nopi_{ro_el}'] = I_nopi
            results_dict[f'Q_nopi_{ro_el}'] = Q_nopi
                
            if meas_type in ['sliced','accumulated','moving_window']:
                hist_pi,_ = self.process_data([I_pi.sum(1), Q_pi.sum(1)], is_mean = False, is_thresh = False, is_calc_stat_error = False)
                hist_nopi,_ = self.process_data([I_nopi.sum(1), Q_nopi.sum(1)], is_mean = False, is_thresh = False, is_calc_stat_error = False)
                
                fdlty, avg_ro = histogram_fidelity(hist_pi, hist_nopi)
                print('The readout fidelity is {}'.format(fdlty))
                
                if is_save_data:
                    time_of_meas_str = datetime.now().strftime("%d_%m_%Y___%H_%M_%S")
                    file_name = 'burst_no_burst_'+'_'.join(res_name)+'_'+time_of_meas_str
                    results_dict['file_name'] = file_name
                    self.pickle_save(results_dict, meas_name='burst_no_burst', filename=file_name, time_of_meas=time_of_meas_str)
                
                if is_plot:
                    self.plot_burst_no_burst(ro_element = ro_el, num_of_bins = num_of_bins,
                                             is_find_burst = is_find_burst, **kwargs)

            else:
                raise ValueError('meas_type must be sliced, accumulated or moving_window!')
                    
        
    def plot_burst_no_burst(self, ro_element, num_of_bins = 300, is_plot_fid = False, is_plot_dist = True,
                            is_find_burst = None, fid_ax = None, dist_ax = None, window_size = 20, meas_type = None,
                            n_consec = 100, is_correct_non_int_chunk = False, **kwargs):
        
        results_dict = self.burst_no_burst_results
        
        if meas_type == None: meas_type = results_dict['meas_type']
        
        if type(ro_element) == list:
            for ro_el in ro_element:
                self.plot_burst_no_burst(ro_el, is_plot_fid = is_plot_fid, is_plot_dist = is_plot_dist, fid_ax = fid_ax, dist_ax = dist_ax, window_size = window_size, n_consec = n_consec, **kwargs)
        else:
            I_pi = results_dict[f'I_pi_{ro_element}']
            Q_pi  = results_dict[f'Q_pi_{ro_element}']
            I_nopi = results_dict[f'I_nopi_{ro_element}']
            Q_nopi = results_dict[f'Q_nopi_{ro_element}'] 
            time = results_dict['time']
            npts = results_dict['npts']
            
            burst_element = results_dict['burst_element']
            burst_delay = results_dict['burst_delay']
            burst_pulse_len = results_dict['burst_pulse_len']
            ramp_up = results_dict['ramp_up']
            res_name = results_dict['res_name']
            ro_amp = results_dict['ro_pulse_amp']
            
            mean_I_pi = I_pi.mean(0)
            mean_I_nopi = I_nopi.mean(0)
            mean_Q_pi = Q_pi.mean(0)
            mean_Q_nopi = Q_nopi.mean(0)
            
            # if is_correct_non_int_chunk:
            #     IF_freq = self.element_IF(ro_element)
            #     mean_I_pi_corr = np.zeros(len(mean_I_pi))
            #     mean_Q_pi_corr = np.zeros(len(mean_Q_pi))
            #     mean_I_nopi_corr = np.zeros(len(mean_I_nopi))
            #     mean_Q_nopi_corr = np.zeros(len(mean_Q_nopi))
            #     for i,(ti,Ii,Qi) in enumerate(zip(time,mean_I_pi,mean_Q_pi)):
            #         mean_I_pi_corr[i], mean_Q_pi_corr[i] = correct_non_integer_demod(Ii,Qi,ti,4*results_dict['chunk_size'][0],IF_freq)
            #     for i,(ti,Ii,Qi) in enumerate(zip(time,mean_I_nopi,mean_Q_nopi)):
            #         mean_I_nopi_corr[i], mean_Q_nopi_corr[i] = correct_non_integer_demod(Ii,Qi,ti,4*results_dict['chunk_size'][0],IF_freq)
            #     mean_I_pi = mean_I_pi_corr.copy()
            #     mean_Q_pi = mean_Q_pi_corr.copy()
            #     mean_I_nopi = mean_I_nopi_corr.copy()
            #     mean_Q_nopi = mean_Q_nopi_corr.copy() 
                
                
            d_I = mean_I_pi.mean()-mean_I_nopi.mean()
            d_Q = mean_Q_pi.mean()-mean_Q_nopi.mean()
            d = np.sqrt(d_I**2+d_Q**2)
            v = (np.around(d_I/d,3), np.around(d_Q/d,3))
            phase = np.around(np.arctan2(v[1],v[0])*180/np.pi,3)
            print(f'Largest distance vector is {v} with phase of {phase} degrees')
            
            if meas_type in ['sliced','moving_window']:
                hist_pi,_ = self.process_data([I_pi.sum(1), Q_pi.sum(1)], is_mean = False, is_thresh = False, is_calc_stat_error = False)
                hist_nopi,_ = self.process_data([I_nopi.sum(1), Q_nopi.sum(1)], is_mean = False, is_thresh = False, is_calc_stat_error = False)
            elif meas_type == 'accumulated':
                hist_pi,_ = self.process_data([np.transpose(I_pi)[-1], np.transpose(Q_pi)[-1]], is_thresh = False, is_calc_stat_error = False, is_mean = False)
                hist_nopi,_ = self.process_data([np.transpose(I_nopi)[-1], np.transpose(Q_nopi)[-1]], is_thresh = False, is_calc_stat_error = False, is_mean = False)
                
            fid_list = np.zeros(npts)
            for i in range(npts):
                if meas_type == 'sliced':
                        hist_pi_i,_ = self.process_data([I_pi[:,:i].sum(1), Q_pi[:,:i].sum(1)], is_mean = False, is_thresh = False, is_calc_stat_error = False)
                        hist_nopi_i,_ = self.process_data([I_nopi[:,:i].sum(1), Q_nopi[:,:i].sum(1)], is_mean = False, is_thresh = False, is_calc_stat_error = False)
                        fid_list[i],_ = histogram_fidelity(hist_pi_i, hist_nopi_i)
                elif meas_type == 'moving_window':
                        hist_pi_i,_ = self.process_data([I_pi[:,max(i-window_size,0):i].sum(1), Q_pi[:,max(i-window_size,0):i].sum(1)], is_mean = False, is_thresh = False, is_calc_stat_error = False)
                        hist_nopi_i,_ = self.process_data([I_nopi[:,max(i-window_size,0):i].sum(1), Q_nopi[:,max(i-window_size,0):i].sum(1)], is_mean = False, is_thresh = False, is_calc_stat_error = False)
                        fid_list[i],_ = histogram_fidelity(hist_pi_i, hist_nopi_i)
            results_dict[f'fidelity_{ro_element}'] = fid_list    
            
            if is_plot_fid:
                if fid_ax is None:
                    _,fid_ax = plt.subplots()
                plt.sca(fid_ax)
                if meas_type == 'moving_window':
                    label = f'{ro_element}, W = {np.round(np.diff(time)[0]*window_size*1e-3,1)}'+r' $[\mu s]$'
                else:
                    label = f'{ro_element}, accumulated'
                plt.plot(time[1:]*1e-3,fid_list[1:], label = label)
                plt.xlabel(r"Time $[\mu s]$", fontsize=18)
                plt.ylabel("Readout Fidelity", fontsize=18)
                plt.fill_between([(burst_delay-ramp_up)*1e-3, (burst_delay-ramp_up+burst_pulse_len)*1e-3], 1,
                                 fc = (0.75,0.75,0.75,0.25), ec = (0.25,0.25,0.25,0.35), lw = 2)
                plt.ylim([0,1])
                plt.grid(1)
                leg = plt.legend(fontsize=10)
                leg.set_draggable(True)
                plt.tight_layout()

            I0 = I_nopi.mean(1).mean()
            Q0 = Q_nopi.mean(1).mean()
            I_dist = (I_pi-I0).mean(0)
            Q_dist = (Q_pi-Q0).mean(0)
            dist = np.sqrt(I_dist**2 + Q_dist**2)
            dist_thresh = np.sqrt((I_nopi-I0).mean(0)**2 + (Q_nopi-Q0).mean(0)**2).std() * 2
            dist_list = np.zeros(npts)
            burst_ind_list = []
            for i in np.arange(1,npts):
                if meas_type == 'sliced':
                    dist_list[i] = dist[:i].mean()
                    if dist_list[i] > dist_thresh: burst_ind_list = np.append(burst_ind_list,int(i))
                elif meas_type == 'moving_window':
                    dist_list[i] = dist[max(i-window_size,0):i].mean()
                    if dist_list[i] > dist_thresh*5: burst_ind_list = np.append(burst_ind_list,int(i))
            results_dict[f'IQ_dist_{ro_element}'] = dist_list    
            
            if is_find_burst:
                if dist_ax is None:
                    _,dist_ax = plt.subplots()
                plt.sca(dist_ax)
                burst_time, burst_time_err, brust_dist_level = self.tb_estimate(ro_element, window_size, n_consec, is_plot_dist)
                results_dict[f'burst_time_{ro_element}'] = burst_time  
                results_dict[f'burst_time_err_{ro_element}'] = burst_time_err  
            else:
                burst_time = None
                burst_time_err = None
                brust_dist_level = None

            if is_plot_dist:
                if meas_type == 'moving_window':
                    label = f'{res_name}, ro={np.round(abs(ro_amp[0]),3)}, W = {np.diff(time)[0]*window_size*1e-3}'+r'$\mu s$'
                else:
                    label = f'{res_name}'
                plt.plot(time[1:]*1e-3,dist_list[1:]*1e6, label = label)
                if burst_time:
                    plt.title(f'Burst Time {np.round(burst_time*1e-3,3)}'+r'$\pm$'+f'{np.round(burst_time_err*1e-3,4)}'+r'$\mu s$',fontsize=18)            
                    plt.errorbar(burst_time*1e-3,brust_dist_level*1e6,xerr=burst_time_err*1e-3, fmt='or', label=None, markersize=10)
                else:
                    plt.title('No automatic burst estimation',fontsize=18)            
                plt.xlabel(r"Time $[\mu s]$", fontsize=18)
                plt.ylabel(r"Distance in IQ plane $[\mu V]$", fontsize=18)
                plt.fill_between([(burst_delay-ramp_up)*1e-3, (burst_delay-ramp_up+burst_pulse_len)*1e-3], max(dist_list*1e6),
                                 fc = (0.75,0.75,0.75,0.25), ec = (0.25,0.25,0.25,0.35), lw = 2)
                plt.ylim([0,max(dist_list*1e6)])
                plt.grid(1)
                leg = plt.legend(fontsize=10)
                leg.set_draggable(True)
                plt.tight_layout()
                    
            
                prog_name = 'Burst no Burst'
                
                fig, axes = plt.subplots(2, 2, figsize = [12,8], num = next_fig_num_by_name(prog_name))
                
                plt.sca(axes[0,0])
                plt.errorbar(time, I_pi.mean(0)*1e6, yerr = sp.stats.sem(I_pi,0)*1e6, fmt = 'r.')
                plt.errorbar(time, I_nopi.mean(0)*1e6, yerr = sp.stats.sem(I_nopi,0)*1e6, fmt = 'b.')
                plt.ylabel(r"I $[\mu V]$", fontsize=18)
                plt.xlabel("Time [ns]", fontsize=18)
                
                plt.sca(axes[0,1])
                plt.errorbar(time, Q_pi.mean(0)*1e6, yerr = sp.stats.sem(Q_pi,0)*1e6, fmt = 'r.')
                plt.errorbar(time, Q_nopi.mean(0)*1e6, yerr = sp.stats.sem(Q_nopi,0)*1e6, fmt = 'b.')
                plt.ylabel(r"Q $[\mu V]$", fontsize=18)
                plt.xlabel("Time [ns]", fontsize=18)
                
                plt.sca(axes[1,0])
                plt.plot(0,0,'kx', markersize=6, alpha = 1)
                if meas_type == 'sliced':
                    plt.plot(I_pi.mean(axis=1)*1e6, Q_pi.mean(axis=1)*1e6, 'ro', markersize=1, alpha=0.5, zorder = 1)
                    plt.plot(I_nopi.mean(axis=1)*1e6, Q_nopi.mean(axis=1)*1e6, 'bo', markersize=1, alpha=0.5, zorder = 1)
                elif meas_type == 'accumulated':
                    plt.plot(I_pi[:,-1]*1e6, Q_pi[:,-1]*1e6, 'ro', markersize=1, alpha=0.5, zorder = 1)
                    plt.plot(I_nopi[:,-1]*1e6, Q_nopi[:,-1]*1e6, 'bo', markersize=1, alpha=0.5, zorder = 1)
                    
                plt.plot(I_pi.mean(0)*1e6, Q_pi.mean(0)*1e6, markersize = 6, color = 'r', markerfacecolor = 'r', marker = 'o', markeredgecolor = 'k')
                plt.plot(I_nopi.mean(0)*1e6, Q_nopi.mean(0)*1e6, markersize = 6, color = 'b', markerfacecolor = 'b', marker = 's', markeredgecolor = 'k')
                plt.xlabel(r"I $[\mu V]$", fontsize=18)
                plt.ylabel(r"Q $[\mu V]$", fontsize=18)
                axes[1,0].set_aspect(1)
        
                
                plt.sca(axes[1,1])
                
                scale = 1e6 if self.which_data != 'Phase' else 1
                
                bins = np.histogram(np.hstack((hist_pi*int(scale), hist_nopi*int(scale))), bins=num_of_bins)[1]
                
                hist_pi_data = plt.hist(hist_pi*scale,
                        bins = bins,
                        histtype='step',
                        color = 'r')
                hist_nopi_data = plt.hist(hist_nopi*scale,
                          bins = bins,
                          histtype='step',
                          color = 'b')
                plt.plot([np.mean(hist_pi)*scale,np.mean(hist_pi)*scale],[0,max([max(hist_pi_data[0]),max(hist_nopi_data[0])])], '--k')
                plt.plot([np.mean(hist_nopi)*scale,np.mean(hist_nopi)*scale],[0,max([max(hist_pi_data[0]),max(hist_nopi_data[0])])], '-k')
                
                plt.ylabel("counts", fontsize=18)
                xlabel = self.which_data
                if self.which_data !='Phase': xlabel += r' $[\mu V]$'
                plt.xlabel(xlabel, fontsize=18)
                
                
                fig.text(0.43, 0.95, "Burst", ha="center", va="bottom", size="large",color="red")
                fig.text(0.52, 0.95, "no Burst", ha="center", va="bottom", size="large",color="blue")
                if ro_element != 'ro':
                    fig.text(0.65, 0.95, "({},{},{})".format(burst_element, ro_element, res_name), ha="center", va="bottom", size="large",color="black")
                else:
                    fig.text(0.65, 0.95, "({})".format(results_dict['burst_element']), ha="center", va="bottom", size="large",color="black")
                plt.suptitle('',
                              fontsize=18)
                
                plt.tight_layout()
    
    def tb_estimate(self, ro_element, window_size=10, n_consec=20, is_plot_dist=None):
        
        results_dict = self.burst_no_burst_results
        time = results_dict['time']
        npts = results_dict['npts']
        burst_delay = results_dict['burst_delay']
        ramp_up = results_dict['ramp_up']
        dist_list = results_dict[f'IQ_dist_{ro_element}'] 
        
        d1 = dist_list[10:np.argwhere(time>burst_delay-ramp_up)[0][0]-10]
        dist_thresh = d1.mean()
        # dist_thresh = d1.max()
        
        burst_ind_list = []
        for i in np.arange(1,npts):
            if dist_list[i] > dist_thresh:
                burst_ind_list = np.append(burst_ind_list,int(i))
                
        burst_ind = None
        i_consec = 0
        for i in range(len(burst_ind_list)-1):
            if burst_ind_list[i+1] - burst_ind_list[i] == 1:
                i_consec += 1
            else:
                i_consec = 0
            if i_consec == n_consec:
                burst_ind = int(burst_ind_list[i-n_consec+1])
                break
            

        def tb_fit2(t,*p):
            tb,a,p1,p2 = p
            if t.size == 1:
                t = np.array([t])
            y = np.zeros(t.shape)
            for i,ti in enumerate(t):
                if ti<tb:
                    y[i] = a
                else:
                    y[i] = a+p1*(ti-tb)+p2*(ti-tb)**2
            return y
        
        if burst_ind:
            t1 = time[10:np.argwhere(time>burst_delay-ramp_up)[0][0]-10]
            t2 = time[burst_ind:burst_ind+n_consec]
            t = np.append(t1,t2)
            d1 = dist_list[10:np.argwhere(time>burst_delay-ramp_up)[0][0]-10]
            d2 = dist_list[burst_ind:burst_ind+n_consec]
            d = np.append(d1,d2)

            tb_0 = t2[0]
            a_0 = d1.mean()
            p2_0 = 0
            p1_0 = (d2[-1]-d2[0])/(t2[-1]-t2[0])
            p0 = [tb_0,a_0,p1_0,p2_0]
            popt, pcov = sp.optimize.curve_fit(tb_fit2,t,d,p0)
            
            burst_time = popt[0]
            burst_time_err =  abs(max(np.roots([popt[3],popt[2],-d1.std()])))
            brust_dist_level = popt[1]
            
            
            if burst_time < burst_delay-ramp_up+0.2e-3:
                
                dist_thresh = dist_list[np.argwhere(time>burst_delay-ramp_up)[0][0]+10 : np.argwhere(time>burst_delay-ramp_up)[0][0]+20].max()
                
                burst_ind_list = []
                for i in np.arange(1,npts):
                    if dist_list[i] > dist_thresh:
                        burst_ind_list = np.append(burst_ind_list,int(i))
                        
                burst_ind = None
                i_consec = 0
                for i in range(len(burst_ind_list)-1):
                    if burst_ind_list[i+1] - burst_ind_list[i] == 1:
                        i_consec += 1
                    else:
                        i_consec = 0
                    if i_consec == n_consec:
                        burst_ind = int(burst_ind_list[i-n_consec+1])
                        break
                    
                if burst_ind:
                    t1 = time[10:np.argwhere(time>burst_delay-ramp_up)[0][0]-10]
                    t2 = time[burst_ind:burst_ind+n_consec]
                    t = np.append(t1,t2)
                    d1 = dist_list[10:np.argwhere(time>burst_delay-ramp_up)[0][0]-10]
                    d2 = dist_list[burst_ind:burst_ind+n_consec]
                    d = np.append(d1,d2)

                    tb_0 = t2[0]
                    a_0 = d1.mean()
                    p2_0 = 0
                    p1_0 = (d2[-1]-d2[0])/(t2[-1]-t2[0])
                    p0 = [tb_0,a_0,p1_0,p2_0]
                    popt, pcov = sp.optimize.curve_fit(tb_fit2,t,d,p0)
                    
                    burst_time = popt[0]
                    burst_time_err =  abs(max(np.roots([popt[3],popt[2],-d1.std()])))
                    brust_dist_level = popt[1]
            
            if is_plot_dist:                    
                plt.plot(t1*1e-3, tb_fit2(t1,*popt)*1e6, '.-k')
                plt.plot(t2*1e-3, tb_fit2(t2,*popt)*1e6, '.-k')
            
        else:
            print(f'Failed to estimate tb! No {n_consec} consequtive points above threshold found.')
            burst_time = None
            burst_time_err = None
            brust_dist_level = None
        
        return burst_time, burst_time_err, brust_dist_level
    
        results_dict[f'burst_time_{ro_element}'] = burst_time  
        results_dict[f'burst_time_err_{ro_element}'] = burst_time_err   
            
        
    def load_qp_lifetime(self, wt_npts = 51, # How many points I measure
               max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
               N_avg = 1000, # How many times each sequence is executed 
               burst_element = None,
               burst_pulse = 'burst_pulse',
               cooling_pulse = 'cooling_pulse',
               ro_element = None,
               res_name = None,
               ro_pulse = 'ro_pulse',
               cooling_amp_start = 0,
               cooling_amp_stop = 0.2,
               cooling_amp_npts = 11,
               ramp_up = 1000,
               **kwargs):
        
        if burst_element is None: burst_element = self.burst_element
        if burst_pulse is None: burst_pulse = self.burst_pulse
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        
        max_seq_clks = int(max_seq_time // 4)
        step_size_clks = max_seq_clks // wt_npts
        # step_size_clks = max_seq_clks // (wt_npts-1)
        
        cooling_amp_list = np.linspace(cooling_amp_start, cooling_amp_stop, cooling_amp_npts)
        if cooling_amp_list.max()!=0: cooling_scale_list = cooling_amp_list/cooling_amp_list.max()
        else: cooling_scale_list = cooling_amp_list*0
        
        self.qp_lifetime_results = {}
        results_dict = self.qp_lifetime_results
        results_dict['time'] = np.arange(step_size_clks*4, max_seq_time + 1, step_size_clks*4)
        results_dict['N_avg'] = N_avg
        results_dict['wt_npts'] = wt_npts
        results_dict['res_name'] = res_name
        results_dict['ro_element'] = ro_element
        results_dict['ro_pulse'] = ro_pulse
        results_dict['ro_pulse_len'] = self.pulse_len(ro_element,ro_pulse) 
        results_dict['ro_pulse_amp'] = self.pulse_amp(ro_element,ro_pulse) 
        results_dict['ro_pulse_freq'] = self.element_freq(ro_element) 
        results_dict['ramp_up'] = ramp_up
        results_dict['burst_element'] = burst_element
        results_dict['burst_pulse'] = burst_pulse
        results_dict['burst_pulse_len'] = self.pulse_len(burst_element,burst_pulse)
        results_dict['burst_pulse_amp'] = self.pulse_amp(burst_element,burst_pulse)
        results_dict['burst_pulse_freq'] = self.element_freq(burst_element)
        results_dict['cooling_amp_list'] = cooling_amp_list
        results_dict['cooling_pulse'] = cooling_pulse
        results_dict['wait_between_seq'] = self.wait_between_seq
        
        run_time = N_avg*wt_npts * (max_seq_time/2 + self.pulse_len(ro_element, ro_pulse) + self.wait_between_seq * 4)
        print('Run time is {}s'.format(round(run_time * 1e-9))) 
        
        ro_length = self.pulse_len(ro_element, ro_pulse)
        extra_time = 1000
        
        
        with program() as self.qp_lifetime_prog:
            wt = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            cooling_scale = declare(fixed)
            
            play(ro_pulse, ro_element, duration = int(ramp_up//4))
            align(ro_element, burst_element)
            
            with for_each_(cooling_scale, cooling_scale_list):
                with for_(n, 0, n < N_avg, n + 1):
                    with for_(wt, step_size_clks, wt <= max_seq_clks, wt + step_size_clks):
                        
                        play(burst_pulse, burst_element)
                        align(ro_element, burst_element)
                        
                        # play(cooling_pulse * amp(cooling_scale), burst_element, duration = wt+int((ro_length + extra_time)//4))
                        # play('ramp_up' * amp(cooling_scale), burst_element)
                        # play(cooling_pulse * amp(cooling_scale), burst_element, duration = wt)
                        # play('ramp_down' * amp(cooling_scale), burst_element) #Pulse is not smooth
                        
                        play('ramp_up' , burst_element)
                        play(cooling_pulse, burst_element, duration = wt)
                        play('ramp_down', burst_element)
                        
                        align(ro_element, burst_element)
                        # wait(1000, ro_element)
                        self.perform_full_measurement(I,Q, I_output_name = 'I', Q_output_name = 'Q', ro_element=ro_element,
                                                      readout_pulse='0_pulse', is_ramp_up=False, is_reset_phase=False)
                        
                        # reference
                        play('ramp_up' , burst_element)
                        play(cooling_pulse, burst_element, duration = wt)
                        play('ramp_down', burst_element)
                        
                        self.perform_full_measurement(I,Q, I_output_name = 'I0', Q_output_name = 'Q0', ro_element=ro_element,
                                                      readout_pulse='0_pulse', is_ramp_up=False, is_reset_phase=False)

            ramp_to_zero(ro_element)
     
        self.last_prog = self.qp_lifetime_prog
    
    
    def run_qp_lifetime(self, is_save_data = None, is_plot = True,  **kwargs):
        
        results_dict = self.qp_lifetime_results
        time = results_dict['time']
        wt_npts = results_dict['wt_npts']
        N_avg = results_dict['N_avg']
        res_name = results_dict['res_name']
        ro_element = results_dict['ro_element']
        burst_element = results_dict['burst_element']
        cooling_pulse = results_dict['cooling_pulse']
        cooling_amp_list = results_dict['cooling_amp_list']
        
        cooling_amp_npts = len(cooling_amp_list)
                  
        if not hasattr(self,'qp_lifetime_prog'): raise ValueError("No qp_lifetime program defined!")

        configured_amp = self.pulse_amp(burst_element, cooling_pulse)
        max_amp = self.qp_lifetime_results['cooling_amp_list'].max()
        
        try:
            if self.sticky(ro_element) is None:
                self.sticky(ro_element, "analog")
                is_sticky_change = True
            else: is_sticky_change = False
            self.pulse_amp(burst_element, cooling_pulse, max_amp)
            
            # execute ptogram
            self.qm_server.clear_all_job_results()            
            job = self.qm.execute(self.qp_lifetime_prog, duration_limit=0, data_limit=0)
            job.result_handles.wait_for_all_values()
            job.execution_report()
            self.last_job = job
            
            #return to initial amp
            if is_sticky_change: self.sticky(ro_element, False, update = False)
            self.pulse_amp(burst_element, cooling_pulse, configured_amp)
        
        except:
            #return to initial amp
            if is_sticky_change: self.sticky(ro_element, False, update = False)
            self.pulse_amp(burst_element, cooling_pulse, configured_amp)
            return
            
        # extract data
        
        I = job.result_handles.get('I').fetch_all()['value'].reshape((cooling_amp_npts, N_avg, -1))
        Q = job.result_handles.get('Q').fetch_all()['value'].reshape((cooling_amp_npts, N_avg, -1))
        
        I_mean = np.zeros((cooling_amp_npts ,wt_npts))
        I_mean_err =  np.zeros((cooling_amp_npts, wt_npts))
        Q_mean = np.zeros((cooling_amp_npts, wt_npts))
        Q_mean_err =  np.zeros((cooling_amp_npts, wt_npts))
        for i in range(cooling_amp_npts):
            I_mean[i,:], I_mean_err[i,:] = self.process_data([I[i,:,:], Q[i,:,:]], which_data = 'I', **kwargs)
            Q_mean[i,:], Q_mean_err[i,:] = self.process_data([I[i,:,:], Q[i,:,:]], which_data = 'Q', **kwargs)
        
        results_dict['I'], results_dict['I_err'] = I_mean, I_mean_err
        results_dict['Q'], results_dict['Q_err'] = Q_mean, Q_mean_err
        
        
        I0 = job.result_handles.get('I0').fetch_all()['value'].reshape((cooling_amp_npts, N_avg, -1))
        Q0 = job.result_handles.get('Q0').fetch_all()['value'].reshape((cooling_amp_npts, N_avg, -1))
        
        I0_mean = np.zeros((cooling_amp_npts ,wt_npts))
        I0_mean_err =  np.zeros((cooling_amp_npts, wt_npts))
        Q0_mean = np.zeros((cooling_amp_npts, wt_npts))
        Q0_mean_err =  np.zeros((cooling_amp_npts, wt_npts))
        for i in range(cooling_amp_npts):
            I0_mean[i,:], I0_mean_err[i,:] = self.process_data([I0[i,:,:], Q0[i,:,:]], which_data = 'I', **kwargs)
            Q0_mean[i,:], Q0_mean_err[i,:] = self.process_data([I0[i,:,:], Q0[i,:,:]], which_data = 'Q', **kwargs)
        
        results_dict['I0'], results_dict['I0_err'] = I0_mean, I0_mean_err
        results_dict['Q0'], results_dict['Q0_err'] = Q0_mean, Q0_mean_err

        if is_save_data:
            time_of_meas_str = datetime.now().strftime("%d_%m_%Y___%H_%M_%S")
            file_name = 'qp_lifetime_'+res_name+'_'+time_of_meas_str
            results_dict['file_name'] = file_name
            self.pickle_save(results_dict, meas_name='qp_lifetime', filename=file_name, time_of_meas=time_of_meas_str)
    
        if is_plot: self.plot_qp_lifetime(is_plot_all = is_plot, **kwargs)
        
    
    def plot_qp_lifetime(self, is_plot_all = True, ax = None, **kwargs):
        
        results_dict = self.qp_lifetime_results
        I, I_err = results_dict['I'], results_dict['I_err'] 
        Q, Q_err = results_dict['Q'], results_dict['Q_err'] 
        I0, I0_err = results_dict['I0'], results_dict['I0_err'] 
        Q0, Q0_err = results_dict['Q0'], results_dict['Q0_err'] 
        cooling_amp_list = results_dict['cooling_amp_list']
        time = results_dict['time']
        
        dist = np.sqrt( (I-I0)**2 + (Q-Q0)**2 )
        dist_err = np.sqrt( (I-I0)**2 * (I_err**2 + I0_err**2) + (Q-Q0)**2 * (Q_err**2 + Q0_err**2) ) / dist
        
       
        def Exp(t, A, gamma, offset):
            return (A*np.exp(-t*gamma) + offset - A/2).astype(np.float64)
            
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            plt.sca(ax)
            
        decays = []
        decays_err = []
        time_fit_plot = np.linspace(time[0], time[-1], 1001)
        for i in range(len(cooling_amp_list)): 
            Amp = np.abs(max(dist[i])-min(dist[i]))
            Offset = dist[i].mean(0)
            Gamma = 0
            p0 = [Amp, Offset, Gamma]
            try:
                popt, pcov = sp.optimize.curve_fit(Exp, time, dist[i], sigma = dist_err[i], p0 = p0)
            except:
                print(f"Could not find fit (cooling amp = {np.round(cooling_amp_list[i],3)} V")
                print('\n')
                popt, pcov = [np.nan]*3, [np.nan]*3
            
            decay, decay_err = 1e-3/popt[1], np.sqrt(pcov[1])*1e-3/popt[1]**2
            
            decays.append(decay if decay is not None else np.nan)
            decays_err.append(decay_err if decay_err is not None else np.nan)
            p = plt.errorbar(time*1e-6, dist[i]*1e6, yerr = dist_err[i]*1e6, fmt = '.-', label = f"Cooling amp = {np.round(cooling_amp_list[i],3)} V")
            plt.plot(time_fit_plot*1e-6, Exp(time_fit_plot, *popt)*1e6, '--', color = p[0].get_color())
        decays = np.array(decays)
        decays_err = np.array(decays_err)
        leg = plt.legend(fontsize = 14)
        leg.set_draggable(True)
        plt.xlabel('Time [ms]')
        plt.ylabel('IQ-plane distance '+r'$[\mu V]$')
        plt.tight_layout()
        
        results_dict['decays'] = decays
        results_dict['decays_err'] = decays_err
        
            
    # def load_burst_detector(self, meas_type,
    #                         npts = 101,
    #                         N_tries = 100,
    #                         npls = 1,
    #                         ro_element = None,
    #                         ro_pulse = None,
    #                         ring_up_time = 2e3,
    #                         stitching_time = 500):
        
    #     if ro_element is None: ro_element = self.main_readout
    #     if ro_pulse is None: ro_pulse = self.ro_pulse
        
    #     self.burst_detector_compiled_program_id = None
        
    #     if type(ro_element)!=list:
    #         ro_element = [ro_element]
        
        
    #     run_time = N_tries * (3 * self.pulse_len(ro_element[0], ro_pulse) + ring_up_time) * 1e-9
    #     print(f'\nRun time is {np.around(run_time)} s')
        
    #     self.burst_detector_results = {}
    #     results_dict = self.burst_detector_results
    #     results_dict['N_tries'] = N_tries
    #     results_dict['npts'] = npts
    #     results_dict['meas_type'] = meas_type
    #     results_dict['ro_element'] = ro_element
    #     results_dict['N_tries'] = N_tries
    #     results_dict['npls'] = npls
    #     results_dict['ring_up_time'] = ring_up_time
        
        
    #     ro_pulse_len = self.config['pulses'][self.config['elements'][ro_element[0]]['operations'][ro_pulse]]['length']
    #     if meas_type in ['sliced', 'accumulated', 'moving_window']: 
    #         chunk_size =  [self.config['pulses'][self.config['elements'][ro_el]['operations'][ro_pulse]]['length'] // npts // 4 for ro_el in ro_element]
    #         results_dict['chunk_size'] = chunk_size
    #     else:
    #         raise ValueError('meas_type must be either sliced, accumulated or moving_window, but is neither!')
    #         results_dict['chunk_size'] = ro_element
            
    #     results_dict['time'] = np.linspace(chunk_size[0] * 4, chunk_size[0] * 4 * npts * 3 , npts*3)
            
        
    #     with program() as self.burst_detector_prog:
    #         n = declare(int)
    #         var_dict = {}
    #         for ro_el in ro_element:
    #             for p in range(npls):
    #                 var_dict[f'I_{ro_el}_pls{p}'] = declare(fixed, size = npts)
    #                 var_dict[f'Q_{ro_el}_pls{p}'] = declare(fixed, size = npts)
    #             var_dict[f'I_{ro_el}_copy'] = declare(fixed, size = npts)
    #             var_dict[f'Q_{ro_el}_copy'] = declare(fixed, size = npts)
    #         i = declare(int)
            
    #         # with strict_timing_():
    #         for ro_el in ro_element:
    #             reset_phase(ro_el)
    #             reset_phase(ro_el + '_copy')
    #         if ring_up_time>0:
    #             for ro_el in ro_element:
    #                 play(ro_pulse, ro_el, duration = int(ring_up_time//4))
    #         with for_(n, 0, n < N_tries, n + 1):
    #             for ro_el, chunk_s in zip(ro_element, chunk_size):
    #                 for p in range(npls):
    #                     if meas_type in ['sliced','moving_window']:
    #                         self.perform_sliced_measurement(var_dict[f'I_{ro_el}_pls{p}'], var_dict[f'Q_{ro_el}_pls{p}'], i, chunk_size = chunk_s, npts = npts, I_output_name = f'I_{ro_el}_pls{p}', Q_output_name = f'Q_{ro_el}_pls{p}', ro_element = ro_el, is_ramp_up = False, is_save = False, is_reset_phase = False, is_wait = False, is_ramp_down = False)
    #                         for i in range(npts):
    #                             for p in range(npls):
    #                                 save(var_dict[f'I_{ro_el}_pls{p}'][i], f'I_{ro_el}_pls{p}')
    #                                 save(var_dict[f'Q_{ro_el}_pls{p}'][i], f'Q_{ro_el}_pls{p}')
    #                         wait(ro_pulse_len//4, ro_el + '_copy')
    #                         self.perform_sliced_measurement(var_dict[f'I_{ro_el}_copy'], var_dict[f'Q_{ro_el}_copy'], i, chunk_size = chunk_s, npts = npts, I_output_name = f'I_{ro_el}_pls{p}', Q_output_name = f'Q_{ro_el}_pls{p}', ro_element = ro_el + '_copy', is_ramp_up = False, is_save = False, is_reset_phase = False, is_wait = False, is_ramp_down = False)
    #                         for i in range(npts):
    #                             save(var_dict[f'I_{ro_el}_copy'][i], f'I_{ro_el}_pls{0}')
    #                             save(var_dict[f'Q_{ro_el}_copy'][i], f'Q_{ro_el}_pls{0}')
    #                         wait(ro_pulse_len//4, ro_el)
    #                         self.perform_sliced_measurement(var_dict[f'I_{ro_el}_pls{p}'], var_dict[f'Q_{ro_el}_pls{p}'], i, chunk_size = chunk_s, npts = npts, I_output_name = f'I_{ro_el}_pls{p}', Q_output_name = f'Q_{ro_el}_pls{p}', ro_element = ro_el, is_ramp_up = False, is_save = False, is_reset_phase = False, is_wait = False, is_ramp_down = False)
    #                         wait(ro_pulse_len//4, ro_el + '_copy')
    #                         play('ro_pulse', ro_el + '_copy', duration = stitching_time//4)
    #             # with for_(i, 0, i < npts, i+1):
    #             for i in range(npts):
    #                 for ro_el in ro_element:
    #                     for p in range(npls):
    #                         save(var_dict[f'I_{ro_el}_pls{p}'][i], f'I_{ro_el}_pls{p}')
    #                         save(var_dict[f'Q_{ro_el}_pls{p}'][i], f'Q_{ro_el}_pls{p}')

                    
                        
    #     self.last_prog = self.burst_detector_prog  
    #     t_start = time.time()
    #     self.burst_detector_compiled_program_id = self.qm.compile(self.burst_detector_prog)
    #     t_stop = time.time()
    #     print('\nCompilation time = ',np.round(t_stop-t_start,2),'s\n')
        
        
        
        
    # def run_burst_detector(self, is_save_data = None):
    #     if is_save_data is None: is_save_data = self.is_save_data
        
    #     npls = self.burst_detector_results['npls']
    #     N_tries = self.burst_detector_results['N_tries']
    #     ro_element = self.burst_detector_results['ro_element']
        
    #     self.qm_server.clear_all_job_results()  
    #     pending_job = self.qm.queue.add_compiled(self.burst_detector_compiled_program_id)
        
    #     t_start = time.time()
    #     job = pending_job.wait_for_execution()
    #     t_stop = time.time()
    #     print('\nWait for execution time = ',np.round(t_stop-t_start,2),'s\n')
    #     res = job.result_handles
    #     t_start = time.time()
    #     res.wait_for_all_values()
    #     t_stop = time.time()
    #     print('\nWait for all values time = ',np.round(t_stop-t_start,2),'s\n')
        
    #     for ro_el in ro_element:
    #         I = np.array([])
    #         Q = np.array([])
    #         for p in range(npls):
    #             I = np.append(I, res.get(f'I_{ro_el}_pls{p}').fetch_all(flat_struct=True))
    #             Q = np.append(Q, res.get(f'Q_{ro_el}_pls{p}').fetch_all(flat_struct=True))
            
    #         I = I.reshape((npls,N_tries,-1)).transpose((1,0,2)).reshape((N_tries,-1))
    #         Q = Q.reshape((npls,N_tries,-1)).transpose((1,0,2)).reshape((N_tries,-1))
    #         self.burst_detector_results[f'I_{ro_el}'] = I
    #         self.burst_detector_results[f'Q_{ro_el}'] = Q
        
    #     if is_save_data: self.pickle_save(self.burst_detector_results, meas_name = 'burst_detector')
        
        
    # def plot_burst_detector(self, I_ref=None, Q_ref=None, thresh=None, window_size=11, is_plot=True,
    #                         is_legend=False, fontsize=15, **kwargs):
        
    #     ro_element = self.burst_detector_results['ro_element']
    #     time = self.burst_detector_results['time']
    #     N_tries = self.burst_detector_results['N_tries']
        
    #     if I_ref == None or Q_ref == None:
    #         # is_legend = False
    #         I_ref = {}
    #         Q_ref = {}
    #         for ro_el in ro_element:
    #             I_ref[ro_el] = None
    #             Q_ref[ro_el] = None
        
    #     if I_ref == 0 and Q_ref == 0:
    #         is_legend = False
            
    #     if is_plot:
    #         fig, axs = plt.subplots(len(ro_element), 1, sharex=True, figsize=(11,8))
    #         fig.subplots_adjust(wspace=0, hspace=0.1)
    #         if type(axs) not in [list, np.ndarray]: axs=[axs]
    #         window_time = np.round(window_size*(time[1]-time[0]),0)
    #         plt.suptitle(f'window = {window_time}ns {ro_element}', fontsize=fontsize+4)
    #         plt.xlabel(r'Time $[\mu s]$', fontsize=fontsize)
    #         for ro_el, ax in zip(ro_element, axs):
    #             plt.sca(ax)
    #             plt.grid()
    #             plt.xticks(fontsize=fontsize)
    #             plt.yticks(fontsize=fontsize)
    #         fig.text(0.05, 0.35, r"Distance in IQ plane $[\mu V]$", fontsize=fontsize, ha='center', rotation='vertical')
        
    #     if not window_size%2:
    #         print('Received even window_size (must be odd)! Increasing by 1')
    #         window_size += 1 
    #     self.burst_detector_results['window_size'] = window_size

    #     if type(thresh) not in [list, np.ndarray]: thresh=[thresh] * len(ro_element)
        
    #     burst_indices = []
    #     for ro_el, th, ax in zip(ro_element, thresh, axs):
    #         if is_plot: plt.sca(ax)
            
    #         I = self.burst_detector_results[f'I_{ro_el}']
    #         Q = self.burst_detector_results[f'Q_{ro_el}']
    #         filt = sp.signal.windows.hann(window_size)
    #         # filt = sp.signal.windows.boxcar(window_size)
    #         if I_ref[ro_el] is None or Q_ref[ro_el] is None:
    #             I0 = 0
    #             Q0 = 0
    #             # th = 0
    #         else:
    #             I0 = np.tile(I_ref[ro_el].mean(0),(N_tries,1))
    #             Q0 = np.tile(Q_ref[ro_el].mean(0),(N_tries,1))
    #             if th is None:
    #                 th = np.sqrt(I_ref[ro_el].std()**2 + Q_ref[ro_el].std()**2) * 5

    #         dist = np.sqrt((I-I0)**2 + (Q-Q0)**2)
            
    #         for j, dist_try in enumerate(dist):
    #             dist_filtered = sp.signal.convolve(dist_try, filt, mode='same')/np.sum(filt)
    #             dist_filtered = dist_filtered[window_size//2:-window_size//2]
    #             time_cut = time[window_size//2:-window_size//2]
    #             for i in range(len(dist_filtered)):
    #                 if dist_filtered[i] > th:
    #                     burst_indices.append(j)
    #                     if is_plot:
    #                         plt.plot(time_cut*1e-3, dist_filtered*1e6, label = j)
    #                     break

    #         if is_plot:
    #             if is_legend: 
    #                 leg = plt.legend(fontsize=fontsize-4)
    #                 leg.set_draggable(True)
        
    #     self.burst_detector_results['burst_indices'] = burst_indices
        
        
    def plot_burst_detector_segment(self, ind1, ind2, I_ref=None, Q_ref=None, window_size=11, fontsize=15, is_shifted=False):
        
        ro_element = self.burst_detector_results['ro_element']
        time = self.burst_detector_results['time']
        ring_up_time = self.burst_detector_results['ring_up_time']
        N_tries = self.burst_detector_results['N_tries']
        
        if I_ref == None or Q_ref == None:
            I_ref = {}
            Q_ref = {}
            for ro_el in ro_element:
                I_ref[ro_el] = None
                Q_ref[ro_el] = None
        
        if not window_size%2:
            print('Received even window_size (must be odd)! Increasing by 1')
            window_size += 1 
        
        plt.figure(figsize=(12,7))
        window_time = np.round(window_size*(time[1]-time[0]),0)
        plt.xlabel('Time [ms]', fontsize=fontsize)
        plt.ylabel(r"Distance in IQ plane $[\mu V]$", fontsize=fontsize)
        plt.title(f'window = {window_time}ns', fontsize=fontsize+4)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid()
        
        data_segment = {ro_el: [] for ro_el in ro_element}
        
        for ro_el in ro_element:
            I = self.burst_detector_results[f'I_{ro_el}']
            Q = self.burst_detector_results[f'Q_{ro_el}']
            filt = sp.signal.windows.hann(window_size)
            # filt = sp.signal.windows.boxcar(window_size)
            
            if I_ref[ro_el] is None or Q_ref[ro_el] is None:
                I0 = 0
                Q0 = 0
            else:
                I0 = np.tile(I_ref[ro_el].mean(0),(N_tries,1))
                Q0 = np.tile(Q_ref[ro_el].mean(0),(N_tries,1))

            dist = np.sqrt((I-I0)**2 + (Q-Q0)**2)
            
            data_segment[ro_el] = sp.signal.convolve(dist[ind1], filt, mode='same')/np.sum(filt)
            data_segment[ro_el] = data_segment[ro_el][window_size//2:-window_size//2]
            time_segment = time[window_size//2:-window_size//2]
            for i in np.arange(ind1+1,ind2):
                data_filtered = sp.signal.convolve(dist[i], filt, mode='same')/np.sum(filt)
                data_filtered = data_filtered[window_size//2:-window_size//2]
                data_segment[ro_el] = np.append(data_segment[ro_el], data_filtered)
                time_shift = time_segment[-1]+time[window_size]-time[1]+ring_up_time
                time_segment = np.append(time_segment, time[window_size//2:-window_size//2] + time_shift)
            if is_shifted:
                data_segment[ro_el] -= data_segment[ro_el][0]
            plt.plot(time_segment*1e-6, data_segment[ro_el]*1e6, 'o', label=ro_el)
                
        leg = plt.legend()
        leg.set_draggable(True)
        plt.tight_layout()
        
        return time_segment, data_segment
        
    
    def load_burst_detector_QM(self,
                            N_tries = 1000,
                            npts = 50,
                            npls = 100,
                            ro_element = None,
                            ro_pulse = 'ro_pulse',
                            zero_pulse = '0_pulse',
                            I_burst_thresh = 0,
                            thresh_type = 'both',
                            ramp_up = 500e3,
                            is_apply_burst = False,
                            burst_element = None,
                            burst_pulse = 'burst_pulse',
                            burst_delay = 520e3):
        
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        if burst_element is None: burst_element = self.burst_element

        
        if (npls % 2) == 1:
            npls += 1
            print('supplied npls is odd! increasing by 1 to make it even')
        if (N_tries % 2) == 1:
            N_tries += 1
            print('supplied N_tries is odd! increasing by 1 to make it even')
            
        self.burst_detector_compiled_program_id = None
        
        if type(ro_element)!=list:
            ro_element = [ro_element]
        if type(I_burst_thresh)!=list:
            I_burst_thresh = [I_burst_thresh]*len(ro_element)
        if type(burst_delay)!=list:
            burst_delay = [burst_delay]

        ro_pulse_len = self.pulse_len(ro_element[0], ro_pulse)
        chunk_size = ro_pulse_len // npts // 4
        
        run_time = N_tries * ro_pulse_len * 1e-9
        print(f'\nRun time is {np.around(run_time)} s')
        
        self.burst_detector_QM_results = {}
        results_dict = self.burst_detector_QM_results
        results_dict['N_tries'] = N_tries
        results_dict['npts'] = npts
        results_dict['npls'] = npls
        results_dict['ro_element'] = ro_element
        results_dict['ro_pulse'] = ro_pulse
        results_dict['ro_pulse_len'] = ro_pulse_len
        results_dict['chunk_size'] = chunk_size
        results_dict['zero_pulse'] = zero_pulse
        results_dict['I_burst_thresh'] = I_burst_thresh
        results_dict['thresh_type'] = thresh_type
        results_dict['ramp_up'] = ramp_up
        results_dict['burst_element'] = burst_element
        results_dict['burst_pulse'] = burst_pulse
        results_dict['burst_delay'] = burst_delay
        
        with program() as self.burst_detector_QM_prog:
            n = declare(int)
            n_copy = declare(int)
            i = declare(int)
            i_copy = declare(int)
            is_burst = declare(bool, value = False)
            is_burst_copy = declare(bool, value = False)
            npls_i = declare(int, value = 0)
            npls_i_copy = declare(int, value = 0)
            # crit = declare(fixed, size = 3)
            # crit_copy = declare(fixed, size = 3)
            
            var_dict = {}
            for ro_el in ro_element:
                var_dict[f'I_{ro_el}'] = declare(fixed, size = npts)
                var_dict[f'Q_{ro_el}'] = declare(fixed, size = npts)
                var_dict[f'I_{ro_el}_copy'] = declare(fixed, size = npts)
                var_dict[f'Q_{ro_el}_copy'] = declare(fixed, size = npts)
                var_dict[f'I_stream_{ro_el}'] = declare_stream()
                var_dict[f'Q_stream_{ro_el}'] = declare_stream()
                var_dict[f'I_stream_{ro_el}_copy'] = declare_stream()
                var_dict[f'Q_stream_{ro_el}_copy'] = declare_stream()
                
            n_burst_stream = declare_stream()
            n_burst_stream_copy = declare_stream()
            
            
            if (is_apply_burst):
                for b_delay in burst_delay:
                    wait(int(b_delay / 4), burst_element)
                    play(burst_pulse, burst_element)
                
            for ro_el in ro_element:
                reset_phase(ro_el)
                reset_phase(ro_el+'_copy')
                play(ro_pulse, ro_el+'_copy')
                align(ro_el,ro_el+'_copy') # why is this nessasery?
                wait(int(ro_pulse_len / 6), ro_el+'_copy') # denominator 6 (instead of 4) is for overlap
            
                wait(int(ramp_up / 4), ro_el)
                wait(int(ramp_up / 4), ro_el+'_copy')
                    
            with for_(n, 0, n < N_tries//2, n + 1):
                
                for ro_el in ro_element:
                    measure(zero_pulse, ro_el, None,
                            demod.sliced("integ_w_I", var_dict[f'I_{ro_el}'], chunk_size, "out1"),
                            demod.sliced("integ_w_Q", var_dict[f'Q_{ro_el}'], chunk_size, "out2"),
                            # dual_demod.full("integ_w_I", "out1", "integ_w_Q", "out2", var_dict[f'I_{ro_el}']),
                            # dual_demod.full("integ_w_minusQ", "out1", "integ_w_I", "out2", var_dict[f'Q_{ro_el}']),
                            timestamp_stream = f'timestamps_{ro_el}')
                    
                with if_(~is_burst):
                    for ro_el, I_th in zip(ro_element, I_burst_thresh):
                        if thresh_type == 'below':
                            with if_(var_dict[f'I_{ro_el}'][-1] - I_th < 0): 
                                assign(is_burst, True)
                        elif thresh_type == 'above':
                            with if_(var_dict[f'I_{ro_el}'][-1] - I_th > 0): 
                                assign(is_burst, True)
                        elif thresh_type == 'both':
                            with if_(  ( (var_dict[f'I_{ro_el}'][npts-1] - I_th[0] < 0) | (var_dict[f'I_{ro_el}'][npts-1] - I_th[1] > 0) )
                                     & ( (var_dict[f'I_{ro_el}'][npts-2] - I_th[0] < 0) | (var_dict[f'I_{ro_el}'][npts-2] - I_th[1] > 0) )
                                     & ( (var_dict[f'I_{ro_el}'][npts-3] - I_th[0] < 0) | (var_dict[f'I_{ro_el}'][npts-3] - I_th[1] > 0) ) ): 
                                assign(is_burst, True)
                                    
                # with if_(~is_burst):
                #     for ro_el in ro_element:
                #         wait(200, ro_el)
                    
                with if_(is_burst):
                    for ro_el in ro_element:
                        with for_(i, 0, i<npts , i+1):
                            save(var_dict[f'I_{ro_el}'][i], var_dict[f'I_stream_{ro_el}'])
                            save(var_dict[f'Q_{ro_el}'][i], var_dict[f'Q_stream_{ro_el}'])
                    save(n, n_burst_stream)
                    assign(npls_i,  npls_i+1)
                    with if_(npls_i > npls//2-1):
                        assign(is_burst, False)
                        assign(npls_i, 0)
                            
            with for_(n_copy, 0, n_copy < N_tries//2, n_copy + 1):
                for ro_el in ro_element:
                    measure(zero_pulse, ro_el+'_copy', None,
                            demod.sliced("integ_w_I", var_dict[f'I_{ro_el}_copy'], chunk_size, "out1"),
                            demod.sliced("integ_w_Q", var_dict[f'Q_{ro_el}_copy'], chunk_size, "out2"),
                            # dual_demod.full("integ_w_I", "out1", "integ_w_Q", "out2", var_dict[f'I_{ro_el}_copy']),
                            # demod.sliced("integ_w_minusQ", "out1", "integ_w_I", "out2", var_dict[f'Q_{ro_el}_copy']),
                            timestamp_stream = f'timestamps_{ro_el}_copy')
                    
                with if_(~is_burst_copy):
                    for ro_el, I_th in zip(ro_element, I_burst_thresh):
                        if thresh_type == 'below':
                            with if_(var_dict[f'I_{ro_el}_copy'][-1] - I_th < 0):
                                assign(is_burst_copy, True)
                        elif thresh_type == 'above': 
                            with if_(var_dict[f'I_{ro_el}_copy'][-1] - I_th > 0):
                                assign(is_burst_copy, True)
                        elif thresh_type == 'both':
                            with if_(  ( (var_dict[f'I_{ro_el}_copy'][npts-1] - I_th[0] < 0) | (var_dict[f'I_{ro_el}_copy'][npts-1] - I_th[1] > 0) )
                                     & ( (var_dict[f'I_{ro_el}_copy'][npts-2] - I_th[0] < 0) | (var_dict[f'I_{ro_el}_copy'][npts-2] - I_th[1] > 0) )
                                     & ( (var_dict[f'I_{ro_el}_copy'][npts-3] - I_th[0] < 0) | (var_dict[f'I_{ro_el}_copy'][npts-3] - I_th[1] > 0) ) ):
                                assign(is_burst_copy, True)

                        
                with if_(is_burst_copy):
                    for ro_el in ro_element:
                        with for_(i_copy, 0, i_copy<npts , i_copy+1):
                            save(var_dict[f'I_{ro_el}_copy'][i_copy], var_dict[f'I_stream_{ro_el}_copy'])
                            save(var_dict[f'Q_{ro_el}_copy'][i_copy], var_dict[f'Q_stream_{ro_el}_copy'])
                    save(n_copy, n_burst_stream_copy)
                    assign(npls_i_copy,  npls_i_copy+1)
                    with if_(npls_i_copy > npls//2-1):
                        assign(is_burst_copy, False)
                        assign(npls_i_copy, 0)
                            
            align(ro_el+'_copy', ro_el) # why is this line nessasery?
            ramp_to_zero(ro_el+'_copy')

            with stream_processing():
                for ro_el in ro_element:
                #     var_dict[f'I_stream_{ro_el}'].save(f'I_{ro_el}')
                #     var_dict[f'Q_stream_{ro_el}'].save(f'Q_{ro_el}')
                #     var_dict[f'I_stream_{ro_el}_copy'].save(f'I_{ro_el}_copy')
                #     var_dict[f'Q_stream_{ro_el}_copy'].save(f'Q_{ro_el}_copy')
                # n_burst_stream.save('n_burst')
                # n_burst_stream_copy.save('n_burst_copy')
                    var_dict[f'I_stream_{ro_el}'].save_all(f'I_{ro_el}')
                    var_dict[f'Q_stream_{ro_el}'].save_all(f'Q_{ro_el}')
                    var_dict[f'I_stream_{ro_el}_copy'].save_all(f'I_{ro_el}_copy')
                    var_dict[f'Q_stream_{ro_el}_copy'].save_all(f'Q_{ro_el}_copy')
                n_burst_stream.save_all('n_burst')
                n_burst_stream_copy.save_all('n_burst_copy')
                    
                #     var_dict[f'I_stream_{ro_el}'].buffer(npls/2 * npts).save(f'I_{ro_el}')

        self.last_prog = self.burst_detector_QM_prog  
        self.burst_detector_QM_compiled_program_id = self.qm.compile(self.burst_detector_QM_prog)
    
        
        
    def run_burst_detector_QM(self, is_save_data = None, is_plot = True, **kwargs):
        t_start = time.time()
        
        if is_save_data is None: is_save_data = self.is_save_data
        
        results_dict = self.burst_detector_QM_results
        ro_element = results_dict['ro_element']
        npts = results_dict['npts']
        ro_pulse_len = results_dict['ro_pulse_len']
        chunk_size = results_dict['chunk_size']
        
        self.qm_server.clear_all_job_results()  
        pending_job = self.qm.queue.add_compiled(self.burst_detector_QM_compiled_program_id)
        job = pending_job.wait_for_execution()
        job = self.qm.execute(self.burst_detector_QM_prog)
        
        res = job.result_handles
        res.wait_for_all_values()
        
        t_pls = np.linspace(0, ro_pulse_len - 4*chunk_size, npts)
        
        t_stop = time.time()
        # print('\n t = ',np.round(t_stop-t_start,2),'s\n')
        
        
        for ro_el in ro_element:
            I_temp = res.get(f'I_{ro_el}').fetch_all()['value'].reshape((npts,-1))
            Q_temp = res.get(f'Q_{ro_el}').fetch_all()['value'].reshape((npts,-1))
            I_temp_copy = res.get(f'I_{ro_el}_copy').fetch_all()['value'].reshape((npts,-1))
            Q_temp_copy = res.get(f'Q_{ro_el}_copy').fetch_all()['value'].reshape((npts,-1))
            
            t_ro = 4*res.get(f'timestamps_{ro_el}').fetch_all()['value']
            t_ro_copy = 4*res.get(f'timestamps_{ro_el}_copy').fetch_all()['value']
            
            n_burst = res.get('n_burst').fetch_all()['value']
            n_burst_copy = res.get('n_burst_copy').fetch_all()['value']
            
            
            t_temp = np.array([])
            for n in n_burst:
                t_temp = np.append(t_temp, t_pls + t_ro[n])
            t_temp_copy = np.array([])
            for n in n_burst_copy:
                t_temp_copy = np.append(t_temp_copy, t_pls + t_ro_copy[n])
            t_temp = t_temp.reshape((npts,-1))
            t_temp_copy = t_temp_copy.reshape((npts,-1))
            
            t = []
            I = []
            Q = []
            if n_burst.size != 0 or n_burst_copy.size != 0:
                t_flat = np.append(t_temp.flatten(), t_temp_copy.flatten())
                I_flat = np.append(I_temp.flatten(), I_temp_copy.flatten())
                Q_flat = np.append(Q_temp.flatten(), Q_temp_copy.flatten())
                
                t_flat , order = np.unique(t_flat-t_flat[0], return_index=True)
                I_flat = I_flat[order]
                Q_flat = Q_flat[order]
                
                t = t_flat
                I = I_flat
                Q = Q_flat
                
                # ind_npls = np.argwhere(np.diff(t_flat) > 10000)
                # if ind_npls.size == 0:
                #     t = [t_flat.tolist()]
                #     I = [I_flat.tolist()]
                #     Q = [Q_flat.tolist()]
                # else:                    
                #     for i_npls in np.append([0],ind_npls[:-1]):
                #         t.append(t_flat[i_npls:i_npls+1])
                #         I.append(I_flat[i_npls:i_npls+1])
                #         Q.append(Q_flat[i_npls:i_npls+1])
                #     t.append(t_flat[int(ind_npls[-1]):])
                #     I.append(I_flat[int(ind_npls[-1]):])
                #     Q.append(Q_flat[int(ind_npls[-1]):])
                    
                
                
                # IF_freq = self.element_IF(ro_el)
                # # print(f'IF_{ro_el}=',IF_freq*1e-6,'MHz')
                # I_flat_corr = np.zeros(len(I_flat))
                # Q_flat_corr = np.zeros(len(Q_flat))
                # for i,(ti,Ii,Qi) in enumerate(zip(t_flat,I_flat,Q_flat)):
                #     I_flat_corr[i], Q_flat_corr[i] = correct_non_integer_demod(Ii,Qi,ti,4*chunk_size,IF_freq, A=0)
            
            else:
                print(f'{ro_el}: No bursts detected!')
            

            results_dict[f'I_{ro_el}'] = I
            results_dict[f'Q_{ro_el}'] = Q
            results_dict[f't_{ro_el}'] = t
        
        
        if is_save_data: self.pickle_save(results_dict, meas_name = 'burst_detector_QM')
        
        if is_plot: self.plot_burst_detector_QM(**kwargs)
        
        
    def plot_burst_detector_QM(self, **kwargs):
        
        if 'is_plot_time' in kwargs.keys(): is_plot_time = kwargs['is_plot_time'] 
        else: is_plot_time = True
        if 'data_type' in kwargs.keys(): data_type = kwargs['data_type'] 
        else: data_type = 'IQ'
            
        results_dict = self.burst_detector_QM_results
        ro_element = results_dict['ro_element']
        
        if data_type == 'IQ':
            
            if 'axs' in kwargs.keys():
                if kwargs['data_type'] is not None:
                    axs = kwargs['data_type'] 
            else: 
                fig, axs = plt.subplots(len(ro_element), 1, sharex=True)
            
            for ro_el, ax, thresh in zip(ro_element, fig.axes, results_dict['I_burst_thresh']):
                t = np.array(results_dict[f't_{ro_el}'])
                I = np.array(results_dict[f'I_{ro_el}'])
                Q = np.array(results_dict[f'Q_{ro_el}'])
                
                if is_plot_time:
                    Iax, = ax.plot(t*1e-9, I*1e6,'.-', label = 'I')
                    Qax, = ax.plot(t*1e-9, Q*1e6,'.-', label = 'Q')
                    if t != []:
                        ax.plot([t[0]*1e-9,t[-1]*1e-9],[thresh[0]*1e6,thresh[0]*1e6],'--',color=Iax.get_color())
                        ax.plot([t[0]*1e-9,t[-1]*1e-9],[thresh[1]*1e6,thresh[1]*1e6],'--',color=Iax.get_color())
                else:
                    Iax, = ax.plot(I*1e6,'.-', label = 'I')
                    Qax, = ax.plot(Q*1e6,'.-', label = 'Q')
                    if t != []:
                        ax.plot([0,len(I)],[thresh[0]*1e6,thresh[0]*1e6],'--',color=Iax.get_color())
                        ax.plot([0,len(I)],[thresh[1]*1e6,thresh[1]*1e6],'--',color=Iax.get_color())
                        
                ax.set_ylabel(f'{ro_el} '+r'$[\mu V]$')
                ax.legend(fontsize = 12)
            
            if is_plot_time:    
                ax.set_xlabel(r'Time [s]')
            else:
                ax.set_xlabel('chunk No.')
            
            fig.tight_layout()
        
        
        elif data_type == 'phase':
            fig, axs = plt.subplots(len(ro_element), 1, sharex=True)
            for ro_el, ax in zip(ro_element, fig.axes):
                t = np.array(results_dict[f't_{ro_el}'])
                I = np.array(results_dict[f'I_{ro_el}'])
                Q = np.array(results_dict[f'Q_{ro_el}'])
                phase = np.angle(I+1j*Q)
                
                phase[phase < -np.pi/2] *= -1
                
                burst_end_ind = np.where(np.diff(t)>1e6)[0]
            
                if is_plot_time:
                    ax.plot(t*1e-9, phase,'.-')
                else:
                    ax.plot(phase,'.-')
                    for i in burst_end_ind:
                        ax.plot([i,i],[min(phase),max(phase)],'--k')
    
                ax.set_ylabel(f'{ro_el} [rad]')
            
            if is_plot_time:    
                ax.set_xlabel(r'Time [s]')
            else:
                ax.set_xlabel('chunk No.')
            
            fig.tight_layout()
            
            
        
    
    def load_burst_detector_single(self,
                            N_tries = 1000,
                            npts = 50,
                            npls = 100,
                            ro_element = None,
                            ro_pulse = 'ro_pulse',
                            zero_pulse = '0_pulse',
                            I_burst_thresh = 0,
                            ramp_up = 500e3,
                            is_apply_burst = False,
                            burst_element = None,
                            burst_pulse = 'burst_pulse',
                            burst_delay = 520e3):
        
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        if burst_element is None: burst_element = self.burst_element

        
        if (npls % 2) == 1:
            npls += 1
            print('supplied npls is odd! increasing by 1 to make it even')
        if (N_tries % 2) == 1:
            N_tries += 1
            print('supplied N_tries is odd! increasing by 1 to make it even')
            
        self.burst_detector_compiled_program_id = None
        
        if type(ro_element)!=list:
            ro_element = [ro_element]
        if type(I_burst_thresh)!=list:
            I_burst_thresh = [I_burst_thresh]*len(ro_element)
        if type(burst_delay)!=list:
            burst_delay = [burst_delay]

        ro_pulse_len = self.pulse_len(ro_element[0], ro_pulse)
        chunk_size = ro_pulse_len // npts // 4
        
        run_time = (N_tries*ro_pulse_len + ramp_up)*1e-9
        print(f'\nRun time is {np.around(run_time)} s')
        
        self.burst_detector_single_results = {}
        results_dict = self.burst_detector_single_results
        results_dict['N_tries'] = N_tries
        results_dict['npts'] = npts
        results_dict['npls'] = npls
        results_dict['ro_element'] = ro_element
        results_dict['ro_pulse'] = ro_pulse
        results_dict['ro_pulse_len'] = ro_pulse_len
        results_dict['chunk_size'] = chunk_size
        results_dict['zero_pulse'] = zero_pulse
        results_dict['I_burst_thresh'] = I_burst_thresh
        results_dict['ramp_up'] = ramp_up
        results_dict['burst_element'] = burst_element
        results_dict['burst_pulse'] = burst_pulse
        results_dict['burst_delay'] = burst_delay
        
        with program() as self.burst_detector_single_prog:
            n = declare(int)
            i = declare(int)
            is_burst = declare(bool, value = False)
            npls_i = declare(int, value = 0)
            var_dict = {}
            for ro_el in ro_element:
                var_dict[f'I_{ro_el}'] = declare(fixed, size = npts)
                var_dict[f'Q_{ro_el}'] = declare(fixed, size = npts)
                var_dict[f'I_stream_{ro_el}'] = declare_stream()
                var_dict[f'Q_stream_{ro_el}'] = declare_stream()
                
            n_burst_stream = declare_stream()
            
            if (is_apply_burst):
                for b_delay in burst_delay:
                    wait(int(b_delay / 4), burst_element)
                    play(burst_pulse, burst_element)
                
            for ro_el in ro_element:
                reset_phase(ro_el+'_copy')
                play(ro_pulse, ro_el+'_copy')
                wait(int(ramp_up / 4), ro_el+'_copy')
                    
            with for_(n, 0, n < N_tries, n + 1):
                for ro_el in ro_element:
                    measure(zero_pulse, ro_el+'_copy', None,
                            demod.sliced("integ_w_I", var_dict[f'I_{ro_el}'], chunk_size, "out1"),
                            demod.sliced("integ_w_Q", var_dict[f'Q_{ro_el}'], chunk_size, "out2"),
                            # dual_demod.full("integ_w_I", "out1", "integ_w_Q", "out2", var_dict[f'I_{ro_el}']),
                            # dual_demod.full("integ_w_minusQ", "out1", "integ_w_I", "out2", var_dict[f'Q_{ro_el}']),
                            timestamp_stream = f'timestamps_{ro_el}')
                    
                with if_(~is_burst):
                    for ro_el, I_th in zip(ro_element, I_burst_thresh):
                        criterion = var_dict[f'I_{ro_el}'][npts-1]
                        # criterion = lib.Math.pow2(var_dict[f'I_{ro_el}'][npts-1]) + lib.Math.pow2(var_dict[f'Q_{ro_el}'][npts-1])
                        with if_(criterion < I_th):
                            assign(is_burst, True)
                    
                with if_(is_burst):
                    for ro_el in ro_element:
                        with for_(i, 0, i<npts , i+1):
                            save(var_dict[f'I_{ro_el}'][i], var_dict[f'I_stream_{ro_el}'])
                            save(var_dict[f'Q_{ro_el}'][i], var_dict[f'Q_stream_{ro_el}'])
                    save(n, n_burst_stream)
                    assign(npls_i,  npls_i+1)
                    with if_(npls_i > npls):
                        assign(is_burst, False)
                        assign(npls_i, 0)
             
            ramp_to_zero(ro_el+'_copy')                 

            with stream_processing():
                for ro_el in ro_element:
                    var_dict[f'I_stream_{ro_el}'].save_all(f'I_{ro_el}')
                    var_dict[f'Q_stream_{ro_el}'].save_all(f'Q_{ro_el}')
                n_burst_stream.save_all('n_burst')
                    
        self.last_prog = self.burst_detector_single_prog  
        self.burst_detector_single_compiled_program_id = self.qm.compile(self.burst_detector_single_prog)
    
        
        
    def run_burst_detector_single(self, is_save_data = None, is_plot = True, **kwargs):
        t_start = time.time()
        
        if is_save_data is None: is_save_data = self.is_save_data
        
        results_dict = self.burst_detector_single_results
        ro_element = results_dict['ro_element']
        npts = results_dict['npts']
        ro_pulse_len = results_dict['ro_pulse_len']
        chunk_size = results_dict['chunk_size']
        
        self.qm_server.clear_all_job_results()  
        pending_job = self.qm.queue.add_compiled(self.burst_detector_single_compiled_program_id)
        job = pending_job.wait_for_execution()
        job = self.qm.execute(self.burst_detector_single_prog)
        
        res = job.result_handles
        res.wait_for_all_values()
        
        t_pls = np.linspace(0, ro_pulse_len - 4*chunk_size, npts)
        
        t_stop = time.time()
        print('\n t = ',np.round(t_stop-t_start,2),'s\n')
        
        
        for ro_el in ro_element:
            I_temp = res.get(f'I_{ro_el}').fetch_all()['value'].reshape((npts,-1))
            Q_temp = res.get(f'Q_{ro_el}').fetch_all()['value'].reshape((npts,-1))
            
            t_ro = 4*res.get(f'timestamps_{ro_el}').fetch_all()['value']
            
            n_burst = res.get('n_burst').fetch_all()['value']
            
            t_temp = np.array([])
            for n in n_burst:
                t_temp = np.append(t_temp, t_pls + t_ro[n])
            t_temp = t_temp.reshape((npts,-1))
            
            t = []
            I = []
            Q = []
            if n_burst.size != 0:
                t = t_temp.flatten()
                I = I_temp.flatten()
                Q = Q_temp.flatten()
                
                # ind_npls = np.argwhere(np.diff(t_flat) > 10000)
                # if ind_npls.size == 0:
                #     t = [t_flat.tolist()]
                #     I = [I_flat.tolist()]
                #     Q = [Q_flat.tolist()]
                # else:                    
                #     for i_npls in np.append([0],ind_npls[:-1]):
                #         t.append(t_flat[i_npls:i_npls+1])
                #         I.append(I_flat[i_npls:i_npls+1])
                #         Q.append(Q_flat[i_npls:i_npls+1])
                #     t.append(t_flat[int(ind_npls[-1]):])
                #     I.append(I_flat[int(ind_npls[-1]):])
                #     Q.append(Q_flat[int(ind_npls[-1]):])
                    
                
                # IF_freq = self.element_IF(ro_el)
                # # print(f'IF_{ro_el}=',IF_freq*1e-6,'MHz')
                # I_flat_corr = np.zeros(len(I_flat))
                # Q_flat_corr = np.zeros(len(Q_flat))
                # for i,(ti,Ii,Qi) in enumerate(zip(t_flat,I_flat,Q_flat)):
                #     I_flat_corr[i], Q_flat_corr[i] = correct_non_integer_demod(Ii,Qi,ti,4*chunk_size,IF_freq, A=0)
            
            else:
                print(f'{ro_el}: No bursts detected!')
            

            results_dict[f'I_{ro_el}'] = I
            results_dict[f'Q_{ro_el}'] = Q
            results_dict[f't_{ro_el}'] = t
        
        
        if is_save_data: self.pickle_save(results_dict, meas_name = 'burst_detector_single')
        if is_plot: self.plot_burst_detector_single( **kwargs)
        
        
    def plot_burst_detector_single(self, axs = None, is_plot_time=True, **kwargs):
        
        results_dict = self.burst_detector_single_results
        ro_element = results_dict['ro_element']
        
        if axs is None:
            fig, axs = plt.subplots(len(ro_element), 1, sharex=True)
        else:
            plt.sca(axs)
            
        for ro_el, ax in zip(ro_element, fig.axes):
            t = results_dict[f't_{ro_el}']
            I = results_dict[f'I_{ro_el}']
            Q = results_dict[f'Q_{ro_el}']
            
            if is_plot_time:
                ax.plot(np.array(t)*1e-3, np.array(I)*1e6,'.-', label = 'I')
                ax.plot(np.array(t)*1e-3, np.array(Q)*1e6,'.-', label = 'Q')
            else:
                ax.plot(np.array(I)*1e6,'.-', label = 'I')
                ax.plot(np.array(Q)*1e6,'.-', label = 'Q')
            
            # for i in range(len(t)):
            #     ax.plot(np.array(t[i])*1e-6, np.array(I[i])*1e6,'.-', label = 'I')
            #     ax.plot(np.array(t[i])*1e-6, np.array(Q[i])*1e6,'.-', label = 'Q')
                
                
                
            #     if i == 0:
            #         t[i] += -t[0,i]
            #     else:
            #         t[:,i] += -t[0,i] + t[-1,i-1] + 1000
            # # t = t.flatten()
            # # I = results_dict[f'I_{ro_el}'].flatten()
            # # Q = results_dict[f'Q_{ro_el}'].flatten()
            
            # ax.plot(t*1e-6,I*1e6,'.-',label = 'I')
            # ax.plot(t*1e-6,Q*1e6,'.-',label = 'Q')
            
            ax.set_ylabel(f'V {ro_el} '+r'$[\mu V]$')
            ax.legend(fontsize = 12)
            
        if is_plot_time:    
            plt.xlabel(r'Time [$\mu s$]')
        else:
            plt.xlabel('chunk No.')
            
        plt.tight_layout()
        
        
   
    def load_ro_spec(self, N_avg = 100, npts = 101,
                         start = None, stop = None,
                         ro_element = None,
                         ro_pulse = None,
                         res_name = None,
                         ramp_up = 0, 
                         wait_time = 0,
                         burst_element = None, 
                         burst_pulse = None,
                         **kwargs
                         ):
        
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        
        if ramp_up > 0:
            is_ramp_up_meas = False
            is_reset_phase_meas = False
        else:
            is_ramp_up_meas = True
            is_reset_phase_meas = True
            
        detuning = np.linspace(start, stop, npts, dtype = int)
            
        IF0 = int(self.element_IF(ro_element))
        
        if burst_element is None: 
            burst_len = 0
            burst_amp = None
        else:
            burst_len = self.pulse_len(burst_element,burst_pulse)
            burst_amp = self.pulse_amp(burst_element,burst_pulse)
        
        self.ro_spec_results = {}
        results_dict = self.ro_spec_results
        results_dict['N_avg'] = N_avg
        results_dict['npts'] = npts
        results_dict['detunings'] = detuning
        results_dict['res_name'] = res_name
        results_dict['ro_element'] = ro_element
        results_dict['ro_pulse'] = ro_pulse
        results_dict['ro_pulse_len'] = self.pulse_len(ro_element,ro_pulse)
        results_dict['ro_pulse_amp'] = self.pulse_amp(ro_element,ro_pulse)
        results_dict['ro_pulse_freq'] = self.element_freq(ro_element)
        results_dict['burst_element'] = burst_element
        results_dict['burst_pulse'] = burst_pulse
        results_dict['burst_pulse_len'] = burst_len
        results_dict['burst_pulse_amp'] = burst_amp
        results_dict['ramp_up'] = ramp_up
        results_dict['wait_time'] = wait_time
        results_dict['wait_between_seq'] = self.wait_between_seq
    
        run_time = N_avg*npts*(self.pulse_len(ro_element, ro_pulse) + self.wait_between_seq + ramp_up)
        print('Run time is {}s'.format(round(run_time * 1e-9)))
        
        with program() as self.ro_spec_prog:
            
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            f = declare(int)
            
            with for_(n, 0, n < N_avg, n + 1):
                with for_each_(f, detuning + IF0):
                    update_frequency(ro_element, f)
                    
                    reset_phase(ro_element)
                    if ramp_up > 0:
                        play(self.configObject.SuperMembers[ro_element].PulseParamsDict[ro_pulse].Additions.ramp['up']['name'], ro_element)
                        play(ro_pulse, ro_element, duration = int((ramp_up+burst_len+wait_time)//4))
                    
                    self.perform_full_measurement(I, Q, I_output_name = 'I', Q_output_name = 'Q', ro_element = ro_element, is_ramp_up = is_ramp_up_meas, is_save = True, is_reset_phase = is_reset_phase_meas)
                    
                    if not burst_element is None:
                        wait(int(ramp_up//4), burst_element)
                        play(burst_pulse, burst_element)
                    
        self.last_prog = self.ro_spec_prog  
    
    
    def run_ro_spec(self, is_plot = True,
                        is_save_data = None,
                        is_calc_stat_error = None,
                        is_tof_correction = True,
                        add_str_to_name = '',
                        **kwargs,
                        ): 
        
        if is_calc_stat_error is None:
            is_calc_stat_error = self.is_calc_stat_error
        
        if is_save_data is None: is_save_data = self.is_save_data
            
        if not hasattr(self, 'ro_spec_prog'):
            raise ValueError('No ro_spec program defined!')
        
        results_dict = self.ro_spec_results
        N_avg = results_dict['N_avg']
        ro_element = results_dict['ro_element']
        res_name = results_dict['res_name']
        detunings = results_dict['detunings']
        
        self.qm_server.clear_all_job_results()
        self.ro_spec_job = self.qm.execute(self.ro_spec_prog, duration_limit=0, data_limit=0)
        self.ro_spec_job.result_handles.wait_for_all_values()
        self.ro_spec_job.execution_report()
        
        I_no_phase_corr = self.ro_spec_job.result_handles.get('I').fetch_all()['value'].reshape((N_avg, -1))
        Q_no_phase_corr = self.ro_spec_job.result_handles.get('Q').fetch_all()['value'].reshape((N_avg, -1))
        
        if is_tof_correction:
            tof = self.tof(ro_element)*1e-9
            phase_corr = -tof*detunings*2*np.pi
            results_dict['I'] = np.cos(phase_corr)*I_no_phase_corr + np.sin(phase_corr)*Q_no_phase_corr
            results_dict['Q'] = -np.sin(phase_corr)*I_no_phase_corr + np.cos(phase_corr)*Q_no_phase_corr
        
        if is_save_data:
            time_of_meas_str = datetime.now().strftime("%d_%m_%Y___%H_%M_%S")
            file_name = 'ro_spec_'+res_name+add_str_to_name+'_'+time_of_meas_str
            results_dict['file_name'] = file_name
            self.pickle_save(results_dict, meas_name='ro_spec', filename=file_name, time_of_meas=time_of_meas_str)
        else:
            results_dict['file_name'] = None
        
        if is_plot:
            self.plot_ro_spec(**kwargs)
        
        
    def plot_ro_spec(self, ax=None, normalize=False, label_str='', title_str='', IQ_corr=False, meas_type='mag', **kwargs):
        
        results_dict = self.ro_spec_results
        detunings = results_dict['detunings']
        res_name = results_dict['res_name']
        ro_amp = results_dict['ro_pulse_amp']
        ro_pulse_len = results_dict['ro_pulse_len']
        burst_amp = results_dict['burst_pulse_amp']
        wait_time = results_dict['wait_time']
        ramp_up = results_dict['ramp_up']
        burst_element = results_dict['burst_element']
        ro_element = results_dict['ro_element']
        ro_pulse = results_dict['ro_pulse']
        
        if ro_amp.imag == 0:
            ro_amp = ro_amp.real
        
        data = [results_dict['I'], results_dict['Q']]
        
        I_list, I_err = self.process_data(data, which_data = 'I', is_calc_stat_error=True)
        Q_list, Q_err = self.process_data(data, which_data = 'Q', is_calc_stat_error=True)
        phase_list, phase_err = self.process_data(data, which_data = 'Phase', is_calc_stat_error=True)
        mag_list, mag_err = self.process_data(data, which_data = 'Mag', is_calc_stat_error=True)

        if IQ_corr:
            T = results_dict['ro_pulse_len']
            t_meas = self.pulse_ramp(ro_element,ro_pulse) + ramp_up + T + wait_time
            IF_freq = self.config['elements'][ro_element]['intermediate_frequency']
            
            I_corr = np.zeros(detunings.size)
            Q_corr = np.zeros(detunings.size)
            for i in range(detunings.size):
                I_corr[i], Q_corr[i] = correct_non_integer_demod(I_list[i], Q_list[i], t_meas, ro_pulse_len, detunings[i]+IF_freq)
            I_list = I_corr    
            Q_list = Q_corr
            mag_list = np.sqrt(I_list**2 + Q_list**2)
            
        fmt = '.-'
        
        # if ax is None:
        #     fig, ax = plt.subplots(figsize=[10,8])
        # else:
        #     plt.sca(ax)
        
        # if meas_type in ['mag','Mag','MAG','magnitude']:
        #     if normalize:
        #         factor = 1/mag_list.max()
        #         plt.ylabel('Mag [Norm.]')
        #         label_str_norm = f'  max[mV]={data_processing.round_sig_dig(mag_list.max()*1e3,3)}' 
        #     else: 
        #         factor = 1
        #         plt.ylabel('Mag [mV]')
        #         label_str_norm = ''
                    
        #     if not label_str == '':
        #         label_str = label_str + ' ' + label_str_norm
    
        #     plt.errorbar(detunings*1e-6, mag_list*1e3*factor, yerr = mag_err*1e3*factor, fmt = fmt, capsize = 5, markersize = 6, label=label_str)
        #     plt.xlabel('Freq [MHz]')
        #     plt.title('ro_spec ('+res_name+')'+' '+title_str)
        #     if not label_str == '':
        #         leg = plt.legend(fontsize = 14)
        #         leg.set_draggable(True)
        #     plt.tight_layout()
           
        # if meas_type in ['phase','Phase','PHASE']:
        #     plt.ylabel('Phase [rad]')
        #     label_str_norm = ''
                    
        #     if not label_str == '':
        #         label_str = label_str + ' ' + label_str_norm
    
        #     plt.errorbar(detunings*1e-6, phase_list, yerr = phase_err, fmt = fmt, capsize = 5, markersize = 6, label=label_str)
        #     plt.xlabel('Freq [MHz]')
        #     plt.title('ro_spec ('+res_name+')'+' '+title_str)
        #     if not label_str == '':
        #         leg = plt.legend(fontsize = 14)
        #         leg.set_draggable(True)
        #     plt.tight_layout()
        
        fig, ax = plt.subplots(2,2, sharex=True, figsize=[10,8])
        plt.sca(ax[0,0])
        plt.errorbar(detunings*1e-6, I_list*1e3, yerr = I_err*1e3, fmt = fmt, capsize = 5, markersize = 6)
        plt.ylabel(f'I [mV]')
        plt.sca(ax[0,1])
        plt.errorbar(detunings*1e-6, Q_list*1e3, yerr = Q_err*1e3, fmt = fmt, capsize = 5, markersize = 6)
        plt.ylabel(f'Q [mV]')
        plt.sca(ax[1,0])
        plt.errorbar(detunings*1e-6, mag_list*1e3, yerr = mag_err*1e3, fmt = fmt, capsize = 5, markersize = 6)
        plt.ylabel(f'Mag [mV]')
        plt.xlabel(f'Freq [MHz]')
        plt.sca(ax[1,1])
        plt.errorbar(detunings*1e-6, phase_list, yerr = phase_err, fmt = fmt, capsize = 5, markersize = 6)
        plt.ylabel('Phase')
        plt.xlabel(f'Freq [MHz]')
        plt.suptitle('ro_spec ('+res_name+')'+' '+title_str + 'ro amp = ' + str(abs(ro_amp))+ ' V')
        plt.tight_layout()
