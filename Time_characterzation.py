# -*- coding: utf-8 -*-
"""
Created on Sun May  3 14:33:59 2020

"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent /'usefulFunctions'))

import traceback

import matplotlib
import numpy as np
# OPX Qunatum Machine:
from qm.qua import *

import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as fig
import scipy as sp
from scipy.optimize import curve_fit, minimize
from Time_domain_base_class import Time_domain_base
from sklearn.cluster import KMeans, MeanShift
from tqdm import tqdm
 # set font for figure axis
font = {'size'   : 20}

matplotlib.rc('font', **font)
# def prin(**kwargs):
#     print(kwargs['tt'])
# prin(tt=5)

from data_processing import round_value_by_error, scale_data_units, histogram_fidelity
from plotting import plot_fft, next_fig_num_by_name, last_fig_num_by_name, plot_hist2d, plot_2D
from smart_fit import fit_circle, sFit, non_TimeDomain_fit
from OneQubitTomography import OneQubitTomo 

from numpy.fft import fft, ifft, fftshift, ifftshift


class TiDo_Chara(Time_domain_base):
#%% initialzation and attributes 
   
    def __init__(self, 
                 Config, #make sure you follow the config example with the names + nust include to sub dicitonaries, one for the opx "opx_config" and an auxillary config "aux_config"
                 hp_readout = False, 
                 thresholding = False, #will assign edge values to data if above or below threshold
                 normalize = False, #will normalize thresholded results to -1 or 1
                 threshold_time_lim = 5, # will know if it should find threshold again if this time has past since finding threshold (in seconds)
                 e_to_f = False,
                 e_to_f_map_back = False,
                 text_size = 10,
                 title_size = 25,
                 **kwargs):# True makes a pi pulse from a single pulse false makes from 2 pi pulses
    
        ##TODO make configuration loadable as string   #isinstance()

        self.subplot = False
        ##TODO add auto
        self.debugging_flag = True
        self.title_size = title_size
        self.text_size = text_size
        self.thresholding = thresholding
        self.normalize = normalize
        self.threshold_time_lim = threshold_time_lim
        self.e_to_f_map_back = e_to_f_map_back
        self.e_to_f = e_to_f
        if self.e_to_f:  self.set_e_to_f(self.e_to_f_map_back)
        else: self.main_qubit_g_to_e = None


        super().__init__(Config, **kwargs)

                    
#TODO add a function to transform the data from raw data to probability and apply it to measurments and plots
#%% pi no pi specific
     
    def load_pinopi(self, npts = 50,
                    N_avg = 10000,
                    meas_type = 'sliced',
                    pi2_pulse = None,
                    ro_pulse = None,
                    is_continuous_drive = False,
                    drive_pulse = None,
                    qubit = None,
                    ro_element = None,
                    is_sb_cool = None,
                    sideband_cool_qubit = ['qb1', 'qb2'],
                    is_active_reset = None,
                    wait_time = 0,
                    **kwargs):
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        if pi2_pulse is None: pi2_pulse = self.pi2_pulse
        if is_sb_cool is None: is_sb_cool = self.is_sb_cool
        if drive_pulse is None: drive_pulse = 'mixer_cal_pulse'
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        if is_active_reset and meas_type=='sliced':
            is_active_reset=False
            print("Can't do active reset with sliced measurement. Set active reset to False.")
        if meas_type == 'sliced' or meas_type == 'accumulated': 
            def is_valid(A,C):
                return A % (4 * C) == 0
            def closest_divisor(A, B):
                # Search outward from B
                for offset in range(0, A):  # A is a safe upper bound
                    for sign in (-1, 1):
                        C = B + sign * offset
                        if C > 0 and is_valid(A,C):
                            return C
                raise ValueError("Can't find npts such that chunk size is whole")
            if not is_valid(self.pulse_len(ro_element, ro_pulse), npts):
                npts = closest_divisor(self.pulse_len(ro_element, ro_pulse), npts)
                print(f'Changed npts to {npts} so chunk size is a whole number.')
            chunk_size =  self.pulse_len(ro_element, ro_pulse) // npts // 4
        
        self.results['PiNoPi'] = {'meas_type': meas_type,
                               'N_avg': N_avg,
                               'npts': npts,
                               'qubit': qubit,
                               'ro_element': ro_element}
        results_dict = self.results['PiNoPi']
        if meas_type == 'sliced' or meas_type == 'accumulated': 
            results_dict['chunk_size'] = chunk_size
        self.pinopi_N_avg = N_avg
        self.pinopi_npts = npts

        if is_sb_cool: run_time = 2*N_avg*(self.pulse_len(ro_element,ro_pulse) + self.pulse_len(qubit, self.sb_cool_kwargs['rabi_sideband_cooling_pulse']))
        elif is_active_reset: run_time = 2*N_avg*self.pulse_len(ro_element,ro_pulse)
        else: run_time = 2*N_avg*(self.pulse_len(ro_element,ro_pulse)+self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))            
        
        with program() as self.pinopi_prog:
            n = declare(int)
                
            if meas_type == 'sliced' or meas_type == 'accumulated':
                I_pi = declare(fixed, size = npts)
                I_nopi = declare(fixed, size = npts)
                Q_pi = declare(fixed, size = npts)
                Q_nopi = declare(fixed, size = npts)
                i = declare(int)
                
            elif meas_type == 'full':
                I_pi = declare(fixed)
                I_nopi = declare(fixed)
                Q_pi = declare(fixed)
                Q_nopi = declare(fixed)
                
            with for_(n, 0, n < N_avg, n + 1):
            
                if self.e_to_f:  self.map_e_to_g(self.main_qubit_g_to_e)  
                if not is_continuous_drive:
                    self._pi_pulse(qubit = qubit, **kwargs)
                    if wait_time >0:
                        wait(wait_time//4,qubit)
                    align(qubit, ro_element)  
                else:
                    play(drive_pulse, qubit, duration = self.pulse_len(ro_element, ro_pulse)//8)
                    
                if self.e_to_f and self.e_to_f_map_back:  self.map_e_to_g(self.main_qubit_g_to_e)
                
                if results_dict['meas_type'] == 'sliced': self.perform_sliced_measurement(I_pi, Q_pi, i, chunk_size = chunk_size, npts = npts, I_output_name = 'I_pi', Q_output_name = 'Q_pi', ro_element = ro_element, is_sb_cool = is_sb_cool,  is_ramp_up = True, readout_pulse = ro_pulse, **kwargs)
                elif results_dict['meas_type'] == 'accumulated': self.perform_accumulated_measurement(I_pi, Q_pi, i, chunk_size = chunk_size, npts = npts, I_output_name = 'I_pi', Q_output_name = 'Q_pi', ro_element = ro_element, is_sb_cool = is_sb_cool,  is_ramp_up = True, readout_pulse = ro_pulse, **kwargs)
                elif results_dict['meas_type'] == 'full': self.perform_full_measurement(I_pi, Q_pi, I_output_name = 'I_pi', Q_output_name = 'Q_pi', ro_element = ro_element, is_sb_cool = is_sb_cool,  is_ramp_up = True, is_active_reset = is_active_reset, readout_pulse = ro_pulse, **kwargs)
                
                align(qubit, ro_element)
                    
                if self.e_to_f:  self.map_e_to_g(self.main_qubit_g_to_e)
                if self.e_to_f and self.e_to_f_map_back:  self.map_e_to_g(self.main_qubit_g_to_e)
                
                align(qubit, ro_element)
                if results_dict['meas_type'] == 'sliced': self.perform_sliced_measurement(I_nopi, Q_nopi, i, chunk_size = chunk_size, npts = npts, I_output_name = 'I_nopi', Q_output_name = 'Q_nopi', ro_element = ro_element, is_sb_cool = is_sb_cool,  is_ramp_up = True, readout_pulse = ro_pulse, **kwargs)
                elif results_dict['meas_type'] == 'accumulated': self.perform_accumulated_measurement(I_nopi, Q_nopi, i, chunk_size = chunk_size, npts = npts, I_output_name = 'I_nopi', Q_output_name = 'Q_nopi', ro_element = ro_element, is_sb_cool = is_sb_cool,  is_ramp_up = True, readout_pulse = ro_pulse, **kwargs)
                elif results_dict['meas_type'] == 'full':self.perform_full_measurement(I_nopi, Q_nopi, I_output_name = 'I_nopi', Q_output_name = 'Q_nopi', ro_element = ro_element, is_ramp_up = True, is_active_reset = is_active_reset, is_sb_cool = is_sb_cool, readout_pulse = ro_pulse, **kwargs)
        self.last_prog = self.pinopi_prog  
                
    def run_pinopi(self, plot = True, is_save_data = None, num_of_bins = 300, ro_pulse = None,
                   is_calc_stat_error = None, is_post_select = False, is_save_discrimination = True, is_plot_fid = True,
                   is_update_readout_analyzer = False,
                   **kwargs): 
        results_dict = self.results['PiNoPi']
        qubit = results_dict['qubit']
        
        if is_calc_stat_error is None: is_calc_stat_error = self.is_calc_stat_error
        if is_save_data is None: is_save_data = self.is_save_data
        
        if not hasattr(self, 'pinopi_prog'): raise ValueError("Idiot! You did not write pinopi program")
        if ro_pulse is None: ro_pulse = self.ro_pulse

        self.qm_server.clear_all_job_results()            
        job = self.qm.execute(self.pinopi_prog, duration_limit=0, data_limit=0)
        job.result_handles.wait_for_all_values()
        self.last_job = job
        
        I_pi = job.result_handles.get('I_pi').fetch_all()['value'].reshape((self.pinopi_N_avg, -1))
        I_nopi = job.result_handles.get('I_nopi').fetch_all()['value'].reshape(( self.pinopi_N_avg, -1))
        Q_pi = job.result_handles.get('Q_pi').fetch_all()['value'].reshape(( self.pinopi_N_avg, -1))
        Q_nopi = job.result_handles.get('Q_nopi').fetch_all()['value'].reshape(( self.pinopi_N_avg, -1))
        results_dict['I_pi'] = I_pi
        results_dict['Q_pi'] = Q_pi
        results_dict['I_nopi'] = I_nopi
        results_dict['Q_nopi'] = Q_nopi
        
        if results_dict['meas_type'] == 'sliced' or results_dict['meas_type'] == 'accumulated':
            chunk_size = results_dict['chunk_size']
            time_since_pulse = np.linspace(results_dict['chunk_size'] * 4, results_dict['chunk_size'] * 4 * self.pinopi_npts , self.pinopi_npts)
            results_dict['time'] = time_since_pulse
            
            
            if is_save_discrimination:
                self.discrim_I = np.repeat(I_pi.mean(0)-I_nopi.mean(0), results_dict['chunk_size'])
                self.discrim_Q = np.repeat(Q_pi.mean(0)-Q_nopi.mean(0), results_dict['chunk_size'])
                
            hist_pi,_ = self.process_data([I_pi.sum(1), Q_pi.sum(1)], is_mean = False, is_thresh = False, is_calc_stat_error = False)
            hist_nopi,_ = self.process_data([I_nopi.sum(1), Q_nopi.sum(1)], is_mean = False, is_thresh = False, is_calc_stat_error = False)
            
            if is_save_data is None: is_save_data = self.is_save_data
            if is_save_data: self.pickle_save(results_dict, 'PiNoPi')
            
            if plot:
                self.plot_pinopi(num_of_bins = num_of_bins, is_plot_fid = is_plot_fid, is_update_readout_analyzer = is_update_readout_analyzer, **kwargs)
                
            fdlty, avg_ro = histogram_fidelity(hist_pi, hist_nopi)
            print('The readout fidelity is {}'.format(round(fdlty,3)))
            
            return avg_ro, fdlty
        
        else:
            is_thresh = self.is_thresh
            
            qubit = results_dict['qubit']
            
            hist_pi, _ = self.process_data([I_pi, Q_pi], is_mean = False, is_thresh = False, is_calc_stat_error = False)
            hist_nopi, _ = self.process_data([I_nopi, Q_nopi], is_mean = False, is_thresh = False, is_calc_stat_error = False)
            
            if is_save_data is None: is_save_data = self.is_save_data
            if is_save_data: self.pickle_save(results_dict, 'PiNoPi')
            if plot:
                self.plot_pinopi(is_update_readout_analyzer = is_update_readout_analyzer)
                
            fdlty, avg_ro = histogram_fidelity(hist_pi, hist_nopi)
            print('The readout fidelity is {}'.format(round(fdlty,3)))
    
            return np.mean(hist_pi), np.mean(hist_nopi), fdlty

    def plot_pinopi(self, num_of_bins = 300, is_plot_fid = True, which_data = None, is_update_readout_analyzer = False, is_kmeans = False, **kwargs):
        if which_data is None: which_data = self.which_data
        
        results_dict = self.results['PiNoPi']
        I_pi = results_dict['I_pi']
        I_nopi = results_dict['I_nopi']
        Q_pi = results_dict['Q_pi']
        Q_nopi = results_dict['Q_nopi']
        qubit = results_dict['qubit']
        
        mean_I_pi = I_pi.mean(0)
        mean_I_nopi = I_nopi.mean(0)
        mean_Q_pi = Q_pi.mean(0)
        mean_Q_nopi = Q_nopi.mean(0)
        
        d_I = mean_I_pi.mean()-mean_I_nopi.mean()
        d_Q = mean_Q_pi.mean()-mean_Q_nopi.mean()
        d = np.sqrt(d_I**2+d_Q**2)
        v = (np.around(d_I/d,3), np.around(d_Q/d,3))
        phase = np.around(np.arctan2(v[1],v[0])*180/np.pi,3)
        results_dict['largest_distance_phase'] = phase
        print(f'Largest distance vector is {v} with phase of {phase} degrees')
        
        
        if results_dict['meas_type'] in ['sliced', 'accumulated']:
            npts = results_dict['npts']
            time = results_dict['time']
            if results_dict['meas_type'] == 'sliced':
                hist_pi,_ = self.process_data([I_pi.sum(1), Q_pi.sum(1)], is_mean = False, is_thresh = False, is_calc_stat_error = False)
                hist_nopi,_ = self.process_data([I_nopi.sum(1), Q_nopi.sum(1)], is_mean = False, is_thresh = False, is_calc_stat_error = False)
            elif results_dict['meas_type'] == 'accumulated':
                hist_pi,_ = self.process_data([np.transpose(I_pi)[-1], np.transpose(Q_pi)[-1]], is_thresh = False, is_calc_stat_error = False, is_mean = False)
                hist_nopi,_ = self.process_data([np.transpose(I_nopi)[-1], np.transpose(Q_nopi)[-1]], is_thresh = False, is_calc_stat_error = False, is_mean = False)
                
            
            if is_plot_fid:
                npts = results_dict['npts']
                fid_list = np.zeros(npts)
                for i in range(npts):
                    hist_pi_i,_ = self.process_data([I_pi[:,:i].sum(1), Q_pi[:,:i].sum(1)], is_mean = False, is_thresh = False, is_calc_stat_error = False)
                    hist_nopi_i,_ = self.process_data([I_nopi[:,:i].sum(1), Q_nopi[:,:i].sum(1)], is_mean = False, is_thresh = False, is_calc_stat_error = False)
                    fid_list[i],_ = histogram_fidelity(hist_pi_i, hist_nopi_i)
                
                
                plt.figure()
                plt.plot(time,fid_list)
                plt.xlabel("Time [ns]", fontsize=18)
                plt.ylabel("Readout Fidelity", fontsize=18)
                plt.ylim([0,1])
                plt.grid()
                plt.tight_layout()
            
            prog_name = 'Pi no Pi'
            
            fig, axes = plt.subplots(2, 2, figsize = [12,8], num = next_fig_num_by_name(prog_name))
            
            plt.sca(axes[0,0])
            plt.errorbar(time, mean_I_pi*1e6, yerr = sp.stats.sem(I_pi,0)*1e6, fmt = 'r.')
            plt.errorbar(time, mean_I_nopi*1e6, yerr = sp.stats.sem(I_nopi,0)*1e6, fmt = 'b.')
            plt.ylabel(r"I $[\mu V]$", fontsize=18)
            plt.xlabel("Time [ns]", fontsize=18)
            
            plt.sca(axes[0,1])
            plt.errorbar(time, mean_Q_pi*1e6, yerr = sp.stats.sem(Q_pi,0)*1e6, fmt = 'r.')
            plt.errorbar(time, mean_Q_nopi*1e6, yerr = sp.stats.sem(Q_nopi,0)*1e6, fmt = 'b.')
            plt.ylabel(r"Q $[\mu V]$", fontsize=18)
            plt.xlabel("Time [ns]", fontsize=18)
            
            plt.sca(axes[1,0])
            plt.plot(0,0,'kx', markersize=6, alpha = 1)
            if results_dict['meas_type'] == 'sliced':
                plt.plot(I_pi.mean(axis=1)*1e6, Q_pi.mean(axis=1)*1e6, 'ro', markersize=1, alpha=0.5, zorder = 1)
                plt.plot(I_nopi.mean(axis=1)*1e6, Q_nopi.mean(axis=1)*1e6, 'bo', markersize=1, alpha=0.5, zorder = 1)
            elif results_dict['meas_type'] == 'accumulated':
                plt.plot(I_pi[:,-1]*1e6, Q_pi[:,-1]*1e6, 'ro', markersize=1, alpha=0.5, zorder = 1)
                plt.plot(I_nopi[:,-1]*1e6, Q_nopi[:,-1]*1e6, 'bo', markersize=1, alpha=0.5, zorder = 1)
                
            plt.plot(mean_I_pi*1e6, mean_Q_pi*1e6, markersize = 6, color = 'r', markerfacecolor = 'r', marker = 'o', markeredgecolor = 'k')
            plt.plot(mean_I_nopi*1e6, mean_Q_nopi*1e6, markersize = 6, color = 'b', markerfacecolor = 'b', marker = 's', markeredgecolor = 'k')
            plt.xlabel(r"I $[\mu V]$", fontsize=18)
            plt.ylabel("Q [mV]", fontsize=18)
            axes[1,0].set_aspect(1)
    
            
            plt.sca(axes[1,1])
            
            scale = 1 if self.which_data == 'Phase' else 1e6
            
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
            if self.which_data in ['I', 'Q', 'Mag', 'mag']: xlabel = self.which_data + r' $[\mu V]$'
            elif which_data == 'Phase': xlabel = 'Phase'
            elif np.isreal(which_data): xlabel = 'Phase' + str(self.which_data*180/np.pi) + r'$^\circ$'
            plt.xlabel(xlabel, fontsize=18)
            
            
            fig.text(0.47, 0.95, "Pi", ha="center", va="bottom", size="large",color="red")
            fig.text(0.52, 0.95, "no Pi", ha="center", va="bottom", size="large",color="blue")
            if results_dict['ro_element'] != 'ro':
                fig.text(0.63, 0.95, "({},{})".format(results_dict['qubit'], results_dict['ro_element']), ha="center", va="bottom", size="large",color="black")
            else:
                fig.text(0.6, 0.95, "({})".format(results_dict['qubit']), ha="center", va="bottom", size="large",color="black")
            plt.suptitle('',
                          fontsize=18)
            
            plt.tight_layout()
            
            fdlty, avg_ro = histogram_fidelity(hist_pi, hist_nopi)
            print('The readout fidelity is {}'.format(round(fdlty,3)))
            
            if is_update_readout_analyzer:
                is_ascending = hist_pi.mean()>hist_nopi.mean()
                thresh, offset, means = self.find_threshold(np.append(I_pi.sum(1),I_nopi.sum(1)), np.append(Q_pi.sum(1),Q_nopi.sum(1)), 
                                                     n_means = 2, is_ascending=is_ascending, is_fit_circle = False)
                if is_ascending: data_g = means[1]; data_e = means[0]
                else: data_g = means[0]; data_e = means[1]
                self.update_readout_analyzer(qubit = qubit, thresh = thresh, is_ascending = is_ascending, offset=offset, 
                                             data_e = data_e, data_g = data_g)
    
            return np.mean(hist_pi), np.mean(hist_nopi), fdlty
        
        elif results_dict['meas_type'] == 'full':
            
            prog_name = 'Pi no Pi'
            fig, axes = plt.subplots(2, 2, figsize = [12,8], num = next_fig_num_by_name(prog_name))  
            
            plt.sca(axes[0,0])
            plt.hist2d(I_pi.flatten(), Q_pi.flatten(), bins = 100, density = False, cmap = 'Greys')
            plt.ylabel("Q [V]", fontsize=18)
            axes[0,0].annotate("Pi",[0.01, 0.9], xycoords = 'axes fraction', color="red")
            axes[0,0].set_aspect(1)
            
            plt.sca(axes[1,0])
            plt.hist2d(I_nopi.flatten(), Q_nopi.flatten(), bins = 100, density = False, cmap = 'Greys')
            plt.xlabel("I [V]", fontsize=18)
            plt.ylabel("Q [V]", fontsize=18)
            axes[1,0].annotate("no Pi", [0.01, 0.9], xycoords = 'axes fraction', color="blue")
            axes[1,0].set_aspect(1)
            
            plt.sca(axes[0,1])
            bins = np.histogram(np.hstack((I_pi, I_nopi)), bins=num_of_bins)[1]
            
            hist_pi_data = plt.hist(I_pi,
                    bins = bins,
                    histtype='step',
                    color = 'r')
            hist_nopi_data = plt.hist(I_nopi,
                     bins = bins,
                     histtype='step',
                     color = 'b')
            plt.plot([np.mean(I_pi),np.mean(I_pi)],[0,max([max(hist_pi_data[0]),max(hist_nopi_data[0])]) + 5], '-k')
            plt.plot([np.mean(I_nopi),np.mean(I_nopi)],[0,max([max(hist_pi_data[0]),max(hist_nopi_data[0])]) + 5], '-k')
            plt.plot([np.mean([np.mean(I_nopi),np.mean(I_pi)]),np.mean([np.mean(I_nopi),np.mean(I_pi)])],[0,max([max(hist_pi_data[0]),max(hist_nopi_data[0])]) + 5], '--k')
            
            plt.ylabel("counts", fontsize=18)
            plt.xlabel('I [V]', fontsize=18)
            
            plt.sca(axes[1,1])
            bins = np.histogram(np.hstack((Q_pi, Q_nopi)), bins=num_of_bins)[1]
            hist_pi_data = plt.hist(Q_pi,
                    bins = bins,
                    histtype='step',
                    color = 'r')
            hist_nopi_data = plt.hist(Q_nopi,
                     bins = bins,
                     histtype='step',
                     color = 'b')
            plt.plot([np.mean(Q_pi),np.mean(Q_pi)],[0,max([max(hist_pi_data[0]),max(hist_nopi_data[0])]) + 5], '-k')
            plt.plot([np.mean(Q_nopi),np.mean(Q_nopi)],[0,max([max(hist_pi_data[0]),max(hist_nopi_data[0])]) + 5], '-k')
            plt.plot([np.mean([np.mean(Q_nopi),np.mean(Q_pi)]),np.mean([np.mean(Q_nopi),np.mean(Q_pi)])],[0,max([max(hist_pi_data[0]),max(hist_nopi_data[0])]) + 5], '--k')
                       
            plt.ylabel("counts", fontsize=18)
            plt.xlabel('Q [V]', fontsize=18)
            
            
            # plt.sca(axes[0])
            # plt.plot(I_pi, Q_pi, 'ro', markersize=1, alpha=0.5, zorder = 1)
            # plt.plot(I_nopi, Q_nopi, 'bo', markersize=1, alpha=0.5, zorder = 1 )
            # plt.plot(0,0,'kx', markersize=6, alpha = 1)
            # plt.xlabel("I [V]", fontsize=18)
            # plt.ylabel("Q [V]", fontsize=18)
            # axes[0].set_aspect(1)
            
            # hist_pi, _ = self.process_data([I_pi, Q_pi], is_mean = False, is_thresh = False, is_calc_stat_error = False)
            # hist_nopi, _ = self.process_data([I_nopi, Q_nopi], is_mean = False, is_thresh = False, is_calc_stat_error = False)
            
            # bins = np.histogram(np.hstack((hist_pi, hist_nopi)), bins=num_of_bins)[1]


            # plt.sca(axes[1])
            # hist_pi_data = plt.hist(hist_pi,
            #         bins = bins,
            #         histtype='step',
            #         color = 'r')
            # hist_nopi_data = plt.hist(hist_nopi,
            #          bins = bins,
            #          histtype='step',
            #          color = 'b')
            # plt.plot([np.mean(hist_pi),np.mean(hist_pi)],[0,max([max(hist_pi_data[0]),max(hist_nopi_data[0])]) + 5], '-k')
            # plt.plot([np.mean(hist_nopi),np.mean(hist_nopi)],[0,max([max(hist_pi_data[0]),max(hist_nopi_data[0])]) + 5], '-k')
            # plt.plot([np.mean([np.mean(hist_nopi),np.mean(hist_pi)]),np.mean([np.mean(hist_nopi),np.mean(hist_pi)])],[0,max([max(hist_pi_data[0]),max(hist_nopi_data[0])]) + 5], '--k')
            
            # plt.ylabel("counts", fontsize=18)
            # plt.xlabel(self.which_data, fontsize=18)
            
                
            fig.text(0.47, 0.95, "Pi", ha="center", va="bottom", size="large", color="red")
            fig.text(0.52, 0.95, "no Pi", ha="center", va="bottom", size="large", color="blue")
            fig.text(0.59, 0.95, f"({qubit})", ha="center", va="bottom", size="large", color="black")
            plt.suptitle('', fontsize=18)
            
            plt.tight_layout()
            
            if is_update_readout_analyzer:
                data_pi,_ = self.process_data([I_pi, Q_pi])
                data_nopi,_ = self.process_data([I_nopi, Q_nopi])
                is_ascending = data_pi.mean()>data_nopi.mean()
                thresh, offset, means = self.find_threshold(np.append(I_pi.sum(1),I_nopi.sum(1)), np.append(Q_pi.sum(1),Q_nopi.sum(1)), 
                                                     n_means = 2, is_ascending=is_ascending, is_fit_circle = False)
                if is_ascending: data_g = means[1]; data_e = means[0]
                else: data_g = means[0]; data_e = means[1]
                self.update_readout_analyzer(qubit = qubit, thresh = thresh, is_ascending = is_ascending, offset=offset, 
                                             data_e = data_e, data_g = data_g)
                
            hist_pi = self.determine_data(I_pi,Q_pi)
            hist_nopi = self.determine_data(I_nopi,Q_nopi)
            fdlty, avg_ro = histogram_fidelity(hist_pi, hist_nopi)
            print('The readout fidelity is {}'.format(round(fdlty,3)))
    
            return np.mean(hist_pi), np.mean(hist_nopi), fdlty

    def load_pinopi_2q(self, N_avg = 10000, 
                            plot = True, 
                            is_return_data = False, 
                            ro_element = None, 
                            ro_pulse = None,
                            wait_time = 0,
                            drive_element = None,
                            drive_pulse: str = None,
                            steady_time = 0,
                            is_active_reset = None,
                            meas_type = 'full',
                            npts = 100,
                            **kwargs):
        if ro_element is None: ro_element = self.main_readout
        if is_active_reset is None: is_active_reset = self.is_active_reset
        if ro_pulse is None: ro_pulse = self.ro_pulse
        self.results['pinopi_2q'] = {'N_avg': N_avg, 'ro_element': ro_element, 'is_active_reset': is_active_reset, 'meas_type': meas_type, 'npts':npts}
        if meas_type == 'sliced' or meas_type == 'accumulated': 
            chunk_size =  self.config['pulses'][self.config['elements'][ro_element]['operations'][ro_pulse]]['length'] // npts // 4
            self.results['pinopi_2q']['chunk_size'] = chunk_size
        
        run_time = 4  *  N_avg * (self.pulse_len(ro_element, self.ro_pulse) + self.wait_between_seq)
        print('Run time is {}s'.format(np.round(run_time * 1e-9)))
        
        with program() as self.pinopi_2q_prog:
            n = declare(int)
            if meas_type == 'sliced':
                j = declare(int)
                I00 = declare(fixed, size = npts)
                I01 = declare(fixed, size = npts)
                I10 = declare(fixed, size = npts)
                I11 = declare(fixed, size = npts)
                Q00 = declare(fixed, size = npts)
                Q01 = declare(fixed, size = npts)
                Q10 = declare(fixed, size = npts)
                Q11 = declare(fixed, size = npts)
            elif meas_type == 'full':
                I00 = declare(fixed)
                I01 = declare(fixed)
                I10 = declare(fixed)
                I11 = declare(fixed)
                Q00 = declare(fixed)
                Q01 = declare(fixed)
                Q10 = declare(fixed)
                Q11 = declare(fixed)
            i = declare(int)
            with for_(n, 0, n < N_avg, n + 1):
                with for_(i, 0, i < 4, i +1):
                    with switch_(i):
                        with case_(0):
                            if drive_element is not None:
                                play(drive_pulse, drive_element)
                            if wait_time>0:
                                wait(wait_time//4, ro_element)
                            if meas_type == 'full': self.perform_full_measurement(I00,Q00,'I00', 'Q00', is_active_reset = is_active_reset)
                            elif meas_type == 'sliced': self.perform_sliced_measurement(I=I00,Q=Q00,I_output_name='I00',Q_output_name= 'Q00',i=j,chunk_size=chunk_size,npts=npts, is_active_reset = is_active_reset)
                        with case_(1):
                            if drive_element is not None:
                                play(drive_pulse, drive_element)
                                if steady_time//4 > 0:
                                    wait(int(steady_time//4), self.main_qubit)
                                    wait(int(steady_time//4), self.scnd_qubit)
                            self._pi_pulse(qubit = self.scnd_qubit)
                            align(self.scnd_qubit, ro_element)
                            if wait_time>0:
                                wait(wait_time//4, ro_element)
                            if meas_type == 'full':self.perform_full_measurement(I10,Q10,'I10', 'Q10', is_active_reset = is_active_reset)
                            elif meas_type == 'sliced': self.perform_sliced_measurement(I=I10,Q=Q10,I_output_name='I10',Q_output_name= 'Q10',i=j,chunk_size=chunk_size,npts=npts, is_active_reset = is_active_reset)
                        with case_(2):
                            if drive_element is not None:
                                play(drive_pulse, drive_element)
                                if steady_time//4 > 0:
                                    wait(int(steady_time//4), self.main_qubit)
                                    wait(int(steady_time//4), self.scnd_qubit)
                            self._pi_pulse(qubit = self.main_qubit)
                            align(self.main_qubit, ro_element)
                            if wait_time>0:
                                wait(wait_time//4, ro_element)
                            if meas_type == 'full':self.perform_full_measurement(I01,Q01,'I01', 'Q01', is_active_reset = is_active_reset)
                            elif meas_type == 'sliced': self.perform_sliced_measurement(I=I01,Q=Q01,I_output_name='I01',Q_output_name= 'Q01',i=j,chunk_size=chunk_size,npts=npts, is_active_reset = is_active_reset)
                        with case_(3):
                            if drive_element is not None:
                                play(drive_pulse, drive_element)
                                if steady_time//4 > 0:
                                    wait(int(steady_time//4), self.main_qubit)
                                    wait(int(steady_time//4), self.scnd_qubit)
                            self._pi_pulse(qubit = self.main_qubit)
                            self._pi_pulse(qubit = self.scnd_qubit)
                            align(self.main_qubit, self.scnd_qubit, ro_element)
                            if wait_time>0:
                                wait(wait_time//4, ro_element)
                            if meas_type == 'full':self.perform_full_measurement(I11,Q11,'I11', 'Q11', is_active_reset = is_active_reset)
                            elif meas_type == 'sliced': self.perform_sliced_measurement(I=I11,Q=Q11,I_output_name='I11',Q_output_name= 'Q11',i=j,chunk_size=chunk_size,npts=npts, is_active_reset = is_active_reset)
                    
        self.last_prog = self.pinopi_2q_prog
        
        
    def run_pinopi_2q(self, is_update_readout_analyzer = False, is_kmeans = True, is_fit_circle = True, is_plot = True, is_save_data = None):
        
        if is_save_data is None: is_save_data = self.is_save_data
        results_dict = self.results['pinopi_2q']
        self.qm_server.clear_all_job_results()
        job = self.qm.execute(self.pinopi_2q_prog, duration_limit=0, data_limit=0)
        job.result_handles.wait_for_all_values()
        self.last_job = job
    
        
        I_00,_ = self.process_data(data = [job.result_handles.get('I00').fetch_all()['value'], job.result_handles.get('Q00').fetch_all()['value']], is_mean = False, is_calc_stat_error=False, which_data = 'I')
        Q_00,_ = self.process_data(data = [job.result_handles.get('I00').fetch_all()['value'], job.result_handles.get('Q00').fetch_all()['value']], is_mean = False, is_calc_stat_error=False, which_data = 'Q')
        I_01,_ = self.process_data(data = [job.result_handles.get('I01').fetch_all()['value'], job.result_handles.get('Q01').fetch_all()['value']], is_mean = False, is_calc_stat_error=False, which_data = 'I')
        Q_01,_ = self.process_data(data = [job.result_handles.get('I01').fetch_all()['value'], job.result_handles.get('Q01').fetch_all()['value']], is_mean = False, is_calc_stat_error=False, which_data = 'Q')
        I_10,_ = self.process_data(data = [job.result_handles.get('I10').fetch_all()['value'], job.result_handles.get('Q10').fetch_all()['value']], is_mean = False, is_calc_stat_error=False, which_data = 'I')
        Q_10,_ = self.process_data(data = [job.result_handles.get('I10').fetch_all()['value'], job.result_handles.get('Q10').fetch_all()['value']], is_mean = False, is_calc_stat_error=False, which_data = 'Q')
        I_11,_ = self.process_data(data = [job.result_handles.get('I11').fetch_all()['value'], job.result_handles.get('Q11').fetch_all()['value']], is_mean = False, is_calc_stat_error=False, which_data = 'I')
        Q_11,_ = self.process_data(data = [job.result_handles.get('I11').fetch_all()['value'], job.result_handles.get('Q11').fetch_all()['value']], is_mean = False, is_calc_stat_error=False, which_data = 'Q')
        
        N_avg = results_dict['N_avg']
        
            
        results_dict['I_00'] = I_00
        results_dict['I_01'] = I_01
        results_dict['I_10'] = I_10
        results_dict['I_11'] = I_11
        results_dict['Q_00'] = Q_00
        results_dict['Q_01'] = Q_01
        results_dict['Q_10'] = Q_10
        results_dict['Q_11'] = Q_11
        results_dict['is_kmeans'] = is_kmeans
        results_dict['is_update_readout_analyzer'] = is_update_readout_analyzer
        
        if results_dict['meas_type'] == 'sliced':
            
            I_00 = I_00.reshape(N_avg,-1).sum(1)
            Q_00 = Q_00.reshape(N_avg,-1).sum(1)
            I_01 = I_01.reshape(N_avg,-1).sum(1)
            Q_01 = Q_01.reshape(N_avg,-1).sum(1)
            I_10 = I_10.reshape(N_avg,-1).sum(1)
            Q_10 = Q_10.reshape(N_avg,-1).sum(1)
            I_11 = I_11.reshape(N_avg,-1).sum(1)
            Q_11 = Q_11.reshape(N_avg,-1).sum(1)
        
        hist_00, hist_00_err = self.process_data(data = [I_00, Q_00], is_mean = False); b00 = np.mean(hist_00)
        hist_01, hist_01_err  = self.process_data(data = [I_01, Q_01], is_mean = False); b01 = np.mean(hist_01)
        hist_10, hist_10_err = self.process_data(data = [I_10, Q_10], is_mean = False); b10 = np.mean(hist_10)
        hist_11, hist_11_err = self.process_data(data = [I_11, Q_11], is_mean = False); b11 = np.mean(hist_11)
        
        
        if is_fit_circle:
            circ = fit_circle([I_00.mean(), I_01.mean(), I_10.mean(), I_11.mean()],[Q_00.mean(), Q_01.mean(), Q_10.mean(), Q_11.mean()])
        else:
            circ = None
            
        results_dict['circle'] = circ
        if is_update_readout_analyzer:
            is_ascending =  b01 > b00
            if is_kmeans:
                I_all = np.concatenate([I_00, I_01, I_10, I_11])
                Q_all = np.concatenate([Q_00, Q_01, Q_10, Q_11])
                
                if circ is not None:
                    I_all -= circ[0]
                    Q_all -= circ[1]
                    offset = [circ[0],circ[1]]
                else:
                    offset=[0,0]
                phase_all,_ = self.process_data(data = [I_all, Q_all], is_mean = False, which_data = 'Phase')
                mag_all,_ = self.process_data(data = [I_all, Q_all], is_mean = False, which_data = 'Mag')
                
                if self.which_data == 'Phase':
                    data_all = phase_all
                elif self.which_data == 'Mag':
                    data_all = mag_all
                elif self.which_data == 'I':
                    data_all = I_all
                elif self.which_data == 'Q':
                    data_all = Q_all
                kmeans = KMeans(n_clusters = 4).fit([[0,d] for d in data_all])
                means = kmeans.cluster_centers_
                means = np.sort(means.transpose()[1])
                
                results_dict['means'] = means
                _,axs = plt.subplots(1,2)
                
                labels = kmeans.labels_
                plt.sca(axs[0])
                for k,c in zip(range(4),['b','r','c','m']):
                    plt.plot(I_all[labels==k], Q_all[labels==k], '.', color = c)
                plt.sca(axs[1])
                num_of_bins = 200
                bins = np.histogram(data_all, bins=num_of_bins)[1]
                for k,c in zip(range(4),['b','r','c','m']):
                    plt.hist(data_all[labels==k], bins = bins, histtype = 'step', color = c)
                    
                if not is_ascending:
                    means = np.flip(means)
                thresh = [np.mean([means[0],means[1]], dtype = float), np.mean([means[1],means[2]], dtype = float), np.mean([means[2],means[3]], dtype = float)]
            else:
                thresh = [np.mean([b00,b01], dtype = float), np.mean([b01,b10], dtype = float), np.mean([b10,b11], dtype = float)]
                offset = [0,0]
            self.update_readout_analyzer(qubit = [self.main_qubit, self.scnd_qubit], thresh = thresh, is_ascending = is_ascending, offset = offset)
        
        # betaII = (b00 + b01 + b10 + b11) / 4
        # betaZI = (b00 + b01 - b10 - b11) / 4
        # betaIZ = (b00 + b01 + b10 - b11) / 4
        # betaZZ = (b00 - b01 - b10 + b11) / 4
        
        if is_save_data: self.pickle_save(self.results['pinopi_2q'], 'pinopi_2q')
        
        print(histogram_fidelity(hist_00,hist_01,hist_10,hist_11))
        
        if is_plot:
            self.plot_pinopi_2q()
            
            
    def plot_pinopi_2q(self):
            
        results_dict = self.results['pinopi_2q']
        circ = results_dict['circle']
        I_00 = results_dict['I_00'] 
        I_01 = results_dict['I_01']
        I_10 = results_dict['I_10'] 
        I_11 = results_dict['I_11'] 
        Q_00 = results_dict['Q_00'] 
        Q_01 = results_dict['Q_01'] 
        Q_10 = results_dict['Q_10']
        Q_11 = results_dict['Q_11'] 
        N_avg = results_dict['N_avg']
        npts = results_dict['npts']
        
        
        if results_dict['meas_type'] == 'sliced':
            I_00_t = I_00.reshape(N_avg,-1).mean(0).cumsum()
            Q_00_t = Q_00.reshape(N_avg,-1).mean(0).cumsum(0)
            I_01_t = I_01.reshape(N_avg,-1).mean(0).cumsum(0)
            Q_01_t = Q_01.reshape(N_avg,-1).mean(0).cumsum(0)
            I_10_t = I_10.reshape(N_avg,-1).mean(0).cumsum(0)
            Q_10_t = Q_10.reshape(N_avg,-1).mean(0).cumsum(0)
            I_11_t = I_11.reshape(N_avg,-1).mean(0).cumsum(0)
            Q_11_t = Q_11.reshape(N_avg,-1).mean(0).cumsum(0)
            
            I_00 = I_00.reshape(N_avg,-1).sum(1)
            Q_00 = Q_00.reshape(N_avg,-1).sum(1)
            I_01 = I_01.reshape(N_avg,-1).sum(1)
            Q_01 = Q_01.reshape(N_avg,-1).sum(1)
            I_10 = I_10.reshape(N_avg,-1).sum(1)
            Q_10 = Q_10.reshape(N_avg,-1).sum(1)
            I_11 = I_11.reshape(N_avg,-1).sum(1)
            Q_11 = Q_11.reshape(N_avg,-1).sum(1)
        
        if circ is not None:
            offset = [circ[0],circ[1]]
        else:
            offset = [0,0]
        
        
        is_kmeans = results_dict['is_kmeans']
        is_update_readout_analyzer = results_dict['is_update_readout_analyzer']
        if is_kmeans and is_update_readout_analyzer: means = results_dict['means'] 
        
        hist_00, hist_00_err = self.process_data(data = [I_00-offset[0], Q_00-offset[1]], is_mean = False); b00 = np.mean(hist_00)
        hist_01, hist_01_err  = self.process_data(data = [I_01-offset[0], Q_01-offset[1]], is_mean = False); b01 = np.mean(hist_01)
        hist_10, hist_10_err = self.process_data(data = [I_10-offset[0], Q_10-offset[1]], is_mean = False); b10 = np.mean(hist_10)
        hist_11, hist_11_err = self.process_data(data = [I_11-offset[0], Q_11-offset[1]], is_mean = False); b11 = np.mean(hist_11)
        
        fig, axs = plt.subplots(1,2, figsize = [10,8])
        
        plt.sca(axs[0])
        if results_dict['meas_type'] == 'sliced':
            plt.plot(I_00_t,Q_00_t, '-b', alpha=0.5)
            plt.plot(I_01_t,Q_01_t, '-r', alpha=0.5)
            plt.plot(I_10_t,Q_10_t, '-c', alpha=0.5)
            plt.plot(I_11_t,Q_11_t, '-m', alpha=0.5)
        plt.plot(I_00,Q_00,'.b',label='00', alpha = 0.15)
        plt.plot(I_01,Q_01,'.r',label='01', alpha = 0.15)
        plt.plot(I_10,Q_10,'.c',label='10', alpha = 0.15)
        plt.plot(I_11,Q_11,'.m',label='11', alpha = 0.15)
        plt.plot([0],[0],'xk')
        
        if circ is not None and is_update_readout_analyzer: 
            x0,y0,r = circ
            phi = np.linspace(0,2*np.pi,100)
            plt.plot(x0+r*np.cos(phi), y0+r*np.sin(phi), '-k')
            R = np.linspace(0,r,2)
            th = self.readout_analyzer_dict['qubits']['2qubits']['thresh']
            for t in th:
                plt.plot(x0+R*np.cos(t), y0+R*np.sin(t), '--k')
        elif is_kmeans and is_update_readout_analyzer:
            th = self.readout_analyzer_dict['qubits']['2qubits']['thresh']
            if self.which_data=='Phase':
                r = np.max(np.sqrt(I_00**2+Q_00**2))
                R = np.linspace(0,r,2)
                for t in th:
                    plt.plot(R*np.cos(t), R*np.sin(t), '--k')
            elif self.which_data == 'I':
                yl = plt.ylim()
                for thresh in th:
                    plt.plot([thresh]*2, yl, '--k')
                plt.ylim(yl)
            elif self.which_data == 'Q':
                xl = plt.xlim()
                for thresh in th:
                    plt.plot(xl, [thresh]*2, '--k')
                plt.xlim(xl)
            elif self.which_data == 'Mag':
                phi = np.linspace(0,2*np.pi,100)
                for thh in th:
                    plt.plot(tth*np.cos(phi), tth*np.sin(phi), '--k')
                
            
        plt.legend()
        plt.xlabel('I [V]')
        plt.ylabel('Q [V]')
        
        plt.sca(axs[1])
        
        num_of_bins = 200
        bins = np.histogram(np.hstack((hist_00, hist_01, hist_10,hist_11)), bins=num_of_bins)[1]
        
        hist_00_data = plt.hist(hist_00,
                bins = bins,
                histtype='step',
                color = 'b')
        hist_01_data = plt.hist(hist_01,
                 bins = bins,
                 histtype='step',
                 color = 'r')
        hist_10_data = plt.hist(hist_10,
                bins = bins,
                histtype='step',
                color = 'c')
        hist_11_data = plt.hist(hist_11,
                 bins = bins,
                 histtype='step',
                 color = 'm')
        
        plt.ylabel("counts", fontsize=18)
        xlabel = self.which_data
        if self.which_data == 'Phase':
            xlabel += ' [Rad]'
        else:
            xlabel +=' [V]'
        plt.xlabel(xlabel, fontsize=18)
        plt.legend(['00','01','10','11'])
        
        ylim = plt.ylim()
        plt.plot([b00,b00],ylim, '--b')
        plt.plot([b01,b01],ylim, '--r')
        plt.plot([b10,b10],ylim, '--c')
        plt.plot([b11,b11],ylim, '--m')
        
        if is_kmeans and is_update_readout_analyzer:
            for m in means:
                plt.plot([m,m],ylim, '--k')
        plt.ylim(ylim)
    
        fig.text(0.4, 0.95, "Pi-Pi,", ha="center", va="bottom", size="large",color="magenta")
        fig.text(0.5, 0.95, "no Pi-Pi,", ha="center", va="bottom", size="large",color="cyan")
        fig.text(0.62, 0.95, "Pi-no Pi,", ha="center", va="bottom", size="large",color="blue")
        fig.text(0.765, 0.95, "no Pi-no Pi", ha="center", va="bottom", size="large",color="red")
        plt.suptitle('', fontsize=18)
        
        plt.tight_layout()
#%% T1 specific
       
    def load_T1(self, npts = 51, # How many points I measure
               max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
               min_seq_time = None,
               N_avg = 1000, # How many times each sequence is executed 
               qubit = None,
               ro_element = None,
               is_sb_cool = None,
               is_active_reset = None,
               **kwargs):
        
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if is_sb_cool is None: is_sb_cool = self.is_sb_cool
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        max_seq_clks= int(max_seq_time // 4)
        if min_seq_time is not None: 
            step_size_clks = (max_seq_time - min_seq_time) // 4 // (npts-1)
            t = np.arange(min_seq_time, max_seq_time + 1, step_size_clks*4)
        else:   
            step_size_clks = max_seq_clks // (npts-1)
            min_seq_time = step_size_clks * 4
            t = np.arange(step_size_clks*4, max_seq_time + 1, step_size_clks*4)
        self.results['T1'] = {'t': t,
                           'qubit': qubit,
                           'ro_element': ro_element,
                           'wait_between_seq': self.wait_between_seq}
        self.T1_N_avg = N_avg
     
        if is_sb_cool: run_time = N_avg*npts * (max_seq_time/2+self.pulse_len(ro_element, self.ro_pulse) + self.pulse_len(qubit, 'rabi_pulse'))
        elif is_active_reset: run_time = N_avg*npts * (max_seq_time/2+self.pulse_len(ro_element, self.ro_pulse))
        else:  run_time = N_avg*npts * (max_seq_time/2+self.pulse_len(ro_element, self.ro_pulse) + self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9))) 
        
        with program() as self.T1_prog:
            wt_time = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            with for_(n, 0, n < N_avg, n + 1):
                with for_(wt_time, min_seq_time//4, wt_time <= max_seq_clks, wt_time + step_size_clks):
                    
                    if self.e_to_f:  self.map_e_to_g(self.main_qubit_g_to_e)
                    
                    self._pi_pulse(qubit = qubit, **kwargs)
                    
                    wait(wt_time, qubit)
                    align(qubit, ro_element)                    
                    if self.e_to_f and self.e_to_f_map_back:  self.map_e_to_g(self.main_qubit_g_to_e)
                    
                    self.perform_full_measurement(I,Q, ro_element = ro_element, is_wait = not is_sb_cool, is_active_reset = is_active_reset, is_sb_cool = is_sb_cool, **kwargs)
        
        self.last_prog = self.T1_prog
        
        
    def run_T1(self, is_continue = False, is_save_data = None, **kwargs): 
    
        if not hasattr(self, 'T1_prog'): raise ValueError("No T1 program defined!")
        
        results_dict = self.results['T1']
        prog_name = 'T1'
        if is_continue and 'I' in results_dict.keys():
            results_dict['I'], results_dict['Q'] = self.continue_run(self.T1_prog, self.T1_N_avg, results_dict)
            fig_num = last_fig_num_by_name(prog_name)(prog_name)
        else:  
            results_dict['I'], results_dict['Q'] = self.run_prog(self.T1_prog, self.T1_N_avg, **kwargs)
            fig_num = next_fig_num_by_name(prog_name)
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'T1')
        return self.plot_T1(fig_num = fig_num, is_continue = is_continue, prog_name = prog_name, **kwargs)

    def plot_T1(self, is_hist2d = False, **kwargs):
        results_dict = self.results['T1']
        if is_hist2d: self.plot_hist2d(results_dict['t'], self.determine_data(results_dict['I'], results_dict['Q']), **kwargs)
        return self.fit_and_plot('Exp', self.process_data(results_dict, **kwargs), results_dict['t'], title_str = f'T1 ({results_dict["qubit"]},{results_dict["ro_element"]})', **kwargs)
    
    
    def T1_complete(self, npts = 50, # How many points I measure
               max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
               N_avg = 1000, # How many times each sequence is executed 
               **kwargs):
        
        self.load_T1(npts=npts,max_seq_time= max_seq_time, N_avg=N_avg,**kwargs)
        
        return self.run_T1()
    

    def load_driven_T1(self, npts = 51, # How many points I measure
               max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
               amp_scale_start = 0,
               amp_scale_stop = 1,
               amp_scale_npts = 11,
               drive_element = None,
               drive_pulse = 'mixer_cal_pulse',
               N_avg = 1000, # How many times each sequence is executed 
               qubit = None,
               ro_element = None,
               is_ramp = None,
               is_sb_cool = None,
               is_active_reset = None,
               drive_detuning = 0,
               ring_down_time = 0,
               **kwargs):
        
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if drive_element is None: drive_element = self.main_readout
        if is_sb_cool is None: is_sb_cool = self.is_sb_cool
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        amp_scale_list = np.linspace(amp_scale_start, amp_scale_stop, amp_scale_npts)
        
        max_seq_clks= int(max_seq_time // 4)
        step_size_clks = max_seq_clks // (npts-1)
        self.results['driven T1'] = {'t': np.arange(step_size_clks*4, max_seq_time + 1, step_size_clks*4),
                               'N_avg': N_avg,
                               'qubit': qubit,
                               'ro_element': ro_element,
                               'drive_element': drive_element,
                               'drive_pulse': drive_pulse,
                               'wait_between_seq': self.wait_between_seq,
                               'amp': amp_scale_list * np.abs(self.pulse_amp(drive_element, drive_pulse))}
        if is_sb_cool: run_time = N_avg*npts*amp_scale_npts * (max_seq_time/2+self.pulse_len(ro_element, self.ro_pulse) + self.pulse_len(qubit, 'rabi_pulse'))
        elif is_active_reset: run_time = N_avg*npts*amp_scale_npts * (max_seq_time/2+self.pulse_len(ro_element, self.ro_pulse))
        else:  run_time = N_avg*npts*amp_scale_npts * (max_seq_time/2+self.pulse_len(ro_element, self.ro_pulse) + self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9))) 
        
        with program() as self.driven_T1_prog:
            wt_time = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            amp_scale = declare(fixed)
            with for_(n, 0, n < N_avg, n + 1):
                with for_each_(amp_scale, amp_scale_list):
                    with for_(wt_time, step_size_clks, wt_time <= max_seq_clks, wt_time + step_size_clks):
                        
                        if is_sb_cool: self.sideband_cool(qubit = qubit, ro_element = ro_element,  **kwargs)
                        
                        if self.e_to_f:  self.map_e_to_g(self.main_qubit_g_to_e)
                        
                        self._pi_pulse(qubit = qubit, **kwargs)
                        if drive_detuning != 0:
                            update_frequency(drive_element, self.element_IF(drive_element)+drive_detuning, keep_phase = True)
                        align(qubit, drive_element)
                        if is_ramp:
                            play((drive_pulse + "_ramp_up")* amp(amp_scale), drive_element)
                        play(drive_pulse * amp(amp_scale), drive_element, duration = wt_time)
                        if is_ramp:
                            play((drive_pulse + "_ramp_down")* amp(amp_scale), drive_element)
                        if ring_down_time > 0: wait(int(ring_down_time//4), ro_element)
                        if self.e_to_f and self.e_to_f_map_back:  self.map_e_to_g(self.main_qubit_g_to_e)
                        if drive_detuning != 0:
                            update_frequency(drive_element, self.element_IF(drive_element), keep_phase = True)
                        align(qubit,ro_element,drive_element)
                        self.perform_full_measurement(I,Q, ro_element = ro_element, is_wait = not is_sb_cool, is_active_reset = is_active_reset, **kwargs)
        
        self.last_prog = self.driven_T1_prog
        
        
    def run_driven_T1(self, is_save_data = None, is_plot_all = True, **kwargs): 
        if not hasattr(self,'driven_T1_prog'): raise ValueError("Idiot! You did not write program")
        results_dict = self.results['driven T1']
        
        qubit = results_dict['qubit']
        drive_element = results_dict['drive_element']
        drive_pulse = results_dict['drive_pulse']
        amp = results_dict['amp']
        
        self.qm_server.clear_all_job_results()
        job = self.qm.execute(self.driven_T1_prog, duration_limit=0, data_limit=0)
        job.result_handles.wait_for_all_values()
        job.execution_report()
        self.last_job = job
        
        N_avg = results_dict['N_avg']
        amp_list = results_dict['amp']
        amp_scale_npts = len(amp_list)
        
        I = job.result_handles.get('I').fetch_all()['value'].reshape((N_avg, amp_scale_npts,-1))
        Q = job.result_handles.get('Q').fetch_all()['value'].reshape((N_avg, amp_scale_npts,-1))
        
        processed_data = []
        stat_error = []
        for i in range(amp_scale_npts):
            data, err = self.process_data([I[:,i,:],Q[:,i,:]], is_calc_stat_error=True)
            processed_data.append(data)
            stat_error.append(err)
        processed_data = np.array(processed_data)
        stat_error = np.array(stat_error)
        decays = []
        decays_error = []
        for amp_data in zip(processed_data, stat_error):
            fit_res, fit_error,_ = self.fit_and_plot('Exp', amp_data, results_dict['t'], is_calc_stat_error=True, title_str = f'T1 with drive ({qubit})', is_return_fit = False, plot = False)
            decays.append(fit_res[0] if fit_res[0] is not None else 0)
            decays_error.append(fit_error[0] if fit_error[0] is not None else 0)
        decays=np.array(decays)
        decays_error = np.array(decays_error)[np.logical_and(decays!=np.nan, decays!=np.inf, decays!=None)]
        
        results_dict['data']=processed_data
        results_dict['error']=stat_error
        results_dict['decays']=decays
        results_dict['decays_error']=decays_error
        
        self.pickle_save(results_dict, meas_name = 'driven T1')
    
        return self.plot_driven_T1(is_plot_all = is_plot_all, **kwargs)

    def plot_driven_T1(self, is_plot_all = False, **kwargs):
        results_dict = self.results['driven T1']
        
        qubit = results_dict['qubit']
        if is_plot_all:
            for amp_data in zip(results_dict['data'], results_dict['error']):
                self.fit_and_plot('Exp', amp_data, results_dict['t'], is_calc_stat_error=True, title_str = f'T1 with drive ({qubit})', is_return_fit = False)
        
        def fit_func(x,A,B):
            return A*x**2+B
        
        amp_list = results_dict['amp']
        decays = results_dict['decays']
        decays_error = results_dict['decays_error']
            
        # fit_results_decay, cov_decay = curve_fit(fit_func, amp_list, 1/decays, sigma = decays_error/decays**2)
        # A_decay, A_decay_err = round_value_by_error(fit_results_decay[0], np.sqrt(cov_decay[0,0]))
        # B_decay, B_decay_err = round_value_by_error(fit_results_decay[1], np.sqrt(cov_decay[1,1]))
            
        plt.figure()
        plt.errorbar(x = amp_list, y = decays, yerr = decays_error, fmt = '.r', label = 'Data', capsize = 4, markersize=4)
        # plt.plot(amp_list, fit_func(amp_list,*fit_results_decay), '-b', label = 'Fit')
        plt.xlabel('Amplitude [V]')
        plt.ylabel('T1 $[\mu s]$')
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        return decays, decays_error
        
    
#%%  Ramsy specific
        
    def load_ramsey(self , 
                    npts = 51, #How many points I measure
                    max_seq_time = 30000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
                    min_seq_time = None,
                    N_avg = 1000, # How many times each sequence is executed 
                    detuning = 0.0005, # The detuned frequency, in units of GHz.
                    is_echo = False,
                    ZZ_interaction = False,
                    pi2_pulse: str= None,
                    pi_pulse: str= None,
                    seq = None,
                    qubit = None,
                    ro_element = None,
                    is_sb_cool = None,
                    is_active_reset = None,
                    prepare = None, prepare_kwargs = {},
                    **kwargs):
        """Time step overrides npts. Only multiples of 4 are possible."""        
        
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if min_seq_time is None:
            if max_seq_time//npts < 16: raise ValueError(f"The minimum wait time is 16ns and your first step is {max_seq_time/npts}")
        else:
            if min_seq_time < 16: raise ValueError(f"The minimum wait time is 16ns and your first step is {max_seq_time/npts}")
            
        if pi2_pulse is None: pi2_pulse=self.pi2_pulse
        if pi_pulse is None: pi_pulse=self.pi_pulse
        if is_sb_cool is None: is_sb_cool = self.is_sb_cool
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        if is_echo: 
            self.results['ramsey echo'] = {}
            self.results['ramsey echo']['detuning'] = detuning
            self.results['ramsey echo']['N_avg'] = N_avg         
            self.results['ramsey echo']['is_ZZ'] = ZZ_interaction
            self.results['ramsey echo']['qubit'] = qubit
            self.results['ramsey echo']['ro_element'] = ro_element
            results_dict = self.results['ramsey echo']
        else:
            self.results['ramsey'] = {}
            self.results['ramsey']['N_avg'] = N_avg
            self.results['ramsey']['detuning'] = detuning
            self.results['ramsey']['is_ZZ'] = ZZ_interaction
            self.results['ramsey']['qubit'] = qubit
            self.results['ramsey']['ro_element'] = ro_element
            results_dict = self.results['ramsey']
        
        self.is_ramsey_echo = is_echo
                
        if is_sb_cool: run_time = N_avg*npts * ((max_seq_time)/2+self.pulse_len(ro_element, self.ro_pulse) + self.pulse_len(qubit, 'rabi_pulse'))
        elif is_active_reset: run_time = N_avg*npts * (max_seq_time/2+self.pulse_len(ro_element, self.ro_pulse))
        else: run_time = N_avg*npts * ((max_seq_time)/2+self.pulse_len(ro_element, self.ro_pulse) + self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))     
        
        if min_seq_time is None:
            times = np.arange(max_seq_time//npts, max_seq_time+1, max_seq_time//npts)//4*4
        else:
            if (max_seq_time-min_seq_time)//(npts-1) <4:
                print("Time step is too small. Seting the time step to 4ns")
                npts = (max_seq_time-min_seq_time)//4
            times = np.arange(min_seq_time, max_seq_time+1, (max_seq_time-min_seq_time)//(npts-1))//4*4
        if npts !=len(times): 
            npts = len(times)
            print(f'Changed <npts> to {npts} to round up the times.')
            
        times_to_wait = times
        if is_echo: times_to_wait = times / 2
            
        phase_off_list = times * detuning * 2 * np.pi
        cos_list = np.cos(phase_off_list)
        sin_list = np.sin(phase_off_list)
        times_to_wait = times_to_wait.astype(int)
        results_dict['t'] = times
        
        with program() as self.ramsey_prog:
        
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            ti = declare(int)
            sin = declare(fixed)
            cos = declare(fixed)
            phase_off = declare(fixed)
            
            with for_(n, 0, n < N_avg, n + 1): 
                
                with for_each_((ti, cos,sin), ((times_to_wait//4).tolist(), cos_list,sin_list)):
                    reset_frame(qubit)
                    if prepare is not None: prepare(**prepare_kwargs)

                    if self.e_to_f:  self.map_e_to_g(self.main_qubit_g_to_e)
                    
                    if results_dict['is_ZZ']: #excite the 2nd qubit to measure (chi~)*sigmaZ*sigmaZ
                        self._pi_pulse(qubit = self.scnd_qubit, **kwargs)
                        
                    play(pi2_pulse, qubit)
                    
                    wait(ti, qubit)
                    
                    if is_echo: self._pi_pulse(qubit = qubit, **kwargs)

                    if is_echo: wait(ti, qubit)
                
                    if results_dict['is_ZZ']: #return the 2nd qubit to ground
                        align(qubit, self.scnd_qubit)
                        self._pi_pulse(qubit = self.scnd_qubit,**kwargs)
                        align(self.scnd_qubit, ro_element)
                        
                    play(pi2_pulse * amp(cos,-sin,sin,cos), qubit)
                    
                    if self.e_to_f and self.e_to_f_map_back:  self.map_e_to_g(self.main_qubit_g_to_e)
                    
                    align(qubit, ro_element)
                    
                    self.perform_full_measurement(I,Q, ro_element = ro_element, is_sb_cool = is_sb_cool, is_active_reset = is_active_reset, **kwargs)
                    
        self.last_prog = self.ramsey_prog
        
    def run_ramsey(self, is_continue = False, is_save_data = None, **kwargs): 
    
        if not hasattr(self, 'ramsey_prog'): raise ValueError("Idiot! You did not write ramsey program")
        
        if self.is_ramsey_echo:
            results_dict = self.results['ramsey echo']
            prog_name = 'Ramsey Echo'
            if is_continue and 'I' in results_dict.keys():
                results_dict['I'], results_dict['Q'] = self.continue_run(self.ramsey_prog, results_dict['N_avg'], results_dict)
                fig_num = last_fig_num_by_name(prog_name)(prog_name)
            else: 
                results_dict['I'], results_dict['Q'] = self.run_prog(self.ramsey_prog, results_dict['N_avg'], **kwargs)
                fig_num = next_fig_num_by_name(prog_name)
            if is_save_data is None: is_save_data = self.is_save_data
            if is_save_data: self.pickle_save(results_dict, 'RamseyEcho')
            return self.plot_ramsey_echo(fig_num = fig_num, is_continue = is_continue, prog_name = prog_name, **kwargs)
        else:
            results_dict = self.results['ramsey']
            prog_name = 'Ramsey'
            if is_continue and 'I' in results_dict.keys():
                results_dict['I'], results_dict['Q'] = self.continue_run(self.ramsey_prog, results_dict['N_avg'], results_dict)
                fig_num = last_fig_num_by_name(prog_name)(prog_name)
            else: 
                results_dict['I'], results_dict['Q'] = self.run_prog(self.ramsey_prog, results_dict['N_avg'], **kwargs)
                fig_num = next_fig_num_by_name(prog_name)
                if is_save_data is None: is_save_data = self.is_save_data
                if is_save_data: self.pickle_save(results_dict, 'Ramsey')
            return self.plot_ramsey(fig_num = fig_num, is_continue = is_continue, prog_name = prog_name, **kwargs)
        

    def plot_ramsey(self, is_hist2d = False, **kwargs):
        results_dict = self.results['ramsey']
        title_str =f'Ramsey ({results_dict["qubit"]})'
        if results_dict['is_ZZ']: title_str = title_str + ' with ZZ interaction'
        
        if is_hist2d: self.plot_hist2d(results_dict['t'], self.determine_data(results_dict['I'], results_dict['Q']), **kwargs)
        return self.fit_and_plot('ExpCos', self.process_data(results_dict,**kwargs), results_dict['t'], title_str = title_str, txt = 'detuning = {} [MHz]'.format(results_dict['detuning']*1e3), **kwargs)
    
    def plot_ramsey_echo(self, is_hist2d = False, **kwargs):
        results_dict = self.results['ramsey echo']
        title_str = f'Ramsey echo ({results_dict["qubit"]})'
        if results_dict['is_ZZ']: title_str = title_str + ' with ZZ interaction'


        if is_hist2d: self.plot_hist2d(results_dict['t'], self.determine_data(results_dict['I'], results_dict['Q']), **kwargs)
        return self.fit_and_plot('ExpCos', self.process_data(results_dict, **kwargs), results_dict['t'], title_str = title_str,txt = f'detuning = {results_dict["detuning"]*1e3} [MHz]', **kwargs)
    
     
    def ramsey_complete(self, **kwargs):
        
        self.load_ramsey(**kwargs)

        return self.run_ramsey(**kwargs)
    
    
    
    def load_driven_ramsey(self, 
                    npts = 51, #How many points I measure
                    max_seq_time = 30000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
                    min_seq_time = None,
                    N_avg = 1000, # How many times each sequence is executed 
                    detuning = 0.0005, # The detuned frequency, in units of GHz.
                    amp_scale_start = 0,
                    amp_scale_stop = 1,
                    amp_npts = 11,
                    is_echo = False,
                    ring_down_time = 0,
                    extra_drive_time = 0,
                    pi2_pulse = None,
                    pi_pulse = None, 
                    drive_element = 'ro',
                    drive_pulse = None,
                    drive_element2 = None,
                    drive_pulse2 = None,
                    drive_detuning = 0,
                    drive_detuning2 = 0,
                    is_ramp = True,
                    qubit = None,
                    ro_element = None,
                    is_sb_cool = None,
                    which_sb_cool = None,
                    sb_cool_kwargs = None,
                    is_active_reset = None,
                    tof_diff = 0,
                    **kwargs):
        
        if drive_pulse is None: drive_pulse = 'mixer_cal_pulse'
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if is_sb_cool is None: is_sb_cool = self.is_sb_cool
        if is_sb_cool:
            if sb_cool_kwargs is None: sb_cool_kwargs = self.sb_cool_kwargs
            if which_sb_cool is None: which_sb_cool = [sb_cool_kwargs[0]['mm']]
        if is_active_reset is None: is_active_reset = self.is_active_reset
        if drive_pulse2 is None: drive_pulse2 = drive_pulse
        
        if pi2_pulse is None: pi2_pulse=self.pi2_pulse
        if pi_pulse is None: pi_pulse=self.pi_pulse
        max_seq_clks= int(max_seq_time // 4)
        

        first_step_clk = max_seq_clks // (npts-1)
        
        amp_scale_list = np.linspace(amp_scale_start, amp_scale_stop, amp_npts, dtype=float)
        amp_list = amp_scale_list*self.pulse_amp(drive_element, drive_pulse)
        
        if is_sb_cool:
            rabi_time = 0
            for cool_dict in sb_cool_kwargs:
                if cool_dict['mm'] in which_sb_cool:
                    rabi_time+=cool_dict['sb_duration']
                else:
                    continue
            run_time = N_avg*npts*amp_npts * (self.pulse_len(ro_element, self.ro_pulse)+self.pulse_len(drive_element, drive_pulse) + rabi_time+self.pulse_len(drive_element, drive_pulse))
        elif is_active_reset: run_time = N_avg*npts*amp_npts * (self.pulse_len(ro_element, self.ro_pulse)+self.pulse_len(drive_element, drive_pulse))
        else: run_time = N_avg*npts*amp_npts * (self.pulse_len(ro_element, self.ro_pulse) +self.pulse_len(drive_element, drive_pulse) + self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))     
        
        
        if min_seq_time is None:
            times = np.arange(max_seq_time//npts, max_seq_time+1, max_seq_time//npts)//4*4
        else:
            if min_seq_time < 4:
                raise ValueError(f'Minimum sequence time can not be shorter than 4ns. Got {min_seq_time}')
            if (max_seq_time-min_seq_time)//(npts-1) <4:
                print("Time step is too small. Seting the time step to 4ns")
                npts = (max_seq_time-min_seq_time)//4
            times = np.arange(min_seq_time, max_seq_time+1, (max_seq_time-min_seq_time)//(npts-1))//4*4
        if npts !=len(times): 
            npts = len(times)
            print(f'Changed <npts> to {npts} to round up the times.')
            
        if is_ramp:
            times_to_wait1 = (int(self.pulse_ramp_len(drive_element, drive_pulse)+ self.pulse_len(drive_element, drive_pulse)-tof_diff-self.pulse_len(qubit, pi2_pulse)*2)-times- extra_drive_time).astype(int)
        else:
            times_to_wait1 = (int(self.pulse_len(drive_element, drive_pulse)-tof_diff-self.pulse_len(qubit, pi2_pulse)*2)-times- extra_drive_time).astype(int)
        times_to_wait2 = times.astype(int)
        if is_echo:
            times_to_wait1 -= self.pulse_len(qubit, 'pi_pulse') 
            times_to_wait2 = times_to_wait2//2
            if times_to_wait1[-1]<16:
                raise ValueError(f"Wait time is too short. Minimum is 16 and got {times_to_wait1[-1]}")
                
        if is_echo:
            self.results['driven ramsey echo'] = {'N_avg': N_avg,
                                            'detuning':detuning,
                                            'drive_element':drive_element,
                                            'drive_element2':drive_element2,
                                            'drive_pulse':drive_pulse,
                                            'drive_pulse2':drive_pulse2,
                                            'drive_detuning':drive_detuning,
                                            'drive_detuning2':drive_detuning2,
                                            'ro_element':ro_element,
                                             'amp_list':amp_list,
                                             't': times,
                                             'qubit': qubit,
                                             'ring_down_time': ring_down_time,
                                             'is_active_reset': is_active_reset,
                                             'is_sb_cool': is_sb_cool,
                                             'ring_down_time': ring_down_time,
                                             'is_ramp': is_ramp
                                             }
            results_dict = self.results['driven ramsey echo']
            self.driven_ramsey_echo_flag = True
        else:
            self.results['driven ramsey'] = {'N_avg': N_avg,
                                                'detuning':detuning,
                                                'drive_element':drive_element,
                                                'drive_element2':drive_element2,
                                                'drive_pulse':drive_pulse,
                                                'drive_pulse2':drive_pulse2,
                                                'drive_detuning':drive_detuning,
                                                'drive_detuning2':drive_detuning2,
                                                'ro_element':ro_element,
                                                 'amp_list':amp_list,
                                                 't': times,
                                                 'qubit': qubit,
                                                 'ring_down_time': ring_down_time,
                                                 'is_active_reset': is_active_reset,
                                                 'is_sb_cool': is_sb_cool,
                                                 'ring_down_time': ring_down_time,
                                                 'is_ramp': is_ramp
                                                 }
            results_dict = self.results['driven ramsey']
            self.driven_ramsey_echo_flag = False
            
        # phase_off_list = np.mod(times * detuning, 1)
        phase_off_list = times * detuning * 2 * np.pi
        cos_list = np.cos(phase_off_list)
        sin_list = np.sin(phase_off_list)
        phase_off_list = np.mod(times * detuning ,1).astype(float)
        results_dict['t'] = times
        
        # if not self.is_loop_amp_scale:
        #     self._load_py_loop_driven_ramsey_prog(times_to_wait1, times_to_wait2, cos_list, sin_list)
        # else:
        with program() as self.driven_ramsey_prog:
        
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            ti1 = declare(int)
            ti2 = declare(int)
            phase_off = declare(fixed)
            sin = declare(fixed)
            cos = declare(fixed)
            amp_scale = declare(fixed)
            with for_(n, 0, n < N_avg, n + 1): 
                with for_each_(amp_scale, amp_scale_list.tolist()):
                    # with for_each_((ti, phase_off), (times_to_wait//4, phase_off_list)):
                    with for_each_((ti1, ti2, cos, sin), (times_to_wait1//4, times_to_wait2//4, cos_list, sin_list)):
                        reset_frame(qubit)
                        reset_frame(drive_element)
                        if drive_detuning != 0 : update_frequency(drive_element, self.element_IF(drive_element)+drive_detuning)
                        if drive_detuning2 != 0 and drive_element2 is not None: update_frequency(drive_element2, self.element_IF(drive_element2)+drive_detuning2)
                        # align(drive_element, qubit, ro_element)
                        if amp_npts>1:
                            if is_ramp:
                                play((drive_pulse + '_ramp_up') * amp(amp_scale), drive_element)
                                if drive_element2 is not None: play((drive_pulse2 + '_ramp_up')* amp(amp_scale), drive_element2)
                            play(drive_pulse * amp(amp_scale), drive_element)
                            if drive_element2 is not None: play(drive_pulse2 * amp(amp_scale), drive_element2)
                            if is_ramp:
                                play((drive_pulse + '_ramp_down') * amp(amp_scale), drive_element)
                                if drive_element2 is not None: play(drive_pulse2 + '_ramp_down', drive_element2)
                        else:
                            if is_ramp:
                                play((drive_pulse + '_ramp_up') * amp(amp_scale_stop), drive_element)
                                if drive_element2 is not None: play((drive_pulse2 + '_ramp_up') * amp(amp_scale_stop), drive_element2)
                            play(drive_pulse * amp(amp_scale_stop), drive_element)
                            if drive_element2 is not None: play(drive_pulse2 * amp(amp_scale_stop), drive_element2)
                            if is_ramp:
                                play((drive_pulse + '_ramp_down') * amp(amp_scale_stop), drive_element)
                                if drive_element2 is not None: play((drive_pulse2 + '_ramp_down') * amp(amp_scale_stop), drive_element2)
                        if drive_detuning != 0 : update_frequency(drive_element, self.element_IF(drive_element), keep_phase = True)
                        if drive_detuning2 != 0 and drive_element2 is not None: update_frequency(drive_element2, self.element_IF(drive_element2), keep_phase = True)
                        wait(ti1, qubit)
                        play(pi2_pulse, qubit)
                        wait(ti2, qubit)
                        if is_echo:
                            play('pi_pulse', qubit)
                            wait(ti2, qubit)
                        play(pi2_pulse * amp(cos, -sin, sin, cos), qubit)
                        
                        align(qubit, ro_element)
                        if drive_element != ro_element:
                            align(drive_element, ro_element)
                            
                        if ring_down_time>0: wait(ring_down_time//4, ro_element)
                        # if drive_element!= ro_element: align(drive_element, ro_element)
                        # if drive_element2 is not None and drive_element2!= ro_element: align(drive_element2, ro_element)
                        self.perform_full_measurement(I,Q, ro_element = ro_element, is_sb_cool = is_sb_cool, sb_cool_kwargs = sb_cool_kwargs, which_sb_cool = which_sb_cool, is_active_reset = is_active_reset, **kwargs)
    
        self.last_prog = self.driven_ramsey_prog
        
    def run_driven_ramsey(self, is_echo = None, kappa = None, chi = None, chi_err = 0, drive_detuning = None, is_plot_all = True, is_plot_fft = False, is_save_data = None, **kwargs):
        if not hasattr(self,'driven_ramsey_prog'): raise ValueError("Idiot! You did not write program")
        
        if is_echo is None: is_echo =  self.driven_ramsey_echo_flag
        if is_echo: results_dict = self.results['driven ramsey echo']
        else: results_dict = self.results['driven ramsey']
        results_dict['kappa'] = kappa
        results_dict['chi'] = chi
        results_dict['drive_detuning'] = drive_detuning
        qubit = results_dict['qubit']
        drive_element = results_dict['drive_element']
        drive_pulse = results_dict['drive_pulse']
        detuning = results_dict['detuning']
        
        N_avg = results_dict['N_avg']
        amp_list = results_dict['amp_list']
        amp_scale_npts = len(amp_list)
        time_npts = len(results_dict['t'])
        I,Q = self.run_prog(self.driven_ramsey_prog, shape = (N_avg, amp_scale_npts,-1), **kwargs)
        # if self.is_loop_amp_scale:
        #     I,Q = self.run_prog(self.driven_ramsey_prog, shape = (N_avg, amp_scale_npts,-1), **kwargs)
        # else:   
        #     prog_id = self.qm.compile(self.driven_ramsey_prog)
        #     amp_scale_list = np.abs(results_dict['amp_list']/self.pulse_amp(results_dict['drive_element'], results_dict['drive_pulse']))
        #     if len(amp_scale_list) == 1:
        #         I,Q = self.run_prog(self.driven_ramsey_prog, shape = (N_avg, amp_scale_npts,-1), **kwargs)
        #     else:
        #         I = np.zeros((N_avg, amp_scale_npts, time_npts))
        #         Q = np.zeros((N_avg, amp_scale_npts, time_npts))
        #         waveform_name = 'wf_' + results_dict['drive_pulse'] + '_' + results_dict['drive_element']
        #         waveform_name_ramp_up = 'wf_' + results_dict['drive_pulse'] + '_ramp_up_' + results_dict['drive_element']
        #         waveform_name_ramp_down = 'wf_' + results_dict['drive_pulse'] + '_ramp_down_' + results_dict['drive_element']
        #         for i,amp_scale in tqdm(enumerate(amp_scale_list)):
                    
        #             I[:,i,:], Q[:,i,:] = self.run_prog(prog_id, shape = (N_avg, time_npts), 
        #                                      overrides_dict={'waveforms': {
        #                                                     (waveform_name): (self.config['waveforms'][(waveform_name)]['sample']*amp_scale).tolist(),
        #                                                     (waveform_name_ramp_up+'_I'): (self.config['waveforms'][(waveform_name_ramp_up+'_I')]['samples']*amp_scale).tolist(),
        #                                                    (waveform_name_ramp_up + '_Q'): (self.config['waveforms'][ (waveform_name_ramp_up + '_Q')]['samples']*amp_scale).tolist(),
        #                                                     (waveform_name_ramp_down+'_I'): (self.config['waveforms'][(waveform_name_ramp_down+'_I')]['samples']*amp_scale).tolist(),
        #                                                    (waveform_name_ramp_down + '_Q'): (self.config['waveforms'][ (waveform_name_ramp_down + '_Q')]['samples']*amp_scale).tolist()}})
                    
        processed_data = []
        stat_error = []
        for i in range(amp_scale_npts):
            data, err = self.process_data([I[:,i,:],Q[:,i,:]], is_calc_stat_error=True)
            processed_data.append(data)
            stat_error.append(err)
        processed_data = np.array(processed_data)
        stat_error = np.array(stat_error)
        freqs = []
        decays = []
        phases = []
        freqs_error = []
        decays_error = []
        phases_error = []
        for amp_data in zip(processed_data, stat_error):
            fit_res, fit_error,_ = self.fit_and_plot('ExpCos', amp_data, results_dict['t'], is_calc_stat_error=True, title_str = f'Ramsey with drive ({qubit})', txt = f'detuning = {detuning*1e3} [MHz]', is_return_fit = True, plot = is_plot_all)
            if is_plot_fft: plot_fft(results_dict['t'], amp_data[0])
            try:
                freqs.append(fit_res[1]*1e3)
                freqs_error.append(fit_error[1]*1e3)
                phases.append(fit_res[3])
                phases_error.append(fit_error[3])
                decays.append(1e-3/fit_res[2])
                decays_error.append(1e-3*fit_error[2]/fit_res[2]**2)
            except:
                freqs.append(np.nan)
                freqs_error.append(np.nan)
                phases.append(np.nan)
                phases_error.append(np.nan)
                decays.append(np.nan)
                decays_error.append(np.nan)
        freqs = np.array(freqs)
        freqs_error = np.array(freqs_error)[np.logical_and(freqs!=np.nan,freqs!=np.inf, freqs!=None)]
        amp_list_fitted = amp_list[np.logical_and(freqs!=np.nan, freqs!=np.inf, freqs!=None)]
        freqs = freqs[np.logical_and(freqs!=np.nan,freqs!=np.inf, freqs!=None)]
        decays=np.array(decays)
        decays_error = np.array(decays_error)[np.logical_and(decays!=np.nan, decays!=np.inf, decays!=None)]
        decays = decays[np.logical_and(decays!=np.nan, decays!=np.inf, decays!=None)]
        
        results_dict['data']=processed_data
        results_dict['error']=stat_error
        results_dict['freqs']=freqs
        results_dict['freqs_error']=freqs_error
        results_dict['decays']=decays
        results_dict['decays_error']=decays_error
        results_dict['phases']=phases
        results_dict['phases_error']=phases_error
        results_dict['amp_list_fitted']=amp_list_fitted
        
        self.pickle_save(results_dict, meas_name = 'driven ramsey')
    
        return self.fit_and_plot_driven_ramsey(is_echo = is_echo, kappa = kappa, chi = chi, chi_err = chi_err, drive_detuning = drive_detuning)
        
        
        
    def fit_and_plot_driven_ramsey(self, is_echo = False, kappa = None, chi = None, chi_err = 0, drive_detuning = None, T1 = None, is_echo_too = False, is_plot_all = False, is_plot_fft = False,
                                   ind_start_fit = None, ind_stop_fit = None):
        print('\n drive_detuning is assumed to be W_drive - W_cavity\n')
        if is_echo: results_dict = self.results['driven ramsey echo']
        else: results_dict = self.results['driven ramsey']
        qubit = results_dict['qubit']
        detuning = results_dict['detuning']
        if is_plot_all:
            for amp_data in zip(results_dict['data'], results_dict['error']):
                self.fit_and_plot('ExpCos', amp_data, results_dict['t'], is_calc_stat_error=True, title_str = f'Ramsey with drive ({qubit})', txt = f'detuning = {detuning*1e3} [MHz]', is_return_fit = False)
                if is_plot_fft: plot_fft(results_dict['t'], amp_data[0], is_filter_dc=True)
        
        def fit_func(x,A,B):
            return A*x**2+B
        
        amp_list_fitted = results_dict['amp_list_fitted'][ind_start_fit:ind_stop_fit]
        if not is_echo_too:
            decays = results_dict['decays'][ind_start_fit:ind_stop_fit]
            decays_error = results_dict['decays_error'][ind_start_fit:ind_stop_fit]
            
        else:
            decays = self.driven_ramsey_echo_results['decays'][ind_start_fit:ind_stop_fit]
            decays_error = self.driven_ramsey_echo_results['decays_error'][ind_start_fit:ind_stop_fit]
            
        freqs = results_dict['freqs'][ind_start_fit:ind_stop_fit]
        freqs_error = results_dict['freqs_error'][ind_start_fit:ind_stop_fit]
        kappa = results_dict['kappa'] if kappa is None else kappa
        chi = results_dict['chi'] if chi is None else chi
        drive_detuning = results_dict['drive_detuning'] if drive_detuning is None else drive_detuning
        delta = drive_detuning
        detuning = results_dict['detuning']
        
        if T1 is not None:
            decays = 1/(-1/2/T1+1/decays)
        if len(amp_list_fitted) == 1:
            stark_shift = -(freqs[0]-detuning*1e3) if detuning > 0 else -(freqs[0]+detuning*1e3)
            stark_shift, stark_shift_err = round_value_by_error(stark_shift, freqs_error[0])
            print(f'Stark shift = {stark_shift} +- {stark_shift_err} MHz')
            V = self.pulse_amp(results_dict['drive_element'], results_dict['drive_pulse'])
            if kappa is not None and delta is not None and chi is not None:
                a = np.sqrt(np.abs(stark_shift * 2 * np.pi * (4*delta**2+kappa**2) * (kappa**2+4*(delta-chi)**2) / (4 * V**2 * chi * (kappa**2+4*delta*(delta-chi)))))
                print(f'a/2pi = {np.round(a/2/np.pi,3)}')
            return stark_shift, stark_shift_err
        else:
            fit_results_freq, cov_freq = curve_fit(fit_func, amp_list_fitted, freqs*2*np.pi, sigma = freqs_error*2*np.pi)
            A_freq, A_freq_err = round_value_by_error(fit_results_freq[0], np.sqrt(cov_freq[0,0]))
            B_freq, B_freq_err = round_value_by_error(fit_results_freq[1], np.sqrt(cov_freq[1,1]))
            plt.figure()
            plt.errorbar(x = amp_list_fitted, y = freqs*2*np.pi, yerr = freqs_error*2*np.pi, fmt = '.r', label = 'Data', capsize = 4, markersize=4)
            plt.plot(amp_list_fitted, fit_func(amp_list_fitted,*fit_results_freq), '-b', label = 'Fit')
            ann = plt.annotate(r'$\delta F = AV^2+B$'+'\n'\
                               +f'A = {A_freq}'+r'$ \pm$'+f'{A_freq_err}'+'\n'\
                               +f'B = {B_freq:.4f}'+r'$ \pm$'+f'{B_freq_err:.4f}',
                               [amp_list_fitted[0], freqs.mean()*2*np.pi], fontsize = 12)
            ann.draggable()
            plt.xlabel('Amplitude [V]')
            plt.ylabel('Frequency detuning [Rad MHz]')
            plt.legend(fontsize=10)
            plt.tight_layout()
            
            if not is_echo_too:
                amp_list_fitted_decay = amp_list_fitted
            else:
                amp_list_fitted_decay = self.driven_ramsey_echo_results['amp_list_fitted']
                
            fit_results_decay, cov_decay = curve_fit(fit_func, amp_list_fitted_decay, 1/decays, sigma = decays_error/decays**2)
            A_decay, A_decay_err = round_value_by_error(fit_results_decay[0], np.sqrt(cov_decay[0,0]))
            B_decay, B_decay_err = round_value_by_error(fit_results_decay[1], np.sqrt(cov_decay[1,1]))
                
            plt.figure()
            plt.errorbar(x = amp_list_fitted_decay, y = (1/decays), yerr = decays_error/decays**2, fmt = '.r', label = 'Data', capsize = 4, markersize=4)
            plt.plot(amp_list_fitted_decay, fit_func(amp_list_fitted_decay,*fit_results_decay), '-b', label = 'Fit')
            ann2 = plt.annotate(r'$\Gamma_D = AV^2+B$'+'\n'\
                               +f'A = {A_decay}'+r'$ \pm$'+f'{A_decay_err}'+'\n'\
                               +f'B = {B_decay:.4f}'+r'$ \pm$'+f'{B_decay_err:.4f}',
                               [amp_list_fitted_decay[0], (1/decays).mean()], fontsize = 12)
            ann2.draggable()
            plt.xlabel('Amplitude [V]')
            plt.ylabel('Dephasing rate [MHz]')
            plt.legend(fontsize=10)
            plt.tight_layout()
            
        
            
            if kappa is not None:
                if chi is None: #find chi and a
                
                    if A_freq>0:
                        print('A_freq should be negative. Taking A_freq = -A_freq. This becomes positive because detuning is set to positive.')
                        A_freq = -A_freq
                
                    chi = (4*delta**2+kappa**2) * A_decay / (2*(2*delta*A_decay+kappa*A_freq))
                    a = np.sqrt(kappa*(A_decay**2+A_freq**2)/(2*A_decay))
                    
                    chi_err = np.sqrt((A_decay_err * (A_decay* delta * (4*delta**2+kappa**2)/(2*A_decay*delta+A_freq*kappa)**2 \
                                                    + (4 * delta**2+kappa**2) / (2*(2*A_decay*delta+A_freq*kappa))))**2 \
                                      + (A_freq_err * (A_decay * kappa * (4*delta**2+kappa**2) / (2*A_decay*delta+A_freq*kappa)**2))**2)
                        
                    a_err = np.sqrt((A_freq_err * A_freq * np.sqrt(kappa) / np.sqrt(2*A_decay*(A_freq**2+A_decay**2)))**2 \
                                    +(A_decay_err * np.sqrt(A_decay*kappa/(2*(A_freq**2+A_decay**2) \
                                                                           -np.sqrt((A_freq**2+A_decay**2)*kappa/8/A_decay**3))))**2)
                    
                    chi,chi_err = round_value_by_error(chi/2/np.pi, chi_err/2/np.pi)
                    a,a_err = round_value_by_error(a/2/np.pi, a_err/2/np.pi)
                    print(f"chi/2pi = {chi} +- {chi_err} MHz")
                    print(f"a/2pi = {a} +- {a_err} MHz/V")
                    
                    return chi, chi_err, a, a_err
                
                elif chi is not None: #find only a
                
                    if A_freq>0:
                        print('A_freq should be negative. Taking A_freq = -A_freq. This becomes positive because detuning is set to positive.')
                        A_freq = -A_freq
                
                    if chi > 0:
                        print('Chi is assumed to be negative. Change to -chi.')
                        chi = -chi
                        
                    a = np.sqrt(A_freq * (4*delta**2+kappa**2) * (4*delta**2+kappa**2-8*delta*chi+4*chi**2) / (4 * chi * (4*delta**2+kappa**2-4*delta*chi)))

                    
                    a_err = np.sqrt((-A_freq_err * a/2/A_freq)**2 + (chi_err * (np.sqrt(-A_freq) * (-8*chi*delta * (kappa**2 + 4*delta * (delta-2*chi))) \
                                                                                   - (4*delta**2+kappa**2)*(-4*delta**2-kappa**2+8*delta*chi+4*chi**2)) \
                                                / (4*chi*np.sqrt(-(kappa**2+4*delta*(delta-chi))*chi / ((4*delta**2+kappa**2)*(kappa*2+4*delta*(delta-2*chi)+4*chi**2))) \
                                                   *(kappa**2+4*delta*(delta-2*chi)+4*chi**2) * (-kappa**2+4*delta*(-delta+chi))))**2 )
                    
                    a,a_err = round_value_by_error(a/2/np.pi, a_err/2/np.pi)
                    print(f"a/2pi = {a} +- {a_err} MHz/V")
                    return a, a_err
            
            
#%% Pi amp calibration
    def load_pi_amp_calibration(self,
                                   amp_scale_start = 0, amp_scale_stop = 1, npts = 11,
                                   N_avg = 1000,
                                   pulse = 'pi_pulse',
                                   npulses = 1,
                                   qubit = None,
                                   ro_element = None,
                                   is_active_reset = None,
                                   **kwargs):
    
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if pulse is None: 'pi_pulse'
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        if is_active_reset: run_time = N_avg*npts*(self.pulse_len(ro_element,self.ro_pulse)+npulses*self.pulse_len(qubit,pulse))
        else: run_time = N_avg*npts*(self.pulse_len(ro_element,self.ro_pulse)+npulses*self.pulse_len(qubit,pulse)+self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))   
        
        amp_scale_list = np.linspace(amp_scale_start, amp_scale_stop, npts)
        amp_list = amp_scale_list * self.pulse_amp(qubit, pulse)
        
        
        self.results['pi_amp_scalibration'] = {'N_avg': N_avg,
                                  'amp_scale_start': amp_scale_start,
                                  'amp_scale_stop': amp_scale_stop,
                                  'npts': npts,
                                  'npulses': npulses,
                                  'pulse': pulse,
                                  'qubit': qubit,
                                  'ro_element': ro_element,
                                  'amp_list':amp_list}
    
        with program() as self.pi_amp_calib_prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            amp_scale = declare(fixed)
            
            with for_(n, 0, n < N_avg, n + 1):
                with for_each_(amp_scale, amp_scale_list):
                    for i in range(npulses):
                        play(pulse * amp(amp_scale), qubit)
                    align(qubit, ro_element)
                    
                    self.perform_full_measurement(I, Q, ro_element = ro_element, is_active_reset = is_active_reset)
                    
        self.last_prog = self.pi_amp_calib_prog 
        
    def run_pi_amp_calibration(self,
                            is_save_data = False,
                            **kwargs): 
        results_dict = self.results['pi_amp_scalibration']
        if is_save_data is None: is_save_data = self.is_save_data
         
        
        self.qm_server.clear_all_job_results()            
        job = self.qm.execute(self.pi_amp_calib_prog, duration_limit=0, data_limit=0, force_execution = True)
        job.result_handles.wait_for_all_values()
        
        I_res = job.result_handles.get('I').fetch_all()['value'].reshape((results_dict['N_avg'], -1))
        Q_res = job.result_handles.get('Q').fetch_all()['value'].reshape((results_dict['N_avg'], -1))
        results_dict['I'] = I_res
        results_dict['Q'] = Q_res
        
        if is_save_data: self.pickle_save(results_dict, 'pi_amp_scalibration')
        
        self.plot_pi_amp_calibration(**kwargs)
        
    def plot_pi_amp_calibration(self, is_calc_stat_error = None, which_data = None, is_fit_cos = False, **kwargs):
        if which_data is None: which_data = self.which_data
        results_dict = self.results['pi_amp_scalibration']
        
        if is_calc_stat_error is None:
            is_calc_stat_error = self.is_calc_stat_error
        
        ro_element = results_dict['ro_element']
        qubit = results_dict['qubit']
        amp_list = results_dict['amp_list']
        npulses = results_dict['npulses']
        
        prog_name = 'Pi amp Calibration'
        fig = plt.figure(next_fig_num_by_name(prog_name))
        
        I = results_dict['I']
        Q = results_dict['Q']
        
        data, stat_error = self.process_data([I, Q], is_calc_stat_error = is_calc_stat_error, **kwargs)
        
        data, units_prefix, scale_factor = self.autoscale_data(data, **kwargs)
        
            
        plt.errorbar(amp_list, data, stat_error * scale_factor, fmt = 'o', capsize = 6)
        if is_fit_cos:
            sfit = sFit('Cos', data, amp_list, stat_error * scale_factor)
            if sfit.is_succeed:
                plt.plot(amp_list, sfit.func(amp_list, *sfit.fit_results), '-b')
                
        plt.xlabel('Pulse amp. [V]')
        if which_data == 'Phase': plt.ylabel('{1} [{0}Rad]'.format(units_prefix, which_data))
        else: plt.ylabel('{1} [{0}V]'.format(units_prefix, which_data))
        plt.tight_layout()
        
        if is_fit_cos:
            print('\nPi pulse amp = {}+-{}\n'.format(*round_value_by_error(npulses/2/sfit.fit_results[1], npulses/2*np.sqrt(sfit.cov_results[1,1])/sfit.fit_results[1]**2)))
        
        return fig
                
#%%  pi2 calibration specific

    def load_pi2_calibration(self,
                                   npts = 12, # How many points I measure
                                   N_avg = 1000, # How many times each sequence is executed 
                                   pi2_pulse: str= None,
                                   start = 0,
                                   qubit = None,
                                   ro_element = None,
                                   drive_element = None,
                                   drive_detuning = 0,
                                   drive_pulse: str = None,
                                   is_ramp = True,
                                   qb_detuning = 0,
                                   steady_time = 0,
                                   ring_down_time = 0,
                                   is_active_reset = None,
                                   is_sb_cool = None,
                                   **kwargs):

        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if drive_pulse is None: drive_pulse = 'mixer_cal_pulse'
        if pi2_pulse is None: pi2_pulse = self.pi2_pulse
        if drive_element is not None:
            pi2_pulse_clks = int(self.pulse_len(qubit, self.pi2_pulse)//4)
            drive_clks = int(self.pulse_len(drive_element, drive_pulse)//4+self.pulse_ramp_len(drive_element, drive_pulse)//4)
            if drive_detuning != 0:
                drive_IF0 = self.element_IF(drive_element)
                drive_IF_detuned = drive_IF0+drive_detuning
                
        if qb_detuning != 0:
            IF0 = self.element_IF(qubit)
            IF_detuned = IF0+qb_detuning
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        def play_drive(drive_element, drive_pulse, duration, is_truncate = False):
            play(drive_pulse, drive_element, duration = duration)
                
        self.results['pi2_calibration'] = {'N_avg': N_avg,
                                  'start': start,
                                  'npts': npts,
                                  'pi2_pulse': pi2_pulse,
                                  'qubit': qubit,
                                  'ro_element': ro_element}
        

        if is_active_reset: run_time = N_avg*npts*(self.pulse_len(ro_element,self.ro_pulse)+npts*self.pulse_len(qubit,self.pi2_pulse))
        elif is_sb_cool: run_time = N_avg*npts*(self.pulse_len(ro_element,self.ro_pulse)+npts*self.pulse_len(qubit,self.pi2_pulse)+self.pulse_len(ro_element, 'constant_sideband_cooling_pulse'))
        else: run_time = N_avg*npts*(self.pulse_len(ro_element,self.ro_pulse)+npts*self.pulse_len(qubit,self.pi2_pulse)+self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))   

        with program() as self.pi2_calib_prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            i = declare(int)
            
            with for_(n, 0, n < N_avg, n + 1):
                with for_(i, start, i < npts + start, i+1):
                    if drive_element is not None:
                        if drive_detuning != 0: update_frequency(drive_element, drive_IF_detuned)
                        if is_ramp: play(drive_pulse + '_ramp_up', drive_element)
                        play(drive_pulse, drive_element)
                    if drive_element is not None:
                        if is_ramp: play(drive_pulse + '_ramp_down', drive_element)
                        wait(drive_clks-i*pi2_pulse_clks, qubit)
                        if drive_detuning != 0 and not is_sb_cool and drive_element == ro_element: update_frequency(drive_element, drive_IF0)
                    if self.e_to_f:  self.map_e_to_g(self.main_qubit_g_to_e)
                    if qb_detuning != 0: update_frequency(qubit, IF_detuned)
                    with switch_(i):
                        for j in range(start, npts+start):
                            with case_(j):
                                for k in range(j):
                                    play(pi2_pulse, qubit)
                            
                    if qb_detuning != 0 and not is_sb_cool: update_frequency(qubit, IF0)
                    
                    if drive_element is not None:
                        if drive_element != ro_element:
                            align(drive_element, ro_element)
                        if ring_down_time//4>0:
                            wait(ring_down_time//4, ro_element)
                    else:
                        align(qubit, ro_element)
                    
                    if self.e_to_f and self.e_to_f_map_back:  self.map_e_to_g(self.main_qubit_g_to_e)
                    
                    self.perform_full_measurement(I, Q, ro_element = ro_element, is_active_reset = is_active_reset, is_sb_cool = is_sb_cool)
                    
        self.last_prog = self.pi2_calib_prog 
        
    def run_pi2_calibration(self,
                            is_save_data = False, is_plot = True,
                            **kwargs): 
        results_dict = self.results['pi2_calibration']
        if is_save_data is None: is_save_data = self.is_save_data
         
        I_res, Q_res = self.run_prog(self.pi2_calib_prog, shape = (results_dict['N_avg'], -1), **kwargs)
        results_dict['I'] = I_res
        results_dict['Q'] = Q_res
        
        if is_save_data: self.pickle_save(results_dict, 'pi2_calibration')
        
        if is_plot: self.plot_pi2_calibration()
        
    def plot_pi2_calibration(self, is_calc_stat_error = None, **kwargs):
        results_dict = self.results['pi2_calibration']
        
        if is_calc_stat_error is None:
            is_calc_stat_error = self.is_calc_stat_error
        
        ro_element = results_dict['ro_element']
        qubit = results_dict['qubit']
        start = results_dict['start']
        npts = results_dict['npts']
        pi2_pulse = results_dict['pi2_pulse']
        
        prog_name = 'Pi2 Calibration'
        fig = plt.figure(next_fig_num_by_name(prog_name))
        
        I = results_dict['I']
        Q = results_dict['Q']
        
        data, stat_error = self.process_data([I, Q], is_calc_stat_error = is_calc_stat_error, **kwargs)
        
        data, units_prefix, scale_factor = self.autoscale_data(data, **kwargs)
        
        num_of_pls = np.linspace(start,start + npts -1, npts)
        plt.plot(num_of_pls, data, 'b-')
        plt.errorbar(num_of_pls, data, yerr = stat_error * scale_factor, fmt = 'ro', capsize = 5, markersize = 6, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))
            
        plt.plot(num_of_pls[1::2], data[1::2],'r--')
        plt.xlabel(r"number of $\pi/2$ pulses")
        if self.which_data == 'Phase': plt.ylabel('{1} [{0}Rad]'.format(units_prefix, self.which_data))
        else: plt.ylabel('{1} [{0}V]'.format(units_prefix, self.which_data))
        
        ro_amp =abs(self.configObject.SuperMembers[ro_element].PulseParamsDict[self.ro_pulse].amp)
        pi2_amp =abs(self.configObject.SuperMembers[qubit].PulseParamsDict[pi2_pulse].amp)
        
        plt.title(f"Pi/2 Calibration ("+qubit+")", fontsize=30)
        if self.pulse_type(qubit, pi2_pulse) == 'gaussian':
            annot = f'Navg={results_dict["N_avg"]}, ' \
            +  r'$\pi/2$ amp={}, '.format(pi2_amp) \
            + r'$\sigma$ = {} ns, '.format(self.pulse_sig(qubit, pi2_pulse))\
            + f'Time multiplier = {self.pulse_time_multiplier(qubit, pi2_pulse)}'
        else:
              annot = f'Navg={results_dict["N_avg"]}, '\
                  + r'$\pi/2$ amp={} '.format(pi2_amp)
        ann = plt.annotate(annot,
                           xy = (float(num_of_pls[0]), float(data.min())),
                           fontsize = 10,
                           )
        ann.draggable()
        
        plt.tight_layout()
        
        
        return fig
    
#%%  pi calibration specific

    def load_pi_calibration(self,
                            npts = 4, # How many points I measure
                            N_avg = 1000, # How many times each sequence is executed 
                            pi2_pulse: str= None,
                            pi_pulse: str= None,
                            is_calc_stat_error= False,
                            start = 0,
                            qubit = None,
                            ro_element = None,
                            drive_element = None,
                            drive_pulse: str = None,
                            steady_time = 0,
                            ring_down_time = 0,
                            is_active_reset = None,
                            **kwargs):
        if pi2_pulse is None: pi2_pulse=self.pi2_pulse
        if pi_pulse is None: pi_pulse=self.pi_pulse
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if drive_pulse is None: drive_pulse = 'mixer_cal_pulse'
        
        if drive_element is not None:
            is_truncate = not self.configObject.SuperMembers[drive_element].PulseParamsDict[drive_pulse].pulseType == 'constant'
            pi2_pulse_time = self.pulse_len(qubit, self.pi2_pulse)
            if is_truncate and self.pulse_len(drive_element, drive_pulse) < self.pulse_len(qubit, self.pi2_pulse) * (npts+start) + steady_time:
                raise ValueError(f"You are trying to truncate the drive pulse, but it is too short. Pulse length >= pi2_pulse_time * (npts+start) + steady_time is required. Got {self.pulse_len(drive_element, drive_pulse)}<{int(pi2_pulse_time * (npts+start))}+{int(steady_time)} ")
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        def play_drive(drive_element, drive_pulse, duration, is_truncate = False):
            if is_truncate:
                play(drive_pulse, drive_element, truncate = duration)
            else:
                play(drive_pulse, drive_element, duration = duration)
                
        
        self.results['pi_calibration'] = {'N_avg': N_avg,
                                       'npts': npts,
                                       'start': start,
                                       'qubit': qubit,
                                       'ro_element': ro_element,
                                       'pi_pulse': pi_pulse,
                                       }
        
        if is_active_reset: run_time = N_avg*((npts)*2)*(self.pulse_len(ro_element, self.ro_pulse))
        else: run_time = N_avg*((npts)*2)*(self.pulse_len(ro_element, self.ro_pulse) + self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))   


        with program() as  self.pi_calib_prog:
            
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            i = declare(int)
            
            with for_(n, 0, n < N_avg, n + 1):
                with for_(i, start, i < start + 2*npts, i+1):
                    with switch_(i):
                        for j in range(start, 2*npts+start):
                            with case_(j):
                                if drive_element is not None:
                                    play_drive(drive_element, drive_pulse, duration = int(steady_time//4) + int(pi2_pulse_time//4) * i, is_truncate = is_truncate)
                                    if steady_time//4 >0:
                                        wait(int(steady_time//4), qubit)
                                        
                                if self.e_to_f:  self.map_e_to_g(self.main_qubit_g_to_e)
                                
                                if np.mod(j, 2) == 1: play(pi2_pulse, qubit)
                                for k in range(j//2):
                                    play(pi_pulse, qubit)
                                        
                                if drive_element is not None:
                                    if drive_element != ro_element:
                                        align(drive_element, ro_element)
                                    if ring_down_time//4>0:
                                        wait(ring_down_time//4, ro_element)
                                else:
                                    align(qubit, ro_element)
                                
                                if self.e_to_f and self.e_to_f_map_back:  self.map_e_to_g(self.main_qubit_g_to_e)
                                    
                    self.perform_full_measurement(I,Q, ro_element = ro_element, is_active_reset = is_active_reset)
                                
   
            self.last_prog = self.pi_calib_prog

    def run_pi_calibration(self, is_calc_stat_error= None, is_save_data = False, is_plot = True, **kwargs): 
        results_dict = self.results['pi_calibration']
        
        qubit = results_dict['qubit']
        ro_element = results_dict['ro_element']
        
        results_dict['I'], results_dict['Q'] = self.run_prog(self.pi_calib_prog, shape = (results_dict['N_avg'], -1), **kwargs)
        
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'Pi_calibration')
        
        if is_plot: self.plot_pi_calibration(is_calc_stat_error = is_calc_stat_error)
        
    def plot_pi_calibration(self, is_calc_stat_error = None, **kwargs):
        results_dict = self.results['pi_calibration']
        
        if is_calc_stat_error is None:
            is_calc_stat_error = self.is_calc_stat_error
        
        start = results_dict['start']
        N_avg = results_dict['N_avg']
        npts = results_dict['npts']
        qubit = results_dict['qubit']
        ro_element = results_dict['ro_element']
        pi_pulse =  results_dict['pi_pulse']
        
        prog_name = 'Pi Calibration'
        fig = plt.figure(next_fig_num_by_name(prog_name))
        
        data, stat_error = self.process_data([results_dict['I'], results_dict['Q']], is_calc_stat_error=is_calc_stat_error, **kwargs)
        
        num_of_pls = np.linspace(start, start + npts - 0.5, npts*2)
        
        data, units_prefix, scale_factor = self.autoscale_data(data, **kwargs)
        
        plt.plot(num_of_pls, data, 'b-')
        
        plt.errorbar(num_of_pls, data, yerr = stat_error * scale_factor, fmt = 'ro', capsize = 5, markersize = 6, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))
            
        plt.plot(num_of_pls[1::2], data[1::2],'r--')

        plt.xlabel("number of pulses")
        if self.which_data == 'Phase': plt.ylabel('{1} [{0}Rad]'.format(units_prefix, self.which_data))
        else: plt.ylabel('{1} [{0}V]'.format(units_prefix, self.which_data))

        ro_amp = abs(self.configObject.SuperMembers[ro_element].PulseParamsDict[self.ro_pulse].amp)
        pi_amp = abs(self.configObject.SuperMembers[qubit].PulseParamsDict[pi_pulse].amp)
         
        plt.title(f"Pi Calibration ("+qubit+")", fontsize=30)
        
        if self.pulse_type(qubit, pi_pulse) == 'gaussian':
            annot = f'Navg={results_dict["N_avg"]}, ' \
            +  r'$\pi$ amp={}, '.format(pi_amp) \
            + r'$\sigma$ = {} ns, '.format(self.pulse_sig(qubit, pi_pulse))\
            + f'Time multiplier = {self.pulse_time_multiplier(qubit, pi_pulse)}'
        else:
              annot = f'Navg={results_dict["N_avg"]}, '\
                  + r'$\pi$ amp={} '.format(pi_amp)
                  
        ann = plt.annotate(annot,
                           # xy = (float((num_of_pls[-1]-num_of_pls[0])/4+num_of_pls[0]), float(data.min())),
                           xy = (float(num_of_pls[0]), float(data.min())),
                           fontsize = 10,
                           )
        ann.draggable()
        plt.tight_layout()
        return fig
   
#%%  rabi specific

    def load_rabi(self,
                npts = 51, # How many points I measure
                max_seq_time = 80000, # Time of longest pulse in units of nano seconds. Must be a multiple of 4
                min_seq_time = None,
                time_step = None,
                N_avg = 1000,
                qubit = None,
                ro_element = None,
                rabi_pulse = 'rabi_pulse',
                is_sb_cool = False,
                is_active_reset = None,
                is_ramp = True,
                is_meas_x = False,
                init_state = 'g',
                is_tomography = False,
                wait_before_readout = 0,
                rabi_detuning = 0,
                **kwargs):
        
        if min_seq_time is not None:
            if min_seq_time > max_seq_time: raise ValueError("Oops... Maximum sequence time < Minimum sequence time.")
        if qubit is None: qubit = [self.main_qubit]
        if type(qubit) is not list: qubit = [qubit]
        if ro_element is None: ro_element = self.main_readout
        if is_active_reset is None: is_active_reset = self.is_active_reset
        if is_tomography: tomo = OneQubitTomo(qubit = qubit)
        
        def play_drive(drive_element, drive_pulse, duration, is_truncate = False):
            if is_truncate:
                play(drive_pulse, drive_element, truncate = duration)
            else:
                play(drive_pulse, drive_element, duration = duration)
        
        if time_step is not None: npts = int((max_seq_time-min_seq_time) // time_step + 1)
        max_seq_clks = int(max_seq_time // 4)
        if min_seq_time is not None: 
            first_clk = min_seq_time//4
            step_size_clks = (max_seq_clks - first_clk) // (npts-1)
            if step_size_clks == 0:
                raise ValueError(f"Time steps are smaller than 4 ns. Got <(max_seq_time-min_seq_time)/4/(npts-1)> =  {np.around((max_seq_clks - first_clk) / (npts-1),4)}")
        else: 
            step_size_clks =  max_seq_time // 4 // (npts-1)
            first_clk = step_size_clks
            if step_size_clks == 0:
                raise ValueError(f"Time steps are smaller than 4 ns. Got <max_seq_time/4 /(npts-1)> =  {np.around(max_seq_clks /(npts-1),4)}")
        self.results['rabi'] = {'t': np.arange(first_clk*4, max_seq_time+1, step_size_clks*4),
                             'npts': npts,
                             'N_avg':N_avg,
                             'qubit': qubit,
                             'ro_element': ro_element,
                             'is_tomography': is_tomography}
        if is_tomography:
            self.results['rabi']['tomo_object'] = tomo
        
        
        if is_sb_cool:
            run_time = N_avg*npts*(max_seq_time/2 + self.pulse_len(ro_element,self.ro_pulse) + self.pulse_len(ro_element, 'sideband_cooling_pulse'))
        elif is_active_reset:
            run_time = N_avg*npts*(max_seq_time/2 + self.pulse_len(ro_element,self.ro_pulse))
        else:
            run_time = N_avg*npts*(max_seq_time/2 + self.pulse_len(ro_element,self.ro_pulse)+self.wait_between_seq)
        if is_tomography: 
            run_time = run_time * len(tomo.pulse_seq_list)
            if is_active_reset: run_time += N_avg * self.pulse_len(ro_element,self.ro_pulse)
            else: run_time += N_avg * (self.pulse_len(ro_element,self.ro_pulse)+self.wait_between_seq)
        
        print('Run time is {}s'.format(round(run_time * 1e-9)))        
        
        
        
        def rabi_sequence(pulse_duration,I,Q, is_tomography, x = None):
            with for_(pulse_duration,
                      first_clk,
                      pulse_duration <= max_seq_clks,
                      pulse_duration + step_size_clks):
                reset_frame(qubit)
                if init_state in ['plus', 'minus']:
                    play('pi2_pulse', 'qb1')
                    frame_rotation_2pi(0.25, 'qb1')
                if self.e_to_f:  self.map_e_to_g(self.main_qubit_g_to_e)
                if rabi_detuning!=0: update_frequency(qubit[0], self.element_IF(qubit[0]) + rabi_detuning)
                for qb in qubit:
                    if is_ramp :
                        play(rabi_pulse + '_ramp_up', qb)
                    play(rabi_pulse, qb, duration = pulse_duration)
                    if is_ramp :
                        play(rabi_pulse + '_ramp_down', qb)
                    if wait_before_readout > 0: wait(wait_before_readout//4, qb)
                if rabi_detuning!=0: update_frequency(qubit[0], self.element_IF(qubit[0]), keep_phase = True)
                        
                if self.e_to_f and self.e_to_f_map_back:  self.map_e_to_g(qubit_g_to_e)
                if is_meas_x and not is_tomography:
                    for qb in qubit:
                        frame_rotation_2pi(0.25, qb) # sigma x measurement
                        play(self.pi2_pulse, qb)
                if is_tomography:
                    with switch_(x):
                        for i in range(len(tomo.pulse_seq_list)):
                            with case_(i):
                                tomo.play_tomo_pulse(tomo.pulse_seq_list[i], self.pulse_len(qubit[0], self.pi2_pulse))
                for qb in qubit:
                    align(qubit, ro_element)
                if is_tomography:
                    I_out_name = 'I_tomo'
                    Q_out_name = 'Q_tomo'
                else:
                    I_out_name = 'I'
                    Q_out_name = 'Q'
                self.perform_full_measurement(I,Q, I_output_name = I_out_name, Q_output_name = Q_out_name, ro_element = ro_element, is_wait = not is_sb_cool, is_active_reset = is_active_reset, **kwargs)

        with program() as self.rabi_prog:
            
            if is_tomography: tomo.declarations()
            else:
                I = declare(fixed)
                Q = declare(fixed)
            pulse_duration = declare(int)
            n = declare(int)
            with for_(n, 0, n < N_avg, n + 1):
                if is_tomography: 
                    tomo.measure_beta_coeffs(ro_element, qubit, meas_func = self.perform_full_measurement)
                    with for_(tomo.x, 0, tomo.x<len(tomo.pulse_seq_list), tomo.x+1):
                        rabi_sequence(pulse_duration, tomo.I, tomo.Q, is_tomography, tomo.x)
                else:
                    rabi_sequence(pulse_duration, I, Q, is_tomography = False)
                        
            self.last_prog = self.rabi_prog
            
        
    def run_rabi(self, is_continue = False, is_save_data = None, **kwargs): 
    
        if not hasattr(self,'rabi_prog'): raise ValueError("Idiot! You did not write rabi program")
        results_dict = self.results['rabi']
        prog_name = 'Rabi'
        if results_dict['is_tomography']: 
            return self.run_one_qubit_tomography(self.rabi_prog, results_dict['npts'], results_dict = results_dict, tomo_object=results_dict['tomo_object'], **kwargs)
        if is_continue and 'I' in results_dict.keys():
                results_dict['I'], results_dict['Q'] = self.continue_run(self.rabi_prog, results_dict['N_avg'], results_dict)
                fig_num = last_fig_num_by_name(prog_name)(prog_name)
        else:
            results_dict['I'], results_dict['Q'] = self.run_prog(self.rabi_prog, results_dict['N_avg'], **kwargs)
            fig_num = next_fig_num_by_name(prog_name)
        
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'rabi')
        if results_dict['is_tomography']: self.plot_one_qubit_tomography()
        return self.plot_rabi(fig_num = fig_num, is_continue = is_continue, prog_name = prog_name, **kwargs)

    def plot_rabi(self, is_hist2d = False, is_fit_decay = True, **kwargs):
        
        results_dict = self.results['rabi']
        
        ylabel = str(self.which_data)
        if self.which_data in ['Phase', 'phase']:
            ylabel += ' [Rad]'
        else:
            ylabel += ' [V]'
        if is_hist2d: plot_hist2d(self.determine_data(results_dict['I'], results_dict['Q']), results_dict['t'], 
                                  xlabel = 'Time [ns]', ylabel = ylabel, cmap = 'Reds', **kwargs)
        
        if is_fit_decay:
            fit_func_name = 'ExpCos'
        else:
            fit_func_name = 'Cos'
            
        return self.fit_and_plot(fit_func_name, self.process_data(results_dict, **kwargs), results_dict['t'], title_str = f'Rabi ({results_dict["qubit"]})', **kwargs)
    
    def rabi_complete(self,wait_overhead = 5,  # this is a time we need to subtract from the wait time to get the correct wait time
                npts = 50, # How many points I measure
                max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
                N_avg = 1000 # How many times each sequence is executed 
                ):
        
        self.load_rabi(npts=npts,max_seq_time= max_seq_time, N_avg=N_avg)
        
        return self.run_rabi()
    
    def load_driven_rabi(self,
                npts = 51, # How many points I measure
                max_seq_time = 80000, # Time of longest pulse in units of nano seconds. Must be a multiple of 4
                min_seq_time = None,
                time_step = None,
                N_avg = 1000,
                amp_scale_start = 1,
                amp_scale_stop = 1,
                amp_scale_npts = 1,
                qubit = None,
                ro_element = None,
                drive_element = None, drive_detuning = 0,
                drive_pulse = None,
                drive_element2 = None, drive_detuning2 = 0,
                drive_pulse2 = None,
                rabi_pulse = 'rabi_pulse',
                is_sb_cool = False,
                is_active_reset = None,
                is_ramp = True,
                is_ramp_drive = False,
                is_meas_x = False,
                init_state = 'g',
                ring_down_time = 0,
                extra_drive_time = 100,
                stark_shift_corr = 0,
                is_tomography = False,
                **kwargs):
        
        # print(stark_shift_corr)
        print(f'\nStark shift correction is {np.round(stark_shift_corr*1e-6,3)} MHz \n')
        if type(stark_shift_corr) is not list:
            stark_shift_corr=[stark_shift_corr]
        if qubit is None: qubit = self.main_qubit
        if type(qubit) is not list:
            qubit = [qubit]
        if min_seq_time is not None:
            if min_seq_time > max_seq_time: raise ValueError("Oops... Maximum sequence time < Minimum sequence time.")
        if ro_element is None: ro_element = self.main_readout
        if drive_element is None: drive_element = self.main_readout
        if drive_pulse is None: drive_pulse = 'mixer_cal_pulse'
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        if is_tomography: tomo = OneQubitTomo(qubit[0])
        is_truncate = not self.configObject.SuperMembers[qubit[0]].PulseParamsDict[rabi_pulse].pulseType == 'constant'
        if is_truncate and self.pulse_len(qubit[0], rabi_pulse) < max_seq_time:
            raise ValueError(f"You are trying to truncate the drive pulse, but it is too short. Pulse length >= max_seq_time is required. Got {self.pulse_len(qubit, rabi_pulse)}<{int(max_seq_time)}")
        ramp_up_length = self.configObject.SuperMembers[qubit[0]].PulseParamsDict[rabi_pulse].Additions.ramp['up']['length']
        ramp_down_length = self.configObject.SuperMembers[qubit[0]].PulseParamsDict[rabi_pulse].Additions.ramp['down']['length']
        if not is_ramp and self.pulse_len(drive_element, drive_pulse) < max_seq_time:
            raise ValueError(f"Drive pulse too short. Must be longer than max_seq_time. Got {self.pulse_len(drive_element, drive_pulse)}<{max_seq_time}")
        elif is_ramp and not is_ramp_drive and int((self.pulse_len(drive_element, drive_pulse)-ramp_up_length-ramp_down_length-extra_drive_time)) < max_seq_time:
            raise ValueError(f"Drive pulse too short. Must be longer than max_seq_time - ramp time. Got {int((self.pulse_len(drive_element, drive_pulse)-ramp_up_length-ramp_down_length-extra_drive_time))}<{max_seq_time}")
        elif is_ramp and is_ramp_drive and int((self.pulse_len(drive_element, drive_pulse)+2*self.pulse_ramp_len(drive_element, drive_pulse)-ramp_up_length-ramp_down_length-extra_drive_time)) < max_seq_time:
            raise ValueError(f"Drive pulse too short. Must be longer than max_seq_time - ramp time. Got {int((self.pulse_len(drive_element, drive_pulse)-ramp_up_length-ramp_down_length-extra_drive_time))}<{max_seq_time}")
        
        if time_step is not None: npts = int((max_seq_time-min_seq_time) // time_step + 1)
        max_seq_clks = int(max_seq_time // 4)
        if min_seq_time is not None: 
            first_clk = min_seq_time//4
            step_size_clks = (max_seq_clks - first_clk) // (npts-1)
            if step_size_clks == 0:
                raise ValueError(f"Time steps are smaller than 4 ns. Got <(max_seq_time-min_seq_time)/4/(npts-1)> =  {np.around((max_seq_clks - first_clk) / (npts-1),4)}")
        else: 
            step_size_clks =  max_seq_time // 4 // (npts-1)
            first_clk = step_size_clks
            if step_size_clks == 0:
                raise ValueError(f"Time steps are smaller than 4 ns. Got <max_seq_time/4 /(npts-1)> =  {np.around(max_seq_clks /(npts-1),4)}")
        amp_scale_list = np.linspace(amp_scale_start, amp_scale_stop, amp_scale_npts)
        
        self.results['driven rabi'] = {'t': np.arange(first_clk*4, max_seq_time+1, step_size_clks*4),
                                    'npts': npts,
                                    'N_avg': N_avg,
                                    'amp_scale': amp_scale_list,
                                     'qubit': qubit,
                                     'ro_element': ro_element,
                                     'drive_element': drive_element,
                                     'drive_element2': drive_element2,
                                     'drive_pulse': drive_pulse,
                                     'drive_pulse2': drive_pulse2,
                                     'is_tomography': is_tomography}
        # if is_tomography:
        #     self.results['driven rabi']['tomo_object'] = tomo
        
        if is_sb_cool:
            run_time = N_avg*npts*amp_scale_npts*(self.pulse_len(drive_element, drive_pulse) + self.pulse_len(ro_element,self.ro_pulse) + self.pulse_len(ro_element, 'constant_sideband_cooling_pulse'))
        elif is_active_reset:
            run_time = N_avg*npts*amp_scale_npts*(self.pulse_len(ro_element,self.ro_pulse) + self.pulse_len(drive_element, drive_pulse))
        else:
            run_time = N_avg*npts*amp_scale_npts*(self.pulse_len(ro_element,self.ro_pulse) + self.pulse_len(drive_element, drive_pulse) +self.wait_between_seq)
        if is_tomography: 
            run_time = run_time * len(tomo.pulse_seq_list)
            if is_active_reset: run_time += N_avg * self.pulse_len(ro_element,self.ro_pulse)
            else: run_time += N_avg * (self.pulse_len(ro_element,self.ro_pulse)+self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))        
        
        base_wait_time = int((self.pulse_len(drive_element, drive_pulse)-extra_drive_time)//4)
        if is_ramp: base_wait_time -= (ramp_down_length+ramp_up_length)//4
        if is_ramp_drive: base_wait_time += self.pulse_ramp_len(drive_element, drive_pulse)//4
        if is_meas_x or is_tomography: base_wait_time -= self.pulse_len(qubit[0], 'pi2_pulse')//4
        if init_state != 'g': base_wait_time -= self.pulse_len(qubit[0], 'pi2_pulse')//4
        
        if init_state == 'plus': init_phase = 0.25
        if init_state == 'minus': init_phase = -0.25


        def rabi_sequence(pulse_duration, I, Q, amp_scale, is_tomography, x = None):
            with for_(pulse_duration,
                      first_clk,
                      pulse_duration <= max_seq_clks,
                      pulse_duration + step_size_clks):
                for qb,sshift in zip(qubit,stark_shift_corr):
                    # if stark_shift_corr != 0: update_frequency(qb, self.element_IF(qb)+sshift, keep_phase = True)
                    if stark_shift_corr != 0: update_frequency(qb, new_IF, keep_phase = True)
                    
                
                if drive_detuning != 0:
                    update_frequency(drive_element, self.element_IF(drive_element)+drive_detuning, keep_phase = True)
                if drive_detuning2 != 0 and drive_element2 is not None:
                    update_frequency(drive_element2, self.element_IF(drive_element2)+drive_detuning2, keep_phase = True)
                reset_frame(drive_element)
                if drive_element2 is not None: 
                    reset_frame(drive_element2)
                for qb in qubit:
                    reset_frame(qb)
                for qb in qubit:
                    align(drive_element, qb)
                    
                if len(amp_scale_list)>1: 
                    if is_ramp_drive:  
                        play((drive_pulse + '_ramp_up') * amp(amp_scale), drive_element)
                        if drive_element2 is not None: play((drive_pulse2 + '_ramp_up') * amp(amp_scale), drive_element2)
                    play(drive_pulse * amp(amp_scale), drive_element)
                    if drive_element2 is not None: play(drive_pulse * amp(amp_scale), drive_element2)
                    if is_ramp_drive:  
                        play((drive_pulse + '_ramp_down')* amp(amp_scale), drive_element)
                        if drive_element2 is not None: play((drive_pulse2 + '_ramp_down')* amp(amp_scale), drive_element2)
                else: 
                    if is_ramp_drive:  
                        play(drive_pulse + '_ramp_up', drive_element)
                        if drive_element2 is not None: play(drive_pulse2 + '_ramp_up', drive_element2)
                    play(drive_pulse, drive_element)
                    if drive_element2 is not None: play(drive_pulse2, drive_element2)
                    if is_ramp_drive:  
                        play(drive_pulse + '_ramp_down', drive_element)
                        if drive_element2 is not None: play(drive_pulse2 + '_ramp_down', drive_element2)
                if drive_detuning != 0:
                    update_frequency(drive_element, self.element_IF(drive_element), keep_phase = True)
                if drive_detuning2 != 0 and drive_element2 is not None:
                    update_frequency(drive_element2, self.element_IF(drive_element2), keep_phase = True)
                
                for qb in qubit:
                    wait(base_wait_time-pulse_duration, qb)
                    
                    if init_state != 'g':
                        if init_state in ['plus', 'minus']: 
                            play('pi2_pulse', qubit[0])
                            frame_rotation_2pi(init_phase, qb)
                        elif init_state == 'e': play('pi_pulse', qb)
                    if is_ramp :
                        play((rabi_pulse + '_ramp_up'), qb)
                    play(rabi_pulse, qb, duration = pulse_duration)
                    if is_ramp :
                        play((rabi_pulse + '_ramp_down'), qb)
                    if is_meas_x and not is_tomography:
                        frame_rotation_2pi(0.25, qb) # sigma x measurement
                        play(self.pi2_pulse, qb)
                        reset_frame(qb)
                
                if ring_down_time>0:
                    wait(int(ring_down_time//4), drive_element)
                    if drive_element2 is not None: wait(int(ring_down_time//4), drive_element2)
                    align(qubit[0], drive_element)
                    
                
                if is_tomography:
                    with switch_(x):
                        for i in range(len(tomo.pulse_seq_list)):
                            with case_(i):
                                tomo.play_tomo_pulse(tomo.pulse_seq_list[i], self.pulse_len(qubit[0], self.pi2_pulse))
                align(qubit[0], ro_element)    
                if is_active_reset:
                    for qb in qubit:
                        # if stark_shift_corr != 0: update_frequency(qb, self.element_IF(qb), keep_phase = True)
                        if stark_shift_corr != 0: update_frequency(qb, old_IF, keep_phase = True)
                
                if drive_element!=ro_element:
                    align(drive_element, ro_element)
                if drive_element2 is not None and drive_element2!=ro_element:
                    align(drive_element2, ro_element)
                    
                align(qubit[0], ro_element)
                if is_tomography:
                    I_out_name = 'I_tomo'
                    Q_out_name = 'Q_tomo'
                else:
                    I_out_name = 'I'
                    Q_out_name = 'Q'
                self.perform_full_measurement(I,Q, I_output_name = I_out_name, Q_output_name = Q_out_name, ro_element = ro_element, is_sb_cool = is_sb_cool, is_active_reset = is_active_reset, **kwargs)
                
        with program() as self.driven_rabi_prog:
            if is_tomography: tomo.declarations()
            else:
                I = declare(fixed)
                Q = declare(fixed)
            for qb,sshift in zip(qubit,stark_shift_corr):
                if stark_shift_corr != 0:
                    new_IF = declare(int, value = int(self.element_IF(qb)+sshift))
                    old_IF = declare(int, value = int(self.element_IF(qb)))
            pulse_duration = declare(int)
            n = declare(int)
            amp_scale = declare(fixed)
            with for_(n, 0, n < N_avg, n + 1):
                with for_each_(amp_scale, amp_scale_list):
                    if is_tomography:
                        for qb in qubit:
                            # if stark_shift_corr != 0: update_frequency(qb, self.element_IF(qb), keep_phase = True)
                            if stark_shift_corr != 0: update_frequency(qb, old_IF, keep_phase = True)
                        tomo.measure_beta_coeffs(ro_element, qubit, meas_func = self.perform_full_measurement, is_active_reset=is_active_reset, is_sb_cool = is_sb_cool, **kwargs)
                        with for_(tomo.x, 0, tomo.x<len(tomo.pulse_seq_list), tomo.x+1):
                            rabi_sequence(pulse_duration, tomo.I, tomo.Q, amp_scale, is_tomography, tomo.x)
                    else:
                        rabi_sequence(pulse_duration, I, Q, amp_scale, is_tomography = False)
                        
            self.last_prog = self.driven_rabi_prog
            
        
    def run_driven_rabi(self, is_continue = False, is_save_data = None,  is_plot_all = True, **kwargs): 
    
        results_dict = self.results['driven rabi']
        if results_dict['is_tomography']: 
            return self.run_one_qubit_tomography(self.driven_rabi_prog, len(results_dict['t']), results_dict = results_dict, **kwargs)
        
        amp_scale_list = results_dict['amp_scale']
        amp_npts = len(amp_scale_list)
        N_avg = results_dict['N_avg']
        t = results_dict['t']
        
        I,Q = self.run_prog(self.driven_rabi_prog, shape = (N_avg, amp_npts,-1), **kwargs)
        
        data, error = self.process_data([I,Q])
        
        results_dict['data'] = data
        results_dict['error'] = error
        
        freqs = []
        decays = []
        freqs_error = []
        decays_error = []
        for data,error,amp_scale in zip(results_dict['data'], results_dict['error'],results_dict['amp_scale']):
            fit_res, fit_error,_ = self.fit_and_plot('ExpCos', [data,error],t, title_str = f'Driven Rabi ({results_dict["qubit"]})', plot = False , is_calc_stat_error=True, txt = f'Amp scale = {amp_scale}')
            if None not in fit_res:
                freqs.append(fit_res[1]*1e3)
                freqs_error.append(fit_error[1]*1e3)
                decays.append(1e-3*fit_res[0])
                decays_error.append(1e-3*fit_error[0])
            else:
                freqs.append(np.nan)
                freqs_error.append(np.nan)
                decays.append(np.nan)
                decays_error.append(np.nan)
            
        freqs = np.array(freqs)
        freqs_error = np.array(freqs_error)
        decays=np.array(decays)
        decays_error = np.array(decays_error)
        
        results_dict['freqs']=freqs
        results_dict['freqs_error']=freqs_error
        results_dict['decays']=decays
        results_dict['decays_error']=decays_error
        
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'driven rabi')
        return self.plot_driven_rabi(**kwargs)

    def plot_driven_rabi(self, is_plot_all = True, **kwargs):
        
        results_dict = self.results['driven rabi']
        t = results_dict['t']
        ylabel = str(self.which_data)
        if self.which_data in ['Phase', 'phase']:
            ylabel += ' [Rad]'
        else:
            ylabel += ' [V]'
        
        amp_scale = results_dict['amp_scale']
        freqs = results_dict['freqs']
        freqs_error = results_dict['freqs_error']
        decays = results_dict['decays']
        decays_error = results_dict['decays_error']
        data = results_dict['data']
        error = results_dict['error']
        
        def fit_func(x,A,B):
            return A*x**2+B
        
        if len(amp_scale) == 1:
            return self.fit_and_plot('ExpCos', [data[0],error[0]], t, title_str = f'Driven Rabi ({results_dict["qubit"]})', is_calc_stat_error=True, txt = f'Amp scale = {amp_scale}', **kwargs)
        
        else:
            if is_plot_all:
                for data,error,amp_scale_temp in zip(results_dict['data'], results_dict['error'],results_dict['amp_scale']):
                    fit_res, fit_error,_ = self.fit_and_plot('ExpCos', [data,error],t, title_str = f'Driven Rabi ({results_dict["qubit"]})', is_calc_stat_error=True, txt = f'Amp scale = {amp_scale_temp}')

            fit_results_freq, cov_freq = curve_fit(fit_func, amp_scale, freqs*2*np.pi, sigma = freqs_error*2*np.pi)
            A_freq, A_freq_err = round_value_by_error(fit_results_freq[0], np.sqrt(cov_freq[0,0]))
            B_freq, B_freq_err = round_value_by_error(fit_results_freq[1], np.sqrt(cov_freq[1,1]))
            plt.figure()
            plt.errorbar(x = amp_scale, y = freqs*2*np.pi, yerr = freqs_error*2*np.pi, fmt = '.r', label = 'Data', capsize = 4, markersize=4)
            plt.plot(amp_scale, fit_func(amp_scale, *fit_results_freq), '-b', label = 'Fit')
            ann = plt.annotate(r'$\delta F = AV^2+B$'+'\n'\
                               +f'A = {A_freq}'+r'$ \pm$'+f'{A_freq_err}'+'\n'\
                               +f'B = {B_freq:.4f}'+r'$ \pm$'+f'{B_freq_err:.4f}',
                               [amp_scale[0], freqs.mean()*2*np.pi], fontsize = 12)
            ann.draggable()
            plt.xlabel('Amplitude [V]')
            plt.ylabel('Frequency [Rad MHz]')
            plt.legend(fontsize=10)
            plt.tight_layout()
            
            fit_results_decay, cov_decay = curve_fit(fit_func, amp_scale, 1/decays, sigma = decays_error/decays**2)
            A_decay, A_decay_err = round_value_by_error(fit_results_decay[0], np.sqrt(cov_decay[0,0]))
            B_decay, B_decay_err = round_value_by_error(fit_results_decay[1], np.sqrt(cov_decay[1,1]))
                
            plt.figure()
            plt.errorbar(x = amp_scale, y = (1/decays), yerr = decays_error/decays**2, fmt = '.r', label = 'Data', capsize = 4, markersize=4)
            plt.plot(amp_scale, fit_func(amp_scale,*fit_results_decay), '-b', label = 'Fit')
            ann2 = plt.annotate(r'$\Gamma_D = AV^2+B$'+'\n'\
                               +f'A = {A_decay}'+r'$ \pm$'+f'{A_decay_err}'+'\n'\
                               +f'B = {B_decay:.4f}'+r'$ \pm$'+f'{B_decay_err:.4f}',
                               [amp_scale[0], (1/decays).mean()], fontsize = 12)
            ann2.draggable()
            plt.xlabel('Amplitude [V]')
            plt.ylabel('Dephasing rate [MHz]')
            plt.legend(fontsize=10)
            plt.tight_layout()
                    
        
    def load_rabi_chevron(self,
                time_npts = 51, # How many points I measure
                max_seq_time = 80000, # Time of longest pulse in units of nano seconds. Must be a multiple of 4
                min_seq_time = None,
                time_step = None,
                detuning_npts = 101,
                detuning_start = -1e6,
                detuning_stop = 1e6,
                N_avg = 1000,
                qubit = None,
                ro_element = None,
                rabi_pulse = 'rabi_pulse',
                is_sb_cool = False,
                is_active_reset = None,
                is_ramp = True,
                **kwargs):
        
        if min_seq_time is not None:
            if min_seq_time > max_seq_time: raise ValueError("Oops... Maximum sequence time < Minimum sequence time.")
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        is_truncate = not self.configObject.SuperMembers[qubit].PulseParamsDict[rabi_pulse].pulseType == 'constant'
        if is_truncate and self.pulse_len(qubit, rabi_pulse) < max_seq_time:
            raise ValueError(f"You are trying to truncate the drive pulse, but it is too short. Pulse length >= max_seq_time is required. Got {self.pulse_len(qubit, rabi_pulse)}<{int(max_seq_time)}")
            
        if time_step is not None: time_npts = int((max_seq_time-min_seq_time) // time_step + 1)
        max_seq_clks = int(max_seq_time // 4)
        if min_seq_time is not None: 
            first_clk = min_seq_time//4
            step_size_clks = (max_seq_clks - first_clk) // (time_npts-1)
            if step_size_clks == 0:
                raise ValueError(f"Time steps are smaller than 4 ns. Got <(max_seq_time-min_seq_time)/4/(time_npts-1)> =  {np.around((max_seq_clks - first_clk) / (time_npts-1),4)}")
        else: 
            step_size_clks =  max_seq_time // 4 // (time_npts-1)
            first_clk = step_size_clks
            if step_size_clks == 0:
                raise ValueError(f"Time steps are smaller than 4 ns. Got <max_seq_time/4 /(time_npts-1)> =  {np.around(max_seq_clks /(time_npts-1),4)}")
        
        detuning_list = np.linspace(detuning_start, detuning_stop, detuning_npts)
        IF_set = self.element_IF(qubit)
        IF_list = (IF_set + detuning_list).astype(int)
        self.results['Rabi Chevron Pattern'] = {'t': np.arange(first_clk*4, max_seq_time+1, step_size_clks*4),
                                     'N_avg': N_avg,
                                     'detuning': detuning_list,
                                     'qubit': qubit,
                                     'ro_element': ro_element}
        
        if is_sb_cool:
            run_time = N_avg*time_npts*detuning_npts*(max_seq_time/2 + self.pulse_len(ro_element,self.ro_pulse) + self.pulse_len(ro_element, 'sideband_cooling_pulse'))
        elif is_active_reset:
            run_time = N_avg*time_npts*detuning_npts*(max_seq_time/2 + self.pulse_len(ro_element,self.ro_pulse))
        else:
            run_time = N_avg*time_npts*detuning_npts*(max_seq_time/2 + self.pulse_len(ro_element,self.ro_pulse)+self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))        
        
        
        with program() as self.rabi_chevron_prog:
            pulse_duration = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            f = declare(int)
            with for_(n, 0, n < N_avg, n + 1):
                with for_each_(f,IF_list):
                    with for_(pulse_duration,
                              first_clk,
                              pulse_duration <= max_seq_clks,
                              pulse_duration + step_size_clks):
                        update_frequency(qubit, f)
                        reset_frame(qubit)
                        if is_sb_cool: self.sideband_cool(qubit = qubit, ro_element = ro_element,  **kwargs)
                        
                        if self.e_to_f:  self.map_e_to_g(self.main_qubit_g_to_e)
                        
                        if is_truncate:
                            play(rabi_pulse, qubit, truncate = pulse_duration)
                        else:
                            if is_ramp :
                                play(rabi_pulse+'_ramp_up', qubit)
                            play(rabi_pulse, qubit, duration = pulse_duration)
                            if is_ramp :
                                play(rabi_pulse+'_ramp_down', qubit)
                                
                        if self.e_to_f and self.e_to_f_map_back:  self.map_e_to_g(qubit_g_to_e); align(qubit, ro_element, self.main_qubit_g_to_e)
                        
                        align(qubit, ro_element)
                        update_frequency(qubit, IF_set)
                        self.perform_full_measurement(I,Q, ro_element = ro_element, is_wait = not is_sb_cool, is_active_reset = is_active_reset, **kwargs)
        
            self.last_prog = self.rabi_chevron_prog
        
    def run_rabi_chevron(self, is_save_data = None, **kwargs):
        if is_save_data is None: is_save_data = self.is_save_data
        
        results_dict = self.results['Rabi Chevron Pattern']
        N_avg = results_dict['N_avg']
        detuning_npts = len(results_dict['detuning'])
        I,Q = self.run_prog(self.rabi_chevron_prog, shape = (N_avg, detuning_npts,-1), **kwargs)
        results_dict['I'] = I
        results_dict['Q'] = Q
        
        if is_save_data: self.pickle_save(results_dict, 'Rabi Chevron Pattern')
        
        self.plot_rabi_chevron(**kwargs)
    
    def plot_rabi_chevron(self, **kwargs):
        
        results_dict = self.results['Rabi Chevron Pattern']
        N_avg = results_dict['N_avg']
        t = results_dict['t']
        detuning = results_dict['detuning']
        data,err = self.process_data(results_dict)
        plot_2D(data.transpose(), detuning*1e-6, t, xlabel = 'Detuning [MHz]', ylabel = 'Time [ns]', cmap = 'Reds')
        
        f = []
        f_err = []
        decay = []
        decay_err = []
        for dat,er in zip(data,err):
            fit_res, fit_err, _ = self.fit_and_plot('ExpCos', [dat,er], t, plot = False)
            f.append(fit_res[1])
            f_err.append(fit_err[1])
            decay.append(fit_res[0])
            decay_err.append(fit_err[0])
        fig,axs = plt.subplots(2,1, sharex=True)
        plt.sca(axs[0])
        plt.errorbar(detuning*1e-6, f, f_err)
        plt.ylabel('Frequency [MHz]')
        plt.sca(axs[1])
        plt.errorbar(detuning*1e-6, decay, decay_err)
        plt.ylabel('Decay time [$\mu$s]')
        plt.xlabel('Detuning [MHz]')
        
    def load_power_rabi_chevron(self,
                amp_npts = 51, 
                amp_start = 1,
                amp_stop = 0,
                detuning_npts = 101,
                detuning_start = -1e6,
                detuning_stop = 1e6,
                N_avg = 1000,
                qubit = None,
                ro_element = None,
                rabi_pulse = 'rabi_pulse',
                is_sb_cool = False,
                is_active_reset = None,
                is_ramp = True,
                drive_element = None,
                drive_pulse = None,
                ss_corr = 0,
                steady_time = 0,
                is_meas_x = False,
                init_state = 'g',
                is_ramp_drive = True,
                **kwargs):
        
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if is_active_reset is None: is_active_reset = self.is_active_reset
        print(f'The Rabi pulse length is {self.pulse_len(qubit, rabi_pulse)}')
        
        amp_scale_list = np.linspace(amp_start, amp_stop, amp_npts)
        detuning_list = np.linspace(detuning_start, detuning_stop, detuning_npts)
        IF_set_qb = self.element_IF(qubit)
        IF_set = self.element_IF(drive_element)
        IF_list = (IF_set + detuning_list).astype(int)
        self.power_rabi_chevron_results = {'amp': amp_scale_list * self.pulse_amp(qubit, rabi_pulse),
                                            'N_avg': N_avg,
                                            'detuning': detuning_list,
                                            'qubit': qubit,
                                            'ro_element': ro_element}
        
        if is_sb_cool:
            run_time = N_avg*amp_npts*detuning_npts*(self.pulse_len(qubit, rabi_pulse) + self.pulse_len(ro_element,self.ro_pulse) + self.pulse_len(ro_element, 'sideband_cooling_pulse'))
        elif is_active_reset:
            run_time = N_avg*amp_npts*detuning_npts*(self.pulse_len(qubit, rabi_pulse) + self.pulse_len(ro_element,self.ro_pulse))
        else:
            run_time = N_avg*amp_npts*detuning_npts*(self.pulse_len(qubit, rabi_pulse) + self.pulse_len(ro_element,self.ro_pulse)+self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))        
        
        if init_state != 'g':
            if init_state =='plus':
                init_phase = 0.25
                init_pulse = 'pi2_pulse'
            if init_state =='minus':
                init_phase = -0.25
                init_pulse = 'pi2_pulse'
            
        
        with program() as self.power_rabi_chevron_prog:
            amp_scale = declare(fixed)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            f = declare(int)
            with for_(n, 0, n < N_avg, n + 1):
                with for_each_(f,IF_list):
                    with for_each_(amp_scale, amp_scale_list):
                        if ss_corr != 0: update_frequency(qubit, IF_set_qb+ss_corr)
                        update_frequency(drive_element, f)
                        reset_frame(qubit)
                        if drive_element is not None and drive_pulse is not None:
                            align(drive_element, qubit)
                        if steady_time > 0:
                            wait(int(steady_time//4), qubit)
                        if drive_element is not None and drive_pulse is not None:
                            if is_ramp_drive:
                                play(drive_pulse+'_ramp_up', drive_element)
                            play(drive_pulse, drive_element)
                            if is_ramp_drive:
                                play(drive_pulse+'_ramp_down', drive_element)
                        if init_state != 'g':
                            play(init_pulse, qubit)
                            frame_rotation_2pi(init_phase, qubit)
                        if is_sb_cool: self.sideband_cool(qubit = qubit, ro_element = ro_element,  **kwargs)
                        if self.e_to_f:  self.map_e_to_g(self.main_qubit_g_to_e)
                        
                        if is_ramp :
                            play((rabi_pulse+'_ramp_up') * amp(amp_scale), qubit)
                        play(rabi_pulse * amp(amp_scale), qubit)
                        if is_ramp :
                            play((rabi_pulse+'_ramp_down') * amp(amp_scale), qubit)
                                
                        if ss_corr != 0: update_frequency(qubit, IF_set_qb+ss_corr)
                        if is_meas_x: 
                            frame_rotation_2pi(0.25, qubit)
                            play('pi2_pulse', qubit)
                        if self.e_to_f and self.e_to_f_map_back:  self.map_e_to_g(qubit_g_to_e); align(qubit, ro_element, self.main_qubit_g_to_e)
                        
                        align(qubit, ro_element)
                        if drive_element is not None and drive_pulse is not None:
                            wait(200, drive_element)
                        # update_frequency(qubit, IF_set)
                        update_frequency(drive_element, IF_set)
                        self.perform_full_measurement(I,Q, ro_element = ro_element, is_wait = not is_sb_cool, is_active_reset = is_active_reset, **kwargs)
        
            self.last_prog = self.power_rabi_chevron_prog
        
    def run_power_rabi_chevron(self, is_save_data = None, **kwargs):
        if is_save_data is None: is_save_data = self.is_save_data
        
        results_dict = self.power_rabi_chevron_results
        N_avg = results_dict['N_avg']
        detuning_npts = len(results_dict['detuning'])
        I,Q = self.run_prog(self.power_rabi_chevron_prog, shape = (N_avg, detuning_npts,-1), **kwargs)
        results_dict['I'] = I
        results_dict['Q'] = Q
        
        if is_save_data: self.pickle_save(results_dict, 'Power Rabi Chevron Pattern')
        
        self.plot_power_rabi_chevron(**kwargs)
    
    def plot_power_rabi_chevron(self, **kwargs):
        
        results_dict = self.power_rabi_chevron_results
        N_avg = results_dict['N_avg']
        amp_list = results_dict['amp']
        detuning = results_dict['detuning']
        data,err = self.process_data(results_dict)
        plot_2D(data.transpose(), detuning*1e-6, amp_list, xlabel = 'Detuning [MHz]', ylabel = 'Drive Amplitude [V]', cmap = 'Reds')
        
        
    def load_driven_rabi_chevron(self,
                time_npts = 51, # How many points I measure
                max_seq_time = 80000, # Time of longest pulse in units of nano seconds. Must be a multiple of 4
                min_seq_time = None,
                time_step = None,
                detuning_npts = 101,
                detuning_start = -1e6,
                detuning_stop = 1e6,
                N_avg = 1000,
                qubit = None,
                ro_element = None,
                rabi_pulse = 'rabi_pulse',
                is_sb_cool = False,
                is_active_reset = None,
                is_ramp = True,
                drive_element = None,
                drive_pulse = 'mixer_cal_pulse',
                extra_drive_time = 0,
                ring_down_time = 0,
                is_meas_x = False,
                **kwargs):
        
        if min_seq_time is not None:
            if min_seq_time > max_seq_time: raise ValueError("Oops... Maximum sequence time < Minimum sequence time.")
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if drive_element is None: drive_element = self.main_readout
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        ramp_up_length = self.configObject.SuperMembers[qubit].PulseParamsDict[rabi_pulse].Additions.ramp['up']['length']
        ramp_down_length = self.configObject.SuperMembers[qubit].PulseParamsDict[rabi_pulse].Additions.ramp['down']['length']
        if not is_ramp and self.pulse_len(drive_element, drive_pulse) < max_seq_time:
            raise ValueError(f"Drive pulse too short. Must be longer than max_seq_time. Got {self.pulse_len(drive_element, drive_pulse)}<{max_seq_time}")
        elif is_ramp and int((self.pulse_len(drive_element, drive_pulse)-ramp_up_length-ramp_down_length-extra_drive_time)) < max_seq_time:
            raise ValueError(f"Drive pulse too short. Must be longer than max_seq_time - ramp time. Got {int((self.pulse_len(drive_element, drive_pulse)-ramp_up_length-ramp_down_length-extra_drive_time))}<{max_seq_time}")
            
            
        is_truncate = not self.configObject.SuperMembers[qubit].PulseParamsDict[rabi_pulse].pulseType == 'constant'
        if is_truncate and self.pulse_len(qubit, rabi_pulse) < max_seq_time:
            raise ValueError(f"You are trying to truncate the drive pulse, but it is too short. Pulse length >= max_seq_time is required. Got {self.pulse_len(qubit, rabi_pulse)}<{int(max_seq_time)}")
            
        if time_step is not None: time_npts = int((max_seq_time-min_seq_time) // time_step + 1)
        max_seq_clks = int(max_seq_time // 4)
        if min_seq_time is not None: 
            first_clk = min_seq_time//4
            step_size_clks = (max_seq_clks - first_clk) // (time_npts-1)
            if step_size_clks == 0:
                raise ValueError(f"Time steps are smaller than 4 ns. Got <(max_seq_time-min_seq_time)/4/(time_npts-1)> =  {np.around((max_seq_clks - first_clk) / (time_npts-1),4)}")
        else: 
            step_size_clks =  max_seq_time // 4 // (time_npts-1)
            first_clk = step_size_clks
            if step_size_clks == 0:
                raise ValueError(f"Time steps are smaller than 4 ns. Got <max_seq_time/4 /(time_npts-1)> =  {np.around(max_seq_clks /(time_npts-1),4)}")
        
        detuning_list = np.linspace(detuning_start, detuning_stop, detuning_npts)
        IF_set = self.element_IF(qubit)
        IF_list = (IF_set + detuning_list).astype(int)
        self.driven_rabi_chevron_results = {'t': np.arange(first_clk*4, max_seq_time+1, step_size_clks*4),
                                             'N_avg': N_avg,
                                             'detuning': detuning_list,
                                             'qubit': qubit,
                                             'ro_element': ro_element}
        
        if is_sb_cool:
            run_time = N_avg*time_npts*detuning_npts*(self.pulse_len(drive_element,drive_pulse) + self.pulse_len(ro_element,self.ro_pulse) + self.pulse_len(ro_element, 'sideband_cooling_pulse'))
        elif is_active_reset:
            run_time = N_avg*time_npts*detuning_npts*(self.pulse_len(drive_element,drive_pulse) + self.pulse_len(ro_element,self.ro_pulse))
        else:
            run_time = N_avg*time_npts*detuning_npts*(self.pulse_len(drive_element,drive_pulse) + self.pulse_len(ro_element,self.ro_pulse)+self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))        
        
        
        
        with program() as self.driven_rabi_chevron_prog:
            pulse_duration = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            f = declare(int)
            with for_(n, 0, n < N_avg, n + 1):
                with for_each_(f,IF_list):
                    with for_(pulse_duration,
                              first_clk,
                              pulse_duration <= max_seq_clks,
                              pulse_duration + step_size_clks):
                        update_frequency(qubit, f)
                        reset_frame(qubit)
                        align(drive_element, qubit)
                        play(drive_pulse, drive_element)
                        if is_ramp:
                            wait(int((self.pulse_len(drive_element, drive_pulse)-ramp_up_length-ramp_down_length-extra_drive_time)//4)-pulse_duration, qubit)
                        else:
                            wait(int((self.pulse_len(drive_element, drive_pulse)-extra_drive_time)//4)-pulse_duration, qubit)
                            
                        if is_ramp :
                            play('ramp_up', qubit)
                        play(rabi_pulse, qubit, duration = pulse_duration)
                        if is_ramp :
                            play('ramp_down', qubit)
                        
                        if ring_down_time>0:
                            wait(int(ring_down_time//4),drive_element)
                        if is_meas_x :
                            frame_rotation_2pi(0.25, qubit) # sigma x measurement
                            play(self.pi2_pulse, qubit)
                            reset_frame(qubit)
                            
                        update_frequency(qubit, IF_set)
                                        
                        if drive_element!=ro_element:
                            align(drive_element, ro_element)
                        align(qubit, ro_element)
    
                        I_out_name = 'I'
                        Q_out_name = 'Q'
                        self.perform_full_measurement(I,Q, I_output_name = I_out_name, Q_output_name = Q_out_name, ro_element = ro_element, is_wait = not is_sb_cool, is_active_reset = is_active_reset, **kwargs)

                
        self.last_prog = self.driven_rabi_chevron_prog
        
    def run_driven_rabi_chevron(self, is_save_data = None, **kwargs):
        if is_save_data is None: is_save_data = self.is_save_data
        
        results_dict = self.driven_rabi_chevron_results
        N_avg = results_dict['N_avg']
        detuning_npts = len(results_dict['detuning'])
        I,Q = self.run_prog(self.driven_rabi_chevron_prog, shape = (N_avg, detuning_npts,-1), **kwargs)
        results_dict['I'] = I
        results_dict['Q'] = Q
        
        if is_save_data: self.pickle_save(results_dict, 'Driven Rabi Chevron Pattern')
        
        self.plot_driven_rabi_chevron(**kwargs)
    
    def plot_driven_rabi_chevron(self, **kwargs):
        
        results_dict = self.driven_rabi_chevron_results
        N_avg = results_dict['N_avg']
        t = results_dict['t']
        detuning = results_dict['detuning']
        data,err = self.process_data(results_dict)
        plot_2D(data.transpose(), detuning*1e-6, t, xlabel = 'Detuning [MHz]', ylabel = 'Time [ns]', cmap = 'Reds')
        
        f = []
        f_err = []
        decay = []
        decay_err = []
        for dat,er in zip(data,err):
            fit_res, fit_err, _ = self.fit_and_plot('ExpCos', [dat,er], t, plot = False)
            f.append(fit_res[1])
            f_err.append(fit_err[1])
            decay.append(fit_res[0])
            decay_err.append(fit_err[0])
        fig,axs = plt.subplots(2,1, sharex=True)
        plt.sca(axs[0])
        plt.errorbar(detuning*1e-6, f, f_err)
        plt.ylabel('Frequency [MHz]')
        plt.sca(axs[1])
        plt.errorbar(detuning*1e-6, decay, decay_err)
        plt.ylabel('Decay time [$\mu$s]')
        plt.xlabel('Detuning [MHz]')
#%% All XY specific

    def load_all_xy(self, N_avg = 1000, qubit = None, ro_element = None, is_active_reset = None):
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        self.all_xy_results = {'qubit': qubit}
        
        self.all_xy_N_avg = N_avg    
        
        self.all_xy_sequence = [('+I', '+I'), ('+X', '+X'), ('+Y', '+Y'), ('+X', '+Y'), ('+Y', '+X'), # ground
                                ('+x', '+I'), ('+y', '+I'), ('+x', '+y'), ('+y', '+x'), ('+x', '+Y'), # g + e
                                ('+y', '+X'), ('+X', '+y'), ('+Y', '+x'), ('+x', '+X'), ('+X', '+x'), # g + e
                                ('+y', '+Y'), ('+Y', '+y'),                                           # g + e
                                ('+X', '+I'), ('+Y', '+I'), ('+x', '+x'), ('+y', '+y')]               # e
        
        amp_and_phase_from_pulse_name = {'+I': (0,0.0),
                                         '+X': (1.0,0.0),
                                         '-X': (1.0,0.5),
                                         '+x': (0.5,0.0),
                                         '-x': (0.5,0.5),
                                         '+Y': (1.0,0.25),
                                         '-Y': (1.0,0.75),
                                         '+y': (0.5,0.25),
                                         '-y': (0.5,0.75),
                                         }
        
        if is_active_reset:run_time = N_avg * len(self.all_xy_sequence) * self.pulse_len(ro_element, 'ro_pulse')
        else: run_time = N_avg * len(self.all_xy_sequence) * self.wait_between_seq
        print('Run time is {}s'.format(np.round(run_time * 1e-9)))
        
        amp1_list = []; phase1_list = []
        amp2_list = []; phase2_list = []
        
        for pulse_seq in self.all_xy_sequence:
            amp1_list.append(amp_and_phase_from_pulse_name[pulse_seq[0]][0])
            phase1_list.append(amp_and_phase_from_pulse_name[pulse_seq[0]][1])
            amp2_list.append(amp_and_phase_from_pulse_name[pulse_seq[1]][0])
            phase2_list.append(amp_and_phase_from_pulse_name[pulse_seq[1]][1])
        
        with program() as self.all_xy_prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            x = declare(int)
            with for_(n, 0, n<N_avg, n+1):
                
                with for_(x,0,x<len(amp1_list),x+1):
                    with switch_(x):
                        for i in range(len(amp1_list)):
                            with case_(i):
                                frame_rotation_2pi(phase1_list[i], qubit)
                                if amp1_list[i] == 1.0:
                                    play(self.pi_pulse, qubit)
                                elif amp1_list[i] == 0.5:
                                    play(self.pi2_pulse, qubit)
                                elif amp1_list[i] == 0.0:
                                    wait(self.pulse_len(qubit, self.pi2_pulse)//4, qubit)
                                reset_frame(qubit)
                                frame_rotation_2pi(phase2_list[i], qubit)
                                if amp2_list[i] == 1.0:
                                    play(self.pi_pulse, qubit)
                                elif amp2_list[i] == 0.5:
                                    play(self.pi2_pulse, qubit)
                                elif amp2_list[i] == 0.0:
                                    wait(self.pulse_len(qubit, self.pi2_pulse)//4, qubit)
                                reset_frame(qubit)
                                align(qubit, ro_element)
                                
                    self.perform_full_measurement(I,Q, ro_element = ro_element, is_active_reset = is_active_reset)
                    
                
                    
        self.last_prog = self.all_xy_prog
    
    def run_all_xy(self,fig_num = None,
                   plot_th = False,
                   is_calc_stat_error = None):
        
        if is_calc_stat_error is None: is_calc_stat_error = self.is_calc_stat_error
        if self.all_xy_prog is None: raise ValueError("Idiot! You did not write all_xy program")
        
        self.all_xy_results['I'], self.all_xy_results['Q'] = self.run_prog(self.all_xy_prog, shape = (self.all_xy_N_avg,-1))
        
        prcd_data, err = self.process_data(self.all_xy_results)
        all_xy_data, units_prefix,scale = self.autoscale_data(prcd_data)
        
        qubit = self.all_xy_results['qubit']
        prog_name = 'All XY'
        if fig_num is None: fig = plt.figure(next_fig_num_by_name(prog_name))
        else: fig = plt.figure(fig_num)
        plt.errorbar(x=np.linspace(0,len(self.all_xy_sequence)-1,len(self.all_xy_sequence)),y=all_xy_data, yerr=err*scale, capsize=6,fmt='-o', alpha = 0.5, label = f'pi/2 drag = {np.round(self.pulse_drag(qubit, self.pi2_pulse),4)}, pi drag = {np.round(self.pulse_drag(qubit, self.pi_pulse),4)}')
        plt.xticks(np.linspace(0,len(self.all_xy_sequence)-1,len(self.all_xy_sequence)), (self.all_xy_sequence), rotation = 60, fontsize = 12)
        plt.title(f'All XY ({qubit})')
        plt.grid()
        # txt = f'DRAG param {self.pi2_pulse} = {self.pulse_drag(qubit, self.pi2_pulse)}'
        # if self.is_pi_pulse: txt = txt + f'\n DRAG param {self.pi_pulse} = {self.pulse_drag(qubit, self.pi_pulse)}'
        # annot = plt.annotate(txt, xy=(0.1,0.7),xycoords ='figure fraction')
        # annot.draggable(True)
        leg = plt.legend(title = 'DRAG param', fontsize = 10, title_fontsize = 10, ncol=2)
        leg.set_draggable(True)
        if self.which_data == 'Phase': plt.ylabel('{1} [{0}Rad]'.format(units_prefix, self.which_data))
        else: plt.ylabel('{1} [{0}V]'.format(units_prefix, self.which_data))
        
        plt.tight_layout()
        
        return fig
    
    def run_drag_calib(self, start, stop, npts, N_avg = 1000, scnd_pulse = 'pi_pulse', qubit = None, ro_element = None, 
                       fig_num = None, is_calc_stat_error= None, is_active_reset = None):
        if is_active_reset is None: is_active_reset = self.is_active_reset
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        
        if np.abs(stop)<np.abs(start): start,stop = stop,start
        if is_calc_stat_error is None:
            is_calc_stat_error=self.is_calc_stat_error
        if is_active_reset: run_time = N_avg * 2 * npts  * self.pulse_len(ro_element, "ro_pulse")
        else: run_time = N_avg * 2 * npts  * self.wait_between_seq
        print('Run time is {}s'.format(np.round(run_time * 1e-9)))
        
        old_pi_drag_param = self.pulse_drag(qubit, 'pi_pulse')
        old_pi2_drag_param = self.pulse_drag(qubit, 'pi2_pulse')
        self.pulse_drag(qubit, 'pi2_pulse', stop)
        self.pulse_drag(qubit, 'pi_pulse', stop)
        
        # self.pulse_drag(qubit, 'x', stop)
        # self.pulse_drag(qubit, 'y', stop)
        
        param_list = np.linspace(start,stop,npts)
        param_scale_list = np.linspace(start/stop, 1, npts).astype(float).tolist()
        
        for param_scale in param_scale_list:
            if param_scale > 2 or param_scale < -2:
                raise ValueError(f"Scaling must be in (-2,2). Got {param_scale}")
        
        
        with program() as self.drag_param_calib_prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            drag_param_scale = declare(fixed)
            with for_(n, 0, n<N_avg, n+1):
                with for_each_(drag_param_scale, param_scale_list):
                    play('pi2_pulse' * amp(1,0,0,drag_param_scale), qubit)
                    frame_rotation_2pi(0.25, qubit)
                    play(scnd_pulse * amp(1,0,0,drag_param_scale), qubit)
                    align(qubit, ro_element)
                    self.perform_full_measurement(I,Q,'Ixy','Qxy', ro_element = ro_element, is_active_reset = is_active_reset)
                    
                    align(qubit, ro_element)
                    play('pi2_pulse' * amp(1,0,0,drag_param_scale), qubit)
                    frame_rotation_2pi(-0.25, qubit)
                    play(scnd_pulse * amp(1,0,0,drag_param_scale), qubit)
                    align(qubit, ro_element)
                    self.perform_full_measurement(I,Q,'Iyx','Qyx', ro_element = ro_element, is_active_reset = is_active_reset)
        self.last_prog = self.drag_param_calib_prog
    
        
        self.qm_server.clear_all_job_results()            
        job = self.qm.execute(self.drag_param_calib_prog, duration_limit = 0, data_limit = 0)
        job.result_handles.wait_for_all_values()
        
        self.pulse_drag(qubit, 'pi_pulse', old_pi_drag_param)
        self.pulse_drag(qubit, 'pi2_pulse', old_pi2_drag_param)
        # self.pulse_drag(qubit, 'x', old_pi_drag_param)
        # self.pulse_drag(qubit, 'y', old_pi2_drag_param)
        
        Ixy = job.result_handles.get('Ixy').fetch_all()['value'].reshape((N_avg,-1))
        Qxy = job.result_handles.get('Qxy').fetch_all()['value'].reshape((N_avg,-1))
        Iyx = job.result_handles.get('Iyx').fetch_all()['value'].reshape((N_avg,-1))
        Qyx = job.result_handles.get('Qyx').fetch_all()['value'].reshape((N_avg,-1))
        
        xy_data, xy_err = self.process_data([Ixy,Qxy])
        yx_data, yx_err = self.process_data([Iyx,Qyx])
        xy_data, units_prefix,factor = self.autoscale_data(xy_data)
        yx_data = yx_data * factor
        xy_err = xy_err*factor
        yx_err = yx_err*factor
        
        prog_name = 'Drag Calibration'
        if fig_num is None: fig = plt.figure(next_fig_num_by_name(prog_name))
        else: fig = plt.figure(fig_num)
        plt.errorbar(param_list,xy_data,xy_err,capsize = 6, fmt =  'or', label = 'xY')
        plt.errorbar(param_list,yx_data,yx_err, capsize = 6, fmt= 'ob', label = 'yX')
        
        sfit_xy = sFit('Line', xy_data, param_list, xy_err)
        sfit_yx = sFit('Line', yx_data, param_list, yx_err)
        
        param_list_for_plot = np.linspace(param_list[0], param_list[-1], 1001)
        
        plt.plot(param_list_for_plot, sfit_xy.func(param_list_for_plot, *sfit_xy.fit_results), '-r')
        plt.plot(param_list_for_plot, sfit_yx.func(param_list_for_plot, *sfit_yx.fit_results), '-b')
        
        ind_best = np.argmin(np.abs(sfit_xy.func(param_list_for_plot, *sfit_xy.fit_results)-sfit_yx.func(param_list_for_plot, *sfit_yx.fit_results)))
        found_drag_param = param_list_for_plot[ind_best]
        
        plt.xticks(fontsize = 10)
        plt.title(f'Drag Calibration ({qubit})')
        leg = plt.legend(fontsize = 10)
        leg.set_draggable(True)
        if self.which_data == 'Phase': plt.ylabel('{1} [{0}Rad]'.format(units_prefix, self.which_data))
        else: plt.ylabel('{1} [{0}V]'.format(units_prefix, self.which_data))
        
        
        print(f"Drag param found is {np.round(found_drag_param,5)}")
    
    
#%% pinopi_spectroscopy
    
    def load_pinopi_spec(self, N_avg = 1000, npts = 101,
                         start = None, stop = None,
                         qubit = None,
                         ro_element = None,
                         pi_pulse = 'pi_pulse',
                         is_continuous_drive = False,
                         drive_pulse = 'mixer_cal_pulse',
                         ro_pulse = None,
                         is_active_reset = False,
                         **kwargs):
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        detuning = np.linspace(start, stop, npts, dtype = int)
        freqs = detuning + self.element_freq(ro_element)
            
        IF0 = int(self.element_IF(ro_element))
        
        self.results['pinopi_spec'] = {'freqs': freqs, 'N_avg': N_avg, 'npts': npts, 'ro_element': ro_element, 'qubit': qubit, 'ro_pulse': ro_pulse}
        detunings = freqs-self.element_freq(ro_element)
        self.results['pinopi_spec']['detunings'] =  detunings
        run_time = 2*N_avg*npts*(self.pulse_len(ro_element, ro_pulse)+self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))
        
        
        mm = 'mm2'
        with program() as self.pinopi_spec_prog:
            
            n = declare(int)
            I_pi = declare(fixed)
            I_nopi = declare(fixed)
            Q_pi = declare(fixed)
            Q_nopi = declare(fixed)
            f = declare(int)
            
            with for_(n, 0, n < N_avg, n + 1):
                with for_each_(f, detuning + IF0):
                    update_frequency(ro_element, f)
                    align(qubit, ro_element)             
                    
                    if self.e_to_f:  self.map_e_to_g(self.main_qubit_g_to_e)
                    play('pi_pulse', qubit)
                    # update_frequency('qb1', -420e6)
                    # play('pi_pulse' * amp(0.727), 'qb1')
                    # update_frequency('qb1', -150e6)
                    if self.e_to_f and self.e_to_f_map_back:  self.map_e_to_g(qubit_g_to_e)
                    self.perform_full_measurement(I_pi, Q_pi,  I_output_name = 'I_pi', Q_output_name = 'Q_pi', is_amplify = False, ro_element = ro_element, is_active_reset = is_active_reset, readout_pulse = ro_pulse, **kwargs)
                    
                    align(qubit, ro_element)
                    if self.e_to_f:  self.map_e_to_g(self.main_qubit_g_to_e)
                    if self.e_to_f and self.e_to_f_map_back:  self.map_e_to_g(self.main_qubit_g_to_e)
                    align(qubit, ro_element)
                    self.perform_full_measurement(I_nopi, Q_nopi, I_output_name = 'I_nopi', Q_output_name = 'Q_nopi', is_amplify = False, ro_element = ro_element, is_active_reset = is_active_reset, readout_pulse = ro_pulse, **kwargs)
                    
        
        self.last_prog = self.pinopi_spec_prog  
    
    
    def run_pinopi_spec(self, is_calc_stat_error = None,
                        is_plot_configured_freq = True,
                        is_fit_mag = False, pnts_to_fit = 11,
                        is_save_data = None,
                        **kwargs): 
        results_dict = self.results['pinopi_spec']
        if is_save_data is None: is_save_data = self.is_save_data
        if is_calc_stat_error is None:
            is_calc_stat_error = self.is_calc_stat_error
            
        
        if not hasattr(self, 'pinopi_spec_prog'): raise ValueError("Idiot! You did not write pinopi program")
        
        freqs = results_dict['freqs']
        N_avg = results_dict['N_avg']
        npts = results_dict['npts']
        ro_element = results_dict['ro_element']
        detunings = results_dict['freqs']-self.element_freq(ro_element)
        self.qm_server.clear_all_job_results()
        self.pinopi_spec_job = self.qm.execute(self.pinopi_spec_prog, duration_limit=0, data_limit=0)
        self.pinopi_spec_job.result_handles.wait_for_all_values()
        self.pinopi_spec_job.execution_report()
        
        results_dict['I_pi'] = self.pinopi_spec_job.result_handles.get('I_pi').fetch_all()['value'].reshape((N_avg, -1))
        results_dict['Q_pi'] = self.pinopi_spec_job.result_handles.get('Q_pi').fetch_all()['value'].reshape((N_avg, -1))
        results_dict['I_nopi'] = self.pinopi_spec_job.result_handles.get('I_nopi').fetch_all()['value'].reshape((N_avg, -1))
        results_dict['Q_nopi'] = self.pinopi_spec_job.result_handles.get('Q_nopi').fetch_all()['value'].reshape((N_avg, -1))
        results_dict['tof'] = self.tof(ro_element)*1e-9
        
        if is_save_data: self.pickle_save(results_dict, 'pinopi_spec')
        self.plot_pinopi_spec(is_fit_mag = is_fit_mag)
        
    def plot_pinopi_spec(self, is_fit_mag = False, pnts_to_fit = 11,
                         is_tof_correction = True, tof = None,
                         is_plot_configured_freq = True, configured_freq = None):
        
        results_dict = self.results['pinopi_spec']
        freqs = results_dict['detunings']
        detunings = results_dict['detunings']
        freqs, freqs_prefix, _ = self.autoscale_data(freqs)
        
        if is_tof_correction:
            
            if tof is None: tof = results_dict['tof']
            phase_corr = -tof*detunings*2*np.pi
            I_pi = np.cos(phase_corr) * results_dict['I_pi'] \
                + np.sin(phase_corr) * results_dict['Q_pi']
                
            Q_pi = -np.sin(phase_corr) * results_dict['I_pi'] \
                + np.cos(phase_corr) * results_dict['Q_pi']
                
            I_nopi = np.cos(phase_corr) * results_dict['I_nopi'] \
                + np.sin(phase_corr) * results_dict['Q_nopi']
                
            Q_nopi = -np.sin(phase_corr) * results_dict['I_nopi'] \
                + np.cos(phase_corr) * results_dict['Q_nopi']
        
        data_pi = [I_pi, Q_pi]
        I_pi_list, I_pi_err = self.process_data(data_pi, which_data = 'I', is_calc_stat_error=True)
        _, I_pi_prefix, I_pi_factor = self.autoscale_data(I_pi_list)
        Q_pi_list, Q_pi_err = self.process_data(data_pi, which_data = 'Q', is_calc_stat_error=True)
        _, Q_pi_prefix, Q_pi_factor = self.autoscale_data(Q_pi_list)
        phase_pi_list, phase_pi_err=self.process_data(data_pi, which_data = 'Phase', is_calc_stat_error=True)
        mag_pi_list, mag_pi_err=self.process_data(data_pi, which_data = 'Mag', is_calc_stat_error=True)
        
        data_nopi = [I_nopi, Q_nopi]
        I_nopi_list, I_nopi_err = self.process_data(data_nopi, which_data = 'I', is_calc_stat_error=True)
        _, I_nopi_prefix, I_nopi_factor = self.autoscale_data(I_nopi_list)
        Q_nopi_list, Q_nopi_err = self.process_data(data_nopi, which_data = 'Q', is_calc_stat_error=True)
        _, Q_nopi_prefix, Q_nopi_factor = self.autoscale_data(Q_nopi_list)
        phase_nopi_list, phase_nopi_err=self.process_data(data_nopi, which_data = 'Phase', is_calc_stat_error=True)
        mag_nopi_list, mag_nopi_err=self.process_data(data_nopi, which_data = 'Mag', is_calc_stat_error=True)
        
        factor = 1
        prfx=''
        for fct, prfx in zip([I_pi_factor,Q_pi_factor,I_nopi_factor,Q_nopi_factor], 
                             [I_pi_prefix,Q_pi_prefix,I_nopi_prefix,Q_nopi_prefix]):
            if fct >= factor:
                factor = fct
                prefix = prfx
                
        I_pi_list, I_pi_err = (I_pi_list*factor, I_pi_err*factor)
        Q_pi_list, Q_pi_err = (Q_pi_list*factor, Q_pi_err*factor)
        I_nopi_list, I_nopi_err = (I_nopi_list*factor, I_nopi_err*factor)
        Q_nopi_list, Q_nopi_err = (Q_nopi_list*factor, Q_nopi_err*factor)
        mag_pi_list, mag_pi_err = (mag_pi_list*factor, mag_pi_err*factor)
        mag_nopi_list, mag_nopi_err = (mag_nopi_list*factor, mag_nopi_err*factor)
           
        I_diff = np.abs(I_pi_list - I_nopi_list)
        Q_diff = np.abs(Q_pi_list - Q_nopi_list)
        phase_diff = np.abs(phase_pi_list - phase_nopi_list)
        mag_diff = np.abs(mag_pi_list - mag_nopi_list)
        total_diff = I_diff + Q_diff
        print(f'Found highest total difference (|Ipi-Inopi|+|Qpi-Qnopi|) at {freqs[np.argmax(total_diff)]} MHz')
        
        pi_fmt = 'r-'
        nopi_fmt = 'b-'
        
        if is_fit_mag: 
            def fit_func(x,A,B,C,D):
                return (A/((x-B)**2+C**2) + D)
            max_pi_ind = np.argmax(np.abs(mag_pi_list - mag_pi_list[0]))
            max_nopi_ind = np.argmax(np.abs(mag_nopi_list - mag_pi_list[0]))
            # max_nopi_ind = np.argmax(mag_nopi_list)
            freqs_to_fit_pi = freqs[max_pi_ind-(pnts_to_fit-1)//2:max_pi_ind+(pnts_to_fit-1)//2+1]
            freqs_to_fit_nopi = freqs[max_nopi_ind-(pnts_to_fit-1)//2:max_nopi_ind+(pnts_to_fit-1)//2+1]
            
            mag_pi_to_fit =  mag_pi_list[max_pi_ind-(pnts_to_fit-1)//2:max_pi_ind+(pnts_to_fit-1)//2+1]
            mag_nopi_to_fit =  mag_nopi_list[max_nopi_ind-(pnts_to_fit-1)//2:max_nopi_ind+(pnts_to_fit-1)//2+1]
            
            C_guess_pi = np.abs(freqs_to_fit_pi[np.argmin(np.abs(mag_pi_to_fit-(max(mag_pi_to_fit)+min(mag_pi_to_fit))/2))]-freqs_to_fit_pi[np.argmax(mag_nopi_to_fit)])   # b_guess :)
            A_guess_pi = (np.max(mag_pi_to_fit)-np.min(mag_pi_to_fit))*C_guess_pi**2
            
            C_guess_nopi = np.abs(freqs_to_fit_nopi[np.argmin(np.abs(mag_nopi_to_fit-(max(mag_nopi_to_fit)+min(mag_nopi_to_fit))/2))]-freqs_to_fit_nopi[np.argmax(mag_nopi_to_fit)])   # b_guess :)
            A_guess_nopi = (np.max(mag_nopi_to_fit)-np.min(mag_nopi_to_fit))*C_guess_nopi**2
            
            B_guess_pi = freqs[max_pi_ind]
            B_guess_nopi = freqs[max_nopi_ind]
            guess_pi = [A_guess_pi, B_guess_pi, C_guess_pi,1]
            guess_nopi = [A_guess_nopi, B_guess_nopi, C_guess_nopi,1]
            try:
                fit_pi, cov_pi = curve_fit(fit_func, freqs_to_fit_pi, mag_pi_to_fit, p0 = guess_pi, sigma = mag_pi_list[max_pi_ind-(pnts_to_fit-1)//2:max_pi_ind+(pnts_to_fit-1)//2+1])
                fit_nopi, cov_nopi = curve_fit(fit_func, freqs_to_fit_nopi, mag_nopi_to_fit, p0 = guess_nopi, sigma = mag_nopi_list[max_nopi_ind-(pnts_to_fit-1)//2:max_nopi_ind+(pnts_to_fit-1)//2+1])
            except:
                is_fit_mag = False
        
        
        fig, ax = plt.subplots(2,2, sharex=True, figsize=[10,8])
        plt.sca(ax[0,0])
        plt.errorbar(freqs, I_pi_list, yerr = I_pi_err, fmt = pi_fmt, capsize = 5, markersize = 6, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1), label = r'$\pi$')
        plt.errorbar(freqs, I_nopi_list, yerr = I_nopi_err, fmt = nopi_fmt, capsize = 5, markersize = 6, ecolor = 'k', mfc=(0,0,1,0.5), mec = (0,0,0,1), label = r'$No-\pi$')
        ylim = plt.ylim()
        if is_plot_configured_freq and configured_freq is not None: plt.plot([configured_freq]*2, ylim, 'k--', alpha = 0.5, label = 'Configured frequency')
        plt.plot([freqs[I_diff.argmax()]]*2, ylim, '--', color='orange', alpha = 0.5, label = 'Best separation')
        plt.ylim(ylim)
        plt.legend(fontsize=6)
        plt.ylabel(f'I [{prefix}V]')
        plt.sca(ax[0,1])
        plt.errorbar(freqs, Q_pi_list, yerr = Q_pi_err, fmt = pi_fmt, capsize = 5, markersize = 6, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1), label = r'$\pi$')
        plt.errorbar(freqs, Q_nopi_list, yerr = Q_nopi_err, fmt = nopi_fmt, capsize = 5, markersize = 6, ecolor = 'k', mfc=(0,0,1,0.5), mec = (0,0,0,1), label = r'$No-\pi$')
        ylim = plt.ylim()
        if is_plot_configured_freq and configured_freq is not None: plt.plot([configured_freq]*2, ylim, 'k--', alpha = 0.5, label = 'Configured frequency')
        plt.plot([freqs[Q_diff.argmax()]]*2, ylim, '--', color='orange', alpha = 0.5, label = 'Best separation')
        plt.ylim(ylim)
        plt.ylabel(f'Q [{prefix}V]')
        plt.legend(fontsize=6)
        plt.sca(ax[1,0])
        plt.errorbar(freqs, mag_pi_list, yerr = mag_pi_err, fmt = pi_fmt, capsize = 5, markersize = 6, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1), label = r'$\pi$')
        plt.errorbar(freqs, mag_nopi_list, yerr = mag_nopi_err, fmt = nopi_fmt, capsize = 5, markersize = 6, ecolor = 'k', mfc=(0,0,1,0.5), mec = (0,0,0,1), label = r'$No-\pi$')
        if is_fit_mag: 
            freqs_to_plot_fit_pi = np.linspace(freqs_to_fit_pi[0],freqs_to_fit_pi[-1], 1001)
            freqs_to_plot_fit_nopi = np.linspace(freqs_to_fit_nopi[0],freqs_to_fit_nopi[-1], 1001)
            plt.plot(freqs_to_plot_fit_pi, fit_func(freqs_to_plot_fit_pi, *fit_pi), ':r', alpha=0.75, lw = 3)
            plt.plot(freqs_to_plot_fit_nopi, fit_func(freqs_to_plot_fit_nopi, *fit_nopi), ':b', alpha=0.75, lw = 3)
        ylim = plt.ylim()
        if is_plot_configured_freq and configured_freq is not None: plt.plot([configured_freq]*2, ylim, 'k--', alpha = 0.5, label = 'Configured frequency')
        plt.plot([freqs[mag_diff.argmax()]]*2, ylim, '--', color='orange', alpha = 0.5, label = 'Best separation')
        plt.ylabel(f'Mag [{prefix}V]')
        plt.xlabel(f'Freq [{freqs_prefix}Hz]')
        plt.ylim(ylim)
        plt.legend(fontsize=6)
        plt.sca(ax[1,1])
        plt.errorbar(freqs, phase_pi_list, yerr = phase_pi_err, fmt = pi_fmt, capsize = 5, markersize = 6, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1), label = r'$\pi$')
        plt.errorbar(freqs, phase_nopi_list, yerr = phase_nopi_err, fmt = nopi_fmt, capsize = 5, markersize = 6, ecolor = 'k', mfc=(0,0,1,0.5), mec = (0,0,0,1), label = r'$No-\pi$')
        ylim = plt.ylim()
        if is_plot_configured_freq and configured_freq is not None: plt.plot([configured_freq]*2, ylim, 'k--', alpha = 0.5, label = 'Configured frequency')
        plt.plot([freqs[phase_diff.argmax()]]*2, ylim, '--', color='orange', alpha = 0.5, label = 'Best separation')
        plt.ylabel('Phase')
        plt.xlabel(f'Freq [{freqs_prefix}Hz]')
        plt.ylim(ylim)
        plt.legend(fontsize=6)
        
        
        if is_fit_mag: 
            chi, chi_err= round_value_by_error((fit_pi[1] - fit_nopi[1]), np.sqrt(cov_pi[1,1]+cov_nopi[1,1]))
            
            print(f'chi = {chi}'+r'+-'f'{chi_err} MHz')
            return chi , chi_err
            
#%% number splitting spectroscopy

    def load_number_split_spec(self, N_avg = 1000, npts = 101,
                               drive_element = None, drive_pulse = None, wait_duration = 500,
                         freq_center = None, span = None, start = None, stop = None,
                         pi_pulse = 'pi_pulse',
                         qubit = None, is_ef = False,
                         ro_element = None,
                         is_active_reset = False,
                         is_ramp = False,
                         is_steady_state = False,
                         rabi_pulse = 'rabi_cooling',
                         ro_pulse = 'ro_pulse',
                         **kwargs):
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        freq0 = self.element_IF(qubit)
        if freq_center is None and span is None and start is not None and stop is not None:
            freqs = freq0 + np.linspace(start, stop, npts)
        elif freq_center is not None and span is not None and start is None and stop is None:
            freqs = freq0 + np.linspace(freq_center - span/2, freq_center + span/2, npts)
        else:
            raise ValueError("You must pass either freq_center and span or start and stop.\n Passed: freq_center={freq_center}, span = {span}, start = {start}, stop ={stop}")
            
        if drive_element is None: drive_element = ro_element
        if drive_pulse is None: drive_pulse = 'mixer_cal_pulse'
        
        wait_duration = int(wait_duration // 4)
        drive_duration = wait_duration + self.pulse_len(qubit, pi_pulse)   // 4
        
        self.results['number_split_spec'] = {'freqs': freqs,
                                          'N_avg': N_avg,
                                          'npts': npts,
                                          'wait duration': wait_duration,
                                          'qubit': qubit,
                                          'ro_element': ro_element}
        
        if is_active_reset:  run_time = N_avg*npts*(self.pulse_len(ro_element, ro_pulse)+self.pulse_len(qubit, pi_pulse)+self.wait_after_reset)
        else: run_time = N_avg*npts*(self.pulse_len(ro_element, ro_pulse)+self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))
        
        with program() as self.number_split_spec_prog:
            
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            freq = declare(int)
                
            with for_(n, 0, n < N_avg, n + 1):
                with for_each_(freq , freqs.astype(int).tolist()): #IF_list.astype(np.int64).tolist()
                    
                    if is_steady_state:
                        update_frequency(qubit,freq)
                        align(qubit, ro_element)
                        play(rabi_pulse, qubit, duration = self.pulse_len(ro_element, ro_pulse)//4)
                        self.perform_full_measurement(I, Q,  I_output_name = 'I', Q_output_name = 'Q', ro_element = ro_element, is_active_reset = False, readout_pulse = 'ro_pulse', is_wait = False)
                        
                    else:
                        if is_ef: 
                            update_frequency(qubit, freq0)
                            self._pi_pulse(qubit = qubit)
                        update_frequency(qubit,freq)
                        align(qubit, ro_element)             
                        play(drive_pulse, drive_element, duration = drive_duration)
                        wait(wait_duration, qubit)
                        if is_ramp:
                            play('ramp_up', qubit)
                            play(pi_pulse, qubit)
                            play('ramp_down', qubit)
                        else:
                            play(pi_pulse, qubit)
                        align(qubit, ro_element)             
                        if is_active_reset: update_frequency(qubit,freq0)
                        self.perform_full_measurement(I, Q,  I_output_name = 'I', Q_output_name = 'Q', ro_element = ro_element, is_active_reset = is_active_reset)
        self.last_prog = self.number_split_spec_prog
        
    def run_number_split_spec(self, is_save_data = None, **kwargs):

        if not hasattr(self, 'number_split_spec_prog'): raise ValueError("Idiot! You did not write spec_number_splitting_")
        results_dict = self.results['number_split_spec']
        results_dict["I"], results_dict["Q"] = self.run_prog(self.number_split_spec_prog, results_dict["N_avg"], **kwargs )      
        
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'sideband_detuning_sweep')
        return self.plot_number_split_spec(**kwargs)


    def plot_number_split_spec(self,prog_name = 'number split spec', fig_num = None, is_calc_stat_error = None,
                               chi_guess = 0, alpha_guess = 0, n_cut = 2, **kwargs):
        
        results_dict = self.results['number_split_spec']
        if fig_num is  None: fig_num = next_fig_num_by_name(prog_name)
        
        is_calc_stat_error = is_calc_stat_error if is_calc_stat_error is not None else self.is_calc_stat_error
        plt.figure(fig_num)
        data, error = self.process_data(data = [results_dict["I"], results_dict["Q"]], is_calc_stat_error = is_calc_stat_error)
        data, data_prefix, factor = self.autoscale_data(data)
        error = error*factor
        scaled_freqs, freq_prefix, _ = self.autoscale_data(results_dict["freqs"]-int(self.element_IF(results_dict["qubit"])))
        plt.errorbar(scaled_freqs, data, yerr = error, label = 'Measured Data', fmt = '--or', capsize = 5, markersize = 6, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1),)
            
        # ann = plt.annotate(r'drive cavity pulse amp = '+self.drive_pulse+'\n'\
        #                     +f'readout amp = {(1e-6*fit_results[0]):.4f}'+r'$ \pm$'+f'{(1e-6*np.sqrt(cov[0,0])):.4f}'+'\n'\,
        #                     [amp_list_fitted[0],freqs.max()], fontsize = 12)
        # ann.draggable()
        
        if self.which_data == 'Phase':
            ylabel = f'{self.which_data} [{data_prefix}Rad]'
        else:
            ylabel = f'{self.which_data} [{data_prefix}V]'
        plt.xlabel("frequencey [{}Hz]".format(freq_prefix))
        plt.ylabel(ylabel)
        
        
        def poisson(n,alpha):
            nbar = np.abs(alpha)**2
            return nbar**(n)*np.exp(-nbar)/np.math.factorial(n)
        
        def normalized_lorentzian(x,freq,width):
            return 1/((x-freq)**2+(width/2)**2)
        
        class poisson_lorentzians_cls():
            def __init__(self, n_cut = 2):
                self.n_cut = n_cut

            def poisson_lorentzians(self, x, amp, freq0, width, chi, alpha, offset):
                res = offset
                for n in range(self.n_cut):
                    res+= poisson(n,alpha) * normalized_lorentzian(x, freq0 + n * chi, width) * amp
                return res
            
        def _fit_number_splitting(x, y, chi_guess = 0, alpha_guess = 0, n_cut = 2):
            #check if negative lorenzian
            if np.abs(y.mean() - y.min()) > np.abs(y.mean() - y.max()): 
                width = np.abs(x[np.argmin(np.abs(y-(max(y)+min(y))/2))]-x[np.argmax(y)])
                amp = (np.min(y)-np.max(y))*width**2
                freq = x[np.argmin(y)]
                offset = np.max(y)
            else:
                width = np.abs(x[np.argmin(np.abs(y-(max(y)+min(y))/2))]-x[np.argmax(y)])
                amp = (np.max(y)-np.min(y))*width**2
                freq = x[np.argmax(y)]
                offset = np.min(y)
                
            guess = [amp, freq, width, chi_guess, alpha_guess, offset]
            fit_class_inst = poisson_lorentzians_cls(n_cut)
            fit, cov = curve_fit(fit_class_inst.poisson_lorentzians, x, y, guess)
            errors = np.sqrt(np.diag(cov))
            return fit, errors, fit_class_inst.poisson_lorentzians
        
        fit, errors, func = _fit_number_splitting(results_dict["freqs"], data, chi_guess = chi_guess, alpha_guess = alpha_guess, n_cut = n_cut)
        plt.plot(scaled_freqs, func(results_dict["freqs"], *fit))        
        ann = plt.annotate(r'$\chi = {}+-{}$ MHz'.format(*round_value_by_error(fit[3]/1e6, errors[3]/1e6))+'\n'+r' $\alpha = {}+-{}$'.format(*round_value_by_error(fit[4], errors[4])),
                     xy = [0.5,0.5],
                     fontsize = 12)
        ann.draggable()
        plt.title(prog_name + f' ({results_dict["qubit"]})')
        plt.tight_layout()
        
        
        
    def load_sideband_detuning_sweep(self, N_avg = 1000, 
                                     npts = 101, detuning_start = 1e3, detuning_stop = -1e3,
                                     drive_element = 'ro', drive_pulse = 'constant_sideband_cooling_pulse',
                                     drive_element2 = None, drive_pulse2 = 'constant_sideband_cooling_pulse',
                                     ring_down_time = 1000,
                                     rabi_pulse = 'rabi_pulse',
                                     is_ramp = True, is_ramp_drive = True,
                                     qubit = None, ro_element = None,
                                     is_meas_x = True,
                                     is_active_reset = None,
                                     is_wait = False,
                                     stark_shift = 0,
                                     **kwargs):
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        relative_detuning_list = np.linspace(detuning_start, detuning_stop, npts)
        detuning_list = relative_detuning_list + self.element_IF(drive_element)
        IF0 = self.element_IF(drive_element)
        if drive_element2 is not None: 
            detuning_list2 = relative_detuning_list + self.element_IF(drive_element2)
            IF0_2 = self.element_IF(drive_element)
        else:  
            detuning_list2 = detuning_list
            IF0_2 = IF0
        
        ramp_up_length = self.pulse_ramp_up_len(qubit, rabi_pulse)
        ramp_down_length = self.pulse_ramp_down_len(qubit, rabi_pulse)
        
        self.results['sideband_detuning_sweep'] = {'qubit': qubit, 'drive_element': drive_element, 'drive_pulse': drive_pulse,
                                                  'detuning': relative_detuning_list, 'N_avg': N_avg, 'is_meas_x': is_meas_x}
        
        if is_active_reset:
            run_time = N_avg * npts * 2 * (self.pulse_len(drive_element, drive_pulse) + self.pulse_len(ro_element, 'ro_pulse'))
        else:
            run_time = N_avg * npts * 2 * (self.pulse_len(drive_element, drive_pulse) + self.pulse_len(ro_element, 'ro_pulse') + self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))
        
        if is_wait:
            wait_time = int((self.pulse_len(drive_element, drive_pulse)-self.pulse_ramp_len(drive_element, drive_pulse) - self.pulse_len(qubit, rabi_pulse))//4)
            if is_ramp: wait_time -= (ramp_down_length + ramp_up_length)//4
            if is_meas_x: wait_time -= self.pulse_len(qubit, 'pi2_pulse')//4
        
        with program() as self.sideband_detuning_sweep_prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            detuning = declare(int)
            detuning2 = declare(int)
            phase = declare(fixed)
            
            if stark_shift != 0: update_frequency(qubit, self.element_IF(qubit) + stark_shift)
            with for_(n,0,n<N_avg,n+1):
                with for_each_((detuning, detuning2), (detuning_list.astype(int), detuning_list2.astype(int))):
                    with for_(phase, 0, phase<0.6, phase+0.5):
                        update_frequency(drive_element, detuning)
                        if drive_element2 is not None: update_frequency(drive_element2, detuning)
                        align(drive_element, qubit)
                        reset_frame(drive_element)
                        if is_ramp_drive:  
                            play(drive_pulse + '_ramp_up', drive_element)
                        play(drive_pulse, drive_element)
                        if is_ramp_drive:  
                            play(drive_pulse + '_ramp_down', drive_element)
                            
                        if drive_element2 is not None:
                            if is_ramp_drive:  
                                play(drive_pulse2 + '_ramp_up', drive_element2)
                            play(drive_pulse2, drive_element2)
                            if is_ramp_drive:  
                                play(drive_pulse2 + '_ramp_down', drive_element2)
                
                        if is_wait: wait(wait_time, qubit)
                    
                        if is_ramp :
                            play(rabi_pulse + '_ramp_up', qubit)
                        play(rabi_pulse, qubit)
                        if is_ramp :
                            play(rabi_pulse + '_ramp_down', qubit)
                        if is_meas_x:
                            frame_rotation_2pi(0.25+phase, qubit) # sigma x measurement
                            play(self.pi2_pulse, qubit)
                            reset_frame(qubit)
                        else:
                            play('pi_pulse', qubit, condition = phase==0)
                        
                        if ring_down_time>0:
                            wait(int(ring_down_time//4), drive_element)
                            if drive_element2 is not None: wait(int(ring_down_time//4), drive_element2)
                                        
                        if drive_element!=ro_element:
                            align(drive_element, ro_element)
            
                        update_frequency(drive_element, IF0)
                        self.perform_full_measurement(I,Q, ro_element = ro_element, is_active_reset = is_active_reset, **kwargs)
        
        self.last_prog = self.sideband_detuning_sweep_prog
            
    def run_sideband_detuning_sweep(self, is_save_data = None, **kwargs):
        results_dict = self.results['sideband_detuning_sweep']
        
        npts = len(results_dict['detuning'])
        N_avg = results_dict['N_avg']
        
        results_dict['I'], results_dict['Q'] = self.run_prog(self.sideband_detuning_sweep_prog, shape = (N_avg, npts,2), **kwargs)
        

        
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'sideband_detuning_sweep')
        return self.plot_sideband_detuning_sweep(**kwargs)

    def plot_sideband_detuning_sweep(self, **kwargs):
        
        results_dict = self.results['sideband_detuning_sweep']
        I = results_dict['I']
        Q = results_dict['Q']
        data, error = self.process_data([I,Q])
        data, units_prefix, factor = self.autoscale_data(data)
        
        ylabel = self.which_data
        if self.which_data in ['Phase', 'phase']:
            ylabel += ' [Rad]'
        else:
            ylabel += ' [V]'
        
        detuning = results_dict['detuning']
        data = (data[:,0]-data[:,1])/2
        error = factor*np.sqrt(error[:,0]**2+error[:,1]**2)/2
        
        plt.figure()
        plt.errorbar(x = detuning * 1e-6, y = data, yerr = error, fmt = '.b', capsize = 4, markersize=4)
        plt.ylabel(f'{self.which_data} [{units_prefix} V]')
        plt.xlabel(r'Sideband detuning [MHz]')
        plt.tight_layout()



    def load_sideband_rabi_amp_sweep(self, N_avg = 1000, 
                                     npts = 101, amp_scale_start = 0.9, amp_scale_stop = 1.1,
                                     drive_element = 'ro', drive_pulse = 'constant_sideband_cooling_pulse',
                                     drive_detuning = 0,
                                     drive_element2 = None, drive_pulse2 = 'constant_sideband_cooling_pulse',
                                     drive_detuning2 = 0,
                                     ring_down_time = 1000,
                                     rabi_pulse = 'rabi_pulse',
                                     is_ramp = True, is_ramp_drive = True,
                                     qubit = None, ro_element = None,
                                     is_meas_x = True,
                                     is_active_reset = None,
                                     is_wait = False,
                                     stark_shift = 0,
                                     **kwargs):
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        amp_scale_list = np.linspace(amp_scale_start, amp_scale_stop, npts)
        amp_list = amp_scale_list * self.pulse_amp(qubit, rabi_pulse)
        
        ramp_up_length = self.pulse_ramp_up_len(qubit, rabi_pulse)
        ramp_down_length = self.pulse_ramp_down_len(qubit, rabi_pulse)
        
        self.results['sideband_rabi_amp_sweep'] = {'qubit': qubit, 'drive_element': drive_element, 'drive_pulse': drive_pulse,
                                                  'amp_list': amp_list, 'N_avg': N_avg, 'is_meas_x': is_meas_x}
        
        if is_active_reset:
            run_time = N_avg * npts * 2 * (self.pulse_len(drive_element, drive_pulse) + self.pulse_len(ro_element, 'ro_pulse'))
        else:
            run_time = N_avg * npts * 2 * (self.pulse_len(drive_element, drive_pulse) + self.pulse_len(ro_element, 'ro_pulse') + self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))
        
        if is_wait:
            wait_time = int((self.pulse_len(drive_element, drive_pulse)-self.pulse_ramp_len(drive_element, drive_pulse) - self.pulse_len(qubit, rabi_pulse))//4)
            if is_ramp: wait_time -= (ramp_down_length + ramp_up_length)//4
            if is_meas_x: wait_time -= self.pulse_len(qubit, 'pi2_pulse')//4
        
        with program() as self.sideband_rabi_amp_sweep_prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            amp_scale = declare(fixed)
            phase = declare(fixed)
            
            with for_(n,0,n<N_avg,n+1):
                with for_each_(amp_scale, amp_scale_list):
                    with for_(phase, 0, phase<0.6, phase+0.5):
                        
                        
                        # if stark_shift != 0: update_frequency(qubit, self.element_IF(qubit) + stark_shift, keep_phase = True)
                        # update_frequency(drive_element, self.element_IF(drive_element) + drive_detuning, keep_phase = True)
                        # if drive_element2 is not None: update_frequency(drive_element2, self.element_IF(drive_element2) + drive_detuning2, keep_phase = True)
                        # reset_frame(qubit)
                        
                        # align(drive_element, qubit)
                        # reset_frame(drive_element)
                        # if drive_element2 is not None: 
                        #     reset_frame(drive_element2)
                        #     align(drive_element2, qubit)
                        # if is_ramp_drive:  
                        #     play(drive_pulse + '_ramp_up', drive_element)
                        # play(drive_pulse, drive_element)
                        # if is_ramp_drive:  
                        #     play(drive_pulse + '_ramp_down', drive_element)
                            
                        # if drive_element2 is not None:
                        #     if is_ramp_drive:  
                        #         play(drive_pulse2 + '_ramp_up', drive_element2)
                        #     play(drive_pulse2, drive_element2)
                        #     if is_ramp_drive:  
                        #         play(drive_pulse2 + '_ramp_down', drive_element2)
                
                        # if is_wait: wait(wait_time, qubit)
                    
                        # if is_ramp :
                        #     play((rabi_pulse + '_ramp_up' ) * amp(amp_scale), qubit)
                        # play(rabi_pulse * amp(amp_scale), qubit)
                        # if is_ramp :
                        #     play((rabi_pulse + '_ramp_down')  * amp(amp_scale), qubit)
                        # if is_meas_x:
                        #     frame_rotation_2pi(0.25+phase, qubit) # sigma x measurement
                        #     play(self.pi2_pulse, qubit)
                        #     reset_frame(qubit)
                        # else:
                        #     play('pi_pulse', qubit, condition = phase==0.5)
                        
                        # if ring_down_time>0:
                        #     wait(int(ring_down_time//4), drive_element)
                        #     if drive_element2 is not None: wait(int(ring_down_time//4), drive_element2)
                                        
                        # if drive_element!=ro_element:
                        #     align(drive_element, ro_element)
            
                        # update_frequency(drive_element, self.element_IF(drive_element))
                        # if drive_element2 is not None: update_frequency(drive_element2, self.element_IF(drive_element2))
                        # if stark_shift != 0: update_frequency(qubit, self.element_IF(qubit))
                        # if ro_element != drive_element: align(qubit, drive_element, ro_element)
                        # self.perform_full_measurement(I,Q, ro_element = ro_element, is_active_reset = is_active_reset, **kwargs)
                        qb_detuning = stark_shift
                        ro_detuning = drive_detuning
                        mm_detuning = drive_detuning2
                        ro_element = drive_element
                        mm = drive_element2
                        is_align_all = False
                        sb_duration = None
                        sideband_drive = drive_pulse
                        mm_sideband_cooling_drive = drive_pulse2
                        sb_wait_time = ring_down_time
                        sb_steady_time = 0
                        rabi_sideband_cooling_drive = rabi_pulse
                        if ro_detuning!=0: update_frequency(ro_element, self.element_IF(ro_element) + ro_detuning, keep_phase = True)
                        if qb_detuning!=0: update_frequency(qubit, self.element_IF(qubit) + qb_detuning, keep_phase = True)
                        if mm_detuning!=0 and mm is not None: update_frequency(mm, self.element_IF(mm) + mm_detuning, keep_phase = True)
                        
                        pi2_phase = 0.25 if ro_detuning > 0 else -0.25
                        
                        if is_align_all: 
                            if mm is not None: align(qubit, mm, ro_element)
                            else: align(qubit, ro_element)
                        if sb_duration is None:
                            if is_ramp: play(sideband_drive+"_ramp_up", ro_element)
                            play(sideband_drive, ro_element)
                            if is_ramp: play(sideband_drive+"_ramp_down", ro_element)
                            for qb in [qubit]:
                                if sb_steady_time>0:
                                    wait(sb_steady_time//4, qb)
                                if is_ramp: play((rabi_sideband_cooling_drive + "_ramp_up")* amp(amp_scale), qubit)
                                play(rabi_sideband_cooling_drive* amp(amp_scale), qb)
                                if is_ramp: play((rabi_sideband_cooling_drive + "_ramp_down")* amp(amp_scale), qubit)
                                if mm is not None: 
                                    if sb_steady_time//4>0:
                                        wait(sb_steady_time//4, mm)
                                    if is_ramp: play(mm_sideband_cooling_drive + "_ramp_up", mm)
                                    play(mm_sideband_cooling_drive, mm)
                                    if is_ramp: play(mm_sideband_cooling_drive + "_ramp_down", mm)
                        if is_meas_x:
                            frame_rotation_2pi(0.25+phase, qubit) # sigma x measurement
                            play(self.pi2_pulse, qubit)
                            reset_frame(qubit)
                        # if is_pi2:
                        #     frame_rotation_2pi(pi2_phase, qb)
                        #     for qb in qubit:
                        #         if type(pi2_amp_scale) is float or type(pi2_amp_scale) is int:
                        #             if pi2_amp_scale == 1.0:
                        #                 play('pi2_pulse', qb)
                        #             else:
                        #                 play('pi2_pulse' * amp(pi2_amp_scale), qb)
                        #         else:
                        #             play('pi2_pulse' * amp(pi2_amp_scale), qb)
                                
                        if ro_detuning!=0: update_frequency(ro_element, self.element_IF(ro_element), keep_phase = True)
                        if qb_detuning!=0: update_frequency(qubit, self.element_IF(qubit), keep_phase = True)
                        if mm_detuning!=0 and mm is not None: update_frequency(mm, self.element_IF(mm), keep_phase = True)
                        
                        align(qubit, ro_element)
                        
                        if sb_wait_time//4  > 0 : 
                            wait(sb_wait_time//4, ro_element)
                            align(qubit, ro_element)
                        self.perform_full_measurement(I,Q, ro_element = ro_element, is_active_reset = is_active_reset, **kwargs)
                        
        self.last_prog = self.sideband_rabi_amp_sweep_prog
            
    def run_sideband_rabi_amp_sweep(self, is_save_data = None, **kwargs):
        results_dict = self.results['sideband_rabi_amp_sweep']
        
        npts = len(results_dict['amp_list'])
        N_avg = results_dict['N_avg']
        
        results_dict['I'], results_dict['Q'] = self.run_prog(self.sideband_rabi_amp_sweep_prog, shape = (N_avg, npts,2), **kwargs)
        
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'sideband_rabi_amp_sweep')
        return self.plot_sideband_rabi_amp_sweep(**kwargs)
    
    def plot_sideband_rabi_amp_sweep(self, **kwargs):
        
        results_dict = self.results['sideband_rabi_amp_sweep']
        I = results_dict['I']
        Q = results_dict['Q']
        data, error = self.process_data([I,Q])
        data, units_prefix, factor = self.autoscale_data(data)
        
        ylabel = self.which_data
        if self.which_data in ['Phase', 'phase']:
            ylabel += ' [Rad]'
        else:
            ylabel += ' [V]'
        
        amps_list = results_dict['amp_list']
        data = (data[:,0]-data[:,1])/2
        # data = (data[:,0]+data[:,1])/2
        error = factor*np.sqrt(error[:,0]**2+error[:,1]**2)/2
        
        plt.figure()
        plt.errorbar(x = amps_list, y = data, yerr = error, fmt = '.b', capsize = 4, markersize=4)
        plt.ylabel(f'{self.which_data} [{units_prefix} V]')
        plt.xlabel(r'Rabi amp [V]')
        plt.tight_layout()
        
        
        
        
    def _load_py_loop_driven_ramsey_prog(self, times_to_wait1, times_to_wait2, cos_list, sin_list):
        results_dict = self.results['driven ramsey']
        N_avg = results_dict['N_avg']
        qubit = results_dict['qubit']
        drive_element = results_dict['drive_element']
        drive_detuning = results_dict['drive_detuning']
        drive_pulse = results_dict['drive_pulse']
        ro_element = results_dict['ro_element']
        
        drive_element2 = results_dict['drive_element2']
        drive_detuning2 = results_dict['drive_detuning2']
        drive_pulse2 = results_dict['drive_pulse2']
        
        is_ramp = results_dict['is_ramp']
        is_active_reset = results_dict['is_active_reset']
        is_sb_cool = results_dict['is_sb_cool']
        ring_down_time = results_dict['ring_down_time']
        
        with program() as self.driven_ramsey_prog:
        
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            ti1 = declare(int)
            ti2 = declare(int)
            phase_off = declare(fixed)
            sin = declare(fixed)
            cos = declare(fixed)
            with for_(n, 0, n < N_avg, n + 1): 
                with for_each_((ti1, ti2, cos, sin), (times_to_wait1//4, times_to_wait2//4, cos_list, sin_list)):
                    reset_frame(qubit)
                    reset_frame(drive_element)
                    if drive_detuning != 0 : update_frequency(drive_element, self.element_IF(drive_element)+drive_detuning)
                    if drive_detuning2 != 0 and drive_element2 is not None: update_frequency(drive_element2, self.element_IF(drive_element2)+drive_detuning2)
                    # align(drive_element, qubit, ro_element)
                    if is_ramp:
                        play(drive_pulse + '_ramp_up', drive_element)
                        if drive_element2 is not None: play(drive_pulse2 + '_ramp_up', drive_element2)
                    play(drive_pulse, drive_element)
                    if drive_element2 is not None: play(drive_pulse2, drive_element2)
                    if is_ramp:
                        play(drive_pulse + '_ramp_down', drive_element)
                        if drive_element2 is not None: play(drive_pulse2 + '_ramp_down', drive_element2)
                    if drive_detuning != 0 : update_frequency(drive_element, self.element_IF(drive_element), keep_phase = True)
                    if drive_detuning2 != 0 and drive_element2 is not None: update_frequency(drive_element2, self.element_IF(drive_element2), keep_phase = True)
                    wait(ti1, qubit)
                    play('pi2_pulse', qubit)
                    wait(ti2, qubit)
                    play('pi2_pulse' * amp(cos, -sin, sin, cos), qubit)
                    
                    align(qubit, ro_element)
                    if ring_down_time>0: wait(ring_down_time//4, ro_element)
                    self.perform_full_measurement(I,Q, ro_element = ro_element, is_sb_cool = is_sb_cool, is_active_reset = is_active_reset)


    def load_driven_power_rabi(self,
                npts = 51, # How many points I measure
                N_avg = 1000,
                amp_scale_start = 1,
                amp_scale_stop = 1,
                time = 1000,
                qubit = None,
                ro_element = None,
                drive_element = None, drive_detuning = 0,
                drive_pulse = None,
                drive_element2 = None, drive_detuning2 = 0,
                drive_pulse2 = None,
                rabi_pulse = 'rabi_pulse',
                is_sb_cool = False,
                is_active_reset = None,
                is_ramp = True,
                is_ramp_drive = True,
                is_meas_x = False,
                init_state = 'g',
                ring_down_time = 0,
                extra_drive_time = 100,
                stark_shift_corr = 0,
                is_tomography = False,
                prepare = None,
                prepare_kwargs = {},
                **kwargs):
        
        print(f'\nStark shift correction is {np.round(stark_shift_corr*1e-6,3)} MHz \n')
        if type(stark_shift_corr) is not list:
            stark_shift_corr=[stark_shift_corr]
        if qubit is None: qubit = self.main_qubit
        if type(qubit) is not list:
            qubit = [qubit]
        if ro_element is None: ro_element = self.main_readout
        if drive_element is None: drive_element = self.main_readout
        if drive_pulse is None: drive_pulse = 'mixer_cal_pulse'
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        if is_tomography: tomo = OneQubitTomo(qubit[0])
        ramp_up_length = self.configObject.SuperMembers[qubit[0]].PulseParamsDict[rabi_pulse].Additions.ramp['up']['length']
        ramp_down_length = self.configObject.SuperMembers[qubit[0]].PulseParamsDict[rabi_pulse].Additions.ramp['down']['length']
        
        
        amp_scale_list = np.linspace(amp_scale_start, amp_scale_stop, npts)
        
        self.results['driven power rabi'] = {'time': time,
                                    'npts': npts,
                                    'N_avg': N_avg,
                                    'amp_scale_list': amp_scale_list,
                                     'qubit': qubit,
                                     'rabi_pulse': rabi_pulse,
                                     'ro_element': ro_element,
                                     'drive_element': drive_element,
                                     'drive_element2': drive_element2,
                                     'drive_pulse': drive_pulse,
                                     'drive_pulse2': drive_pulse2,
                                     'is_tomography': is_tomography}
        
        if is_sb_cool:
            run_time = N_avg*npts*(self.pulse_len(drive_element, drive_pulse) + self.pulse_len(ro_element,self.ro_pulse) + self.pulse_len(ro_element, 'constant_sideband_cooling_pulse'))
        elif is_active_reset:
            run_time = N_avg*npts*(self.pulse_len(ro_element,self.ro_pulse) + self.pulse_len(drive_element, drive_pulse))
        else:
            run_time = N_avg*npts*(self.pulse_len(ro_element,self.ro_pulse) + self.pulse_len(drive_element, drive_pulse) +self.wait_between_seq)
        if is_tomography: 
            run_time = run_time * len(tomo.pulse_seq_list)
            if is_active_reset: run_time += N_avg * self.pulse_len(ro_element,self.ro_pulse)
            else: run_time += N_avg * (self.pulse_len(ro_element,self.ro_pulse)+self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))        
        
        base_wait_time = 0
        if is_ramp_drive: base_wait_time += self.pulse_ramp_len(drive_element, drive_pulse)//4
        
        sideband_time = time
        if is_ramp: sideband_time += ramp_up_length + ramp_down_length
        if is_meas_x or is_tomography: sideband_time += self.pulse_len(qubit[0], 'pi2_pulse')//4
        if init_state != 'g': sideband_time += self.pulse_len(qubit[0], 'pi2_pulse')//4
        if init_state == 'plus': init_phase = 0.25
        if init_state == 'minus': init_phase = -0.25
    
    
        def rabi_sequence(I, Q, amp_scale, is_tomography, x = None):
            with for_each_(amp_scale, amp_scale_list):
                
                if prepare is not None:
                    prepare(**prepare_kwargs)
                    
                for qb,sshift in zip(qubit,stark_shift_corr):
                    # if stark_shift_corr != 0: update_frequency(qb, self.element_IF(qb)+sshift, keep_phase = True)
                    if stark_shift_corr != 0: update_frequency(qb, new_IF, keep_phase = True)
                    
                if drive_detuning != 0:
                    update_frequency(drive_element, self.element_IF(drive_element)+drive_detuning, keep_phase = True)
                if drive_detuning2 != 0 and drive_element2 is not None:
                    update_frequency(drive_element2, self.element_IF(drive_element2)+drive_detuning2, keep_phase = True)
                reset_frame(drive_element)
                if drive_element2 is not None: 
                    reset_frame(drive_element2)
                for qb in qubit:
                    reset_frame(qb)
                for qb in qubit:
                    align(drive_element, qb)
                    
                    if is_ramp_drive:  
                        play(drive_pulse + '_ramp_up', drive_element)
                        if drive_element2 is not None: play(drive_pulse2 + '_ramp_up', drive_element2)
                    play(drive_pulse, drive_element, duration = sideband_time//4)
                    if drive_element2 is not None: play(drive_pulse2, drive_element2, duration = sideband_time//4)
                    if is_ramp_drive:  
                        play(drive_pulse + '_ramp_down', drive_element)
                        if drive_element2 is not None: play(drive_pulse2 + '_ramp_down', drive_element2)
                if drive_detuning != 0:
                    update_frequency(drive_element, self.element_IF(drive_element), keep_phase = True)
                if drive_detuning2 != 0 and drive_element2 is not None:
                    update_frequency(drive_element2, self.element_IF(drive_element2), keep_phase = True)
                
                for qb in qubit:
                    wait(base_wait_time, qb)
                    
                    if init_state != 'g':
                        if init_state in ['plus', 'minus']: 
                            play('pi2_pulse', qubit[0])
                            frame_rotation_2pi(init_phase, qb)
                        elif init_state == 'e': play('pi_pulse', qb)
                    if is_ramp :
                        play(((rabi_pulse + '_ramp_up') * amp(amp_scale)), qb)
                    play(rabi_pulse * amp(amp_scale), qb, duration = time//4)
                    if is_ramp :
                        play(((rabi_pulse + '_ramp_down') * amp(amp_scale)), qb)
                    if is_meas_x and not is_tomography:
                        frame_rotation_2pi(0.25, qb) # sigma x measurement
                        play(self.pi2_pulse, qb)
                        reset_frame(qb)
                
                if ring_down_time>0:
                    wait(int(ring_down_time//4), drive_element)
                    if drive_element2 is not None: wait(int(ring_down_time//4), drive_element2)
                    align(qubit[0], drive_element)
                    
                
                if is_tomography:
                    with switch_(x):
                        for i in range(len(tomo.pulse_seq_list)):
                            with case_(i):
                                tomo.play_tomo_pulse(tomo.pulse_seq_list[i], self.pulse_len(qubit[0], self.pi2_pulse))
                align(qubit[0], ro_element)    
                if is_active_reset:
                    for qb in qubit:
                        # if stark_shift_corr != 0: update_frequency(qb, self.element_IF(qb), keep_phase = True)
                        if stark_shift_corr != 0: update_frequency(qb, old_IF, keep_phase = True)
                
                if drive_element!=ro_element:
                    align(drive_element, ro_element)
                if drive_element2 is not None and drive_element2!=ro_element:
                    align(drive_element2, ro_element)
                    
                align(qubit[0], ro_element)
                if is_tomography:
                    I_out_name = 'I_tomo'
                    Q_out_name = 'Q_tomo'
                else:
                    I_out_name = 'I'
                    Q_out_name = 'Q'
                self.perform_full_measurement(I,Q, I_output_name = I_out_name, Q_output_name = Q_out_name, ro_element = ro_element, is_sb_cool = is_sb_cool, is_active_reset = is_active_reset, **kwargs)
                
        with program() as self.driven_power_rabi_prog:
            if is_tomography: tomo.declarations()
            else:
                I = declare(fixed)
                Q = declare(fixed)
            for qb,sshift in zip(qubit,stark_shift_corr):
                if stark_shift_corr != 0:
                    new_IF = declare(int, value = int(self.element_IF(qb)+sshift))
                    old_IF = declare(int, value = int(self.element_IF(qb)))
            pulse_duration = declare(int)
            n = declare(int)
            amp_scale = declare(fixed)
            with for_(n, 0, n < N_avg, n + 1):
                if is_tomography:
                    for qb in qubit:
                        # if stark_shift_corr != 0: update_frequency(qb, self.element_IF(qb), keep_phase = True)
                        if stark_shift_corr != 0: update_frequency(qb, old_IF, keep_phase = True)
                    tomo.measure_beta_coeffs(ro_element, qubit, meas_func = self.perform_full_measurement, is_active_reset=is_active_reset, is_sb_cool = is_sb_cool, **kwargs)
                    with for_(tomo.x, 0, tomo.x<len(tomo.pulse_seq_list), tomo.x+1):
                        rabi_sequence(tomo.I, tomo.Q, amp_scale, is_tomography, tomo.x)
                else:
                    rabi_sequence(I, Q, amp_scale, is_tomography = False)
                        
            self.last_prog = self.driven_power_rabi_prog
            
        
    def run_driven_power_rabi(self, is_continue = False, is_save_data = None,  is_plot_all = True, **kwargs): 
    
        results_dict = self.results['driven power rabi']
        if results_dict['is_tomography']: 
            return self.run_one_qubit_tomography(self.driven_rabi_prog, len(results_dict['t']), results_dict = results_dict, **kwargs)
        
        amp_scale_list = results_dict['amp_scale_list']
        amp_npts = len(amp_scale_list)
        N_avg = results_dict['N_avg']
        results_dict['pulse_amp'] = self.pulse_amp(results_dict['qubit'][0], results_dict['rabi_pulse'])
        
        results_dict['I'], results_dict['Q'] = self.run_prog(self.driven_power_rabi_prog, shape = (N_avg, amp_npts), **kwargs)
        
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'driven power rabi')
        return self.plot_driven_power_rabi(**kwargs)
    
    def plot_driven_power_rabi(self, which_data = None, **kwargs):
        if which_data is None: which_data = self.which_data
        results_dict = self.results['driven power rabi']
        amp_scale_list = results_dict['amp_scale_list']
        amps = amp_scale_list * results_dict['pulse_amp']
        
        data, err = self.process_data(results_dict)\
        
        data, data_units_prefix, data_factor = self.autoscale_data(data)
        err = err * data_factor
        
        prog_name = 'Driven Power Rabi'
        plt.figure(next_fig_num_by_name(prog_name))    
        plt.errorbar(amps, data, err, fmt = '-o')
        
        plt.xlabel('Rabi amp. [V]')
        plt.ylabel(which_data + f'[{data_units_prefix}V]')
        plt.tight_layout()
            

        
    def load_ro_decay(self, npts = 50,
                    N_avg = 10000,
                    ro_element = None,
                    ro_pulse = None,
                    meas_pulse = None,
                    wait_time = 0,
                    **kwargs):
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        if meas_pulse is None: meas_pulse = self.ro_pulse
        
        def is_valid(A,C):
            return A % (4 * C) == 0
        def closest_divisor(A, B):
            # Search outward from B
            for offset in range(0, A):  # A is a safe upper bound
                for sign in (-1, 1):
                    C = B + sign * offset
                    if C > 0 and is_valid(A,C):
                        return C
            raise ValueError("Can't find npts such that chunk size is whole")
        if not is_valid(self.pulse_len(ro_element, ro_pulse), npts):
            npts = closest_divisor(self.pulse_len(ro_element, ro_pulse), npts)
            print(f'Changed npts to {npts} so chunk size is a whole number.')
        chunk_size =  self.pulse_len(ro_element, ro_pulse) // npts // 4
        
        self.results['ro_decay'] = {'N_avg': N_avg,
                               'npts': npts,
                               'ro_pulse': ro_pulse,
                               'ro_element': ro_element}
        results_dict = self.results['ro_decay']
        results_dict['chunk_size'] = chunk_size

        run_time = 2*N_avg*(self.pulse_len(ro_element,ro_pulse)+self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))            
        
        with program() as self.ro_decay_prog:
            n = declare(int)
                
            I = declare(fixed, size = npts)
            Q = declare(fixed, size = npts)
            i = declare(int)
                
            with for_(n, 0, n < N_avg, n + 1):
                play(ro_pulse, ro_element)
                if wait_time>=16: wait(int(wait_time//4), ro_element)
                self.perform_sliced_measurement(I, Q, i, chunk_size = chunk_size, npts = npts, ro_element = ro_element,  readout_pulse = ro_pulse,
                                                is_sb_cool = False,  is_ramp_up = False, amp_scale = 0, **kwargs)
        self.last_prog = self.ro_decay_prog  
                
    def run_ro_decay(self, is_save_data = None,
                     ro_pulse = None,
                   **kwargs): 
        results_dict = self.results['ro_decay']
        
        if is_save_data is None: is_save_data = self.is_save_data
        
        if not hasattr(self, 'ro_decay_prog'): raise ValueError("Idiot! You did not write pinopi program")
        if ro_pulse is None: ro_pulse = results_dict['ro_pulse']
        
        results_dict['I'], results_dict['Q'] = self.run_prog(self.ro_decay_prog, shape = (results_dict['N_avg'], results_dict['npts']), **kwargs)
        
        results_dict['time'] = np.linspace(results_dict['chunk_size'] * 4, results_dict['chunk_size'] * 4 * results_dict['npts'] , results_dict['npts'])
        
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'ro_decay')
        
        self.plot_ro_decay(**kwargs)

    def plot_ro_decay(self, **kwargs):
        results_dict = self.results['ro_decay']
        t = results_dict['time']
        
        data, err = self.process_data(results_dict, is_mean = True, is_calc_stat_error = True, which_data = 'Mag')
        res, fit_err, fig = self.fit_and_plot('Exp', [data, err], t, )
        plt.ylabel('Readout Amplitude [V]')
        print('kappa/2pi = {}+-{} MHz'.format(*round_value_by_error(1/res[0]/2/np.pi, fit_err[0]/res[0]**2/2/np.pi)))


    def load_ckp(self, N_avg = 1000, 
                 ro_detuning_start = -5e3, ro_detuning_stop = 5e3, ro_detuning_npts = 101,
                 qb_detuning_start = -5e3, qb_detuning_stop = 5e3, qb_detuning_npts = 101,
                 qubit = None, ro_element = None,
                 steady_time = 2e3,  ringdown_time = 800,
                 is_ramp = True, pi_pulse = None, drive_pulse = 'mixer_cal_pulse',
                 is_active_reset = None,
                 **kwargs):
        "from https://arxiv.org/pdf/2402.00413"
        
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if pi_pulse is None: pi_pulse=self.pi_pulse
        
        if is_active_reset is None: is_active_reset = self.is_active_reset
        
        ro_detuning = np.linspace(ro_detuning_start, ro_detuning_stop, ro_detuning_npts)
        qb_detuning = np.linspace(qb_detuning_start, qb_detuning_stop, qb_detuning_npts)
        
        ro_f0 = self.element_IF(ro_element)
        qb_f0 = self.element_IF(qubit)
        
        drive_duration = int(self.pulse_len(qubit, pi_pulse) + steady_time)
        
        self.results['ckp'] = {'ro_detuning': ro_detuning,
                               'qb_detuning': qb_detuning,
                               'N_avg': N_avg,
                               'steady_time': steady_time,
                               'qubit': qubit,
                               'ro_f0': self.element_freq(ro_element),
                               'drive_pulse': drive_pulse,
                               'ro_element': ro_element}
        results_dict = self.results['ckp']
        
        if is_active_reset: run_time = N_avg*ro_detuning_npts*qb_detuning_npts * (steady_time+self.pulse_len(ro_element, self.ro_pulse))
        else: run_time = N_avg*ro_detuning_npts*qb_detuning_npts * (steady_time+self.pulse_len(ro_element, self.ro_pulse) + self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))     
        
        with program() as self.ckp_prog:
        
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            f_qb = declare(int)
            f_ro = declare(int)
            
            with for_(n, 0, n < N_avg, n + 1): 
                with for_each_(f_ro, (ro_f0+ro_detuning).astype(int).tolist()): 
                    with for_each_(f_qb, (qb_f0+qb_detuning).astype(int).tolist()): 
                        update_frequency(ro_element, f_ro)
                        update_frequency(qubit, f_qb)
                        if is_ramp and self.configObject.SuperMembers[ro_element].PulseParamsDict[drive_pulse].Additions.ramp['up'] is not None:
                            play(self.configObject.SuperMembers[ro_element].PulseParamsDict[drive_pulse].Additions.ramp['up']['name'], ro_element)
                        play(drive_pulse, ro_element, duration = drive_duration//4)
                        if is_ramp and self.configObject.SuperMembers[ro_element].PulseParamsDict[drive_pulse].Additions.ramp['down'] is not None:
                            play(self.configObject.SuperMembers[ro_element].PulseParamsDict[drive_pulse].Additions.ramp['down']['name'], ro_element)
                        wait(int(steady_time//4), qubit)
                        play(pi_pulse, qubit)
                        update_frequency(ro_element, ro_f0)
                        update_frequency(qubit, qb_f0)
                        if ringdown_time >=16: wait(ringdown_time//4, ro_element)
                        self.perform_full_measurement(I,Q, ro_element = ro_element, is_active_reset = is_active_reset, **kwargs)
                        
                        align(qubit, ro_element)
                        play(pi_pulse, qubit)
                        align(qubit, ro_element)
                        update_frequency(ro_element, f_ro)
                        update_frequency(qubit, f_qb)
                        if is_ramp and self.configObject.SuperMembers[ro_element].PulseParamsDict[drive_pulse].Additions.ramp['up'] is not None:
                            play(self.configObject.SuperMembers[ro_element].PulseParamsDict[drive_pulse].Additions.ramp['up']['name'], ro_element)
                        play(drive_pulse, ro_element, duration = drive_duration//4)
                        if is_ramp and self.configObject.SuperMembers[ro_element].PulseParamsDict[drive_pulse].Additions.ramp['down'] is not None:
                            play(self.configObject.SuperMembers[ro_element].PulseParamsDict[drive_pulse].Additions.ramp['down']['name'], ro_element)
                        wait(int(steady_time//4), qubit)
                        play(pi_pulse, qubit)
                        update_frequency(ro_element, ro_f0)
                        update_frequency(qubit, qb_f0)
                        if ringdown_time >=16: wait(ringdown_time//4, ro_element)
                        align(qubit, ro_element)
                        play(pi_pulse, qubit)
                        self.perform_full_measurement(I,Q, ro_element = ro_element, is_active_reset = is_active_reset, **kwargs)
                        
                    
        self.last_prog = self.ckp_prog
        
    def run_ckp(self, is_continue = False, is_save_data = None, **kwargs): 
    
        if not hasattr(self, 'ckp_prog'): raise ValueError("Idiot! You did not write ramsey program")
        
        results_dict = self.results['ckp']
        prog_name = 'ckp'
        results_dict['I'], results_dict['Q'] = self.run_prog(self.ckp_prog, 
                                                             shape = (results_dict['N_avg'], len(results_dict['ro_detuning']),len(results_dict['ro_detuning']), 2),
                                                             **kwargs)
        results_dict['pulse_amp'] = self.pulse_amp(results_dict['ro_element'], results_dict['drive_pulse'])
        fig_num = next_fig_num_by_name(prog_name)
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(results_dict, 'ckp')
        return self.plot_ckp(fig_num = fig_num, prog_name = prog_name, **kwargs)

    def plot_ckp(self, which_data = None, **kwargs):
        if which_data == None: which_data = self.which_data
        results_dict = self.results['ckp']
        
        data, err = self.process_data(results_dict, which_data = which_data)
        ro_detuning = results_dict['ro_detuning']
        qb_detuning = results_dict['qb_detuning']
        
        fitted_freqs = np.zeros((len(results_dict['ro_detuning']), 2))
        fitted_freqs_err = np.zeros((len(results_dict['ro_detuning']), 2))
        
        print('\n\nStart fitting\n\n')
        for i,f in enumerate(ro_detuning):
            sft = non_TimeDomain_fit('Lorentzian', data[i,:,0], qb_detuning*1e-6, err[i,:,0])
            fit,cov = sft.fit_params, sft.fit_cov
            if sft.is_succeed:
                fitted_freqs[i,0] = fit[3]
                fitted_freqs_err[i,0] = np.sqrt(cov[3,3])
            else:
                fitted_freqs[i,0] = np.nan
                fitted_freqs_err[i,0] = np.nan
            
            sft = non_TimeDomain_fit('Lorentzian', data[i,:,1], qb_detuning*1e-6, err[i,:,0])
            fit,cov = sft.fit_params, sft.fit_cov
            if sft.is_succeed:
                fitted_freqs[i,1] = fit[3]
                fitted_freqs_err[i,1] = np.sqrt(cov[3,3])
            else:
                fitted_freqs[i,1] = np.nan
                fitted_freqs_err[i,1] = np.nan
                
        inds_g = np.isfinite(fitted_freqs[:,0])
        inds_e = np.isfinite(fitted_freqs[:,1])
        ro_detuning_fitted_g = ro_detuning[inds_g]
        fitted_freqs_g = fitted_freqs[inds_g, 0]
        ro_detuning_fitted_e = ro_detuning[inds_e]
        fitted_freqs_e = fitted_freqs[inds_e, 1]
        fitted_freqs_err_e = fitted_freqs[inds_e, 1]
            
        plot_2D(data[:,:,0].transpose(), ro_detuning*1e-6, qb_detuning*1e-6, 
                xlabel = r'$\delta_\mathrm{ro}$ [MHz]', ylabel = r'$\delta_\mathrm{qb}$ [MHz]',
                cmap = 'Reds', is_colorbar = False)
        plt.text(ro_detuning.min(),qb_detuning.max(),r'$|g\rangle$', ha = 'left', va = 'top')
        plt.plot(ro_detuning_fitted_g*1e-6, fitted_freqs_g, 'b-')
        
        
        plot_2D(data[:,:,1].transpose(), ro_detuning*1e-6, qb_detuning*1e-6, 
                xlabel = r'$\delta_\mathrm{ro}$ [MHz]', ylabel = r'$\delta_\mathrm{qb}$ [MHz]',
                cmap = 'Reds', is_colorbar = False)
        plt.text(ro_detuning.min(),qb_detuning.max(),r'$|e\rangle$', ha = 'left', va = 'top')
        plt.plot(ro_detuning_fitted_e*1e-6, fitted_freqs_e, 'b-')
        
        plt.figure()
        if fitted_freqs_err.max() > np.abs(fitted_freqs).max()*10:
            plt.plot(ro_detuning*1e-6, fitted_freqs[:,0], 'o')
            plt.plot(ro_detuning*1e-6, fitted_freqs[:,1], 'o')
        else:
            plt.errorbar(ro_detuning*1e-6, fitted_freqs[:,0], fitted_freqs_err[:,0], fmt = 'ob')
            plt.errorbar(ro_detuning*1e-6, fitted_freqs[:,1], fitted_freqs_err[:,1], fmt = 'or')
        
        sft_g = non_TimeDomain_fit('Lorentzian', fitted_freqs_g, ro_detuning*1e-6)
        fit_g, cov_g = sft_g.fit_params, sft_g.fit_cov
        if sft_g.is_succeed: plt.plot(ro_detuning*1e-6, sft.func(ro_detuning*1e-6, *fit_g), '-b', label = r'$|g\rangle$')
        sft_e = non_TimeDomain_fit('Lorentzian', fitted_freqs_e, ro_detuning*1e-6)
        fit_e, cov_e = sft_e.fit_params, sft_e.fit_cov
        if sft_e.is_succeed: plt.plot(ro_detuning*1e-6, sft.func(ro_detuning*1e-6, *fit_e), '-r', label = r'$|e\rangle$')
        plt.legend()
        plt.xlabel( r'$\delta_\mathrm{ro}$ [MHz]')
        plt.ylabel( r'$\delta_\mathrm{qb}$ [MHz]')
        plt.tight_layout()
        
        if sft_e.is_succeed and sft_g.is_succeed:
            chi = fit_e[3] - fit_g[3]
            chi_err = np.sqrt(cov_g[3,3] + cov_e[3,3])
            f_ro = (fit_e[3] + fit_g[3])/2
            f_ro_err = np.sqrt(cov_g[3,3] + cov_e[3,3])/2
            kappa = (2*fit_e[1] + 2*fit_g[1])/2
            kappa_err = np.sqrt(cov_g[1,1] + cov_e[1,1])
            # depth = chi * 4 a**2
            a_mean = (fit_g[0] + fit_e[0])/2
            a_mean_err = np.sqrt(cov_g[0,0] + cov_e[0,0])/2
            
            a = np.sqrt(a_mean)/np.sqrt(chi)*kappa/2/results_dict['pulse_amp']
            
            a_err = a*np.sqrt((a_mean_err/a_mean/2)**2 + (chi_err/chi/2)**2 + (kappa_err/kappa)**2) / 2/results_dict['pulse_amp']
        
            print('omega_ro/2pi = {} +- {} MHz'.format(*round_value_by_error(results_dict['ro_f0']*1e-6 + f_ro, f_ro_err)))
            print('chi/2pi = {} +- {} MHz'.format(*round_value_by_error(chi, chi_err)))
            print('kappa/2pi = {} +- {} MHz'.format(*round_value_by_error(kappa, kappa_err)))
            print('a/2pi = {} +- {} MHz/V'.format(*round_value_by_error(a, a_err)))
        else: print('Fit failed')
