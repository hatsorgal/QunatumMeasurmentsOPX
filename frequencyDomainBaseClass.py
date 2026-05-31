# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:36:41 2022

@author: Eliya
"""
import pathlib
import sys
import os
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent /'usefulFunctions'))
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\measurementClass') # for fridge 2 pc
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\measurementClass') # for fridge 1 pc
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\InstrumentControl') # for fridge 2 pc
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\InstrumentControl') # for fridge 1 pc
import pickle
from MXG5183A import MXG5183A
from AgilentPNA import AgilentPNA
from B2962A import B2962A
from CXA_SA import CXA_SA
from general_functions import in_hz, in_s, in_dbm
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Cambria']
from labrad.units import V,mV, s,ms,us,ns, Hz,kHz,MHz,GHz, Ohm, dBm, Value
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta, date
from scipy import fftpack
from scipy.optimize import curve_fit
import scipy as sp
from tqdm import tqdm
import json
import requests
from data_processing import scale_data_units, round_value_by_error
from plotting import plot_2D, next_fig_num_by_name, last_fig_num_by_name
# from qkit.analysis.circle_fit.circle_fit_classic import circuit
from scipy import signal, interpolate, optimize
import math as math
from qkit.analysis.circle_fit.circle_fit_classic import circuit
import re


#%%
def sleep_in_steps(sleep_time, step = 1):
    t=0
    while t+step < sleep_time:
        sleep(step)
        t+=step
    sleep(sleep_time-t)
    
def normalized_lorentzian(x,freq,width):
    return 1/((x-freq)**2+(width/2)**2)

def lorentzian(x,amp,freq,width,offset):
    return amp/((x-freq)**2+(width/2)**2) + offset

def hanging_res(x,freq,kappa_l,kappa_i,offset):
    """ function describing log10(|S21|) of hanging resonator (no assymetry), with offset """
    return offset + np.sqrt(1 - (kappa_l**2-kappa_i**2)/4/((x-freq)**2 + (kappa_l/2)**2))

def poisson(n,alpha):
    nbar = np.abs(alpha)**2
    return nbar**(n)*np.exp(-nbar)/np.math.factorial(n)
        
class poisson_lorentzians_cls():
    def __init__(self, n_cut = 2):
        self.n_cut = n_cut

    def poisson_lorentzians(self, x, amp, freq0, width, chi, alpha, offset):
        res = offset
        for n in range(self.n_cut):
            res+= poisson(n,alpha) * normalized_lorentzian(x, freq0 + n * chi, width) * amp
        return res

def reflection_S11_delay(f, k_in, k_ext, f_c, phase_off, amp, delay_distance):
    c = 299792458 * 1e-9
    return  amp * np.exp(1j*(phase_off/180*np.pi + delay_distance/c*f *2*np.pi)) * ((k_in - k_ext)*np.pi + 1j * ((f-f_c)*2*np.pi) )  / ( (k_in+k_ext) * np.pi + 1j * (f - f_c) * 2 * np.pi)

def reflection_S11(f, k_in, k_ext, f_c, phase_off, amp):
    c = 299792458 * 1e-9
    return  amp * np.exp(1j*phase_off/180*np.pi) * ((k_in - k_ext)*np.pi + 1j * ((f-f_c)*2*np.pi) )  / ( (k_in+k_ext) * np.pi + 1j * (f - f_c) * 2 * np.pi)

def reflection_S11_flat(f, k_in, k_ext, f_c, phase_off, amp):
    S11 = reflection_S11(f, k_in, k_ext, f_c, phase_off, amp)
    return np.append(S11.real, S11.imag)
    
class FreqDo():
    
    def __init__(self,
                 mxg_name = None,
                 which_data = 'Both',
                 in_port = None,
                 out_port = None,
                 ps_channel = 1,
                 save_folder = None):
        self.which_data = which_data
        self.ps_channel = ps_channel
        self.get_instruments(mxg_name)
        self.save_folder = save_folder
        if save_folder is not None: self.is_save_data = True
        else: self.is_save_data = False
        self.temp_update_interval = 0
        self.last_temp_time = datetime.now()

        self.pna.s_parameters(1,in_port,out_port)
        self.in_port = in_port
        self.out_port = out_port
        self.results = {}
        self.exp_name = ''
        
    def check_s_parameters(self):
        s_params_current = self.pna.s_parameters()
        in_port = s_params_current[2]
        out_port = s_params_current[1]
        if in_port != self.in_port or out_port != self.out_port:
            self.pna.s_parameters(1,self.in_port,self.out_port)
            
    def get_instruments(self, mxg_name):
        self.pna = AgilentPNA()
        try:
            self.ps = B2962A(channel = self.ps_channel)
        except:
            print('\n\nDid not connect to power source\n\n')
        try:
            self.mxg = MXG5183A(mxg_name)
            self.is_mxg = True
        except:
            self.is_mxg = False
            print('\n\nDid not connect to mxg\n\n')
        try:
            self.cxa = CXA_SA()
        except:
            print('\n\nDid not connect to spectrum analyzer \n\n')
            
    def one_tone_flux_sweep(self,
                            freq_center,
                            freq_span,
                            freq_npts,
                            power,
                            navg,
                            if_bw,
                            curr_start,
                            curr_stop,
                            curr_npts,
                            is_plot_while_running = False,
                            plot_interval = 10,
                            which_data = None,
                            is_check_temp = False,
                            T_thresh = 20e-3,
                            is_save_data = None,
                            is_find_resonance = True,
                            ):
        
        if freq_npts > 100001: raise ValueError(f"max number of frequency points is 10001. got <{freq_npts}>")
        is_save_data = is_save_data if is_save_data is not None else self.is_save_data 
        if which_data is None: which_data = self.which_data
        self.check_s_parameters()
        self.set_pna_format(which_data = which_data)
        self.results['one_tone_flux_sweep'] = {}
        single_sweep_time = self.set_pna_sweep(freq_center, freq_span, freq_npts, power, if_bw, navg)
        self.pna.set_triger_source(source='IMM')
        
        if self.is_mxg: self.mxg.off()
        
        freq_list = np.linspace(in_hz(freq_center-freq_span/2), in_hz(freq_center+freq_span/2), freq_npts)
        curr_list = np.linspace(curr_start, curr_stop, curr_npts)
        logmag_array = np.zeros([freq_npts, curr_npts])
        phase_array = np.zeros([freq_npts, curr_npts])
        
        self.pna.powerOnOff(state='ON');
        self.ps.on()
        
        total_run_time = single_sweep_time * curr_npts
        print(f'Run time is {(total_run_time/60):.2f} minutes')
        
        if is_plot_while_running and which_data in ['LogMag', 'Logmag','logmag', 'Both']:
            fig, ax_logmag = plt.subplots()
        else:
            ax_logmag = None
        if is_plot_while_running and which_data in ['Phase','phase', 'Both']:
            fig, ax_phase = plt.subplots()
        else:
            ax_phase = None
        
        for i, curr in tqdm(enumerate(curr_list)):
            self.ps.set_curr_smooth(curr)
            self.pna.trigger_single()
            sleep_in_steps(single_sweep_time)
            self.pna.auto_scale()
            if is_check_temp: 
                self.check_heating(T_thresh = T_thresh)
            data = self.pna.getData(typ ='Scale')
            logmag_array[:,i] = 20*np.log10(np.abs(data))
            logmag_array[:,i+1:] = logmag_array[:,i+1:]*0 + np.min(logmag_array[:,:i+1])
            phase_array[:,i] = np.angle(data, deg=True)
            if is_plot_while_running and i%plot_interval==0 and i!=0:
                if which_data in ['LogMag', 'Logmag','logmag', 'Both']:
                    plot_2D(logmag_array, curr_list*1e3, freq_list*1e-9,
                                 ax=ax_logmag, is_colorbar = False,
                                 xlabel = 'Current [mA]', ylabel = 'Frequency [GHz]', zlabel = 'LogMag [dB]', cmap = 'Reds')
                plt.pause(0.01)
                if which_data in ['Phase','phase', 'Both']:
                    plot_2D(phase_array,  curr_list*1e3, freq_list*1e-9,
                                 ax=ax_phase, is_colorbar = False,
                                 xlabel = 'Current [mA]', ylabel = 'Frequency [GHz]', zlabel = r'Phase $^\circ$')
                    plt.pause(0.01)
        
            
        if is_check_temp: temperature = self.get_temperature()
        else: temperature = 0
        
        results_dict = self.results['one_tone_flux_sweep']
        results_dict['logmag'] = logmag_array
        results_dict['phase'] = phase_array
        results_dict['frequency'] = freq_list
        results_dict['current'] = curr_list
        results_dict['power'] = power
        results_dict['if_bw'] = if_bw
        results_dict['navg'] = navg
        results_dict['temperature'] = temperature
        
        if is_find_resonance:
            if which_data in ['LogMag', 'Logmag','logmag', 'both', 'Both']:
                res_freq_list = freq_list[np.argmin(logmag_array,0)]
                results_dict['resonance_frequency'] = res_freq_list
            else:
                res_freq_list = freq_list[np.argmin(np.abs(phase_array),0)]
                results_dict['resonance_frequency'] = res_freq_list
        
        self.plot_one_tone_flux_sweep(which_data = which_data)
        
        self.pickle_save(results_dict, 'one_tone_flux_sweep')
            
        
    def two_tone_flux_sweep(self,
                            freq_center,
                            freq_span,
                            freq_npts,
                            mxg_power,
                            pna_freq,
                            pna_power,
                            navg,
                            if_bw,
                            curr_start,
                            curr_stop,
                            curr_npts,
                            is_follow_resonance = False,
                            is_plot_while_running = False,
                            plot_interval = 10,
                            which_data = None,
                            is_check_temp = False,
                            T_thresh = 20e-3,
                            is_save_data = None
                            ):
        
        self.check_s_parameters()
        if freq_npts > 100001: raise ValueError(f"max number of frequency points is 10001. got <{freq_npts}>")
        is_save_data = is_save_data if is_save_data is not None else self.is_save_data 
        if which_data is None: which_data = self.which_data
        self.set_pna_format(which_data = which_data)
        self.set_pna_cw(freq = pna_freq, power = pna_power, if_bw = if_bw, npts = freq_npts, navg = navg)
        self.set_pna_ext_trigger(action = 'POIN')
        dwell_time = (1.2 /if_bw+4*ms)
        
        freq_list = np.linspace(in_hz(freq_center-freq_span/2), in_hz(freq_center+freq_span/2), freq_npts)
        curr_list = np.linspace(curr_start, curr_stop, curr_npts)
        logmag_array = np.zeros([freq_npts, curr_npts])
        phase_array = np.zeros([freq_npts, curr_npts])
        
        self.set_mxg_freq_sweep(dwell_time = dwell_time, start = freq_list[0]*Hz, stop = freq_list[-1]*Hz, npts = freq_npts, power = mxg_power)
        
        self.pna.powerOnOff(state='ON');
        self.mxg.on()
        self.ps.on()
        
        single_sweep_time = in_s((dwell_time+6*ms)*freq_npts*navg)
        
        total_run_time = single_sweep_time * curr_npts
        print(f'Run time is {(total_run_time/60):.2f} minutes')
        
        time_of_meas = datetime.now().strftime("%d_%m_%Y___%H_%M_%S")
        
        if is_plot_while_running and which_data in ['LogMag', 'Logmag','logmag', 'Both', 'both']:
            fig, ax_logmag = plt.subplots()
        else:
            ax_logmag = None
        if is_plot_while_running and which_data in ['Phase','phase', 'Both', 'both']:
            fig, ax_phase = plt.subplots()
        else:
            ax_phase = None
        
        if is_follow_resonance:
            self.get_resonance_vs_flux(which_data=which_data)
        for i, curr in tqdm(enumerate(curr_list)):
            if is_follow_resonance:
                self.pna.CwFreq(f = self.res_vs_flux(curr))
            self.ps.set_curr_smooth(curr)
            if is_check_temp: 
                self.check_heating(T_thresh = T_thresh)
            for n in range(navg):
                self.mxg.start_sweep('SING')
                sleep_in_steps(single_sweep_time+0.1)
                self.pna.auto_scale()
            data = self.pna.getData(typ ='Scale')
            logmag_array[:,i] = 20*np.log10(np.abs(data))
            logmag_array[:,i+1:] = logmag_array[:,i+1:]*0 + np.min(logmag_array[:,:i+1])
            # phase_array[:,i] = np.unwrap(np.angle(data))
            phase_array[:,i] = np.angle(data, deg=True)
            # phase_array[:,i+1:] = phase_array[:,i+1:] * 0 + np.mean(phase_array[:,:i])
            if is_plot_while_running and i%plot_interval==0 and i!=0:
                if which_data in ['LogMag', 'Logmag','logmag', 'Both', 'both']:
                    plot_2D(logmag_array, curr_list*1e3, freq_list*1e-9, 
                                 ax=ax_logmag, is_colorbar = False,
                                 xlabel = 'Current [mA]', ylabel = 'Frequency [GHz]', zlabel = 'LogMag [dB]', cmap = 'Reds')
                    plt.pause(0.1)
                if which_data in ['Phase','phase', 'Both', 'both']:
                    plot_2D(phase_array,  curr_list*1e3, freq_list*1e-9, 
                                 ax=ax_phase, is_colorbar = False,
                                 xlabel = 'Current [mA]', ylabel = 'Frequency [GHz]', zlabel = 'Phase $^\circ$')
                    plt.pause(0.1)
                
                self.results['two_tone_flux_sweep'] = {}
                results_dict = self.results['two_tone_flux_sweep']
                results_dict['logmag'] = logmag_array
                results_dict['phase'] = phase_array
                results_dict['frequency'] = freq_list
                results_dict['current'] = curr_list
                results_dict['mxg_power'] = mxg_power
                results_dict['pna_freq'] = pna_freq
                results_dict['pna_power'] = pna_power
                results_dict['if_bw'] = if_bw
                results_dict['navg'] = navg
                self.pickle_save(results_dict, meas_name = 'two_tone_flux_sweep', time_of_meas = time_of_meas, is_print = False)
        self.mxg.off()
            
        if is_check_temp: temperature = self.get_temperature()
        else: temperature = 0
        
        results_dict['logmag'] = logmag_array
        results_dict['phase'] = phase_array
        results_dict['frequency'] = freq_list
        results_dict['current'] = curr_list
        results_dict['mxg_power'] = mxg_power
        results_dict['pna_freq'] = pna_freq
        results_dict['pna_power'] = pna_power
        results_dict['if_bw'] = if_bw
        results_dict['navg'] = navg
        results_dict['temperature'] = temperature
        results_dict['is_follow_resonance'] = is_follow_resonance
        
        self.pickle_save(results_dict, meas_name = 'two_tone_flux_sweep', time_of_meas = time_of_meas)
        
        self.plot_two_tone_flux_sweep(which_data = which_data)
        
    def plot_one_tone_flux_sweep(self, ax = None, which_data = None):
        if which_data == None: which_data = self.which_data
        
        if which_data in ['Both', 'both']:
            if ax is not None:
                print("You passed axis and also which data is set to both. Using passed axis to plot phase.")
            self.plot_one_tone_flux_sweep(which_data = 'phase' , ax = ax)
            self.plot_one_tone_flux_sweep(which_data = 'logmag')
            
        results_dict = self.results['one_tone_flux_sweep']
        logmag_array = results_dict['logmag']
        phase_array = results_dict['phase']
        freq_list = results_dict['frequency']
        curr_list = results_dict['current']
        power = results_dict['power']
        if_bw = results_dict['if_bw']
        navg = results_dict['navg']
        
        # temperature = self.one_tone_flux_sweep_results['temperature']
        temperature = None
        # annot = r'${}mK$'.format(temperature*1e3) if temperature is not None else None
        
        if ax is None: _,ax = plt.subplots()
        if which_data in ['LogMag', 'Logmag','logmag']:
            plot_2D(logmag_array, curr_list*1e3, freq_list*1e-9, cmap = 'Reds', ax=ax,
                         xlabel = 'Current [mA]', ylabel = 'Frequency [GHz]', zlabel = 'LogMag [dB]', is_colorbar = True)
        elif which_data in ['Phase','phase']:
            plot_2D(phase_array,  curr_list*1e3, freq_list*1e-9, ax=ax,
                         xlabel = 'Current [mA]', ylabel = 'Frequency [GHz]', zlabel = 'Phase $^\circ$', is_colorbar = True)
        plt.sca(ax)
        txt = 'Power = {}\n IF-BW = {}'.format(power,if_bw)
        if navg != 1: txt += '\nNavg = {}'.format(navg)
        an = plt.annotate(txt, xy = (0.65, 0.05), xycoords = 'axes fraction', fontsize = 12)
        an.draggable()
            
        
    def plot_two_tone_flux_sweep(self, 
                                 ax = None, 
                                 which_data = None,
                                 vmax = None,
                                 vmin = None,
                                 is_normalize_all = False,
                                 freq_start = None,
                                 freq_stop = None,
                                 curr_start = None,
                                 curr_stop = None):
        which_data = self.which_data if which_data is None else which_data
        
        if which_data in ['Both', 'both']:
            if ax is not None:
                print("You passed axis and also which data is set to both. Using passed axis to plot phase.")
            self.plot_two_tone_flux_sweep(which_data = 'phase' , ax = ax)
            self.plot_two_tone_flux_sweep(which_data = 'logmag')
            
        results_dict = self.results['two_tone_flux_sweep']
        logmag_array = results_dict['logmag'][freq_start:freq_stop,curr_start:curr_stop]
        phase_array = results_dict['phase'][freq_start:freq_stop,curr_start:curr_stop]
        freq_list = results_dict['frequency'][freq_start:freq_stop]
        curr_list = results_dict['current'][curr_start:curr_stop]
        mxg_power = results_dict['mxg_power']
        pna_power = results_dict['pna_power']
        pna_freq = results_dict['pna_freq']
        if_bw = results_dict['if_bw']
        navg = results_dict['navg']
        if 'is_follow_resonance' in results_dict.keys(): is_follow_resonance = results_dict['is_follow_resonance']
        else: is_follow_resonance = True
        
        if is_normalize_all:
            phase_array = phase_array.copy()
            logmag_array = logmag_array.copy()
            for i,phase in enumerate(phase_array.transpose()):
                phase_array[:,i] = 2*(phase_array[:,i] - (phase.max()+phase.min()) / 2) / (phase.max()-phase.min())
            for i,logmag in enumerate(logmag_array.transpose()):
                logmag_array[:,i] = 2*(logmag_array[:,i] - (logmag.max()+logmag.min()) / 2) / (logmag.max()-logmag.min())
        
        # temperature = results_dict['temperature']
        temperature = None
        # annot = r'${}mK$'.format(temperature*1e3) if temperature is not None else None
        
        if ax is None: _,ax = plt.subplots()
        if which_data in ['LogMag', 'Logmag','logmag']:
            plot_2D(logmag_array, curr_list*1e3, freq_list*1e-9, 
                         ax=ax, cmap = 'Reds',
                         xlabel = 'Current [mA]', ylabel = 'Frequency [GHz]', zlabel = 'LogMag [dB]',
                         vmax = vmax, vmin = vmin)
        if which_data in ['Phase','phase']:
            plot_2D(phase_array,  curr_list*1e3, freq_list*1e-9, 
                     ax=ax, 
                     xlabel = 'Current [mA]', ylabel = 'Frequency [GHz]', zlabel = 'Phase $^\circ$',
                     vmax = vmax, vmin = vmin)
            
        plt.sca(ax)
        if is_follow_resonance:
            txt = 'MXG Power = {}\n PNA Power = {}\n IF-BW = {}'.format(mxg_power,pna_power,if_bw)
        else:
            txt = 'MXG Power = {}\n PNA Power = {}\n PNA Freq = {}\n IF-BW = {}'.format(mxg_power,pna_power,pna_freq,if_bw)
        if navg != 1: txt += '\n Navg = {}'.format(navg)
        an = plt.annotate(txt, xy = (0.65, 0.05), xycoords = 'axes fraction', fontsize = 12)
        an.draggable()
                
    def set_mxg_freq_sweep(self,
                           dwell_time, 
                           start,
                           stop,
                           power,
                           npts,
                           which_data = None
                           ):
        self.mxg.atten_prot('off')
        self.mxg.power(power)
        self.mxg.sweep_mode('FREQ')
        self.mxg.set_freq_start(start)    
        self.mxg.set_freq_stop(stop)   
        self.mxg.set_dwell(dwell_time)
        self.mxg.Num_SWE_PT(npts)
        self.mxg.triger_source('IMM') #set SG trigering to IMM
        self.mxg.EXT_trig_source('TRIG2')
        self.mxg.set_sweep_type('STEP')
        
    def one_tone_sweep(self,
                       center,
                       span,
                       npts,
                       power,
                       if_bw,
                       navg,
                       which_data = None,
                       ax = None,
                       is_save_data = None,
                       meas_name = 'one_tone_sweep',
                       is_get_temp = False,
                       is_plot = True,
                       phase_offset = None,
                       electrical_delay = None,
                       is_get_data_only = False
                       ):
        
        if npts > 100001: raise ValueError(f"max number of points is 10001. got <{npts}>")
        is_save_data = is_save_data if is_save_data is not None else self.is_save_data 
        if which_data is None: which_data = self.which_data
        if phase_offset is None:
            phase_offset = self.pna.get_phase_offset()
        if electrical_delay is None:
            electrical_delay = self.pna.electrical_delay()['s']
        
        if not is_get_data_only:
            self.check_s_parameters()
            self.set_pna_format(which_data = which_data)
            self.pna.powerOnOff('ON')
            self.pna.set_triger_source(source='IMM')
            sweep_time = self.set_pna_sweep(center,span,npts,power,if_bw,navg)
            for n in range(navg):
                self.pna.trigger_single()
                sleep_in_steps(sweep_time)
        freq_list = np.linspace(in_hz(center-span/2), in_hz(center+span/2), npts)
        self.pna.auto_scale()
        data = self.pna.getData(typ = 'Scale')
        
        self.one_tone_sweep_results={}
        self.one_tone_sweep_results['logmag'] = 20*np.log10(np.abs(data))
        self.one_tone_sweep_results['phase'] = np.angle(data, deg=True)
        self.one_tone_sweep_results['frequency'] = freq_list
        self.one_tone_sweep_results['power'] = power
        self.one_tone_sweep_results['if_bw'] = if_bw
        self.one_tone_sweep_results['navg'] = navg
        self.one_tone_sweep_results['phase offset'] = phase_offset
        self.one_tone_sweep_results['electrical delay'] = electrical_delay
        
        
        if is_get_temp:
            temp = self.get_temperature()
            self.one_tone_sweep_results['temp'] = temp
        
        if is_save_data:
            self.pickle_save(self.one_tone_sweep_results, meas_name = meas_name)
        
        if is_plot:
            self.plot_one_tone_sweep(which_data = which_data, ax = ax)
    
           
    def get_resonance_vs_flux(self,
                              which_data = None):
        
        if which_data is None: which_data = self.which_data
        results_dict = self.results['one_tone_flux_sweep']
        current = results_dict['current']
        phase_array = results_dict['phase']
        logmag_array = results_dict['logmag']
        freq_list = results_dict['frequency']
            
        if which_data in ['LogMag', 'Logmag','logmag', 'both', 'Both']:
            res_freq_list = freq_list[np.argmin(logmag_array,0)]
        else:
            res_freq_list = freq_list[np.argmin(np.abs(phase_array),0)]
        if current[1]-current[0]<0:
            current = np.flip(current.copy())
            res_freq_list = np.flip(res_freq_list.copy())
        self.res_vs_flux = sp.interpolate.UnivariateSpline(current, res_freq_list)
        return self.res_vs_flux, current
    
    def one_tone_power_sweep(self,
                            freq_center,
                            freq_span,
                            freq_npts,
                            power_start,
                            power_stop,
                            power_npts,
                            if_bw,
                            navg,
                            is_plot_while_running = False,
                            plot_interval = 10,
                            which_data = None,
                            ax = None,
                            is_save_data = None):
        
        if freq_npts > 100001: raise ValueError(f"max number of frequency points is 10001. got <{freq_npts}>")
        is_save_data = is_save_data if is_save_data is not None else self.is_save_data 
        if which_data is None: which_data = self.which_data
        self.check_s_parameters()
        self.set_pna_format(which_data = which_data)
        self.results['one_tone_power_sweep_results'] = {}
        results_dict = self.results['one_tone_power_sweep_results']
        self.pna.set_triger_source(source='IMM')
        single_sweep_time = self.set_pna_sweep(freq_center, freq_span, freq_npts, power_start, if_bw, navg)
        
        
        freq_list = np.linspace(in_hz(freq_center-freq_span/2), in_hz(freq_center+freq_span/2), freq_npts)
        power_list = np.linspace(power_start['dBm'], power_stop['dBm'], power_npts)
        logmag_array = np.zeros([freq_npts, power_npts])
        phase_array = np.zeros([freq_npts, power_npts])
        
        self.pna.powerOnOff(state='ON');
        
        total_run_time = single_sweep_time * power_npts
        print(f'Run time is {(total_run_time/60):.2f} minutes')
        
        
        if is_plot_while_running and which_data in ['LogMag', 'Logmag','logmag', 'Both']:
            fig, ax_logmag = plt.subplots()
        else:
            ax_logmag = None
        if is_plot_while_running and which_data in ['Phase','phase', 'Both']:
            fig, ax_phase = plt.subplots()
        else:
            ax_phase = None
        
        for i, power in tqdm(enumerate(power_list)):
            self.pna.power(p = power *dBm)
            self.pna.trigger_single()
            sleep_in_steps(single_sweep_time)
            self.pna.auto_scale()
            data = self.pna.getData(typ ='Scale')
            logmag_array[:,i] = 20*np.log10(np.abs(data))
            phase_array[:,i] = np.angle(data, deg=True)
            if is_plot_while_running and i%plot_interval==0 and i!=0:
                if which_data in ['LogMag', 'Logmag','logmag', 'Both']:
                    plot_2D(logmag_array, power_list, freq_list*1e-9,
                                 ax=ax_logmag, is_colorbar = False,
                                 xlabel = 'Power [dBm]', ylabel = 'Frequency [GHz]', zlabel = 'LogMag [dB]', cmap = 'Reds')
                plt.pause(0.1)
                if which_data in ['Phase','phase', 'Both']:
                    plot_2D(phase_array,  power_list, freq_list*1e-9,
                                 ax=ax_phase, is_colorbar = False,
                                 xlabel = 'Power [dBm]', ylabel = 'Frequency [GHz]', zlabel = 'Phase $^\circ$')
                plt.pause(0.1)
            
        results_dict['logmag'] = logmag_array
        results_dict['phase'] = phase_array
        results_dict['frequency'] = freq_list
        results_dict['power'] = power_list
        results_dict['if_bw'] = if_bw
        results_dict['navg'] = navg
        
        self.pickle_save(results_dict, 'one_tone_power_sweep')
        
        self.plot_one_tone_power_sweep(which_data = which_data)
        
    def one_tone_time_sweep(self,
                            freq_center,
                            freq_span,
                            freq_npts,
                            ntimes,
                            power,
                            if_bw,
                            is_plot_while_running = False,
                            plot_interval = 10,
                            which_data = None,
                            is_save_data = None):
        
        if freq_npts > 100001: raise ValueError(f"max number of frequency points is 10001. got <{freq_npts}>")
        is_save_data = is_save_data if is_save_data is not None else self.is_save_data 
        if which_data is None: which_data = self.which_data
        self.check_s_parameters()
        self.set_pna_format(which_data = which_data)
        self.one_tone_time_sweep_results = {}
        self.pna.set_triger_source(source='IMM')
        single_sweep_time = self.set_pna_sweep(freq_center, freq_span, freq_npts, power, if_bw, 1)
        
        
        freq_list = np.linspace(in_hz(freq_center-freq_span/2), in_hz(freq_center+freq_span/2), freq_npts)
        time_list = np.linspace(0,ntimes * (1/in_hz(if_bw)*freq_npts), ntimes)
        logmag_array = np.zeros([freq_npts, ntimes])
        phase_array = np.zeros([freq_npts, ntimes])
        self.pna.powerOnOff(state='ON');
        
        total_run_time = single_sweep_time * ntimes
        print(f'Run time is {(total_run_time/60):.2f} minutes')
        
        
        if is_plot_while_running and which_data in ['LogMag', 'Logmag','logmag', 'Both']:
            fig, ax_logmag = plt.subplots()
        else:
            ax_logmag = None
        if is_plot_while_running and which_data in ['Phase','phase', 'Both']:
            fig, ax_phase = plt.subplots()
        else:
            ax_phase = None
        
        for i in range(ntimes):
            self.pna.trigger_single()
            sleep_in_steps(single_sweep_time)
            self.pna.auto_scale()
            data = self.pna.getData(typ ='Scale')
            logmag_array[:,i] = 20*np.log10(np.abs(data))
            phase_array[:,i] = np.angle(data, deg=True)
            if is_plot_while_running and i%plot_interval==0 and i!=0:
                if which_data in ['LogMag', 'Logmag','logmag', 'Both']:
                    plot_2D(logmag_array, time_list, freq_list*1e-9,
                                 ax=ax_logmag, is_colorbar = False,
                                 xlabel = 'Time [s]', ylabel = 'Frequency [GHz]', zlabel = 'LogMag [dB]', cmap = 'Reds')
                plt.pause(0.1)
                if which_data in ['Phase','phase', 'Both']:
                    plot_2D(phase_array,  time_list, freq_list*1e-9,
                                 ax=ax_phase, is_colorbar = False,
                                 xlabel = 'Time [s]', ylabel = 'Frequency [GHz]', zlabel = 'Phase $^\circ$')
                plt.pause(0.1)
            
        self.one_tone_time_sweep_results['logmag'] = logmag_array
        self.one_tone_time_sweep_results['phase'] = phase_array
        self.one_tone_time_sweep_results['frequency'] = freq_list
        self.one_tone_time_sweep_results['if_bw'] = if_bw
        self.one_tone_time_sweep_results['time_list'] = time_list
        
        self.pickle_save(self.one_tone_time_sweep_results, 'one_tone_time_sweep')
        
        self.plot_one_tone_time_sweep(which_data = which_data)
        
    def plot_one_tone_time_sweep(self, which_data = None):
        if which_data == None: which_data = self.which_data
        
        if which_data in ['Both', 'both']:
            self.plot_one_tone_time_sweep(which_data = 'phase')
            self.plot_one_tone_time_sweep(which_data = 'logmag')
            
        logmag_array = self.one_tone_time_sweep_results['logmag']
        phase_array = self.one_tone_time_sweep_results['phase']
        freq_list = self.one_tone_time_sweep_results['frequency']
        time_list = self.one_tone_time_sweep_results['time_list']
        
        annot = None
        
        if which_data in ['LogMag', 'Logmag','logmag']:
            plot_2D(logmag_array, time_list, freq_list*1e-9, cmap = 'Reds',
                         xlabel = 'Time [s]', ylabel = 'Frequency [GHz]', zlabel = 'LogMag [dB]', is_colorbar = True,
                         annot = annot)
        if which_data in ['Phase','phase']:
            plot_2D(phase_array,  time_list, freq_list*1e-9,
                         xlabel = 'Time [s]', ylabel = 'Frequency [GHz]', zlabel = 'Phase $^\circ$', is_colorbar = True,
                         annot = annot)
            
    def cw_sweep(self,
                freq = None,
                npts = None,
                power = None,
                if_bw = None,
                navg = None,
                which_data = None,
                is_plot_fft = False,
                is_save_data = None
                ):
        
        if freq == None: freq = self.pna.CwFreq()
        if npts == None: npts = self.pna.num_points()
        if npts > 100001: raise ValueError(f"Max number of points is 10001. got <{npts}>")
        if power == None: power = self.pna.power()
        if if_bw == None: if_bw = self.pna.bandwidth()
        if navg == None: navg = int(self.pna.averages()[1])
        is_save_data = is_save_data if is_save_data is not None else self.is_save_data 
        if which_data is None: which_data = self.which_data
        self.check_s_parameters()
        self.set_pna_format(which_data = which_data)
        self.pna.set_triger_source(source='IMM')
        sweep_time = self.set_pna_cw(freq=freq, power=power, if_bw=if_bw, npts=npts, navg = navg)
        self.pna.trigger_single()
        sleep_in_steps(sweep_time)
        time_list = np.linspace(0, in_s(npts/if_bw), npts)
        self.pna.auto_scale()
        data = self.pna.getData(typ = 'Scale')
        
        self.cw_sweep_results={}
        self.cw_sweep_results['logmag'] = 20*np.log10(np.abs(data))
        self.cw_sweep_results['phase'] = np.angle(data, deg=True)
        self.cw_sweep_results['time'] = time_list
        self.cw_sweep_results['power'] = power
        self.cw_sweep_results['if_bw'] = if_bw
        self.cw_sweep_results['navg'] = navg
        
        if is_save_data:
            self.pickle_save(self.cw_sweep_results, meas_name = 'cw_time')
        
        self.plot_cw_sweep(which_data = which_data, is_plot_fft = is_plot_fft)
        
        
    def two_tone_sweep(self,
                       center,
                       span,
                       npts,
                       mxg_power,
                       pna_power,
                       pna_freq,
                       if_bw,
                       navg,
                       which_data = None,
                       is_fit = False,
                       n_cut = 1,
                       is_save_data = None,
                       meas_name = 'two_tone_sweep',
                       **kwargs,
                       ):
        """ If you are getting no signal, the trigger is not connected!!!"""
        """Run a two tone sweep using the PNA and the MXG.
        To fit number splitting pass is_fit = True and use kwargs to pass chi_guess, alpha_guess"""
        if npts > 100001: raise ValueError(f"Max number of points is 10001. got <{npts}>")
        if in_dbm(mxg_power) < -20 or in_dbm(mxg_power) > 30: raise ValueError(f"MXG power must be between -20dBm and 30dBm. got <{mxg_power}>dBm")
        if which_data is None: which_data = self.which_data
        self.check_s_parameters()
        self.set_pna_format(which_data = which_data)
        
        is_save_data = is_save_data if is_save_data is not None else self.is_save_data 
        self.results['two_tone_sweep'] = {}
        results_dict = self.results['two_tone_sweep']
        self.set_pna_cw(freq = pna_freq, power = pna_power, if_bw = if_bw, npts = npts, navg = navg)
        self.set_pna_ext_trigger(action = 'POIN')
        dwell_time = (1.2/if_bw+2*ms)
        
        freq_list = np.linspace(in_hz(center-span/2), in_hz(center+span/2), npts)
        
        self.set_mxg_freq_sweep(dwell_time = dwell_time, start = freq_list[0]*Hz, stop = freq_list[-1]*Hz, npts = npts, power = mxg_power)
        
        self.pna.powerOnOff(state='ON');
        self.mxg.on()
        
        sleep_step = 1
        
        single_sweep_time = in_s((dwell_time+4*ms)*npts)
        total_run_time = single_sweep_time*navg
        print(f'Run time is {(total_run_time/60):.2f} minutes')
        
        time_of_meas = datetime.now().strftime("%d_%m_%Y___%H_%M_%S")
        
        for n in range(navg):
            self.mxg.start_sweep('SING')
            sleep_in_steps(single_sweep_time)
            self.pna.auto_scale()
            
        data = self.pna.getData(typ ='Scale')
        logmag_array = 20*np.log10(np.abs(data))
        phase_array = np.angle(data, deg=True)
        
        self.mxg.off()        
            
        results_dict['logmag'] = logmag_array
        results_dict['phase'] = phase_array
        results_dict['frequency'] = freq_list
        results_dict['mxg_power'] = mxg_power
        results_dict['pna_freq'] = pna_freq
        results_dict['pna_power'] = pna_power
        results_dict['if_bw'] = if_bw
        results_dict['navg'] = navg
        
        if is_save_data:
                
            self.pickle_save(results_dict, meas_name = meas_name, time_of_meas = time_of_meas)
        
        if is_fit:
            try:
                if n_cut == 1:
                    if which_data in ['Phase','phase', 'Both']: self.plot_resonance(freq_list, phase_array)
                    if which_data in ['LogMag', 'Logmag','logmag', 'Both']: self.plot_resonance(freq_list, logmag_array, ylabel = 'LogMag [dB]')
                else:
                     if which_data in ['Phase','phase', 'Both']: self.plot_number_splitting(freq_list, phase_array, n_cut = n_cut, **kwargs)
                     if which_data in ['LogMag', 'Logmag','logmag', 'Both']: self.plot_number_splitting(freq_list, logmag_array, n_cut = n_cut, ylabel = 'LogMag [dB]',**kwargs)
            except:
                self.plot_two_tone_sweep(which_data = which_data, **kwargs)
        else:
            self.plot_two_tone_sweep(which_data = which_data, **kwargs)
                
        
        
    def plot_two_tone_sweep(self, which_data = None, ax = None, is_fit = False, n_cut = 1, **kwargs):
        if which_data is None: which_data = self.which_data
        results_dict = self.results['two_tone_sweep']

        if which_data in ['Both', 'both']:
            if ax is not None:
                print("You passed axis and also which data is set to both. Using passed axis to plot phase.")
            self.plot_two_tone_sweep(which_data = 'phase' , ax = ax)
            self.plot_two_tone_sweep(which_data = 'logmag')
            
        elif which_data in ['Phase','phase']:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                plt.sca(ax)
            x = results_dict['frequency'] * 1e-9
            y = results_dict['phase']
            plt.plot(x,y)
            plt.ylabel('Phase [Deg]', fontsize = 15)
            yl = plt.ylim()
            # plt.yticks(ticks = [-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
            #            labels = [r'$-\pi$', r'$-\frac{3\pi}{4}$', r'$-\frac{\pi}{2}$',
            #                      r'$-\frac{\pi}{4}$', r'$0$', r'$\frac{\pi}{4}$',  r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'], fontsize = 15)
            plt.xlabel('Frequency [GHz]', fontsize=15)
            plt.grid()
            plt.xticks(fontsize=15)
            plt.ylim(yl)
            ax.format_coord = lambda x, y: 'x={:g}, y={:g}'.format(x, y)
            mxg_power = results_dict['mxg_power']
            pna_power = results_dict['pna_power']
            pna_freq = results_dict['pna_freq']
            if_bw = results_dict['if_bw']
            navg = results_dict['navg']
            txt = 'MXG Power = {}\n PNA Power = {}\n PNA Freq = {}\n IF-BW = {}'.format(mxg_power,pna_power,pna_freq,if_bw)
            if navg == 1: txt += '\n Navg = {}'.format(navg)
            an = plt.annotate(txt, xy = (0.65, 0.05), xycoords = 'axes fraction', fontsize = 12)
            an.draggable()
            plt.tight_layout()
        elif which_data in ['LogMag','Logmag','logmag']:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                ax_logmag = ax
                plt.sca(ax_logmag)
            x = results_dict['frequency'] * 1e-9
            y = results_dict['logmag']
            plt.plot(x,y)
            plt.ylabel('LogMag [dB]', fontsize = 15)
            yl = plt.ylim()
            plt.xlabel('Frequency [GHz]', fontsize=15)
            plt.xticks(fontsize=15)
            plt.ylim(yl)
            ax.format_coord = lambda x, y: 'x={:g}, y={:g}'.format(x, y)
            plt.grid()
            mxg_power = results_dict['mxg_power']
            pna_power = results_dict['pna_power']
            pna_freq = results_dict['pna_freq']
            if_bw = results_dict['if_bw']
            navg = results_dict['navg']
            txt = 'MXG Power = {}\n PNA Power = {}\n PNA Freq = {}\n IF-BW = {}'.format(mxg_power,pna_power,pna_freq,if_bw)
            if navg == 1: txt += '\n Navg = {}'.format(navg)
            an = plt.annotate(txt, xy = (0.65, 0.05), xycoords = 'axes fraction', fontsize = 12)
            an.draggable()
            plt.tight_layout()
        else:
            raise ValueError(f"Unknown data type. (got <{which_data}>")
            
        if is_fit:
            if n_cut == 1:
                self.plot_resonance(x, y)
            else:
                self.plot_number_splitting(x, y, n_cut = n_cut, **kwargs)
            
    def plot_one_tone_sweep(self, which_data = None, ax = None):
        if which_data is None: which_data = self.which_data
        
        if which_data in ['Both', 'both']:
            self.plot_one_tone_sweep(which_data = 'phase' , ax = ax)
            self.plot_one_tone_sweep(which_data = 'logmag')
        
        power = self.one_tone_sweep_results['power']
        if which_data in ['Phase','phase', 'Both', 'both']:
            if ax is None:
                fig, ax = plt.subplots()
                is_annot = True
            else:
                is_annot = False
                plt.sca(ax)
            x = self.one_tone_sweep_results['frequency'] * 1e-9
            y = self.one_tone_sweep_results['phase']
            
            plt.plot(x,y, label = 'Power = {}'.format(power))
            plt.ylabel('Phase $^\circ$', fontsize = 15)
            plt.yticks(fontsize = 15)
            plt.xlabel('Frequency [GHz]', fontsize=15)
            plt.grid()
            if is_annot:
                an = plt.annotate('Power = {}'.format(power), xy = (0.5, 0.8), xycoords = 'axes fraction')
                an.draggable()
            plt.tight_layout()
            
        if which_data in ['LogMag', 'Logmag','logmag', 'Both', 'both']:
            if ax is None:
                fig, ax = plt.subplots()
                is_annot = True
            else:
                plt.sca(ax)
                is_annot = False
            x = self.one_tone_sweep_results['frequency'] * 1e-9
            y = self.one_tone_sweep_results['logmag']
            plt.plot(x,y, label = 'Power = {}'.format(power))
            plt.ylabel('LogMag [dB]', fontsize = 15)
            plt.yticks(fontsize = 15)
            plt.xlabel('Frequency [GHz]', fontsize=15)
            plt.grid()
            if is_annot:
                an = plt.annotate('Power = {}'.format(power), xy = (0.5, 0.8), xycoords = 'axes fraction')
                an.draggable()
            plt.tight_layout()
        if which_data not in ['Phase','phase','LogMag', 'Logmag','logmag', 'Both', 'both']:
            raise ValueError(f"Unknown data type. (got <{which_data}>")
    
    def plot_one_tone_power_sweep(self, which_data = None):
        if which_data == None: which_data = self.which_data
        
        if which_data in ['Both', 'both']:
            self.plot_one_tone_power_sweep(which_data = 'phase')
            self.plot_one_tone_power_sweep(which_data = 'logmag')
            
        results_dict = self.results['one_tone_power_sweep_results']
        logmag_array = results_dict['logmag']
        phase_array = results_dict['phase']
        freq_list = results_dict['frequency']
        power_list = results_dict['power']
        
        if which_data in ['LogMag', 'Logmag','logmag']:
            plot_2D(logmag_array, power_list, freq_list*1e-9, cmap = 'Reds',
                         xlabel = 'Power [dBm]', ylabel = 'Frequency [GHz]', zlabel = 'LogMag [dB]', is_colorbar = True)
        if which_data in ['Phase','phase']:
            plot_2D(phase_array,  power_list, freq_list*1e-9,
                         xlabel = 'Power [dBm]', ylabel = 'Frequency [GHz]', zlabel = 'Phase $^\circ$', is_colorbar = True)
        
    def plot_cw_sweep(self, which_data = None, is_plot_fft = False, **kwargs):
        if which_data is None: which_data = self.which_data
        if which_data in ['Both','both']:
            self.plot_cw_sweep(which_data='phase')
            self.plot_cw_sweep(which_data='logmag')
            
        elif which_data in ['Phase','phase']:
            plt.figure()
            x = self.cw_sweep_results['time'] * 1e3
            y = self.cw_sweep_results['phase']
            plt.plot(x,y)
            plt.ylabel('Phase $^\circ$', fontsize = 15)
            plt.yticks(fontsize = 15)
            plt.xlabel('Time [ms]', fontsize=15)
            plt.tight_layout()
            if is_plot_fft: self.plot_fft(x*1e-3,y, is_filter_dc = True, **kwargs)
        elif which_data in ['LogMag', 'Logmag','logmag']:
            plt.figure()
            x = self.cw_sweep_results['time'] * 1e3
            y = self.cw_sweep_results['logmag']
            plt.plot(x,y)
            plt.ylabel('LogMag [dB]', fontsize = 15)
            plt.xlabel('Time [ms]', fontsize=15)
            plt.tight_layout()
            if is_plot_fft: self.plot_fft(x*1e-3,y, is_filter_dc = True,  **kwargs)
        else:
            raise ValueError(f"Unknown data type. (got <{which_data}>")
        
        
    def set_pna_format(self, which_data):
        if which_data is None: which_data = self.which_data
        if which_data in ['Phase','phase', 'Both']:
            self.pna.format_typ('PHAS')
        elif which_data in ['LogMag', 'Logmag','logmag']:
            self.pna.format_typ('MLOG')
            
            
    def set_pna_sweep(self,
                      center,
                      span,
                      npts,
                      power,
                      if_bw,
                      navg
                      ):

        if npts > 100001: raise ValueError(f"Max number of points is 10001. got <{npts}>")
        self.pna.power(p = power)
        self.pna.set_sweep_type('LIN')
        self.pna.frequencey_span(CenFreq = center,span = span)
        self.pna.bandwidth(if_bw)
        self.pna.averages(av =navg)
        self.pna.num_points(n = npts)
        self.pna.Triger_to_cont()
        sweep_time = npts * (1.1/in_hz(if_bw) + 100e-6)
        return sweep_time
    
    def set_pna_cw(self,
                   freq,
                   power,
                   if_bw,
                   npts,
                   navg):
        if npts > 100001: raise ValueError(f"Max number of points is 10001. got <{npts}>")
        self.pna.bandwidth(bw = if_bw)#in seconds
        self.pna.num_points(n = npts)# the number of points of the PNA sweep 
        self.pna.averages(av =navg)
        self.pna.power(p = power)
        self.pna.set_sweep_type('CW')
        self.pna.CwFreq(f = freq)
        self.pna.Triger_to_cont()
        sweep_time = npts / in_hz(if_bw) + 2e-3
        return sweep_time

    def set_pna_ext_trigger(self, action):
        self.pna.set_triger_source()
        self.pna.set_triger_type()
        self.pna.set_triger_slope()
        self.pna.set_following_triger(action)
        
        
    
        

    def pickle_save(self, to_save_dict = None, meas_name = None, folder_name = None, time_of_meas = None, is_print = True):
            
        if time_of_meas is None:
            time_of_meas = datetime.now().strftime("%d_%m_%Y___%H_%M_%S")
        
        to_save_dict.update({'time_of_meas': time_of_meas})
        
        if folder_name is None:
            if self.save_folder is not None:
                folder_name = self.save_folder
                if not os.path.exists(self.save_folder):                
                    os.makedirs(self.save_folder)
            else:
                folder_name = ''
        else:
            if not os.path.exists(self.save_folder):                
                    os.makedirs(self.save_folder)
                    
        if meas_name is None:
            meas_name = 'unnamed_meas'
        to_save_dict.update({'meas_name': meas_name})
        
        filename = meas_name + '_' + time_of_meas
        
        filepath = folder_name + '\\' + filename
        
        with open(f'{filepath}.pickle', 'wb') as handle:
            pickle.dump(to_save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        if is_print:
            print('\n'+f'Saved data to {filepath}'+'\n')
            
    def pickle_load(self, filepath):
        with open(f'{filepath}.pickle', 'rb') as handle:
            loaded_dict = pickle.load(handle)
        self.results[loaded_dict['meas_name']] = loaded_dict
        return loaded_dict
            
    def plot_from_file(self, filepath, ax = None, which_data = None):
        loaded_dict = self.pickle_load(filepath)
        if loaded_dict['meas_name'] == 'two_tone_flux_sweep':
            self.plot_two_tone_flux_sweep(ax=ax,which_data = which_data)
        elif loaded_dict['meas_name'] == 'one_tone_flux_sweep':
            self.plot_one_tone_flux_sweep(ax=ax, which_data = which_data)
        elif loaded_dict['meas_name'] == 'one_tone_sweep':
            self.plot_one_tone_sweep(ax=ax, which_data = which_data)
        elif loaded_dict['meas_name'] == 'two_tone_sweep':
            self.plot_two_tone_sweep(ax=ax, which_data = which_data)
        elif loaded_dict['meas_name'] == 'one_tone_power_sweep':
            self.plot_one_tone_power_sweep(ax=ax, which_data = which_data)
        elif loaded_dict['meas_name'] == 'cw_time':
            self.plot_cw_sweep(which_data = which_data)
            
    
    def sleep_in_steps(self, sleep_time, step = 1):
        t=0
        while t < sleep_time:
            sleep(step)
            t+=step
        sleep(sleep_time+step-t)
        
        
    def plot_fft(self, times=None, data=None, title_str = None, fig_num = None, lgd = '', is_filter_dc = False, **kwargs):
        """Assumes times is in seconds and points are equally spread"""
        fig = plt.figure(fig_num) if fig_num is not None else plt.figure()
        
        Tfinal = times[-1]
        N = len(times)
        x = np.linspace(0, Tfinal, N)
        if is_filter_dc: yf = fftpack.fft(data - data.mean(),N)
        else: yf = fftpack.fft(data,N)
        xf = np.linspace(0.0, 1.0/(2.0*(Tfinal/N)), N//2)
        plt.plot(xf, 2.0/N * np.abs(yf[:N//2]),'o-', label = lgd)
        # leg =plt.legend()
        # leg.set_draggable(True)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(r'$|F(f)|$')
        plt.tight_layout()
        if title_str is not None:
            plt.title(title_str, fontsize = 10)
        else:
            plt.title(' FFT', fontsize = 10)
            
            
    def get_temperature(self, ch=6, device_ip='192.168.0.30', timeout=10, update_interval=None):
              
        if update_interval is None: update_interval = self.temp_update_interval
        
        now = datetime.now()
        time_interval = now - self.last_temp_time
        
        if  time_interval.seconds > update_interval:
        
            url = 'http://{}:5001/channel/historical-data'.format(device_ip)
            prev = now - timedelta(hours=5) # minimum working timedelta is 4 hours for some reason
            keys = {
                'channel_nr': ch,        
                'start_time': prev.strftime("%Y-%m-%dT%H:%M:%SZ"),
                'stop_time': now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                'fields': ['temperature']
                }  
            req = requests.post(url, json=keys, timeout=timeout)
            data = req.json()   
            self.last_temp_time = now
            self.temp = data['measurements']['temperature'][-1]
            
        return self.temp
    
    
    def check_heating(self, T_thresh = 20e-3):
        if self.get_temperature()>T_thresh: 
            self.ps.set_curr_smooth(0)
            raise ValueError(f"The coil is heating up the fridge. T>{T_thresh*1e3} mK")
        else: pass
    
    
    def get_heater_params(self, ch, device_ip='192.168.0.30', timeout=10):
        url = 'http://{}:5001/heater'.format(device_ip)
        keys = {         
            'heater_nr': ch
            }
        req = requests.post(url, json=keys, timeout=timeout)
        data = req.json()
        
        return data
    
    
    def set_heater_params(self, setpoint, ch=4, P_pid=0.01, I_pid=250, D_pid=0, max_power=0.001, device_ip='192.168.0.30', timeout=10):
        
        active_status = self.get_heater_params(ch)['active']
        
        control_algorithm_settings = {'proportional': P_pid,
                                      'integral': I_pid,
                                      'derivative': D_pid}
        
        url = 'http://{}:5001/heater/update'.format(device_ip)
        data = {
            'heater_nr': ch,  
            'active': active_status,         
            'setpoint': setpoint,    #[K]
            'control_algorithm_settings': control_algorithm_settings,
            'max_power': max_power
        }
        req = requests.post(url, json=data, timeout=timeout)
        data = req.json()
        print('Response: \n{}'.format(json.dumps(data, indent=2)))    
    
    
    def set_heater_on(self, ch=4, device_ip='192.168.0.30', timeout=10):
        url = 'http://{}:5001/heater/update'.format(device_ip)
        data = {
            'active': True,         
            'heater_nr': ch   
        }
        req = requests.post(url, json=data, timeout=timeout)
        data = req.json()
        print('Response: \n{}'.format(json.dumps(data, indent=2)))   
        
    def set_heater_off(self, ch=4, device_ip='192.168.0.30', timeout=10):
        #Turn off heater
        url = 'http://{}:5001/heater/update'.format(device_ip)
        data = {
            'active': False,         
            'heater_nr': ch,     
            'setpoint': 0e-3      #[K]
        }
        req = requests.post(url, json=data, timeout=timeout)
        data = req.json()
        print('Response: \n{}'.format(json.dumps(data, indent=2)))    
    
    
    def wait_temp_stabilize(self,TempStabilizationTime=100):
        #TODO sample temp and pressure every 30 seconds, abort if overpressure
        #TODO return some meaningful value of stabilization success
        sleep(TempStabilizationTime)
    
     #TODO install BF Com
    # def get_maxigauge_logs(self):
        
    #     subprocess.check_output("net use B: /delete /y", shell=True)
    #     subprocess.check_output('net use B: \\\\79FZ593\\BFLogs BranJosephson /user:79FZ593\\BFlogs', shell=True)
    #     today = date.today()
    #     yymmdd = today.strftime("%y-%m-%d")
    #     with open("B:\\"+yymmdd+"\\maxigauge "+yymmdd+".log","r") as file:
    #         for line in file:
    #             pass #"line" holds the last line now
    #     dt_string = line.split(",")[0]+','+line.split(",")[1]
    #     dt_log = datetime.strptime(dt_string, "%d-%m-%y,%H:%M:%S")
    #     logage=(datetime.now()-dt_log).total_seconds() #sanity check - should be less than 300
    #     P4=float(line.split(",")[23]) #P4 pressure sensor in mbar - should be less than 750 (V13 opens at 800, BPV3 at 1000 to avoid He3 loss)
    #     return P4
       
    
    def _fit_resonance(self, x, y):
        
        #check if negative lorenzian
        if np.abs(y.mean() - y.min()) > np.abs(y.mean() - y.max()): 
            width = np.abs(x[np.argmin(np.abs(y-(max(y)+min(y))/2))]-x[np.argmax(y)])
            amp = (np.min(y)-np.max(y))*width**2
            freq = x[np.argmin(y)]
            offset = np.max(y)
        else:
            width = np.abs(x[np.argmax(np.abs(y-(max(y)+min(y))/2))]-x[np.argmin(y)])
            amp = -(np.max(y)-np.min(y))*width**2
            freq = x[np.argmax(y)]
            offset = np.min(y)
    
        guess = [amp, freq, width, offset]
        fit, cov = curve_fit(lorentzian, x, y, guess)
        errors = np.sqrt(np.diag(cov))
        return fit, errors
    
    def _fit_resonance_complex(self, x,y, k_in = None, k_ext = None, f_c = None, phase_off = None, amp = None):
        y_abs = np.abs(y)
        k_tot = (x[-1]-x[0]) / 4
        print(k_tot)
        if k_ext is None:
            k_ext = ((y_abs[0]-y_abs[len(y)//2])/y_abs[0] + 1) * k_tot  / 2
        if k_in is None:
            k_in = k_tot - k_ext
        if amp is None:
            amp = (y_abs[-1]+y_abs[0])/2
        if phase_off is None:
            phase_off = np.angle(y[0])
        if f_c is None:
            f_c = x[len(x)//2]
        y2 = np.append(y.real, y.imag)
        guess = [k_in, k_ext, f_c, phase_off, amp]
        print(guess)
        fit, cov = curve_fit(reflection_S11_flat, x, y2, guess, bounds = ([0,0,0,0,0],[0.1,0.1,10,360,1]))
        print(fit)
        errors = np.sqrt(np.diag(cov))
        return fit, errors
        
    def plot_resonance_complex(self, x,y, k_in = None, k_ext = None, f_c = None, phase_off = None, amp = None):
        try:
            fit, errors = self._fit_resonance_complex(x, y, k_in, k_ext, f_c, phase_off, amp)
            fit_success = True
        except RuntimeError:
            print('Could not find fit')
            fit = None
            errors = None
            fit_success = False
              
        plt.figure()
        plt.plot(y.real, y.imag, 'o')
        
        if fit_success: 
            fitted_data = reflection_S11(x,*fit)
            
            plt.plot(fitted_data.real, fitted_data.imag)
        plt.xlabel('Re[S11]')
        plt.ylabel('Im[S11]')
        plt.grid()
        plt.tight_layout()
        if fit_success : 
            k_in = round_value_by_error(fit[0],errors[0])
            k_ext = round_value_by_error(fit[1],errors[1])
            f_c = round_value_by_error(fit[2],errors[2])
            phase_off = round_value_by_error(fit[3],errors[3])
            amp = round_value_by_error(fit[4],errors[4])
            # delay_distance = round_value_by_error(fit[5],errors[5])
            # print(f'Amp = {fit[0]} +- {errors[0]}')
            # print(f'Freq = {fit[1]} +- {errors[1]}')
            # print(f'Width = {fit[2]} +- {errors[2]}')
            # print(f'Offset = {fit[3]} +- {errors[3]}')
            annot = r'$\kappa_{in}$' + f' = {np.around(k_in[0]*1e3,5)} +- {np.around(k_in[1]*1e3,5)} MHz \n' \
                    + r'$\kappa_{ext}$' + f' = {np.around(k_ext[0]*1e3,5)} +- {np.around(k_ext[1]*1e3,5)} MHz \n' \
                    + f'f = {f_c[0]} +- {f_c[1]} GHz\n'
            an = plt.annotate(annot, xy = (0.5, 0.8), xycoords = 'axes fraction')
            an.draggable()
        return fit, errors
        
    def _fit_resonance_hanging(self, x, y):
        """fits the function self.hanging_res to data in linear scaling"""
        freq = x[np.argmin(y)]
        offset = max(y)-1
        width = 2*np.abs(x[np.argmin(np.abs(y-(max(y)+min(y))/2))]-freq)
        temp1 = min(y)-offset
        temp2 = ((max(y)+min(y))/2 - offset)**2
        temp3 = width**2 * (1-temp2)
        kappa_l = np.sqrt(temp3/(temp2 - temp1**2))        
        kappa_i = temp1*kappa_l
        
        guess = [freq, kappa_l, kappa_i, offset]
        fit, cov = curve_fit(hanging_res, x, y, guess)
        errors = np.sqrt(np.diag(cov))
        return fit, errors
    
    
    def plot_resonance_hanging(self, x,y, xlabel='Frequency', ylabel='LogMag'):
        """fits and plots a non assymetric hanging resonator S21 in LogMag. Validity verified in simulation"""
        scaled_x, prefix_x, factor_x = scale_data_units(x)
        # scaled_y, prefix_y, factor_y = scale_data_units(y)
        scaled_y  = y
        if ylabel == 'Phase' or ylabel == 'phase':
            y_units = 'Degrees'
        elif ylabel in ['LogMag', 'logmag', 'Logmag']:
            y_units = 'dB'
            y_lin = 10**(scaled_y/10)
        else:
            y_units = ''
        try:
            fit, errors = self._fit_resonance_hanging(scaled_x, y_lin)
            fit_success = True
        except:
              print('Could not find fit')
              fit = None
              errors = None
              fit_success = False
        plt.figure()
        plt.plot(scaled_x, scaled_y, 'o')
        if fit_success: plt.plot(scaled_x, 10*np.log10(hanging_res(scaled_x,*fit)))
        plt.xlabel(f'{xlabel} [{prefix_x}Hz]')
        plt.ylabel(f'{ylabel} [{y_units}]')
        plt.grid()
        plt.title('Hanging Resonator S21')
        if fit_success: 
            Freq = round_value_by_error(fit[0],errors[0])
            kappa_l = round_value_by_error(fit[1],errors[1])
            kappa_i = round_value_by_error(fit[2],errors[2])
            Offset = round_value_by_error(fit[3],errors[3])
            annot = f'$Freq = {np.round(Freq[0],6)} \pm {np.round(Freq[1],6)}$ GHz'+'\n' \
                    + f'$\kappa_l/2\pi = {np.round(kappa_l[0]*1e3,4)} \pm {np.round(kappa_l[1]*1e3,4)}$ MHz'+'\n' \
                    + f'$\kappa_i/2\pi = {np.round(kappa_i[0]*1e3,4)} \pm {np.round(kappa_i[1]*1e3,4)}$ MHz'+'\n' \
                    + f'$Offset = {np.round(Offset[0],3)} \pm {np.round(Offset[1],3)}$ LIN'
            an = plt.annotate(annot, xy = (0.15, 0.2), xycoords = 'axes fraction')
            an.draggable()
        plt.tight_layout()
        return fit, errors
    
    
    def plot_resonance(self, x,y, xlabel = 'Frequency', ylabel = 'Phase'):
        scaled_x, prefix_x, factor_x = scale_data_units(x)
        # scaled_y, prefix_y, factor_y = scale_data_units(y)
        scaled_y  = y
        if ylabel == 'Phase' or ylabel == 'phase':
            y_units = 'Degrees'
        elif ylabel in ['LogMag', 'logmag', 'Logmag']:
            y_units = 'dB'
        else:
            y_units = ''
        try:
            fit, errors = self._fit_resonance(scaled_x, scaled_y)
            fit_success = True
        except:
              print('Could not find fit')
              fit = None
              errors = None
              fit_success = False
        plt.figure()
        plt.plot(scaled_x, scaled_y, 'o')
        if fit_success: plt.plot(scaled_x, lorentzian(scaled_x,*fit))
        plt.xlabel(f'{xlabel} [{prefix_x}Hz]')
        plt.ylabel(f'{ylabel}')
        plt.grid()
        if fit_success : 
            Amp = round_value_by_error(fit[0],errors[0])
            Freq = round_value_by_error(fit[1],errors[1])
            Width = round_value_by_error(fit[2],errors[2])
            Offset = round_value_by_error(fit[3],errors[3])
            # print(f'Amp = {fit[0]} +- {errors[0]}')
            # print(f'Freq = {fit[1]} +- {errors[1]}')
            # print(f'Width = {fit[2]} +- {errors[2]}')
            # print(f'Offset = {fit[3]} +- {errors[3]}')
            annot = f'Amp = {Amp[0]} +- {Amp[1]} {y_units}\n' \
                    + f'Freq = {Freq[0]} +- {Freq[1]} GHz\n' \
                    + f'Width = {np.round(Width[0]*1e3,4)} +- {np.round(Width[1]*1e3,4)} MHz\n' \
                    + f'Offset = {Offset[0]} +- {Offset[1]} {y_units}'
            an = plt.annotate(annot, xy = (0.5, 0.8), xycoords = 'axes fraction')
            an.draggable()
        plt.tight_layout()
        return fit, errors
    
    def plot_number_splitting(self, freqs, data, chi_guess = 0, alpha_guess = 0, n_cut = 2, xlabel = 'Frequency', ylabel = 'Phase'):
        scaled_x, prefix_x, factor_x = scale_data_units(freqs)
        scaled_y, prefix_y, factor_y = scale_data_units(data)
        fit, errors, func = self._fit_number_splitting(freqs, data, chi_guess, alpha_guess, n_cut)
        plt.figure()
        plt.plot(freqs, data, 'o')
        plt.plot(freqs, func(freqs, *fit))
        plt.xlabel(f'{xlabel} [{prefix_x}Hz]')
        plt.ylabel(f'{ylabel}')
        plt.tight_layout()
        fit, errors = round_value_by_error(fit, errors)
        print(f'Amp = {fit[0]} +- {errors[0]}')
        print(f'Freq = {fit[1]} +- {errors[1]}')
        print(f'Width = {fit[2]} +- {errors[2]}')
        print(f'Chi = {fit[3]} +- {errors[3]}')
        print(f'Alpha = {fit[4]} +- {errors[4]}')
        print(f'Offset = {fit[5]} +- {errors[5]}')
        return fit, errors
    
    def _fit_number_splitting(self, x, y, chi_guess = 0, alpha_guess = 0, n_cut = 2):
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
    
    
    # Aron function.
    """
    We need to break this up into a lot of smaller frequencies.  
    - Then have a function that analyses all the one tone sweep files we did and does circle fits
    - Then do a function that sweeps temperature and uses the previous functions. 
    - Then also do a function that sweeps power and uses the previous functions 
    Code takes in a dictionary containing frequencies, and d
    """
    
    def Aron_multitone_scan(self, res_params,
                            save_params = True): 
        
        if save_params:
            saveFolder = self.save_folder
            self.pickle_save(to_save_dict=res_params, meas_name='res_params', folder_name=saveFolder, is_print=False)
        
        self.pna.electrical_delay(res_params['electrical_delay'])
        self.pna.phase_offset(res_params['phase_offset'])
        names = res_params['names']
        freq_center = res_params['freq_center']
        power_ro = res_params['power_ro'] 
        span = res_params['span']
        npts = res_params['num_points']
        
        for i in range(len(names)):
            print(names[i]+': \n')
            fridge_temp = self.get_temperature()
            meas_name = f'one_tone_sweep {self.exp_name} {names[i]} {freq_center[i]/1e9} GHz {power_ro[i]} dBm temp {fridge_temp*1e3:.2f} mK '
            self.one_tone_sweep(
                                    center = freq_center[i],
                                    span = span[i],
                                    npts = npts[i],
                                    power = power_ro[i],
                                    navg = res_params['averages'],
                                    if_bw = res_params['bandwidth'],
                                    is_save_data = True,
                                    meas_name = meas_name,
                                    is_get_temp = True,
                                    is_plot = False
                                    )
            
            
    def Guy_multitone_circle_fit(self, res_params):
        
        exp = self.exp_name
        saveFolder = self.save_folder
        names = res_params['names']

        data_all = {name:{'freq':np.array([]),'freq_err':np.array([]),'kappa_i':np.array([]),'kappa_i_err':np.array([]),
                      'kappa_c':np.array([]),'kappa_c_err':np.array([]),'kappa_l':np.array([]),
                      'kappa_l_err':np.array([]),'theta0':np.array([]),'phi0':np.array([]),'phi0_err':np.array([]),
                      'chi_square':np.array([]),'temperature':np.array([]),'ro_power':np.array([])} for name in names}

        for root, subdirs, files in os.walk(saveFolder):
            for fname in files:
                
                counter = 1
                
                if '.pickle' not in fname:
                    continue
                if exp not in fname:
                    continue
                if 'data_all' in fname:
                    continue
                
                fname = fname[:-7]
                data = self.pickle_load(root+'\\'+fname)
                print(data)
                
                f = data['frequency']
                logmag = data['logmag']
                phase = data['phase']*np.pi/180
                temp = data['temp']
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
                   r'$phi0=%.2f\pm%.2f[deg]$' % (vals['phi0']*180/np.pi, vals['phi0_err']*180/np.pi, ),
                   r'$theta0=%.2f[deg]$' % (vals['theta0']*180/np.pi, ),
                   r'$\chi^2=%.3f$' % (vals['chi_square'], )))
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.59, 0.47, textstr, fontsize=12,
                        verticalalignment='top', bbox=props)
                plt.show()

                plt.savefig(root+'\\'+fname+'.png')
                plt.clf()
                
                print('\n')
                print('kappa_i/2*pi =',np.round(vals['fr']/vals['Qi_dia_corr']*1e-3,2),'kHz')        
                print('kappa_e/2*pi =',np.round(vals['fr']/vals['Qc_dia_corr']*1e-3,2),'kHz')       
                
                name = "Op" + str(counter)
                
                if data_all[name]['ro_power']==[]: data_all[name]['ro_power'] = ro_pwr
                
                data_all[name]['temperature'] = np.append(data_all[name]['temperature'],temp)
                
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

        self.pickle_save(data_all,'data_all '+exp+' ')
    
    
    def temp_multitone_circle_fit(self, res_params):
        
        exp = self.exp_name
        saveFolder = self.save_folder
        names = res_params['names']

        data_all = {name:{'freq':np.array([]),'freq_err':np.array([]),'kappa_i':np.array([]),'kappa_i_err':np.array([]),
                      'kappa_c':np.array([]),'kappa_c_err':np.array([]),'kappa_l':np.array([]),
                      'kappa_l_err':np.array([]),'theta0':np.array([]),'phi0':np.array([]),'phi0_err':np.array([]),
                      'chi_square':np.array([]),'temperature':np.array([]),'ro_power':np.array([])} for name in names}

        for root, subdirs, files in os.walk(saveFolder):
            for fname in files:
                
                if '.pickle' not in fname:
                    continue
                if 'one_tone_sweep' not in fname:
                    continue
                if 'data_all' in fname:
                    continue

                # --- Determine resonator name from filename (e.g., "Op1", "Op2", ...) ---
                m = re.search(r'\bOp\d+\b', fname)
                if m is None:
                    continue
                name = m.group(0)
                if name not in data_all:
                    continue
                # ---------------------------------------------------------------
                
                fname = fname[:-7]
                data = self.pickle_load(root+'\\'+fname)
                #print(data)
                
                f = data['frequency']
                logmag = data['logmag']
                phase = data['phase']*np.pi/180
                temp = data['temp']
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
                   r'$phi0=%.2f\pm%.2f[deg]$' % (vals['phi0']*180/np.pi, vals['phi0_err']*180/np.pi, ),
                   r'$theta0=%.2f[deg]$' % (vals['theta0']*180/np.pi, ),
                   r'$\chi^2=%.3f$' % (vals['chi_square'], )))
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.59, 0.47, textstr, fontsize=12,
                        verticalalignment='top', bbox=props)
                plt.show()

                plt.savefig(root+'\\'+fname+'.png')
                plt.clf()
                
                print('\n')
                print('kappa_i/2*pi =',np.round(vals['fr']/vals['Qi_dia_corr']*1e-3,2),'kHz')        
                print('kappa_e/2*pi =',np.round(vals['fr']/vals['Qc_dia_corr']*1e-3,2),'kHz')       
                
                if data_all[name]['ro_power']==[]: 
                    data_all[name]['ro_power'] = ro_pwr
                
                data_all[name]['temperature'] = np.append(data_all[name]['temperature'],temp)
                
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

        self.pickle_save(data_all,'data_all '+exp+' ')
        
    def plot_experiment_graphs(self, res_params, resname = ''):
        
        names = res_params['names']
        
        for root, subdirs, files in os.walk(self.save_folder):
            for fname in files:
                
                if '.pickle' not in fname:
                    continue
                if 'data_all' in fname:
                    continue
                if 'res_params' in fname: 
                    continue 
                if resname in fname and 'one_tone_sweep' in fname:
                    fname = fname[:-7]
                    data = self.pickle_load(root+'\\'+fname)
                    frequency = np.array(data['frequency'])
                    logmag = np.array(data['logmag'])
                    
                    # Plot
                    plt.figure(figsize=(8, 5))
                    plt.plot(frequency, logmag)
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Log Magnitude (dB)')
                    plt.title(f'LogMag vs Frequency: {fname}')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()
                    #"""
                    
                    
                    
                    
    def analyse_temp_data(self, res_params):
        
        names = res_params['names']
        
        for root, subdirs, files in os.walk(self.save_folder):
            for fname in files:
                
                if '.pickle' not in fname:
                    continue
                if 'one_tone_sweep' in fname:
                    continue
                if 'data_all' in fname:
                    fname = fname[:-7]
                    temp_data = self.pickle_load(root+'\\'+fname)
                    for name in names: 
                        freq = temp_data[name]['freq']
                        freq_error = temp_data[name]['freq_err']
                        temperature = temp_data[name]['temperature']
                        
                        # Add in a plot here of frequency as it depends on temperature
                        plt.figure()
                        plt.errorbar(temperature, freq, yerr=freq_error, fmt='o')
                        plt.xlabel('Temperature (mK)')
                        plt.ylabel('Resonance frequency (Hz)')
                        plt.title(f'{name}: Frequency vs Temperature')
                        plt.tight_layout()
                        plt.show()    
                  
                    
            
    def Aron_multitone_circle_fit(self, ):
        
        all_data = []
        for root, subdirs, files in os.walk(self.save_folder):
            for fname in files:

                if not fname.endswith('.pickle'):
                    continue
                if self.exp_name not in fname:
                    continue
                if 'data_all' in fname:
                    continue

                fname_noext = fname[:-7]
                data = self.pickle_load(root + '\\' + fname_noext)
                all_data.append(data)
                print(data)
                f = np.asarray(data['frequency'])              # Hz
                logmag = np.asarray(data['logmag'])            # dB
                phase = np.asarray(data['phase']) * np.pi/180  # rad
                temp = data['temp']
                freq = data['frequency'][data['frequency'].size//2]
                ro_pwr = data['power']
    
    
    def do_temp_sweep(self, temps, res_params, save_params = True):
        
        print("Okay let's start the temperature sweep")
        
        if save_params:
            saveFolder = self.save_folder
            self.pickle_save(to_save_dict= res_params, meas_name='res_params', folder_name=saveFolder, is_print=False)
        
        self.pna.electrical_delay(res_params['electrical_delay'])
        self.pna.phase_offset(res_params['phase_offset'])
        names = res_params['names']
        freq_center = res_params['freq_center']
        power_ro = res_params['power_ro'] 
        span = res_params['span']
        npts = res_params['num_points']
        bandwidth = res_params['bandwidth']
        averages = res_params['averages']
        
        
        Temps = temps 
        TempStabilizationTime = 2000 #[s]

        P_pid=0.04
        I_pid=250
        D_pid=0
        max_power=0.005 #[W]

        self.set_heater_on()


        for T in Temps:
            self.set_heater_params(setpoint=T, ch=4, P_pid=P_pid, I_pid=I_pid, D_pid=D_pid, max_power=max_power)
            self.sleep_in_steps(TempStabilizationTime)
            
            for i in np.arange(len(names)):
                
                temperature = self.get_temperature() # [K]
                if temperature >= 0.55:
                    npts[i] = 2001
                FileName = f'{names[i]} {freq_center[i]/1e9} GHz {power_ro[i]} dBm {np.round(temperature*1e3,3)} mK'

                self.one_tone_sweep(center = freq_center[i],
                                      span = span[i],
                                      npts = npts[i],
                                      power = power_ro[i],
                                      if_bw = bandwidth,
                                      navg = averages,
                                      meas_name = 'one_tone_sweep '+FileName,
                                      is_get_temp = True,
                                      is_plot = False)
            
        self.set_heater_off()
        
        
        
        
    def do_power_sweep(self, powers_array, res_params,
                            save_params = True):
        print("Okay, let's start the power sweep")
        
        if save_params:
            saveFolder = self.save_folder
            self.pickle_save(to_save_dict=res_params, meas_name='res_params', folder_name=saveFolder, is_print=False)
        
        self.pna.electrical_delay(res_params['electrical_delay'])
        self.pna.phase_offset(res_params['phase_offset'])
        names = res_params['names']
        freq_center = res_params['freq_center']
        span = res_params['span']
        npts = res_params['num_points']
        bandwidth = res_params['bandwidth']
    
        for i in range(len(names)):
            
            for power in powers_array:
                
                if power >= -40: 
                    number_pts = 501
                elif power <= -75: 
                    number_pts = 2001
                    #bandwidth = 0.001e3
                else:
                    number_pts = npts[i]
                    
                print(names[i]+': \n')
                fridge_temp = self.get_temperature()
                meas_name = f'one_tone_sweep {self.exp_name} {names[i]} {freq_center[i]/1e9} GHz {power} dBm temp {fridge_temp*1e3:.2f} mK '
                self.one_tone_sweep(
                                        center = freq_center[i],
                                        span = span[i],
                                        npts = number_pts,
                                        power = power,
                                        navg = res_params['averages'],
                                        if_bw = bandwidth,
                                        is_save_data = True,
                                        meas_name = meas_name,
                                        is_get_temp = True,
                                        is_plot = False
                                        )
        
    
    def calibrate_hanging_resoators(self, name_list, freq_list, width_list, npts = 1001, if_bw = 1000, power = -30,
                            is_plot = False, is_save_data = None, max_attempts = 10):
        
        if is_save_data == None: is_save_data = self.is_save_data
        
        freq_list_final = []
        kappa_list_final = []
        width_list_final = []
        for name,freq,width in zip(name_list,freq_list,width_list):
            print('res',name)
            
            is_calibration_done = False
            n_attempt = 0
            while not is_calibration_done:
                n_attempt += 1
                if n_attempt+1 == max_attempts:
                    print('number of attempts exceeded! \n')
                    continue
                print(f'attempt {n_attempt}/{max_attempts}')
                print(f'f={freq*1e-9}GHz ; width={width*1e-6}MHz')
                
                self.one_tone_sweep(center = freq,
                                       span = width,
                                       npts = npts,
                                       power = power,
                                       if_bw = if_bw,
                                       navg = 1,
                                       is_save_data = is_save_data,
                                       meas_name = 'one_tone_sweep '+name,
                                       is_get_temp = False,
                                       is_plot = is_plot)
                
                data = self.one_tone_sweep_results
                
                f = data['frequency']
                logmag = data['logmag']
                phase = data['phase']*np.pi/180
                sig = 10**(logmag/20) * (np.cos(phase) + 1j*np.sin(phase))
                
                sig_real_filt = signal.savgol_filter(np.real(sig), 21, 3)
                sig_imag_filt = signal.savgol_filter(np.imag(sig), 21, 3)
                sig_filt = sig_real_filt + 1j*sig_imag_filt
    
                port = circuit.notch_port()
                port.add_data(f,sig_filt)
                try:
                    port.autofit()
                    vals = port.fitresults
                    if not any(np.isnan([vals['fr'],vals['Ql']])):
                        freq = vals['fr']
                        kappa = freq/vals['Ql']
                    else:
                        print('Fit failed')
                        kappa = 0
                        freq = f[np.argmin(logmag)]
                        width *= 0.2
                        
                    if kappa > 0.2*width:
                        is_calibration_done = True
                        print('res',name,'Calibration done! \n')
                    else:
                        width *= 0.2
                            
                        
                except:
                    print('Fit failed')
                    freq = f[np.argmin(logmag)]
                    width *= 0.2
            freq_list_final = np.append(freq_list_final,freq)
            kappa_list_final = np.append(kappa_list_final,kappa)
            width_list_final = np.append(width_list_final,width)
        
        return freq_list_final, kappa_list_final, width_list_final
            
        
    
if __name__ == '__main__':
    
    freqdo = FreqDo('MIDDLE',
                    which_data = 'Phase',
                    save_folder = r'F:\OneDrive - Technion\Eliya\Experiment Data\entanglement stabilization\Spectroscopy\Two tone flux sweep')
