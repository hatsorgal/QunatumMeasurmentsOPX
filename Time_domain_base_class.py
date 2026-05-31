# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:22:45 2020

"""

# for reltive folder imports:
import pathlib
import sys, os
import os.path
import itertools

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent /'usefulFunctions'))

sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\measurementClass') # for fridge 2 new pc
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\ConfigurationClasses') # for fridge 2 new pc
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\InstrumentControl') # for fridge 2 new pc

sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\ConfigurationClasses') # for fridge 1 new pc
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\InstrumentControl') # for fridge 1 new pc


from time import sleep, time
try:
    from qm.QuantumMachinesManager import QuantumMachinesManager
except: 
    from qm.quantum_machines_manager import QuantumMachinesManager
from qm.qua import *
from qm.octave import *
from qm.octave.octave_manager import ClockMode
from qm import SimulationConfig, CompilerOptionArguments
import numpy as np
import scipy as sp
from scipy import signal, fftpack
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as fig
from smart_fit import sFit as sft, fit_circle
from smart_fit import non_TimeDomain_fit
from scipy.optimize import curve_fit, minimize, Bounds, NonlinearConstraint
from time import time
from warnings import warn
from MXG5183A import MXG5183A
from AgilentPNA import AgilentPNA
from B2962A import B2962A
from CXA_SA import CXA_SA
from windfreak import SynthHD
import qutip as qt
from Compute_Density_Matrix import calculate_eigvals, MLE_density_matrix
from datetime import datetime
import pickle
import json
import types
# import visa, pyvisa
import pyvisa
from sklearn.cluster import KMeans

from remove_unpicklable import remove_unpicklable

import yaml
# Our Classes:
from ConfigurationCreator import Config as ConfigObject
from OneQubitTomography import OneQubitTomo 
from data_processing import data_to_sigma_z, round_value_by_error, scale_data_units, histogram_fidelity
from plotting import plot_2D, plot_fft, next_fig_num_by_name

#%%
class Time_domain_base(object):
    
    def __init__(self, Config, get_external = True, which_data = 'I', wait_between_seq = 120e3,
                 main_qubit = 'qb1', #The qubit/element you want meausre as the name in the dictionary 
                 scnd_qubit = None, #The qubit/element you want meausre as the name in the dictionary 
                 main_readout  = 'ro',
                 main_paramp = None, #The parametric amplifier pump element
                 ro_pulse = 'ro_pulse',#readout_pulse
                 pi2_pulse = 'pi2_pulse',
                 pi_pulse = 'pi_pulse',
                 pump_pulse = 'pump_pulse',
                 # check_overflow = False, 
                 is_pi_pulse = False, 
                 is_sb_cool = False,
                 host = None,
                 is_amplify = False,
                 is_thresh = False,
                 is_save_data = False,
                 save_folder = None,
                 is_calc_stat_error = False,
                 is_autoscale_data = True,
                 N_avg_for_thresh = 20000,
                 simulation_channels = [1,3,5],
                 is_get_opx = True,
                 is_get_ps = False,
                 fridge = None,
                 is_smooth_fit_plot = True,
                 is_active_reset = False,
                 is_loop_reset = True,
                 reset_qubits = None,
                 is_ro_cancellation = False,
                 cancellation_delay = 0,
                 parameters_dict = None,
                 wait_after_reset = 1000,
                 is_working_in_parallel = False,
                 is_add_explicit_align_at_end = False,
                 wait_if_busy_timeout = 3600,
                 **kwargs):
        
        if fridge is None:
            raise ValueError("You must pass fridge number. 1 for old or 2 for new.")
        
        self.results = {}
        if parameters_dict is not None:
            self.results['parameters'] = parameters_dict
            
        self.CXA = None
        self.get_external = get_external
        
        self.is_get_ps = is_get_ps
        
        self.simulation_channels = simulation_channels
        self.last_prog = None
        self.is_autoscale_data = is_autoscale_data
        
        self.sg_dict = None
        
        self.octave_config = Config.get('octave_config')
        
        if self.octave_config is not None: self.octave_name = next(iter(self.octave_config.get_devices()))
        
        
        if self.octave_config is not None:
            if fridge == 2:
                self.qm_server = QuantumMachinesManager(host = '192.168.0.31', cluster_name='Cluster_1', octave=self.octave_config)
            if fridge == 1:
                self.qm_server = QuantumMachinesManager(host = '192.168.0.33', cluster_name='Cluster_1', octave=self.octave_config)
            self.octave_clock(Config.octave_clock)
        else:
            if fridge == 2:
                self.qm_server = QuantumMachinesManager(host = '192.168.0.31', cluster_name='Cluster_1')
            if fridge == 1:
                # self.qm_server = QuantumMachinesManager(host = host, port = 9510)
                self.qm_server = QuantumMachinesManager(host = '192.168.0.33', cluster_name = 'Cluster_1')
                # self.qm_server = QuantumMachinesManager(host = '192.168.0.33', port = 9510)
        
            
        if save_folder is None: save_folder = pathlib.Path(__file__).resolve().parent /'saved_results'
        if is_save_data: self.generate_folder(save_folder)
        self.is_save_data = is_save_data
        self.save_folder = str(save_folder)
        self.is_smooth_fit_plot = is_smooth_fit_plot
        self.main_qubit = main_qubit
        self.scnd_qubit =scnd_qubit
        self.main_readout = main_readout
        self.main_paramp = main_paramp
        self.is_thresh = is_thresh
        self.is_calc_stat_error = is_calc_stat_error
        self.N_avg_for_thresh = N_avg_for_thresh
        self.is_amplify = is_amplify
        self.is_sb_cool = is_sb_cool
        self.is_active_reset = is_active_reset
        self.is_loop_reset = is_loop_reset
        self.loop_repetition = 1
        self.is_ro_cancellation = is_ro_cancellation
        self.cancellation_delay = cancellation_delay
        self.is_add_explicit_align_at_end = is_add_explicit_align_at_end
        
        self.pi2_pulse = pi2_pulse
        self.pi_pulse = pi_pulse   
        self.ro_pulse = ro_pulse
        self.pump_pulse = pump_pulse
        
        self.reset_qubits = reset_qubits
        self.readout_analyzer_dict = {'qubits':{}}
        self.is_sb_cool = is_sb_cool
        self.sb_cool_kwargs = {}
        
        self.is_working_in_parallel = is_working_in_parallel
        
        self.set_mxgs_flag = False
        
        self.wait_if_busy_timeout = wait_if_busy_timeout
        
        # Given config can be either an object from ConfigurationCreator or just the dict:        
        if is_get_opx: self.update_config(Config)

        self.WF = None
        
        
        self.which_data = which_data
        self.threshold = None
        self.threshold_time = 0

        #check if all the configurations are updated proprely 
        self.wait_between_seq = wait_between_seq
        #start the system
        if get_external: self.get_ext_inst()
        else: self.rm = None
        self.is_pi_pulse = is_pi_pulse  
        self.wait_after_reset = wait_after_reset  

        self.new_rot_dict = {
            '+I':(1,0,0,1, 0),
            '-I':(-1,0,0,-1, 0),
            '+X': (1,0,0,1, 2),
            '+x': (1,0,0,1, 1),
            '-x': (-1,0,0, -1, 1),
            '-X': (-1,0,0, -1, 2),
            '+y': (0,1,-1,0, 1),
            '-y': (0,-1,1,0, 1),
            '-Y': (0,-1,1,0, 2),
            }
        
        self.to_save_dict = {}
        
        

    def get_ext_inst(self, mxg_curr = None, cxa_curr = None, PNA_curr = None, PS_curr = None, WF_curr = None):
        
        try:self.rm = visa.ResourceManager()
        except: self.rm = pyvisa.highlevel.ResourceManager()
        
        self.set_all_mxgs()
        
        # if cxa_curr is None:
        #     cxa_curr = CXA_SA(resourceManager = self.rm)
        # self.CXA = cxa_curr

        if self.is_get_ps:
            try:
                self.PS = B2962A(NetworkAddress = 'TCPIP0::192.168.0.6::inst0::INSTR', channel = 1, resourceManager =  self.rm)
            except:
                print('Could not get PS')
        for element in self.configObject.SuperMembers.values():
            if element.current_source is not None:
                address = element.current_source[0]
                channel = element.current_source[1]
                current = element.current_source[2]
                try:
                    self.PS = B2962A.B2962A(NetworkAddress = address, channel = channel, resourceManager =  self.rm)
                    self.PS.curr()
                    # self.PS.set_volt_lim()
                    # self.PS.protect_on()
                    self.PS.set_curr(current)
                except:
                    print('Could not get PS')
        
    def set_PNA_by_element(self, element):
        self.PNA.set_sweep_type('CW')
        self.PNA.CwFreq(self.config['elements'][element]['mixInputs']['lo_frequency'])
        self.PNA.power(3)
        self.PNA.s_parameters(1,1,4)
        self.PNA.Triger_tO_cont()
        
    def set_cxa_to_element(self, element, vid_bw=1000, res_bw=1000, main_marker ='R', mul_span = 4, center = 'LO'):
        if self.CXA is None: self.CXA = CXA_SA(resourceManager = self.rm)
        
        LO_f = self.config['elements'][element]['mixInputs']['lo_frequency']
        IF_f = self.config['elements'][element]['intermediate_frequency']

        self.CXA.Freq.Span((mul_span*(np.abs(IF_f)+1e6)/1e6))
        if center == 'LO': 
            self.CXA.Freq.Zoom_Center(LO_f/1e6)
        elif center == 'USB':
            self.CXA.Freq.Zoom_Center(LO_f/1e6+np.abs(IF_f/1e6))
        elif center == 'LSB':
            self.CXA.Freq.Zoom_Center(LO_f/1e6-np.abs(IF_f/1e6))
        else:
            raise ValueError(f"Unkown center. Got {center}, expecter <LO>, <USB> or <LSB>.")
        self.CXA.Bandwidth.ResBW(res_bw)
        self.CXA.Bandwidth.VidBW(vid_bw)
     
        self.CXA.Marker.Marker_Mode(1, 'POS')
        self.CXA.Marker.Frequency(1, (LO_f-np.abs(IF_f))/1e6)
        self.CXA.Marker.Marker_Mode(2, 'POS')
        self.CXA.Marker.Frequency(2, LO_f/1e6)
        self.CXA.Marker.Marker_Mode(3, 'POS')
        self.CXA.Marker.Frequency(3, (LO_f+np.abs(IF_f))/1e6)
        
        if main_marker =='L':
            self.CXA.Marker.Marker_Mode(1, 'POS')
            self.CXA.Marker.Frequency(1, (LO_f-np.abs(IF_f))/1e6)
        elif main_marker=='C':
            self.CXA.Marker.Marker_Mode(2, 'POS')
            self.CXA.Marker.Frequency(2, LO_f/1e6)
    
    
    
    def start_wf(self, address, freq, power, ref):
        #check if wf is already initialized
        for sg_name, sg_subdict in self.sg_dict.items():
            
            if sg_subdict['type'] == 'wf':
                if sg_subdict['inst'] is None: continue
                if address[:4] == sg_subdict['inst']._devpath:
                    self.start_wf_channel(sg_subdict['inst'], int(address[-1]), freq, power)
                    return sg_subdict['inst']
        #if not, initialize it
        try:
            wf = SynthHD(address[:4])
            wf.init()
        except:
            raise ValueError('The WindFreak could not connect (port {})'.format(address[:4]))
        #set reference:        
        if ref in ['Int', 'int', 'internal', 'Internal']:
            wf.reference_mode = 'internal 10mhz'
            wf.reference_frequency = 10e6
        elif ref in ['Ext', 'ext', 'External', 'external']:
            wf.reference_mode = 'external'
            wf.reference_frequency = 10e6
            
        self.start_wf_channel(wf, int(address[-1]), freq, power)
        return wf
        
                    
    def start_wf_channel(self, wf, channel, freq, power):
        channel = int(channel)
        wf[channel].power = power
        wf[channel].frequency = freq
        wf[channel].enable = True
        if not wf[channel].calibrated:
            print(f"\n\n\n !! Warning !! \n The windfreak is unleveled with power of {power} and frequency {freq}. Please consider setting lower power.\n\n\n")
        if not wf[channel].lock_status:
            print("\n\n\n !! Warning !! \n The windfreak is not phase locked. Check the clock.\n\n\n")
                
    def set_all_mxgs(self):
        
        if self.sg_dict is None: self.sg_dict = dict()
        
        for sg_name, sg in self.configObject.mxgs.items():
            
            if sg.inst_type == 'mxg':
                self.sg_dict[sg_name] = dict(type = 'mxg',
                                             inst = self.start_mxg(sg.address, sg.lo_freq_actual, sg.power))
                if sg.is_using_octave_externally: 
                    for element, _ in sg.PairedMembers.items():
                        self.set_octave_lo_source(element, 'External')
                        # self.set_octave_external_lo_freq(element)
                        self.octave_switch(element, sg.trig_mode)
                        self.octave_gain(element, sg.gain)
                        octave_rf_in_port = self.configObject.SuperMembers[element].octave_rf_in_port
                        if octave_rf_in_port is not None:
                            self.qm.octave.set_qua_element_octave_rf_in_port(element, self.octave_name, octave_rf_in_port)
                            if octave_rf_in_port == 1: 
                                self.qm.octave.set_downconversion(element, lo_source=RFInputLOSource.Dmd1LO)
                            elif octave_rf_in_port == 2: 
                                self.qm.octave.set_downconversion(element, lo_source=RFInputLOSource.Dmd2LO)
            elif sg.inst_type == 'wf':
                if sg_name not in self.sg_dict:
                    self.sg_dict[sg_name] = dict(type = 'wf',
                                                 inst = self.start_wf(sg.address, sg.lo_freq_actual, sg.power, sg.ref))
                else:
                    self.start_wf_channel(self.sg_dict[sg_name]['inst'], int(sg.address[-1]), sg.lo_freq_actual, sg.power)
                if sg.is_using_octave_externally: 
                    for element, _ in sg.PairedMembers.items():
                        # self.set_octave_external_lo_freq(element)
                        self.set_octave_lo_source(element, 'External')
                        self.octave_switch(element, sg.trig_mode)
                        self.octave_gain(element, sg.gain)
                        octave_rf_in_port = self.configObject.SuperMembers[element].octave_rf_in_port
                        if octave_rf_in_port is not None:
                            self.qm.octave.set_qua_element_octave_rf_in_port(element, self.octave_name, octave_rf_in_port)
                            if octave_rf_in_port == 1: self.qm.octave.set_downconversion(element,lo_source=RFInputLOSource.Dmd1LO,
                                                                                         if_mode_i=IFMode.direct, if_mode_q=IFMode.direct)
                            elif octave_rf_in_port == 2: self.qm.octave.set_downconversion(element,lo_source=RFInputLOSource.Dmd2LO,
                                                                                           if_mode_i=IFMode.direct, if_mode_q=IFMode.direct)
                            
            elif sg.inst_type == 'octave':
                if len(self.configObject.mxgs[sg_name].PairedMembers) > 1:
                    Warn(f'Two or more elements are paired with mxg {sg_name} and the octave. You should make sure that their OPX output channel is also the same')
                element = next(iter(sg.PairedMembers))
                if not sg.is_using_octave_externally:
                    self.set_octave_lo_source(element, 'Internal')
                    self.set_octave_lo_freq(element, self.configObject.SuperMembers[element]._initDerivedLoFreq(1))
                else:
                    self.set_octave_lo_source(element, 'External')
                    # self.set_octave_external_lo_freq(element)
                    
                self.octave_switch(element, sg.trig_mode)
                self.octave_gain(element, sg.gain)
                octave_rf_in_port = self.configObject.SuperMembers[element].octave_rf_in_port
                if octave_rf_in_port is not None:
                    self.qm.octave.set_qua_element_octave_rf_in_port(element, self.octave_name, octave_rf_in_port)
                    if octave_rf_in_port == 1: self.qm.octave.set_downconversion(element,lo_source=RFInputLOSource.Internal,
                                                                                 if_mode_i=IFMode.direct, if_mode_q=IFMode.direct)
                    elif octave_rf_in_port == 2: self.qm.octave.set_downconversion(element,lo_source=RFInputLOSource.Dmd2LO,
                                                                                   if_mode_i=IFMode.direct, if_mode_q=IFMode.direct)
            else:
                raise ValueError(f"Unknown signal generator type. Got {sg.inst_type}")
     ##TODO turn off all MXG
     
    def set_master_slave(self, element1, element2):
        """First element will be master and second will be slave"""
        
        
        self.configObject.SuperMembers[element1].mxg = (self.configObject.SuperMembers[element1].mxg[0], self.configObject.SuperMembers[element1].mxg[0].MasterSlaveEnum.SLAVE) # looks like master and slave Enum are reversed in Nir's code
        self.configObject.SuperMembers[element2].mxg = (self.configObject.SuperMembers[element2].mxg[0], self.configObject.SuperMembers[element2].mxg[0].MasterSlaveEnum.MASTER) 
        self.update_config(self.configObject)
        self.set_all_mxgs()
        
    def start_mxg(self,networkaddress,freq,
                  pwr = 18,
                  ):#power in dBm
    
        mxg = MXG5183A(networkaddress, resourceManager = self.rm)
        
        mxg.set_power(pwr); mxg.set_freq(freq); mxg.on()
        
        return mxg
       
    def mxg_power_switch(self,power):
        for key in self.mxg_dic.keys():
            if power == 1 or power=='ON':
                self.mxg_dic[key].on()
            else:
                self.mxg_dic[key].off()
    
    def mxg_power_ON(self):
         self.mxg_power_switch(self,'ON')
         
    def mxg_power_OFF(self,power):
         self.mxg_power_switch(self,'OFF')
         
    # def set_JPA(self, setup):
    #     self.PS.set_curr_smooth(self.aux_config['elements']['paramp']['setups'][setup]['current'])
    #     self.mxg_dic[self.aux_config['elements']['paramp']['MXG']].set_freq(self.aux_config['elements']['paramp']['setups'][setup]['frequency'])
    #     self.mxg_dic[self.aux_config['elements']['paramp']['MXG']].set_power(self.aux_config['elements']['paramp']['setups'][setup]['power'])
    #     self.mxg_dic[self.aux_config['elements']['paramp']['MXG']].on()


    #%% general functions

    def serialize_prog(self, filename='debug', save_folder = None, prog = None):
        """ prog can be self.burst_detector_prog, self.burst_no_burst_prog, etc."""
        if save_folder is None: save_folder = self.save_folder
        if prog is None: prog = self.last_prog
        from qm import generate_qua_script
        sourceFile = open(f'{save_folder}\\{filename}.py', 'w')
        print(generate_qua_script(prog, self.config), file=sourceFile)
        sourceFile.close()

    def pickle_save(self, to_save_dict = None, meas_name = None, foldername = None, filename = None, time_of_meas = None, **kwargs):
        """Use **kwargs to add to the basic dictionary of the tido class"""
        
        if time_of_meas is None: time_of_meas = datetime.now().strftime("%d_%m_%Y___%H_%M_%S")
        
        if to_save_dict is None:
            to_save_dict = self.to_save_dict
        to_save_dict.update({'time_of_meas': time_of_meas})
        
        to_save_dict.update(kwargs)
        
        if foldername is None:
            if self.save_folder is not None:
                foldername = self.save_folder
            else:
                foldername = ''
        if not os.path.exists(foldername):                
            os.makedirs(foldername)
                    
        if meas_name is None:
            meas_name = 'unnamed_meas'
        to_save_dict.update({'meas_name': meas_name})
        
        if filename is None:
            filename = meas_name + '_' + time_of_meas
        
        filepath = foldername + '\\' + filename
        
        with open(f'{filepath}.pickle', 'wb') as handle:
            pickle.dump(remove_unpicklable(to_save_dict), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print('\n'+f'Saved data to {filepath}'+'\n')
        
    def pickle_load(self, filepath):
        if filepath[-7:] == '.pickle':
            with open(f'{filepath}', 'rb') as handle:
                loaded_dict = pickle.load(handle)
        else:
            with open(f'{filepath}.pickle', 'rb') as handle:
                loaded_dict = pickle.load(handle)
        self.results[loaded_dict['meas_name']] = loaded_dict
        if 'burst_no_burst' in loaded_dict['meas_name']:
            self.burst_no_burst_results = loaded_dict
        elif 'ro_spec' in loaded_dict['meas_name']:
            self.ro_spec_results = loaded_dict
        elif 'qp_lifetime' in loaded_dict['meas_name']:
            self.qp_lifetime_results = loaded_dict
        elif 'burst_detector_QM' in loaded_dict['meas_name']:
            self.burst_detector_QM_results = loaded_dict
            
        return loaded_dict
    
    def plot_from_file(self, filepath):
        loaded_file = self.pickle_load(filepath)
        if loaded_file['meas_name'] == 'driven_ramsey':
            self.fit_and_plot_driven_ramsey()
        elif loaded_file['meas_name'] == 'Rabi':
            self.plot_rabi()
        elif loaded_file['meas_name'] == 'ramsey':
            self.plot_ramsey()
        elif loaded_file['meas_name'] == 'ramsey_echo':
            self.plot_ramsey_echo()
        elif loaded_file['meas_name'] == 'T1':
            self.plot_T1()
        
  
    def run_prog(self, 
                 prog,
                 shape = -1,
                 duration_limit=0,
                 data_limit=0,
                 flags = None,
                 overrides_dict = None,
                 **kwargs):
        """Executes a program and returns t [ns], I(t)[V] and Q(t)[V]"""
        is_compiled = type(prog) is str
            
        my_compiler_options = CompilerOptionArguments(flags=flags)
        
        if self.is_working_in_parallel:
            if not self.is_qm_open():
                if self.qm_server.version_dict()['qm-qua'] == '1.2.2':
                    if len(self.qm_server.list_open_qms()) > 0:
                        self.qm = self.qm_server.get_qm(self.qm_server.list_open_qms()[0])
                        self.wait_if_busy()
                    self.qm = self.qm_server.open_qm(self.config)
                else:
                    if len(self.qm_server.list_open_quantum_machines()) > 0:
                        self.qm = self.qm_server.get_qm(self.qm_server.list_open_quantum_machines()[0])
                        self.wait_if_busy()
                    self.qm = self.qm_server.open_qm(self.config)
                
        self.qm_server.clear_all_job_results()
        if is_compiled:
            pending_job = self.qm.queue.add_compiled(prog, overrides=overrides_dict)
            job = pending_job.wait_for_execution()
        else:
            job = self.qm.execute(prog, duration_limit=duration_limit, data_limit=data_limit, compiler_options = my_compiler_options)
            self.last_prog = prog
        self.last_job = job
        try:
            job.result_handles.wait_for_all_values()
            job.execution_report()
        except KeyboardInterrupt:
            print('Canceled job')
            job.halt()
            raise KeyboardInterrupt
            
        
        if type(shape) is int:
            if shape != -1:
                shape = (shape,-1)
        
        I_res =     job.result_handles.get('I').fetch_all()['value'].reshape(shape)
        Q_res =     job.result_handles.get('Q').fetch_all()['value'].reshape(shape)
                
        return I_res , Q_res
    

    def fit_and_plot(self,
                     fittype: str,
                     data: list,
                     ti: list,
                     txt = None,
                     title_str = None,
                     start_x_fit = None,
                     is_calc_stat_error=None,
                     is_spec = False,
                     x_type = 't',
                     **kwargs):
        
        if is_calc_stat_error is None: is_calc_stat_error = self.is_calc_stat_error
        
        if is_calc_stat_error:
            stat_error = data[1]
            data = data[0]
        else:
            stat_error=None
            
        if x_type=='t':
            if start_x_fit is None:
                sft1 = None if fittype is None else sft(fittype, data, ti, error = stat_error)
            else:
                sft1 = None if fittype is None else sft(fittype, data[ti>=start_x_fit], ti[ti>=start_x_fit], error = stat_error)
            return self.plot_results(sft1, data, ti, txt = txt, title_str = title_str, error = stat_error,x_type = x_type, **kwargs)#,np.sqrt(np.diag(cov)),fit
        
        self.data = data
        self.x_axis = ti
        sft1 = None if fittype is None else non_TimeDomain_fit(fittype,data[start_x_fit:], ti[start_x_fit:], error = stat_error)
        
        return self.plot_non_time_domain_results(sft1, data, ti, txt = txt, title_str = title_str, error = stat_error,x_type = x_type, **kwargs)
        # sft1 = None if fittype is None else self.fit(fittype,data[ti>=start_x_fit], ti[ti>=start_x_fit]
    # def fit_non_time_domain(self,fittype,data,x_axis):
        
    def plot_non_time_domain_results(self, sft1, Y_axis, X_axis, 
                                     fig_num = None, txt = None, title_str = None, plot = True,
                                     subplot ='111', error = None, is_plot_fft = False, is_continue = False,
                                     is_print_alpha=False, label =  None, is_thresh=None, is_autoscale_data = None, x_type = 'amp',
                                     markersize = 6,**kwargs):#      
        if is_autoscale_data is None:
            is_autoscale_data = self.is_autoscale_data
        if is_autoscale_data: 
            scaled_X, X_units_prefix, X_scaling_factor = scale_data_units(X_axis) 
            scaled_Y, Y_units_prefix, Y_scaling_factor = scale_data_units(Y_axis)
        else:
            scaled_X, X_units_prefix, X_scaling_factor = X_axis, '', 1
            scaled_Y, Y_units_prefix, Y_scaling_factor = Y_axis, '', 1
            
        x_text = np.mean(scaled_X)
        y_text = max(scaled_Y)
        
        if is_thresh is None: is_thersh = self.is_thresh
       
        if sft1 is None:
                   if subplot[2]=='1':
                       if fig_num is None:
                           fig = plt.figure()
                       else:
                           fig = plt.figure(fig_num)
                           if is_continue: plt.clf()
                   ax = plt.subplot(int(subplot))
                   if error is not None: ax.errorbar(scaled_X, scaled_Y, yerr = Y_scaling_factor * error, fmt = '--or', capsize = 5, markersize = markersize, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1), label = label)
                   else: ax.plot(scaled_X, scaled_Y,'--or', markersize = markersize, mfc=(1,0,0,0.5), mec = (0,0,0,1), label = label)
                   return ax,X_scaling_factor,Y_scaling_factor
        

        fit = sft1.fit_params
        cov = sft1.fit_cov
        fit_fidelty = np.sqrt(np.diag(cov))   
        
             
        if plot:
            x_text = None
            if subplot[2]=='1':
                if fig_num is None:
                    fig = plt.figure()
                else:
                    fig = plt.figure(fig_num)
                    if is_continue: plt.clf()
            ax = plt.subplot(int(subplot))    
            
            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
#TODO calculte error in fit             
            if is_thresh: ax.set_ylabel(r'$\langle Z \rangle$')
            elif self.which_data == 'Phase': ax.set_ylabel('{1} [{0}Rad]'.format(Y_units_prefix, self.which_data))
            else: ax.set_ylabel('{1} [{0}V]'.format(Y_units_prefix, self.which_data))
              
            if error is not None: ax.errorbar(scaled_X, scaled_Y, yerr = Y_scaling_factor * error, fmt = '--or', capsize = 5, markersize = markersize, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1), label = label)
            else: ax.plot(scaled_X, scaled_Y,'--or', markersize = markersize, mfc=(1,0,0,0.5), mec = (0,0,0,1), label = label)

            if sft1.is_succeed:
                if sft1.fittype == 'Gaussian':
                    results = [abs(1/fit[2]),fit[1],fit[0],fit[3]]
                    fit_bars = [fit_fidelty[2]/fit[2]**2,fit_fidelty[1],fit_fidelty[0],fit_fidelty[3]]
                    if txt is None: txt =  r'$ \sigma $' +' =  {} $\pm$ {} \n $\mu$ = {} $\pm$ {}  \n $A$ = {} $\pm$ {} \n $C$ = {} $\pm$ {}'
                    txt = txt.format(*round_value_by_error(results[0],fit_bars[0]),*round_value_by_error(results[1],fit_bars[1]),*round_value_by_error(results[2],fit_bars[2]),*round_value_by_error(results[3],fit_bars[3]))
                    y_text =y_text/1.5
                    x_text = scaled_X[0]/1.5
                elif  sft1.fittype ==  'Cos':
                    y_text =y_text/1.5
                    x_text = scaled_X[0]/1.5
                    results = [fit[1],fit[2],fit[0],fit[3]]
                    fit_bars = [fit_fidelty[1],fit_fidelty[2],fit_fidelty[0],fit_fidelty[3]]
                    if txt is None: txt =  r'frequency' +' =  {} $\pm$ {} \n $\phi_0$ = {} $\pm$ {}  \n $A$ = {} $\pm$ {} \n $C$ = {} $\pm$ {}'
                    try:
                        txt = txt.format(*round_value_by_error(results[0],fit_bars[0]),*round_value_by_error(results[1],fit_bars[1]),*round_value_by_error(results[2],fit_bars[2]),*round_value_by_error(results[3],fit_bars[3]))
                    except:
                        txt =''
                elif  sft1.fittype ==  'Line':
                    y_text =y_text/1.5
                    x_text = scaled_X[0]/1.5
                    results = [fit[0],fit[1]]
                    fit_bars = [fit_fidelty[0],fit_fidelty[1]]
                    if txt is None: txt =  r'a' +' =  {} $\pm$ {} \n $C$ = {} $\pm$ {}'
                    txt = txt.format(*round_value_by_error(results[0],fit_bars[0]),*round_value_by_error(results[1],fit_bars[1]))
                elif  sft1.fittype ==  'Line180':
                    y_text =y_text/1.5
                    x_text = scaled_X[0]/1.5
                    results = [fit[0]]
                    fit_bars = [fit_fidelty[0]]
                    if txt is None: txt = '$C$ = {} $\pm$ {}'
                    txt = txt.format(*round_value_by_error(results[0],fit_bars[0]))
                elif  sft1.fittype ==  'GaussianCos':
                    y_text =y_text/1.5
                    x_text = scaled_X[0]/1.5
                    results = [fit[1],fit[2],fit[3],fit[0],fit[4]]
                    fit_bars = [fit_fidelty[1],fit_fidelty[2],fit_fidelty[3],fit_fidelty[0],fit_fidelty[4]]
                    if txt is None: txt =  r'frequency' +' =  {} $\pm$ {} \n $\sigma$ = {} $\pm$ {}  \n$\phi_0$ = {} $\pm$ {}  \n $A$ = {} $\pm$ {} \n $C$ = {} $\pm$ {}'
                    try:
                        txt = txt.format(*round_value_by_error(results[0],fit_bars[0]),*round_value_by_error(results[1],fit_bars[1]),*round_value_by_error(results[2],fit_bars[2]),*round_value_by_error(results[3],fit_bars[3]),*round_value_by_error(results[4],fit_bars[4]))
                    except:
                        txt =''

                # ann = plt.annotate( r'decay time =  {0} $\pm$ {1} [$\mu$s]'.format(*round_value_by_error(fit[2],fit_fidelty[2]), xy = (x_text,y_text), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
                if not txt is None:
                    plt.title(title_str)
                    ann = plt.annotate(txt, xy = (x_text,y_text), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
                    ann.draggable()
                    print(fit)
                    ax.plot(scaled_X, Y_scaling_factor * sft1.func(np.array(X_axis), *fit),'b')   
            else:
                fit         = None
                fit_fidelty = None
        
        return ax, fit,fit_fidelty, x_text,y_text

    def plot_results(self, sft1, Y_axis, X_axis, 
                     fig_num = None, x_type='t', txt = None, is_autoscale_data = None, subplot = 111, title_str = None, plot = True,
                     error = None, is_plot_fft = False, is_continue = False, is_print_alpha=False, is_smooth_fit_plot = None,
                     label =  None, is_thresh=None, markersize = 3, is_return_fit = False, is_annotate = True,
                     **kwargs):#additional_data=[]
 #TODO with ASAF+ELIYA split and add different but related treatment to different ploting schemes: time domain (ramsey.....), spectroscopy (number splitting), Calibration (pinopi,pi2_calibration....)  and defaults
#TODO     choose how  to plot,legend and text many similar experamints on different figures.
#TODO consider changing to work with axes rather than figs
        
        txt_add = ''  if txt is None else '\n' + txt
        
        if self.which_data == 'Phase':
            is_autoscale_data = False
        if is_smooth_fit_plot is None: is_smooth_fit_plot = self.is_smooth_fit_plot
        if is_autoscale_data is None: is_autoscale_data = self.is_autoscale_data
        if is_autoscale_data: 
            scaled_X, X_units_prefix, X_scaling_factor = scale_data_units(X_axis*(1e-9)) 
            scaled_Y, Y_units_prefix, Y_scaling_factor = scale_data_units(Y_axis)
        else:
            scaled_X, X_units_prefix, X_scaling_factor = X_axis, '', 1
            scaled_Y, Y_units_prefix, Y_scaling_factor = Y_axis, '', 1
            
        if  sft1 is None:
            if fig_num is None:
                fig = plt.figure()
            else:
                fig = plt.figure(fig_num)
                if is_continue: plt.clf()

            ax = plt.subplot(subplot)    
            if error is not None: ax.errorbar(scaled_X, scaled_Y, yerr = Y_scaling_factor * error, fmt = '--or', capsize = 5, markersize = markersize, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1), label = label)
            else: ax.plot(scaled_X, scaled_Y,'--or', markersize = markersize, mfc=(1,0,0,0.5), mec = (0,0,0,1), label = label)

            return ax
        
        fit, cov = sft1.get_fit_results()
        fit_fidelty = np.sqrt(np.diag(cov))
        
        if not sft1.internal_sfit is None: self.plot_results(sft1.internal_sfit, sft1.internal_sfit.trace, sft1.internal_sfit.time, txt = txt, title_str = title_str, error = None,is_print_alpha = True, label= label, plot = False,**kwargs)
     
        if plot:
            x_text = None
            if fig_num is None:
                fig = plt.figure()
            else:
                fig = plt.figure(fig_num)
                if is_continue: plt.clf()
                
            ax = plt.subplot(subplot)               
            if error is not None: ax.errorbar(scaled_X, scaled_Y, yerr = Y_scaling_factor * error, fmt = '--or', capsize = 5, markersize = markersize, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1), label = label)
            else: ax.plot(scaled_X, scaled_Y,'--or', markersize = markersize, mfc=(1,0,0,0.5), mec = (0,0,0,1), label = label)

            if sft1.is_succeed:
                if is_smooth_fit_plot:
                    X_fit = np.linspace(X_axis[0], X_axis[-1], len(X_axis)*100)
                    scaled_X_fit = X_fit * X_scaling_factor*(1e-9)
                    ax.plot(scaled_X_fit, Y_scaling_factor * sft1.func(X_fit, *fit),'b')
                else:
                    ax.plot(scaled_X, Y_scaling_factor * sft1.func(X_axis, *fit),'b')
            
            ax.set_xlabel("Time [{}s]".format(X_units_prefix)) 
            
            if is_thresh: ax.set_ylabel(r'$\langle Z \rangle$')
            elif self.which_data == 'Phase': ax.set_ylabel('{1} [{0}Rad]'.format(Y_units_prefix, self.which_data))
            else: ax.set_ylabel('{1} [{0}V]'.format(Y_units_prefix, self.which_data))
            
            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            
            if title_str is not None:
                  plt.title(title_str, fontsize=self.title_size)
        
        if  sft1.fittype ==  'Exp':
            if sft1.is_succeed:
                fit_bars =[fit_fidelty[1]*1e-3/fit[1]**2]
                results = [1e-3/fit[1]]

                txt = r'decay time = {0} $\pm$ {1} [$\mu$s]'.format(*round_value_by_error(results[0], fit_bars[0]))
                if is_print_alpha and sft1.is_succeed:
                    try: txt = txt + '\n' + r'$|\alpha |^2$'+' = {0} $\pm$ {1}'.format(*round_value_by_error(-fit[0], fit_fidelty[0]))
                    except: print('Something went wrong with error of |alpha|^2')

            else:
                txt=''
                results = [None]
                fit_bars = [None]
        
        elif  sft1.fittype ==  'Cos':
            if sft1.is_succeed:
                fit_bars =[fit_fidelty[1],max(scaled_Y)]
                results = [1e3*fit[1]]
                txt = r'frequency = {0} $\pm$ {1} [MHz]'.format(*round_value_by_error(results[0], fit_bars[0]))
                print(fit)
            else:
                fit_bars =[None,max(scaled_Y)]
                results = [None]
        elif sft1.fittype == 'ExpCos':
            if sft1.is_succeed:
                results = [1e-3/fit[2], 1e3*fit[1]]
                fit_bars = [fit_fidelty[2]*1e-3/fit[2]**2, 1e3*fit_fidelty[1]]

                txt = r'decay time = {0} $\pm$ {1} [$\mu$s]'.format(*round_value_by_error(results[0], fit_bars[0])) 
                txt = txt+'\n'+r'frequency = {0} $\pm$ {1} [MHz]'.format(*round_value_by_error(results[1], fit_bars[1]))
                # txt = txt+'\n'+r'Rabi power = {0}'.format(self.pulse_amp('qb1','rabi_pulse'))
                
            else: 
                results  = [None, None]
                fit_bars = [None, None]
                
        elif  sft1.fittype ==  'Exp(Exp)':
            if sft1.is_succeed:
                # txt = f'decay time =  {str(round((1e-3/fit[2]), 4))} {str(round((additional_data[1]), 4))} [$\mu$s] \n alpha = {str(round(np.sqrt(fit[1]),3))} ({str(round(np.sqrt(additional_data[0],3))}))'
                results = [1e-3/fit[2], fit[1]]#,*additional_data]#[1/kappa (in micro seconds),alpha**2, 1/kappa from fit of log(in micro seconds),alpha**2 from fit of log]
                fit_bars = [fit_fidelty[2]*1e-3/fit[2]**2, fit_fidelty[1]] #,*additional_data]#[1/kappa (in micro seconds),alpha**2, 1/kappa from fit of log(in micro seconds),alpha**2 from fit of log]
                
                txt = r'decay time =  {0} $\pm$ {1} [$\mu$s]'.format(*round_value_by_error(results[0], fit_bars[0]))
                txt =txt + '\n' + r'$|\alpha |^2$ = {0} $\pm$ {1}'.format(*round_value_by_error(results[1], fit_bars[1]))
                x_text = 3*np.mean(scaled_X)/2
            else:
                txt=''
                results  = [None, None]
                fit_bars = [None, None]
        if plot:
            if is_annotate: 
                txt += txt_add
                if x_text is None: x_text = np.mean(scaled_X)
                ann = plt.annotate(txt, xy = (x_text, max(scaled_Y)), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
                ann.draggable()
                
            #fig.show()
            plt.tight_layout()
            
        if is_plot_fft:
            if title_str is None:
                fft_title = 'FFT'
            else:
                fft_title = title_str + ' FFT'
            plot_fft(X_axis, Y_axis, title_str = fft_title, **kwargs)
            plt.tight_layout()
            #fig.show()
            

        if is_return_fit: return fit,fit_fidelty,None
        if plot: return results, fit_bars, ax
        return results,fit_bars,None
    
    def autoscale_data(self, data):
        return scale_data_units(data)
    
    def set_e_to_f(self, map_back):
        if self.e_to_f: print('e to f is already on'); pass
        self.e_to_f = True
        self.e_to_f_map_back = map_back
        self.main_qubit_g_to_e = self.main_qubit
        self.main_qubit = self.main_qubit + 'ef'
        print(f'main qubit is {self.main_qubit} now ')
    def cancel_e_to_f(self):
        self.e_to_f = False
        if self.main_qubit_g_to_e is not None:
            self.main_qubit = self.main_qubit_g_to_e
            self.main_qubit_g_to_e = None
        print(f'main qubit is {self.main_qubit} now ')

    def simulate_prog(self, prog =None, duration = 10000, channels = None, is_digital = False, flags = None, is_fft = False, ax = None, is_IF = True, element_list = None, **kwargs):
        """simulate the signal of the program. Can choose different channels. You might want to decrease reset time (wait_between_seq) so the simulation is faster"""
        if prog is None: prog = self.last_prog
        if channels is None: channels = self.simulation_channels
        duration = int(duration)
        job = self.qm_server.simulate(config = self.config, program = prog, simulate = SimulationConfig(
        duration=duration // 4,           # duration of simulation in units of 4ns
        include_analog_waveforms=True,    # include analog waveform names
        include_digital_waveforms=True,    # include digital waveform names
        ),
        compiler_options = CompilerOptionArguments(flags=flags))
        samples = job.get_simulated_samples()
        
        wfs = job.simulated_analog_waveforms()
        
        prog_name_time = 'Simulation'
        if ax is None:
            fig, axs = plt.subplots(len(channels), 1,num = next_fig_num_by_name(prog_name_time), sharex = True, squeeze = False)
            
        colors = plt.rcParams["axes.prop_cycle"]()
        if is_IF:
            for i in range(len(channels)):
                if ax is None: 
                    plt.sca(axs[i][0])
                    c = next(colors)["color"]
                    plt.plot(samples.con1.analog['{}'.format(channels[i])],'.-', color = c)
                else: 
                    plt.sca(ax)
                    plt.plot(samples.con1.analog['{}'.format(channels[i])],'.-')
                if is_digital:
                    plt.plot(samples.con1.digital['{}'.format(channels[i])],':')
                    plt.legend(['analog', 'digital'])
                plt.title('Channel {}'.format(channels[i]))
                plt.ylabel("Signal [V]")
            plt.xlabel("Time [ns]")
            plt.tight_layout()
        elif element_list != None:
            for i in range(len(channels)):
                if ax is None: 
                    plt.sca(axs[i][0])
                else: 
                    plt.sca(ax)
                c = next(colors)["color"]
                IF = self.element_IF(element_list[i])*1e-9
                t = np.linspace(0, len(samples.con1.analog['{}'.format(channels[i])])-1, len(samples.con1.analog['{}'.format(channels[i])]))
                I_DC = self.set_output_dc_offset_by_element(element_list[i], 'I')
                Q_DC = self.set_output_dc_offset_by_element(element_list[i], 'Q')
                inverse_corr_mat = np.linalg.inv(self.get_mixer_correction(element_list[i]))
                I = samples.con1.analog['{}'.format(channels[i])]-I_DC
                Q = samples.con1.analog['{}'.format(channels[i]+1)]-Q_DC
                I_corr = I*inverse_corr_mat[0,0]+Q*inverse_corr_mat[0,1]
                Q_corr = I*inverse_corr_mat[1,0]+Q*inverse_corr_mat[1,1]
                I_waveform = I_corr * np.cos(-2*np.pi*t*IF) - Q_corr * np.sin(-2*np.pi*t*IF)
                Q_waveform = I_corr * np.sin(-2*np.pi*t*IF) + Q_corr * np.cos(-2*np.pi*t*IF)
                plt.plot(t,I_waveform,'-', color = c)
                plt.plot(t,Q_waveform,'--', color = c)
                if is_digital:
                    plt.plot(samples.con1.digital['{}'.format(channels[i])],':')
                plt.ylabel("Signal [V]")
            plt.xlabel("Time [ns]")
            plt.tight_layout()
        else:
            raise ValueError('you must give element list for <is_IF> = False option. The list should be the same length as the channels list.')
        
        if is_fft:
            prog_name_freq = 'Simulation FFT'
            fig_fft, axs_fft = plt.subplots(len(channels), 1, num = next_fig_num_by_name(prog_name_freq), sharex = True, squeeze = False)
            for i in range(len(channels)):
                data = samples.con1.analog['{}'.format(channels[i])] + 1j * samples.con1.analog['{}'.format(channels[i]+1)]
                plot_fft(np.arange(0, duration), data, ax = axs_fft[i][0], lgd = 'analog{}'.format(channels[i]), **kwargs)
            lgd_freq = plt.legend()
            lgd_freq.set_draggable(True)
            
        return samples.con1
    
    def print_prog(self, prog =None, duration = 10000, time_start = 0, channels = None, is_digital = False, flags = None):
        """simulate the signal of the program. Can choose different channels. You might want to decrease reset time (wait_between_seq) so the simulation is faster"""
        if prog is None: prog = self.last_prog
        if channels is None: channels = self.simulation_channels
        if type(channels) is not list: channels = [channels]
        duration = int(duration)
        job = self.qm_server.simulate(config = self.config, program = prog, simulate = SimulationConfig(
        duration=duration // 4,           # duration of simulation in units of 4ns
        include_analog_waveforms=True,    # include analog waveform names
        include_digital_waveforms=True,    # include digital waveform names
        ),
        compiler_options = CompilerOptionArguments(flags=flags))
        wfs_dict = job.simulated_analog_waveforms()
        
        for ch in channels:
            print(f'channel {ch}\n')
            if str(ch) in wfs_dict['controllers']['con1']['ports'].keys():
                for wf_dict in wfs_dict['controllers']['con1']['ports'][str(ch)]:
                    if wf_dict['timestamp'] > time_start:
                        print(wf_dict['name'],' '
                              ,int(wf_dict['timestamp']-time_start),'-',int(wf_dict['timestamp']+wf_dict['duration']-1-time_start),
                              ', f = ',np.round(wf_dict['frequency']*1e-6,3), 'MHz',
                              ', phase = ', np.round(wf_dict['phase']*180/np.pi,3), 'deg',
                              '\n')
            else:
                print('No pulses\n')
        
    def process_data(self, data, is_mean = True, is_calc_stat_error = None,
                     mean_axis = 0,
                       is_thresh = None, #this is true when Data with processed and thresholed before this 
                       which_data = None,
                       **kwargs):
        """Processes the data.
        If data is a list of I and Q, automatically calls determine_data(I,Q)
        Can average the data.
        Can return the statistical error.
        Can threshold the data"""
        if len(data) == 2 and type(data) is list: data = self.determine_data(data[0], data[1], which_data = which_data)
        if type(data) is dict: data = self.determine_data(data['I'], data['Q'], which_data = which_data)
        
        if is_thresh is None: is_thresh = self.is_thresh
        if is_thresh: data = self.thresh_data(data, **kwargs)

        if is_calc_stat_error is None: is_calc_stat_error = self.is_calc_stat_error
        if is_calc_stat_error: stat_error = sp.stats.sem(data, axis = mean_axis)
        else: stat_error = None
        
        if is_mean: data = data.mean(mean_axis)
        
        return data, stat_error
    
    def determine_data(self, I, Q, which_data = None):
        """
        takes all the data and returns the appropriate data according to self.which_data
        """
        if which_data is None: which_data = self.which_data
        if which_data == 'I': data = I
        elif which_data == 'Q': data =  Q
        elif which_data in ['Mag', 'mag']: data = np.sqrt(I**2+Q**2)
        elif which_data in ['Phase', 'phase']: data = np.arctan2(Q,I)
        elif np.isreal(which_data): data = I * np.cos(which_data) + Q * np.sin(which_data)
        else:
            raise ValueError('No such data type! Must be one of ["I","Q","Mag","Phase", a float representing a phase]')
        return data
        
   
    
    def calibrate_readout(self, qubit, ro, I, Q, N_avg = 1, is_sb_cool = False, **kwargs):
        nn = declare(int)
        if N_avg < 1:
            N_avg = 1
        with for_(nn, 0, nn<int(N_avg), nn+1):
            play('pi_pulse', qubit)
            align(qubit, ro)
            self.perform_full_measurement(I,Q, I_output_name = 'I_calib_pi', Q_output_name = 'Q_calib_pi', is_sb_cool = is_sb_cool, **kwargs)
            align(qubit, ro)
            self.perform_full_measurement(I,Q, I_output_name = 'I_calib_nopi', Q_output_name = 'Q_calib_nopi', is_sb_cool = is_sb_cool, **kwargs)
            align(qubit, ro)
            
        
    # def thresh_data(self, data, ref, is_scale_probability = True):
    #     """Assumes ref is a set of distributions, that do not have to be normalized. Returns the data sorted into the most likely set."""
    #     n_ref_smpls = len(ref[0])
    #     n_data_smpls = data.shape[0]
    #     mean_ref = ref.mean(0)
        
    #     if mean_ref[0] > mean_ref[-1]:
    #         mean_ref = -mean_ref
    #         data = -data
    #         ref = -ref
            
    #     th = [sum(mean_ref[i:i+1]) for i in range(mean_ref-1)]
        
        
    #     ref_counts = np.zeros(len(ref[1]))
    #     for i, ref_set in enumerate(ref.transpose()):
    #         if i==0:
    #             ref_counts[i] = sum(r < betas[i] for r in ref_set)
    #         elif i==len(ref[1])-1:
    #             ref_counts[i] = sum(r > betas[i] for r in ref_set)
    #         else:
    #             ref_counts[i] = sum((r < betas[i+1] and r > betas[i-1]) for r in ref_set)
                
                
    #     if len(data.shape)==2:
    #         data_counts_array = np.zeros([data.shape[1], len(ref[1])])
    #         for data_set, data_counts in zip(data.transpose(), data_counts_array):
    #             for i in range(len(data_counts)):
    #                 if i==0:
    #                     data_counts[i] += sum(d < betas[i] for d in data_set)
    #                 elif i==len(ref[1])-1:
    #                     data_counts[i] += sum(d > betas[i] for d in data_set)
    #                 else:
    #                     data_counts[i] += sum((d < betas[i] and d > betas[i-1]) for d in data_set)
            
    #     probabilities0 = ref_counts/n_ref_smpls
        
    #     populations = (data_counts_array/len(data))
    #     if is_scale_probability:
    #         populations = populations / probabilities0
            
    #     return populations
        
        
    def project_to_distribution(self, data, ref):
        
        num_of_bins = 200
        bins = np.histogram(np.hstack([data[~np.isnan(data)].flatten(),ref.flatten()]), bins=num_of_bins, density = True)[1]
        # bins_w = bins[1]-bins[0]
        
        ref_hist_list = np.zeros([len(ref), num_of_bins])
        
        for i, ref_set in enumerate(ref):
            ref_hist_list[i,:] = np.histogram(ref_set, bins = bins, density = True)[0]
        
        ref_hist_list = ref_hist_list.transpose()
        
        hist_data_list = np.zeros([len(data), num_of_bins])
        for i,data_set in enumerate(data):
            hist_data_list[i,:] = np.histogram(data_set, bins = bins, density = True)[0]
        
        
        prob = np.zeros([len(data), len(ref)])
        
        
        
        def sum_dist(v):
            return np.sum(v * ref_hist_list, axis = 1)
        
        def fit_dist(bins, a,b,c,d):
            return sum_dist(np.array([a,b,c,d]))
        
        for i,hist_data in enumerate(hist_data_list):
            fit,cov = curve_fit(fit_dist, bins, hist_data)
            prob[i] += fit
            
        return prob
            
        # N =  np.linalg.norm(ref_hist_list, axis = 1)**2
        # proj_coeffs = np.tensordot(ref_hist_list, ref_hist_list, axes = 1) / N
        # # proj_coeffs = proj_coeffs/np.max(proj_coeffs)
        # probability = np.tensordot(hist_data_list, ref_hist_list, axes = 1)
        # probability = probability / N
        # probability = probability - np.mean(proj_coeffs, axis = 1) 
        # # probability = probability/np.max(probability)
        
        # p = np.tensordot(probability, proj_coeffs, axes = 1)/4
        
        # np.tensordot(proj_coeffs, proj_coeffs, axes = 1)/4
        
        # probability0 = 
        
        # probability = probability  (np.max(probability))
        
        # return probability
        
    def find_thresh(self):
        self.load_pinopi(N_avg = self.N_avg_for_thresh,is_sliced = False)
        mean_g, mean_e, _ = self.run_pinopi()
        return mean_g, mean_e, np.mean([mean_g,mean_e])
                
            
    def run_continuous(self, element, pulse ='mixer_cal_pulse', run = True, detuning = 0):
        if type(element) != list:
            element = [element]
            
        with program() as prog:
            if detuning != 0: update_frequency(element, self.element_IF(element[0])+detuning)
            with infinite_loop_():
                for el in element:
                    play(pulse, el)
        if run:
            self.qm_server.clear_all_job_results()            
            return self.qm.execute(prog, duration_limit=0, data_limit=0, force_execution=True)
        else:
            return prog         
    
    def compare_config2opx(self):
        
        return {'in_self_config':self.compare_dic(self.config, self.qm.get_config()),'from_opx':self.compare_dic(self.qm.get_config,self.config())}
        
    def compare_dic(self,x,y):
        return  {k: x[k] for k in x if not(k in y) or not( x[k] == y[k])}
       
        
#%% Set/Update dictionary or Atrributes  functions
#%% general to all elements
    def element_freq(self,element, freq=None, update=True, suppress_msg=False):  
        
        if self.configObject.SuperMembers[element].mxg[1]==2:
            if not suppress_msg:
                print("\n\n"+'='*50+f"\nThis element <{element}> is enslaved. Cant change the LO frequency. You must first set it to be the master and the current master to be slave using set_master_slave.\n\n"+'='*60)
        #freq in Hz
        if freq is None:
            return self.configObject.SuperMembers[element].ElementParams['freq']
        else:
            if not self.get_external:
                raise ValueError('Can\'t change LO value on external machine')
        
        self.configObject.SuperMembers[element].ElementParams['freq'] = freq
        
        self.set_mxgs_flag = True
        if update: 
            self.update_config(self.configObject)
    
    def element_IF(self,element, freq = None, is_change_element_freq = True, update = True):   
        #freq in Hz
        if freq is None:
            return self.configObject.SuperMembers[element].ElementParams['intermediateFreq']
        
        if is_change_element_freq: self.element_freq(element, freq = self.element_freq(element) + freq - self.element_IF(element), update =False)
        else: self.set_mxgs_flag = True
        self.configObject.SuperMembers[element].ElementParams['intermediateFreq'] = freq
        
        if update: 
            self.update_config(self.configObject)
            
    def print_pulse(self,element,pulse,is_print = True ):        
        
        pulTyp = self.configObject.SuperMembers[element].PulseParamsDict[pulse].pulseType 
        txt = '*********** \n' + pulse + ' is of type ' +pulTyp +'\n *********** \n'
        
        if pulTyp =='gaussian':
            txt += f'Amp:   {self.pulse_amp(element, pulse)},  isCutTail:       {not (self.configObject.SuperMembers[element].PulseParamsDict[pulse].isCutTail is None)} \n'
            txt += f'Sigma: {self.pulse_sig(element, pulse)} [ns],  time_multiplier: {self.pulse_time_multiplier(element, pulse)}'
        elif pulTyp == 'constant':
            txt += f'Amp: {self.pulse_amp(element, pulse)},  duration: {self.configObject.SuperMembers[element].PulseParamsDict[pulse].length} [ns]'
        else:
            txt += f'duration: {self.configObject.SuperMembers[element].PulseParamsDict[pulse].length} [ns]'
        
        if is_print:
            print(txt)
        else:
            return txt                
   
    def plot_pulse(self,element, pulse, is_ramp = None, is_param_text = False, fig_num = None, is_fixed_pnt = True):
        
        fig = self.configObject.SuperMembers[element].plotPulse(pulse,fig_num=fig_num)
        
        if is_param_text: 
            txt = plt.annotate(self.print_pulse(element, pulse, is_print = False ), (0.05,0.6), xycoords = 'axes fraction')
            txt.draggable() 
        
        return fig
    
    def plot_pulse_fft(self,element,pulse,**kwargs):
        _, IWaveform, QWaveform, _ = self.configObject.SuperMembers[element]._generatePulseAndWaveforms( pulse )


        pulseLength = self.pulse_len(element, pulse)
        def fix_signal(sig):
            # get length:
            # ==========
            try: 
                sigLength = len(sig)
            except:
                sigLength = 1
            # fix signal:
            # ==========
            if   sigLength==pulseLength:                    pass
            elif sigLength > pulseLength:                   raise ValueError(f"Length of signal = {sigLength} > {pulseLength} = length of pulse")
            elif sigLength < pulseLength and sigLength==1:  sig = np.array( [sig] * pulseLength )
            else:                                           raise ValueError(f"Can't fix signal of length = {sigLength} > 1")
            # End:
            # ===
            return sig
        ti = np.arange(pulseLength)
        data_dic = dict(I={'data':fix_signal(IWaveform.sample),'times': ti}, Q={'data':fix_signal(QWaveform.sample),'times': ti})
        
        plot_fft(data_dic =data_dic,**kwargs)
        
        return fig
    
    def sticky(self, element, stickyInput = None, update = True):
        if stickyInput is None:
            return self.configObject.SuperMembers[element].element.stickyInput
        
        
        if stickyInput == "analog": 
            stickyInput = {'analog':True, 'digital':False, 'duration': 4}
        elif stickyInput is False:
            stickyInput = None
        elif type(stickyInput) != dict:
            raise ValueError("Invalid stickyInput. Should be either <'analog'> or a dictionary as in QUA documentation entry for sticky element.")
        
        self.configObject.SuperMembers[element].ElementParams['stickyInput'] = stickyInput
        self.configObject.SuperMembers[element].is_update = True
        if update: self.update_config(self.configObject)
    
    def pulse_tail(self, element, pulse, is_cut_tail = None, update=True):
        if is_cut_tail is None:
            return self.configObject.SuperMembers[element].PulseParamsDict[pulse].isCutTail
        else:
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].isCutTail = is_cut_tail
        if update: self.update_config(self.configObject)
        
    def pulse_amp(self, element, pulse, amp=None, keep_phase = False, update=True):
        """ Get or set the amplitude of a pulse for an element. \n
            to change the amplitude but keep the phase as is send a real amplitude and choose keep_phase = True"""
        if amp is None:
            return self.configObject.SuperMembers[element].PulseParamsDict[pulse].amp
        if type(amp) is not np.ndarray:
            if np.imag(amp) == 0 and keep_phase: 
                amp *= np.exp(1j * np.angle(self.configObject.SuperMembers[element].PulseParamsDict[pulse].amp, deg=False))
        self.configObject.SuperMembers[element].PulseParamsDict[pulse].amp=amp
        
        if update: self.update_config(self.configObject)
        
    def pulse_ramp_up_len(self, element, pulse, length=None, update=True):
        """ Get or set the length of a pulse ramp up for an element."""
        if length is None:
            if self.configObject.SuperMembers[element].PulseParamsDict[pulse].Additions.ramp['up'] is not None:
                length = self.configObject.SuperMembers[element].PulseParamsDict[pulse].Additions.ramp['up']['length']
            else:
                length = self.configObject.SuperMembers[element].PulseParamsDict[pulse].ramp_up_length
            if length is None:
                raise ValueError(f"Pulse <{pulse}> of element <{element}>has no ramp defined.")
            return length
            
        if self.configObject.SuperMembers[element].PulseParamsDict[pulse].Additions.ramp['up'] is not None:
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].Additions.ramp['up']['length'] = length
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].is_update = True
        else:
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].ramp_up_length=length
        
        if update: self.update_config(self.configObject)
        
    def pulse_ramp_down_len(self, element, pulse, length=None, update=True):
        """ Get or set the length of a pulse ramp down for an element."""
        if length is None:
            if self.configObject.SuperMembers[element].PulseParamsDict[pulse].Additions.ramp['down'] is not None:
                return self.configObject.SuperMembers[element].PulseParamsDict[pulse].Additions.ramp['down']['length']
            else:
                return self.configObject.SuperMembers[element].PulseParamsDict[pulse].ramp_down_length
            
        if self.configObject.SuperMembers[element].PulseParamsDict[pulse].Additions.ramp['down'] is not None:
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].Additions.ramp['down']['length'] = length
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].is_update = True
        else:
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].ramp_down_length=length
        
        if update: self.update_config(self.configObject)
        
    def pulse_ramp_len(self, element, pulse, length=None, update=True):
        """ Set the length of a pulse ramp up and down for an element."""
        
        if length is not None :
            self.pulse_ramp_down_len(element, pulse, length, update = update)
            self.pulse_ramp_up_len(element, pulse, length, update = update)
        
        else :
            return self.pulse_ramp_down_len(element, pulse, length, update = update) # Assume ramp down and ramp up length are same
        
        if update: self.update_config(self.configObject)
        
    def pulse_ramp_type(self, element, pulse, ramp_type = None, update=True):
        """ Get or the type of the ramp up for a pulse of an element. Can be <erf> or <sin2>"""
        
        if ramp_type is None :
            return self.configObject.SuperMembers[element].PulseParamsDict[pulse].Additions.ramp['up']['ramp_type'], self.configObject.SuperMembers[element].PulseParamsDict[pulse].Additions.ramp['down']['ramp_type']
        else:
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].Additions.ramp['down']['ramp_type'] = ramp_type
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].Additions.ramp['up']['ramp_type'] = ramp_type
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].is_update = True
        
        if update: self.update_config(self.configObject)
        
   
    def pulse_phase(self, element, pulse, phase=None, is_in_deg = True, update=True):
        """ Get or set the phase of a pulse for an element.
            Pass phase in degrees"""
        if phase is None:
            return np.angle(self.configObject.SuperMembers[element].PulseParamsDict[pulse].amp, deg=is_in_deg)
        self.configObject.SuperMembers[element].PulseParamsDict[pulse].amp = np.abs(self.configObject.SuperMembers[element].PulseParamsDict[pulse].amp)*np.exp(1j*phase*np.pi/180)
        
        if update: self.update_config(self.configObject)
        
    def pulse_drag(self, element, pulse, drag_param=None, update=True):
        """ Get or set the DRAG parameter of a pulse for an element"""
        if drag_param is None:
            return self.configObject.SuperMembers[element].PulseParamsDict[pulse].drag_param
        
        self.configObject.SuperMembers[element].PulseParamsDict[pulse].drag_param = drag_param
        
        if update: self.update_config(self.configObject)
        
    def pulse_len(self, element, pulse, pulse_len=None, update=True):
        """ Get or set the length of a pulse for an element"""
        
        # if self.configObject.SuperMembers[element].PulseParamsDict[pulse].length is None:
        #     raise ValueError('length not defined for this pulse')

        if pulse_len is None:
            if self.configObject.SuperMembers[element].PulseParamsDict[pulse].length is None:
                return self.pulse_sig(element,pulse)*self.pulse_time_multiplier(element,pulse)
            return self.configObject.SuperMembers[element].PulseParamsDict[pulse].length

                
        self.configObject.SuperMembers[element].PulseParamsDict[pulse].length=int(pulse_len)
        
        if update: self.update_config(self.configObject)
        
    def pulse_type(self, element, pulse, pulse_type=None, update=True):
        
        if pulse_type is None:
            return self.configObject.SuperMembers[element].PulseParamsDict[pulse].pulseType
                
        self.configObject.SuperMembers[element].PulseParamsDict[pulse].pulseType=pulse_type
        
        if update: self.update_config(self.configObject)
                                      
    def pulse_sig(self,element, pulse, sig=None, update=True):
        """ Get or set the sigma (standard deviation of the Gaussian) of a pulse for an element"""
        if self.configObject.SuperMembers[element].PulseParamsDict[pulse].sigma is None:
            raise ValueError('sigma not defined for this pulse')
            
        if sig is None:
            return self.configObject.SuperMembers[element].PulseParamsDict[pulse].sigma
        self.configObject.SuperMembers[element].PulseParamsDict[pulse].sigma=sig
        
        if update: self.update_config(self.configObject)    
        
    def pulse_detuning(self,element, pulse, det=None, update=True):
        """ Get or set the detuning (for fast conditional displacment) of a pulse for an element"""
        if self.configObject.SuperMembers[element].PulseParamsDict[pulse].detuning is None:
            raise ValueError('sigma not defined for this pulse')
            
        if det is None:
            return self.configObject.SuperMembers[element].PulseParamsDict[pulse].detuning
        self.configObject.SuperMembers[element].PulseParamsDict[pulse].detuning=det
        
        if update: self.update_config(self.configObject)    

    def pulse_frequency(self,element, pulse, freq=None, update=True):
        """ Get or set the frequencey (for fast conditional displacment) of a pulse for an element"""
        if self.configObject.SuperMembers[element].PulseParamsDict[pulse].frequency is None:
            raise ValueError('frequency not defined for this pulse')
            
        if freq is None:
            return self.configObject.SuperMembers[element].PulseParamsDict[pulse].frequency
        self.configObject.SuperMembers[element].PulseParamsDict[pulse].frequency=freq
        
        if update: self.update_config(self.configObject)    

        
    def pulse_time_multiplier(self,element,pulse,time_multiplier = None, update=True):
        """ Get or set the time multiplier of a pulse for an element"""
        if self.configObject.SuperMembers[element].PulseParamsDict[pulse].time_multiplier is None:
            raise ValueError('time multiplier not defined for this pulse')
            
        if time_multiplier is None:
            return self.configObject.SuperMembers[element].PulseParamsDict[pulse].time_multiplier
        self.configObject.SuperMembers[element].PulseParamsDict[pulse].time_multiplier=time_multiplier
        
        if update: self.update_config(self.configObject)
        
    def prepare_pulses_for_tomo(self, qubit = None):
        if qubit is None: qubit = self.main_qubit
        pi2_amp = self.pulse_amp(qubit, 'pi2_pulse')
        pi_amp = self.pulse_amp(qubit, 'pi_pulse')
        pi2_drag = self.pulse_drag(qubit, 'pi2_pulse')
        pi_drag = self.pulse_drag(qubit, 'pi_pulse')
        
        
        
        self.pulse_amp(qubit, 'x', pi2_amp, update = False)
        self.pulse_amp(qubit, '-x', -pi2_amp, update = False)
        self.pulse_amp(qubit, '-y', -np.around(pi2_amp*np.exp(np.pi*1j/2),9), update = False)
        self.pulse_amp(qubit, 'y', np.around(pi2_amp*np.exp(np.pi*1j/2),9), update = False)
        
        self.pulse_drag(qubit, 'x', pi2_drag, update = False)
        self.pulse_drag(qubit, '-x', pi2_drag, update = False)
        self.pulse_drag(qubit, '-y', pi2_drag, update = False)
        self.pulse_drag(qubit, 'y', pi2_drag, update = False)
        
        self.pulse_amp(qubit, 'X', pi_amp, update = False)
        self.pulse_amp(qubit, '-X', -pi_amp, update = False)
        self.pulse_amp(qubit, '-Y', -np.around(pi_amp*np.exp(np.pi*1j/2),9), update = False)
        self.pulse_amp(qubit, 'Y', np.around(pi_amp*np.exp(np.pi*1j/2),9), update = True)
        
        self.pulse_drag(qubit, 'X', pi_drag, update = False)
        self.pulse_drag(qubit, '-X', pi_drag, update = False)
        self.pulse_drag(qubit, '-Y', pi_drag, update = False)
        self.pulse_drag(qubit, 'Y', pi_drag, update = False)
        
        if self.pulse_type(qubit, 'pi_pulse') == 'gaussian':
            sigma = self.pulse_sig(qubit, 'pi_pulse')
            tmult = self.pulse_time_multiplier(qubit, 'pi_pulse')
            self.pulse_sig(qubit, 'x', sigma, update = False)
            self.pulse_sig(qubit, '-x', sigma, update = False)
            self.pulse_sig(qubit, '-y', sigma, update = False)
            self.pulse_sig(qubit, 'y', sigma, update = False)
            self.pulse_sig(qubit, 'X', sigma, update = False)
            self.pulse_sig(qubit, '-X', sigma, update = False)
            self.pulse_sig(qubit, '-Y', sigma, update = False)
            self.pulse_sig(qubit, 'Y', sigma, update = False)
            
            self.pulse_time_multiplier(qubit, 'x', tmult, update = False)
            self.pulse_time_multiplier(qubit, '-x', tmult, update = False)
            self.pulse_time_multiplier(qubit, '-y', tmult, update = False)
            self.pulse_time_multiplier(qubit, 'y', tmult, update = False)
            self.pulse_time_multiplier(qubit, 'X', tmult, update = False)
            self.pulse_time_multiplier(qubit, '-X', tmult, update = False)
            self.pulse_time_multiplier(qubit, '-Y', tmult, update = False)
            self.pulse_time_multiplier(qubit, 'Y', tmult, update = False)
        elif self.pulse_type(qubit, 'pi_pulse') == 'sin2':
            
            length = self.pulse_len(qubit, 'pi_pulse')
            self.pulse_len(qubit, 'x', length, update = False)
            self.pulse_len(qubit, '-x', length, update = False)
            self.pulse_len(qubit, '-y', length, update = False)
            self.pulse_len(qubit, 'y', length, update = False)
            self.pulse_len(qubit, 'X', length, update = False)
            self.pulse_len(qubit, '-X', length, update = False)
            self.pulse_len(qubit, '-Y', length, update = False)
            self.pulse_len(qubit, 'Y', length, update = False)
        
        self.update_config()
        
    def sidebands_pulse(self, element, pulse = 'sidebands_pulse',
                        amp = None, ramp_up_length = None, ramp_down_length = None, length = None,
                        upper_frequency = None, lower_frequency = None, lower_amp = None, upper_amp = None, phase_diff = None,
                        update = True):
        """Get or set the frequency and amp of sidebands pulse """
        if amp is None:
            print(f'Amp is {self.configObject.SuperMembers[element].PulseParamsDict[pulse].amp}')
        else:
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].amp = amp
        if upper_frequency is None:
            print(f'Upper frequency is {self.configObject.SuperMembers[element].PulseParamsDict[pulse].upper_frequency}')
        else:
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].upper_frequency = upper_frequency
        if lower_frequency is None:
            print(f'Lower frequency is {self.configObject.SuperMembers[element].PulseParamsDict[pulse].lower_frequency}')
        else:
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].lower_frequency = lower_frequency
        if upper_amp is None:
            print(f'Upper amp is {self.configObject.SuperMembers[element].PulseParamsDict[pulse].upper_amp}')
        else:
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].upper_amp = upper_amp
        if lower_amp is None:
            print(f'Lower amp is {self.configObject.SuperMembers[element].PulseParamsDict[pulse].lower_amp}')
        else:
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].lower_amp = lower_amp
        if phase_diff is None:
            print(f'Phase difference is {self.configObject.SuperMembers[element].PulseParamsDict[pulse].phase_diff}')
        else:
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].phase_diff = phase_diff
        if ramp_up_length is None:
            print(f'Ramp up length is {self.configObject.SuperMembers[element].PulseParamsDict[pulse].ramp_up_length}')
        else:
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].ramp_up_length = ramp_up_length
        if ramp_down_length is None:
            print(f'Ramp down length is {self.configObject.SuperMembers[element].PulseParamsDict[pulse].ramp_down_length}')
        else:
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].ramp_down_length = ramp_down_length
        if length is None:
            print(f'Length is {self.configObject.SuperMembers[element].PulseParamsDict[pulse].length}')
        else:
            self.configObject.SuperMembers[element].PulseParamsDict[pulse].length = int(length)
            
        if amp is not None or upper_frequency is not None or lower_frequency is not None or upper_amp is not None or lower_amp is not None or ramp_down_length is not None or ramp_up_length is not None or length is not None:
            if update: self.update_config(self.configObject)
    
    def sidebands_pulse_phase_diff(self,element, pulse, phase_diff=None, update=True):
        """ Get or set the frequencey (for fast conditional displacment) of a pulse for an element"""
        if self.configObject.SuperMembers[element].PulseParamsDict[pulse].phase_diff is None:
            raise ValueError('phase_diff not defined for this pulse')
            
        if phase_diff is None:
            return self.configObject.SuperMembers[element].PulseParamsDict[pulse].phase_diff
        self.configObject.SuperMembers[element].PulseParamsDict[pulse].phase_diff=phase_diff
        
        if update: self.update_config(self.configObject)   
        
    def pulse_no_drive_len(self, element, pulse, no_drive_length = None, update = True):
        """ Get or set the no drive lenght of a ring down pulse for an element"""
        if self.configObject.SuperMembers[element].PulseParamsDict[pulse].ring_down_no_drive_length is None:
            raise ValueError('ring_down_no_drive_length not defined for this pulse')
            
        if no_drive_length is None:
            return self.configObject.SuperMembers[element].PulseParamsDict[pulse].ring_down_no_drive_length
        self.configObject.SuperMembers[element].PulseParamsDict[pulse].ring_down_no_drive_length=no_drive_length
        
        if update: self.update_config(self.configObject)
        
    def get_port_by_element(self, element, channel = 'I'):
        return self.configObject.SuperMembers[element].ElementParams[channel+'Con']

    def set_output_dc_offset_by_element(self, element, channel, offset = None, update = True):
        if offset is None: return self.configObject.controllers.analog_outputs[self.get_port_by_element(element, channel)]
        self.qm.set_output_dc_offset_by_element(element, channel, offset)
        self.configObject.controllers.analog_outputs[self.get_port_by_element(element, channel)] = offset
        self.configObject.controllers.is_update = True
        if update: self.update_config(self.configObject)
        
    
    def set_input_dc_offset_by_element(self, element, channel, offset = None, update = True):
        if offset is None: return self.configObject.controllers.analog_inputs[self.get_port_by_element(element, channel)]
        self.configObject.controllers.analog_inputs[self.get_port_by_element(element, channel)] = offset
        self.configObject.controllers.is_update = True
        if update: self.update_config(self.configObject)
        
        
    def output_filter(self, element, channel, feedforward = [], feedback = []):
        if feedforward is None and feedback is None:
            return feedforward, feedback
        else:
            port = self.get_port_by_element(element = element, channel = channel)
            self.configObject.controllers.output_filters[port] = [feedforward, feedback]
            self.configObject.controllers.is_update = True
            self.update_config(self.configObject)
            
  #IW =intergtation weights  set get functions   
    def IW_I_amp(self,I_amp=None, ro_element= None,ro_pulse = None,update =True):
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        integrationWeights = self.configObject.SuperMembers[ro_element].PulseParamsDict[ro_pulse].integrationWeights
        if I_amp is None: return integrationWeights.I_amp
        integrationWeights.I_amp=I_amp
        if update: self.update_config(self.configObject)
        
    def IW_Q_amp(self,Q_amp=None, ro_element= None,ro_pulse = None,update =True):
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        integrationWeights = self.configObject.SuperMembers[ro_element].PulseParamsDict[ro_pulse].integrationWeights
        if Q_amp is None: return integrationWeights.Q_amp
        integrationWeights.Q_amp=Q_amp
        if update: self.update_config(self.configObject)
    
    def IW_I_phase(self,I_phase=None, ro_element= None,ro_pulse = None,update =True):
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        integrationWeights = self.configObject.SuperMembers[ro_element].PulseParamsDict[ro_pulse].integrationWeights
        if I_phase is None: return integrationWeights.I_phase
        integrationWeights.I_phase=I_phase
        if update: self.update_config(self.configObject)
        
    def IW_Q_phase(self,Q_phase=None, ro_element= None,ro_pulse = None,update =True):
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        integrationWeights = self.configObject.SuperMembers[ro_element].PulseParamsDict[ro_pulse].integrationWeights
        if Q_phase is None: return integrationWeights.Q_phase
        integrationWeights.Q_phase=Q_phase
        if update: self.update_config(self.configObject)
        
    def IW_SOf(self,SOf=None, ro_element= None,ro_pulse = None,update =True):
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        integrationWeights = self.configObject.SuperMembers[ro_element].PulseParamsDict[ro_pulse].integrationWeights
        if SOf is None: return integrationWeights.SOf
        integrationWeights.SOf= SOf
        if update: self.update_config(self.configObject)
        
    def set_IQ_rot_phase(self, phase=None, ro_element = None, ro_pulse = None, update=True):
        """ rotates the IQ plane using the integration weights of the OPX.
        pass phase in degrees
        """
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        integrationWeights = self.configObject.SuperMembers[ro_element].PulseParamsDict[ro_pulse].integrationWeights
        if phase is None:
            return integrationWeights.rot_phase
        integrationWeights.rot_phase = phase
        if update: self.update_config(self.configObject)
        
    def IW_discrim_I(self,discrim_I=None, ro_element= None, ro_pulse = None, is_normalize = True, update =True):
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        integrationWeights = self.configObject.SuperMembers[ro_element].PulseParamsDict[ro_pulse].integrationWeights
        if discrim_I is None: return integrationWeights.discrim_I
        if is_normalize: discrim_I = discrim_I/np.max(np.abs(discrim_I))
        integrationWeights.discrim_I=discrim_I
        if update: self.update_config(self.configObject)
        
    def IW_discrim_Q(self,discrim_Q=None,ro_element= None,ro_pulse = None, is_normalize = True, update =True):
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        integrationWeights = self.configObject.SuperMembers[ro_element].PulseParamsDict[ro_pulse].integrationWeights
        if discrim_Q is None: return integrationWeights.discrim_Q
        if is_normalize: discrim_Q = discrim_Q/np.max(np.abs(discrim_Q))
        integrationWeights.discrim_Q=discrim_Q
        if update: self.update_config(self.configObject)
        
    # def IW_discrim_phase(self, discrim_phase=None,ro_element= None,ro_pulse = None, update =True):
    #     if ro_element is None: ro_element = self.main_readout
    #     if ro_pulse is None: ro_pulse = self.ro_pulse
    #     integrationWeights = self.configObject.SuperMembers[ro_element].PulseParamsDict[ro_pulse].integrationWeights
    #     if discrim_phase is None: return integrationWeights.discrim_phase 
    #     integrationWeights.discrim_phase=discrim_phase
    #     if update: self.update_config(self.configObject)

    def set_integration_weights(self, I_phase = None,Q_phase = None, I_amp = None, Q_amp = None, SOf=None, ro_element = None, ro_pulse = None, rot_phase= None, update = True):
        self.IW_I_amp(I_amp=I_amp, ro_element = ro_element, ro_pulse = ro_pulse, update =False)
        self.IW_Q_amp(Q_amp =Q_amp, ro_element = ro_element,ro_pulse = ro_pulse, update =False)
        self.IW_I_phase(I_phase=I_phase, ro_element = ro_element, ro_pulse = ro_pulse, update =False)
        self.IW_Q_phase(Q_phase =Q_phase, ro_element = ro_element, ro_pulse = ro_pulse, update =False)
        self.IW_SOf(SOf=SOf, ro_element = ro_element, ro_pulse = ro_pulse, update =False)
        self.set_IQ_rot_phase(phase = rot_phase,ro_element=ro_element, ro_pulse = ro_pulse, update =False)
        
        if update: self.update_config(self.configObject)
       
            
    def run_down_conversion_mixer_calibration(self, N_avg = 1000, npts = 100, wait_time = None, ro_element = None, ro_pulse = None, fit = True):
            """calibrate the phase and the amplitude corrections for the down conversion mixer. will print the results to paste in the configuration."""
            if wait_time is None: wait_time = self.wait_between_seq
            if ro_element is None: ro_element = self.main_readout
            if ro_pulse is None: ro_pulse = self.ro_pulse
            
            self._single_DCMC(ro_element = ro_element, N_avg = N_avg, npts=npts, wait_time = wait_time, ro_pulse = ro_pulse, label = 'current')
            plt.suptitle('current config')
            plt.pause(0.1)
            
            rot_phase = self.set_IQ_rot_phase(ro_element=ro_element)
            self.set_integration_weights(ro_element = ro_element, ro_pulse = ro_pulse, I_phase=0, Q_phase=-np.pi/2, I_amp= 1,Q_amp=1, rot_phase=0)

            phaseI, phaseQ, Iamp, Qamp = self._single_DCMC(ro_element = ro_element, N_avg = N_avg, npts=npts, wait_time = wait_time, ro_pulse = ro_pulse, label = 'default')
            plt.suptitle('default config')
            plt.pause(0.1)
            self.set_integration_weights(ro_element = ro_element, ro_pulse = ro_pulse, I_phase=phaseI, Q_phase=phaseQ, I_amp=Iamp, Q_amp=Qamp, rot_phase=0)

            self._single_DCMC(ro_element = ro_element, N_avg = N_avg, npts=npts, wait_time = wait_time, ro_pulse = ro_pulse, label = 'corrected', is_print_error = True)
            plt.suptitle('corrected config')
            plt.pause(0.1)
            print("Copy this to the configuration:"); print(f"I_phase_{ro_element} = ",phaseI); print(f"Q_phase_{ro_element} = ",phaseQ);  print(f"I_amp_{ro_element} = ",Iamp);  print(f"Q_amp_{ro_element} = ",Qamp)
            
    def _single_DCMC(self, ro_element, ro_pulse, wait_time = None, N_avg = 1000, npts = 100, label = '', is_print_error = False, is_ro_cancellation = None):
        if wait_time is None: wait_time = self.wait_between_seq
        else: wait_time = wait_time//4
        if is_ro_cancellation is None: is_ro_cancellation = self.is_ro_cancellation
        
        if is_ro_cancellation: ro_cncl = ro_element + '_canceller'
        phase_list = np.linspace(0.0,5.0,npts).tolist()
        
        with program() as self.DCMC_prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            phase = declare(fixed)
            with for_(n, 0, n < N_avg, n + 1):
                with for_each_(phase, phase_list):
                    if is_ro_cancellation:
                        reset_frame(ro_cncl)
                        reset_phase(ro_cncl)
                        frame_rotation_2pi(phase, ro_cncl)
                        play('ro_cancel_pulse', ro_cncl, duration = self.pulse_len(ro_element, ro_pulse)//4)
                    reset_frame(ro_element)
                    reset_phase(ro_element)
                    frame_rotation_2pi(phase, ro_element)
                    # play('ramp_up' , readout)
                    measure(ro_pulse, ro_element, None,
                            demod.full("integ_w_I", I, "out1"),
                            demod.full("integ_w_Q", Q, "out2"))
                    wait(wait_time, ro_element)
                    save(I, 'I')
                    save(Q, 'Q')
    
        self.qm_server.clear_all_job_results()
        job = self.qm.execute(self.DCMC_prog, duration_limit=0, data_limit=0)
        job.result_handles.wait_for_all_values()
        
        I = job.result_handles.get('I').fetch_all()['value'].reshape((N_avg, -1))
        Q = job.result_handles.get('Q').fetch_all()['value'].reshape((N_avg, -1))
        I_mean = I.mean(0)
        Q_mean = Q.mean(0)
        I_err = sp.stats.sem(I,0)
        Q_err = sp.stats.sem(Q,0)
        phase_list = np.array(phase_list )* 2 * np.pi
        # fit
        sft1 = sft('Cos', I_mean, phase_list)
        [fit1, cov1] = curve_fit( sft1.func, xdata = phase_list, ydata = I_mean, p0 = sft1.guess)
        sft2 = sft('Cos', Q_mean, phase_list)
        [fit2, cov2] = curve_fit( sft2.func, xdata = phase_list, ydata = Q_mean, p0 = sft2.guess)
        
        plt.figure('Down Conversion')
        ax=plt.subplot(111)
        plt.errorbar(I_mean,Q_mean,Q_err,I_err, capsize=6, label = label)
        ax.set_aspect(1)
        plt.legend()
        plt.figure()
        plt.subplot(2,1,1)
        plt.errorbar(phase_list,I_mean,I_err, capsize=6, label = label)
        plt.plot(phase_list, sft1.func(phase_list, *fit1))
        plt.subplot(2,1,2)
        plt.errorbar(phase_list,Q_mean,Q_err, capsize=6, label = label)
        plt.plot(phase_list, sft2.func(phase_list, *fit2))
            
        
        Iamp = fit1[0]
        if Iamp < 0 :
            Iamp = -Iamp
            Iph = fit1[2] + np.pi
        else: 
            Iph = fit1[2]
            
        Qamp = fit2[0]
        if Qamp < 0 :
            Qamp = -Qamp
            Qph = fit2[2] + np.pi
        else:
            Qph = fit2[2]
    
        if is_print_error: print(f'The standard deviation of I is {I_err.mean(0)} and of Q is {Q_err.mean(0)}')
        return Iph,Qph,Iamp,Qamp
        
            
    def tof(self, element = None, tof= None, update = True):
         if element is None: element = self.main_readout        # self.config = self.qm.get_config()
         
         if tof is None:
             # TOF = self.config['elements'][key]['time_of_flight']
            tof = self.configObject.SuperMembers[element].ElementParams['time_of_flight']
            return tof
         
         if tof % 4 != 0 or tof <24: 
             raise ValueError(f'Time of flight must be a multiple of 4 and greater or equal than 24 (got {tof})')
     
         self.configObject.SuperMembers[element].ElementParams['time_of_flight'] = tof
         
         if update: self.update_config(self.configObject)

    def update_config(self, Config=None):
        """update_config Update the OPX and this object with a new set of configurations.

        A centrelized functions for updating both this object (self) and the OPX (@ self.qm) with a new set of configurations.

        Args:
            Config ( dict or Config() ): Your new or updated configurations/
            
        """
        # If not received an object, try to get it from self:
        if Config is None:
            if isinstance(self.configObject, ConfigObject): 
                Config = self.configObject
            
        # act according to given config:
        if  type(Config).__name__ == 'Config':
            # temp_on_conflict =Config.DictAdditionsConfig['onConflicts']
            # Config.DictAdditionsConfig['onConflicts'] = 'override'
            self.configObject   = Config
            if hasattr(self, 'config'):
                self.config         = Config.get('config', self.config)
            else:
               self.config         = Config.get('config')
                
            # self.aux_config     = Config.get('aux_config')
            self.octave_config  = Config.get('octave_config')
            
        elif type(Config)==dict:
            self.config = Config
            self.octave_config  = Config.get('octave_config')
        else:
            self.configObject   = None
            self.config         = Config['opx_config']
            self.octave_config  = Config.get('octave_config')
        
        if self.is_working_in_parallel:
            if not self.is_qm_open():
                if self.qm_server.version_dict()['qm-qua'] == '1.2.2':
                    if len(self.qm_server.list_open_qms()) > 0:
                        self.qm = self.qm_server.get_qm(self.qm_server.list_open_qms()[0])
                        self.wait_if_busy()
                else:
                    if len(self.qm_server.list_open_quantum_machines()) > 0:
                        self.qm = self.qm_server.get_qm(self.qm_server.list_open_quantum_machines()[0])
                        self.wait_if_busy()
                
        self.qm = self.qm_server.open_qm(self.config, use_calibration_data = False)
        
        if self.set_mxgs_flag: 
            self.set_all_mxgs()
            self.set_mxgs_flag = False
            
        else:
            #Fixed octave bug
            for sg_name, sg in self.configObject.mxgs.items():
                if sg.is_using_octave_externally: 
                    for element, _ in sg.PairedMembers.items():
                        # self.set_octave_external_lo_freq(element)
                        self.set_octave_lo_source(element, 'External')


    def is_qm_open(self):
        if not hasattr(self, 'qm'):
            return False
        return self.qm.id in self.qm_server.list_open_quantum_machines()
        
    def wait_if_busy(self, sleep_interval = 1, timeout = None):
        if timeout == None: timeout = self.wait_if_busy_timeout
        job = self.qm.get_running_job()
        if job is None: 
            return
        try:
            t=0
            dots = itertools.cycle(['.', '..', '...'])
            while job.status in ['running', 'pending']:
                sys.stdout.write('\rWaiting for a job from other user'+ next(dots) + '  ')
                sys.stdout.flush()
                sleep(sleep_interval)
                t += sleep_interval
                if t>timeout: raise TimeoutError(f'Running job did not finish after {timeout} seconds.')
        except:
            print('\nStopped waiting for job.\nDo you want to kill the job?\nPrint <Y> or <y> or <yes> to kill or anything else to keep waiting.')
            if input() in ['Y', 'y', 'yes', 'Yes']:
                return
            else:
                self.wait_if_busy()
    
    # def close_qm(self):
    #     self.qm = self.qm_server.get_qm(self.qm_server.list_open_quantum_machines()[0])
    #     self.qm.close()
        
    def update_mixer_by_element(self, element, phase_correction =None, g_correction=None, update = True):
        """sets the mixer correctin matrix of element in opx and updates the matrix in self.config based on the values in self.aux_config
        returns the phase correction and the amp corrections of a mixer"""
        if phase_correction is None: phase_correction= self.configObject.SuperMembers[element].ElementParams['phaseCorrection']
        if g_correction is None: g_correction= self.configObject.SuperMembers[element].ElementParams['gCorrection']
        
        self.configObject.SuperMembers[element].ElementParams['phaseCorrection'] = phase_correction
        self.configObject.SuperMembers[element].ElementParams['gCorrection']    = g_correction
        self.configObject.SuperMembers[element].is_update = True
        self.configObject.SuperMembers[element].element.is_update = True
       
        corr_mat =  self.calc_cmat(phase_correction, g_correction)
        self.qm.set_mixer_correction(self.configObject.SuperMembers[element].element.mixerName, 
                                      int(self.configObject.SuperMembers[element].ElementParams['intermediateFreq']), 
                                      int(self.configObject.SuperMembers[element].element.loFreq), corr_mat)
           
        if update: self.update_config(self.configObject)

        return phase_correction, g_correction
    
    def get_mixer_correction(self, element):
        phase_correction= self.configObject.SuperMembers[element].ElementParams['phaseCorrection']
        g_correction= self.configObject.SuperMembers[element].ElementParams['gCorrection']
        
        return np.array(self.calc_cmat(phase_correction, g_correction)).reshape((2,2))
        
            
#%%pulse/operation  functions 
#TODO we can add as a kwarg a phase or any value which may be required for manipulating the pulse if we wish in the future 
    def _pi_pulse(self,
                pulse_type: str =None, #None for Eliya code or just standard, 'Un'/'Con' otherwise
                pulse_init: str = None,#'' for gaussian, can be such as drag and to add additional features  ) 
                **kwargs):

        if pulse_init is None: pulse_init = ''
        else: pulse_init = pulse_init +'_'
    
        if pulse_type is None: self._default_pi_pulse(pulse_init= pulse_init,**kwargs) 
        elif pulse_type == 'Un': self._Un_pi_pulse(pulse_init= pulse_init,**kwargs)     
        elif pulse_type == 'Con': self._Con_pi_pulse(pulse_init= pulse_init,**kwargs)     
        else: raise ValueError('Dummy proofing')
        
    def _default_pi_pulse(self, qubit = None,
        pulse_init ='',**kwargs):#'' for gaussian, can be such as drag )  

        if qubit is None: qubit = self.main_qubit
        
        if self.is_pi_pulse:
            play(pulse_init+self.pi_pulse, qubit)
        else:
            play(pulse_init+self.pi2_pulse, qubit)
            play(pulse_init+self.pi2_pulse, qubit)

    def _Con_pi_pulse(self,*args,qubit = None,pulse_init ='',**kwargs):
        if qubit is None: 
            qubit = self.main_qubit
        if self.is_pi_pulse:
            play(pulse_init+'ConRotpi_pulse', qubit)
        else:
            play(pulse_init+'ConRotpi2_pulse', qubit)
            play(pulse_init+'ConRotpi2_pulse', qubit)

    def _Un_pi_pulse(self,*args,qubit = None,pulse_init ='',**kwargs):
        if qubit is None: 
            qubit = self.main_qubit
        if self.is_pi_pulse:
            play(pulse_init+'UnRotpi_pulse', qubit)
        else:
            play(pulse_init+'UnRotpi2_pulse',qubit)
            play(pulse_init+'UnRotpi2_pulse', qubit)  
    
    
    def calibrate_rabi(self, qubit = None, ro_element = None, mm = None, #This is a work in progress, might be buggy
                      periods = 4, rabi_freq = 12.5e6,
                      **kwargs):
        tido.load_rabi(npts = 1, min_seq_time = (periods + 0.75) * (2*np.pi/rabi_freq),**kwargs)
        tido.run_rabi(is_return_fit = True, is_fit_decay = True, is_MLE = False, is_plot_fft = False)

    
    
    def sideband_cool(self, qubit = None, ro_element = None, mm = None,
                      sb_steady_time = None, sb_wait_time = None, sb_duration = None, sb_extra_duration = None,
                      rabi_sideband_cooling_pulse = 'rabi_pulse', sideband_pulse = 'constant_sideband_cooling_pulse', 
                      mm_sideband_cooling_pulse = 'constant_sideband_cooling_pulse',
                      is_ramp = None,
                      is_pi2 = None,  pi2_amp_scale = 1.0,
                      ro_detuning = None, qb_detuning = None, mm_detuning = None,
                      is_align_all = True,
                      disp_amp_scale = 0, disp_phase_2pi = 0,
                      **kwargs):
        if ro_element is None: ro_element = self.main_readout
        if qubit is None: qubit = self.main_qubit
        if type(qubit) is not list: qubit = [qubit]
        if mm is None: mm = self.sb_cool_kwargs[0]['mm']
        is_using_mm = mm != False
        if is_using_mm: mm_list = [mm] if type(mm) is not list else mm
        
        for qb in qubit:
            reset_frame(qb)
            
        if ro_detuning is None: ro_detuning = self.sb_cool_kwargs['ro_detuning']
        if qb_detuning is None: qb_detuning = self.sb_cool_kwargs['qb_detuning']
        if mm_detuning is None and mm is not None: mm_detuning = self.sb_cool_kwargs['mm_detuning']
        if is_pi2 is None: is_pi2 = self.sb_cool_kwargs['is_pi2']
        if is_ramp is None: is_ramp = self.sb_cool_kwargs['is_ramp']
        if sb_wait_time is None: sb_wait_time = self.sb_cool_kwargs['sb_wait_time']
        if sb_steady_time is None: sb_steady_time = self.sb_cool_kwargs['sb_steady_time']
        if sb_extra_duration is None: sb_extra_duration = self.sb_cool_kwargs['sb_extra_duration']
        if sb_duration is None: sb_duration = self.sb_cool_kwargs['sb_duration']
        
        if ro_detuning!=0: update_frequency(ro_element, self.element_IF(ro_element) + ro_detuning, keep_phase = True)
        if qb_detuning!=0: update_frequency(qubit[0], self.element_IF(qubit[0]) + qb_detuning, keep_phase = True)
        if is_using_mm:
            for mm in mm_list:
                if mm_detuning!=0 and mm: update_frequency(mm, self.element_IF(mm) + mm_detuning, keep_phase = True)
        
        if sb_extra_duration == 'auto':
            sb_extra_duration = 0
            if is_pi2: sb_extra_duration+= self.pulse_len(qubit[0], 'pi2_pulse')
            if is_ramp: sb_extra_duration+= 2*self.pulse_ramp_len(qubit[0], rabi_sideband_cooling_pulse)
        qubit_wait_time = sb_steady_time
        if is_ramp: qubit_wait_time += self.pulse_ramp_len(ro_element, sideband_pulse)
            
        if type(sb_duration) not in [int, float]:
            actual_sb_duration = declare(int)
            assign(actual_sb_duration, sb_duration+(sb_steady_time+sb_extra_duration)//4)
        else:
            if sb_duration is None: 
                sb_duration = self.pulse_len(qubit[0], rabi_sideband_cooling_pulse)//4
            actual_sb_duration = int(sb_duration//4+(sb_steady_time+sb_extra_duration)//4)
            sb_duration = int(sb_duration//4)
        pi2_phase = 0.25 if ro_detuning > 0 else -0.25
        
        if is_align_all: 
            if is_using_mm:
                for mm in mm_list:
                    align(qubit[0], mm, ro_element)
            else:
                align(*qubit, ro_element)
            
            
        if is_ramp: play((sideband_pulse+"_ramp_up"), ro_element)
        play(sideband_pulse, ro_element, duration = actual_sb_duration)
        if is_ramp: play((sideband_pulse+"_ramp_down"), ro_element)
        for qb in qubit:
            if qubit_wait_time//4>0:
                wait(qubit_wait_time//4, qb)
                
            if is_ramp:
                play((rabi_sideband_cooling_pulse + "_ramp_up"), qb)
            if sb_duration!=0: play(rabi_sideband_cooling_pulse, qb, duration = sb_duration)
            if is_ramp:
                play((rabi_sideband_cooling_pulse + "_ramp_down"), qb)
                
        if is_using_mm:
            for mm in mm_list:
                if is_ramp: play((mm_sideband_cooling_pulse + "_ramp_up"), mm)
                play(mm_sideband_cooling_pulse, mm, duration = actual_sb_duration)
                if is_ramp: play((mm_sideband_cooling_pulse + "_ramp_down"), mm)
        
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
        if is_using_mm:
            for mm in mm_list:
                if mm_detuning!=0 and mm: update_frequency(mm, self.element_IF(mm), keep_phase = True)
        
        align(*qubit, ro_element)
        
        if sb_wait_time//4  > 0: 
            wait(int(sb_wait_time//4), ro_element)
            if is_using_mm: wait(int(sb_wait_time//4), mm)
            align(*qubit, ro_element)
        
        frame_rotation_2pi(disp_phase_2pi, mm)
        if disp_amp_scale != 0: 
            play('disp' * amp(disp_amp_scale), mm)        
    
    def perform_full_measurement(self, I, Q, I_output_name = 'I', Q_output_name = 'Q', ro_element = None, is_amplify = None, is_ramp_up = True, readout_pulse = None, is_save=True, is_wait = True, is_active_reset = None, reset_qubit = None, is_ro_cancellation = None, cancellation_delay = None,
                                 is_sb_cool = None, which_sb_cool = None, sb_cool_kwargs = None, amp_scale = 1,
                                 is_reset_phase = True, is_loop_reset = None, wait_time = None, wait_after_reset = None, **kwargs):
        """Performs a full demod. measurement using main readout.\n
        Will automatically add ramp up to the measurement pulse if it is added in the config. Ramp up can be removed manually with argument.\n
        Will add pump pulse to the main_paramp element if it is attributed to the tido class and specified in the argument."""
        if wait_time is None: wait_time = self.wait_between_seq
        if ro_element is None: ro_element = self.main_readout
        if readout_pulse is None: readout_pulse = self.ro_pulse
        if is_amplify is None: is_amplify = self.is_amplify
        if is_active_reset is None: is_active_reset = self.is_active_reset
        if is_loop_reset is None: is_loop_reset = self.is_loop_reset
        if is_ro_cancellation is None: is_ro_cancellation = self.is_ro_cancellation
        if cancellation_delay is None: cancellation_delay = self.cancellation_delay
        if reset_qubit is None: reset_qubit = self.reset_qubits
        if is_sb_cool is None: is_sb_cool = self.is_sb_cool
        if is_sb_cool:
            if sb_cool_kwargs is None: sb_cool_kwargs = self.sb_cool_kwargs
            if which_sb_cool is None: which_sb_cool = [sb_cool_kwargs[0]['mm']]
        if wait_after_reset is None: wait_after_reset = self.wait_after_reset
        if is_amplify and self.main_paramp is not None:
            align(ro_element, self.main_paramp)
            reset_phase(self.main_paramp)
            if is_ramp_up and self.configObject.SuperMembers[self.main_paramp].PulseParamsDict[self.pump_pulse].Additions.ramp['up'] is not None: play('ramp_up', self.main_paramp)
            play(self.pump_pulse, self.main_paramp)
        if is_ro_cancellation:
            ro_cncl = ro_element + '_canceller'
            cancel_pulse = 'ro_cancel_pulse'
            align(ro_element, ro_cncl)
            if cancellation_delay > 0: wait(cancellation_delay//4, ro_cncl)
            reset_phase(ro_cncl)
            if is_ramp_up and self.configObject.SuperMembers[ro_cncl].PulseParamsDict[cancel_pulse].Additions.ramp['up'] is not None: play(self.configObject.SuperMembers[ro_cncl].PulseParamsDict[cancel_pulse].Additions.ramp['up']['name'], ro_cncl)
            play(cancel_pulse, ro_cncl, duration = self.pulse_len(ro_element, readout_pulse)//4)
            if is_ramp_up and self.configObject.SuperMembers[ro_cncl].PulseParamsDict[cancel_pulse].Additions.ramp['down'] is not None: play(self.configObject.SuperMembers[ro_cncl].PulseParamsDict[cancel_pulse].Additions.ramp['down']['name'], ro_cncl)
        if is_reset_phase: reset_phase(ro_element)
        if amp_scale ==1:
            if is_ramp_up and self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['up'] is not None:
                play(self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['up']['name'], ro_element)
            measure(readout_pulse, ro_element, None,
                demod.full('integ_w_I', I, 'out1'),
                demod.full('integ_w_Q', Q, 'out2'))
            if is_ramp_up and self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['down'] is not None:
                play(self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['down']['name'], ro_element)
        else: 
            if is_ramp_up and self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['up'] is not None:
                play(self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['up']['name'] * amp(amp_scale), ro_element)
                
            if self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].integrationWeights.is_dual:
                measure(readout_pulse * amp(amp_scale), ro_element, None,
                    dual_demod.full('integ_w_IfromI', 'out1', 'integ_w_IfromQ', 'out2', I),
                    dual_demod.full('integ_w_QfromI', 'out1', 'integ_w_QfromQ', 'out2', Q))
            else:
                measure(readout_pulse * amp(amp_scale), ro_element, None,
                    demod.full('integ_w_I', I, 'out1'),
                    demod.full('integ_w_Q', Q, 'out2'))
            if is_ramp_up and self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['down'] is not None:
                play(self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['down']['name'] * amp(amp_scale), ro_element)
        if is_wait and not is_active_reset and not is_sb_cool:
            # Rest time between measurements:
            wait(int(wait_time//4), ro_element)  # reset time
        if is_save:
            # Save resulsts:
            save(I, I_output_name)
            save(Q, Q_output_name)
        if is_sb_cool:
            print(f'Sideband cooling {which_sb_cool}')
            for cool_dict in sb_cool_kwargs:
                if cool_dict['mm'] in which_sb_cool:
                    self.sideband_cool(**cool_dict)
                else:
                    continue
        if is_active_reset:
            if type(reset_qubit) != list:
                align(ro_element, reset_qubit)
            else:
                for qb in reset_qubit:
                    align(ro_element, qb)
            state_ind_to_reset = declare(int)
            state_ind_to_reset = self.analyze_readout(reset_qubit, I, Q, is_in_prog = True, state_ind_to_reset = state_ind_to_reset)
            reset_count = declare(int)
            assign(reset_count, 1-state_ind_to_reset)
            # save(I,'reset_I')
            # save(Q,'reset_Q')
            # save(state_ind_to_reset,'reset_index')
            if wait_after_reset >= 16: wait(int(wait_after_reset//4), ro_element)
            if type(reset_qubit) != list:
                align(ro_element, reset_qubit)
            else:
                for qb in reset_qubit:
                    align(ro_element, qb)
            self.active_reset(qubit=reset_qubit, state_ind_to_reset = state_ind_to_reset)
            if is_loop_reset:
                with while_(~(reset_count==self.loop_repetition)):
                    I,Q = self.perform_full_measurement(I,Q, is_active_reset = False, is_wait = False, is_save = False, is_ramp_up = is_ramp_up, readout_pulse = readout_pulse)
                    state_ind_to_reset = self.analyze_readout(reset_qubit, I, Q, is_in_prog = True, state_ind_to_reset = state_ind_to_reset)
                    with if_(state_ind_to_reset==0):
                        assign(reset_count, reset_count+1)
                    with else_():
                        assign(reset_count, 0)
                    if wait_after_reset >= 16: wait(int(wait_after_reset//4), ro_element)
                    if type(reset_qubit) != list:
                        align(ro_element, reset_qubit)
                    else:
                        for qb in reset_qubit:
                            align(ro_element, qb)
                    self.active_reset(qubit=reset_qubit, state_ind_to_reset = state_ind_to_reset)
                    # save(I,'reset_I')
                    # save(Q,'reset_Q')
                    # save(state_ind_to_reset,'reset_index')
                assign(state_ind_to_reset,2)
                save(state_ind_to_reset,'state')
                # save(I,'reset_I')
                # save(Q,'reset_Q')
                # save(state_ind_to_reset,'reset_index')
        if self.is_add_explicit_align_at_end:
            align(*self.config['elements'].keys())
        return I,Q
        
    def perform_sliced_measurement(self, I, Q, i, chunk_size, npts, I_output_name = 'I', Q_output_name = 'Q', ro_element = None, is_save = True, is_amplify = None, is_ramp_up = True, is_ramp_down = True, readout_pulse = None, is_wait = True, amp_scale=1, is_reset_phase=True, is_ro_cancellation = None, cancellation_delay = None, **kwargs):
        """Performs a sliced demod. measurement using main readout.\n
        Will automatically add ramp up to the measurement pulse if it is added in the config. Ramp up can be removed manually with argument.\n
        Will add pump pulse to the main_paramp element if it is attributed to the tido class and specified in the argument."""
        if readout_pulse is None: readout_pulse = self.ro_pulse
        if is_amplify is None: is_amplify = self.is_amplify
        if ro_element is None: ro_element = self.main_readout
        if is_ro_cancellation is None: is_ro_cancellation = self.is_ro_cancellation
        if cancellation_delay is None: cancellation_delay = self.cancellation_delay
        if is_amplify and self.main_paramp is not None:
            align(ro_element, self.main_paramp)
            reset_phase(self.main_paramp)
            if is_ramp_up and self.configObject.SuperMembers[self.main_paramp].PulseParamsDict[self.pump_pulse].Additions.ramp['up'] is not None: play('ramp_up', self.main_paramp)
            play(self.pump_pulse, self.main_paramp)
        
        if is_ro_cancellation:
            ro_cncl = ro_element + '_canceller'
            cancel_pulse = 'ro_cancel_pulse'
            align(ro_element, ro_cncl)
            if cancellation_delay > 0: wait(cancellation_delay//4, ro_cncl)
            if is_reset_phase: reset_phase(ro_cncl)
            if is_ramp_up and self.configObject.SuperMembers[ro_cncl].PulseParamsDict[cancel_pulse].Additions.ramp['up'] is not None: play(self.configObject.SuperMembers[ro_cncl].PulseParamsDict[cancel_pulse].Additions.ramp['up']['name'], ro_cncl)
            play(cancel_pulse, ro_cncl, duration = self.pulse_len(ro_element, readout_pulse)//4)
            if is_ramp_up and self.configObject.SuperMembers[ro_cncl].PulseParamsDict[cancel_pulse].Additions.ramp['down'] is not None: play(self.configObject.SuperMembers[ro_cncl].PulseParamsDict[cancel_pulse].Additions.ramp['down']['name'], ro_cncl)
        
        if is_reset_phase: reset_phase(ro_element)
        if is_ramp_up and self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['up'] is not None: play(self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['up']['name'], ro_element)
        # if is_ramp_up and self.configObject.SuperMembers[ro].PulseParamsDict[readout_pulse].Additions.ramp['up'] is not None: play('ramp_up', ro_element)
        
        if self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].integrationWeights.is_dual:
            if amp_scale == 1:
                measure(readout_pulse * amp(amp_scale), ro_element, None,
                    dual_demod.sliced('integ_w_IfromI', 'out1', 'integ_w_IfromQ', 'out2', I, chunk_size),
                    dual_demod.sliced('integ_w_QfromI', 'out1', 'integ_w_QfromQ', 'out2', Q, chunk_size))
            else:
                measure(readout_pulse*amp(amp_scale), ro_element, None,
                    dual_demod.sliced('integ_w_IfromI', 'out1', 'integ_w_IfromQ', 'out2', I, chunk_size),
                    dual_demod.sliced('integ_w_QfromI', 'out1', 'integ_w_QfromQ', 'out2', Q, chunk_size))
        else:
            if amp_scale == 1:
                measure(readout_pulse, ro_element, None,
                        demod.sliced('integ_w_I', I, chunk_size, 'out1'),
                        demod.sliced('integ_w_Q', Q, chunk_size, 'out2'))
            else:
                measure(readout_pulse*amp(amp_scale), ro_element, None,
                    demod.sliced('integ_w_I', I, chunk_size, 'out1'),
                    demod.sliced('integ_w_Q', Q, chunk_size, 'out2'))
                
            
        if is_ramp_down and self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['down'] is not None: play(self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['down']['name'], ro_element)
        if is_wait:
            # Rest time between measurements:
            wait(int(self.wait_between_seq//4), ro_element)  # reset time
        
        if is_save:
            # Save resulsts:
            with for_(i, 0, i < npts, i+1):
                save(I[i], I_output_name)
                save(Q[i], Q_output_name)
            
    def perform_accumulated_measurement(self, I, Q, i, chunk_size, npts, I_output_name = 'I', Q_output_name = 'Q', ro_element = None, is_save = True, is_amplify = None, is_ramp_up = True, readout_pulse = None, is_wait = True, amp_scale=1, is_reset_phase=True, is_ro_cancellation = None, **kwargs):
        """Performs a full demod. measurement using main readout.\n
        Will automatically add ramp up to the measurement pulse if it is added in the config. Ramp up can be removed manually with argument.\n
        Will add pump pulse to the main_paramp element if it is attributed to the tido class and specified in the argument."""
        if readout_pulse is None: readout_pulse = self.ro_pulse
        if is_ro_cancellation is None: is_ro_cancellation = self.is_ro_cancellation
        if ro_element is None: ro_element = self.main_readout
        if is_amplify is None: is_amplify = self.is_amplify
        if is_amplify and self.main_paramp is not None:
            align(ro_element, self.main_paramp)
            reset_phase(self.main_paramp)
            if is_ramp_up and self.configObject.SuperMembers[self.main_paramp].PulseParamsDict[self.pump_pulse].Additions.ramp['up'] is not None: play('ramp_up', self.main_paramp)
            play(self.pump_pulse, self.main_paramp)
        
        if is_ro_cancellation:
            ro_cncl = ro_element + '_canceller'
            cancel_pulse = 'ro_cancel_pulse'
            align(ro_element, ro_cncl)
            if cancellation_delay > 0: wait(cancellation_delay//4, ro_cncl)
            if is_reset_phase: reset_phase(ro_cncl)
            if is_ramp_up and self.configObject.SuperMembers[ro_cncl].PulseParamsDict[cancel_pulse].Additions.ramp['up'] is not None: play(self.configObject.SuperMembers[ro_cncl].PulseParamsDict[cancel_pulse].Additions.ramp['up']['name'], ro_cncl)
            play(cancel_pulse, ro_cncl, duration = self.pulse_len(ro_element, readout_pulse)//4)
            if is_ramp_up and self.configObject.SuperMembers[ro_cncl].PulseParamsDict[cancel_pulse].Additions.ramp['down'] is not None: play(self.configObject.SuperMembers[ro_cncl].PulseParamsDict[cancel_pulse].Additions.ramp['down']['name'], ro_cncl)
        
        if is_reset_phase:
            reset_phase(ro_element)
        if is_ramp_up and self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['up'] is not None: play(self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['up']['name'], ro_element)
        # if is_ramp_up and self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['up'] is not None: play('ramp_up', ro_element)
        if amp_scale == 1:
            measure(readout_pulse, ro_element, None,
                    demod.accumulated('integ_w_I', I, chunk_size, 'out1'),
                    demod.accumulated('integ_w_Q', Q, chunk_size, 'out2'))
        else:
                measure(readout_pulse*amp(amp_scale), ro_element, None,
                    demod.accumulated('integ_w_I', I, chunk_size, 'out1'),
                    demod.accumulated('integ_w_Q', Q, chunk_size, 'out2'))
                
        if is_ramp_up and self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['down'] is not None: play(self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['down']['name'], ro_element)
        if is_wait:
            # Rest time between measurements:
            wait(int(self.wait_between_seq//4), ro_element)  # reset time
        
        # Save resulsts:
        with for_(i, 0, i < npts, i+1):
            save(I[i], I_output_name)
            save(Q[i], Q_output_name)
            
    def perform_moving_window_measurement(self, I, Q, i, chunk_size, chunks_per_window, npts, I_output_name = 'I', Q_output_name = 'Q', ro_element = None, is_save = True, is_amplify = None, is_ramp_up = True, readout_pulse = None, is_wait = True, amp_scale=1, is_reset_phase=True, **kwargs):
        raise ValueError("moving window did not work. Fix if you want it.")
        """Performs a moving window demod. measurement using main readout.\n
        Will automatically add ramp up to the measurement pulse if it is added in the config. Ramp up can be removed manually with argument.\n
        Will add pump pulse to the main_paramp element if it is attributed to the tido class and specified in the argument."""
        if readout_pulse is None: readout_pulse = self.ro_pulse
        if is_amplify is None: is_amplify = self.is_amplify
        if is_amplify and self.main_paramp is not None:
            align(ro_element, self.main_paramp)
            reset_phase(self.main_paramp)
            if is_ramp_up and self.configObject.SuperMembers[self.main_paramp].PulseParamsDict[self.pump_pulse].Additions.ramp['up'] is not None: play('ramp_up', self.main_paramp)
            play(self.pump_pulse, self.main_paramp)
        
        if is_reset_phase:
            reset_phase(ro_element)
        if is_ramp_up and self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['up'] is not None: play(self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['up']['name'], ro_element)
        # if is_ramp_up and self.configObject.SuperMembers[ro_element].PulseParamsDict[readout_pulse].Additions.ramp['up'] is not None: play('ramp_up', ro_element)
        if amp_scale == 1:
            measure(readout_pulse, ro_element, None,
                    demod.moving_window('integ_w_I', I, chunk_size, chunks_per_window, 'out1'),
                    demod.moving_window('integ_w_Q', Q, chunk_size, chunks_per_window, 'out2'))
        else:
            measure(readout_pulse*amp(amp_scale), ro_element, None,
                demod.moving_window('integ_w_I', I, chunk_size, chunks_per_window, 'out1'),
                demod.moving_window('integ_w_Q', Q, chunk_size, chunks_per_window, 'out2'))
                
        if is_wait:
            # Rest time between measurements:
            wait(int(self.wait_between_seq//4), ro_element)  # reset time
        
        if is_save:
        # Save resulsts:
            with for_(i, 0, i < npts, i+1):
                save(I[i], I_output_name)
                save(Q[i], Q_output_name)


    def update_readout_analyzer(self, qubit, thresh, is_ascending = True, unclear_range = 0, offset = [0,0], data_g = None, data_e = None):
        if type(qubit) is not list:
            self.readout_analyzer_dict['qubits'][qubit] = {'thresh': thresh,
                                                           'is_ascending': is_ascending,
                                                           'unclear_range': unclear_range,
                                                           'offset': offset,
                                                           'data_g': data_g,
                                                           'data_e': data_e}
        elif len(qubit)==2:
            self.readout_analyzer_dict['qubits']['2qubits'] = {'thresh': thresh,
                                                               'is_ascending': is_ascending,
                                                               'unclear_range': unclear_range,
                                                               'offset': offset}
            
    def find_threshold(self, I, Q, n_means, is_ascending, is_fit_circle = True, is_plot = True, is_for_in_prog = True):
        
        if is_fit_circle:
            circ = fit_circle(I.mean(), Q.mean())
            offset = circ[0],circ[1]
        else:
            offset = [0,0]
            
        I = I-offset[0]
        Q = Q-offset[0]
        
        data,_ = self.process_data([I, Q], is_mean = False)
        
        kmeans = KMeans(n_clusters = n_means).fit([[0,d] for d in data])
        means = kmeans.cluster_centers_
        means = np.sort(means.transpose()[1])
        
        if is_plot:
            _,axs = plt.subplots(1,2)
            
            labels = kmeans.labels_
            plt.sca(axs[0])
            for k,c in zip(range(n_means),['b','r','c','m']):
                plt.plot(I[labels==k], Q[labels==k], '.', color = c)
            plt.sca(axs[1])
            num_of_bins = 200
            bins = np.histogram(data, bins=num_of_bins)[1]
            for k,c in zip(range(n_means),['b','r','c','m']):
                plt.hist(data[labels==k], bins = bins, histtype = 'step', color = c)
        
        if is_ascending != means[0]<means[1]:
            means = np.flip(means)
        if self.which_data == 'Phase' and is_for_in_prog:
            thresh = [np.mean([np.tan(means[i]), np.tan(means[i+1])], dtype = float) for i in range(n_means-1)]
        else:
            thresh = [np.mean([means[i],means[i+1]], dtype = float) for i in range(n_means-1)]
            
        return thresh, offset, means
    
    
    def analyze_readout(self, qubit, I, Q, is_in_prog = True, state_ind_to_reset = None, thresh = None):
        if type(qubit) is list:
            if len(qubit)==2:
                qubit = '2qubits'
                is_ascending = thresh[1] > thresh[0]

        else:
            if qubit not in self.readout_analyzer_dict['qubits'].keys():
                raise ValueError("Active reset parameters are not calibrated. Please run Pi-no Pi with <is_update_readout_analyzer> = True.")
            is_ascending = self.readout_analyzer_dict['qubits'][qubit]['is_ascending']
                
        unclear_range = self.readout_analyzer_dict['qubits'][qubit]['unclear_range']
        if thresh is None: thresh = self.readout_analyzer_dict['qubits'][qubit]['thresh']
        
        offset = self.readout_analyzer_dict['qubits'][qubit]['offset']
        if type(thresh) is list and qubit != '2qubits':
            thresh = thresh[0]

        if offset != [0,0]:
            
            if is_ascending:
                if is_in_prog:
                    if self.which_data!='Phase':
                        data = declare(fixed)
                        if self.which_data == 'Q':
                            assign(data, Q-offset[1])
                        elif self.which_data == 'I':
                            assign(data, I-offset[0])
                        elif self.which_data == 'Mag':
                            assign(data, (I-offset[0])*(I-offset[0])+(Q-offset[1])*(Q-offset[1]))
                        elif np.isreal(self.which_data):
                            assign(data, (I-offset[0])*np.cos(self.which_data)+(Q-offset[1])*np.sin(self.which_data))
                        # save(data, 'reset_data')
                        
                        if qubit == '2qubits':
                            with if_(data<thresh[0]):
                                assign(state_ind_to_reset,0)
                            with elif_((data>thresh[0]) & (data<thresh[1])):
                                assign(state_ind_to_reset,1)
                            with elif_((data>thresh[1]) & (data<thresh[2])):
                                assign(state_ind_to_reset,2)
                            with else_():
                                assign(state_ind_to_reset,3)
                            return state_ind_to_reset
                        else:
                            with if_(data<thresh):
                                assign(state_ind_to_reset,0)
                            with else_():
                                assign(state_ind_to_reset,1)
                            return state_ind_to_reset
                    else:
                        if qubit == '2qubits':
                            with if_(Q-offset[1]<np.tan(thresh[0])*(I-offset[0])):
                                assign(state_ind_to_reset,0)
                            with elif_((Q-offset[1]>np.tan(thresh[0])*(I-offset[0])) & (Q-offset[1]<np.tan(thresh[1])*(I-offset[0]))):
                                assign(state_ind_to_reset,1)
                            with elif_((Q-offset[1]>np.tan(thresh[1])*(I-offset[0])) & (Q-offset[1]<np.tan(thresh[2])*(I-offset[0]))):
                                assign(state_ind_to_reset,2)
                            with else_():
                                assign(state_ind_to_reset,3)
                            return state_ind_to_reset
                        else:
                            with if_(Q-offset[1]<np.tan(thresh)*(I-offset[0])):
                                assign(state_ind_to_reset,0)
                            with else_():
                                assign(state_ind_to_reset,1)
                            return state_ind_to_reset
                        
                            
                else: #not in prog
                    data = self.determine_data(I,Q)
                    if qubit == '2qubits':
                        if(data<thresh[0]):
                            return 0
                        elif((data>thresh[0]) & (data<thresh[1])):
                            return 1
                        elif((data>thresh[1]) & (data<thresh[2])):
                            return 2
                        else:
                            return 3
                    else:
                        with if_(data<thresh):
                            return 0
                        with else_():
                            return 1
            else: #decending
                if is_in_prog:
                    if self.which_data!='Phase':
                        data = declare(fixed)
                        if self.which_data == 'Q':
                            assign(data, Q)
                        elif self.which_data == 'I':
                            assign(data, I)
                        elif self.which_data == 'Mag':
                            assign(data, I*I+Q*Q)
                        elif np.isreal(self.which_data):
                            assign(data, (I-offset[0])*np.cos(self.which_data)+(Q-offset[1])*np.sin(self.which_data))
                        # save(data, 'reset_data')
                        if qubit == '2qubits':
                            with if_(data>thresh[0]):
                                assign(state_ind_to_reset,0)
                            with elif_((data<thresh[0]) & (data>thresh[1])):
                                assign(state_ind_to_reset,1)
                            with elif_((data<thresh[1]) & (data>thresh[2])):
                                assign(state_ind_to_reset,2)
                            with else_():
                                assign(state_ind_to_reset,3)
                            return state_ind_to_reset
                        else:
                            with if_(data>thresh):
                                assign(state_ind_to_reset,0)
                            with else_():
                                assign(state_ind_to_reset,1)
                            return state_ind_to_reset
                    else:
                        if qubit == '2qubits':
                            with if_((Q-offset[1]>np.tan(thresh[0])*(I-offset[0])) & ((Q-offset[1]<np.tan(2*thresh[0]-thresh[1])*(I-offset[0])))):
                                assign(state_ind_to_reset,0)
                            with elif_((Q-offset[1]<np.tan(thresh[0])*(I-offset[0])) & (Q-offset[1]>np.tan(thresh[1])*(I-offset[0]))):
                                assign(state_ind_to_reset,1)
                            with elif_((Q-offset[1]<np.tan(thresh[1])*(I-offset[0])) & (Q-offset[1]>np.tan(thresh[2])*(I-offset[0]))):
                                assign(state_ind_to_reset,2)
                            with elif_((Q-offset[1]<np.tan(thresh[2])*(I-offset[0])) & ((Q-offset[1]>np.tan(2*thresh[2]-thresh[1])*(I-offset[0])))):
                                assign(state_ind_to_reset,3)
                            with else_():
                                assign(state_ind_to_reset,4)
                            return state_ind_to_reset
                        else:
                            with if_(Q-offset[1]>np.tan(thresh)*(I-offset[0])):
                                assign(state_ind_to_reset,0)
                            with else_():
                                assign(state_ind_to_reset,1)
                            return state_ind_to_reset
                        
                            
                else: #not in prog
                    data = self.determine_data(I,Q)
                    if qubit == '2qubits':
                        if(data>thresh[0]):
                            return 0
                        elif((data<thresh[0]) & (data>thresh[1])):
                            return 1
                        elif((data<thresh[1]) & (data>thresh[2])):
                            return 2
                        else:
                            return 3
                    else: #only one qubit
                        with if_(data<thresh):
                            return 0
                        with else_():
                            return 1
        else: #no offset
            if is_ascending:
                if is_in_prog:
                    if self.which_data!='Phase':
                        data = declare(fixed)
                        if self.which_data == 'Q':
                            assign(data, Q)
                        elif self.which_data == 'I':
                            assign(data, I)
                        elif self.which_data == 'Mag':
                            assign(data, I*I+Q*Q)
                        elif np.isreal(self.which_data):
                            assign(data, I*np.cos(self.which_data)+Q*np.sin(self.which_data))
                        # save(data, 'reset_data')
                        
                        if qubit == '2qubits':
                            with if_(data<thresh[0]):
                                assign(state_ind_to_reset,0)
                            with elif_((data>thresh[0]) & (data<thresh[1])):
                                assign(state_ind_to_reset,1)
                            with elif_((data>thresh[1]) & (data<thresh[2])):
                                assign(state_ind_to_reset,2)
                            with else_():
                                assign(state_ind_to_reset,3)
                            return state_ind_to_reset
                        else:
                            with if_(data<thresh):
                                assign(state_ind_to_reset,0)
                            with else_():
                                assign(state_ind_to_reset,1)
                            return state_ind_to_reset
                    else:
                        if qubit == '2qubits':
                            with if_(Q<np.tan(thresh[0])*I):
                                assign(state_ind_to_reset,0)
                            with elif_((Q>np.tan(thresh[0])*I) & (Q<np.tan(thresh[1])*I)):
                                assign(state_ind_to_reset,1)
                            with elif_((Q>np.tan(thresh[1])*I) & (Q<np.tan(thresh[2])*I)):
                                assign(state_ind_to_reset,2)
                            with else_():
                                assign(state_ind_to_reset,3)
                            return state_ind_to_reset
                        else:
                            with if_(Q<np.tan(thresh)*I):
                                assign(state_ind_to_reset,0)
                            with else_():
                                assign(state_ind_to_reset,1)
                            return state_ind_to_reset
                        
                            
                else: #not in prog
                    data = self.determine_data(I,Q)
                    if qubit == '2qubits':
                        if(data<thresh[0]):
                            return 0
                        elif((data>thresh[0]) & (data<thresh[1])):
                            return 1
                        elif((data>thresh[1]) & (data<thresh[2])):
                            return 2
                        else:
                            return 3
                    else:
                        with if_(data<thresh):
                            return 0
                        with else_():
                            return 1
            else: #decending
                if is_in_prog:
                    if self.which_data!='Phase':
                        data = declare(fixed)
                        if self.which_data == 'Q':
                            assign(data, Q)
                        elif self.which_data == 'I':
                            assign(data, I)
                        elif self.which_data == 'Mag':
                            assign(data, I*I+Q*Q)
                        elif np.isreal(self.which_data):
                            assign(data, I*np.cos(self.which_data)+Q*np.sin(self.which_data))
                        # save(data, 'reset_data')
                        if qubit == '2qubits':
                            with if_(data>thresh[0]):
                                assign(state_ind_to_reset,0)
                            with elif_((data<thresh[0]) & (data>thresh[1])):
                                assign(state_ind_to_reset,1)
                            with elif_((data<thresh[1]) & (data>thresh[2])):
                                assign(state_ind_to_reset,2)
                            with else_():
                                assign(state_ind_to_reset,3)
                            return state_ind_to_reset
                        else:
                            with if_(data>thresh):
                                assign(state_ind_to_reset,0)
                            with else_():
                                assign(state_ind_to_reset,1)
                            return state_ind_to_reset
                    else:
                        if qubit == '2qubits':
                            with if_((Q>np.tan(thresh[0])*I) & ((Q<np.tan(2*thresh[0]-thresh[1])*I))):
                                assign(state_ind_to_reset,0)
                            with elif_((Q<np.tan(thresh[0])*I) & (Q>np.tan(thresh[1])*I)):
                                assign(state_ind_to_reset,1)
                            with elif_((Q<np.tan(thresh[1])*I) & (Q>np.tan(thresh[2])*I)):
                                assign(state_ind_to_reset,2)
                            with elif_((Q<np.tan(thresh[2])*(I)) & ((Q>np.tan(2*thresh[2]-thresh[1])*I))):
                                assign(state_ind_to_reset,3)
                            with else_():
                                assign(state_ind_to_reset,4)
                            return state_ind_to_reset
                        else:
                            with if_(Q>np.tan(thresh)*I):
                                assign(state_ind_to_reset,0)
                            with else_():
                                assign(state_ind_to_reset,1)
                            return state_ind_to_reset
                        
                            
                else: #not in prog
                    data = self.determine_data(I,Q)
                    if qubit == '2qubits':
                        if(data>thresh[0]):
                            return 0
                        elif((data<thresh[0]) & (data>thresh[1])):
                            return 1
                        elif((data<thresh[1]) & (data>thresh[2])):
                            return 2
                        else:
                            return 3
                    else: #only one qubit
                        with if_(data<thresh):
                            return 0
                        with else_():
                            return 1
                
    def active_reset(self, qubit, state_ind_to_reset):
        
        if type(qubit) is not list:
            play('pi_pulse', qubit, condition= (state_ind_to_reset == 1))
        else:
            # wait(200//4,qubit[0])
            # wait(200//4,qubit[1])
            play('pi_pulse', qubit[0], condition = ((state_ind_to_reset==1) | (state_ind_to_reset==3)))
            play('pi_pulse', qubit[1], condition = ((state_ind_to_reset==2) | (state_ind_to_reset==3)))
                
        
        
    def play_seq(self,start_with,*args,elements = None,**kwargs):
        if elements is None: elements = [self.main_qubit]
        if type(start_with) is tuple: 
            play(*start_with)
            align(start_with[1],*elements)
        elif type(start_with) is types.FunctionType:
            start_with(*args,**kwargs)

    
    def octave_clock(self, clock = None):
        if clock is None: return self.qm.octave.get_clock(self.octave_name)
        else:
            if clock == '10MHz':
                self.qm_server.octave_manager.set_clock(self.octave_name, ClockType.External, ClockFrequency.MHZ_10)
                print('Set Octave clock to external ' +clock)
            elif clock == '100MHz':
                self.qm_server.octave_manager.set_clock(self.octave_name, ClockType.External, ClockFrequency.MHZ_100)
                print('Set Octave clock to external ' +clock)
            elif clock == '1000MHz' or clock == '1GHz':
                self.qm_server.octave_manager.set_clock(self.octave_name, ClockType.External, ClockFrequency.MHZ_1000)
                print('Set Octave clock to external ' +clock)
            else:
                self.qm_server.octave_manager.set_clock(self.octave_name, ClockType.Internal, ClockFrequency.MHZ_1000)
                print('Set Octave clock to internal')
                
    def set_octave_lo_source(self, element, source):
        """ """
        if source in ['internal', 'Internal']:
            self.qm.octave.set_lo_source(element, OctaveLOSource.Internal)
        elif source in ['external','External'] :
            I_out = self.configObject.elements[element].ICon
            if I_out == 1: self.qm.octave.set_lo_source(element, OctaveLOSource.LO1)
            elif I_out == 3: self.qm.octave.set_lo_source(element, OctaveLOSource.LO2)
            elif I_out == 5: self.qm.octave.set_lo_source(element, OctaveLOSource.LO3)
            elif I_out == 7: self.qm.octave.set_lo_source(element, OctaveLOSource.LO4)
            elif I_out == 9: self.qm.octave.set_lo_source(element, OctaveLOSource.LO5)
            else: raise ValueError("This element has an invalid I output")
        elif source in ['off','Off'] :
            self.qm.octave.set_lo_source(element, OctaveLOSource.Off)
            
    def load_octave_lo_from_config(self, element):
        self.qm.octave.load_lo_frequency_from_config(element)
        
    def set_octave_lo_freq(self, element, freq):
        self.qm.octave.set_lo_frequency(element, freq)
        
    def set_octave_external_lo_freq(self, element):
        self.qm.octave.update_external_lo_frequency(element, self.configObject.elements[element].loFreq)
    
    def octave_gain(self, element, gain = None):
        mxg_name = self.configObject.SuperMembers[element].mxg[0].name
        if gain == None: return self.configObject.mxgs[mxg_name].gain
        self.configObject.mxgs[mxg_name].gain = gain
        self.qm.octave.set_rf_output_gain(element, gain)
        
    def octave_switch(self, element, mode):
        """Mode can be: 
        <on> - always on
        <off> - always off
        <normal> - on when triggered
        <inverse> - off when triggered"""
        #TODO: add get mode after the get function will be implemented by qm
        if mode in ['on', 'On','ON']:
            self.qm.octave.set_rf_output_mode(element, RFOutputMode.on)
        elif mode in ['off', 'Off','OFF']:
            self.qm.octave.set_rf_output_mode(element, RFOutputMode.off)
        elif mode in ['normal', 'Normal']:
            self.qm.octave.set_rf_output_mode(element, RFOutputMode.trig_normal)
        elif mode in ['inverse', 'Inverse','inv','Inv']:
            self.qm.octave.set_rf_output_mode(element, RFOutputMode.trig_inverse)
        else:
            raise ValueError(f'Mode {mode} is not a valid RF-switch mode')
            
    def octave_calib(self, element):
        LO = self.configObject.elements[element].loFreq
        IF = self.configObject.elements[element].intermediateFreq
        self.qm.octave.calibrate_element(element, [(LO, IF)])
        self.qm =  self.qm_server.open_qm(self.config)
        #Fixed octave bug
        for sg_name, sg in self.configObject.mxgs.items():
            if sg.is_using_octave_externally: 
                for element, _ in sg.PairedMembers.items():
                    # self.set_octave_external_lo_freq(element)
                    self.set_octave_lo_source(element, 'External')
 #%% Readout Calibration
    
    def readout_calibration (self,N_avg = 100, ro_element = None,
                         fig_num = None):
        if ro_element is None: ro_element = self.main_readout
        with program() as prog:
            A = declare(fixed)
            n = declare(int)  
            I = declare(fixed)
            Q = declare(fixed)
            with for_(n, 0, n < N_avg, n + 1):

                reset_phase(ro_element)
                measure(self.ro_pulse, ro_element, 'samples',
                            demod.full('integ_w_I', I, 'out1'),
                            demod.full('integ_w_Q', Q, 'out2'))
        
        
                wait(int(self.wait_between_seq//4), ro_element)  # This wait time is needed to allow transferring samples
        
                save(I, 'I')
                save(Q, 'Q')
        self.last_prog = prog
        self.qm_server.clear_all_job_results()            
        job = self.qm.execute(prog, duration_limit=0, data_limit=0)
        self.last_job = job
        job.result_handles.wait_for_all_values()

        if fig_num is None:
            fig = plt.figure()
        else:
            fig = plt.figure(fig_num)
            
        self.rawI = job.result_handles.samples_input1.fetch_all()['value'].reshape((N_avg, -1)).mean(axis=0) / 4096
        self.rawQ = job.result_handles.samples_input2.fetch_all()['value'].reshape((N_avg, -1)).mean(axis=0) / 4096
        plt.subplot(211)
        plt.plot(self.rawI)
        plt.suptitle("TOF={0} nsec, Navg={1}".format(self.config['elements'][ro_element]['time_of_flight'],N_avg), fontsize=18)

        plt.xlabel("time, nsec", fontsize=14)
        plt.ylabel("ADC [-2048 to 2047]", fontsize=14)
        plt.title("Raw I", fontsize=14)
        plt.subplot(212)
        plt.plot(self.rawQ)
        plt.xlabel("time, nsec", fontsize=14)
        plt.ylabel("ADC [-2048 to 2047]", fontsize=14)
        plt.title("Raw Q", fontsize=14)
        print('The offset of I is - {} and the offset of Q is - {}'.format(-np.mean(self.rawI),-np.mean(self.rawQ)))
        return fig
    
    def run_paramp_phase_check(self, N_avg = 10000, ro_element = None, is_meas_vac = True, is_meas_disp = True):
        """ Runs a measurement of a squeezed vacuum state by pumping the paramp and of a vacuum state for reference.
            Returns the phase of the amplification"""
        n=1
        if is_meas_vac: n+=1
        if is_meas_disp: n+=2
        if ro_element is None: ro_element = self.main_readout
        run_time = n*N_avg*(self.pulse_len(ro_element,self.ro_pulse)+4*self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))            
        
        with program() as self.paramp_phase_check:
            n = declare(int)
            I_squeezed = declare(fixed)
            I_vac = declare(fixed)
            I_disp = declare(fixed)
            I_squeezed_disp = declare(fixed)
            Q_squeezed = declare(fixed)
            Q_disp = declare(fixed)
            Q_squeezed_disp = declare(fixed)
            Q_vac = declare(fixed)
                
            with for_(n, 0, n < N_avg, n + 1):
                
                reset_phase(self.main_paramp)
                reset_phase(ro_element)
                
                play(self.pump_pulse, self.main_paramp)
                measure(self.ro_pulse * amp(0), ro_element, None,
                        demod.full('integ_w_I', I_squeezed, 'out1'),
                        demod.full('integ_w_Q', Q_squeezed, 'out2'))
                
                save(I_squeezed, 'I_squeezed')
                save(Q_squeezed, 'Q_squeezed')
                
                wait(int(self.wait_between_seq//4), ro_element)
                
                if is_meas_disp:
                    align(ro_element, self.main_paramp)
                    reset_phase(self.main_paramp)
                    reset_phase(ro_element)
                    
                    play(self.pump_pulse, self.main_paramp)
                    measure(self.ro_pulse, ro_element, None,
                            demod.full('integ_w_I', I_squeezed_disp, 'out1'),
                            demod.full('integ_w_Q', Q_squeezed_disp, 'out2'))
                    
                    save(I_squeezed_disp, 'I_squeezed_disp')
                    save(Q_squeezed_disp, 'Q_squeezed_disp')
                    
                    wait(int(self.wait_between_seq//4), ro_element)
                    
                    reset_phase(ro_element)
                    
                    measure(self.ro_pulse, ro_element, None,
                            demod.full('integ_w_I', I_disp, 'out1'),
                            demod.full('integ_w_Q', Q_disp, 'out2'))
                    
                    save(I_disp, 'I_disp')
                    save(Q_disp, 'Q_disp')
                    
                    wait(int(self.wait_between_seq//4), ro_element)
                if is_meas_vac:
                    reset_phase(ro_element)
                    
                    measure(self.ro_pulse * amp(0), ro_element, None,
                            demod.full('integ_w_I', I_vac, 'out1'),
                            demod.full('integ_w_Q', Q_vac, 'out2'))
                    
                    save(I_vac, 'I_vac')
                    save(Q_vac, 'Q_vac')
                    
                    wait(int(self.wait_between_seq//4), ro_element)
                

        self.last_prog = self.paramp_phase_check 
        
        self.qm_server.clear_all_job_results()            
        job = self.qm.execute(self.paramp_phase_check, duration_limit=0, data_limit=0)
        job.result_handles.wait_for_all_values()
        
        I_squeezed = job.result_handles.get('I_squeezed').fetch_all()['value']
        Q_squeezed = job.result_handles.get('Q_squeezed').fetch_all()['value']
        
        _, I_units_prefix, I_factor = self.autoscale_data(I_squeezed)
        _, Q_units_prefix, Q_factor = self.autoscale_data(Q_squeezed)

        if I_factor > Q_factor:
            factor = I_factor
            Q_squeezed *= factor
            I_squeezed *= factor
            units_prefix = I_units_prefix
        else:
            factor = Q_factor
            Q_squeezed *= factor
            I_squeezed *= factor
            units_prefix = Q_units_prefix
        
        prog_name = 'Paramp Phase Check'
        fig = plt.figure(next_fig_num_by_name(prog_name))    
        ax = fig.add_subplot(111)
        plt.plot(I_squeezed, Q_squeezed, 'bo', alpha = 0.75, label = 'Squeezed', zorder = 1)
        print(f"Squeezed vacuum covariance is  {np.cov(I_squeezed, Q_squeezed)}")
        
        if is_meas_vac: 
            I_vac = job.result_handles.get('I_vac').fetch_all()['value'] * factor
            Q_vac = job.result_handles.get('Q_vac').fetch_all()['value'] * factor
            plt.plot(I_vac, Q_vac, 'ro', alpha = 0.25, label = 'Vacuum', zorder = 1)
            print(f"Vacuum covariance is  {np.cov(I_vac, Q_vac)}")
        if is_meas_disp:
            I_disp = job.result_handles.get('I_disp').fetch_all()['value'] * factor
            Q_disp = job.result_handles.get('Q_disp').fetch_all()['value'] * factor
            I_squeezed_disp = job.result_handles.get('I_squeezed_disp').fetch_all()['value'] * factor
            Q_squeezed_disp = job.result_handles.get('Q_squeezed_disp').fetch_all()['value'] * factor
            plt.plot(I_disp, Q_disp, 'go', alpha = 0.25, label = 'Displacement', zorder = 1)
            plt.plot(I_squeezed_disp, Q_squeezed_disp, 'mo', alpha = 0.25, label = 'Squeezed Displacement', zorder = 1)
            print(f"Displaced vacuum covariance is  {np.cov(I_disp, Q_disp)}")
            print(f"Squeezed displaced vacuum covariance is  {np.cov(I_squeezed_disp, Q_squeezed_disp)}")
            
        plt.xlabel(f"I [{units_prefix}V]", fontsize=18)
        plt.ylabel(f"Q [{units_prefix}V]", fontsize=18)
        ax.set_aspect(1)
        leg = plt.legend()
        leg.set_draggable(True)
        
        covMat = np.cov([I_squeezed,Q_squeezed])
        eigVals, eigVecs = np.linalg.eig(covMat)
        max_ind = np.where(eigVals==eigVals.max())[0][0]
        amp_phase = np.arctan(eigVecs[max_ind][1]/eigVecs[max_ind][0])/np.pi*180
        print(f'The phase of the amplified axis is {amp_phase} degrees with respect to the frame of the OPX')
        
        return amp_phase
        
    def run_paramp_phase_calib(self, start, stop, npts, N_avg = 1000, ro_element = None, which_data = None, is_ramp_up = True, is_plot_all = True, **kwargs):
        """ Sweeps over the phase of the pump to find the best readout fidelity using pinopi.
            Returns the phase that gives the best fidelity.
            Will use the default data type, unless which_data is passed. Will not change the attributed which_data"""
        
        if ro_element is None: ro_element = self.main_readout
        run_time = 2*npts*N_avg*(self.pulse_len(ro_element,self.ro_pulse)+4*self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))            
        
        phase_list = np.linspace(start, stop, npts)
        
        print('The phase of the readout pulse is set to 0 momentarily')
        old_phase = self.pulse_phase(ro_element, self.ro_pulse)
        self.pulse_phase(ro_element, self.ro_pulse, 0)
        
        with program() as self.paramp_phase_calib_prog:
            n = declare(int)
            phase=declare(fixed)
            I_pi = declare(fixed)
            I_nopi = declare(fixed)
            Q_pi = declare(fixed)
            Q_nopi = declare(fixed)
                
            with for_(n, 0, n < N_avg, n + 1):
                with for_each_(phase, (phase_list*np.pi/180).tolist()):
                    reset_frame(ro_element)
                    reset_frame(self.main_paramp)
                    self._pi_pulse(qubit = self.main_qubit,**kwargs)
                    
                    align(self.main_qubit, ro_element, self.main_paramp)
                    reset_phase(self.main_paramp)
                    reset_phase(ro_element)
                    frame_rotation(phase, ro_element)
                    
                    if is_ramp_up and self.configObject.SuperMembers[self.main_paramp].PulseParamsDict[self.pump_pulse].Additions.ramp['up'] is not None: play('ramp_up', self.main_paramp)
                    if is_ramp_up and self.configObject.SuperMembers[ro].PulseParamsDict[self.ro_pulse].Additions.ramp['up'] is not None: play('ramp_up', ro_element)
                    play(self.pump_pulse, self.main_paramp)
                    measure(self.ro_pulse, ro_element, None,
                            demod.full('integ_w_I', I_pi, 'out1'),
                            demod.full('integ_w_Q', Q_pi, 'out2'))
                    
                    wait(int(self.wait_between_seq//4), ro_element)  # reset time
        
                    save(I_pi, 'I_pi')
                    save(Q_pi, 'Q_pi')
                    reset_frame(ro_element)
                    reset_frame(self.main_paramp)
                    
                    align(ro_element, self.main_paramp)
                    reset_phase(self.main_paramp)
                    reset_phase(ro_element)
                    frame_rotation(phase, ro_element)
                    
                    if is_ramp_up and self.configObject.SuperMembers[self.main_paramp].PulseParamsDict[self.pump_pulse].Additions.ramp['up'] is not None: play('ramp_up', self.main_paramp)
                    if is_ramp_up and self.configObject.SuperMembers[ro].PulseParamsDict[self.ro_pulse].Additions.ramp['up'] is not None: play('ramp_up', ro_element)
                    play(self.pump_pulse, self.main_paramp)
                    measure(self.ro_pulse, ro_element, None,
                            demod.full('integ_w_I', I_nopi, 'out1'),
                            demod.full('integ_w_Q', Q_nopi, 'out2'))
                    
                    wait(int(self.wait_between_seq//4), ro_element)  # reset time
        
                    save(I_nopi, 'I_nopi')
                    save(Q_nopi, 'Q_nopi')
                    
        self.last_prog = self.paramp_phase_calib_prog  
                
        self.qm_server.clear_all_job_results()            
        job = self.qm.execute(self.paramp_phase_calib_prog, duration_limit=0, data_limit=0)
        job.result_handles.wait_for_all_values()
        
        print(f'The phase of the readout pulse is reset to its previouse value {old_phase}')
        self.pulse_phase(ro_element, self.ro_pulse, old_phase)
            
        I_pi = job.result_handles.get('I_pi').fetch_all()['value'].reshape((N_avg, npts))
        Q_pi = job.result_handles.get('Q_pi').fetch_all()['value'].reshape((N_avg, npts))
        I_nopi = job.result_handles.get('I_nopi').fetch_all()['value'].reshape((N_avg, npts))
        Q_nopi = job.result_handles.get('Q_nopi').fetch_all()['value'].reshape((N_avg, npts))
        
        _, I_units_prefix, I_factor = self.autoscale_data(I_nopi)
        _, Q_units_prefix, Q_factor = self.autoscale_data(Q_nopi)
        
        if I_factor > Q_factor:
            factor = I_factor
            units_prefix = I_units_prefix
        else:
            factor = Q_factor
            units_prefix = Q_units_prefix
        I_pi *= factor
        I_nopi *= factor
        Q_pi *= factor
        Q_nopi *= factor
            
        fdlty_list=[]
        
        max_I = max([I_pi.max(), I_nopi.max()])
        min_I = min([I_pi.min(), I_nopi.min()])
        max_Q = max([Q_pi.max(), Q_nopi.max()])
        min_Q = min([Q_pi.min(), Q_nopi.min()])
        
        old_which_data = self.which_data
        if which_data is not None:
            self.which_data = which_data
        
        if is_plot_all: 
            prog_name = 'Paramp Phase Calibration All'
            plt.figure(next_fig_num_by_name(prog_name))
            print('==========================\n Press any key to view next phase\n==========================')
            
        for i_pi, q_pi, i_nopi, q_nopi, j in zip(I_pi.T, Q_pi.T, I_nopi.T, Q_nopi.T, range(npts)):
            data_pi = self.determine_data(i_pi, q_pi)
            data_nopi = self.determine_data(i_nopi, q_nopi)
            
            avg_ro = np.mean([data_pi,data_nopi])
            fdlty = 1
            
            for i in range(len(data_pi)):
                if np.mean(data_pi) > np.mean(data_nopi):
                    if data_pi[i] <= avg_ro: fdlty = fdlty - 1/len(data_pi)/2
                    if data_nopi[i] >= avg_ro: fdlty = fdlty - 1/len(data_nopi)/2
                else:
                    if data_pi[i] >= avg_ro: fdlty = fdlty - 1/len(data_pi)/2
                    if data_nopi[i] <= avg_ro: fdlty = fdlty - 1/len(data_nopi)/2
            fdlty_list.append(fdlty)
            
            if is_plot_all:
                plt.clf()
                ax=plt.subplot(111)
                plt.xlim([min_I,max_I])
                plt.ylim([min_Q,max_Q])
                plt.plot(i_pi, q_pi, 'ob', alpha = 0.5)
                plt.plot(i_nopi, q_nopi, 'or', alpha = 0.5)
                plt.title(f'Phase = {phase_list[j]}')
                plt.xlabel(f"I [{units_prefix}V]", fontsize=18)
                plt.ylabel(f"Q [{units_prefix}V]", fontsize=18)
                ax.set_aspect(1)
                plt.pause(0.1)
                plt.waitforbuttonpress()
        if is_plot_all: plt.close()
                
        self.which_data = old_which_data
        
        prog_name = 'Paramp Phase Calibration'
        fig = plt.figure(next_fig_num_by_name(prog_name))            
        plt.plot(phase_list, fdlty_list, 'o-')
        plt.xlabel(r'Phase$^{\circ}$')
        plt.ylabel('Readout Fidelity')
        max_fdlty = max(fdlty_list)
        max_ind = np.where(np.array(fdlty_list)==max_fdlty)[0][0]
        print(f"The best readout fidelity is {max_fdlty} for a phase of {phase_list[max_ind]}")
        return phase_list[max_ind]
    
    def run_paramp_amp_calib(self, start, stop, npts, N_avg = 1000, ro_element = None, which_data = None, is_ramp_up = True, is_plot_all = True, **kwargs):
        """ Sweeps over the amp of the pump to find the best readout fidelity using pinopi.
            Returns the amp that gives the best fidelity.
            Will use the default data type, unless which_data is passed. Will not change the attributed which_data"""
        
        if ro_element is None: ro_element = self.main_readout
        run_time = 2*npts*N_avg*(self.pulse_len(ro_element,self.ro_pulse)+4*self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))            
        
        amp_list = np.linspace(start, stop, npts)
        amp_list_for_prog = amp_list/stop
        print(f'The amp of the pump pulse is set to {stop} momentarily')
        old_amp = self.pulse_amp(self.main_paramp, self.pump_pulse)
        self.pulse_amp(self.main_paramp, self.pump_pulse, stop, keep_phase = True)
        
        with program() as self.paramp_amp_calib:
            n = declare(int)
            a=declare(fixed)
            I_pi = declare(fixed)
            I_nopi = declare(fixed)
            Q_pi = declare(fixed)
            Q_nopi = declare(fixed)
                
            with for_(n, 0, n < N_avg, n + 1):
                with for_each_(a, amp_list_for_prog.tolist()):
                    
                    self._pi_pulse(qubit = self.main_qubit,**kwargs)
                    
                    align(self.main_qubit, ro_element, self.main_paramp)
                    reset_phase(self.main_paramp)
                    reset_phase(ro_element)
                    
                    if is_ramp_up and self.configObject.SuperMembers[self.main_paramp].PulseParamsDict[self.pump_pulse].Additions.ramp['up'] is not None: play('ramp_up' * amp(a), self.main_paramp)
                    if is_ramp_up and self.configObject.SuperMembers[ro].PulseParamsDict[self.ro_pulse].Additions.ramp['up'] is not None: play('ramp_up', ro_element)
                    play(self.pump_pulse * amp(a), self.main_paramp)
                    measure(self.ro_pulse, ro_element, None,
                            demod.full('integ_w_I', I_pi, 'out1'),
                            demod.full('integ_w_Q', Q_pi, 'out2'))
                    
                    wait(int(self.wait_between_seq//4), ro_element)  # reset time
        
                    save(I_pi, 'I_pi')
                    save(Q_pi, 'Q_pi')
    
                    align(ro_element, self.main_paramp)
                    reset_phase(self.main_paramp)
                    reset_phase(ro_element)
                    
                    if is_ramp_up and self.configObject.SuperMembers[self.main_paramp].PulseParamsDict[self.pump_pulse].Additions.ramp['up'] is not None: play('ramp_up' * amp(a), self.main_paramp)
                    if is_ramp_up and self.configObject.SuperMembers[ro].PulseParamsDict[self.ro_pulse].Additions.ramp['up'] is not None: play('ramp_up', ro_element)
                    play(self.pump_pulse * amp(a), self.main_paramp)
                    measure(self.ro_pulse, ro_element, None,
                            demod.full('integ_w_I', I_nopi, 'out1'),
                            demod.full('integ_w_Q', Q_nopi, 'out2'))
                    
                    wait(int(self.wait_between_seq//4), ro_element)  # reset time
        
                    save(I_nopi, 'I_nopi')
                    save(Q_nopi, 'Q_nopi')
                    
        self.last_prog = self.paramp_amp_calib  
                
        self.qm_server.clear_all_job_results()            
        job = self.qm.execute(self.paramp_amp_calib, duration_limit=0, data_limit=0)
        job.result_handles.wait_for_all_values()
        
        print(f'The amp of the pump pulse is reset to its previouse value {old_amp}')
        self.pulse_phase(self.main_paramp, self.pump_pulse, old_amp)
            
        I_pi = job.result_handles.get('I_pi').fetch_all()['value'].reshape((N_avg, npts))
        Q_pi = job.result_handles.get('Q_pi').fetch_all()['value'].reshape((N_avg, npts))
        I_nopi = job.result_handles.get('I_nopi').fetch_all()['value'].reshape((N_avg, npts))
        Q_nopi = job.result_handles.get('Q_nopi').fetch_all()['value'].reshape((N_avg, npts))
        
        _, I_units_prefix, I_factor = self.autoscale_data(I_nopi)
        _, Q_units_prefix, Q_factor = self.autoscale_data(Q_nopi)
        
        if I_factor > Q_factor:
            factor = I_factor
            units_prefix = I_units_prefix
        else:
            factor = Q_factor
            units_prefix = Q_units_prefix
        I_pi *= factor
        I_nopi *= factor
        Q_pi *= factor
        Q_nopi *= factor
            
        fdlty_list=[]
        
        max_I = max([I_pi.max(), I_nopi.max()])
        min_I = min([I_pi.min(), I_nopi.min()])
        max_Q = max([Q_pi.max(), Q_nopi.max()])
        min_Q = min([Q_pi.min(), Q_nopi.min()])
        
        old_which_data = self.which_data
        if which_data is not None:
            self.which_data = which_data
        
        if is_plot_all: 
            prog_name = 'Paramp Amp Calibration All'
            plt.figure(next_fig_num_by_name(prog_name))
            print('==========================\n Press any key to view next phase\n==========================')
            
        for i_pi, q_pi, i_nopi, q_nopi, j in zip(I_pi.T, Q_pi.T, I_nopi.T, Q_nopi.T, range(npts)):
            data_pi = self.determine_data(i_pi, q_pi)
            data_nopi = self.determine_data(i_nopi, q_nopi)
            
            avg_ro = np.mean([data_pi,data_nopi])
            fdlty = 1
            
            for i in range(len(data_pi)):
                if np.mean(data_pi) > np.mean(data_nopi):
                    if data_pi[i] <= avg_ro: fdlty = fdlty - 1/len(data_pi)/2
                    if data_nopi[i] >= avg_ro: fdlty = fdlty - 1/len(data_nopi)/2
                else:
                    if data_pi[i] >= avg_ro: fdlty = fdlty - 1/len(data_pi)/2
                    if data_nopi[i] <= avg_ro: fdlty = fdlty - 1/len(data_nopi)/2
            fdlty_list.append(fdlty)
            
            if is_plot_all:
                plt.clf()
                ax=plt.subplot(111)
                plt.xlim([min_I,max_I])
                plt.ylim([min_Q,max_Q])
                plt.plot(i_pi, q_pi, 'ob', alpha = 0.5)
                plt.plot(i_nopi, q_nopi, 'or', alpha = 0.5)
                plt.title(f'Amp = {amp_list[j]}')
                plt.xlabel(f"I [{units_prefix}V]", fontsize=18)
                plt.ylabel(f"Q [{units_prefix}V]", fontsize=18)
                ax.set_aspect(1)
                plt.pause(0.1)
                plt.waitforbuttonpress()
        if is_plot_all: plt.close()
                
        self.which_data = old_which_data
        
        prog_name = 'Paramp Amp Calibration'
        fig = plt.figure(next_fig_num_by_name(prog_name))            
        plt.plot(amp_list, fdlty_list, 'o-')
        plt.xlabel('Amp [V]')
        plt.ylabel('Readout Fidelity')
        max_fdlty = max(fdlty_list)
        max_ind = np.where(np.array(fdlty_list)==max_fdlty)[0][0]
        print(f"The best readout fidelity is {max_fdlty} for an amp of {amp_list[max_ind]}")
        return amp_list[max_ind]



    def load_TWPA_pump_sweep(self, mxg, N_avg = 1000, ro_element=None, freq_start=None, freq_stop=None, freq_npts=1, power_start=None, power_stop=None, power_npts=1, ro_pulse = None, wait_time = 5e3, extra_dwell_time = 1e6, qubit = None):
        if power_npts * freq_npts > 3201:
            raise ValueError(f"Total npts cannot exceed 3201 when sweeping over both power and frequency. Got power_npts*freq_npts={power_npts*freq_npts}")
            
        if ro_element is None: ro_element = self.main_readout
        if ro_pulse is None: ro_pulse = self.ro_pulse
        
        if power_npts == 1 and freq_npts == 1:
            sweep_type = 'const'
            shape = (N_avg,1)
            powers = power_start
            freqs = freq_start
        
        elif power_npts == 1:
            freqs = np.linspace(freq_start, freq_stop, freq_npts)
            sweep_type = 'freq'
            if power_start is None:
                powers = mxg.power()
            else:
                powers = power_start
                mxg.power(powers)
        elif freq_npts == 1:
            powers = np.linspace(power_start, power_stop, power_npts)
            sweep_type = 'power'
            if freq_start is None:
                freqs = mxg.freq()
            else:
                freqs = freq_start
                mxg.freq(freqs)
        else:
            sweep_type = 'power_and_freq'
            powers = np.linspace(power_start, power_stop, power_npts)
            freqs = np.linspace(freq_start, freq_stop, freq_npts)
        
        dwell_time = ((wait_time + self.pulse_len(ro_element, ro_pulse)) * N_avg + extra_dwell_time) * 1e-9
        if dwell_time < 100e-6: 
            dwell_time = 100e-6
            print('\n Set the MXG dwell time to the minimum 100 us\n')
        self.results['TWPA_pump_sweep'] = {'powers': powers, 'freqs': freqs, 'sweep_type': sweep_type,
                                           'N_avg': N_avg,
                                           'ro_element': ro_element, 'ro_pulse': ro_pulse,
                                           'dwell_time': dwell_time,
                                           'qubit': qubit}
        run_time = (dwell_time) * freq_npts * power_npts
        print('Run time is {}s'.format(np.round(run_time)))
        
        with program() as self.TWPA_pump_sweep_prog:
            I=declare(fixed)
            Q=declare(fixed)
            n=declare(int)
            i=declare(int)
            
            with for_(i,0,i<int(freq_npts*power_npts),i+1):
                if sweep_type!='const': wait_for_trigger(ro_element)
                with for_(n,0,n<N_avg,n+1):
                    if qubit is not None:
                        for i in range(1):
                            play('pi_pulse', qubit)
                        align(qubit, ro_element)
                    self.perform_full_measurement(I,Q,ro_element=ro_element,ro_pulse=ro_pulse,is_wait = False, is_active_reset = False)
                    wait(int(wait_time//4), ro_element)
                
    def run_TWPA_pump_sweep(self, mxg, trigger = 'TRIG2'):
        powers = self.results['TWPA_pump_sweep']['powers']
        freqs = self.results['TWPA_pump_sweep']['freqs']
        N_avg = self.results['TWPA_pump_sweep']['N_avg']
        sweep_type = self.results['TWPA_pump_sweep']['sweep_type']
        ro_element = self.results['TWPA_pump_sweep']['ro_element']
        dwell_time = self.results['TWPA_pump_sweep']['dwell_time']
        
        if sweep_type == 'power_and_freq': 
            shape = (len(freqs), len(powers), N_avg)
            dwells = [dwell_time]
            powers_grid, freqs_grid = np.meshgrid(powers,freqs)
            mxg.visa.write(':LIST:TYPE LIST') # selects a list sweep
            mxg.sweep_mode('PWR_FREQ')
            mxg.set_list_sweep(freqs_grid.flatten(),powers_grid.flatten(),dwells)
            mean_axis = 2
        elif sweep_type == 'power': 
            mxg.visa.write(':LIST:TYPE STEP') # selects a step sweep
            shape = (len(powers), N_avg)
            mxg.power(powers)
            mxg.sweep_mode('PWR')
            mxg.freq(freqs)    
            mxg.set_dwell(dwell_time)
            mxg.Num_SWE_PT(len(powers))
            mean_axis = 1
        elif sweep_type == 'freq': 
            shape = (len(freqs), N_avg)
            mxg.visa.write(':LIST:TYPE STEP') # selects a step sweep
            mxg.power(powers)
            mxg.sweep_mode('FREQ')
            mxg.set_freq_start(freqs[0])    
            mxg.set_freq_stop(freqs[-1])    
            mxg.set_dwell(dwell_time)
            mxg.Num_SWE_PT(len(freqs))
            mean_axis = 1
        elif sweep_type == 'const':
            shape = (1,N_avg)
            mxg.power(powers)
            mxg.freq(freqs)
            mean_axis = 1
            
        else: raise ValueError("Unkown sweep type")
        
        
        mxg.triger_source('IMM') #set SG trigering to IMM
        mxg.EXT_trig_source(trigger)
        mxg.on()
        
        self.qm_server.clear_all_job_results()
        job = self.qm.execute(self.TWPA_pump_sweep_prog, duration_limit=0, data_limit=0)
        
        mxg.start_sweep('SING')
        
        job.result_handles.wait_for_all_values()
        mxg.off()
        self.last_prog = self.TWPA_pump_sweep_prog
        self.last_job = job
        
        I_res = job.result_handles.get('I').fetch_all()['value'].reshape(shape)
        Q_res = job.result_handles.get('Q').fetch_all()['value'].reshape(shape)
                
        if sweep_type == 'const':
            is_mean = False
        else:
            is_mean = True
        readout_amplitude, readout_amplitude_err = self.process_data([I_res,Q_res], which_data = 'Mag', mean_axis = mean_axis, is_mean=is_mean)
        
        self.results['TWPA_pump_sweep']['readout_amplitude'] = readout_amplitude.transpose()
        self.results['TWPA_pump_sweep']['readout_amplitude_err'] = readout_amplitude_err.transpose()
        
        self.plot_TWPA_pump_sweep()
        
    def plot_TWPA_pump_sweep(self):
        powers = self.results['TWPA_pump_sweep']['powers']
        freqs = self.results['TWPA_pump_sweep']['freqs']
        N_avg = self.results['TWPA_pump_sweep']['N_avg']
        sweep_type = self.results['TWPA_pump_sweep']['sweep_type']
        ro_element = self.results['TWPA_pump_sweep']['ro_element']
        
        readout_amplitude, prefix, factor = self.autoscale_data(self.results['TWPA_pump_sweep']['readout_amplitude'])
        readout_amplitude_err = self.results['TWPA_pump_sweep']['readout_amplitude_err']*factor
        
        if sweep_type == 'power_and_freq':
            fig, axs = plt.subplots(2,1,num = next_fig_num_by_name('TWAPA Pump Sweep'), sharex=True)
            plot_2D(readout_amplitude, freqs*1e-9, powers, xlabel = None, ylabel = 'Pump power [dBm]', zlabel = f'Mag [{prefix}V]', ax=axs[0], cmap = 'Reds')
            plot_2D(readout_amplitude_err, freqs*1e-9, powers, xlabel = 'Pump frequency [GHz]', ylabel = 'Pump power [dBm]', zlabel = f'Mag error [{prefix}V]', ax=axs[1], cmap = 'Reds')
        elif sweep_type == 'power':
            fig, ax = plt.subplots(num = next_fig_num_by_name('TWAPA Pump Sweep'))
            plt.errorbar(powers, readout_amplitude, readout_amplitude_err) 
            plt.xlabel('Pump power [dBm]')
            plt.ylabel(f'Readout amplitude [{prefix}V]')
        elif sweep_type == 'freq':
            fig, ax = plt.subplots(num = next_fig_num_by_name('TWAPA Pump Sweep'))
            plt.errorbar(freqs*1e-9, readout_amplitude, readout_amplitude_err) 
            plt.xlabel('Pump frequency [GHz]')
            plt.ylabel(f'Readout amplitude [{prefix}V]')
        elif sweep_type == 'const':
            fig, ax = plt.subplots(num = next_fig_num_by_name('TWAPA Pump Sweep'))
            plt.hist(readout_amplitude, bins = 100, histtype = 'step')
            plt.xlabel(f'Readout amplitude [{prefix}V]')
            plt.ylabel('Counts')
            readout_amplitude, readout_amplitude_err = self.process_data([readout_amplitude,readout_amplitude*0], which_data = 'I', mean_axis = 0, is_mean=True)
            readout_amplitude, readout_amplitude_err = round_value_by_error(readout_amplitude[0], readout_amplitude_err[0])
            print(f'{readout_amplitude}+-{readout_amplitude_err} {prefix}V')
        else:
            raise ValueError(f"Unkown sweep_type. Got {sweep_type}")
        
        plt.tight_layout()
        
    def hp_readout_amp_calib(self, N_avg = 1000, start_amp = 0.0, stop_amp = 0.45, npts = 10, ro_duration = 3200, qubits = ['qubit1'], ro_element = None, set_amp = True): 
        """ sweep over amplitudes to find the best amp to distinguish between states. Only for transmision - measures the output magnitude."""    
        
        if ro_element is None: ro_element = self.main_readout
        if npts <2: raise ValueError("number of points must be greater than 1")
        
        amps = ( 2.5 *  np.linspace(start_amp, stop_amp, npts)).tolist() 
        if self.ro_pulse == 'hp_ro_pulse':
            current_ro_amp = self.config['waveforms']['wf_hp_' + ro]['sample']
            self.set_hp_readout_amplitude(ro_element, 0.4) # * amp(a) has range of (-2,2) so first we set the amp to 0.4 and then reduce it, so amp(a) isn't too big
        else: 
            current_ro_amp = self.config['waveforms']['wf_'+ro]['sample']
            self.set_readout_amplitude(ro_element, 0.4) # * amp(a) has range of (-2,2) so first we set the amp to 0.4 and then reduce it, so amp(a) isn't too big

        
        with program() as prog:
            
            n = declare(int)
            i = declare(int)
            
            a = declare(fixed)

            I = declare(fixed)
            Q = declare(fixed)

            with for_(n, 0, n < N_avg, n + 1): # loop over averages
                with for_each_(a, amps): # loop over amps
                    with for_(i, 0, i <= 2 * len(qubits) - 1 , i+1): # loop over states
                        for j in range(len(qubits)):
                            with if_((i == j + 1) | (i == 2 * len(qubits) - 1)):
                                self._pi_pulse(args, kwargs, qubit= qubits[j])
                                align(*qubits)
                        align(*qubits, ro_element)
                        measure(self.ro_pulse * amp(a), ro_element, None,
                                demod.full("integ_w_I", I, "out1"),
                                demod.full('integ_w_Q', Q, 'out2'))
                        save(I, 'I')
                        save(Q, 'Q')
                        wait(int(self.wait_between_seq//4), ro_element)  # reset time
        try:
            run_time = (2 * len(qubits)) * N_avg * npts * (ro_duration + 2 * self.config['pulses']['pi2_pulse_qb1_in']['length'] + 4 * self.wait_between_seq)
            print('Run time is {}s'.format(np.round(run_time * 1e-9)))
        except:
            warn('unknown run time')
        self.qm_server.clear_all_job_results()            
        job = self.qm.execute(prog, duration_limit=0, data_limit=0)
        job.result_handles.wait_for_all_values()

        I = job.result_handles.get('I').fetch_all()['value'].reshape((N_avg, -1)).mean(axis=0).reshape((npts,-1))
        Q = job.result_handles.get('Q').fetch_all()['value'].reshape((N_avg, -1)).mean(axis=0).reshape((npts,-1))
        
        fig = plt.figure()
        amps_for_fig = np.array(amps) / 2.5
        mags = []; max_sep = []; max_ind = []
        clrs = ['b','r','c','m']

        for i in range(2 * len(qubits)):
            mags.append(np.sqrt(I[:,i]**2 + Q[:,i]**2))
            max_sep.append(max(mags[i] - mags[0]))
            max_ind.append(np.where(mags[i] - mags[0]  == max_sep[i]))
            plt.plot(amps_for_fig, mags[i], clrs[i])
            
        for i in range(1, 2 * len(qubits)):
            if max_sep[i] == min(max_sep[1:2 * len(qubits)]): 
                if set_amp: print('setting the readout amp for best separation from 00 at amp = {}'.format(amps_for_fig[max_ind[i]][0]))
                else: print('readout amp for best separation from 00 at amp = {}'.format(amps_for_fig[max_ind[i]][0]))
                
        for i in range(2 * len(qubits)):
            plt.plot([amps_for_fig[max_ind[i]],amps_for_fig[max_ind[i]]], [mags[0][max_ind[i]], mags[i][max_ind[i]]], '--' + clrs[i])
        if len(qubits) == 1: lgd = ['0','1']
        elif len(qubits) == 2: lgd = ['00','01','10','11']
        else: lgd = 'you have {} qubits - Well Done! edit this code for your legend'.format(len(qubits))

        plt.legend(lgd)
        
        if self.ro_pulse == 'hp_ro_pulse':
            if set_amp : self.set_hp_readout_amplitude(ro_element, hp_amplitude = amps_for_fig[max_ind[i]][0])
            else: self.set_hp_readout_amplitude(ro_element, hp_amplitude = current_ro_amp)
        else:
            if set_amp : self.set_readout_amplitude(ro_element, amplitude = amps_for_fig[max_ind[i]][0])
            else: self.set_readout_amplitude(ro_element, amplitude = current_ro_amp)
        return fig
      
    
    def rotation_mat_from_pulse_name(self, pulse_name, qubit):
        pi_to_pi2_ratio = abs(self.configObject.SuperMembers[self.main_qubit].PulseParamsDict[self.pi2_pulse].amp) / abs(self.configObject.SuperMembers[self.main_qubit].PulseParamsDict[self.pi_pulse].amp)
        rot_dict = {
            '+I':(0,0,0,0),
            '-I':(0,0,0,0),
            '+X': (1,0,0,1),
            '+x': (pi_to_pi2_ratio,0,0,pi_to_pi2_ratio),
            '-x': (-pi_to_pi2_ratio,0,0, -pi_to_pi2_ratio),
            '-X': (-1,0,0, -1),
            '+y': (0,pi_to_pi2_ratio,-pi_to_pi2_ratio,0),
            '+Y': (0,1,-1,0),
            '-y': (0,-pi_to_pi2_ratio,pi_to_pi2_ratio,0),
            '-Y': (0,-1,1,0),
            }
        return rot_dict[pulse_name]
    
    def rotation_mat_from_pulse_name(self, pulse_name, qubit):
        pi_to_pi2_ratio = abs(self.configObject.SuperMembers[self.main_qubit].PulseParamsDict[self.pi2_pulse].amp) / abs(self.configObject.SuperMembers[self.main_qubit].PulseParamsDict[self.pi_pulse].amp)
        rot_dict = {
            '+I':(0,0,0,0),
            '-I':(0,0,0,0),
            '+X': (1,0,0,1),
            '+x': (pi_to_pi2_ratio,0,0,pi_to_pi2_ratio),
            '-x': (-pi_to_pi2_ratio,0,0, -pi_to_pi2_ratio),
            '-X': (-1,0,0, -1),
            '+y': (0,pi_to_pi2_ratio,-pi_to_pi2_ratio,0),
            '+Y': (0,1,-1,0),
            '-y': (0,-pi_to_pi2_ratio,pi_to_pi2_ratio,0),
            '-Y': (0,-1,1,0),
            }
        return rot_dict[pulse_name]
    
        
    def find_1q_beta_coeffs(self, N_avg = 10000, plot = False):
        print('Not made yet')
        
    
#%%

    def load_sideband_cooling_reset_calib(self, phase_start, phase_stop, phase_npts, amp_start, amp_stop, amp_npts, N_avg = 1000, ro_element = None, qubit = None, **kwargs):
        
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        
        self.sideband_cooling_reset_calib_results = {}
        self.sideband_cooling_reset_calib_results['N_avg'] = N_avg
        self.sideband_cooling_reset_calib_results['qubit'] = qubit
        self.sideband_cooling_reset_calib_results['ro_element'] = ro_element
        
        run_time = N_avg*phase_npts*amp_npts * (self.pulse_len(ro_element, self.ro_pulse) + self.pulse_len(qubit, 'rabi_sideband_cooling_pulse'))
        print('Run time is {}s'.format(round(run_time * 1e-9)))    
        
        phase_list = np.linspace(phase_start, phase_stop, phase_npts)
        amp_list = np.linspace(amp_start, amp_stop, amp_npts)
        amp_scale_list = np.linspace(amp_start/amp_stop, 1, amp_npts)
        self.sideband_cooling_reset_calib_results['phase_list'] = phase_list
        self.sideband_cooling_reset_calib_results['amp_list'] = amp_list
        
        with program() as self.sideband_cooling_reset_calib_prog:
        
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            phase = declare(fixed)
            pi2_amp_scale = declare(fixed)
            
            with for_(n, 0, n < N_avg, n + 1): 
                with for_each_(phase, (phase_list/360).astype(float).tolist()):
                    with for_each_(pi2_amp_scale, amp_scale_list.astype(float).tolist()):
                        reset_phase(qubit)
                        reset_frame(qubit)
                        self.sideband_cool(qubit = qubit, ro_element = ro_element, is_pi2 = True, extra_phase = phase, pi2_amp_scale = pi2_amp_scale, **kwargs)
    
                        align(qubit, ro_element)
                        
                        self.perform_full_measurement(I,Q, ro_element = ro_element, is_wait = False)
                    
        self.last_prog = self.sideband_cooling_reset_calib_prog
        
    def run_sideband_cooling_reset_calib(self, is_save_data = None, **kwargs):
        prog_name = 'Sideband Cooling Phase'
        N_avg = self.sideband_cooling_reset_calib_results['N_avg']
        phase_npts = len(self.sideband_cooling_reset_calib_results['phase_list'])
        amp_npts = len(self.sideband_cooling_reset_calib_results['amp_list'])
        amp_stop = self.sideband_cooling_reset_calib_results['amp_list'][-1]
        qubit = self.sideband_cooling_reset_calib_results['qubit']
        shape = (N_avg, phase_npts, amp_npts)
        
        
        try:
            if amp_npts > 1:
                old_amp = self.pulse_amp(qubit, 'pi2_reset_pulse')
                self.pulse_amp(qubit, 'pi2_reset_pulse', amp_stop, keep_phase = True)
            self.sideband_cooling_reset_calib_results['I'], self.sideband_cooling_reset_calib_results['Q'] = self.run_prog(self.sideband_cooling_reset_calib_prog, shape, **kwargs)
        except:
            if amp_npts > 1: self.pulse_amp(qubit, 'pi2_reset_pulse', old_amp)
        fig_num = next_fig_num_by_name(prog_name)
        if is_save_data is None: is_save_data = self.is_save_data
        if is_save_data: self.pickle_save(self.sideband_cooling_reset_calib_results, 'Sideband Cooling Phase')
        return self.plot_sideband_cooling_reset_calib(fig_num = fig_num, prog_name = prog_name, **kwargs)
        
    def plot_sideband_cooling_reset_calib(self, fig_num = None, is_calc_stat_error = None, **kwargs):
        title_str =f'Sideband Cooling Phase ({self.sideband_cooling_reset_calib_results["qubit"]})'
        
        phase_list = self.sideband_cooling_reset_calib_results['phase_list']
        amp_list = self.sideband_cooling_reset_calib_results['amp_list']
        
        if is_calc_stat_error is None: is_calc_stat_error = self.is_calc_stat_error
        
        if len(phase_list) == 1:
            x = amp_list
            xlabel = 'Amp [V]'
            is_2D = False
        elif len(amp_list) == 1:
            x = phase_list
            xlabel = r'Phase$^{\circ}$'
            is_2D = False
        else:
            x = phase_list
            xlabel = r'Phase$^{\circ}$'
            y = amp_list
            ylabel = 'Amp [V]'
            is_2D = True
        
        if not is_2D:
            fig, ax = plt.subplots(num = fig_num)
            y, y_err = self.process_data(self.sideband_cooling_reset_calib_results, is_calc_stat_error = is_calc_stat_error, **kwargs)
            y  = y.flatten()
            y_err  = y_err.flatten()
            if self.which_data != 'Phase':
                scaled_y, y_units_prefix, y_scaling_factor = scale_data_units(y)
            plt.errorbar(x,scaled_y, y_scaling_factor * y_err, fmt = '--or', capsize = 5, markersize = 6, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1),)
            plt.xlabel(xlabel)
            plt.ylabel(f'{self.which_data} [{y_units_prefix}V]')
        else:
            z, z_err = self.process_data(self.sideband_cooling_reset_calib_results, is_calc_stat_error = is_calc_stat_error, **kwargs)
            scaled_z, z_units_prefix, z_scaling_factor = scale_data_units(z)
            
            plot_2D(np.transpose(z),x,y, xlabel = xlabel, ylabel = ylabel)
            
            
            
            
            
    
#%% tomography
    
    def load_tomography(self, prepare = None, N_avg = 1000, qubit = None, ro_element = None, is_sb_cool = False, is_active_reset = False, **kwargs):
        if qubit is None: qubit = self.main_qubit
        if ro_element is None: ro_element = self.main_readout
        if type(qubit) != list:
            self.load_one_qubit_tomography(prepare=prepare, N_avg=N_avg, qubit=qubit, ro_element=ro_element, is_sb_cool=is_sb_cool, is_active_reset = is_active_reset, **kwargs)

    def load_one_qubit_tomography(self, N_avg, qubit, ro_element, prepare = None, prepare_dict = {}, is_active_reset_calib = None, is_sb_cool_calib = None, **kwargs):
        self.tomo = OneQubitTomo(qubit = qubit)
        self.tomography_results = {"N_avg": N_avg,
                                   "qubit": qubit
                                   }
        # if is_active_reset: run_time = N_avg*(self.wait_between_seq+len(self.tomo.pulse_seq_list)*(self.wait_between_seq))
        # else: run_time = N_avg*(self.wait_between_seq+len(self.tomo.pulse_seq_list)*(self.wait_between_seq))
        # print('Run time is {} minutes'.format(np.around(run_time * 1e-9 / 60,1)))
        calib_kwargs = kwargs.copy()
        if 'is_active_reset' in calib_kwargs.keys(): del calib_kwargs['is_active_reset']
        if 'is_sb_cool' in calib_kwargs.keys(): del calib_kwargs['is_sb_cool']
        with program() as self.tomography_prog:
            
            self.tomo.declarations()
            
            I_betas = declare(fixed)
            Q_betas = declare(fixed)
            
            x = declare(int)
            
            with for_(self.tomo.n, 0, self.tomo.n < N_avg, self.tomo.n + 1):
                with for_(x, 0, x<2, x+1):
                    with switch_(x):
                        with case_(0):
                            align(qubit, ro_element)
                            if is_active_reset_calib or is_sb_cool_calib: self.perform_full_measurement(I_betas, Q_betas,is_save = False, ro_element = ro_element, is_active_reset = is_active_reset_calib, is_sb_cool = is_sb_cool_calib, **calib_kwargs )
                        with case_(1):
                            self._pi_pulse(qubit = qubit)
                            align(qubit, ro_element)
                    self.perform_full_measurement(I_betas, Q_betas,'I_betas', 'Q_betas', ro_element = ro_element, is_active_reset = is_active_reset_calib, is_sb_cool = is_sb_cool_calib, **calib_kwargs )
                            
                with for_(x, 0, x<len(self.tomo.pulse_seq_list), x+1):
                    if prepare is not None: 
                        prepare(**prepare_dict)
                    with switch_(x):
                        for i in range(len(self.tomo.pulse_seq_list)):
                            with case_(i):
                                self.tomo.play_tomo_pulse(self.tomo.pulse_seq_list[i], self.pulse_len(qubit, self.pi2_pulse))
                    align(qubit, ro_element)
                    self.perform_full_measurement(self.tomo.I, self.tomo.Q, I_output_name = 'I_tomo', Q_output_name = 'Q_tomo', ro_element = ro_element,  **kwargs)
                    
        self.last_prog = self.tomography_prog
        
    def run_tomography(self):
        if type(self.tomography_results['qubit']) != list:
            self.run_one_qubit_tomography()
            
    def run_one_qubit_tomography(self, prog = None, npts = 1, results_dict = None, **kwargs):
        if results_dict is None: results_dict = self.tomography_results
        N_avg = results_dict['N_avg']
        if npts > 1:
            data_shape = (N_avg, -1, npts)
        else:
            data_shape = (N_avg, -1)
            
        if prog is None: prog = self.tomography_prog
        self.qm_server.clear_all_job_results()
        self.tomography_job = self.qm.execute(prog, duration_limit=0, data_limit=0)
        self.tomography_job.result_handles.wait_for_all_values()
        self.tomography_job.execution_report()
        
        I_tomo = self.tomography_job.result_handles.get('I_tomo').fetch_all()['value'].reshape(data_shape)
        Q_tomo = self.tomography_job.result_handles.get('Q_tomo').fetch_all()['value'].reshape(data_shape)
        I_betas = self.tomography_job.result_handles.get('I_betas').fetch_all()['value'].reshape((N_avg,-1))
        Q_betas = self.tomography_job.result_handles.get('Q_betas').fetch_all()['value'].reshape((N_avg,-1))
        
        results_dict['I_tomo'] = I_tomo
        results_dict['Q_tomo'] = Q_tomo
        results_dict['I_betas'] = I_betas
        results_dict['Q_betas'] = Q_betas
        results_dict['npts'] = npts
        
        self.pickle_save(results_dict, meas_name = 'OneQubitTomography')
        bloch_sphere = qt.Bloch()
        self.plot_one_qubit_tomography(bloch_sphere = bloch_sphere, results_dict = results_dict, **kwargs)
        
        
        
    def plot_tomography(self, guess = 'random', is_plot_no_MLE = True):
        if type(self.tomography_results['qubit']) != list:
            self.plot_one_qubit_tomography(guess = guess, is_plot_no_MLE=is_plot_no_MLE)
        
    def plot_one_qubit_tomography(self, I_tomo = None, Q_tomo = None, I_betas = None, Q_betas = None, npts = None, tomo_object = None,
                                  guess = 'random', is_MLE = True,
                                  is_thersh = True, is_kmeans = True,
                                  bloch_sphere = None,
                                  results_dict = None, 
                                  pts_method = 'l',
                                  **kwargs):
        if results_dict is None: results_dict = self.tomography_results
        if I_tomo is None: I_tomo = results_dict['I_tomo']
        if Q_tomo is None: Q_tomo = results_dict['Q_tomo']
        if I_betas is None: I_betas = results_dict['I_betas']
        if Q_betas is None: Q_betas = results_dict['Q_betas']
        if npts is None: npts = results_dict['npts']
        if tomo_object is None: tomo_object = OneQubitTomo(results_dict['qubit'])
        if is_kmeans: 
            data_e, err_e = self.process_data([results_dict['I_betas'][:,0], results_dict['Q_betas'][:,0]])
            data_g, err_g = self.process_data([results_dict['I_betas'][:,1], results_dict['Q_betas'][:,1]])
            
            _,_,means = self.find_threshold(results_dict['I_betas'].flatten(), results_dict['Q_betas'].flatten(), n_means = 2, is_fit_circle = False, is_ascending = data_g.mean()<data_e.mean())
            
            betas = means
        else: betas, betas_err = self.process_data([I_betas, Q_betas], is_calculate_stat_error = True)
        plt.figure()
        plt.hist(self.determine_data(I_betas, Q_betas), bins = 300, histtype = 'step')
        
        if is_thersh:
            X = []
            Y = []
            Z = []
            betas = np.array([1,-1])
            if npts>1:
                for n in range(npts):
                    data, err = self.process_data([I_tomo[:,:,n], Q_tomo[:,:,n]], is_mean=False)
                    data, err = data_to_sigma_z(data=data, data_g=means[0], data_e=means[1], err=err, data_g_err=err_g, data_e_err=err_e, 
                                                is_thresholding = is_thersh)
                    data = data.mean(0)
                    tomo_object.create_measured_ops()
                    tomo_object.calculate_eigenvalues(betas, data, err)
                    tomo_object.calculate_density_mat()
                    if is_MLE:
                        tomo_object.calculate_MLE_density_matrix(guess = guess)
                        rho = tomo_object.MLE_density_mat
                    else:
                        rho = tomo_object.density_mat
                    X.append(qt.expect(qt.sigmax(), rho))
                    Y.append(qt.expect(qt.sigmay(), rho))
                    Z.append(qt.expect(qt.sigmaz(), rho))
            else:
                data, err = self.process_data([I_tomo, Q_tomo], is_mean=False)
                data, err = data_to_sigma_z(data=data, data_g=means[0], data_e=means[1], err=err, data_g_err=err_g, data_e_err=err_e, 
                                            is_thresholding = is_thersh)
                data = data.mean(0)
                tomo_object.create_measured_ops()
                tomo_object.calculate_eigenvalues(betas, data, err)
                tomo_object.calculate_density_mat()
                if is_MLE:
                    tomo_object.calculate_MLE_density_matrix(guess = guess)
                    rho = tomo_object.MLE_density_mat
                else:
                    rho = tomo_object.density_mat
                X.append(qt.expect(qt.sigmax(), rho))
                Y.append(qt.expect(qt.sigmay(), rho))
                Z.append(qt.expect(qt.sigmaz(), rho))
        else:
            X = []
            Y = []
            Z = []
            if npts>1:
                for n in range(npts):
                    data, err = self.process_data([I_tomo[:,:,n], Q_tomo[:,:,n]], is_calc_stat_error=True)
                    tomo_object.create_measured_ops()
                    tomo_object.calculate_eigenvalues(betas, data, err)
                    tomo_object.calculate_density_mat()
                    if is_MLE:
                        tomo_object.calculate_MLE_density_matrix(guess = guess)
                        rho = tomo_object.MLE_density_mat
                    else:
                        rho = tomo_object.density_mat
                    X.append(qt.expect(qt.sigmax(), rho))
                    Y.append(qt.expect(qt.sigmay(), rho))
                    Z.append(qt.expect(qt.sigmaz(), rho))
            else:
                data, err = self.process_data([I_tomo[:,:], Q_tomo[:,:]], is_calc_stat_error=True)
                tomo_object.create_measured_ops()
                tomo_object.calculate_eigenvalues(betas, data, err)
                tomo_object.calculate_density_mat()
                if is_MLE:
                    tomo_object.calculate_MLE_density_matrix(guess = guess)
                    rho = tomo_object.MLE_density_mat
                else:
                    rho = tomo_object.density_mat
                X.append(qt.expect(qt.sigmax(), rho))
                Y.append(qt.expect(qt.sigmay(), rho))
                Z.append(qt.expect(qt.sigmaz(), rho))
            
        if bloch_sphere is None:
            bloch_sphere = qt.Bloch()
        bloch_sphere.add_points([X[0],Y[0],Z[0]], 's', alpha = 1)
        bloch_sphere.add_points([X[-1],Y[-1],Z[-1]], 's', alpha = 1)
        bloch_sphere.add_points([X,Y,Z], pts_method, alpha = 0.5)
                
        bloch_sphere.make_sphere()
        
        # self.plot_beta_coeffs(I_betas, Q_betas, data, err)
        if npts ==  1:
            theta = np.arccos(qt.expect(qt.sigmaz(), rho)/(2 * (rho.purity() - 0.5)) ** 0.5) * 180/np.pi
            phi  = np.arctan(qt.expect(qt.sigmay(), rho)/qt.expect(qt.sigmax(),rho)) * 180/np.pi
            if qt.expect(qt.sigmax(),rho)<0 and qt.expect(qt.sigmay(),rho)>0:
                phi= phi + 180
            elif qt.expect(qt.sigmax(),rho)<0 and qt.expect(qt.sigmay(),rho)<0:
                phi = phi - 180
    
            print('\n ##############purity and direction of spinor###############')
            print('Purity No MLE = %f' % (tomo_object.density_mat.purity()))
            print('Theta, phi = {theta}, {phi}'.format(theta=theta, phi=phi))
            print('Purity MLE = %f' % (rho.purity()))
            
        if 't' in results_dict.keys():
            t = results_dict['t']
            if len(t)>1:
                fig,axs = plt.subplots(4,1, sharex = True)
                plt.sca(axs[0])
                plt.plot(t,X)
                plt.ylabel('X')
                plt.sca(axs[1])
                plt.plot(t,Y)
                plt.ylabel('Y')
                plt.sca(axs[2])
                plt.plot(t,Z)
                plt.ylabel('Z')
                plt.sca(axs[3])
                plt.plot(t,np.sqrt(np.array(Z)**2+np.array(X)**2+np.array(Y)**2))
                plt.ylabel('Purity')
                plt.xlabel('Times [ns]')
                
                plt.tight_layout()
        
        
    def plot_beta_coeffs(self, I, Q, data = None, err = None):
        if I.shape[1] == 2:
            self.plot_one_qubit_beta_coeffs(I=I, Q=Q, data=data, err=err)
        elif I.shape[1]==4:
            self.plot_two_qubits_beta_coeffs(I=I, Q=Q, data=data, err=err)
            
    def plot_one_qubit_beta_coeffs(self, I, Q, data = None, err = None):
        hist_0, hist_0_err = self.process_data(data = [I[:,0],Q[:,0]], is_mean = False); b0 = np.mean(hist_0)
        hist_1, hist_1_err  = self.process_data(data = [I[:,1],Q[:,1]], is_mean = False); b1 = np.mean(hist_1)
        
        fig, axs = plt.subplots(1,2)
        
        plt.sca(axs[0])
        plt.plot(I[:,0],Q[:,0],'.b',label='0', alpha = 0.15)
        plt.plot(I[:,1],Q[:,1],'.r',label='1', alpha = 0.15)
        plt.plot([0],[0],'xk')
        plt.legend()
        
        plt.sca(axs[1])
        
        num_of_bins = 200
        bins = np.histogram(np.hstack((hist_0, hist_1)), bins=num_of_bins)[1]
        
        hist_0_data = plt.hist(hist_0,
                bins = bins,
                histtype='step',
                color = 'b')
        hist_1_data = plt.hist(hist_1,
                 bins = bins,
                 histtype='step',
                 color = 'r')
        
        plt.ylabel("counts", fontsize=18)
        plt.xlabel(self.which_data, fontsize=18)
        plt.legend(['0','1'])
        
        max_count = max([max(hist_0_data[0]), max(hist_1_data[0])])
        plt.plot([b0,b0],[0, max_count + 5], '--b')
        plt.plot([b1,b1],[0, max_count + 5], '--r')

        plt.suptitle(r'Calibrate $\beta$ Coefficients')
        
        if data is not None:
            plt.sca(axs[1])
            plt.errorbar(x = data, y = [10]*len(data), xerr = err, capsize = 6, ecolor = 'k', fmt = '.')
            
    def plot_two_qubits_beta_coeffs(self, I, Q, data = None, err = None):
        
        hist_00, hist_00_err = self.process_data(data = [I[:,0],Q[:,0]], is_mean = False); b00 = np.mean(hist_00)
        hist_01, hist_01_err  = self.process_data(data = [I[:,1],Q[:,1]], is_mean = False); b01 = np.mean(hist_01)
        hist_10, hist_10_err = self.process_data(data = [I[:,2],Q[:,2]], is_mean = False); b10 = np.mean(hist_10)
        hist_11, hist_11_err = self.process_data(data = [I[:,3],Q[:,3]], is_mean = False); b11 = np.mean(hist_11)
        
        fig, axs = plt.subplots(1,2)
        
        plt.sca(axs[0])
        plt.plot(I[:,0],Q[:,0],'.b',label='00', alpha = 0.15)
        plt.plot(I[:,1],Q[:,1],'.r',label='01', alpha = 0.15)
        plt.plot(I[:,2],Q[:,2],'.c',label='10', alpha = 0.15)
        plt.plot(I[:,3],Q[:,3],'.m',label='11', alpha = 0.15)
        plt.plot([0],[0],'xk')
        plt.legend()
        
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
        plt.xlabel(self.which_data, fontsize=18)
        plt.legend(['00','01','10','11'])
        
        max_count = max([max(hist_00_data[0]),max(hist_01_data[0]), max(hist_10_data[0]), max(hist_11_data[0])])
        plt.plot([b00,b00],[0, max_count + 5], '--b')
        plt.plot([b01,b01],[0, max_count + 5], '--r')
        plt.plot([b10,b10],[0, max_count + 5], '--c')
        plt.plot([b11,b11],[0, max_count + 5], '--m')

        plt.suptitle(r'Calibrate $\beta$ Coefficients')
        
        if data is not None:
            plt.sca(axs[1])
            plt.errorbar(x = data, y = [10]*len(data), xerr = err, fmt = 'ok')
            
            
    def map_e_to_g(self, qubit, ro_element = None): #this is actually for measuring e to f but it fits here
        if ro_element is None: ro_element = self.main_readout
        
        align(qubit, qubit + 'ef', ro_element)
        if self.is_pi_pulse:
            play('pi_pulse', qubit)
        else: 
            play('pi2_pulse', qubit)
            play('pi2_pulse', qubit)
        align(qubit, qubit + 'ef', ro_element)
        
        
#%% mixer functions
    def MixCal(self, element, SSB_id = None, DC_os =None, fig = False, 
               repetition = 5, LO_acc = -70, SB_acc = -50, g_os = 0.2, g_c = None, phi_os = 30, phi_c = None):
    #side band choice is automatic form dictionary and close all other MXG and reopen them after calibration and make the choice automatic according to type of sideband depending on
        # self.remove_leakage(element,DC_os = DC_os, fig = fig, repetition = repetition, LO_acc = LO_acc,is_print =False, is_running = False)
        
        self.remove_leakage(element, DC_os = DC_os, fig = fig, repetition = repetition, LO_acc = LO_acc,is_print =False)
       
        self.remove_side_band(element,SSB_id = SSB_id, fig = fig, repetition = repetition, SB_acc = SB_acc, g_os = g_os, g_c = g_c, phi_os = phi_os, phi_c = phi_c,is_print =False)

        self.remove_leakage(element,DC_os = DC_os, fig = fig, repetition = repetition, LO_acc = LO_acc,is_print = False)
       
        self.print_mixer(element)
        

        #self.set_output_dc_offset_by_element(element, 'I',self.qm.get_output_dc_offset_by_element(element,'I'))
        return
    
    def print_mixer(self,element):
        
        # g_str = [str(np.round(g,3)) for g in self.configObject.SuperMembers[element].element.gCorrection]
        g_str = str(np.round(self.configObject.SuperMembers[element].element.gCorrection,3))
        # phi_str = [str(np.round(phi,7)) for phi in self.configObject.SuperMembers[element].element.phaseCorrection]
        phi_str = str(np.round(self.configObject.SuperMembers[element].element.phaseCorrection,7))
        print('offsets[I_{4}] = {0}\noffsets[Q_{4}] = {1} \nssb_cor_phase_{4} = {2} \ng_corr_{4}= {3}'.format(
            np.round(self.qm.get_output_dc_offset_by_element(element,'I'),7),
                            np.round(self.qm.get_output_dc_offset_by_element(element,'Q'),7),
                            phi_str,
                            g_str, element)
    )
    
    
    def remove_leakage(self, element, I_os = 0.01, I_c = None, Q_os = 0.01, Q_c = None, fig = False, repetition = 5, LO_acc = -70, is_print = True, is_run=False,
                       freq_span_ratio = 4, res_bw = 1000, vid_bw = 1000):
        
        if is_run: job = self.run_continuous(element)
        self.set_cxa_to_element(element, vid_bw = vid_bw, res_bw = res_bw, mul_span = freq_span_ratio)
        
        sleep_t = np.max([0.1, 1/vid_bw])
        # if vid_bw <=1:
        #     sleep_t = 1
        try:
            I_C, I_os = self.DC_OS(element, repetition = 2, V_os = I_os, V_c = I_c, accuracy = LO_acc, fig = fig, is_run = is_run, sleep_t = sleep_t )
            for rep in [repetition-3]: 
                 Q_C, Q_os = self.DC_OS(element,  MixIn = 'Q', V_os = Q_os, V_c = Q_c, repetition = rep+2, accuracy=LO_acc, fig = fig, is_run = is_run, sleep_t = sleep_t)
                 I_C,I_os = self.DC_OS(element, V_os = I_os, V_c = I_c, repetition = rep+2, accuracy=LO_acc, fig = fig , is_run = is_run, sleep_t = sleep_t)
        except raise_accuracy as err:
                print('success \n leakage is {0}'.format(err.args[0]))
        if is_print: self.print_mixer(element)

    def remove_side_band(self,element, SSB_id = None, fig = False, repetition = 5, SB_acc = -50, g_os = 0.2, g_c =None, phi_os = 30, phi_c = None, is_print =True,
                         vid_bw = 1000, res_bw = 1000, freq_span_ratio = 4):
        "g_c, phi_c are center of scan vector. g_os, phi_os are range of scan vector"
        
        if (self.config['elements'][element]['intermediate_frequency']>0) and SSB_id == None: 
            SSB_id = 'USB'
            cxa_center_freq = 'LSB'
        elif SSB_id == None: 
            SSB_id = 'LSB'
        cxa_center_freq = 'USB' if SSB_id == 'LSB' else 'LSB'
        
        self.set_cxa_to_element(element, vid_bw = vid_bw, res_bw = res_bw, mul_span = freq_span_ratio, center = cxa_center_freq)

        sleep_t = np.max([0.1, 1/vid_bw])
        # if phi_c is None: 
        #     phi_c  = self.configObject.SuperMembers[element].element.phaseCorrection
            # if phi_c is None and SSB_id == 'LSB': phi_c = 0
            # elif phi_c is None: phi_c = 180
        
        try: #TODO fix this try clause ?WHY? the print to include value of side band?
            job = self.run_continuous(element)
            phi_c,phi_os = self.CMat_phi(element, job, repetition = 1, Npts =101, accuracy=SB_acc , phi_os=phi_os, phi_c = phi_c, SSB_id=SSB_id, fig = fig, sleep_t = sleep_t)
            for rep in [repetition-3]:#range(max(repetition-2,1)):
                job = self.run_continuous(element)
                _,_ = self.CMat_g(element, job, repetition = rep+2, g_os=g_os, g_c=g_c, accuracy=SB_acc ,  SSB_id=SSB_id, fig = fig, sleep_t = sleep_t)
                job = self.run_continuous(element)
                _,_ = self.CMat_phi(element, job, repetition = 3, Npts =101, accuracy=SB_acc , phi_os=phi_os, phi_c=phi_c, SSB_id=SSB_id, fig = fig, sleep_t = sleep_t)
        except raise_accuracy as err:
                print('success')
        if is_print: self.print_mixer(element)
        
    
    def DC_OS_sweep(self,element, MixIn = 'I', V_os = 0.01, V_c = 0.0, Npts=21, sleep_t = 0.01):
        LSB_P = []
        LO_P = []
        USB_P = []
        SSB_P = []
     
        DC_os_vec = np.linspace((V_c - V_os), (V_c + V_os), Npts).tolist()
        

        if max(list(abs(np.array(DC_os_vec))))>0.2: raise ValueError('Too high DC offset')
        for v in DC_os_vec:
            self.qm.set_output_dc_offset_by_element(element, MixIn, v)
            sleep(sleep_t)
            LSB_P.append(self.CXA.Marker.Get_Power(1))
            LO_P.append(self.CXA.Marker.Get_Power(2))
            USB_P.append(self.CXA.Marker.Get_Power(3))
            
        return LSB_P,LO_P, USB_P, DC_os_vec
     
    def DC_OS(self,element, MixIn = 'I', V_os = 0.01, V_c = None, Npts=21, sleep_t = 0.1, accuracy = -40, repetition = 3, fig = True, is_run = False):

        print('optimizing leakage of {0} port {1}'.format(element,MixIn))
        if is_run: self.run_continuous(element)

        if V_c is None:
            V_c  = self.qm.get_output_dc_offset_by_element(element,MixIn)
            
        i=0 ;j=0
        while ((i<repetition) and (j<2*repetition)):
            LSB_P,LO_P, USB_P, DC_os_vec = self.DC_OS_sweep(element, MixIn = MixIn, V_os = V_os , V_c = V_c , Npts=Npts, sleep_t = sleep_t)
            if fig:
                plt.figure()              
                plt.plot(DC_os_vec, LSB_P, label='LSB')
                plt.plot(DC_os_vec, LO_P, label='Leakage')
                plt.plot(DC_os_vec, USB_P, label='USB')
                plt.xlabel(MixIn + " DC Offset [V]")
                plt.ylabel("Signal Power [dBm]")
                plt.legend()
            
            
            if min(LO_P)< accuracy:
                DC_v = DC_os_vec[LO_P.index(min(LO_P))]
                self.set_output_dc_offset_by_element(element, MixIn, DC_v)
                if is_run: self.run_continuous(element)
                print('DC offeset of ' +element + ' in port ' + MixIn + ' is ' + str(DC_v))
                print('sucess')
                raise raise_accuracy(min(LO_P))
            
            if LO_P.index(min(LO_P))<2 :
                V_c = DC_os_vec[LO_P.index(min(LO_P))]-V_os    
                
            elif LO_P.index(min(LO_P))>len(LO_P)-3:
                V_c = DC_os_vec[LO_P.index(min(LO_P))]+V_os       
                
            else:
                V_c = DC_os_vec[LO_P.index(min(LO_P))]    
                V_os = DC_os_vec[LO_P.index(min(LO_P))]- DC_os_vec[LO_P.index(min(LO_P))+2]
                i = i+1
            j=j+1
        DC_v = DC_os_vec[LO_P.index(min(LO_P))]
        self.set_output_dc_offset_by_element(element, MixIn, DC_v)
        if is_run: self.run_continuous(element)
        print('DC offeset of ' +element + ' in port ' + MixIn + ' is ' + str(DC_v))
        return DC_v,V_os
    
    def side_band_optimazation(self,SSB_id, USB_P, LSB_P,LO_P, is_relative = False):
        """
        What to optimize according to what side band you want to use
        example, if SSB_id == 'LSB' it will reduce the upper side band 
        compared to the lower sideband
        """ 
        if is_relative:
            if SSB_id == 'ESB':
                 return list(abs(np.array(LSB_P)-np.array(USB_P)))
            elif SSB_id == 'LSB':
                return list(np.array(USB_P)-np.array(LSB_P))#+np.array(LO_P))
            elif SSB_id == 'USB':
                 return list(np.array(LSB_P)-np.array(USB_P))#+np.array(LO_P))
        else:
             if SSB_id == 'ESB':
                  return list(abs(np.array(LSB_P)-np.array(USB_P)))
             elif SSB_id == 'LSB':
                 return list(np.array(USB_P))#+np.array(LO_P))
             elif SSB_id == 'USB':
                  return list(np.array(LSB_P))#+np.array(LO_P))    


    
    def CMat_phi_sweep(self,element, job, SSB_id='LSB', phi_os = 30, phi_c = None, Npts = 21, sleep_t = 0.1):

        if phi_c is None: phi_c  = self.configObject.SuperMembers[element].ElementParams['phaseCorrection']
        
        g_v = self.configObject.SuperMembers[element].element.gCorrection
        LO_f = self.configObject.SuperMembers[element].element.loFreq
        IF_f = self.configObject.SuperMembers[element].element.intermediateFreq
        mixer = self.configObject.SuperMembers[element].element.mixerName
        # print(L)
        # LO_f = self.config['elements'][element]['mixInputs']['lo_frequency']
        # IF_f = self.config['elements'][element]['intermediate_frequency']
        # g_v = self.aux_config['elements'][element]['mixer_g']
        # mixer = self.config['elements'][element]['mixInputs']['mixer']

        LSB_P = []
        LO_P = []
        USB_P = []
    
        phi_vec = np.linspace((phi_c - phi_os), (phi_c + phi_os), Npts).tolist()
    
        for phi in phi_vec:
            # self.qm.set_mixer_correction(mixer, int(IF_f), int(LO_f), self.calc_cmat(phi, g_v))
            job.set_element_correction(element, self.calc_cmat(phi, g_v))
            sleep(sleep_t)
            LSB_P.append(self.CXA.Marker.Get_Power(1))
            LO_P.append(self.CXA.Marker.Get_Power(2))
            USB_P.append(self.CXA.Marker.Get_Power(3))
 
        return LSB_P,LO_P, USB_P, phi_vec
 
    def CMat_phi(self,element, job, SSB_id='LSB',  phi_os = 30, phi_c = None, Npts = 21, sleep_t = 0.1, accuracy = -30, repetition = 5, fig = True):#TODO add to all such functions and there sub functions a break if there is no real improvment or the results are worst than the best

        print('optimizing {0} to the single {1}'.format(element,SSB_id))
            
        i=0; j=0
        while((i<repetition) and (j<2*repetition)):
            LSB_P,LO_P, USB_P, phi_vec = self.CMat_phi_sweep(element, job, SSB_id, phi_os = phi_os , phi_c = phi_c , Npts=Npts, sleep_t = sleep_t)
            
            if fig:
                plt.figure()
                plt.plot(phi_vec, LSB_P, label='LSB')
                plt.plot(phi_vec, LO_P, label='Leakage')
                plt.plot(phi_vec, USB_P, label='USB')
                plt.xlabel("C Matrix Rotation [Phas]")
                plt.ylabel("Signal Power [dBm]")
                plt.legend()      
            # print()
            SSB_P = self.side_band_optimazation(SSB_id,USB_P,LSB_P,LO_P)

            if min(SSB_P) < accuracy:
                phi_SSB = phi_vec[SSB_P.index(min(SSB_P))]

                self.update_mixer_by_element(element,phase_correction = phi_SSB)
                print('Power of unwanted sideband is {0}'.format(SSB_P[SSB_P.index(min(SSB_P))]))
                print('success')
                self.run_continuous(element)
                raise raise_accuracy(SSB_P[SSB_P.index(min(SSB_P))],SSB_id)
           
            if SSB_P.index(min(SSB_P))<2 :
                phi_c = phi_vec[SSB_P.index(min(SSB_P))]#-phi_os    
                
            elif SSB_P.index(min(SSB_P))>len(SSB_P)-3:
                phi_c = phi_vec[SSB_P.index(min(SSB_P))]#+phi_os       
                
            else:
                phi_c = phi_vec[SSB_P.index(min(SSB_P))]    
                phi_os = phi_os*3/Npts
                i = i+1 
            # print( 'phi_c'+ str(phi_c))
            # print( 'phi_os'+ str(phi_os))
            j = j+1           
            
        phi_SSB = phi_vec[SSB_P.index(min(SSB_P))]; print(phi_SSB)
        print(f'best phi is {phi_SSB}')
        self.update_mixer_by_element(element,phase_correction=phi_SSB)
        print(f'updated to {self.configObject.SuperMembers[element].element.phaseCorrection}')
        self.run_continuous(element)
        # self.config = self.qm.get_config()
        return phi_SSB, phi_os
    
    def CMat_g_sweep(self, element, job, SSB_id ='LSB',  g_os = 0.2, g_c =None, Npts = 21, sleep_t = 0.1):
        
        if g_c is None: g_c  =  self.configObject.SuperMembers[element].element.gCorrection

        phi_v = self.configObject.SuperMembers[element].element.phaseCorrection
        LO_f = self.configObject.SuperMembers[element].element.loFreq
        IF_f = self.configObject.SuperMembers[element].element.intermediateFreq
        mixer = self.configObject.SuperMembers[element].element.mixerName
        
        # LO_f = self.config['elements'][element]['mixInputs']['lo_frequency']
        # IF_f = self.config['elements'][element]['intermediate_frequency']
        # phi_v = self.aux_config['elements'][element]['mixer_phase']
        # mixer = self.config['elements'][element]['mixInputs']['mixer']
    
        LSB_P = []
        LO_P = []
        USB_P = []

    
        g_vec = np.linspace((g_c - g_os), (g_c + g_os), Npts).tolist()  
        
        for g in g_vec:
    
            # self.qm.set_mixer_correction(mixer, int(IF_f), int(LO_f), self.calc_cmat(phi_v, g))
            job.set_element_correction(element, self.calc_cmat(phi_v, g))
            sleep(sleep_t)
            LSB_P.append(self.CXA.Marker.Get_Power(1))
            LO_P.append(self.CXA.Marker.Get_Power(2))
            USB_P.append(self.CXA.Marker.Get_Power(3))
        
        return LSB_P,LO_P, USB_P, g_vec
    

    def CMat_g(self, element, job, SSB_id ='LSB',  g_os = 0.5, g_c = None, Npts = 21, sleep_t = 0.1, accuracy = -80, repetition =5, fig = True):
      
        if g_c is None: g_c  =  self.configObject.SuperMembers[element].element.gCorrection
      
        print('calibrating g')  
        
        i=0; j=0
        while((i<repetition) and (j<2*repetition)):

            LSB_P,LO_P, USB_P, g_vec = self.CMat_g_sweep(element, job, SSB_id = SSB_id, g_os = g_os , g_c = g_c , Npts=Npts, sleep_t = sleep_t)
                
            if fig:
                plt.figure()
                plt.plot(g_vec, LSB_P, label='LSB')
                plt.plot(g_vec, LO_P, label='Leakage')
                plt.plot(g_vec, USB_P, label='USB')
                plt.xlabel("C Matrix g [U.L]")
                plt.ylabel("Signal Power [dBm]")
                plt.legend()    
            
       
            SSB_P = self.side_band_optimazation(SSB_id,USB_P,LSB_P,LO_P)

            if min(SSB_P) < accuracy:
                print('Power of unwanted sideband is {0}'.format(SSB_P[SSB_P.index(min(SSB_P))]))
                print('success')
                g_SSB = g_vec[SSB_P.index(min(SSB_P))]
                self.configObject.SuperMembers[element].element.gCorrection= g_SSB
                self.update_mixer_by_element(element,g_correction = g_SSB)
                # self.config = self.qm.get_config()
                self.run_continuous(element)

                raise raise_accuracy(SSB_P[SSB_P.index(min(SSB_P))],SSB_id)
            
           
            if SSB_P.index(min(SSB_P))<2 :
                g_c = g_vec[SSB_P.index(min(SSB_P))]-g_os    
                
            elif SSB_P.index(min(SSB_P))>len(SSB_P)-3:
                g_c = g_vec[SSB_P.index(min(SSB_P))]+g_os       
                
            else:
                g_c = g_vec[SSB_P.index(min(SSB_P))]    
                g_os = g_vec[SSB_P.index(min(SSB_P))]-g_vec[SSB_P.index(min(SSB_P))+2]
                i = i+1
            j = j+1
            
        g_SSB = g_vec[SSB_P.index(min(SSB_P))]
        
        self.configObject.SuperMembers[element].element.gCorrection = g_SSB
        print(g_SSB)
        self.update_mixer_by_element(element,g_correction = g_SSB)
        self.run_continuous(element)

        # self.config = self.qm.get_config()
        return g_SSB, g_os
    
    def calc_cmat(self,phi, g):
       return Calc_corr_mat(phi,g)
    
    def cart2pol(self,x, y):
        return cart2pol(x, y)
    
    def pol2cart(self, rho, phi):    
        return pol2cart(self, rho, phi)
    def remove_leakage_gradient(self, element, is_start_with_current = True):
        
        job = self.run_continuous(element)
        self.set_cxa_to_element(element)
        if is_start_with_current:
            I0 = self.qm.get_output_dc_offset_by_element(element,'I')
            Q0 = self.qm.get_output_dc_offset_by_element(element,'Q')
        else:
            I0=0
            Q0=0
        results = minimize(self.change_DC_and_read_power, [I0,Q0], args = (element), bounds = Bounds(-0.1,0.1), method = 'SLSQP', options = {'maxiter': 100, 'eps': 0.04, 'ftol': 1e-8, 'disp': True})
        print(f'I DC offset = {results.x[0]}\nQ DC offset = {results.x[1]}')
        
    # def change_DC_and_read_power(self, output_voltages, element):
    #     if (np.abs(output_voltages[0]) > 0.1) or (np.abs(output_voltages[1]) > 0.1): raise ValueError('DC offset too high')
    #     self.qm.set_output_dc_offset_by_element(element, 'I', output_voltages[0] )
    #     self.qm.set_output_dc_offset_by_element(element, 'Q', output_voltages[1] )
    #     sleep(0.2)
    #     return self.CXA.Marker.Get_Power(2)
    
    def remove_sideband_gradient(self, element, SSB = None, init_guess = None):
        
        job = self.run_continuous(element)
        self.set_cxa_to_element(element)

        LO_frequency = int(self.config['elements'][element]['mixInputs']['lo_frequency'])
        intermediate_frequency = int(self.config['elements'][element]['intermediate_frequency'])
        mixer = self.config['elements'][element]['mixInputs']['mixer']
        
        if (self.config['elements'][element]['intermediate_frequency']>0) and SSB is None: SSB = 'USB'
        elif SSB is None: SSB = 'LSB'
        
        if init_guess is None:
            init_guess = [0,0]
        elif init_guess == 'current':
            # init_guess = [self.aux_config['elements'][element]['mixer_phase'],self.aux_config['elements'][element]['mixer_g']]
            init_guess = [0,0]
        
        constraints = (({'type': 'ineq', 'fun': lambda x: -np.tan(x[0]* np.pi / 180)},
                        {'type': 'ineq', 'fun': lambda x: 1 / ((1 + x[1]) * np.cos(x[0]* np.pi / 180))}))
        
        
        results = minimize(self.change_correction_mat_and_read_power, init_guess, args = (element, SSB, mixer, intermediate_frequency, LO_frequency), bounds = Bounds(-1.95, 1.95), constraints = constraints, method = 'SLSQP', options = {'maxiter': 2000, 'eps': 0.1 ,'ftol': 0.1, 'disp': True})
        print(f'phase = {results.x[0]}\n g = {results.x[1]}')

        
    # def change_correction_mat_and_read_power(self, corr_params, element, SSB, mixer, intermediate_frequency, LO_frequency):
    #     corr_mat = self.calc_cmat(corr_params[0], corr_params[1])
    #     self.qm.set_mixer_correction(mixer, intermediate_frequency, LO_frequency, corr_mat)
    #     sleep(0.2)
    #     LSB_power = self.CXA.Marker.Get_Power(1)
    #     USB_power = self.CXA.Marker.Get_Power(3)
    #     if SSB == 'USB': return LSB_power-USB_power
    #     elif SSB == 'LSB': return USB_power-LSB_power
        

    def change_qubit(self, main = None):
        if main is None:
            if self.scnd_qubit is not None:
                       main_temp = self.main_qubit
                       self.main_qubit = self.scnd_qubit
                       self.scnd_qubit = main_temp
                       print(f"Changed the main qubit to {self.main_qubit}")
            else: raise ValueError("No second qubit found")
        elif self.main_qubit == main:
            print(f"The main qubit is already {self.main_qubit}")  
        else:
            self.scnd_qubit = self.main_qubit
            self.main_qubit = main
            print(f"Changed the main qubit to {self.main_qubit}")
    
    def generate_folder(self,name):
        """checks if a folder called name exists otherwise it generates a folder"""
        if not os.path.exists(name):                
            os.makedirs(name)
        return
    
    @staticmethod
    def check_fixed(num):
        if num >= 8 or num <=-8:
            raise ValueError(f"{num} is not in (-8,8) and therefore it overflows")
        else: return num
#%%
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(x,y):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)   

def Calc_corr_mat(phi, g):
    "phi in degrees"
    c00 = 1; 
    c01 = -np.tan(phi* np.pi / 180)
    c10 = 0
    c11 = 1 / ((1 + g) * np.cos(phi* np.pi / 180))
    # for value in [c00, c01, c10, c11]: 
        # if (value >2) or (value <-2): raise ValueError('overflow in correction matrix with phi, g = {},{}. {} is out of range (-2,2)'.format(phi, g, value))
    return [c00, c01, c10, c11]

class raise_accuracy(Exception):
        pass

def multiple_gaussians(x,a,b,c, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y + a*x**2+b*x+c              
def Lorentzian(x,a,b,c,d):
    return (a/((x-d)**2+b**2) + c)
def Gaussian(x,amp,center,std,C):
    return amp*np.exp(-((x-center)**2)/(2*std**2))+C


