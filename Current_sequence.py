# -*- coding: utf-8 -*-
"""
Created on Mon Sep 02 10:22:22 2019

@author: Nir and Boaz
"""
"""
-----------------------
current tests
-----------------------
"""
import datetime
import Usefull_fun as ufun
import sys
# Do not change order, this needs to be called before calling import B2962A
sys.path.append(r'C:\Users\Shay\Anaconda2\envs\exp-env\Lib\site-packages\qnl_ctrl\InstrumentControl')
from InstrumentControl import B2962A

import percure_data
import MXG_SWEEP

import matplotlib.pyplot as plt
from time import sleep
import numpy as np
from labrad.units import V,mV, s,ms,us,ns, Hz,kHz,MHz,GHz, Ohm, dBm, Value, mA
import yaml
# import h5py

from sequence_runner_base import Sequence_runner

class Current_sequence(Sequence_runner):
    def __init__(self, **kwargs):
        pass
    
    def SingleFrequencySweepForCurrent(self, pna,#PNA_object 
                              pna_power = -40*dBm, # pna power
                              pna_freq=[6*GHz,1*GHz],# a lab_rad list of [frequncey center, span]
                              frq_pts=101, #number of MXG sweep_points
                              Data_format = 'Raw', #sets the Format of Data extracted from PNA
                              pna_numAv = 1, #number of times the each point of the data is averged over
                              IF_BW = 100*Hz,ElDel_flag=True, 
                              stabilization_time=0.1*s#???
                              ):
        
        """ Perform a spectrosopy with a frequency sweep of the current current """
        dwell_time = self.initilaitze_PNA(pna,pna_freq,pna_power,IF_BW,frq_pts,pna_numAv)
        dwell_time = dwell_time+ stabilization_time
        #sweep_list = np.linspace(in_hz(sg_freq[0]-sg_freq[1]/2),in_hz(sg_freq[0]+sg_freq[1]/2),frq_pts)*Hz        
        pna_X_axis = np.linspace(pna_freq[0]['Hz']-pna_freq[1]['Hz']/2,pna_freq[0]['Hz']+pna_freq[1]['Hz']/2,frq_pts)*Hz
    #    index=0;
    #    while(index<pna_numAv):
    #        sleep(dwell_time['s'])
    #        if index%10==0 and index>0: print(index)
        pna.auto_scale();#.index=index+1
        pna.visa.write('SENS:SWE:MODE SING')
        sleep(dwell_time['s'])
#        if Data_format == 'Raw':
            #frequencies_for_elDel = pna_X_axis*GHz
        DATA = pna.compensate_ElDel(freq_list=pna_X_axis)
#        else: 
#            DATA = pna.getData(typ =Data_format)   
    
        return  DATA
    
    """The running script """
    def RunCurrentFreqSweep(self, fig_num = 203, 
                            current_range = [0.17*mA,0.19*mA],# a lab_rad list of [start current, stop current]
                            current_pts = 21,# number of steps in the current sweep
                            magnetization_delay=2*s, # The delay time between setting the current and performing the frequency sweep
                            pna_power = -40*dBm, # pna power
                            pna_freq=[6.2*GHz,0.5*GHz],# a lab_rad list of [frequncey center, span]
                            frq_pts=501, #number of MXG sweep_points
                            IF_BW = 100*Hz,
                            averging_folder_name= None,
                            averging_file_name = None
                            ):
        #initialization
        complex_data = np.zeros([current_pts,frq_pts],dtype = 'complex128')
        freq_Y_axis = np.linspace((pna_freq[0]-pna_freq[1]/2.0)[pna_freq[0].units],(pna_freq[0]+pna_freq[1]/2.0)[pna_freq[0].units],frq_pts)*pna_freq[0].unit
        
        PNA_object = percure_data.get_network_analzer()
        PowerSource = B2962A.B2962A()
        try:    
        
            # unitless list of currents
            current_list = np.linspace(current_range[0]['A'],current_range[1]['A'],num=current_pts)
            
            # running of the script
            for index,current in enumerate(current_list):
                PowerSource.set_curr(current=current)
                sleep(magnetization_delay['s'])
                complex_data[index,:] = self.SingleFrequencySweepForCurrent(PNA_object, 
                                             frq_pts=frq_pts, 
                                             pna_freq=pna_freq,
                                             pna_power=pna_power,
                                             IF_BW = IF_BW)
     #           pna_freq_Y_axis, complex_data[index,:] = LIN_spectroscopy(pna=PNA_object,
     #                                                                     frq_pts=frq_pts,pna_freq=pna_freq,
     #                                                                     IF_BW=IF_BW)
           
                                                                            
            current_list = current_list*mA
            # averging_folder_name = 'BoazGraphs\\'
            # averging_file_name = ''
            now = datetime.datetime.now()
            Data_dic = {}
            Data_dic['IF_BW']= IF_BW
            Data_dic['ElDe'] =PNA_object.electrical_delay()
            Data_dic['phase']= PNA_object.get_phase_offset()
            PNA_object.save_Yaml(OutYaml = averging_folder_name+averging_file_name+'current_scan'+now.strftime("_%Y_%m_%d_%H-%M-%S")+'_Config.yaml', Dic = Data_dic)
            Data_dic['X_axis']= current_list
            Data_dic['Y_axis']= freq_Y_axis
            Data_dic['Raw']=complex_data
            np.savez(averging_folder_name+'\\'+averging_file_name+'current_scan'+now.strftime("_%Y_%m_%d_%H-%M-%S")+'.npz',DATA= Data_dic)
      #      PowerSource.set_curr(current=0)
        finally:
           if PNA_object is not None:
               PNA_object.visa.close()
           if PowerSource is not None:
               PowerSource.visa.close()
        ############## Close visa connection of PNA_object ###############
        return self.plot_2D_sweep(current_list, freq_Y_axis, complex_data.transpose(),title= 'Flux Tuning Curve')

