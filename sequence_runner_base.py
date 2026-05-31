# -*- coding: utf-8 -*-
"""
Created on Mon Sep 02 10:59:45 2019

@author: Nir and Boaz
"""


"""
problem to solve later

the first run of a frequency sweep in power is often very misaligned, getting giant phase numbers, possibly due to wrong electrical delay
"""
import sys

sys.path.append(r'C:\Users\Shay\Anaconda2\envs\exp-env\Lib\site-packages\qnl_ctrl\InstrumentControl')
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\freqDomain\low level old stuff')
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\freqDomain\low level old stuff')

import datetime
import Usefull_fun as ufun

from InstrumentControl.general_functions import *
from MXG_SWEEP import *
# Do not change order, this needs to be called before calling import B2962A
from percure_data import *
from InstrumentControl import B2962A

import percure_data

import matplotlib.pyplot as plt
from time import sleep
import numpy as np
from labrad.units import V,mV, s,ms,us,ns, Hz,kHz,MHz,GHz, Ohm, dBm, Value, mA
import yaml
# import h5py

class Sequence_runner(object):
    
    def __init__(self, 
                 saveFolderName = None,
                 **kwargs):
        
        self.saveFolderName = saveFolderName
    
    """ Copy from percure data """
    def Quick_save(self, fle = 'Pna_Config_Example.yaml', NetAnal = None, loc ='TCPIP0::K-N5232B-81186::inst0::INSTR',
                   averging_folder_name = None,averging_file_name = None, Big_Dic = None, ToPlot = True  ): 
      
        """Gets: fle- the full yaml filename with all the data for the mesurment,  
        loc - the IP/location of the vna network analyzer
        NetAnal- the pna instance
        
        DATA dic with c
        Returns: the instance of the network analyzer as well as the directory  
        of the DATA.npz file which  containing the key 'DATA':
                Data_dic['X_axis']= the pna x axis as a labrad
                Data_dic['Raw']= the pna raw data
                Data_dic['Y_axis']= the  pna formated data
                Data_dic['format'] = the pna format
            
        also returns the network analyzer configuration as a conifg.yaml at the same folder"""
        #print(fle)
        now = datetime.datetime.now()
        Data_dic ={}
        if NetAnal is None:
            NetAnal = percure_data.get_network_analzer(loc=loc, filename = fle)
        """ # The problem here is that PNA, and both MXGs are not imported here. The solution is probably to change all the parameters to be taken from BigWorkingDic
        if Big_Dic is not None:
            update_Big_Dic(PNA = PNA, MXG1 = MXGTop, MXG2 = MXGBottom , Dictionary = Big_Dic)
            Data_dic['BigWorkingDic'] = Big_Dic
        """
        Data_dic['X_axis']=NetAnal.get_X_axis()
        Data_dic['Raw']=NetAnal.getData(typ='Raw')
        Data_dic['Y_axis']=NetAnal.getData()
        Data_dic['format'] =NetAnal.format_typ()
        # Insert: PNA power, consider also putting PNA power and frequency in filename
        
        
        
        
        if averging_folder_name is None:
            averging_folder_name = 'ForsakenQuickSaves\\'+ now.strftime("%Y_%m_%d")
        if averging_file_name is None:
            averging_file_name = '\\'
        
        SavePath = averging_folder_name+'\\'+averging_file_name+now.strftime("_%Y_%m_%d_%H-%M-%S")     
        ufun.create_folder(averging_folder_name)
        np.savez(SavePath+'.npz',DATA= Data_dic)
        if ToPlot:
            percure_data.raw_data_plot(SavePath+'.npz')
    
        print ('File path is: ', SavePath+'.npz')
        #NEED TO ENSURE THAT THE FILE IS NOT RRUN OVER 
        
        NetAnal.save_Yaml(OutYaml = SavePath+'_Config.yaml')
    
        return NetAnal, SavePath+'.npz'
    
    def initilaitze_PNA(self, pna, PNA_freq, pna_power, IF_BW, scan_pts, pna_numAv, s_parameters = None, elDelay = None):
        """intializes the PNA according to the following type of sweep
        PNA_freq can be a lab_rad list of [frequncey center, span] or a labrad value of CW_frequencey
         used by two_tone_spectroscopy 
         for power sweep pna power is a list of labrad dBm units 
         give measure in the form 'S11', 'S34', etc.
        Returns:
                delay time for the SG in labrad seconds
                """
        pna.powerOnOff(state='ON');# pna.s_parameters(in_port=2,out_port =3 )
        pna.bandwidth(bw = IF_BW)#in seconds
        pna.num_points(n = scan_pts)# the number of points of the PNA sweep 
        
        #the power of the PNA in dbm
        pna.averages(av =pna_numAv);pna.set_triger_source();pna.set_triger_type();pna.set_triger_slope()
        #pna.visa.write('SENS:SWE:MODE CONT') 
        pna.averages(av=pna_numAv) #the number of times that each point is averged
        if type(pna_power)==list:
            pna.set_sweep_type('POW'); pna.set_following_triger('SWE');pna.set_triger_source(source='IMM')
            pna.CwFreq(f = PNA_freq);pna.power_sweep(rang=pna_power);sleep(1)
        elif type(PNA_freq)== list:
            pna.power(p = pna_power)
            pna.set_sweep_type('LIN'); pna.set_following_triger('SWE');pna.set_triger_source(source='IMM')
            pna.frequencey_span(CenFreq = PNA_freq[0], span = PNA_freq[1]); sleep(1)
        else:
            pna.power(p = pna_power)
            pna.set_sweep_type('CW');pna.CwFreq(f = PNA_freq);pna.set_following_triger('POIN');
        
        if s_parameters is not None: pna.s_parameters(in_port = s_parameters[1], out_port = s_parameters[2])
        
        if elDelay is not None: pna.electrical_delay(elDelay)
        
        dwell_time = 1.1*(1./IF_BW)*scan_pts+2*ms
        return dwell_time
    

    def plot_2D_sweep(self,
                      X_axis, # X axis values, in labrad units
                      Y_axis, # Y axis values, in labrad units
                      DATA,
                      fig_num = None,
                      title= None, 
                      Xlabel = None,
                      Ylabel = None,
                      txt=None, 
                      Unwrap= False,
                      cmap='seismic',
                      textsize = 15):
        plt.figure(num = fig_num, figsize = [10,6])
        plt.clf()
        # X,Y=np.meshgrid(X_axis[X_axis.units],Y_axis[Y_axis.units])
        #if DATA.dtype==np.complex:
        ax1 = plt.subplot(121)
        if Unwrap:
            color_data = (np.unwrap(np.angle(DATA)))
        else:
             color_data = (np.angle(DATA))
        
        # plt.pcolormesh(Y,X,color_data,cmap=cmap,shading ='gouraud' )#,edgecolors='k', linewidths=0.02)
        plt.imshow(color_data, extent = (X_axis[X_axis.units].min(), X_axis[X_axis.units].max(), Y_axis[Y_axis.units].min(), Y_axis[Y_axis.units].max()), cmap=cmap, aspect = 'auto')#,edgecolors='k', linewidths=0.02)
        plt.title('Phase',fontweight="bold")
        if Ylabel is not None:
            Ylabel += '[{0}]'.format(Y_axis.units)
            plt.ylabel(Ylabel, fontsize = textsize)
        if Xlabel is not None: 
            Xlabel += ' [{0}]'.format(X_axis.units)
            plt.xlabel(Xlabel, fontsize = textsize)
        plt.xticks(fontsize = textsize)
        plt.yticks(fontsize = textsize)
        cb = plt.colorbar()
        cb.set_label(label='Phase', fontsize = textsize)
        
        ax2 =plt.subplot(122, sharey=ax1)
        # plt.pcolormesh(Y,X, (np.log10(np.abs(DATA))).transpose(),cmap=cmap,shading ='gouraud' )#,edgecolors='k', linewidths=0.02)
        plt.imshow(np.log10(np.abs(DATA)), extent = (X_axis[X_axis.units].min(), X_axis[X_axis.units].max(), Y_axis[Y_axis.units].min(), Y_axis[Y_axis.units].max()), cmap=cmap, aspect = 'auto')#,edgecolors='k', linewidths=0.02)
        plt.title('LogMag',fontweight="bold")
        if Xlabel is not None: plt.xlabel(Xlabel.format(X_axis.units))
        plt.xticks(fontsize = textsize)
        plt.yticks(fontsize = textsize)
        cb = plt.colorbar()
        cb.set_label(label='LogMag [dB]', fontsize = textsize)
    
        if not title is None:
            plt.suptitle(title,fontweight="bold")
        if not txt is None:
            plt.text(0,0.5,txt,fontweight="bold")
        plt.show() 

