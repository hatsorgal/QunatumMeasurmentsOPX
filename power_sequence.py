# -*- coding: utf-8 -*-
"""
Created on Mon Sep 02 10:53:42 2019

@author: Boaz
"""
import percure_data
import Usefull_fun as ufun
import sys
import os
import datetime
import numpy as np
from labrad.units import V,mV, s,ms,us,ns, Hz,kHz,MHz,GHz, Ohm, dBm, Value, mA
from time import sleep
from sequence_runner_base import Sequence_runner

class Power_sequence(Sequence_runner):
    
    def __init__(self, **kwargs):
        pass
        
         
    def Sexy_plot(self, pna, 
                 pna_power = [-5*dBm,-40*dBm],# a lab_rad list of [Power_sweep_start, power_sweep_stop]     
                 pna_freq=[6.5*GHz,6.68*GHz],# a lab_rad list of [start, finish]
                 pna_power_pts=1001, #number of MXG sweep_points if sweep is of the 2D sweep is of the same insturment This is power
                 pna_freq_pts= 2, #number of points in a sweep of the PNA if sweep is of the 2D sweep is of the same insturment This is frequencey
                 ElDe_flag =True,#determines if to compensate for electreical delay
                 Data_format = 'Raw', #sets the Format of Data extracted from PNA
                 elDelay = None,
                 IF_BW = 50*Hz,
                 pna_numAv=1,
                 s_parameters = None,
                 fig_num = None, 
                 save_two_tone = True,
                 plot_at_int = 3,
                 isSave = None,
                 saveFolderName= None,
                 saveFileName = None,
                 Unwrap = False):
        """BOAZ NEEDS to update documantation 
        
        prefoms a power sweep of the pna for different values of frequencey
            returns:
                X_axis- the sweep points of the frequencey
                Y_axis  - the sweep points of the power 
                DATA - an array of complex numbers of size pna_freq_pts*pna_power_pts which
                is the unwraped raw data"""
        
        
        Data_dic ={}
        DATA = np.zeros([pna_power_pts,pna_freq_pts],dtype = 'complex128')
    
        title= saveFolderName; Xlabel='Freq  [{0}]';Ylabel='Power [{0}]';txt=None
        txt=''.join([key+'='+str(itm)+'\n' for key,itm in Data_dic.items()])
        
        power_axis = np.linspace(pna_power[0][pna_power[0].units],pna_power[1][pna_power[0].units],pna_power_pts)*pna_power[0].unit
        freq_axis =    np.linspace(pna_freq[0][pna_freq[0].units],pna_freq[1][pna_freq[0].units],pna_freq_pts)*pna_freq[0].unit
        
        #the following line should replace above line if I ever change the power input into a span
      #  Y_axis = np.linspace((sg_power[0]-sg_power[1]/2.0)[sg_power[0].units],(sg_power[0]+sg_power[1]/2.0)[sg_power[0].units],sg_pts)*sg_power[0].unit
      
        print (pna_power)        
        for i, fre in enumerate(freq_axis):
            DATA[i,:] = np.flip(self.SingleFrequencyPowerSweep(pna,pna_power=pna_power,
                                                               pna_freq=fre,
                                                               power_pts=pna_power_pts,
                                                               pna_numAv= pna_numAv,
                                                               s_parameters = s_parameters, 
                                                               IF_BW = IF_BW,
                                                               fig_num = None, 
                                                               Data_format= Data_format,
                                                               save_data= False,
                                                               folder_name = None,
                                                               file_name = None))
    
        Ylabel = 'Power'
        Xlabel = 'Frequency'
        if saveFolderName is not None:
            if saveFileName is None: saveFileName = ''
            now = datetime.datetime.now()
            
            ufun.create_folder(saveFolderName)
            Data_dic['IF_BW']= IF_BW; 
            Data_dic['ElDe'] = pna.electrical_delay()
            Data_dic['phaseOffset'] = pna.get_phase_offset()
            Data_dic['pnaAvgNum'] = pna_numAv
            Data_dic['X_axis']= freq_axis
            Data_dic['Y_axis']= power_axis
            Data_dic['Raw']= DATA
            completeFilePath = saveFolderName+r'\\'+saveFileName+now.strftime("_%Y_%m_%d_%H-%M-%S")+'.npz'
            np.savez(completeFilePath, DATA= Data_dic)
            print(f'Save data to [{completeFilePath}]')
            
                                 
        self.plot_2D_sweep(X_axis = freq_axis, Y_axis = power_axis, DATA = DATA.transpose(), title = title, Xlabel = Xlabel, Ylabel = Ylabel, txt = None, Unwrap=Unwrap, fig_num = fig_num)#works
        resultsDict = {'power':power_axis, 'freq':freq_axis, 'data':DATA.transpose()}
        return resultsDict
    
    """ based on POW_spectroscopy from MXG_Sweep """
    def SingleFrequencyPowerSweep(self, pna,#PNA_object 
                              pna_power = [-40*dBm,-10*dBm],
                              pna_freq=7.5364*GHz,#7.30898*GHz,# labrad value
                              power_pts=11, #number of MXG sweep_points
                              Data_format = 'Raw', #sets the Format of Data extracted from PNA
                              swp_typ ='SING',#determines if to initate a single ('SING') or continues ('CONT') sweep of the MXG
                              pna_numAv = 1, #number of times the each point of the data is averged over
                              s_parameters = None,
                              elDelay = None,
                              IF_BW = 100*Hz,fig_num = 400, save_data= True, folder_name = None,file_name = None):
                
        """ Perform a spectrosopy with a power sweep of  the pna frequencey """
        dwell_time = self.initilaitze_PNA(pna,pna_freq,pna_power,IF_BW,scan_pts = power_pts, s_parameters = s_parameters, pna_numAv=pna_numAv, elDelay = elDelay)
        #i=0;
        #while(i<pna_numAv):
        pna.visa.write('SENS:SWE:MODE {0}'.format(swp_typ))
        sleep(dwell_time['s'])
        pna.auto_scale()
        #    if i%10==0 and i>0: print(i) 
        #    pna.auto_scale();i=i+1
        
        DATA = pna.getData(typ =Data_format) *np.exp(2j*np.pi*(pna_freq*pna.electrical_delay()+(pna.get_phase_offset())/360.0))
        return DATA
