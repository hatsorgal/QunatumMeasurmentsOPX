# -*- coding: utf-8 -*-
"""
Created on Sun May 12 14:55:34 2019

@author: Shay


Chen:
    Please note the pronunciation of averging_file_name and averging_folder_name.
    Should be pronounced A-Ver-ging, like Avenging. Just saying...
"""
import sys
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\freqDomain\low level old stuff')
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\freqDomain')
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\freqDomain\low level old stuff')
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\freqDomain')
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\InstrumentControl')
import Usefull_fun as ufun
# import percure_data
from scipy import optimize
import re
import glob
import os
from matplotlib import cm
import matplotlib.pyplot as plt
import yaml
# import h5py
from time import sleep
import numpy as np
try: 
    from  InstrumentControl import MXG5183A, AgilentPNA, AgilentVNA
    from  InstrumentControl.general_functions import *
except: 
    from MXG5183A import MXG5183A
    from AgilentPNA import AgilentPNA
    from B2962A import B2962A
    from CXA_SA import CXA_SA
    from general_functions import *
from labrad.units import V,mV, s,ms,us,ns, Hz,kHz,MHz,GHz, Ohm, dBm, Value
from warnings import warn

import Usefull_fun as Ufun
# from pyvisa.errors import ret_VisaIOError
import time
# from percure_data import *
from time import time
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter

# from percure_data import Quick_save  #mark out for first import!


#Chen's
import datetime



def save_config(func):
    def inner(*args, **kwargs):
        for key, value in kwargs.items():
            config ={}
            if (value is True) or (a is False) or (a is None):
                continue
            config[key]= value            
        x = func(*args, **kwargs)
        return x
    return inner
def time_spent(func):
    def inner(*args, **kwargs):
        time_start = time()
        x = func(*args, **kwargs)
        print('time of runing was {0}'.format(time()- time_start))
        return x
    return inner

class MXG(MXG5183A):
    def load_config(self,filename, dic_name ='ConfigInit'):
        if type(filename)==str:
            with open(filename) as f:
                self.loaded_config = yaml.load(f)[dic_name]
        elif type(filename)== dict:
            self.loaded_config = filename  
        else:
            raise ValueError('filename is neither dict nor yaml file')
        self.loaded_config = check_config_4None(self.loaded_config)
        return self.loaded_config.copy()

#### It should moved to percure_data along with MXG ###

def get_signal_generator(loc ='TCPIP0::192.168.0.11::inst0::INSTR',filename ='MXG_Config_Example.yaml'):        
    """ I am assuming that the error occures only in the initilization of comunication"""
    succsesful_comunication = 0
    while succsesful_comunication<10:
        try:
            SG = MXG(loc) 
            break
        except ret_VisaIOError as e:
            succsesful_comunication = succsesful_comunication +1
            print('failed comunication time {0}'.format(succsesful_comunication))
    else:
        print('last try')
        SG =  MXG(loc) 
    return SG


def initilaitze_PNA(pna,PNA_freq,pna_power,IF_BW,frq_pts,pna_numAv):
    """intializes the PNA according to the following type of sweep
    PNA_freq can be a lab_rad list of [frequncey center, span] or a labrad value of CW_frequencey
     used by two_tone_spectroscopy 
     for power sweep pna power is a list of labrad dBm units 
    Returns:
            delay time for the SG in labrad seconds
            """
    pna.powerOnOff(state='ON');# pna.s_parameters(in_port=2,out_port =3 )
    pna.bandwidth(bw = IF_BW)#in seconds
    pna.num_points(n = frq_pts)# the number of points of the PNA sweep 
    
    #the power of the PNA in dbm
    pna.averages(av =pna_numAv);pna.set_triger_source();pna.set_triger_type();pna.set_triger_slope()
    #pna.visa.write('SENS:SWE:MODE CONT') 
    pna.averages(av=pna_numAv) #the number of times that each point is averged
    if type(pna_power)==list:
        pna.set_sweep_type('POW'); pna.set_following_triger('SWE');pna.set_triger_source(source='IMM')
        pna.CwFreq(f = PNA_freq);pna.power_sweep(rang=pna_power);sleep(1)
        return 1.1*frq_pts*(1./IF_BW)+2*ms, pna.get_X_axis()
    elif type(PNA_freq)== list:
        pna.power(p = pna_power)
        pna.set_sweep_type('LIN'); pna.set_following_triger('SWE');pna.set_triger_source(source='IMM')
        pna.frequencey_span(CenFreq =PNA_freq[0],span =PNA_freq[1]);sleep(1)
        return 1.1*frq_pts*(1./IF_BW)+2*ms, pna.get_X_axis()
    else:
        pna.power(p = pna_power)
        pna.set_sweep_type('CW');pna.CwFreq(f = PNA_freq);pna.set_following_triger('POIN');
        sleep(1)   
    return 1.2*(1./IF_BW)+2*ms, None

def initilaitze_MXG(sg,sg_freq,sg_pwr,frq_pts,dwell_time):
    """intializes the MXG  type of sweep 
       sg_freq (sg_power) is a lab rad list than we get a frequencey sweep of [frequncey(power) center, span]
        or a fixed power and frequencey if both are a non list labrad variable 
         used by two_tone_spectroscopy
         
         returns:
             the total time required to complete the sweep, or the dwell time which should be required for a single point and PNA sweep
             the sweep points in labrad array or NON if this is a constant point
       """ 
    
    sg.on(); sg.attenuator_protection()
    if type(sg_freq)== list and type(sg_pwr)== list:
        raise ValueError('Unprepered for a 2 D sweep of single MXG')
    elif type(sg_freq)== list:
        sg.sweep_mode('FREQ')
        sweep_list = np.linspace(in_hz(sg_freq[0]-sg_freq[1]/2),in_hz(sg_freq[0]+sg_freq[1]/2),frq_pts)*Hz
    elif type(sg_pwr)== list:
        sg.sweep_mode('PWR')
        sweep_list = np.linspace(in_dbm(sg_pwr[0]-sg_pwr[1]/2),in_dbm(sg_pwr[0]+sg_pwr[1]/2),frq_pts)*dBm
    else:
        sg.power(sg_pwr);sg.freq(sg_freq)
        sg.sweep_mode('CONST')
        return 1.1*dwell_time + 4*ms, None
    #sg.visa.write(':SWE:GEN STEP')## selects a step sweep
    sg.Num_SWE_PT(Num_of_points=frq_pts);sg.set_dwell(dwell_time)
    sg.triger_source() #set SG trigering to IMM
    sg.power(sg_pwr);sg.freq(sg_freq)
    sg.EXT_trig_source() #set automaticly to TRIG2
    sg.visa.write(':SWE:GEN STEP')## selects a step sweep
    return (1.0*dwell_time + 4*ms)*frq_pts, sweep_list #takes into account the 1ms overhead time for each point trigered

def save_config_file():
    warn('need to write config update for mxg the config file')
    return
#@Ufun.time_spent
def plot_1D_sweep(fig_num, X_axis, DATA, label = None):
    plt.figure(fig_num)#;plt.clf()
    
    ax2 =plt.subplot(212)
    plt.plot(X_axis['GHz'], 20*np.log10(np.abs(DATA)),'.-',label = label)
    plt.ylabel('LogMag [dB]', fontsize = 15)
    plt.xlabel('Frequency [GHz]', fontsize = 15)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True)
    
    ax1 = plt.subplot(211)
    plt.plot(X_axis['GHz'], np.unwrap(np.angle(DATA)), '.-', label = label)
    plt.ylabel('Phase [Rad]', fontsize = 15)
    plt.xticks(plt.xticks()[0],[None]*len(plt.xticks()[0]), fontsize = 11)
    plt.yticks(fontsize=11)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    return plt.figure(fig_num)

def plot_2D_sweep(fig_num,
                  X_axis, # X axis values, in labrad units
                  Y_axis, # Y axis values, in labrad units
                  DATA,titl= 'None', Xlabel='PNA PTs [{0}]',Ylabel='MXG PTs [{0}]',txt=None,cmap='seismic_r',Unwrap=True):
    plt.figure(fig_num);plt.clf()
    X,Y=np.meshgrid(X_axis[X_axis.units],Y_axis[Y_axis.units])
    if DATA.dtype==np.complex:
        ax1 = plt.subplot(121)
        if Unwrap:
            plt.pcolormesh(X,Y, np.unwrap(np.angle(DATA)),cmap=cmap,shading ='gouraud')
        else:
             plt.pcolormesh(X,Y, np.angle(DATA),cmap=cmap,shading ='gouraud')
    #    norm1 = cm.colors.Normalize(vmax=np.unwrap(np.angle(DATA)).max(), vmin=-np.unwrap(np.angle(DATA)).max())
     #   cmap = cm.PRGn 
        plt.pcolormesh(X,Y, np.unwrap(np.angle(DATA)),cmap=cmap,shading ='gouraud')#,edgecolors='k', linewidths=0.02)
        plt.title('phase',fontweight="bold")
        plt.ylabel(Ylabel.format(Y_axis.units))
        plt.xlabel(Xlabel.format(X_axis.units))
        plt.colorbar()
        ax2 =plt.subplot(122, sharey=ax1)
        plt.pcolormesh(X,Y, np.log10(np.abs(DATA)),cmap=cmap,shading ='gouraud')#,edgecolors='k', linewidths=0.02)
        plt.title('MLOG',fontweight="bold")
        plt.colorbar()
        plt.xlabel(Xlabel.format(X_axis.units))
    else:
        plt.pcolormesh(X,Y,DATA)
        plt.ylabel(Ylabel.format(Y_axis.units))
        plt.xlabel(Xlabel.format(X_axis.units))


    plt.colorbar()
    if not titl is None:
        plt.suptitle(titl,fontweight="bold")
    if not txt is None:
        plt.text(0,0.5,txt,fontweight="bold")
    plt.show()
    return plt.figure(fig_num)
def plot_2D_cuts_of_SG(fig_num,interval,X_axis,Y_axis,DATA):
    try:
        Y_axis = Y_axis[Y_axis.units]
    except:
        Y_axis= Y_axis
    for i,Yval in enumerate(Y_axis[0::interval]):
        fig = plot_1D_sweep(fig_num,X_axis,DATA[i,:],label='sg_point= {0}'.format(Yval))
    plt.legend()
    return fig



def CW_spectroscopy(pna,#PNA_object 
                          sg, #SG_object
                          pna_freq=None,#7.30898*GHz,# labrad value
                          sg_power = 10*dBm,# a lab_rad list of [Power_sweep_start, power_sweep_stop] or  a labrad value of SG power depending if we sweep for power or frequencey
                          sg_freq = [4.4*GHz,1000*MHz], # a lab_rad list of [sg_frequncey_center (guess of qubit frequence), sg_frequncey_span] or  a labrad value of SG single fequencey depending if we sweep for power or frequencey
                          frq_pts=None, #number of MXG sweep_points
                          Data_format = 'Raw', #sets the Format of Data extracted from PNA 
                          #to be added
                          swp_typ ='SING',#determines if to initate a single ('SING') or continues ('CONT') sweep of the MXG
                          pna_numAv = 1, #number of times the each point of the data is averged over
                          IF_BW = None,pna_power = None, fig_num = 300, save_data= True, in_flag=True,
                          folder_name = None,file_name = None, #Names for saving the file
                          ThreeToneFreq = None, # Option for automatic 3 tone spectroscopy run
                          Big_Dic = None, # dictionary that saves all running parameters
                          return_fig =False):

    """ Perform a spectrosopy with a sweep of a constant frequencey pna frequencey and a 
    frequencey or power sweep of the mxg"""
    if pna_freq is None:
        pna_freq = pna.CwFreq()
        print("PNA freq is set to {} GHz".format(pna_freq['Hz']*1e-9) )
        
    # if in_flag and False:
    #     _,_= CW_spectroscopy(pna, sg, pna_freq=pna_freq,sg_power = sg_power,
    #                           sg_freq=sg_freq,frq_pts= 3, swp_typ ='SING',pna_numAv =1,IF_BW = IF_BW,pna_power=pna_power,fig_num = None, save_data = False,in_flag=False)
    
    dwell_time, pna_X_axis = initilaitze_PNA(pna,pna_freq,pna_power,IF_BW,frq_pts,pna_numAv)
    total_time, sg_X_axis = initilaitze_MXG(sg,sg_freq,sg_power,frq_pts,dwell_time)
    i=0
    pna.visa.write('SENS:SWE:MODE CONT');
    while(i<pna_numAv):
        sg.start_sweep(swp_typ)
        sleep(0.5)
        sleep_time=0
        sleep_step = 3
        while sleep_time < total_time['s'] + sleep_step:
            sleep(sleep_step)
            sleep_time+=sleep_step
        # sleep(total_time['s'])
        if i%10==0 and i>0: print(i)
        pna.auto_scale()
        i+=1
    DATA = pna.getData(typ ='Raw')
    #sg.abort_sweep();#pna.reset_measure();pna.powerOnOff(state='OFF');    
    if fig_num:
        fig = plot_1D_sweep(fig_num, sg_X_axis, DATA)
    if save_data:
        if folder_name:
            if file_name is None:
                file_name ='' 
            Quick_save(NetAnal = pna, averging_folder_name = folder_name,averging_file_name = file_name+ 'CW_'+'Center'+ np.str(sg_freq[0][sg_freq[0].units])+sg_freq[0].units+'_Span'+np.str( sg_freq[1][sg_freq[1].units])+sg_freq[1].units+'_MXGPower'+np.str( sg_power[sg_power.units])+sg_power.units, ToPlot = False  )##TODO change the saving technique
        else:
            if file_name is None:
                file_name =''            
            Quick_save(averging_file_name = file_name+ 'CW_'+'Center'+ np.str(sg_freq[0][sg_freq[0].units])+sg_freq[0].units+'_Span'+np.str( sg_freq[1][sg_freq[1].units])+sg_freq[1].units+'_ScanPower'+np.str( sg_power[sg_power.units])+sg_power.units , ToPlot = False, NetAnal = pna)
            save_config_file()
    if return_fig:
        return sg_X_axis, DATA,fig
    return sg_X_axis, DATA




def POW_spectroscopy(pna,#PNA_object 
                          pna_power = [-40*dBm,-10*dBm],
                          pna_freq=7.5364*GHz,#7.30898*GHz,# labrad value
                          frq_pts=1001, #number of MXG sweep_points
                          Data_format = 'Raw', #sets the Format of Data extracted from PNA
                          swp_typ ='SING',#determines if to initate a single ('SING') or continues ('CONT') sweep of the MXG
                          pna_numAv = 1, #number of times the each point of the data is averged over
                          IF_BW = 100*Hz,fig_num = 400, save_data= True,ElDe_flag=True, folder_name = None,file_name = None):
    
    """ Perform a spectrosopy with a power sweep of  the pna frequencey """
    dwell_time, pna_X_axis = initilaitze_PNA(pna,pna_freq,pna_power,IF_BW,frq_pts,pna_numAv)
    #pna.visa.write('SENS:SWE:MODE CONT');
    i=0;
    while(i<pna_numAv):
        pna.visa.write('SENS:SWE:MODE {0}'.format(swp_typ))
        sleep(dwell_time['s'])
        if i%10==0 and i>0: print(i)
        pna.auto_scale();i=i+1
    DATA =pna.getData(typ =Data_format) 
    if Data_format == 'Raw' and ElDe_flag:
        DATA = DATA*np.exp(2j*np.pi*(pna_freq*pna.electrical_delay()+(pna.get_phase_offset())/360.0))
         
    #sg.abort_sweep();#pna.reset_measure();pna.powerOnOff(state='OFF');    
    if fig_num:
        plot_1D_sweep(fig_num,pna_X_axis,DATA)
    if save_data:
        if folder_name:
            if file_name is None:
                file_name ='' 
            Quick_save(averging_folder_name = folder_name,averging_file_name = file_name+ 'CW_'+'Center'+ np.str(sg_freq[0][sg_freq[0].units])+sg_freq[0].units+'_Span'+np.str( sg_freq[1][sg_freq[1].units])+sg_freq[1].units+'_MXGPower'+np.str( sg_power[sg_power.units])+sg_power.units  )
        else:
            if file_name is None:
                file_name =''            
            Quick_save(averging_file_name = file_name+ 'CW_'+'Center'+ np.str(sg_freq[0][sg_freq[0].units])+sg_freq[0].units+'_Span'+np.str( sg_freq[1][sg_freq[1].units])+sg_freq[1].units+'_ScanPower'+np.str( sg_power[sg_power.units])+sg_power.units)
            save_config_file()
    return pna_X_axis, DATA
def printKwrg(**kwargs):
   for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))
def LIN_spectroscopy(pna,#PNA_object 
                          sg, #SG_object
                          pna_freq=[7.304*GHz,100*MHz],# a lab_rad list of [frequncey center, span]
                          sg_power = -20*dBm,#  a labrad value of SG power 
                          Data_format = 'Raw', #sets the Format of Data extracted from PNA 
                           #to be added
                          sg_freq = 6.135*GHz,#  a labrad value of SG single fequencey 
                          frq_pts= 1001, #number of sweep_points
                          swp_typ ='SING',#determines if to initate a single ('SING') or continues ('CONT') sweep of the MXG
                          pna_numAv = 1, #number of times the each point of the data is averged over
                          IF_BW = 100*Hz,pna_power = -65*dBm,fig_num = 100, 
                          averging_folder_name = None,save_data = False,ElDe = False):
    """ Perform a spectrosopy with a constant frequencey and power of the mxg and with frequencey sweep of the pna, 
    returns the compensted acquired data and the X axis of the sweep"""
    if ElDe:
        pna.config['ElecDelay'] = pna.electrical_delay('AUTO')
    dwell_time, pna_X_axis = initilaitze_PNA(pna,pna_freq,pna_power,IF_BW,frq_pts,pna_numAv)
    total_time, _ = initilaitze_MXG(sg,sg_freq,sg_power,frq_pts,dwell_time)
    i=0
    while(i<pna_numAv):
        pna.visa.write('SENS:SWE:MODE SING')
        sleep(dwell_time['s'])
        if i%10==0 and i>0: print(i)
        pna.auto_scale();i=i+1
    if Data_format == 'Raw':
        #frequencies_for_elDel = pna_X_axis*GHz
        DATA = pna.compensate_ElDel(freq_list=pna_X_axis)
    else: 
        DATA =pna.getData(typ =Data_format)   
#pna.getData(typ ='Raw')#
    #sg.abort_sweep();#pna.reset_measure();pna.powerOnOff(state='OFF');    
    if fig_num:
        plot_1D_sweep(fig_num,pna_X_axis,DATA)
    if save_data:
        Quick_save()
        save_config_file()
    return pna_X_axis, DATA
# @time_spent
'''def DC_Spectroscopy(pna,#PNA_object 
                          sg, #SG_object
                          pna_freq=[7.304*GHz,100*MHz],# a lab_rad list of [frequncey center, span]
                          sg_power = -20*dBm,#  a labrad value of SG power 
                          sg_freq = 6.135*GHz,#  a labrad value of SG single fequencey 
                          frq_pts= 1001, #number of sweep_points
                          swp_typ ='SING',#determines if to initate a single ('SING') or continues ('CONT') sweep of the MXG
                          pna_numAv = 1, #number of times the each point of the data is averged over
                          IF_BW = 100*Hz,pna_power = -65*dBm,fig_num = 100, 
                          averging_folder_name = None,save_data = False,ElDe = False):
    """ Perform a spectrosopy with a constant frequencey and power of the mxg and with frequencey sweep of the pna, 
    returns the compensted acquired data and the X axis of the sweep"""
    if ElDe:
        pna.config['ElecDelay'] = pna.electrical_delay('AUTO')
    dwell_time, pna_X_axis = initilaitze_PNA(pna,pna_freq,pna_power,IF_BW,frq_pts,pna_numAv)
    total_time, _ = initilaitze_MXG(sg,sg_freq,sg_power,frq_pts,dwell_time)
    i=0
    while(i<pna_numAv):
        pna.visa.write('SENS:SWE:MODE SING')
        sleep(dwell_time['s'])
        if i%10==0 and i>0: print(i)
        pna.auto_scale();i=i+1
    DATA =pna.compensate_ElDel()#pna.getData(typ ='Raw')#
    #sg.abort_sweep();#pna.reset_measure();pna.powerOnOff(state='OFF');    
    if fig_num:
        plot_1D_sweep(fig_num,pna_X_axis,DATA)
    if save_data:
        Quick_save()
        save_config_file()
    return pna_X_axis, DATA
# @time_spent    '''
def twoD_sweep(pna,#PNA_object 
                          sg, #SG_object
                          pna_freq=[7.5*GHz,500*MHz],# a lab_rad list of [frequncey center, span] y depending if we sweep two tones or a single tone
                          sg_power = -10*dBm,# a lab_rad list of [Power_sweep_start, power_sweep_stop] 
                          sg_freq = [7.3*GHz,500*MHz], # a lab_rad list of [sg_frequncey_center (guess of qubit frequence), sg_frequncey_span] or  a labrad value of SG single fequencey depending if we sweep for power or frequencey
                          # This is the Start and the Stop of the frequency!!! CHEN 03/07/19 ASAF you are killing me!!!!!!!!!!!!!
                          sg_pts=1001, #number of MXG sweep_points if sweep is of the 2D sweep is of the same insturment This is power
                          pna_pts=2001, #number of points in a sweep of the PNA if sweep is of the 2D sweep is of the same insturment This is frequencey
                          IF_BW = 100*Hz,pna_power = -45*dBm,pna_numAv=1,fig_num =None, 
                          Data_format = 'Raw', #sets the Format of Data extracted from PNA 
                          #To be added
                          save_two_tone = True, ElDe_flag =False,plot_at_int = None,
                          averging_folder_name= None,averging_file_name = None, #Directory and name to save the file
                          ThreeToneFreq = None # Option for automatic 3 tone spectroscopy run
                          ):
    """returns:
            X_axis- the sweep points of the pna, if only single insturment sweep than this is the frequencey
            Y_axis  - the sweep points of the mxg, if only single insturment sweep than this is the power 
            DATA - dictionary that includes an array of complex numbers of size SG_pts*pna_pts which
            is the unwraped raw data"""
            
            
            # Chen: This function should have an Esc option like in TestSequencer where it saves all the info it gathered so far
    
    Data_dic ={}
    if pna_pts is None: pna_pts = sg_pts
    DATA = np.zeros([sg_pts,pna_pts],dtype = 'complex128')
    if not (type(pna_freq)== list): Data_dic['pna_freq'] = pna_freq
    if not (type(pna_power)== list): Data_dic['pna_power']= pna_power
    if not (type(sg_freq)== list): Data_dic['sg_freq'] = sg_freq
    if not (type(sg_power)== list): Data_dic['sg_power'] =sg_power
    
    titl= averging_folder_name; Xlabel='PNA PTs [{0}]';Ylabel='MXG PTs [{0}]';txt=None
    txt=''.join([key+'='+str(itm)+'\n' for key,itm in Data_dic.iteritems()])
    if type(sg_freq)== list and type(sg_power)== list:
        if type(pna_freq)== list or type(pna_power)== list:
            raise ValueError('too many paramter lists')
        Y_axis = np.linspace((sg_power[0])[sg_power[0].units],(sg_power[1])[sg_power[0].units],sg_pts)*sg_power[0].unit
        #the following line should replace above line if I ever change the power input into a span
      #  Y_axis = np.linspace((sg_power[0]-sg_power[1]/2.0)[sg_power[0].units],(sg_power[0]+sg_power[1]/2.0)[sg_power[0].units],sg_pts)*sg_power[0].unit
        for i,pwr in enumerate(Y_axis):
            X_axis, DATA[i,:] = CW_spectroscopy(pna,sg, pna_freq=pna_freq,sg_power = pwr,
                              sg_freq=sg_freq,frq_pts= pna_pts,pna_numAv =pna_numAv,IF_BW = IF_BW,
                              pna_power=pna_power,fig_num = None,save_data=False)#,label = 'sg_power = {0}'.format(pwr))
            ElDe_flag =False
        Ylabel= 'MXG power [{0}]'
        Xlabel = 'MXG freq [{0}]'
    elif type(sg_freq)== list:
        Y_axis = np.linspace((sg_freq[0])[sg_freq[0].units],(sg_freq[1])[sg_freq[0].units],sg_pts)*sg_freq[0].unit  # Chen's
       # Y_axis = np.linspace((sg_power[0])[sg_power[0].units],(sg_power[1])[sg_power[0].units],sg_pts)*sg_power[0].unit  #    Asaf's
        #the following line should replace above line if I ever change the power input into a span
       # Y_axis = np.linspace((sg_power[0]-sg_power[1]/2.0)[sg_power[0].units],(sg_power[0]+sg_power[1]/2.0)[sg_power[0].units],sg_pts)*sg_power[0].unit
        for i,fre in enumerate(Y_axis):
            X_axis, DATA[i,:] = LIN_spectroscopy(pna,sg, pna_freq=pna_freq,sg_power = sg_power,
                              sg_freq=fre,frq_pts= pna_pts,pna_numAv =pna_numAv,IF_BW = IF_BW,
                              pna_power=pna_power,fig_num = None, save_data = False,ElDe =ElDe_flag)#,label = 'sg_freq = {0}'.format(fre))
            ElDe_flag =False
    elif type(sg_power)== list: 
        Y_axis = np.linspace((sg_power[0]-sg_power[1]/2.0)[sg_power[0].units],(sg_power[0]+sg_power[1]/2.0)[sg_power[0].units],sg_pts)*sg_power[0].unit
        for i,pwr in enumerate(Y_axis):
            X_axis, DATA[i,:] = LIN_spectroscopy(pna,sg, pna_freq=pna_freq,sg_power = pwr,
                              sg_freq=sg_freq,frq_pts= pna_pts,pna_numAv =pna_numAv,IF_BW = IF_BW,
                              pna_power=pna_power,fig_num = None, ElDe =ElDe_flag)#,label = 'sg_power = {0}'.format(pwr))
            ElDe_flag =False
    else:
        raise ValueError('this is a single  paramter sweep')
    
    if save_two_tone:
        if averging_folder_name is None:
            averging_folder_name = 'ForsakenSweeps\\'+pna.config['out_folder']
        if averging_file_name is None:
            averging_file_name = ''
    if type(sg_freq)== list:
        if type(sg_power)!= list:
            averging_file_name = '\\' + averging_file_name + '2D' + '_MXGPower'+ np.str( sg_power[sg_power.units])+sg_power.units + '_MXGStart'+ np.str(sg_freq[1][sg_freq[1].units])+sg_freq[1].units+'_MXGStop'+np.str( sg_freq[0][sg_freq[0].units])+sg_freq[0].units
        else:
            averging_file_name = '\\' + averging_file_name + '2D_PowerSweep' + '_MXGPowerStart'+ np.str( sg_power[0][sg_power[0].units])+sg_power[0].units+ '_MXGPowerStop'+ np.str( sg_power[1][sg_power[1].units])+sg_power[1].units + '_MXGCenter'+ np.str(sg_freq[0][sg_freq[0].units])+sg_freq[0].units+'_MXGSpan'+np.str( sg_freq[1][sg_freq[1].units])+sg_freq[1].units
   
    #Chen's: Insert here all other lists cases.... (power sweep/frequency sweep etc.)         
        
        
        
        
        now = datetime.datetime.now()
        
        Ufun.create_folder(averging_folder_name)
        #Data_dic['X_axis']=[X_axis[X_axis.units][0],X_axis[X_axis.units][-1]]*X_axis.unit
        #Data_dic['Y_axis']=[Y_axis[Y_axis.units][0],Y_axis[Y_axis.units][-1]]*Y_axis.unit
        Data_dic['IF_BW']= IF_BW; 
        Data_dic['ElDe'] =pna.electrical_delay()
        Data_dic['phase']= pna.get_phase_offset()
        Data_dic['pnaAvgNum'] = pna_numAv
        pna.save_Yaml(OutYaml = averging_folder_name+averging_file_name+now.strftime("_%Y_%m_%d_%H-%M-%S")+'_Config.yaml', Dic = Data_dic)
        Data_dic['X_axis']=X_axis
        Data_dic['Y_axis']=Y_axis
        Data_dic['Raw']=DATA
        np.savez(averging_folder_name+averging_file_name+now.strftime("_%Y_%m_%d_%H-%M-%S")+'.npz',DATA= Data_dic)
        print ('File path is: ', averging_folder_name+file_name+now.strftime("_%Y_%m_%d_%H-%M-%S")+'.npz')
    if fig_num:
        plot_2D_sweep(fig_num,X_axis,Y_axis,DATA,titl,Xlabel,Ylabel,txt)
        if plot_at_int: 
            plt.figure(fig_num+1);plt.clf()
            plot_2D_cuts_of_SG(fig_num+100,plot_at_int,X_axis,Y_axis,DATA)
        
   
    return X_axis,Y_axis,DATA



def Punchout(pna, 
             pna_power = [-10*dBm,-35*dBm],# a lab_rad list of [Power_sweep_start, power_sweep_stop]     
             pna_freq=[6.6*GHz,800*MHz],# a lab_rad list of [frequncey center, span]
             pna_pwr_pts=21, #number of MXG sweep_points if sweep is of the 2D sweep is of the same insturment This is power
             pna_freq_pts= None, #number of points in a sweep of the PNA if sweep is of the 2D sweep is of the same insturment This is frequencey
             ElDe_flag =True,#determines if to compensate for electreical delay
             Data_format = 'Raw', #sets the Format of Data extracted from PNA
             IF_BW = 100*Hz,pna_numAv=1,fig_num =100, 
             save_two_tone = True, plot_at_int = 3,averging_folder_name= None,averging_file_name = None):
    """prefoms a power sweep of the pna for different values of frequencey
        returns:
            X_axis- the sweep points of the frequencey
            Y_axis  - the sweep points of the power 
            DATA - an array of complex numbers of size pna_freq_pts*pna_pwr_pts which
            is the unwraped raw data"""
    
    
     # Chen: This function should have an Esc option like in TestSequencer where it saves all the info it gathered so far
     
     
    Data_dic ={}
    if pna_freq_pts is None: pna_freq_pts = pna_pwr_pts
    if Data_format == 'Raw':
        DATA = np.zeros([pna_pwr_pts,pna_freq_pts],dtype = 'complex128')
    else:
        DATA = np.zeros([pna_pwr_pts,pna_freq_pts],dtype = 'float64')

    titl= averging_folder_name; Xlabel='Freq  [{0}]';Ylabel='Power [{0}]';txt=None
    txt=''.join([key+'='+str(itm)+'\n' for key,itm in Data_dic.iteritems()])
    X_axis = np.linspace((pna_freq[0]-pna_freq[1]/2.0)[pna_freq[0].units],(pna_freq[0]+pna_freq[1]/2.0)[pna_freq[0].units],pna_freq_pts)*pna_freq[0].unit
    
    #the following line should replace above line if I ever change the power input into a span
  #  Y_axis = np.linspace((sg_power[0]-sg_power[1]/2.0)[sg_power[0].units],(sg_power[0]+sg_power[1]/2.0)[sg_power[0].units],sg_pts)*sg_power[0].unit
            
    for i,fre in enumerate(X_axis):
        Y_axis,DATA[:,i]=POW_spectroscopy(pna,pna_power=pna_power, pna_freq=fre,ElDe_flag=ElDe_flag,
                   frq_pts=pna_pwr_pts, pna_numAv= pna_numAv, IF_BW = IF_BW, fig_num = None, Data_format= Data_format,save_data= False, folder_name = None,file_name = None)

   # DATA = DATA.transpose()
    Ylabel= 'power [{0}]'
    Xlabel = 'freq [{0}]'
    if save_two_tone:
        if averging_folder_name is None:
            averging_folder_name = 'ForsakenSweeps\\'
        if averging_file_name is None:
            averging_file_name = ''


        
        
        now = datetime.datetime.now()
        
        Ufun.create_folder(averging_folder_name)
        #Data_dic['X_axis']=[X_axis[X_axis.units][0],X_axis[X_axis.units][-1]]*X_axis.unit
        #Data_dic['Y_axis']=[Y_axis[Y_axis.units][0],Y_axis[Y_axis.units][-1]]*Y_axis.unit
        Data_dic['IF_BW']= IF_BW; 
        Data_dic['ElDe'] =pna.electrical_delay()
        Data_dic['phase']= pna.get_phase_offset()
        Data_dic['pnaAvgNum'] = pna_numAv
        pna.save_Yaml(OutYaml = averging_folder_name+averging_file_name+'Punchout'+now.strftime("_%Y_%m_%d_%H-%M-%S")+'_Config.yaml', Dic = Data_dic)
        Data_dic['X_axis']=X_axis
        Data_dic['Y_axis']=Y_axis
        Data_dic['Raw']=DATA
        np.savez(averging_folder_name+averging_file_name+'Punchout'+now.strftime("_%Y_%m_%d_%H-%M-%S")+'.npz',DATA= Data_dic)
   # return X_axis,Y_axis,DATA
    if fig_num:
        plot_2D_sweep(fig_num,X_axis,Y_axis,DATA,titl,Xlabel,Ylabel,txt)
#        if plot_at_int: 
#            plt.figure(fig_num+1);plt.clf()
#            plot_2D_cuts_of_SG(fig_num+100,plot_at_int,Y_axis,X_axis,np.transpose(DATA))
#            plot_2D_cuts_of_SG(fig_num+102,plot_at_int,X_axis,Y_axis,DATA)
        
   
    return X_axis,Y_axis,DATA
"""    
def 2D_spectroscopy(pna,      #PNA_object 
                          sg, #SG_object
                          pna_freq=7.30898*GHz,# a lab_rad list of [frequncey center, span] or a labrad value of CW_frequencey depending if we sweep two tones or a single tone
                          sg_power = 14*dBm,# a lab_rad list of [Power_sweep_start, power_sweep_stop] or  a labrad value of SG power depending if we sweep for power or frequencey
                          sg_freq = [5.135*GHz,300*MHz], # a lab_rad list of [sg_frequncey_center (guess of qubit frequence), sg_frequncey_span] or  a labrad value of SG single fequencey depending if we sweep for power or frequencey
                          sg_pts= 3001, #number of MXG sweep_points
                          pna_pts=None, #number of points in a sweep of the PNA
                          IF_BW = 1*kHz,pna_power = -40 *dBm,plot = True, save_two_tone = False):
    
    D1 = False #to determine if for each point in the sweep of the mxg multiple points the PNA must be saved or only a single
    if pna_frq_pts is None or (~type(PNA_freq)== list):
        pna_frq_pts = frq_pts
 """      

#CW_X_axis, CW_DATA=CW_spectroscopy(PNA_object,SG_object_up,fig_num=401,pna_numAv =1,pna_power= -65*dBm,sg_power = 3*dBm,sg_freq = [6.7*GHz,400*MHz],frq_pts=101)
#aa,bb = LIN_spectroscopy(PNA_object,SG_object)
#SG_object.sweep_mode(SWE_MOD ='FREQ')

# PNA_object = percure_data.get_network_analzer()
# SG_object_up = get_signal_generator()

#Chen: Set trigger settings from default to the trigger settings we need to work with
# PNA_object.set_triger_source()
# PNA_object.set_triger_slope()
# PNA_object.set_triger_type()

"""
CW_spectroscopy(PNA_object ,SG_object_up,
                          pna_freq=7.53642*GHz,#7.30898*GHz,# labrad value
                          sg_power = 15*dBm,# a lab_rad list of [Power_sweep_start, power_sweep_stop] or  a labrad value of SG power depending if we sweep for power or frequencey
                          sg_freq = [3.6*GHz,600*MHz], # a lab_rad list of [sg_frequncey_center (guess of qubit frequence), sg_frequncey_span] or  a labrad value of SG single fequencey depending if we sweep for power or frequencey
                          frq_pts=1001, #number of MXG sweep_points
                          swp_typ ='SING',#determines if to initate a single ('SING') or continues ('CONT') sweep of the MXG
                          pna_numAv = 10, #number of times the each point of the data is averged over
                          IF_BW = 100*Hz,pna_power = -40*dBm,fig_num = 300, save_data= False,in_flag=True)

"""
if __name__ == '__main__' and False:
    SG_object_Down = get_signal_generator(loc ='BOTTOM')
    SG_object_Down.on()
    Pna_freq_list = [7.562*GHz,7.582*GHz,7.5801*GHz,7.5786*GHz]
    for i, bottom_power in enumerate([8.0*dBm,4*dBm,0*dBm,'off']):
        SG_object_Down.power(bottom_power)
        SG_object_up.off()
        aa,bb = LIN_spectroscopy(PNA_object,SG_object_up, pna_freq=[7.578*GHz,100*MHz], sg_power = -10*dBm,sg_freq = 6.135*GHz,frq_pts =1001,fig_num=False)
        plot_1D_sweep(10,aa,bb,label='sg_bottom_power = {0}'.format(bottom_power))
        #SG_object_Down.power(bottom_power)
        SG_object_up.on()

        X_axis,Y_axis,DATA=twoD_sweep(PNA_object,SG_object_up,
                                  plot_at_int=3,pna_numAv =10,pna_power= -45*dBm,
                                  save_two_tone = True,sg_pts = 2001,pna_pts =4001,
                                  pna_freq=Pna_freq_list[i],fig_num= 105+i,
                                  sg_power = [-20*dBm,10*dBm],sg_freq = [3.8*GHz,800*MHz],
                                  averging_folder_name='sweep\\' +'power 2 is '+str(bottom_power))
"""
plt.figure(10)
plt.legend()
plt.show()
"""