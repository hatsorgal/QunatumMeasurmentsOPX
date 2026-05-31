# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 18:46:25 2019

@author: Shay
"""

import pathlib, sys
#for new pc
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\freqDomain\low level old stuff')
#for old pc
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\freqDomain\low level old stuff')
from formating_autonames_control_prefernces import *
from MXG_SWEEP import *
# from MXG_SWEEP import MXG5183A
from scipy import optimize
import re
import glob
import os
import matplotlib.pyplot as plt
import yaml; 
# import h5py
from time import sleep
import numpy as np
try: 
    from  InstrumentControl import MXG5183A, AgilentPNA, AgilentVNA 
    from InstrumentControl.general_functions import *
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
from scipy.optimize import curve_fit

# Chen's
import datetime
from scipy.stats.distributions import  t # Necessary for confidence intervals of fit parameters in Resonator_fit_conf

"""  
procure definition: 
    1. to get something, especially after an effort: 
    2. to get a prostitute for someone else to have sex with

"""

def Lorentzian(x,a,b,c,d):
    return (a/((x-d)**2+b**2) + c)

    
    

class Network_analyzer(AgilentPNA): #TODO include all PNA class mesurements in this class
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
    def InitPar(self,filename ='Defult_SetUP.yaml'):
        #self.powerOnOff(state=1)
        self.DATA= {}
        self.config = self.load_config(filename)
        self.inital_config = self.config.copy()
        self.s_parameters(in_port = self.config['in_port'], out_port=self.config['out_port'])
        self.config['Pwr'] = self.power(p=self.config['Pwr'])
        self.config['ElecDelay'] = None
        #self.config['PhConst'] =None
        warn('elecdelay is set to None')
        if self.config['ElecDelay'] =='AUTO':
            self.auto_Elec_Delay()
            self.config['ElecDelay'] = self.electrical_delay()
        else:
            self.config['ElecDelay'] = self.electrical_delay(corr=self.config['ElecDelay'])
        self.config['Sweep']= self.sweep_type(typ= self.config['Sweep'])
        if self.config['Sweep'] =='LIN':
            self.frequencey_span(CenFreq=self.config['FreqCen'], span=self.config['FreqSpan'] )
            sleep(self.get_sweep_time()['s'])
            temp = self.frequencey_span()
            self.config['FreqCen']=temp[0]
            self.config['FreqSpan']= temp[1]
        elif  self.config['Sweep'] =='CW' or self.config['Sweep'] =='POW':
            self.config['FreqCen']= self.CwFreq(f=self.config['FreqCen'])
        else:
            raise ValueError('no such sweep')
        self.config['BanWid'] = self.bandwidth(bw = self.config['BanWid'])      
        self.config['NumOfPt'] =self.num_points(self.config['NumOfPt'])
        self.config['DatFrm'] = [self.format_typ(self.config['DatFrm'])]
        self.config['PhConst'] = self.phase_offset(self.config['PhConst'])
        if 'out_folder' in self.config.keys():
            if self.config['out_folder'] is None:
                self.config['out_folder']= ''# os.getcwd()
        else:
            self.config['out_folder']= ''#os.getcwd()
        #self.DATA ={}
        self.auto_scale()
      #  sleep(self.get_sweep_time()['s'])
        #self.in_port=temp_port[0]
        #self.out_port=temp_port[1]
        return #self.frequencey_span(CenFreq=self.FreqCen, span=self.FreqSpan )
    def save_Yaml(self,OutYaml ='Test_outfile.yaml', Dic = None):
        if Dic is None:
            Dic = self.config
        with open(OutYaml, 'w') as f: 
            yaml.dump(Dic, f)
        return OutYaml 

def noise_data(x_axis, data):
    """ gets the  array data of with shape(len  in and and the time at which it was extrapolated as 1D array
    (x_axis which is assumed to have time labrad units
    and retun the data frequncies as the same thpe of array"""
    return FFT_withfreq(Ufun.labrad_value(in_s(x_axis))[1]-Ufun.labrad_value(in_s(x_axis))[0],data)
def FFT_withfreq(Time_step,data):
    FoTr = np.fft.fft(data)
  #  print(data.shape[-1])
    freq_array=np.fft.fftfreq(data.shape[-1],(in_s(Time_step)))*Hz
    return  freq_array,FoTr
def Extract_plotting_data(Dic_name='DATA',data_type='Y_axis',X_axis='X_axis',full_file_name =None):
    if full_file_name is None:
        return Ufun.extract_dic(Dic_name,folder_name+filename)[X_axis], Ufun.extract_dic(Dic_name,folder_name+filename)[data_type]
    else:
        return Ufun.extract_dic(Dic_name,full_file_name)[X_axis], Ufun.extract_dic(Dic_name,full_file_name)[data_type]
def noise_plot(full_file_name,ax =None,data_type = 'Y_axis'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    freq_array,FoTr = Averg_noise_from_file(full_file_name)
    unit_string = 'Hz'
    if max(abs(freq_array['Hz']))>1000:
        unit_string= 'kHz'
    if max(abs(freq_array['kHz']))>1000:
        unit_string= 'MHz'
    ax.semilogy(freq_array[unit_string],FoTr,'.-')
    plt.title(subfolder_name(full_file_name))    
    plt.xlabel('freq [{0}]'.format(unit_string))
    return ax
def subplot_noise_data( folder_list=None):
    if folder_list is None:
        folder_list = Ufun.file_list()
    nrow,ncol = des_subplot(len(folder_list))
    fig = plt.figure()
    for i,folder_name in enumerate(folder_list):
        ax = fig.add_subplot(nrow,ncol,i+1)
        ax =noise_plot(folder_name,ax=ax)
 #       ax =None
    #fig.tight_layout() 
    return fig
def des_subplot(n): 
    """gets the number of figures and returns the nrow, ncol desired"""
    if n%2 == 0:
       return 2, n/2
    if n%3 == 0:
        return 3,n/3
    return n,1
def raw_data_plot_old(Dic_name = 'DATA', folder_name='CW_off_Res_BW_1000', filename ='\DATA.npz',data_type = 'PHAS'):
    x_axis,data =Extract_plotting_data(Dic_name, folder_name,data_type)
    plt.figure()
    #freq_array,FoTr = FFT_withfreq(data,Ufun.labrad_value(in_s(x_axis))[1]-Ufun.labrad_value(in_s(x_axis))[0])
    plt.plot(x_axis[x_axis.units],data,'.-')
    plt.xlabel(x_axis.units)
    plt.title(folder_name)    
    return 

def plot_axis(ax =None,x=None,y=None,Lintyp='.-',label = False):#Dic_name = 'DATA', folder_name='CW_off_Res_BW_1000', filename ='\DATA.npz',data_type = 'PHAS'):
    #x,y = Extract_plotting_data(Dic_name, folder_name,data_type, filename)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    ax.plot(Ufun.labrad_value(x),Ufun.labrad_value(y),Lintyp,label = label)
    plt.xlabel(Ufun.labrad_units(x))
    plt.ylabel(Ufun.labrad_units(y))
    return ax
def subfolder_name(full_filename,start='avg\\', stop='\\DATA'):
    #name = re.match(start +'.*' +stop,full_filename)
    #print(start)
    #print(stop)
    return full_filename# name.group()[len(start):-len(stop)]
def ploter_main_example(Dic_list=[{'Dic_name':'DATA','directory':'avg\\CW_5to7_LOG_MAG_8_BW_10', 'filename':'\DATA.npz','data_type':'MLOG','title':None}]):# ):
    nrow,ncol = des_subplot(len(Dic_list))
    fig = plt.figure()
    for i,case in enumerate(Dic_list):
        if  type(case) in [dict,str]:
            if type(case) is dict:
                x,y = Extract_plotting_data(case['Dic_name'], case['folder_name'],case['data_type'], case['filename'])
                case_temp =case.copy()
            elif type(case) is str:
                x,y = Extract_plotting_data('DATA',full_file_name= case)
                case_temp = {'title': case[case[:case.rindex('\\')].rindex('\\DATA.npz'),case.rindex('\\DATA.npz')]}
            try:
                if 'title' in case_temp.keys():
                    title=(case_temp['title'])
                else:
                    title = case_temp['folder_name']
            except:
                title =None
        else:
            x,y = case[0],case[1]
            try:
                title = case[2]
            except:
                title =None
        ax = fig.add_subplot(nrow,ncol,i+1)
        """ax = plot_function(x,y,ax)"""
        ax = plot_axis(ax,x,y)
        #ax.plot(Ufun.labrad_value(x),Ufun.labrad_value(y),'.-')

        plt.title(title)
        plt.yscale('log')
        #plt.tight_layout()

    return fig
    #fig = plt.figure()
def data_dic(Dic_name = 'DATA', folder_name='CW_off_Res_BW_1000', filename ='\DATA.npz',data_type = 'MLOG'):
    return {'Dic_name':Dic_name, 'folder_name':folder_name, 'filename':filename,'data_type':data_type}

def get_network_analzer(loc ='TCPIP0K-N5232B-81186inst0INSTR', filename = None):        
    """ I am assuming that the error occures only in the initilization of communication"""
    # filename = 'Pna_Config_Example.yaml' %% Eliya: I'm stopping this insanity
    
    # succsesful_comunication = 0
    # while succsesful_comunication<10:
    #     try:
    #         NetAnal =  Network_analyzer(loc) 
    #         break
    #     except ret_VisaIOError as e:
    #         succsesful_comunication = succsesful_comunication +1
    #         print('failed comunication time {0}'.format(succsesful_comunication))
    # else:
    #     print('last try')
    #     NetAnal =  Network_analyzer(loc)
    NetAnal =  Network_analyzer(loc)
    # NetAnal.InitPar(filename= filename)
    return NetAnal

def Get_Data4averging(fle = 'Pna_Config_Example.yaml',num_of_averges =100,loc ='TCPIP0::K-N5232B-81186::inst0::INSTR'): 
    """ADD away to get data for addtional format types?
    
    Gets: fle- the full yaml filename with all the data for the mesurment,  
    num_of_averges- the number of required averges
    loc - the IP/location of the vna network analyzer
    
    Returns: the instance of the network analyzer as well as the dirictory  
    of the DATA.npz file which  containes the data meant for averging. stored
    the x_axis is saved with the dic_key 'X_axis' and the y_axis is saved as 'Y_axis' """
    #print(fle)
    NetAnal = get_network_analzer(loc=loc, filename = fle)
    print(NetAnal.config['out_folder'])
    if NetAnal.config['out_folder'] is None: print(1)
    x,y=NetAnal.Triger_data()
    raw_data = y
    Data_dic = {} 
    Data_dic['X_axis'] = x 
    Data_dic['format'] =NetAnal.format_typ()
    if NetAnal.config['out_folder'] == 'date':
        outfile_name= date_time_as_string()
    else:
         outfile_name =  NetAnal.config['out_folder']
    averging_folder_name = 'avg\\temp\\'+outfile_name+'.npz'
    Ufun.create_folder(averging_folder_name)
 #   raw_axis = x
    for i in range(max(0,num_of_averges-1)):
       _,y=NetAnal.Triger_data()
       raw_data= np.vstack((raw_data,y))
#       raw_axis = np.hstack((raw_axis,x))
       if i%10==0:
           print(i)
       if i>0 and i%200==0:
           Data_dic['Y_axis'] = raw_data   
           np.savez(averging_folder_name+'\DATA.npz',DATA= Data_dic)
           print('saved {0} realizations'.format(i))
    
    print(averging_folder_name)
    Data_dic['Y_axis'] = raw_data
    
    #NEED TO ENSURE THAT THE FILE IS NOT RRUN OVER 
    np.savez(averging_folder_name+'\DATA.npz',DATA= Data_dic)
    NetAnal.config['Number_of_sweeps']=num_of_averges
    NetAnal.save_Yaml(OutYaml = averging_folder_name+'\config.yaml')
    return NetAnal, averging_folder_name+'\DATA.npz'


def Averg_noise(time,raw_data):
    """ gets  a time vector of length N in labrad seconds as time var
    and a an array of mesutred data of size (number of mesuerments) X N
    finds the freuquncies amplitude of each mesuremnt and returns the frequncey 
    array (with labrad units of frequncey) with the averge of the absoulte value of the frequncey values"""
    freq_arr,Data = noise_data(time,raw_data)
    Data_abs = np.abs(Data)
    return freq_arr,Data_abs.mean(axis = 0)
    
#Continue here#
    """ plt.figure()
    plt.plot(freq_arr['kHz'],mean_data,'.-')
    plt.yscale('log')
    plt.title(temp.config['out_folder'])"""
def Averg_noise_from_file(filename,data_key='DATA',data_type='Y_axis') :
    temp = Extract_plotting_data(data_key,full_file_name =filename,data_type=data_type)
    return Averg_noise(temp[0],temp[1])
   # return temp[0],temp[1]
def raw_data_from_file(filename,data_key='DATA',data_type='Y_axis',realiztion_start= None,number_of_realiztions = 1):
    """extracts all the raw datat from of realiztion_ start to realization_start+number_of_realzations -1 
    if realiztion_start is None will extract all realiztions"""
    temp = Extract_plotting_data(data_key,full_file_name =filename,data_type=data_type)
    if realiztion_start is None:
        return temp[0],temp[1]
    try:
        return temp[0],temp[1][realiztion_start:realiztion_start+number_of_realiztions,:]
    except Exception as e:
        print(e)
        return temp[0],temp[1]
def raw_data_plot(filename= None,data_key='DATA',data_type='Y_axis',realiztion_start= 0,number_of_realiztions = None):
    if filename is None:
        filename = Ufun.file_list()[0]
    x,y = raw_data_from_file(filename,data_key= data_key,data_type=data_type,realiztion_start= realiztion_start,number_of_realiztions = number_of_realiztions)
    ax = None
    try:
        for i in range(y.shape[0]):
            ax = plot_axis(ax,x=x,y =y[i,:],label = 'Realiztion Number: {0}'.format(i))
            plt.legend()
    except:
         ax = plot_axis(ax,x=x,y =y)
    return ax

def smart_Mode_guess(x,y,guess = None):
    
    if  guess:
        return guess
    
    try:
        b_guess = np.abs(x[np.argmin(np.abs(y-(max(y)+min(y))/2))]-x[np.argmax(y)])   # b_guess :)
        a_guess = (np.max(y)-np.min(y))*b_guess**2
        return a_guess, b_guess, np.min(y)
    
    except:
        print ('Could not make a smart guess!!!')
            # [10**11, 10**11, -400*10**(-6),Wr] # Initial guess for the parameters
        return [ 10**11, 10**11, -400*10**(-6)]# Nir 10_09_19  Fit works better also for undercoupled cavities
    
def Mode_center(X_axis,freq_cen=None):
    
    if freq_cen is None:
        try:
            freq_cen = in_hz(X_axis[int(len(X_axis)/2)])
        except:
            freq_cen = in_hz((X_axis[0]+X_axis[-1])/2)
    return in_hz(freq_cen)

def Mode_Data_from_file(filename =None,data_key='DATA',freq_cen =None):
    
    if filename is None:
        filename = Ufun.file_list()[0]        
  
    X_axis,Y_axis = raw_data_from_file(filename,data_key= data_key)
 
    try:
       _,freq_cen = in_hz(raw_data_from_file(filename,data_key= 'Wr'))
    except:
        freq_cen = None
        
    return X_axis,Y_axis,freq_cen   

def Mode_fitNplot(X_axis=None, Y_axis=None,filename =None,data_key='DATA',
                       freq_cen =None,
                       inital_guess=None,
                       fitfunc = Lorentzian,
                       Punchout= None): 

    if X_axis is None:
        X_axis,Y_axis,freq_cen= Mode_Data_from_file(filename =filename,data_key=data_key,freq_cen=freq_cen)   
    
    try:
        Data_dic,popt,pcov= Mode_fit(X_axis,Y_axis,freq_cen =freq_cen, inital_guess=inital_guess, is_correlated_error=True)
    
        return Data_dic, Eliya_new_Mode_printNplot(popt,in_hz(X_axis),Y_axis, Data_dic=Data_dic, Punchout = Punchout, )
    except:
        print("The error were calculated assuming no correlations because the correlations caused an error")
        Data_dic,popt,pcov= Mode_fit(X_axis,Y_axis,freq_cen =freq_cen, inital_guess=inital_guess, is_correlated_error=False)
    
        return Data_dic, Eliya_new_Mode_printNplot(popt,in_hz(X_axis),Y_axis, Data_dic=Data_dic, Punchout = Punchout, )
        
    
def Mode_fit(X_axis, Y_axis,freq_cen =None,
                       inital_guess=None,
                       fitfunc = Lorentzian,
                       is_correlated_error = False):                                                                                   

    
    Temp= smart_Mode_guess(in_hz(X_axis),in_hz(Y_axis))
    Wr = Mode_center(X_axis,freq_cen=freq_cen)
    p0=[Temp[0],Temp[1],Temp[2],Wr]
    print('p0 is')
    print(p0)
    
    
   # def Lorentzian(x,a,b,c,d):
     #   return (a/((x-d)**2+b**2) + c)

    try:
        popt, pcov =  curve_fit(fitfunc, in_hz(X_axis), in_hz(Y_axis),p0 =p0)
        n = X_axis.size    # number of data points
        p = len(popt) # number of parameters
        alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
                            
        dof = max(0, n - p) # number of degrees of freedom
        
                            # student-t value for the dof and confidence level
        tval = t.ppf(1.0-alpha/2., dof) 
        FitRawPar_conf = np.zeros([len(popt),3])
        FitPar_conf = np.zeros([3,3]) # 3 for value, lower and upper bounds, and 3 for Kappa in, Kappa ext and Wr
                            
        for i, p,var in zip(range(n), popt, np.diag(pcov)):
            sigma = var**0.5
            FitRawPar_conf[i] = [p, p - sigma*tval, p + sigma*tval]
        for i in range(3):                   
            FitPar_conf[i]  =  [-FitRawPar_conf[0][i]/FitRawPar_conf[2][i]*np.sqrt(1/(FitRawPar_conf[1][i])**2)/(10**6), 2*np.sqrt((FitRawPar_conf[1][i])**2)/(10**6)+FitRawPar_conf[0][i]/FitRawPar_conf[2][i]*np.sqrt(1/(FitRawPar_conf[1][i])**2)/(10**6),FitRawPar_conf[3][i]] # [KappaExt,KappaInt,Wr]
            
        a = popt[0]; b = popt[1]; c = popt[2]; d = popt[3]
        a_err = np.sqrt(pcov[0,0]); b_err = np.sqrt(pcov[1,1]); c_err = np.sqrt(pcov[2,2]); d_err = np.sqrt(pcov[3,3])
        if is_correlated_error:
            ab_err = pcov[0,1]; ac_err = pcov[0,2]; bc_err = pcov[1,2]
        else:
            ab_err =0; ac_err = 0; bc_err = 0
            
        Data_dic={'X_axis' : X_axis,'Y_axis' : Y_axis, 'Wr' : d,
                  'Kappa_Ext' : -a/c/np.abs(b)/1e6,
                  'Kappa_Int' : 2*np.abs(b)/1e6+a/c/np.abs(b)/1e6,
                  'Wr_error' : d_err,
                   'Kappa_Ext_error' : np.sqrt(np.abs((b**2*c**2*a_err**2+a*(c*(-2*b*c*ab_err-2*b**2*ac_err+a*c*b_err**2+2*a*b*bc_err)+a*b**2*c_err**2))/b**4/c**4))/1e6,
                    'Kappa_Int_error' : np.sqrt((b**2*c**2*a_err**2-2*b*c**2*(a-2*b**2*c)*ab_err-2*a*b**2*c*ac_err+c**2*(a-2*b**2*c)**2*b_err**2+2*a*b*c*(a-2*b**2*c)*bc_err+a**2*b**2*c_err**2)/b**4/c**4)/1e6,
                    }
        FitParNames = ['Kappa External','Kappa Internal','Resonance Frequency']
        FitParUnits = ['MHz','MHz','Hz']
        print('Wr={}+-{}\n kappa_ext={}+-{}\n kappa_int={}+-{}\n'.format(Data_dic['Wr'],Data_dic['Wr_error'],Data_dic['Kappa_Ext'], Data_dic['Kappa_Ext_error'], Data_dic['Kappa_Int'], Data_dic['Kappa_Int_error']))
             
        return Data_dic,popt,pcov
    except:
         return [],[],[]
    

def Normalize_y_axis(y):
    max_y = np.max(y)
    min_y = np.min_(y)
    Mean_y =(max_y+min_y)/2
    return (y-Mean_y)

def Mode_printNplot(popt,x,y,
                    FitPar_conf, # [KappaExt,KappaInt,Wr]
                    fitfunc=Lorentzian,
                    Punchout= None):
   
    W_r = FitPar_conf[2]
    
    max_y = np.max(y)
    min_y = np.min(y)
    
    Mean_y =0#(max_y+min_y)/2
    
    
    Delta_y = max_y-min_y
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    
    res_line = (np.arange(np.min(y),np.max(y),(np.max(y)-np.min(y))/100.0)-Mean_y)/Delta_y
    X_axis =(x-in_hz(W_r))/1000.0
    ax.plot(X_axis,(y-Mean_y)/Delta_y, "b.",X_axis, (fitfunc(x, *popt)-Mean_y)/Delta_y, "r-",np.zeros_like(res_line),res_line,'--k')
    plt.xlabel(r'$\Delta f$'+ '  [kHz]',fontsize=16),plt.ylabel(r'$\Re(S_{11})$' +" Normalized",fontsize=16)
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    if Punchout is None:  
        t = ax.text(X_axis[int(7*len(X_axis)/15)], res_line[int(len(res_line)/8)], r'$f_{r} \approx$' +'{0} [GHz]'.format(np.round(in_hz(W_r)/(10**9),2)), ha="right", va="center", rotation=10,
                size=20 ,
                bbox=bbox_props)
    else:
          t = ax.text(X_axis[int(7*len(X_axis)/15)], res_line[int(len(res_line)/8)], r'$f_{r} \approx$' +'{0} [GHz]'.format(np.round(in_hz(W_r)/(10**9),2))+'\n Power= {0} [dBm]'.format(Punchout),
                ha="right", va="center", rotation=10,
                size=20 ,
                bbox=bbox_props)
    bb = t.get_bbox_patch()
    bb.set_boxstyle("rarrow", pad=0.2)
   # lt.annotate( r'$f_{r} \approx$' +'{0} [GHz]'.format(np.round(in_hz(W_r)/(10**9),2)), xy=(1,1),
    #          xytext=(1.5, .5),
     #         arrowprops=dict( arrowstyle="->" )
    legend = plt.legend(('Data points', 'Fit'),fontsize=16) 
    legend.set_draggable(True)
    if FitPar_conf[0]<0.1:
        FitPar_conf[0]=FitPar_conf[0]*1000.0
        k_ext_text = 'kHz'
    else:
         k_ext_text = 'MHz'
    if FitPar_conf[1]<0.1:
        FitPar_conf[1]=FitPar_conf[1]*1000.0
        k_int_text = 'kHz'
    else:
         k_int_text = 'MHz'
         
    kappa_text = ax.text(X_axis[int(len(X_axis)/5)],res_line[int(7*len(res_line)/15)], r'$\kappa _{ext}/2\pi$'+ 
                                 '={0:.2f}'.format(FitPar_conf[0])+ ' [{}] \n'.format(k_ext_text)+ 
                                 r'$\kappa _{int}/2\pi$'+'={0:.2f}'.format(FitPar_conf[1])+ ' [{}] \n'.format(k_int_text),
                                 ha ='center',va ='center',fontsize = 20)#,fontweight ='bold'
    #plt.title("Cavity data and fit")# \n Parameters: \n"+ str(popt)+"\n'")  
    print('Parameters: \n'+ str(popt)+'\n')
    
#    print('Kappa External: \n'+str(-popt[0]/popt[2]*np.sqrt(1/popt[1])/(10**6))+'MHz \n')
    # Print fit results
    
    return ax

def Eliya_new_Mode_printNplot(popt,x,y,
                    Data_dic, # [KappaExt,KappaInt,Wr]
                    fitfunc=Lorentzian,
                    Punchout = None):
   
    W_r = Data_dic['Wr']
    W_r_err =  Data_dic['Wr_error']
    # W_r_err = (0.5*(np.abs(W_r-Data_dic['Wr_conf'][1])+np.abs(W_r-Data_dic['Wr_conf'][2])))
    kappa_ext = Data_dic['Kappa_Ext']
    kappa_ext_err = Data_dic['Kappa_Ext_error']
    # kappa_ext_err = 0.5*(np.abs(kappa_ext-Data_dic['Kappa_Ext_conf'][1])+np.abs(kappa_ext-Data_dic['Kappa_Ext_conf'][2]))
    kappa_int = Data_dic['Kappa_Int']
    kappa_int_err = Data_dic['Kappa_Int_error']
    # kappa_int_err = 0.5*(np.abs(kappa_int-Data_dic['Kappa_Int_conf'][1])+np.abs(kappa_int-Data_dic['Kappa_Int_conf'][2]))
    
    
    W_r_err_dig = int((np.floor(np.log10(W_r_err))))
    W_r=np.floor(W_r/10**W_r_err_dig)*10**W_r_err_dig if W_r/10**W_r_err_dig<5 else np.floor(W_r/10**W_r_err_dig+1) * 10**W_r_err_dig
    W_r_err=np.floor(W_r_err/10**W_r_err_dig)*10**W_r_err_dig if W_r_err/10**W_r_err_dig<5 else np.floor(W_r_err/10**W_r_err_dig+1.0) * 10**W_r_err_dig
    
    kappa_ext_err_dig = int(np.floor(np.log10(kappa_ext_err)))
    kappa_ext=np.round(np.floor(kappa_ext/10**kappa_ext_err_dig) * 10**kappa_ext_err_dig,5) if kappa_ext/10**kappa_ext_err_dig<5 else np.round(np.floor(kappa_ext/10**kappa_ext_err_dig+1.0) * (10**kappa_ext_err_dig),5)
    kappa_ext_err=np.round(np.floor(kappa_ext_err / 10**kappa_ext_err_dig) * 10**kappa_ext_err_dig,5) if kappa_ext_err/10**kappa_ext_err_dig<5 else np.round(np.floor(kappa_ext_err/10**kappa_ext_err_dig+1.0) * 10**kappa_ext_err_dig,5)
    
    
    kappa_int_err_dig = int(np.floor(np.log10(kappa_int_err)))
    kappa_int=np.round(np.floor(kappa_int/10**kappa_int_err_dig) * 10**kappa_int_err_dig,5) if kappa_int/10**kappa_int_err_dig<5 else np.round(np.floor(kappa_int/10**kappa_int_err_dig+1.0) * 10**kappa_int_err_dig,5)
    kappa_int_err=np.round(np.floor(kappa_int_err/10**kappa_int_err_dig) *10**kappa_int_err_dig,5) if kappa_int_err/10**kappa_int_err_dig<5 else np.round(np.floor(kappa_int_err/10**kappa_int_err_dig+1.0) * 10**kappa_int_err_dig,5)
    
    max_y = np.max(y)
    min_y = np.min(y)
    
    Mean_y =0#(max_y+min_y)/2
    
    
    Delta_y = max_y-min_y
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    X_axis =(x-in_hz(W_r))/1000.0
    
    ax.plot(X_axis, (y-Mean_y)/Delta_y, "b.",X_axis, (fitfunc(x, *popt)-Mean_y)/Delta_y, "r-")
    
    plt.xlabel(r'$\Delta f$'+ '  [KHz]',fontsize=16),plt.ylabel(r'$\Re(S_{11})$' +" Normalized",fontsize=16)
    
    legend = plt.legend(('Data', 'Fit'),fontsize=16) 
    legend.set_draggable(True)
    
    if kappa_ext<0.1 or kappa_int<0.1:
        kappa_ext=kappa_ext*1000.0
        kappa_int=kappa_int*1000.0
        kappa_ext_err=kappa_ext_err*1000
        kappa_int_err=kappa_int_err*1000
        k_ext_text = 'KHz'
        k_int_text = 'KHz'
    else:
         k_ext_text = 'MHz'
         k_int_text = 'MHz'
         
    kappa_text = plt.annotate(r'$f_{r} =$'+'{0}'.format(np.round(W_r/(10**9),4)) + r'$\pm{}$'.format(W_r_err/(10**9)) +' [GHz]' +
                              # '\n'+'Power={0} [dBm]'.format(Punchout) +
                              '\n' + r'$\frac{\kappa _{ext}}{2\pi}$'+ '={}'.format(kappa_ext)+r'$\pm{}$'.format(kappa_ext_err) + ' [{}]'.format(k_ext_text)+ 
                              '\n' + r'$\frac{\kappa _{int}}{2\pi}$'+'={}'.format(kappa_int)+r'$\pm{}$'.format(kappa_int_err)+ ' [{}]'.format(k_int_text),
                              (0.05,0.5),
                              fontsize = 10, 
                              xycoords = 'axes fraction')
    kappa_text.draggable() 
            
    #plt.title("Cavity data and fit")# \n Parameters: \n"+ str(popt)+"\n'")  
    print('Parameters: \n'+ str(popt)+'\n')
    
#    print('Kappa External: \n'+str(-popt[0]/popt[2]*np.sqrt(1/popt[1])/(10**6))+'MHz \n')
    # Print fit results
    
    return ax


def Characterize_Mode(freq_cen =None, averging_folder_name = None, averging_file_name = None,PNA = None,
                     initial_guess=None,triger_data = True, Punchout= None):##TODO add averging

    #raise ValueError('this needs to be completed')
    
    print('Make sure you are on real format \n Also frequencey center is on resonance')
    if PNA is None: # Chen 04/09/19
        PNA = get_network_analzer(filename = 'Pna_Config_Example.yaml') # 

    if freq_cen is None:
        freq_cen = PNA.frequencey_span()[0]
    else:
        PNA.frequencey_span(CenFreq=freq_cen, span=PNA.config['FreqSpan'] )
    if triger_data:    
        X,y=PNA.Triger_data()
    else:
        y=PNA.getData(typ ='Format')
        X=PNA.get_X_axis()    #If you want to average use Triger_data_cont function
    if not (Punchout is None):
        Punchout = PNA.power()['dBm'] 
        
    Data_dic, ax =Mode_fitNplot(X_axis=X['Hz'],Y_axis=y,freq_cen =in_hz(freq_cen),inital_guess=initial_guess,Punchout= Punchout)

    ##TODO separte as own funct
    now = datetime.datetime.now()
    if averging_file_name is not None:
        FileName = averging_file_name
    else:
        FileName = ''
    if averging_folder_name is not None:
        Ufun.create_folder(averging_folder_name)
        path = averging_folder_name+'\\'+FileName+'ResonatorFit'+now.strftime("_%Y_%m_%d_%H-%M-%S")+'.npz'
    else:
        Ufun.create_folder('ForsakenResonatorFits')
        path = 'ForsakenResonatorFits'+'\\'+FileName+'ResonatorFit'+now.strftime("_%Y_%m_%d_%H-%M-%S")+'.npz'
        
    np.savez(path,DATA= Data_dic)
    print ('\n' +'File path: ' + path)
    if triger_data:
        PNA.reset_measure()
        
    plt.tight_layout()
    return Data_dic, ax, path



    
    
def Quick_run(fle='Pna_Config_Example.yaml',num_of_averges =1,noise=False):
    NetAnal,outfile_name =Get_Data4averging(fle,num_of_averges =num_of_averges)
    #plt.ylabel(NetAnal.format_typ())
    ax = raw_data_plot(outfile_name)
    return NetAnal,outfile_name   




# Asaf's: 
    """
def Quick_save(fle = 'Pna_Config_Example.yaml', NetAnal = None, loc ='TCPIP0::K-N5232B-81186::inst0::INSTR'): 
  
#    Gets: fle- the full yaml filename with all the data for the mesurment,  
#    loc - the IP/location of the vna network analyzer
#    NetAnal- the pna instance
#    
#    DATA dic with c
#    Returns: the instance of the network analyzer as well as the directory  
#    of the DATA.npz file which  containing the key 'DATA':
#            Data_dic['X_axis']= the pna x axis as a labrad
#            Data_dic['Raw']= the pna raw data
#            Data_dic['Y_axis']= the  pna formated data
#            Data_dic['format'] = the pna format
#        
#    also returns the network analyzer configuration as a conifg.yaml at the same folder
    #print(fle)
    Data_dic ={}
    if NetAnal is None:
        NetAnal = get_network_analzer(loc=loc, filename = fle)
    Data_dic['X_axis']=NetAnal.get_X_axis()
    Data_dic['Raw']=NetAnal.getData(typ='Raw')
    Data_dic['Y_axis']=NetAnal.getData()
    Data_dic['format'] =NetAnal.format_typ()
    averging_folder_name = 'temp\\'+NetAnal.config['out_folder']
    Ufun.create_folder(averging_folder_name)
    np.savez(averging_folder_name+'\DATA.npz',DATA= Data_dic)

    #NEED TO ENSURE THAT THE FILE IS NOT RRUN OVER 
    NetAnal.save_Yaml(OutYaml = averging_folder_name+'\config.yaml')
    raw_data_plot(averging_folder_name+'\DATA.npz')
    return NetAnal, averging_folder_name+'\DATA.npz'
"""

# Chen's: 
def Quick_save(fle = 'Pna_Config_Example.yaml', NetAnal = None, loc ='TCPIP0::192.168.0.11::inst0::INSTR',
               averging_folder_name = None,averging_file_name = None, Big_Dic = None, ToPlot = True ): 
  
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
        NetAnal = get_network_analzer(loc=loc, filename = fle)
    # The problem here is that PNA, and both MXGs are not imported here. The solution is probably to change all the parameters to be taken from BigWorkingDic
    if Big_Dic is not None:
        update_Big_Dic(PNA = PNA, MXG1 = MXGTop, MXG2 = MXGBottom , Dictionary = Big_Dic)
        Data_dic['BigWorkingDic'] = Big_Dic
    
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
    Ufun.create_folder(averging_folder_name)
    np.savez(SavePath+'.npz',DATA= Data_dic)
    
    # Plot!!
    if ToPlot:
        raw_data_plot(SavePath+'.npz')
    
    
    print( 'File path is: ', SavePath+'.npz')
    #NEED TO ENSURE THAT THE FILE IS NOT RRUN OVER 
    
    # NetAnal.save_Yaml(OutYaml = SavePath+'_Config.yaml')

    return NetAnal, SavePath+'.npz'

def change_yaml_param(filename,dicname ='ConfigInit', param_dic={},new_filename = None):    
    """changes the yamal  params  in a dic of param files according to """   
    with open(filename) as f:
        Yamfile = yaml.load(f)
    if dicname is None:
        pass
    else:
        Yamfile[dicname]=change_dic_params(Yamfile[dicname], param_dic= param_dic)
    if new_filename is None:
        with open(filename, "w") as f:
            yaml.dump(Yamfile, f)
    else:
        with open(outfile_name, "w") as f:
            yaml.dump(Yamfile, f)
def change_dic_params(dic, param_dic={}):
    for key in param_dic.keys():
        dic[key] = param_dic[key]
    return dic


#%%  Chen's functions



def AlmightyPlotFromNPZ(  # I really made an effort not to write BigDicPlot Here
         FilePath,
         title = None,
         WhatToPlot = None,
         CalibOrBase = None, # C for Calibration B for BaseCode
         ylabel = None, # for inserting a new ylabel to 'quick'
         TimeAvgNum = None, # for TimeDrifts, amount of runs to average on each time.
         Freq = False, # For time drift plot, plot Freq if True, plot Decay if False.
         Delay = 1, # Amount of runs to wait before collecting 
         fignum = None # For a few cavity plots on the same figure
         ):
    

   if CalibOrBase is None:
       ToPlot_BigDic=Ufun.extract_dic(name='DATA',file_name=FilePath)
       x = ToPlot_BigDic['X_axis']
       y = ToPlot_BigDic['Y_axis'] 
       
   if CalibOrBase == 'Base':
       FilePath = 'F:\Dropbox (Technion Dropbox)\programing\BaseCode\\' + FilePath
       ToPlot_BigDic=Ufun.extract_dic(name='DATA',file_name=FilePath)
       x = ToPlot_BigDic['PulseDict']['X_axis_accum']
       y =  ToPlot_BigDic['PulseDict']['Y_axis_accum']
       
   if CalibOrBase == 'Cal':
       FilePath = 'F:\Dropbox (Technion Dropbox)\programing\Calibration\\'  + FilePath
       ToPlot_BigDic=Ufun.extract_dic(name='DATA',file_name=FilePath)
       x = ToPlot_BigDic['X_axis']
       y = ToPlot_BigDic['Y_axis'] 

   
      
       
   if WhatToPlot == 'cavity':
        def fitfunc(x,a,b,c,d):
            return (a/((x-d)**2+b**2) + c)
        Wr = ToPlot_BigDic['Wr']


        
       # plt.plot(x,ToPlot_BigDic['Y_axis'])
        try:
            c_guess = np.min(y)
            b_guess = np.abs(x[np.argmin(np.abs(y-(max(y)+min(y))/2))]-x[np.argmax(y)])   # b_guess :)
            a_guess = (np.max(y)-c_guess)*b_guess**2
            p0 = [a_guess, b_guess, c_guess, Wr]
        except:
            print ('Could not make a smart guess!!!')
            p0 = [10**11, 10**11, -400*10**(-6), Wr] # Initial guess for the parameters
 
        popt, pcov =  curve_fit(fitfunc, x, y,p0 =p0 )  #Chen  27_10_19 Fit for lower Q
        n = x.size    # number of data points
        p = len(popt) # number of parameters
        alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
                            
        dof = max(0, n - p) # number of degrees of freedom
        
                            # student-t value for the dof and confidence level
        tval = t.ppf(1.0-alpha/2., dof) 
        FitRawPar_conf = np.zeros([len(popt),3])
        FitPar_conf = np.zeros([3,3]) # 3 for value, lower and upper bounds, and 3 for Kappa in, Kappa ext and Wr
                            
        for i, p,var in zip(range(n), popt, np.diag(pcov)):
            sigma = var**0.5
            FitRawPar_conf[i] = [p, p - sigma*tval, p + sigma*tval]
        for i in range(3):                   
            FitPar_conf[i]  =  [-FitRawPar_conf[0][i]/FitRawPar_conf[2][i]*np.sqrt(1/(FitRawPar_conf[1][i])**2)/(10**6), 2*np.sqrt((FitRawPar_conf[1][i])**2)/(10**6)+FitRawPar_conf[0][i]/FitRawPar_conf[2][i]*np.sqrt(1/(FitRawPar_conf[1][i])**2)/(10**6),FitRawPar_conf[3][i]] # [KappaExt,KappaInt,Wr]
        

            
        if fignum is not None and plt.gcf().number == fignum:
            plt.figure(fignum)
            plt.plot(x/10**9-0.1,y, "k.", x/10**9-0.1, Lorentzian(x, *popt), "r",linewidth=3,  markersize=15 )
            if title is None:
                plt.title("Cavity resonance Lorentzian fit", fontsize = 30,fontweight = 'bold')
            else:
                plt.title(title, fontsize = 25,fontweight = 'bold')
            plt.xlabel("Frequency [GHz]", fontsize = 25)
            plt.ylabel("Real Reflection [AU]", fontsize = 25)
            #plt.legend(('Data', 'fit'), fontsize = 25) 
            plt.tick_params(labelsize = 30)
            print('Parameters: \n'+ str(popt)+'\n')           
                
        else:
            if fignum is None:
                        plt.figure()
            else:
                plt.figure(fignum)
           # fig, ax = plt.subplots()
            plt.plot(x/10**9-0.1,y, "b.", x/10**9-0.1, Lorentzian(x, *popt), "r",linewidth=3,  markersize=15 )
            if title is None:
                plt.title("Cavity resonance Lorentzian fit", fontsize = 30,fontweight = 'bold')
            else:
                plt.title(title, fontsize = 25,fontweight = 'bold')
            plt.xlabel("Frequency [GHz]", fontsize = 25)
            plt.ylabel("Real Reflection [AU]", fontsize = 25)
            #Qplt.legend(('Data', 'fit'), fontsize = 25) 
            plt.tick_params(labelsize = 30)
            print('Parameters: \n'+ str(popt)+'\n')        
            #For Zeno Shifts plot Chen 31/12/19
            """
            fig, ax = plt.subplots()
            #plt.plot(x/10**9-0.1,y, "b.", x/10**9-0.1, Lorentzian(x, *popt), "r",linewidth=3,  markersize=15 )
            plt.plot(x/10**9,y, "b-",   linewidth=5 )
            plt.plot(x/10**9-0.011,y, "k-", linewidth=5 )
            plt.plot(x/10**9-0.022,y, "r-",  linewidth=5 )
            plt.plot(x/10**9-0.033,y, "g-", linewidth=5 )
            if title is None:
                plt.title("Cavity resonance", fontsize = 30,fontweight = 'bold')
            else:
                plt.title(title, fontsize = 25,fontweight = 'bold')
            plt.xlabel("Frequency [GHz]", fontsize = 25)
            plt.ylabel("Real Reflection [AU]", fontsize = 25)
            #plt.legend(('Qubits at ground state', '1 Qubit at excited state',  '2 Qubits at excited state'), fontsize = 25) 
            plt.tick_params(labelsize = 30)
            print('Parameters: \n'+ str(popt)+'\n')
            """

   if WhatToPlot == 'quick':   
        fig, ax = plt.subplots()
        if x.inBaseUnits().units == 'Hz':
            plt.plot(x['GHz'],y, "b.")
            plt.xlim(np.min(x['GHz']),np.max(x['GHz']))
        else:
            startindxSpan = FilePath.find('Span')+4
            stopindxSpan = FilePath.find('GHz',startindxSpan)
            Span = float(FilePath[startindxSpan:stopindxSpan])
            startindxCenter = FilePath.find('Center')+6
            stopindxCenter = FilePath.find('GHz',startindxCenter)
            Center = float(FilePath[startindxCenter:stopindxCenter])
            x = np.linspace(Center-Span/2,Center+Span/2,x[x.units].size)
            plt.plot(x,y, "b.")
            plt.xlim(np.min(x),np.max(x))
        
        if title is None:
            plt.title("Raw Data", fontsize = 30,fontweight = 'bold')
        else:
            plt.title(title, fontsize = 30,fontweight = 'bold')
        plt.xlabel("Frequency [GHz]", fontsize = 25)
        if ylabel is None:
            if ToPlot_BigDic['format'] == 'MLOG':
                plt.ylabel('Power [dBm]', fontsize = 25)
            elif ToPlot_BigDic['format'] == 'PHAS':
                plt.ylabel('Phase [Rad]', fontsize = 25)
            else:
                plt.ylabel('Intensity [AU]', fontsize = 25)
        else:
            plt.ylabel(ylabel, fontsize = 25)
        plt.legend(('Data','fit'), fontsize = 25) 
        plt.tick_params(labelsize = 30)
        
        
   if WhatToPlot == '2DParamp':
        DATA=ToPlot_BigDic['Raw']     
        plot_2D_sweepParamp(fig_num = 763,
                  X_axis = x, # X axis values, in labrad units
                  Y_axis = y, # Y axis values, in labrad units
                  DATA = DATA,title= 'Paramp Current Scan', Xlabel='Scanning Current [mA]',Ylabel='Scanning Frequency [GHz]', cmap = 'Spectral',Unwrap=True)
        
        
        
   if WhatToPlot == 'TimeDriftsAverage': 
       # Assumes 6 seconds per run!!!!!
        TimePerRun = 6
        LoopNum = ToPlot_BigDic['PulseDict']['NumberOfLoops']
        newx = np.linspace(Delay*TimeAvgNum*TimePerRun,np.floor(LoopNum/TimeAvgNum)*TimeAvgNum*TimePerRun/60,np.floor((LoopNum)/TimeAvgNum)-1-Delay)
        fig, ax = plt.subplots()
        newy=np.zeros(newx.size)
        if Freq == True:  
            Array = [item[0] for item in ToPlot_BigDic['PulseDict']['FitParameters']]
            plt.ylabel("Freq [GHz]", fontsize = 25)
        else:
            Array = [item[1] for item in ToPlot_BigDic['PulseDict']['FitParameters']]
            plt.ylabel("Decay [us]", fontsize = 25)
        for count in range(newx.size):
            newy[count] = np.mean(Array[TimeAvgNum*(count+Delay):TimeAvgNum*(count+Delay+1)])
        

        x = newx
        y = newy
        
        
        
        
        
        plt.plot(x,y, "b.")
        if title is None:
            if ToPlot_BigDic['action']=='0':
                if ToPlot_BigDic['RamseyEcho'] ==True:
                    plt.title('Qubit Ramsey Echo (T2e) Time Drift', fontsize = 30,fontweight = 'bold')
                else:
                    plt.title('Qubit Ramsey (T2*) Time Drift', fontsize = 30,fontweight = 'bold')
            elif ToPlot_BigDic['action']=='4':
                plt.title('Qubit Rabi Oscillations', fontsize = 30,fontweight = 'bold')
        else:
            plt.title(title, fontsize = 30,fontweight = 'bold')            
        plt.xlabel("Time [minutes]", fontsize = 25)
        plt.ylim(np.min(y)-np.abs(np.max(y)-np.min(y))*0.05,np.max(y)+np.abs(np.max(y)-np.min(y))*0.05)
        plt.legend(('Data', 'fit'), fontsize = 20) 
        plt.tick_params(labelsize = 30)
       
       
       
       
       
       
       
       
        
   if WhatToPlot == 'FreqFit' :
        def fitfunc(x,a,b,c,d,e):
            return (a+c*np.exp(-b*x)*np.cos(d+e*2*np.pi*x))        
        
        #Smart guess:
        if np.argmin(y)<np.argmax(y):
            aGuess = np.max(y)
        else:
            aGuess = np.min(y)
        
        cGuess = ToPlot_BigDic['PulseDict']['FitParameters'][-1][3]
        bGuess = 1/ToPlot_BigDic['PulseDict']['FitParameters'][-1][1]
        dGuess = ToPlot_BigDic['PulseDict']['FitParameters'][-1][2]
        eGuess = ToPlot_BigDic['PulseDict']['FitParameters'][-1][0]
        
        p0 = [aGuess, bGuess,cGuess,dGuess,eGuess] # Initial guess for the parameters
 
        popt, pcov =  curve_fit(fitfunc, x, y,p0 =p0 )  #Chen  27_10_19 Fit for lower Q
        n = x.size    # number of data points
        p = len(popt) # number of parameters
        alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
                            
        dof = max(0, n - p) # number of degrees of freedom
        
                            # student-t value for the dof and confidence level
        tval = t.ppf(1.0-alpha/2., dof) 
        FitRawPar_conf = np.zeros([len(popt),3])
        FitPar_conf = np.zeros([3,2]) # 3 for value, lower and upper bounds, 1 for decay 1 for freq.
                            
        for i, p,var in zip(range(n), popt, np.diag(pcov)):
            sigma = var**0.5
            FitRawPar_conf[i] = [p, p - sigma*tval, p + sigma*tval]
        for i in range(3):                   
            FitPar_conf[i]  =  [1/FitRawPar_conf[1][i],FitRawPar_conf[4][i]] # [T1,omega]
        xfit = np.linspace(x[0],x[-1],x.size*10)
        fig, ax = plt.subplots()
        plt.plot(xfit, fitfunc(xfit, *popt), "k-",linewidth=4)
        plt.plot(x,y, "b.",   markersize=10)

        if title is None:
            if ToPlot_BigDic['action']=='0':
                if ToPlot_BigDic['RamseyEcho'] ==True:
                    plt.title('Qubit Ramsey Echo (T2e)', fontsize = 30,fontweight = 'bold')
                else:
                    plt.title('Qubit Ramsey (T2*)', fontsize = 30,fontweight = 'bold')
            elif ToPlot_BigDic['action']=='4':
                plt.title('Qubit Rabi Oscillations', fontsize = 30,fontweight = 'bold')
        else:
            plt.title(title, fontsize = 30,fontweight = 'bold')
            
        plt.xlabel("Time [us]", fontsize = 25)
        plt.ylim(np.min(y)-np.abs(np.max(y)-np.min(y))*0.05,np.max(y)+np.abs(np.max(y)-np.min(y))*0.05)
        plt.ylabel("Signal Quadrature [mV]", fontsize = 25)
        plt.legend(('Data', 'Decaying oscillations fit'), fontsize = 25) 
        plt.tick_params(labelsize = 30)
        plt.tight_layout()
        #print('Parameters: \n'+ str(popt)+'\n')
        print('T1: \n'+ str([item[0] for item in FitPar_conf])+'\n')        
        print('Freq: \n'+ str([item[1] for item in FitPar_conf])+'\n') 
        
   if WhatToPlot == 'T1':
        def fitfunc(x,a,b,c):
            return (a+c*np.exp(-b*x))        
        
        #Smart guess:
        if np.argmin(y)<np.argmax(y):
            cGuess = 1
            aGuess = np.max(y)
        else:
            cGuess = -1
            aGuess = np.min(y)
            
        bGuess = 1/(np.argmin(np.diff(y)-np.mean(np.diff(y)))*np.mean(np.diff(x))/np.exp(1))
        p0 = [aGuess, bGuess,cGuess] # Initial guess for the parameters
 
        popt, pcov =  curve_fit(fitfunc, x, y,p0 =p0 )  #Chen  27_10_19 Fit for lower Q
        n = x.size    # number of data points
        p = len(popt) # number of parameters
        alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
                            
        dof = max(0, n - p) # number of degrees of freedom
        
                            # student-t value for the dof and confidence level
        tval = t.ppf(1.0-alpha/2., dof) 
        FitRawPar_conf = np.zeros([len(popt),3])
        FitPar_conf = np.zeros([1,3]) # 3 for value, lower and upper bounds, and 1 for T1
                            
        for i, p,var in zip(range(n), popt, np.diag(pcov)):
            sigma = var**0.5
            FitRawPar_conf[i] = [p, p - sigma*tval, p + sigma*tval]
        for i in range(1):                   
            FitPar_conf[i]  =  [1/FitRawPar_conf[1][i]] # [T1]
        
        xfit = np.linspace(x[0],x[-1],x.size*10)
        fig, ax = plt.subplots()
        plt.plot(x,y, "b.", xfit, fitfunc(xfit, *popt), "r-",linewidth=4,  markersize=15)
        if title is None:
            plt.title('Qubit Excitation Coherence Time T1', fontsize = 25,fontweight = 'bold')
        else:
            plt.title(title, fontsize = 23,fontweight = 'bold')
        plt.xlabel("Time [us]", fontsize = 25)
        plt.ylim(np.min(y)-np.abs(np.max(y)-np.min(y))*0.05,np.max(y)+np.abs(np.max(y)-np.min(y))*0.05)
        plt.ylabel("Signal Quadrature [mV]", fontsize = 25)
        plt.legend(('Data', 'Decaying exponent fit'), fontsize = 25) 
        plt.tick_params(labelsize = 30)
        print('Parameters: \n'+ str(popt)+'\n')
        print('T1: \n'+ str(FitPar_conf)+'\n')
    
   if WhatToPlot == 'PiNoPi':
 
        plt.plot(x,y, "b.",  markersize=15)
        if title is None:
            plt.title('Qubit Excitation Coherence Time T1', fontsize = 25,fontweight = 'bold')
        else:
            plt.title(title, fontsize = 23,fontweight = 'bold')
        plt.xlabel("Time [us]", fontsize = 25)
        plt.ylim(np.min(y)-np.abs(np.max(y)-np.min(y))*0.05,np.max(y)+np.abs(np.max(y)-np.min(y))*0.05)
        plt.ylabel("Signal Quadrature [mV]", fontsize = 25)
        plt.legend(('Data', 'Decaying exponent fit'), fontsize = 25) 
        plt.tick_params(labelsize = 30)
    
    
def Resonator_fit_conf(freq_cen =None, averging_folder_name = None, averging_file_name = None,PNA = None):
    def fitfunc(x,a,b,c,d):
        return (a/((x-d)**2+b**2) + c)
    #raise ValueError('this needs to be completed')
    print('Make sure you are on real format \n Also frequencey center is on resonance')
    if PNA is None: # Chen 04/09/19
        NetAnal = get_network_analzer(filename = 'Pna_Config_Example.yaml') # 
        if freq_cen is None:
            freq_cen =NetAnal.frequencey_span()[0]
           # return freq_cen
        else:
            NetAnal.frequencey_span(CenFreq=freq_cen, span=NetAnal.config['FreqSpan'] )
        # Fit For resonator
        X,y=NetAnal.Triger_data()
        x =X['Hz']
    else: # Chen 04/09/19 I changed this to have the option not to open a new network analyzer each time annoy Visa
        if freq_cen is None:
            freq_cen =PNA.frequencey_span()[0]
           # return freq_cen
        else:
            PNA.frequencey_span(CenFreq=freq_cen, span=NetAnal.config['FreqSpan'] )
        # Fit For resonator
        X,y=PNA.Triger_data() 
        'If you want to average use Triger_data_cont()'
        x = X['Hz']
        
    Wr = freq_cen['Hz']
    try:
        c_guess = np.min(y)
        b_guess = np.abs(x[np.argmin(np.abs(y-(max(y)+min(y))/2))]-x[np.argmax(y)])   # b_guess :)
        a_guess = (np.max(y)-c_guess)*b_guess**2
        p0 = [a_guess, b_guess, c_guess, Wr]
    except:
        print ('Could not make a smart guess!!!')
        p0 = [10**11, 10**11, -400*10**(-6), Wr] # Initial guess for the parameters
    #p0 = [10**11, 10**11, -1*10**(-1),Wr] # Nir 10_09_19  Fit works better also for undercoupled cavities, Q of a few MHz
    #p0 = [1.5*10**(7), 9*10**(8), -1*10**(-2),Wr] # Chen 27_10_19  Fit for higher Q
    ##work from here
    popt, pcov =  curve_fit(fitfunc, x, y,p0 =p0 )  #Chen  27_10_19 Fit for lower Q
    # DON'T USE THE LINES BELOW!!!
 #  Chen  27_10_19 test this to make the function converge
   # popt, pcov =  curve_fit(fitfunc, x, y,p0 =p0, absolute_sigma =True, sigma = (np.abs(y))/100000) # Chen 27_10_19  Fit for higher Q. This makes the algorithm require better convergence.
    
 #   p1, success = optimize.leastsq(errfunc, p0[:], args=(X[ending:starting], Y[ending:starting]))
    n = x.size    # number of data points
    p = len(popt) # number of parameters
    alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
                        
    dof = max(0, n - p) # number of degrees of freedom
    
                        # student-t value for the dof and confidence level
    tval = t.ppf(1.0-alpha/2., dof) 
    FitRawPar_conf = np.zeros([len(popt),3])
    FitPar_conf = np.zeros([3,3]) # 3 for value, lower and upper bounds, and 3 for Kappa in, Kappa ext and Wr
                        
    for i, p, var in zip(range(n), popt, np.diag(pcov)):
        sigma = var**0.5
        FitRawPar_conf[i] = [p, p - sigma*tval, p + sigma*tval]
    for i in range(3):                   
        FitPar_conf[i]  =  [-FitRawPar_conf[0][i]/FitRawPar_conf[2][i]*np.sqrt(1/(FitRawPar_conf[1][i])**2)/(10**6), 2*np.sqrt((FitRawPar_conf[1][i])**2)/(10**6)+FitRawPar_conf[0][i]/FitRawPar_conf[2][i]*np.sqrt(1/(FitRawPar_conf[1][i])**2)/(10**6),FitRawPar_conf[3][i]] # [KappaExt,KappaInt,Wr]
    

    Data_dic = {} 
    Data_dic['Raw_Parameters'] = popt
    Data_dic['X_axis'] = x
    Data_dic['Y_axis'] = y
    Data_dic['Kappa_Ext'] = -popt[0]/popt[2]*np.sqrt(1/(popt[1])**2)/(10**6)
    Data_dic['Kappa_Int'] = 2*np.sqrt((popt[1])**2)/(10**6)+popt[0]/popt[2]*np.sqrt(1/(popt[1])**2)/(10**6)
    Data_dic['Wr'] = popt[3]
    Data_dic['Kappa_Ext_conf'] = FitPar_conf[:][0]# [Value,Lower,Upper]
    Data_dic['Kappa_Int_conf'] = FitPar_conf[:][1] # [Value,Lower,Upper]
    Data_dic['Wr_conf'] = FitPar_conf[:][2] # [Value,Lower,Upper]
    plt.figure()
    plt.plot(x,y, "b.", x, fitfunc(x, *popt), "r-")
    plt.title("Cavity data and fit")# \n Parameters: \n"+ str(popt)+"\n'")
    plt.xlabel("frequency [Hz]")
    plt.ylabel("Real Reflection [AU]")
    plt.legend(('data', 'fit'))    
    print('Parameters: \n'+ str(popt)+'\n')
    
#    print('Kappa External: \n'+str(-popt[0]/popt[2]*np.sqrt(1/popt[1])/(10**6))+'MHz \n')
#    print('Kappa Internal: \n'+str(2*np.sqrt(popt[1])/(10**6)+popt[0]/popt[2]*np.sqrt(1/popt[1])/(10**6))+'MHz \n') # This is wrong!!!
#    print('Resonance Frequency: \n'+str(Wr/10**9)+'GHz \n')




    # Print fit results
    FitParNames = ['Kappa External','Kappa Internal','Resonance Frequency']
    FitParUnits = ['MHz','MHz','Hz']
    for i in range(len(FitPar_conf)):
        print ('\n' + '{0}: {1} {4} [{2}  {3}]'.format(FitParNames[i], FitPar_conf[0][i], FitPar_conf[1][i], FitPar_conf[2][i],FitParUnits[i]))
         
    #Data_dic['format'] =NetAnal.format_typ()
    now = datetime.datetime.now()
    if averging_file_name is not None:
        FileName = averging_file_name
    else:
        FileName = ''
    if averging_folder_name is not None:
        Ufun.create_folder(averging_folder_name)
        path = averging_folder_name+'\\'+FileName+'ResonatorFit'+now.strftime("_%Y_%m_%d_%H-%M-%S")+'.npz'
    else:
        Ufun.create_folder('ForsakenResonatorFits')
        path = 'ForsakenResonatorFits'+'\\'+FileName+'ResonatorFit'+now.strftime("_%Y_%m_%d_%H-%M-%S")+'.npz'
        
    np.savez(path,DATA= Data_dic)
    print ('\n' +'File path: ' + path)
     



   
    # Sets the dictionary that holds the entire state of the system
    ######### Complete this with all relevant parameters #########
def update_Big_Dic(PNA, MXG1,MXG2 , Dictionary):

    for i in ['MXG1','MXG2']:
        if locals()[i].get_output_state():
            Dictionary [  i+'_powerOnOff'] = locals()[i].get_output_state()
            Dictionary [  i+'_power'] = locals()[i].get_power()
            Dictionary [  i+'_frequency'] = locals()[i].get_freq()
            Dictionary [  i+'_sweepMode'] = locals()[i].sweep_mode()
        else:
            Dictionary [  i+'_powerOnOff'] = locals()[i].get_output_state() 
    
    if PNA.powerOnOff():
        Dictionary [  'PNA_powerOnOff'] = PNA.powerOnOff()
        Dictionary [  'PNA_NumPoints'] = PNA.num_points()
        Dictionary [  'PNA_Bandwidth'] = PNA.bandwidth()
        Dictionary [  'PNA_S_parameters'] = PNA.s_parameters()
        Dictionary [  'PNA_ElectricalDelay'] = PNA.electrical_delay()
        Dictionary [  'PNA_AveragesNum'] = np.int(PNA.averages())
        
        Dictionary [  'PNA_SweepType'] = PNA.get_sweep_type()
        
        Dictionary [  'PNA_FreqStart'] = PNA.get_frequency_start()
        Dictionary [  'PNA_FreqEnd'] = PNA.get_frequency_End()
        Dictionary [  'PNA_CWFreq'] = PNA.CwFreq()
    else:
        Dictionary [  'PNA_powerOnOff'] = PNA.powerOnOff() 
        
 
def initialize_system_for_spectroscopy(PNA = None, MXGTop = None, MXGBottom = None ):
    # Set triggers to fit the triggers we work with
    
    if PNA is not None:
       PNA.visa.close()
    if MXGTop is not None:
       MXGTop.visa.close()
    if MXGBottom is not None:
       MXGBottom.visa.close()
    
    PNA = get_network_analzer()
    PNA.set_triger_source()
    PNA.set_triger_type()
    PNA.set_triger_slope()
    PNA.set_following_triger()
    
    # Initialize MXGs for spectroscopy
    try:
        MXGMiddle = MXG5183A(NetworkAddress = 'MIDDLE')
    except:
        print('Could not connect MXG MIDDLE')
        MXGMiddle = None
    try:
        MXGTop = MXG5183A(NetworkAddress ='TOP')
    except:
        print('Could not connect MXG TOP')
        MXGTOP = None
    try:
        MXGBottom = MXG5183A(NetworkAddress ='BOTTOM')
    except:
        print('Could not connect MXG BOTTOM')
        MXGTOBottom = None
    return PNA,MXGTop,MXGMiddle,MXGBottom
    

      
            ############ Make here a function that sets the state of the system according to the dictionary ########




def plot_2D_sweepParamp(fig_num,
                      X_axis, # X axis values, in labrad units
                      Y_axis, # Y axis values, in labrad units
                      DATA,title= 'None', 
                      Xlabel='PNA PTs [{0}]',
                      Ylabel='MXG PTs [{0}]',
                      txt=None,Unwrap= True,
                      cmap='Spectral'):
        plt.figure(fig_num);plt.clf()
        X,Y=np.meshgrid(X_axis[X_axis.units],Y_axis[Y_axis.units])
        #if DATA.dtype==np.complex:
        #ax1 = plt.subplot(121)
        if Unwrap:
            color_data = (np.unwrap(np.angle(DATA))).transpose()
        else:
             color_data = (np.angle(DATA)).transpose()
        
        
        plt.pcolormesh(X,Y,color_data,cmap=cmap,shading ='gouraud' )#,edgecolors='k', linewidths=0.02)
        #plt.title('Phase',fontweight="bold", fontsize = 35)
        plt.ylabel(Ylabel.format(Y_axis.units), fontsize = 25)
        plt.xlabel(Xlabel.format(X_axis.units), fontsize = 25)
        cbar = plt.colorbar()
        cbar.set_label('Phase', fontsize = 25)

        
        """      
        ax2 =plt.subplot(122, sharey=ax1)
        plt.pcolormesh(X,Y, (np.log10(np.abs(DATA))).transpose(),cmap=cmap,shading ='gouraud' )#,edgecolors='k', linewidths=0.02)
        plt.title('Power [dBm]',fontweight="bold")
        plt.xlabel(Xlabel.format(X_axis.units))
        plt.colorbar()
        """
    
    
        if not title is None:
            plt.suptitle(title,fontweight="bold", fontsize = 35)
        if not txt is None:
            plt.text(0,0.5,txt,fontweight="bold", fontsize = 25)
        plt.show() 
        
    
    

#        
    ############ Main ############
if __name__ == '__main__' and False: # Chen 16/10/19 I inserted the False again
    if  NetAnal is None:
        NetAnal=get_network_analzer()
        NetAnal.config['out_folder'] = 'date'
    if False:
        try:
            NetAnal.powerOnOff(state=1)
        except:
            pass
    if NetAnal is not None:
       NetAnal.visa.close()
    NetAnal=get_network_analzer()
    time_start = time.time()
    outfile_list =[]
    fl_list =['Pna_Config_Example.yaml']#'Run_files\CW_Reso*PHAS_BW_10000.*','Run_files\CW_Reso*PHAS_BW_1000.*','Run_files\CW_Reso*PHAS_BW_10.*','Run_files\CW_Reso*PHAS_BW_1.*']#['Pna_Config_Example.yaml']'Pna_Config_Example.yaml']#'Run_files\CW_free*MLOG_BW_10000.*','Run_files\CW_free_MLOG_BW_1000.*''Pna_Config_Example.yaml']#,'Run_files\CW_Reso*PHAS_BW_0.1.*']#,
    j =0
    if len(fl_list)==0 and False:
        fl_list =Ufun.file_list()
    for fl in (fl_list):     
       file_list = glob.glob(fl)
       print (file_list)
       for i,fle in enumerate(file_list):
               NetAnal,outfile_name =Get_Data4averging(fle,num_of_averges =25000)
               outfile_list.append(outfile_name)
               #_ = noise_plot(outfile_name)
              # if i+1%4 ==0 and i>=0 :
               #    fig = subplot_noise_data(outfile_list[j+i-4:j+i]) 
       #j = j+len(file_list)
    print('time for data is: {0}'.format(time.time() -time_start))
    if False:
        for fle in glob.glob('re*10.yaml') :#'Run_files\CW_Reso*PHAS_BW_0.1.*'):
            NetAnal,outfile_name =Get_Data4averging(fle,num_of_averges =10000)
            outfile_list.append(outfile_name)
            _ = noise_plot(outfile_name)
        for i in range(len(outfile_list)):
            if (i+1)%4 ==0:
                print(i)
                fig =subplot_noise_data(folder_list=outfile_list[i-3:i+1])
    for fl in outfile_list:
       _ = noise_plot(fl)
       print (' CW Freq = {0} GHz \n Power = {1} dBm \n IF BW = {2} Hz'.format(NetAnal.CwFreq()['GHz'],NetAnal.power()['dBm'], NetAnal.bandwidth()['Hz'] ))
       plt.title((' CW Freq = {0} GHz \n Power = {1} dBm \n IF BW = {2} Hz'.format(NetAnal.CwFreq()['GHz'],NetAnal.power()['dBm'], NetAnal.bandwidth()['Hz'] )))
      # plt.title(0.5, 0.5,' CW Freq = (0) GHz \n Power = (1) dBm \n IF BW = (2) Hz'.format(NetAnal.CwFreq()['GHz'],NetAnal.power()['dBm'], NetAnal.bandwidth()['Hz'] ),horizontalalignment='center',verticalalignment='center')
        
    #NetAnal.powerOnOff(state=0)
    #temp.Triger_to_cont()
                
    time_total = time.time() - time_start 
    print('total run time is: {0}'.format(time_total))    