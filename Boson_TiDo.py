# -*- coding: utf-8 -*-
"""
Created on Thursday August 12 12:17:59 2020

@author: Asaf Diringer
"""

# common:
import matplotlib
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from warnings import warn
from time import sleep, time
# OPX Qunatum Machine:
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig

import types
from smart_fit import sFit, non_TimeDomain_fit
import Time_characterzation
from Time_domain_base_class import *
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plotting import plot_2D, plot_fft, next_fig_num_by_name

class Boson_TiDo_Chara( Time_characterzation.TiDo_Chara ):
#%% initialzation and attributes    
    def __init__(self,Config, #make sure you follow the config example with the names + nust include to sub dicitonaries, one for the opx "opx_config" and an auxillary config "aux_config"
                 main_mem      = 'mm1' ,  # memmory mod
                 sec_mem       =  'mm2' ,  # memmory mod
                 mem_chi_list  = None, #chi of main mode 
                 UnDisp1_pulse = 'UnDisp1' ,
                 ConDisp1_pulse = 'Fast_ConDisp1',
                 UnRotPi2_pulse ='UnRotPi2_pulse',
                 ConRotPi2_pulse ='ConRotPi2_pulse',
                 UnRotPi_pulse ='pi2_pulse',#UnRotPi_pulse',
                 ConRotPi_pulse ='ConRotPi_pulse',
                 is_fixCD_cPhase = False,
                 CD_cPhase_dic = None,
                 is_Yale =True, #determines wheter the fast conditional displacment pulse inculdes a pi_pulse and a negative displacment as done in Yale or not
                 element_dics= {},#dict=None,#a dictionary where each key is a dic with paramters important that element like {self.main_mem:{'kappa':10(in Hz)}}
                 #mem_chi = [],#
                 **kwargs):
        
        #flags:
        self.is_internal_sweep = True
        self.is_subplot  = False
        self.is_debugging= True  
        self.is_two_mode_disp = False 
        
        #input:    
        self.main_mem      = main_mem
        self.scnd_mem      = sec_mem
#        self.mem_chi       = mem_chi
        
        self.UnDisp1_pulse = UnDisp1_pulse
        self.ConDisp1_pulse= ConDisp1_pulse

        self.ConRotPi2_pulse=ConRotPi2_pulse
        self.ConRotPi_pulse =ConRotPi_pulse

        self.UnRotPi2_pulse =UnRotPi2_pulse
        self.UnRotPi_pulse=UnRotPi_pulse
        
        #Expermients params:
        # self.cavity_T1_params  = dict()
        self.is_Yale    = is_Yale # will be changes to OPX prog  after call to load_cavity_T1_measurement()
        self.ConDisp_correction = 0
        
        #correctig phase for fast conditional displacment
        self.CD_cPhase_dic = CD_cPhase_dic#dict('mm1'= dict('Fast_ConDisp1'=0.1[in units of 2pi])  )
        self.is_fixCD_cPhase = is_fixCD_cPhase 
        #Parent class init:
        self.element_dics = element_dics
        super().__init__(Config, **kwargs)
#%%
    def calc_run_time(self, ExperimentName):
        if ExperimentName is None:
            raise ValueError("Call this function with experiment name")
        if ExperimentName == 'T1':
            N_avg               = self.cavity_T1_params["N_avg"]             
            step_size_clks      = self.cavity_T1_params["step_size_clks"]   
            num_of_pnts         = self.cavity_T1_params["num_of_pnts"]  
            max_seq_time        = self.cavity_T1_params["max_seq_time"]  
            measurement_length  = self.config['pulses']['meas_pulse_in']['length'] 
            pi2_length          = self.config['pulses']['pi2_pulse_qb1_in']['length']
            Displacement_length = self.config['pulses']['Displacement']['length'] 

            run_time = N_avg * num_of_pnts * (
                      Displacement_length
                    + max_seq_time / 2 
                    + measurement_length
                    + 2 * pi2_length 
                    + self.wait_between_seq*4
                )
        return run_time


        
    def normalize_readout(self):# a function to test for shift in readout due to cross kerr
        pass
    #split in to cases and call functions to implement
#%%

#%% cross kerr measurment  
    
    def load_CrossKerr(self, num_of_pnts = 50, # How many points to measure, .
                                    max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    Displacement_pulse = None,
                                    Scale_displacment = [1.0],
                                    mem_mode=None,
                                    Start_time = 0,
                                    CrossKerr_params = None,#if true than no conditonal rotation will be performed and the fit will be done and fit is through regular exp
                                    update_prog = True):
        
        self.CrossKerr_results = dict()
        self.CrossKerr_results['t'] = np.linspace(max_seq_time/(num_of_pnts-1) , max_seq_time, num_of_pnts)
        if CrossKerr_params is None:
            if mem_mode is None: mem_mode = self.main_mem
            if Displacement_pulse is None: Displacement_pulse = self.UnDisp1_pulse
            # if seq_func is None: seq_func = self.cavity_T1_seq
    
            # max_seq_time = int(max_seq_clks * 4)                            
            max_seq_clks = int((max_seq_time-Start_time) // 4)
            time_init    = int(Start_time // 4)
            step_size_clks =  max_seq_clks // (num_of_pnts-1)
            # T1_N_avg = N_avg

            #Save parameters:
        
            CrossKerr_params = {}
            CrossKerr_params["N_avg"]              = N_avg
            CrossKerr_params["Start_time"]         = Start_time
            CrossKerr_params["step_size_clks"]     = step_size_clks
            CrossKerr_params["max_seq_time"]       = max_seq_time
            CrossKerr_params["num_of_pnts"]        = num_of_pnts
            CrossKerr_params["Displacement_pulse"] = Displacement_pulse
            CrossKerr_params["mode"]               = mem_mode          
        else:
            # N_avg              = CrossKerr_params["N_avg"]              
            # step_size_clks     = CrossKerr_params["step_size_clks"]     
            Start_time       = CrossKerr_params["Start_time"] 
            max_seq_time       = CrossKerr_params["max_seq_time"] 
            num_of_pnts        = CrossKerr_params["num_of_pnts"]
            max_seq_clks       = int(max_seq_time // 4)
            step_size_clks     =  max_seq_clks // num_of_pnts
            Displacement_pulse = CrossKerr_params["Displacement_pulse"]
            Scale_displacment  = CrossKerr_params["Scale_displacment"] 
            mem_mode           = CrossKerr_params["mem_mode"] 
            
            

        if update_prog: self.CrossKerr_params=CrossKerr_params
            
        #Continue with T1_from cross kerr
        #Calculate and show expeced runtime:
  
        try:
            run_time = N_avg*num_of_pnts*(max_seq_time/2+self.pulse_len(self.main_readout,self.ro_pulse)+self.pulse_len(mem_mode,Displacement_pulse)+self.wait_between_seq * 4)
            print('Run time is {}s'.format(round(run_time * 1e-9)))         
        except:
            print('cant calculate time')
        
        #Create program:
        with program() as prog:
            DeltaT  = declare(int)
            n       = declare(int)
            I       = declare(fixed)
            Q       = declare(fixed)
            with for_(n,0,n<N_avg,n+1):
                with for_(DeltaT, time_init , DeltaT<=max_seq_clks + time_init, DeltaT + step_size_clks):
                    for s_disp in Scale_displacment:
                        play(Displacement_pulse*amp(s_disp) , mem_mode )
                    align(mem_mode,self.main_readout)
                    # Wait DeltaT time to observe T1 decay:
                    with if_(~(DeltaT==0)):
                       wait(DeltaT, mem_mode)
                       align(mem_mode,self.main_readout)
                       
                    self.perform_full_measurement(I,Q)
                    save(DeltaT, 't')
        
        if update_prog: self.CrossKerr_prog = prog
        self.last_prog = prog
        
        return prog,CrossKerr_params
    
    
    def normalize_CrossKerr(self,params,
                            results, 
                            prog = None,
                            multiply = None,
                            prev_CrossKerr = None,
                            is_spec = False,
                            **kwargs):
        if multiply is None:
            if prog is None:
                results['I']
                # _,results['I_CrossKerr'],results["Q_CrossKerr"] =  self.CrossKerr_Removal(params= params,**kwargs) if prev_CrossKerr is None else [None,prev_CrossKerr["I_CrossKerr"],prev_CrossKerr["Q_CrossKerr"]]
            elif is_spec:
                _,results['I_CrossKerr'],results["Q_CrossKerr"]   =  self.run_spec(prog, params['N_avg'],**kwargs )
            else:
                _,results['I_CrossKerr'],results["Q_CrossKerr"] =  self.run_prog(prog, params['N_avg'],**kwargs ) #if prev_CrossKerr is None else [None,prev_CrossKerr["I_CrossKerr"],prev_CrossKerr["Q_CrossKerr"]]
            
            results["I_orig"] =  results["I_orig"] if 'I_orig' in results.keys() else results["I"]
            results["Q_orig"] =  results["I_orig"] if 'Q_orig' in results.keys() else results["Q"]
            
            results["I"] = results["I_orig"] - results["I_CrossKerr"]
            results["Q"] = results["Q_orig"] - results["Q_CrossKerr"]
        else:
            raise ValueError('single measurment not yey supported')
        
        return results
    
    def CrossKerr_Removal(self,
                          params, #The dic of the original program params
                          N_avg =None,**kwargs):
        
        
        if N_avg is None: N_avg= params['N_avg']
        
        CrossKerr_prog,dump=self.load_CrossKerr(N_avg=N_avg,CrossKerr_params =params,update_prog =False)
                
        return self.run_prog(CrossKerr_prog ,N_avg ) #cross kerr t, cross kerr I ,cross kerr Q
    
    # self.perform_full_measurement(I,Q)
    
    def complete_CrossKerr_measurement(self,num_of_pnts = 50, # How many points to measure, .
                                    max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    normalize = False,#if normalize is true the program will run 
                                    just_run=True,
                                    **kwargs):
        raise ValueError('not working')
       
        self.cavity_T1_results["t"], self.cavity_T1_results["I"], self.cavity_T1_results["Q"] = self.run_prog(self.cavity_T1_prog ,self.cavity_T1_params["N_avg"] )        
        
        self.cavity_T1_results["t"], self.cavity_T1_results["I"], self.cavity_T1_results["Q"] = self.run_prog(self.cavity_T1_prog ,self.cavity_T1_params["N_avg"] )        
        return self.plot_cavity_T1_measurement(**kwargs)
    
    
    def run_CrossKerr_measurement(self,
                                 **kwargs):
        if not hasattr(self, 'CrossKerr_prog'): raise ValueError("Idiot! You did not write T1 program")

        self.CrossKerr_results["I"], self.CrossKerr_results["Q"] = self.run_prog(self.CrossKerr_prog ,self.CrossKerr_params["N_avg"] )        
        
        return self.plot_CrossKerr_measurement(**kwargs)
    
    
    def plot_CrossKerr_measurement(self,start_x_fit=None,**kwargs):
        prog_name = 'T1 CrossKerr'
        fig_num = prog_name + ' ' + str(self.next_fig_num_by_prog_name(prog_name))

        try:
            results, fit_fidelty, fig = self.fit_and_plot( 'Exp', self.process_data(self.CrossKerr_results, **kwargs), ti = self.CrossKerr_results["t"] ,  title_str = f"Cross Kerr, ({self.CrossKerr_params['mode']})",is_thresh = False,fig_num=fig_num,**kwargs ) #txt="Insert My Great Text Here" 
        except:
            results, fit_fidelty = self.fit_and_plot( 'Exp' , self.process_data(self.CrossKerr_results, **kwargs), ti = self.CrossKerr_results["t"] ,  title_str = f"Cross Kerr, ({self.CrossKerr_params['mode']})",is_thresh = False,fig_num = fig_num, **kwargs ) #txt="Insert My Great Text Here" 
            fig = False
        self.CrossKerr_fit = dict(T1 = results[0],T1_error = fit_fidelty[0])
        return
    ##TODO can I make this work for any program?

        # return self.cat_and_back_results]
    def run_for_phase(self,prog_function,is_use_previous = False,is_remove= False,is_phase=False,is_measure_Z = False,var_name ='scale_amp',normalize = False,plot_normaliztion =False,
                      duration_limit=0,
                      is_grid = False,
                      data_limit=0,**kwargs):
        
        prog,params = prog_function(is_remove = is_remove, is_phase = is_phase, is_measure_Z = is_measure_Z,normalize=normalize,**kwargs)
        pi2_phase_list = params["pi2_phase_list"]
        if is_grid:
            x_grid,y_grid       = np.meshgrid(np.array(params[var_name[0]+"_list"]),np.array(params[var_name[1]+"_list"]))
            new_shape      = [params['N_avg'],x_grid.size, len(pi2_phase_list)]
            var_list            = [x_grid,y_grid]
        else:
            var_list       = np.array(params[var_name+"_list"])
            new_shape      = [params['N_avg'],*var_list.shape, len(pi2_phase_list)]
        self.qm_server.clear_all_job_results()
        job = self.qm.execute(prog, flags = ['fix-literal-duration'], duration_limit=duration_limit, data_limit=data_limit)
        job.result_handles.wait_for_all_values()
      
        self.last_prog = prog
        self.last_job = job
       
        res_handles = job.result_handles
        I_handle = res_handles.get('I')
        Q_handle = res_handles.get('Q')

        I_handle.wait_for_all_values()
        Q_handle.wait_for_all_values()

        I_res_orig = I_handle.fetch_all()['value'].reshape(*new_shape)
        Q_res_orig = Q_handle.fetch_all()['value'].reshape(*new_shape)    #.r
        results = dict(var = var_list,I_unprocessed = I_res_orig, Q_unprocessed = Q_res_orig)
      
        if is_remove:
            results_I = np.squeeze(I_res_orig[:,:,::2]-I_res_orig[:,:,1::2])
            results_Q = np.squeeze(Q_res_orig[:,:,::2]-Q_res_orig[:,:,1::2])
        else:
            results_I = np.squeeze(I_res_orig)
            results_Q = np.squeeze(Q_res_orig)
        if is_phase:
            results['Re'] = dict(var = var_list,I = results_I[:,:,0],Q = results_Q[:,:,0] )
            results['Im'] = dict(var = var_list,I = results_I[:,:,1],Q = results_Q[:,:,1] )
            if is_measure_Z: results['Z'] = dict(var = var_list,I = results_I[:,:,-1],Q = results_Q[:,:,-1])
            plot_normaliztion =False
            
        else:
            if normalize:
                results['I'] = np.squeeze(I_res_orig[:,:,0] - I_res_orig[:,:,1])
                results['Q'] = np.squeeze(Q_res_orig[:,:,0] - Q_res_orig[:,:,1])

                results['I_orig'] = np.squeeze(I_res_orig[:,:,0]) 
                results['Q_orig'] = np.squeeze(Q_res_orig[:,:,0])
                
                results['I_CrossKerr'] = np.squeeze(I_res_orig[:,:,1])
                results['Q_CrossKerr'] = np.squeeze(Q_res_orig[:,:,1])
            else:
                results = dict(var = var_list,I =  results_I, Q = results_Q)
                plot_normaliztion =False
                
        return results,params,plot_normaliztion
    
 
    def get_qb_measure_seq_list(self,pi2_phase=0.5,is_remove= False,is_phase=False,is_measure_Z= False,normalize = False,**kwargs):
        # Used when the main sweep is not on the qubit but on the memory mode such as for cavity ramsey vaccum 
        if normalize:
            return [pi2_phase]*2,[1,0]
            
            
        pi2_phase_list = []
        # pi_amp_list    = []
        Delta_remove = [0,0.5]  if is_remove else [0]
        Delta_phase  = [0,0.25] if is_phase  else [0]

        for Dp in Delta_phase:
            for Dr in Delta_remove:
                pi2_phase_list.append(pi2_phase-Dr+Dp)
                
        qb_pulse_list = [1]*len(pi2_phase_list)
        # qb_amp_list   = [1.0]*len(pi2_phase_list)
        
        if is_measure_Z:
            pi2_phase_list.append(0)
            qb_pulse_list.append(2)
            # qb_amp_list.append(0.0)
            if is_remove: 
                pi2_phase_list.append(0)
                qb_pulse_list.append(0)
                # qb_amp_list.append(1.0)
        return np.round(pi2_phase_list,6),qb_pulse_list#,qb_amp_list

    
    def plot_normaliztion(self):
        return
    def plot_phase(self, params,Displacement_pulse= None, mem_mode = None,scnd_mem=None,xlabel='', prog_name='',fig_num= None,results=None,is_two_mode_disp = None, ti=None,fit_types =[None,None,None,None],**kwargs):
        
        if is_two_mode_disp is None: is_two_mode_disp = self.is_two_mode_disp

        try:
            if mem_mode is None: mem_mode  = params["mem_mode"]
            if Displacement_pulse is None: Displacement_pulse = params["Displacement_pulse"]
            if is_two_mode_disp and scnd_mode is None: 
                scnd_mode = params[scnd_mem]  if "scnd_mem" in params.keys else self.scnd_mode    
        except: 
            mem_mode = self.main_mem
            scnd_mode = ''
        
        print 
        prog_name = prog_name+f' {mem_mode}'
        if is_two_mode_disp: prog_name = prog_name + f' {is_two_mode_disp} {scnd_mem}'
            
        if fig_num is None: fig_num = prog_name + ' ' + str(self.next_fig_num_by_prog_name(prog_name))
        
        fig = plt.figure(fig_num)
        ax1 = plt.subplot(221)
        ax1.set_title('Real')

        ax2 = plt.subplot(222)
        ax2.set_title('Imag')
        
        ax3= plt.subplot(223)
        ax3.set_title('phase')
        ax3.set_xlabel(xlabel)
        
        ax4= plt.subplot(224)
        ax4.set_title('Radius')
        ax4.set_xlabel(xlabel)
        
        try:
            tit = r'$\sigma $ = {0} [ns] amp = {1} [V]'.format(self.pulse_sig(mem_mode,Displacement_pulse),np.round(self.pulse_amp(mem_mode, Displacement_pulse),3))
            try:
                if is_two_mode_disp:  tit = '\n' + r'$\sigma $ = {0} [ns] amp = {1} [V]'.format(self.pulse_sig(scnd,Displacement_pulse),np.round(self.pulse_amp(scnd, Displacement_pulse),3))
            except:
                a =2 
            plt.suptitle(tit)        
        except:
            a =1 
        return fig_num,fig
    def plot_sigZ(self, params,Displacement_pulse= None, mem_mode = None,xlabel='', prog_name='',fig_num= None,results=None, ti=None,fit_types =[None,None,None,None],**kwargs):
        try:
            if mem_mode is None: mem_mode  = params["mem_mode"]
            if Displacement_pulse is None: Displacement_pulse = params["Displacement_pulse"]
        except: 
            mem_mode = self.main_mem
        
        
        prog_name = prog_name+f'Sz of {mem_mode}'
        if fig_num is None: fig_num = prog_name + ' ' + str(self.next_fig_num_by_prog_name(prog_name))
        
        fig = plt.figure(fig_num)
        ax1 = plt.subplot('211')
        ax1.set_title(r'$\Sigma_z$ ')

        ax2 = plt.subplot('212')
        ax2.set_title(r'$\phi$')
        
        ax3= plt.subplot('313')
        ax3.set_title('Radius')
        ax3.set_xlabel(xlabel)

        try:
            tit = r'$\sigma $ = {0} [ns] amp = {1} [V]'.format(self.pulse_sig(mem_mode,Displacement_pulse),np.round(self.pulse_amp(mem_mode, Displacement_pulse),3))
            if is_two_mode_disp:  tit = '\n' + r'$\sigma $ = {0} [ns] amp = {1} [V]'.format(self.pulse_sig(scnd,Displacement_pulse),np.round(self.pulse_amp(scnd, Displacement_pulse),3))
            plt.suptitle(tit)
        except:
            a =1 
        return fig_num
    def fitNplot_phase(self,results,params,ti= None, xlabel    = r'$amplitude$',x_type    = 'amp',fig_num= None,plot_normaliztion= False,
                       fit_types = [None,None,None,None],is_autoscale_data_list = None,**kwargs):
        
        if ti is None: ti = results['Re']["var"]
        
        axes = [None]*4;x_text = [None]*4;y_text = [None]*4
        fit = [None]*4;fit_fidelty = [None]*4
                
        fig_num,fig      = self.plot_phase(params,xlabel=xlabel,**kwargs)
        # complex_data = self.process_data(results['Re'],is_mean = True,is_calc_stat_error = False) +1j*self.process_data(results['Im'],is_mean = True,is_calc_stat_error = False)
        complex_data = self.process_data(results['Re'])[0] +1j*self.process_data(results['Im'])[0]#,self.process_data(results['Re'])[1] +1j*self.process_data(results['Im'])[1]]
        ABS = np.abs(complex_data)
        Angle =360*np.angle(complex_data)/2/np.pi
        if self.is_calc_stat_error: 
            ABS = [ABS, np.sqrt((self.process_data(results['Re'])[1])**2+(self.process_data(results['Im'])[1])**2)]

        DATA = [self.process_data(results['Re']),self.process_data(results['Im']),Angle,ABS ]
        Title = ['Real',"Imag",'Phase','Radius']
        if is_autoscale_data_list is None:
            is_autoscale_data_list = [self.is_autoscale_data,self.is_autoscale_data,False,self.is_autoscale_data]
        is_calc_stat_error= [True,True,False,True]
        for i,fityp in enumerate(fit_types): 
            if not fityp is None:
                axes[i],fit[i],fit_fidelty[i],x_text[i],y_text[i] = self.fit_and_plot( fityp , DATA[i]  , ti = ti  , title_str = Title[i],label = 'Fitted data',is_calc_stat_error=is_calc_stat_error[i], fig_num = fig_num,x_type = x_type,subplot = f'22{i+1}',is_autoscale_data=is_autoscale_data_list[i], **kwargs ) #
            else:
                axes[i],x_text[i],y_text[i] = self.fit_and_plot( fityp , DATA[i]  , ti = ti  , title_str = Title[i],label = 'Fitted data',is_calc_stat_error=is_calc_stat_error[i], fig_num = fig_num,x_type = x_type,subplot = f'22{i+1}',is_autoscale_data=is_autoscale_data_list[i], **kwargs ) #

            if plot_normaliztion:#TODO change the self.....results to approprite for each of the phases
                warn('plot normalization is holder')
                self.plot_normaliztion()
        plt.tight_layout()
        if 'Z' in results.keys(): 
            is_calc_stat_error= [False,False,False]
            comp_Z_data =self.process_data(results['Z'])[0]+ 1j*np.abs(complex_data)
            DATA = [self.process_data(results['Z'])[0],360*np.angle(comp_Z_data)/2/np.pi,np.abs(comp_Z_data) ]
            fig_numZ=self.plot_sigZ(params,xlabel=xlabel,**kwargs)
            for i,fityp in enumerate([None,None,None]): 
                if not fityp is None:
                    axes[i],fit[i],fit_fidelty[i],x_text[i],y_text[i] = self.fit_and_plot( fityp , DATA[i]  , ti = ti  , title_str = Title[i],label = 'Fitted data',is_calc_stat_error=is_calc_stat_error, fig_num = fig_numZ,x_type = x_type,subplot = f'31{i+1}', **kwargs ) #
                else:
                    axes[i],x_text[i],y_text[i] = self.fit_and_plot( fityp , DATA[i]  , ti = ti  , title_str = '',label = 'Fitted data',is_calc_stat_error=is_calc_stat_error[i], fig_num = fig_numZ,x_type = x_type,subplot = f'31{i+1}', **kwargs ) #
        return fig_num,fit,fit_fidelty,fig
   
    def plot_continuous_tomography(self,grid_X,grid_Y,Data,title_str='',fig_num= None,**kwargs):
        i = 0
        j = 0
        while grid_Y.size < grid_X.size:
            i=i+1
            grid_Y_temp =grid_Y
            grid_Y = np.zeros(grid_Y_temp.size+2)
            grid_Y[1:-1] = grid_Y_temp
            grid_Y[0] = grid_Y_temp[0]-np.diff(grid_Y_temp)[0]
            grid_Y[-1] = grid_Y_temp[-1]+np.diff(grid_Y_temp)[-1]
        while grid_X.size < grid_Y.size:
            j=j+1
            grid_X_temp  =grid_X
            grid_X = np.zeros(grid_X_temp.size+2)
            grid_X[1:-1] = grid_X_temp
            grid_X[0] = grid_X_temp[0]-np.diff(grid_X_temp)[0]
            grid_Y[-1] = grid_X_temp[-1]+np.diff(grid_X_temp)[-1]
        
        Data = np.pad(Data,[(j,j),(i,i)])                
        grid_X_extend = np.zeros(grid_X.size+2)
        # print(grid_X)
        grid_X_extend[1:-1] = grid_X # fill up with original values
        grid_X_extend[0] = grid_X[0]-np.diff(grid_X)[0]
        grid_X_extend[-1] = grid_X[-1]+np.diff(grid_X)[-1]
        grid_X_midpoints = grid_X_extend[:-1]+0.5*(np.diff(grid_X_extend))
        
        grid_Y_extend = np.zeros(grid_Y.size+2)
        grid_Y_extend[1:-1] = grid_Y
        grid_Y_extend[0] = grid_Y[0]-np.diff(grid_Y)[0]
        grid_Y_extend[-1] = grid_Y[-1]+np.diff(grid_Y)[-1]
        grid_Y_midpoints = grid_Y_extend[:-1]+0.5*(np.diff(grid_Y_extend))
        
        
        fig,ax = plt.subplots(num=fig_num)
        plt.title(title_str)
        im = ax.pcolormesh(grid_X_midpoints,grid_Y_midpoints, Data, cmap = 'seismic',vmax = 1,vmin = -1)
        # im = ax.pcolormesh(grid_X_midpoints,grid_Y_midpoints, Data, cmap = 'RPGn')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.07)
        plt.colorbar(im, cax=cax)
        return fig,ax
    #%% Ramsey for displacment calibration
    def load_cavity_ramsey_vaccum(self, num_of_pnts = 201, # How many points to measure, .
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    scale_amp_list = None,
                                    Displacement_pulse = None,
                                    mem_mode=None,
                                    scnd_mem = None,
                                    start_with: tuple = None, #(self.pi_pulse,self.main_qubits)
                                    update_prog = True,
                                    pi2_phase = 0, #phase of second pi2 pulse in units of pi 
                                    normalize = False,
                                    qb_phase_correction = None,
                                    is_Yale= False,
                                    _is_cat_and_back=False,
                                    is_remove = False,
                                    is_phase = False, #are measuring the phase of the qubit b
                                    is_measure_Z = False,#only relevent when _is_calibrate_SigZ is True
                                    _is_calibrate_SigZ=False,
                                    _is_sweep_qubit_phase = False, #when true scale_amp_list is used as a qubit phase list
                                    disp_amp = 1.0, #only used when we are not sweeping on the amplitude size 
                                    detuning = 0, 
                                    **kwargs):
        
        if qb_phase_correction is None: qb_phase_correction = self.ConDisp_correction
        if mem_mode is None: mem_mode =self.main_mem
        if scnd_mem is None: scnd_mem =self.sec_mem
        
        if Displacement_pulse is None: Displacement_pulse = self.ConDisp1_pulse

        pi2_phase_list,qb_pulse_list =self.get_qb_measure_seq_list(pi2_phase=pi2_phase,is_remove= is_remove, is_phase = is_phase, is_measure_Z=is_measure_Z,normalize=normalize)
            
        if scale_amp_list is None: 
            if not (num_of_pnts%2): num_of_pnts =num_of_pnts+1
            scale_amp_list = np.linspace(-1.9,1.9,num_of_pnts,dtype=float)
      
        if type(scale_amp_list) == list: scale_amp_list= np.array(scale_amp_list)
        # rot_phase_list =  (scale_amp_list**2)*detuning/2/np.pi + pi2_phase
        rot_phase_list =  (scale_amp_list**2)*detuning/2/np.pi 
        
        if  _is_sweep_qubit_phase:
            rot_phase_list = scale_amp_list
            scale_amp_list = disp_amp*np.ones(len(rot_phase_list))
            
        
        qb_phase_corr_list =scale_amp_list**2*qb_phase_correction/2/np.pi
                
        rot_phase_list     = rot_phase_list.tolist()
        scale_amp_list     = scale_amp_list.tolist()
        qb_phase_corr_list = qb_phase_corr_list.tolist()
        num_of_pnts = len(scale_amp_list)
        # ramsey_vaccum_N_avg = N_avg
       
        #Save parameters:
        cavity_ramsey_vaccum_params = {}
        cavity_ramsey_vaccum_params["N_avg"]              = N_avg
        cavity_ramsey_vaccum_params["num_of_pnts"]        = num_of_pnts
        cavity_ramsey_vaccum_params["Displacement_pulse"] = Displacement_pulse
        cavity_ramsey_vaccum_params["mem_mode"]           = mem_mode  
        cavity_ramsey_vaccum_params["scnd_mem"]           = scnd_mem  
        cavity_ramsey_vaccum_params["start_with"]         = start_with
        cavity_ramsey_vaccum_params["scale_amp_list"]     = scale_amp_list
        cavity_ramsey_vaccum_params["is_Yale"]            = is_Yale
        cavity_ramsey_vaccum_params["rot_phase_list"]     = rot_phase_list
        cavity_ramsey_vaccum_params["detuning"]           = detuning
        cavity_ramsey_vaccum_params["qb_phase_correction"]= qb_phase_correction
        cavity_ramsey_vaccum_params["pi2_phase_list"]     = pi2_phase_list
        
        # cavity_ramsey_vaccum_params["_is_cat_and_back"]    = _is_cat_and_back
        
        # cavity_ramsey_vaccum_params["step_size_clks"]     = step_size_clks
        if update_prog: self.cavity_ramsey_vaccum_params=cavity_ramsey_vaccum_params

        assert  (not self.pulse_len(self.main_qubit,self.pi_pulse)%4), 'pi pulse length does not divide by 4'
        pi_wait_time = self.pulse_len(self.main_qubit,self.pi_pulse)//4
        #Calculate and show expeced runtime:
        try:
            run_time = len(pi2_phase_list)*N_avg*num_of_pnts*(self.pulse_len(self.main_readout,self.ro_pulse)+self.wait_between_seq * 4 +self.pulse_len(mem_mode,Displacement_pulse))
            print('Run time is {}s'.format(round(run_time * 1e-9)))         
        except:
            print('cant calculate time')
        print(pi2_phase_list)
        #Create program:
        with program() as prog:
            n             = declare(int)
            i             = declare(int)
            j             = declare(int)
            I             = declare(fixed)
            Q             = declare(fixed)
            I_st          = declare_stream()
            Q_st          = declare_stream()
            pi2_phase_arr = declare(fixed,value = pi2_phase_list)
            qb_pulse_arr  = declare(int,value =qb_pulse_list)
            
            rot_phase     = declare(fixed)
            rot_phase_arr = declare(fixed,value = rot_phase_list)
            qb_phasCor_arr= declare(fixed,value = qb_phase_corr_list)
            scale_amp_arr = declare(fixed,value =scale_amp_list)
            scale_amp     = declare(fixed)
            qb_pulse      = declare(int)
            
            detuning      = declare(fixed)
            qb_phase_corr = declare(fixed)
            # qb_amp        = declare(fixed)
            # qb_amp_arr    = declare(fixed,value = pi_amp_mul_list)
            with for_(n,0,n<N_avg,n+1):
                with for_(i,0,i<len(scale_amp_list),i+1):
                    assign(scale_amp,scale_amp_arr[i])
                    assign(detuning,rot_phase_arr[i])
                    assign(qb_phase_corr,qb_phasCor_arr[i])
                    with for_(j,0,j<pi2_phase_arr.length(),j+1):
                        assign(qb_pulse,qb_pulse_arr[j])
                        # assign(qb_amp, qb_amp_arr[j])
                        assign(rot_phase, detuning+pi2_phase_arr[j])
                            
                        reset_phase(self.main_qubit)
                        reset_phase(mem_mode)
                        if self.is_two_mode_disp or self.is_two_mode_disp=='Sim' or self.is_two_mode_disp=='Sam': 
                            reset_phase(scnd_mem)
                            reset_frame(mem_mode,scnd_mem,self.main_qubit)
                        else:
                            reset_frame(mem_mode,self.main_qubit)
                            
                        
                        
                        if type(start_with) is tuple: 
                            play(*start_with)
                            align(start_with[1],self.main_qubit)
                        elif type(start_with) is types.FunctionType or type(start_with) is types.MethodType:
                            start_with(mem_mode = mem_mode,scale_amp = scale_amp,**kwargs)
                        else:
                            print('\n \n ******** \n no starting sequence \n ******** \n \n')
                        
                                        
                        play(self.pi2_pulse,self.main_qubit)
                            
                        # align(self.main_qubit, mem_mode)
                        
                        self._ConDisp_pulse(qubit = self.main_qubit, mem_mode = mem_mode,mem_mode2= scnd_mem, scale_amp =scale_amp,Displacement_pulse = Displacement_pulse,is_Yale = is_Yale,qb_phase_correction =qb_phase_corr,**kwargs) 
                        # self._ConDisp_pulse(qubit = self.main_qubit, mem_mode = mem_mode,mem_mode2= scnd_mem, scale_amp =scale_amp,Displacement_pulse = Displacement_pulse,is_Yale = is_Yale,qb_phase_correction =qb_phase_corr,is_two_mode_disp = 'Sim',**kwargs) 
                        #not for jiwon  start
                        if _is_calibrate_SigZ:
                            # wait(self.pulse_sig(mem_mode,Displacement_pulse),mem_mode)
                            self._ConDisp_pulse(qubit = self.main_qubit, mem_mode = mem_mode, mem_mode2= scnd_mem,scale_amp =-scale_amp,Displacement_pulse = Displacement_pulse,is_Yale = is_Yale,qb_phase_correction =qb_phase_corr,**kwargs) 
                            # if detuning:
                        elif _is_cat_and_back:
                            align(self.main_qubit, mem_mode)
                            play(self.pi_pulse,self.main_qubit)
                            
                            if is_Yale:
                                
                                if 'scale_amp2' in kwargs.keys():
                                    if not kwargs['scale_amp2'] is None:
                                        if type(kwargs['scale_amp2']) is list: 
                                            kwargs['scale_amp2'] =[-kwargs['scale_amp2'][0],-kwargs['scale_amp2'][1],-kwargs['scale_amp2'][2],-kwargs['scale_amp2'][3]]
                                        else:
                                            kwargs['scale_amp2'] = -kwargs['scale_amp2']
                                                                      
                                if self.is_fixCD_cPhase:
                                    frame_rotation_2pi(-np.round(self.pulse_len(self.main_qubit,self.pi_pulse)/2*self.mem_chi[mem_mode],6),mem_mode)
                                    # try: 
                                    frame_rotation_2pi(-np.round(self.pulse_len(self.main_qubit,self.pi_pulse)/2*self.mem_chi[scnd_mem],6),scnd_mem)
                                    # except:
                                        # print ('fail')
                                align(self.main_qubit, mem_mode)
                                self._ConDisp_pulse(qubit = self.main_qubit, mem_mode = mem_mode,mem_mode2= scnd_mem, scale_amp =-scale_amp,Displacement_pulse = Displacement_pulse,is_Yale = is_Yale,qb_phase_correction =qb_phase_corr,**kwargs) 
                                # self._ConDisp_pulse(qubit = self.main_qubit, mem_mode =scnd_mem ,mem_mode2= mem_mode, scale_amp =-scale_amp,Displacement_pulse = Displacement_pulse,is_Yale = is_Yale,qb_phase_correction =qb_phase_corr,**kwargs) 
                                # warn('Asaf fix me to not prioritizing ')
                            else:
                                align(self.main_qubit, mem_mode)
                                self._ConDisp_pulse(qubit = self.main_qubit, mem_mode = mem_mode, mem_mode2= scnd_mem,scale_amp =-scale_amp,Displacement_pulse = Displacement_pulse,is_Yale = is_Yale,qb_phase_correction =qb_phase_corr,**kwargs) 
                                if 1:
                                    align(self.main_qubit, mem_mode)
                                    play(self.pi_pulse,self.main_qubit)
                                    align(self.main_qubit, mem_mode)
                                    self._ConDisp_pulse(qubit = self.main_qubit, mem_mode = mem_mode,mem_mode2= scnd_mem, scale_amp =-scale_amp,Displacement_pulse = Displacement_pulse,is_Yale = is_Yale,qb_phase_correction =qb_phase_corr,**kwargs) 
                                    align(self.main_qubit, mem_mode)
                                    play(self.pi_pulse,self.main_qubit)
                                    align(self.main_qubit, mem_mode)
                                    self._ConDisp_pulse(qubit = self.main_qubit, mem_mode = mem_mode, mem_mode2= scnd_mem,scale_amp =scale_amp,Displacement_pulse = Displacement_pulse,is_Yale = is_Yale,qb_phase_correction =qb_phase_corr,**kwargs) 
                      #not for  jiwon end  
                        frame_rotation_2pi(rot_phase,self.main_qubit)
                        
                        align(self.main_qubit, mem_mode)
                        # play(self.pi2_pulse,self.main_qubit)
                        # frame_rotation_2pi(pi2_phase,self.main_qubit)
                        with switch_(qb_pulse,unsafe = True):
                            with case_(1): 
                                play(self.pi2_pulse,self.main_qubit)
                            with case_(2): 
                                play(self.pi_pulse,self.main_qubit)
                            with case_(0): 
                                wait(pi_wait_time,self.main_qubit)
                        align(self.main_qubit, self.main_readout)
                        # self.perform_full_measurement(I,Q)#,I_output_name = I_st,Q_output_name = Q_st)
                        self.perform_full_measurement(I,Q,I_output_name = I_st,Q_output_name = Q_st)
                        
                        save(scale_amp, 'scale_amp')
                
            with stream_processing():
                I_st.save_all('I')
                Q_st.save_all('Q')
                # I_st.buffer(len(scale_amp_list),len(qb_pulse_list)).save_all('I')
                # Q_st.buffer(len(scale_amp_list),len(qb_pulse_list)).save_all('Q')
                # I_st.auto_reshape().save_all('I')
                
        if update_prog: self.cavity_ramsey_vaccum_prog = prog
        # self.simulate_prog(prog)
        return prog,cavity_ramsey_vaccum_params

    def run_cavity_ramsey_vaccum(self,is_phase= True,#normalize = True,is_remove= False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):
        
        self.cavity_ramsey_vaccum_results,_ , self.ramsey_vaccum_plot_normaliztion=self.run_for_phase(self.load_cavity_ramsey_vaccum,is_phase=is_phase,var_name ='scale_amp',**kwargs)
        
        if is_phase: return self.plot_cavity_ramsey_vaccum_phase(**kwargs)
        return self.plot_cavity_ramsey_vaccum(**kwargs)
      


    
    def plot_cavity_ramsey_vaccum(self,start_x_fit=None,plot_normaliztion=None,txt= '',fig_num = None,**kwargs):
        plot_normaliztion = self.ramsey_vaccum_plot_normaliztion if plot_normaliztion is None else plot_normaliztion
        # Check what's happening here:
        #           fit_and_plot(self,fittype, I, Q, ti, N_avg,  txt=None,title_str =None, fig_num=None ):  
        prog_name = 'ramsey_vaccum EM'
        if fig_num is None: fig_num = prog_name + ' ' + str(self.next_fig_num_by_prog_name(prog_name))
        
        fig, fit,fit_fidelty, x_text,y_text = self.fit_and_plot( 'Gaussian' , self.process_data(self.cavity_ramsey_vaccum_results), ti = self.cavity_ramsey_vaccum_results["var"]  , title_str = f"ramsey_vaccum EM ({self.main_mem})",label = 'Fitted data', fig_num = fig_num,x_type = 'amp',txt = r'$| \alpha |$ = {} $\pm$ {}', **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        plt.xlabel('amplitude scaling')
        # ret  p=lt.plot( self.autoscale_data(self.process_data(self.cavity_ramsey_vaccum_results))#, ti = self.cavity_ramsey_vaccum_results["amp"]  , title_str = f"ramsey_vaccum EM ({self.main_mem})",label = 'Fitted data', fig_num = fig_num, **kwargs ) #            max_seq_clks = int(max_seq_time // 4)\
        if plot_normaliztion:
            try:
                plt.plot(self.autoscale_data(self.cavity_ramsey_vaccum_results["var"])[0], self.autoscale_data(self.process_data(data = [self.cavity_ramsey_vaccum_results["I_orig"], self.cavity_ramsey_vaccum_results["Q_orig"]])[0])[0],'+',label = 'Measured Data')
                plt.plot(self.autoscale_data(self.cavity_ramsey_vaccum_results["var"])[0], self.autoscale_data(self.process_data(data = [self.cavity_ramsey_vaccum_results["I_CrossKerr"], self.cavity_ramsey_vaccum_results["Q_CrossKerr"]], is_calc_stat_error = False, is_thresh = False))[0],'o', label = 'Cross Kerr')
                leg = plt.legend()
                leg.set_draggable(True)
            except:
                a=1
             
        plt.title(r'$\sigma $ = {0} amp = {1}'.format(self.pulse_sig(self.cavity_ramsey_vaccum_params["mem_mode"] ,self.cavity_ramsey_vaccum_params["Displacement_pulse"] ),self.pulse_amp(self.cavity_ramsey_vaccum_params["mem_mode"] , self.cavity_ramsey_vaccum_params["Displacement_pulse"] )))
        # #fig.show()
        plt.tight_layout()
        return fig,fit,fit_fidelty
        
    def plot_cavity_ramsey_vaccum_phase(self,results =None,params = None,fig_num= None,plot_normaliztion= None
                                        ,**kwargs):
        return self.fitNplot_phase(self.cavity_ramsey_vaccum_results,
                                   self.cavity_ramsey_vaccum_params,ti= self.cavity_ramsey_vaccum_results['Re']["var"], 
                                   xlabel    = r'$amplitude$',x_type    = 'amp',
                                   prog_name = f'ramsey vaccum phase',
                                   plot_normaliztion = self.ramsey_vaccum_plot_normaliztion,
                                   fit_types =  ['Gaussian',None,None,'Gaussian'],**kwargs)

        # plt.suptitle(#f"C&B {mem_mode} \n"+

#%% prograrms for calibarating sigma z rotation due to condtional displacment based on cavity_ramsey_vaccum
    def load_calibrate_sigma_z(self,scale_amp_list = None, num_of_pnts = 51,update_prog = True,is_Yale = True,detuning = 0.0,**kwargs):
        if scale_amp_list is None:
            if not (num_of_pnts%2): num_of_pnts =num_of_pnts+1
            scale_amp_list = np.array([*np.linspace(-3.61,3.61,num_of_pnts)])
            scale_amp_list = np.sign(scale_amp_list)*np.sqrt(np.abs(scale_amp_list))

        prog,params = self.load_cavity_ramsey_vaccum(scale_amp_list=scale_amp_list,_is_calibrate_SigZ=True,is_Yale = False,update_prog = False,detuning = detuning,**kwargs)
        
        if update_prog: 
            self.calibrate_sigma_z_prog   = prog
            self.calibrate_sigma_z_params = params
            self.calibrate_sigma_z_params['detuning'] = detuning
            
        return prog, params

    def run_calibrate_sigma_z(self,is_phase=False,#normalize = True,is_remove= False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):
        if is_phase: 
            self.calibrate_sigma_z_phase_results, _,self.calibrate_sigma_z_plot_normaliztion=self.run_for_phase(self.load_calibrate_sigma_z,var_name ='scale_amp',is_phase=is_phase,**kwargs)
            return self.plot_calibrate_sigma_z_phase(**kwargs)
        self.calibrate_sigma_z_results, _,self.calibrate_sigma_z_plot_normaliztion=self.run_for_phase(self.load_calibrate_sigma_z,var_name ='scale_amp',**kwargs)
        return self.plot_calibrate_sigma_z(**kwargs)


    
    def plot_calibrate_sigma_z(self,start_x_fit=None,plot_normaliztion=None,txt= '',omega=None,omega_prime= 0,**kwargs):
        plot_normaliztion = self.calibrate_sigma_z_plot_normaliztion if plot_normaliztion is None else plot_normaliztion
        # Check what's happening here:
        #           fit_and_plot(self,fittype, I, Q, ti, N_avg,  txt=None,title_str =None, fig_num=None ):  
        prog_name = 'calibrate_sigma_z EM'
        fig_num = prog_name + ' ' + str(self.next_fig_num_by_prog_name(prog_name))
        try:
            squared_amp = np.sign(self.calibrate_sigma_z_results["amp"])*abs(self.calibrate_sigma_z_results["amp"])**2
        except:
            squared_amp = np.sign(self.calibrate_sigma_z_results['var'])*abs(self.calibrate_sigma_z_results['var'])**2
        fig,fit,fit_fidelty,x_text,y_text = self.fit_and_plot( 'Cos' , self.process_data(self.calibrate_sigma_z_results), ti = squared_amp  , title_str = f"calibrate_sigma_z EM ({self.main_mem})",label = 'Fitted data', fig_num = 'fake_'+fig_num,x_type = 'amp', **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        print(fit)
        Data = self.process_data(self.calibrate_sigma_z_results,is_thresh = False)
        max_Data = np.max(Data[0])
        min_Data = np.min(Data[0])
        if omega is None: omega = fit[1]*2*np.pi
        guess =[(max_Data-min_Data)/2,omega,omega_prime,(max_Data+min_Data)/2]
        try: 
            is_succeed = True
            fit,cov= curve_fit(self.cos_xi_prime, xdata = squared_amp, ydata = self.process_data(self.calibrate_sigma_z_results)[0], p0 = guess)
            fit_fidelty = np.sqrt(np.diag(cov))
        except: 
            is_succeed = False
            print('***************************\n Could not find fit \n***************************')
        if not fit is None:
            txt = txt + r'$\omega_\phi$ =  {0} $\pm$ {1} $[\pi /amp^2]$'.format(*round_value_by_error(fit[1],fit_fidelty[1]))
            if is_succeed:
                txt = txt +'\n'+ r'$\omega_\phi $ prime =  {0} $\pm$ {1} $[\pi /amp^2]$'.format(*round_value_by_error(fit[2],fit_fidelty[2]))
                # fig.close()
                fig1= self.fit_and_plot( None, self.process_data(self.calibrate_sigma_z_results), ti = squared_amp  , title_str = f"calibrate_sigma_z EM ({self.main_mem})",label = 'Fitted data', fig_num = fig_num,x_type = 'amp', **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
                y_fit = self.autoscale_data(self.cos_xi_prime(squared_amp,*fit))[0]
                # x_fit = self.autoscale_data(self.cavity_charactaristic_function_decay_results["t"]*1e-9)[0]                         
                x_fit = self.autoscale_data(squared_amp  )[0]                  
                plt.plot(x_fit,y_fit)
                    
       
        txt = txt + '\n detuning = {0}'.format(self.calibrate_sigma_z_params['detuning'])
            
        plt.xlabel(r'$amplitude^2$')
        # ret  p=lt.plot( self.autoscale_data(self.process_data(self.calibrate_sigma_z_results))#, ti = self.calibrate_sigma_z_results["amp"]  , title_str = f"calibrate_sigma_z EM ({self.main_mem})",label = 'Fitted data', fig_num = fig_num, **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        if plot_normaliztion:
            plt.plot(self.autoscale_data(squared_amp)[0], self.autoscale_data(self.process_data(data = [self.calibrate_sigma_z_results["I_orig"], self.calibrate_sigma_z_results["Q_orig"]])[0])[0],'+',label = 'Measured Data')
            plt.plot(self.autoscale_data(squared_amp)[0], self.autoscale_data(self.process_data(data = [self.calibrate_sigma_z_results["I_CrossKerr"], self.calibrate_sigma_z_results["Q_CrossKerr"]], is_calc_stat_error = False, is_thresh = False))[0],'o', label = 'Cross Kerr')
            leg = plt.legend()
            leg.set_draggable(True)
      
            # ann = plt.annotate( r'$\phi$/amp =  {0} $\pm$ {1} [us]'.format(*round_value_by_error(fit[2],fit_fidelty[2]), xy = (x_text,y_text), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top')) 
        if not txt =='':
            # ann = plt.annotate(txt, xy = (self.autoscale_data(self.calibrate_sigma_z_results["amp"])[0][0]/1.5,y_text/1.25), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
            ann = plt.annotate(txt, xy = (0,0), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
            ann.draggable()
        plt.title(r'$\sigma $ = {0} amp = {1}'.format(self.pulse_sig(self.main_mem,self.ConDisp1_pulse),self.pulse_amp(self.main_mem, self.ConDisp1_pulse)))
         
        #fig.show()
        plt.tight_layout()
        return fit,fit_fidelty
    def cos_xi_prime(self,x,A,chi,chi_prime,C):
        return A*np.cos(chi*x + chi_prime*(x**2))+C
    # -*- coding: utf-8 -*-
    def plot_calibrate_sigma_z_phase(self,results =None,params = None,fig_num= None,plot_normaliztion= None
                                        ,**kwargs):
        
        squared_amp = np.sign( self.calibrate_sigma_z_phase_results['Re']["var"])*abs(self.calibrate_sigma_z_phase_results['Re']["var"])**2

        return self.fitNplot_phase(self.calibrate_sigma_z_phase_results,
                                   self.calibrate_sigma_z_params,ti=squared_amp, 
                                   xlabel    = r'$amplitude^2$',x_type    = 'amp',
                                   prog_name = f'rcalibrate sigma_z phase',
                                   plot_normaliztion = self.calibrate_sigma_z_plot_normaliztion,
                                   fit_types =  ['Cos',None,None,None],**kwargs)

        # plt.suptitle(#f"C&B {mem_mode} \n"+
#%% cat and back

    def load_cat_and_back(self,update_prog = True,_is_cat_and_back= True, num_of_pnts=51,scale_amp_list= None,**kwargs):
        if scale_amp_list is None:
            if not (num_of_pnts%2): num_of_pnts =num_of_pnts+1
            scale_amp_list = np.array([*np.linspace(-3.61,3.61,num_of_pnts)])
            scale_amp_list = np.sign(scale_amp_list)*np.sqrt(np.abs(scale_amp_list))
            
        prog,params = self.load_cavity_ramsey_vaccum(_is_cat_and_back =True,update_prog = False,scale_amp_list=scale_amp_list,**kwargs)
        
        if update_prog: 
            self.cat_and_back_prog   = prog
            self.cat_and_back_params = params
            
        return prog, params

    def run_cat_and_back(self,is_phase = False,# normalize = True,is_remove= False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):
        self.cat_and_back_results,_ , self.cat_and_back_plot_normaliztion=self.run_for_phase(self.load_cat_and_back,var_name ='scale_amp',is_phase=is_phase,**kwargs)
        
        if is_phase: return self.plot_cat_and_back_phase(**kwargs)
        return self.plot_cat_and_back(**kwargs)
   
    
    
    def plot_cat_and_back(self,start_x_fit=None,results= None,plot_normaliztion=None,txt= '',omega=None,omega_prime= 0,**kwargs):

        if results is None: results= self.cat_and_back_results 

        plot_normaliztion = self.cat_and_back_plot_normaliztion if plot_normaliztion is None else plot_normaliztion
        # Check what's happening here:
        #           fit_and_plot(self,fittype, I, Q, ti, N_avg,  txt=None,title_str =None, fig_num=None ):  
        
        squared_amp = np.sign(results['var'])*abs(results['var'])**2
        # squared_amp = np.sign(results['var'])*abs(results['var'])
        prog_name = 'cat_and_back {}'.format(self.cat_and_back_params['mem_mode'])
        fig_num = prog_name + ' ' + str(self.next_fig_num_by_prog_name(prog_name))
        fig,fit,fit_fidelty,x_text,y_text = self.fit_and_plot( 'Cos' , self.process_data(results ), ti = squared_amp  , title_str = f"cat_and_back EM ({self.main_mem})",label = 'Fitted data', fig_num = fig_num,x_type = 'amp', **kwargs ) #            max_seq_clks = int(max_seq_time // 4)

        # txt = txt + '\n detuning = {0}'.format(self.cat_and_back_params['detuning'])
            
        plt.xlabel(r'$amplitude^2$')
        # ret  p=lt.plot( self.autoscale_data(self.process_data(results ))#, ti = results ["amp"]  , title_str = f"cat_and_back EM ({self.main_mem})",label = 'Fitted data', fig_num = fig_num, **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        if plot_normaliztion:
            plt.plot(self.autoscale_data(squared_amp)[0], self.autoscale_data(self.process_data(data = [results ["I_orig"], results ["Q_orig"]])[0])[0],'+',label = 'Measured Data')
            plt.plot(self.autoscale_data(squared_amp)[0], self.autoscale_data(self.process_data(data = [results ["I_CrossKerr"], results ["Q_CrossKerr"]], is_calc_stat_error = False, is_thresh = False))[0],'o', label = 'Cross Kerr')
            leg = plt.legend()
            leg.set_draggable(True)
      
        #     # ann = plt.annotate( r'$\phi$/amp =  {0} $\pm$ {1} [us]'.format(*round_value_by_error(fit[2],fit_fidelty[2]), xy = (x_text,y_text), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top')) 
        # if not txt =='':
        #     # ann = plt.annotate(txt, xy = (self.autoscale_data(results ["amp"])[0][0]/1.5,y_text/1.25), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
        #     ann = plt.annotate(txt, xy = (0,0), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
        #     ann.draggable()
        # plt.title(r'$\sigma $ = {0} amp = {1}'.format(self.pulse_sig(self.main_mem,self.ConDisp1_pulse),self.pulse_amp(self.main_mem, self.ConDisp1_pulse)))
         
        # #fig.show()
        # plt.tight_layout()
        # return fit,fit_fidelty

    def plot_cat_and_back_phase(self,results =None,fig_num= None,plot_normaliztion= None,is_amp_ploted_squared = True, **kwargs):
        
        squared_amp = np.sign( self.cat_and_back_results['Re']["var"])*abs(self.cat_and_back_results['Re']["var"])
        if is_amp_ploted_squared:   squared_amp = np.sign( self.cat_and_back_results['Re']["var"])*abs(self.cat_and_back_results['Re']["var"])**2
        if 1:
            return self.fitNplot_phase( self.cat_and_back_results,
                           self.cat_and_back_params,ti= squared_amp, 
                           xlabel    = r'$amplitude^2$',x_type    = 'amp',
                           prog_name = f'C&B phase',
                           plot_normaliztion =self.cat_and_back_plot_normaliztion,
                           fit_types =  ['Cos','Cos',None,None],**kwargs)

        mem_mode           = self.cat_and_back_params["mem_mode"]
        Displacement_pulse = self.cat_and_back_params["Displacement_pulse"]
        
        if results is None: results = self.cat_and_back_results
        if plot_normaliztion is None: plot_normaliztion = self.cat_and_back_plot_normaliztion  
        
        x_label =r'$amplitude^2$' if is_amp_ploted_squared else r'$amplitude$'
        
        self.plot_phase(self.cat_and_back_params,xlabel=x_label,prog_name= f'C&B phase {mem_mode}')
        
        squared_amp = np.sign(results['Re']["var"])*abs(results['Re']["var"])**2
        
        complex_data = self.process_data(results['Re'])[0] +1j*self.process_data(results['Im'])[0]
        
        ax1,fit2,fit_fidelty2,  x_text1,y_text1 = self.fit_and_plot( 'Cos' ,  self.process_data(results['Re']), ti = squared_amp  , title_str = "Real",label = 'Fitted data', fig_num = fig_num,x_type = 'amp',subplot = '221', **kwargs ) #            max_seq_clks = int(max_seq_time // 4)

        ax2, fit2,fit_fidelty2, x_text2,y_text2 = self.fit_and_plot( 'Cos' ,  self.process_data(results['Im']), ti = squared_amp  , title_str = "Imag",label = 'Fitted data', fig_num = fig_num,x_type = 'amp',subplot = '222', **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        
        ax3, x_text3,y_text3 = self.fit_and_plot( None ,np.angle(complex_data)  , ti = squared_amp  , title_str = "phase",label = 'Fitted data', fig_num = fig_num,x_type = 'amp',subplot = '223',is_calc_stat_error=False, **kwargs )
        
        ax4, x_text4,y_text4 = self.fit_and_plot( None ,np.abs(complex_data)  , ti = squared_amp  , title_str = "Radius",label = 'Fitted data', fig_num = fig_num,x_type = 'amp',subplot = '224',is_calc_stat_error=False, **kwargs )
    
        print(1111111111111)
        if plot_normaliztion:#TODO change the self.....results to approprite for each of the phases
            self.plot_normaliztion()
            # plt.plot(self.autoscale_data(squared_amp)[0], self.autoscale_data(self.process_data(data = [self.cavity_ramsey_vaccum_results["I_orig"], self.cavity_ramsey_vaccum_results["Q_orig"]])[0])[0],'+',label = 'Measured Data')
            # plt.plot(self.autoscale_data(squared_amp)[0], self.autoscale_data(self.process_data(data = [self.cavity_ramsey_vaccum_results["I_CrossKerr"], self.cavity_ramsey_vaccum_results["Q_CrossKerr"]], is_calc_stat_error = False, is_thresh = False))[0],'o', label = 'Cross Kerr')
            # leg = plt.legend()
            # leg.set_draggable(True)
        
        # plt.suptitle(#f"C&B {mem_mode} \n"+
                     # r'$\sigma $run_cat_and_back = {0} [ns] amp = {1} [V]'.format(self.pulse_sig(mem_mode,Displacement_pulse),self.pulse_amp(mem_mode, Displacement_pulse)))
#%% Disentagle cat
    def which_cat(self,cat_phase='+',**kwargs):
        if cat_phase == '+':
            print('+')
            print('+')
            play(self.pi2_pulse*amp(0,-1,1,0),self.main_qubit)  
        elif cat_phase == '-':
            print('-')
            print('-')
            play(self.pi2_pulse*amp(0,1,-1,0),self.main_qubit)   
            return [1]
        elif cat_phase == '+i':
            print('+i')
            print('+i')
            play(self.pi2_pulse,self.main_qubit)    
            return [1]
        elif cat_phase == '-i':
            print('-i')
            play(self.pi2_pulse*amp(-1),self.main_qubit)    
            print('-i')
            return [1]
        else:
            play(self.pi2_pulse*amp(np.cos(cat_phase),-np.sin(cat_phase),np.sin(cat_phase),np.cos(cat_phase)),self.main_qubit)
            return [1]
        return #add which phase to return according to which cat created, should be important for la     
            
    def cat_seq(self,cat_phase='+',mem_mode =None,first_pulse= 'Fast_ConDisp1', first_amp =1.0,
                second_pulse= 'VFast_ConDisp1',second_amp = None, qubit_phase = -0.0,**kwargs):
              
            if mem_mode is None: mem_mode = self.main_mem

            pi2_amp_list = self.which_cat(cat_phase = cat_phase)
            align(self.main_qubit,mem_mode)

            # self._ConDisp_pulse(qubit = self.main_qubit,mem_mode = mem_mode,scale_amp =[0,first_amp,-first_amp,0],Displacement_pulse = first_pulse,is_Yale = True )
            self._ConDisp_pulse(qubit = self.main_qubit,mem_mode = mem_mode,scale_amp =first_amp,Displacement_pulse = first_pulse,is_Yale = True )
            qubit_phase = qubit_phase#*2*np.pi
            if not second_amp is None:
                # print(qubit_phase)
                play(self.pi2_pulse,self.main_qubit)
                # align(self.main_qubit,mem_mode)
                if self.is_two_mode_disp or self.is_two_mode_disp == 'Sim':
                    warn('Disentagling pulse is simultanios')
                    self._ConDisp_pulse(qubit = self.main_qubit,mem_mode = mem_mode,scale_amp =second_amp,Displacement_pulse = second_pulse,is_two_mode_disp = 'Sim')
                else:    
                    self._ConDisp_pulse(qubit = self.main_qubit,mem_mode = mem_mode,scale_amp =second_amp,Displacement_pulse = second_pulse)
                # frame_rotation_2pi(qubit_phase,self.main_qubit)
                # align(self.main_mem,self.main_qubit)
                # play(self.pi2_pulse*amp(pi2_amp_list[0]),self.main_qubit)
                # play(self.pi2_pulse*amp(0,1,-1,0),self.main_qubit)
                play(self.pi2_pulse*amp(np.cos(np.pi/2+qubit_phase) ,np.sin(np.pi/2+qubit_phase),-np.sin(np.pi/2+qubit_phase),np.cos(np.pi/2+qubit_phase)),self.main_qubit)
                # play(self.pi2_pulse*amp(np.cos(np.pi/2) ,np.sin(np.pi/2),-np.sin(np.pi/2),np.cos(np.pi/2)),self.main_qubit)
                # play(self.pi2_pulse,self.main_qubit)
  
    def load_disentangle_cat(self,cat_phase = '+',update_frequency = None,
                             first_amp = 1.95,first_pulse = None,second_pulse = None,
                             update_prog = True, **kwargs):
        if first_pulse is None: first_pulse = self.Fast_ConDisp1
        if second_pulse is None: second_pulse = 'VFast_ConDisp1'
        
        prog,params = self.load_cavity_ramsey_vaccum(update_prog = False,start_with = self.cat_seq,Displacement_pulse=second_pulse,first_pulse = first_pulse,first_amp = first_amp,**kwargs)
        
        if update_prog: 
            params['first_pulse'] = first_pulse
            params['first_amp']   = first_amp
            params['cat_phase']   = cat_phase
            self.disentangle_cat_prog   = prog
            self.disentangle_cat_params = params
            
        return prog, params

    def run_disentangle_cat(self,is_phase = False,# normalize = True,is_remove= False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):
        self.disentangle_cat_results,_ , self.disentangle_cat_plot_normaliztion=self.run_for_phase(self.load_disentangle_cat,var_name ='scale_amp',is_phase=is_phase,**kwargs)
        
        if is_phase: return self.plot_disentangle_cat_phase(**kwargs)
        return self.plot_disentangle_cat(**kwargs)
  
    def plot_disentangle_cat_phase(self,results =None,params = None,fig_num= None,plot_normaliztion= None
                                        ,**kwargs):
        
        fig_num,fit,fit_fidelty,fig = self.fitNplot_phase(self.disentangle_cat_results,
                                   self.disentangle_cat_params,ti= self.disentangle_cat_results['Re']["var"], 
                                   xlabel    = r'$amplitude$',x_type    = 'amp',
                                   prog_name = f'Disentangle cat phase',
                                   plot_normaliztion = self.disentangle_cat_plot_normaliztion,
                                   fit_types =  ['Gaussian',None,None,'Gaussian'],**kwargs)
        first_pulse_attr=  [self.disentangle_cat_params['mem_mode'],self.disentangle_cat_params['first_pulse']]     
        try:
            ann = plt.annotate('first pulse \n'+r'$\sigma $ = {0}  amp = {1}'.format(self.pulse_sig(*first_pulse_attr),np.round(self.disentangle_cat_params['first_amp']*self.pulse_amp(*first_pulse_attr),4)),(-1.5,0))
        except :
            ann = plt.annotate('first pulse \n'+r'$time_tot $ = {0}  amp = {1}'.format(self.pulse_len(*first_pulse_attr),np.round(self.disentangle_cat_params['first_amp']*self.pulse_amp(*first_pulse_attr),4)),(-1.5,0))
        ann.draggable()
        return  fig_num,fit,fit_fidelty,fig
#%%  qubitPhase_4cat
    def load_qubitPhase_4cat(self,num_of_pnts=51,scale_phase_list = None,cat_phase = '+',update_frequency = None,
                         first_amp = 1.95,first_pulse = None,second_pulse = None,
                         update_prog = True, **kwargs):
    
        if first_pulse is None: first_pulse = self.Fast_ConDisp1
        if second_pulse is None: second_pulse = 'VFast_ConDisp1'
        if scale_phase_list is None:
            scale_phase_list = np.linspace(-1.0,1.0,num_of_pnts)
        prog,params = self.load_cavity_ramsey_vaccum(scale_amp_list = scale_phase_list, update_prog = False,start_with = self.cat_seq,Displacement_pulse=second_pulse,first_pulse = first_pulse,first_amp = first_amp,_is_sweep_qubit_phase= True,cat_phase=cat_phase,**kwargs)
        
        if update_prog: 
            params['first_pulse'] = first_pulse
            params['first_amp']   = first_amp
            params['cat_phase']   = cat_phase
            params['scale_phase_list']   = scale_phase_list*2
            self.qubitPhase_4cat_prog   = prog
            self.qubitPhase_4cat_params = params
            
        return prog, params

    def run_qubitPhase_4cat(self,is_phase = False,disp_amp = 1.0,# normalize = True,is_remove= False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):
        self.qubitPhase_4cat_results,_ , self.qubitPhase_4cat_plot_normaliztion=self.run_for_phase(self.load_qubitPhase_4cat,var_name ='scale_amp',is_phase=is_phase,disp_amp =disp_amp,**kwargs)
        
        if is_phase: return self.plot_qubitPhase_4cat_phase(**kwargs)
        return self.plot_disentangle_cat(**kwargs)
  
    def plot_qubitPhase_4cat_phase(self,results =None,params = None,fig_num= None,plot_normaliztion= None
                                        ,**kwargs):
        
        fig_num,fit,fit_fidelty,fig = self.fitNplot_phase(self.qubitPhase_4cat_results,
                                   self.qubitPhase_4cat_params,ti= self.qubitPhase_4cat_params['scale_phase_list'], 
                                   xlabel    = r'$Rad$',x_type    = 'phase',
                                   prog_name = f'qubit phase 4 cat phase',
                                   plot_normaliztion = self.qubitPhase_4cat_plot_normaliztion,
                                   fit_types =  ['Cos','Cos','Line180',None],**kwargs)
        first_pulse_attr=  [self.qubitPhase_4cat_params['mem_mode'],self.qubitPhase_4cat_params['first_pulse']]     
        ann = plt.annotate('first pulse \n'+r'$\sigma $ = {0}  amp = {1}'.format(self.pulse_sig(*first_pulse_attr),np.round(self.qubitPhase_4cat_params['first_amp']*self.pulse_amp(*first_pulse_attr),4)),(0,0))
        ann.draggable()
        return  fig_num,fit,fit_fidelty,fig
 #%% measure cat single axis
    def load_measure_cat_1Axis(self,num_of_pnts=51,scale_phase_list = None,cat_phase = '+',update_frequency = None,
                         first_amp = 1.95,second_amp = 1.0,first_pulse = None,second_pulse = None,measure_pulse = None,
                         update_prog = True, **kwargs):
    
        if first_pulse is None: first_pulse = self.Fast_ConDisp1
        if second_pulse is None: second_pulse = 'VFast_ConDisp1'
        if measure_pulse is None: measure_pulse = 'Cat_ConDisp1'
        
        prog,params = self.load_cavity_ramsey_vaccum( update_prog = False,start_with = self.cat_seq,Displacement_pulse=measure_pulse,second_pulse= second_pulse,second_amp = second_amp,first_pulse = first_pulse,first_amp = first_amp,_is_sweep_qubit_phase= False,cat_phase=cat_phase,**kwargs)
        
        if update_prog: 
            params['first_pulse'] = first_pulse
            params['first_amp']   = first_amp
            params['second_pulse'] = second_pulse
            params['second_amp'] = second_amp
            params['cat_phase']   = cat_phase
            self.measure_cat_1Axis_prog   = prog
            self.measure_cat_1Axis_params = params
            
        return prog, params

    def run_measure_cat_1Axis(self,is_phase = False,disp_amp = 1.0,# normalize = True,is_remove= False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):
        self.measure_cat_1Axis_results,_ , self.measure_cat_1Axis_plot_normaliztion=self.run_for_phase(self.load_measure_cat_1Axis,var_name ='scale_amp',is_phase=is_phase,disp_amp =disp_amp,**kwargs)
        
        if is_phase: return self.plot_measure_cat_1Axis_phase(**kwargs)
        return self.plot_disentangle_cat(**kwargs)
  
    def plot_measure_cat_1Axis_phase(self,results =None,params = None,fig_num= None,plot_normaliztion= None
                                        ,**kwargs):
        
        fig_num,fit,fit_fidelty,fig = self.fitNplot_phase(self.measure_cat_1Axis_results,
                                   self.measure_cat_1Axis_params,ti= self.measure_cat_1Axis_results['Re']["var"], 
                                   xlabel    = r'$amplitude$',x_type    = 'amp',
                                   prog_name = f'measure cat 1 axis phase',
                                   plot_normaliztion = self.measure_cat_1Axis_plot_normaliztion,
                                   fit_types =  [None,None,None,None],**kwargs)
        first_pulse_attr=  [self.measure_cat_1Axis_params['mem_mode'],self.measure_cat_1Axis_params['first_pulse']]     
        ann = plt.annotate('first pulse \n'+r'$\sigma $ = {0}  amp = {1}'.format(self.pulse_sig(*first_pulse_attr),np.round(self.measure_cat_1Axis_params['first_amp']*self.pulse_amp(*first_pulse_attr),4)),(1.5,0))
        ann.draggable()
        return  fig_num,fit,fit_fidelty,fig
        
   #%%angle calibration
    def load_cavity_calibrate_angle(self, num_of_pnts = 51, # How many points to measure, .
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    N_repeat = 3, #How many times is the sequence reptead for better SNR 
                                    scale_angle_list= None,
                                    Displacement_pulse = None,
                                    mem_mode=None,
                                    start_with: tuple = None, #(self.pi_pulse,self.main_qubits)
                                    update_prog = True,
                                    pi2_phase = 0.5, #phase of second pi2 pulse in units of pi 
                                    normalize = False,
                                    is_displaced = True, #both states are displaced in the same direction 
                                    is_remove = False,
                                    is_phase = False,
                                    is_measure_Z = False,
                                    scale_amp = 1.0,
                                    **kwargs):
        
        if mem_mode is None: mem_mode =self.main_mem
        if Displacement_pulse is None: Displacement_pulse = self.ConDisp1_pulse
        # max_seq_time = int(max_seq_clks * 4)                            
        
        pi2_phase_list,qb_pulse_list =self.get_qb_measure_seq_list(pi2_phase=pi2_phase,is_remove= is_remove, is_phase = is_phase, is_measure_Z=is_measure_Z,normalize=normalize)

        if scale_angle_list is None: 
            if not (num_of_pnts%2): num_of_pnts =num_of_pnts+1
            scale_angle_list = np.linspace(-0.5,0.5,num_of_pnts,dtype=float)
      
        if type(scale_angle_list) == list: scale_angle_list= np.array(scale_angle_list)
        # rot_phase_list =  (scale_angle_list**2)*detuning/2/np.pi + pi2_phase
        if is_displaced:
            N_repeat = 1
            
        scale_angle_list    = scale_angle_list.tolist()
        num_of_pnts = len(scale_angle_list)
        # calibrate_angle_N_avg = N_avg
        #Save parameters:
        cavity_calibrate_angle_params = {}
        cavity_calibrate_angle_params["N_avg"]              = N_avg
        cavity_calibrate_angle_params["num_of_pnts"]        = num_of_pnts
        cavity_calibrate_angle_params["N_repeat"]           = N_repeat
        cavity_calibrate_angle_params["Displacement_pulse"] = Displacement_pulse
        cavity_calibrate_angle_params["mem_mode"]           = mem_mode  
        cavity_calibrate_angle_params["start_with"]         = start_with
        cavity_calibrate_angle_params["scale_amp"]          = scale_amp
        cavity_calibrate_angle_params["scale_angle_list"]   = scale_angle_list
        cavity_calibrate_angle_params["pi2_phase"]          = pi2_phase
        cavity_calibrate_angle_params["is_displaced"]       = is_displaced
        cavity_calibrate_angle_params["pi2_phase_list"]     = pi2_phase_list
        
        # cavity_calibrate_angle_params["step_size_clks"]     = step_size_clks
        
        # pi_amp_mul = 0.0 if normalize else 1.0
        
        if update_prog: self.cavity_calibrate_angle_params=cavity_calibrate_angle_params
        
        assert  (not self.pulse_len(self.main_qubit,self.pi_pulse)%4), 'pi pulse length does not divide by 4'
        pi_wait_time = self.pulse_len(self.main_qubit,self.pi_pulse)//4

        #Calculate and show expeced runtime:
        try:
            run_time = N_avg*num_of_pnts*(self.pulse_len(self.main_readout,self.ro_pulse)+self.wait_between_seq * 4 +N_repeat*2*self.pulse_len(mem_mode,Displacement_pulse))*len(pi2_phase_list)
            print('Run time is {}s'.format(round(run_time * 1e-9)))         
        except:
            print('cant calculate time')
        
        print(pi2_phase_list)
        #Create program:
        with program() as prog:
            n             = declare(int)
            # n_rep         = declare(int)
            i             = declare(int)
            j             = declare(int)
            I             = declare(fixed)
            Q             = declare(fixed)
            I_st          = declare_stream()
            Q_st          = declare_stream()
            pi2_phase_arr = declare(fixed,value = pi2_phase_list)
            qb_pulse_arr  = declare(int,value =qb_pulse_list)
            rot_phase     = declare(fixed)
            qb_pulse      = declare(int)
            scale_angle_arr= declare(fixed,value = scale_angle_list)
            scale_angle   = declare(fixed)
            with for_(n,0,n<N_avg,n+1):
                with for_(i,0,i<len(scale_angle_list),i+1):
                    # assign(qb_phase_corr,qb_phasCor_arr[i])
                    
                    assign(scale_angle, scale_angle_arr[i])
                    with for_(j,0,j<pi2_phase_arr.length(),j+1):
                        assign(qb_pulse,qb_pulse_arr[j])
                        # assign(qb_amp, qb_amp_arr[j])
                        assign(rot_phase, pi2_phase_arr[j])
                        
                        reset_frame(mem_mode,self.main_qubit)
                        reset_phase(mem_mode)
                        reset_phase(self.main_qubit)

                        if type(start_with) is tuple: 
                            play(*start_with)
                            align(start_with[1],self.main_qubit)
                        elif type(start_with) is types.FunctionType:
                            start_with(mem_mode = mem_mode,**kwargs)
                        else:
                            warn('\n ******** \n no starting sequence \n ******** \n')
                        
                                        
                        play(self.pi2_pulse,self.main_qubit)
                        align(self.main_qubit, mem_mode)
                        
                        # with for_(n_rep,0,n_rep<N_repeat,n_rep+1):
                        for n_rep in range(N_repeat):
                            align(self.main_qubit, mem_mode)
                            # align(self.main_qubit, mem_mode,'mm2')

                            if n_rep>0:
                                play(self.pi_pulse,self.main_qubit)
                                align(self.main_qubit, mem_mode)
                                # align(self.main_qubit, mem_mode,'mm2')
                                
                                
                            play(Displacement_pulse*amp(scale_amp),mem_mode)   
                            # play(Displacement_pulse*amp(scale_amp),'mm2')   

                            align(self.main_qubit, mem_mode)
                            # align(self.main_qubit, mem_mode,'mm2')
                            
                            play(self.pi_pulse,self.main_qubit)
                            frame_rotation_2pi(scale_angle,mem_mode)
                            align(self.main_qubit, mem_mode)
                            # align(self.main_qubit, mem_mode,'mm2')

                            if is_displaced:
                                play(Displacement_pulse*amp(scale_amp),mem_mode)   
                            else:
                                play(Displacement_pulse*amp(-scale_amp),mem_mode)   
                                if 1:
                                    align(self.main_qubit, mem_mode)
                                    play(self.pi_pulse,self.main_qubit)
                                    # frame_rotation_2pi(scale_angle,mem_mode)
                                    align(self.main_qubit, mem_mode)
                                    play(Displacement_pulse*amp(-scale_amp),mem_mode)   
                                    align(self.main_qubit, mem_mode)
                                    play(self.pi_pulse,self.main_qubit)
                                    frame_rotation_2pi(scale_angle,mem_mode)
                                    align(self.main_qubit, mem_mode)
                                    play(Displacement_pulse*amp(scale_amp),mem_mode)   
                                    # align(self.main_qubit, mem_mode)
                            # with if_(n_rep>N_repeat-2):
                        frame_rotation_2pi(rot_phase,self.main_qubit)
                        align(self.main_qubit, mem_mode)
                            # frame_rotation_2pi(pi2_phase,self.main_qubit)
                        with switch_(qb_pulse,unsafe = True):
                            with case_(1): 
                                play(self.pi2_pulse,self.main_qubit)
                            with case_(2): 
                                play(self.pi_pulse,self.main_qubit)
                            with case_(0): 
                                wait(pi_wait_time,self.main_qubit)
                        align(self.main_qubit, self.main_readout)
                            # self.perform_full_measurement(I,Q)#,I_output_name = I_st,Q_output_name = Q_st)
                        self.perform_full_measurement(I,Q,I_output_name = I_st,Q_output_name = Q_st)
    
    
                        save(scale_angle, 'scale_angle')
            with stream_processing():
                I_st.save_all('I')
                Q_st.save_all('Q')
        if update_prog: self.cavity_calibrate_angle_prog = prog
        # self.simulate_prog(prog)
        return prog,cavity_calibrate_angle_params

    def run_cavity_calibrate_angle(self,#normalize = True,is_remove= False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):
        #TODO orginize to complete and run sperate
        if 1:
             self.cavity_calibrate_angle_results,_ , self.calibrate_angle_plot_normaliztion=self.run_for_phase(self.load_cavity_calibrate_angle,var_name ='scale_angle',**kwargs)
             return self.plot_calibrate_angle_phase(**kwargs)
        else:
     
            if not hasattr(self, 'cavity_calibrate_angle_prog'): raise ValueError("Idiot! You did not write calibrate_angle program")
         
            self.cavity_calibrate_angle_results = dict()
            if True:
                self.cavity_calibrate_angle_results["angle"], self.cavity_calibrate_angle_results["I_orig"], self.cavity_calibrate_angle_results["Q_orig"] = self.run_prog(self.cavity_calibrate_angle_prog , self.cavity_calibrate_angle_params["N_avg"],var = 'scale_angle' )
             
                prog,_ = self.load_cavity_calibrate_angle(**self.cavity_calibrate_angle_params,is_remove = True,update_prog = False)
                _, self.cavity_calibrate_angle_results["I_rev"], self.cavity_calibrate_angle_results["Q_rev"] = self.run_prog(prog ,self.cavity_calibrate_angle_params["N_avg"],var = 'scale_angle' )
                
                self.cavity_calibrate_angle_results["I"] = self.cavity_calibrate_angle_results["I_orig"]-self.cavity_calibrate_angle_results["I_rev"]
                self.cavity_calibrate_angle_results["Q"] = self.cavity_calibrate_angle_results["Q_orig"]-self.cavity_calibrate_angle_results["Q_rev"]
                plot_normaliztion = False
            else:
                self.cavity_calibrate_angle_results["angle"], self.cavity_calibrate_angle_results["I"], self.cavity_calibrate_angle_results["Q"] = self.run_prog(self.cavity_calibrate_angle_prog ,self.cavity_calibrate_angle_params["N_avg"],var = 'scale_angle' )        
                # return self.cavity_calibrate_angle_results["angle"], self.cavity_calibrate_angle_results["I"], self.cavity_calibrate_angle_results["Q"] 
                if False:
                    prog,_ = self.load_cavity_calibrate_angle(**self.cavity_calibrate_angle_params,normalize = True,update_prog = False)
                    self.cavity_calibrate_angle_results =  self.normalize_CrossKerr(self.cavity_calibrate_angle_params, self.cavity_calibrate_angle_results,prog = prog,var = 'scale_angle')
                else:
                    plot_normaliztion =False        
            
            self.calibrate_angle_plot_normaliztion = plot_normaliztion
        # return self.cavity_calibrate_angle_results
        return self.plot_cavity_calibrate_angle(**kwargs)


    def plot_calibrate_angle_phase(self,results =None,params = None,fig_num= None,plot_normaliztion= None
                                        ,**kwargs):
        return self.fitNplot_phase(self.cavity_calibrate_angle_results,
                                   self.cavity_calibrate_angle_params,ti= self.cavity_calibrate_angle_results['Re']["var"], 
                                   xlabel    = r'$angle$ [rad]',x_type    = 'amp',
                                   prog_name = f'calibrate angle phase',
                                   plot_normaliztion = False,
                                   fit_types =  [None,None,None,'Gaussian'],**kwargs)
    
    def plot_cavity_calibrate_angle(self,start_x_fit=None,plot_normaliztion=None,txt= '',fig_num = None,**kwargs):
        plot_normaliztion = self.calibrate_angle_plot_normaliztion if plot_normaliztion is None else plot_normaliztion
        # Check what's happening here:
        #           fit_and_plot(self,fittype, I, Q, ti, N_avg,  txt=None,title_str =None, fig_num=None ):  
        prog_name = 'calibrate_angle EM'
        if fig_num is None: fig_num = prog_name + ' ' + str(self.next_fig_num_by_prog_name(prog_name))
        
        plt.xlabel('angle [Rad]')
        
        fig, fit,fit_fidelty, x_text,y_text = self.fit_and_plot( 'Cos' , self.process_data(self.cavity_calibrate_angle_results), ti = self.cavity_calibrate_angle_results["var"]  , title_str = f"calibrate_angle EM ({self.main_mem})",label = 'Fitted data', fig_num = fig_num,x_type = 'angle', **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        # fig, x_text,y_text = self.fit_and_plot( None , self.process_data(self.cavity_calibrate_angle_results), ti = self.cavity_calibrate_angle_results["angle"]  , title_str = f"calibrate_angle EM ({self.main_mem})",label = 'Fitted data', fig_num = fig_num,x_type = 'angle', **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        # ret  p=lt.plot( self.autoscale_data(self.process_data(self.cavity_calibrate_angle_results))#, ti = self.cavity_calibrate_angle_results["angle"]  , title_str = f"calibrate_angle EM ({self.main_mem})",label = 'Fitted data', fig_num = fig_num, **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        print(x_text)
        print(y_text)
        # print(fit)
        # print(fit)

        if plot_normaliztion:
            plt.plot(self.autoscale_data(self.cavity_calibrate_angle_results["angle"])[0], self.autoscale_data(self.process_data(data = [self.cavity_calibrate_angle_results["I_orig"], self.cavity_calibrate_angle_results["Q_orig"]])[0])[0],'+',label = 'Measured Data')
            plt.plot(self.autoscale_data(self.cavity_calibrate_angle_results["angle"])[0], self.autoscale_data(self.process_data(data = [self.cavity_calibrate_angle_results["I_CrossKerr"], self.cavity_calibrate_angle_results["Q_CrossKerr"]], is_calc_stat_error = False, is_thresh = False))[0],'o', label = 'Cross Kerr')
            leg = plt.legend()
            leg.set_draggable(True)
        if not fit is None:
            # txt = txt + r'$| \alpha |$' +' =  {0}'.format(np.round(abs(1/fit[2]),4))
            txt = r'angle to add =  {0} $\pm$ {1} [us] ] \n # repeats = {2}'.format(*round_value_by_error(fit[2],fit_fidelty[2]),self.cavity_calibrate_angle_params["N_repeat"])
            ann = plt.annotate(txt, xy = (x_text,y_text), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
            ann.draggable()
        
        # plt.title(r'$\sigma $ = {0} amp = {1}'.format(self.pulse_sig(self.cavity_calibrate_angle_params["mem_mode"] ,self.cavity_calibrate_angle_params["Displacement_pulse"] ),self.pulse_amp(self.cavity_calibrate_angle_params["mem_mode"] , self.cavity_calibrate_angle_params["Displacement_pulse"] )))
        #fig.show()
        # plt.tight_layout()
        return fig,fit,fit_fidelty
    #%% Ramsey for un conditional displacment calibration
    def load_cavity_ramsey_displacment(self, num_of_pnts = 51, # How many points to measure, .
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    scale_amp_list = None,
                                    Displacement_pulse = None,
                                    mem_mode=None,
                                    start_with: tuple = None, #(self.pi_pulse,self.main_qubits)
                                    update_prog = True,
                                    pi2_phase = 0.5, #phase of second pi2 pulse in units of pi 
                                    normalize = False,
                                    is_Yale= None,
                                    ConDisp_pulse= None,
                                    is_remove = False,
                                    **kwargs):
        
        if mem_mode is None: mem_mode =self.main_mem
        if Displacement_pulse is None: Displacement_pulse = self.UnDisp1_pulse
        if ConDisp_pulse is None: ConDisp_pulse   = self.ConDisp1_pulse
        if is_Yale is None: is_Yale   = self.is_Yale
        # max_seq_time = int(max_seq_clks * 4)                            
        if is_remove:
            pi2_phase = pi2_phase -0.5
        
        if scale_amp_list is None: 
            if not (num_of_pnts%2): num_of_pnts =num_of_pnts+1
            scale_amp_list = np.linspace(-1.9,1.9,num_of_pnts,dtype=float).tolist()
        
        num_of_pnts = len(scale_amp_list)
        # ramsey_displacment_N_avg = N_avg

        #Save parameters:
        cavity_ramsey_displacment_params = {}
        cavity_ramsey_displacment_params["N_avg"]              = N_avg
        cavity_ramsey_displacment_params["num_of_pnts"]        = num_of_pnts
        cavity_ramsey_displacment_params["Displacement_pulse"] = Displacement_pulse
        cavity_ramsey_displacment_params["ConDisp_pulse"]      = ConDisp_pulse
        cavity_ramsey_displacment_params["mem_mode"]           = mem_mode  
        cavity_ramsey_displacment_params["start_with"]         = start_with
        cavity_ramsey_displacment_params["scale_amp_list"]     = scale_amp_list
        cavity_ramsey_displacment_params["is_Yale"]            = is_Yale
        cavity_ramsey_displacment_params["pi2_phase"]          = pi2_phase
        
        # cavity_ramsey_displacment_params["step_size_clks"]     = step_size_clks
        
        pi_amp_mul = 0.0 if normalize else 1.0
        
        if update_prog: self.cavity_ramsey_displacment_params=cavity_ramsey_displacment_params
            
        #Calculate and show expeced runtime:
        try:
            run_time = N_avg*num_of_pnts*(self.pulse_len(self.main_readout,self.ro_pulse)+self.wait_between_seq * 4 +self.pulse_len(mem_mode,Displacement_pulse))
            print('Run time is {}s'.format(round(run_time * 1e-9)))         
        except:
            print('cant calculate time')
        
        #Create program:
        with program() as prog:
            n             = declare(int)
            I             = declare(fixed)
            Q             = declare(fixed)
            scale_amp     = declare(fixed)
            with for_(n,0,n<N_avg,n+1):
                with for_each_(scale_amp, scale_amp_list):

                    reset_frame(mem_mode)
                    reset_frame(self.main_qubit)
                    
                    if not start_with is None: 
                        play(*start_with)
                        align(start_with[1],mem_mode)
                    

                    play(self.pi2_pulse,self.main_qubit)
                        
                    align(self.main_qubit, mem_mode)
                    self._ConDisp_pulse(qubit = self.main_qubit, mem_mode = mem_mode, scale_amp =1,Displacement_pulse = ConDisp_pulse,is_Yale = is_Yale,**kwargs)
                    # play(ConDisp_pulse,mem_mode)
                    
                    # frame_rotation_2pi(0.0,mem_mode)
                    play(Displacement_pulse*amp(scale_amp) , mem_mode )
                    # reset_frame(mem_mode)

                    # align(self.main_qubit, mem_mode)
                    # self._pi_pulse(qubit = self.main_qubit,**kwargs)
                    # align(self.main_qubit, mem_mode)

                    self._ConDisp_pulse(qubit = self.main_qubit, mem_mode = mem_mode, scale_amp =1,Displacement_pulse = ConDisp_pulse,is_Yale = is_Yale,**kwargs)
                    # play(ConDisp_pulse*amp(-1),mem_mode)

                    # frame_rotation_2pi(0.0,mem_mode)
                    play(Displacement_pulse*amp(-scale_amp) , mem_mode )
                    # reset_frame(mem_mode)
                    
                    align(self.main_qubit, mem_mode)
                    frame_rotation_2pi(pi2_phase,self.main_qubit)
                    play(self.pi2_pulse*amp(pi_amp_mul),self.main_qubit)
                    
                    align(self.main_qubit, self.main_readout)
                    self.perform_full_measurement(I,Q)

                    save(scale_amp, 'scale_amp')
        
        if update_prog: self.cavity_ramsey_displacment_prog = prog
        
        return prog,cavity_ramsey_displacment_params
    def complete_cavity_ramsey_displacment(self,num_of_pnts = 50, # How many points to measure, .
                                    max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    # normalize = False,#if normalize is true the program will run 
                                    just_run=True,
                                    **kwargs):
       #TODO finish this to work with normalize
       self.cavity_ramsey_displacment_results["amp"], self.cavity_ramsey_displacment_results["I"], self.cavity_ramsey_displacment_results["Q"] = self.run_prog(self.cavity_ramsey_displacment_prog ,self.cavity_ramsey_displacment_params["N_avg"],var = 'scale_amp' )        
       return self.cavity_ramsey_displacment_results["amp"], self.cavity_ramsey_displacment_results["I"], self.cavity_ramsey_displacment_results["Q"] 
        
        # self.cavity_ramsey_displacment_results["t"], self.cavity_ramsey_displacment_results["I"], self.cavity_ramsey_displacment_results["Q"] = self.run_prog(self.cavity_ramsey_displacment_prog ,self.cavity_ramsey_displacment_params["N_avg"] )        
        # return self.plot_cavity_ramsey_displacment(**kwargs)

    def run_cavity_ramsey_displacment(self,normalize = True,is_remove= False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):
        #TODO orginize to complete and run sperate

        if not hasattr(self, 'cavity_ramsey_displacment_prog'): raise ValueError("Idiot! You did not write ramsey_displacment program")
     
        self.cavity_ramsey_displacment_results = dict()
        if is_remove:
            self.cavity_ramsey_displacment_results["amp"], self.cavity_ramsey_displacment_results["I_orig"], self.cavity_ramsey_displacment_results["Q_orig"] = self.run_prog(self.cavity_ramsey_displacment_prog , self.cavity_ramsey_displacment_params["N_avg"],var = 'scale_amp' )
         
            prog,_ = self.load_cavity_ramsey_displacment(**self.cavity_ramsey_displacment_params,is_remove = True,update_prog = False)
            _, self.cavity_ramsey_displacment_results["I_rev"], self.cavity_ramsey_displacment_results["Q_rev"] = self.run_prog(prog ,self.cavity_ramsey_displacment_params["N_avg"],var = 'scale_amp' )
            
            self.cavity_ramsey_displacment_results["I"] = self.cavity_ramsey_displacment_results["I_orig"]-self.cavity_ramsey_displacment_results["I_rev"]
            self.cavity_ramsey_displacment_results["Q"] = self.cavity_ramsey_displacment_results["Q_orig"]-self.cavity_ramsey_displacment_results["Q_rev"]
            plot_normaliztion = False
        else:
            self.cavity_ramsey_displacment_results["amp"], self.cavity_ramsey_displacment_results["I"], self.cavity_ramsey_displacment_results["Q"] = self.run_prog(self.cavity_ramsey_displacment_prog ,self.cavity_ramsey_displacment_params["N_avg"],var = 'scale_amp' )        
            # return self.cavity_ramsey_displacment_results["amp"], self.cavity_ramsey_displacment_results["I"], self.cavity_ramsey_displacment_results["Q"] 
            if normalize:
                prog,_ = self.load_cavity_ramsey_displacment(**self.cavity_ramsey_displacment_params,normalize = True,update_prog = False)
                self.cavity_ramsey_displacment_results =  self.normalize_CrossKerr(self.cavity_ramsey_displacment_params, self.cavity_ramsey_displacment_results,prog = prog,var = 'scale_amp')
            else:
                plot_normaliztion =False        
        
        self.ramsey_displacment_plot_normaliztion = plot_normaliztion
        # return self.cavity_ramsey_displacment_results
        return self.plot_cavity_ramsey_displacment(**kwargs)


    
    def plot_cavity_ramsey_displacment(self,start_x_fit=None,plot_normaliztion=None,txt= '',**kwargs):
        plot_normaliztion = self.ramsey_displacment_plot_normaliztion if plot_normaliztion is None else plot_normaliztion
        # Check what's happening here:
        #           fit_and_plot(self,fittype, I, Q, ti, N_avg,  txt=None,title_str =None, fig_num=None ):  
        prog_name = 'ramsey_displacment EM'
        fig_num = prog_name + ' ' + str(self.next_fig_num_by_prog_name(prog_name))
        
        
        fig,fit,fit_fidelty,x,y = self.fit_and_plot( 'Cos' , self.process_data(self.cavity_ramsey_displacment_results), ti = self.cavity_ramsey_displacment_results["amp"]  , title_str = f"ramsey_displacment EM ({self.main_mem})",label = 'Fitted data', fig_num = fig_num,x_type = 'amp', **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        # plt.figure()
        plt.xlabel('amplitude scaling')
        # return  plt.plot( self.autoscale_data(self.process_data(self.cavity_ramsey_displacment_results), ti = self.cavity_ramsey_displacment_results["amp"]  , title_str = f"ramsey_displacment EM ({self.main_mem})",label = 'Fitted data', fig_num = fig_num, **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        if plot_normaliztion:
            plt.plot(self.autoscale_data(self.cavity_ramsey_displacment_results["amp"])[0], self.autoscale_data(self.process_data(data = [self.cavity_ramsey_displacment_results["I_orig"], self.cavity_ramsey_displacment_results["Q_orig"]])[0])[0],'+',label = 'Measured Data')
            plt.plot(self.autoscale_data(self.cavity_ramsey_displacment_results["amp"])[0], self.autoscale_data(self.process_data(data = [self.cavity_ramsey_displacment_results["I_CrossKerr"], self.cavity_ramsey_displacment_results["Q_CrossKerr"]], is_calc_stat_error = False, is_thresh = False))[0],'o', label = 'Cross Kerr')
            leg = plt.legend()
            leg.set_draggable(True)
        if not fit is None:
            txt = txt + r'$| \beta |$' +r' =  ({0} $\pm$ {1})/$| \alpha |$ '.format(*round_value_by_error(fit[1]*2*np.pi,fit_fidelty[1]*2*np.pi))
            # txt = txt + r'$| \beta |$' +r' =  ({0} $\pm$ {1})/$| \alpha |$ '.format(*round_value_by_error(abs(2*fit[0]),2*fit_fidelty[0]))
            # ann = plt.annotate( r'decay time =  {0} $\pm$ {1} [us]'.format(*round_value_by_error(fit[2],fit_fidelty[2]), xy = (x_text,y_text), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
        if not txt == '':
            ann = plt.annotate(txt, xy = (0,y), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
            ann.draggable()
            
        #fig.show()
        plt.tight_layout()
        return fig,fit,fit_fidelty

       
#%% displacment calibration

    def load_displacment_calib_spectroscopy(self, chis_list = None, scale_amp_list = np.arange(0.1,3,0.1),
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    rotation_pulse = None,
                                    Displacement_pulse = None,
                                    mem_mode=None,
                                    normalize = False,
                                    update_prog = True):
        
        if rotation_pulse is None:     rotation_pulse = self.ConRotPi_pulse
        if mem_mode is None:           mem_mode =self.main_mem
        if Displacement_pulse is None: Displacement_pulse=self.UnDisp1_pulse 

        IF_list     = chis_list + self.element_IF(self.main_qubit)
        
        num_of_pnts = len(IF_list)
        
        #Save parameters:
        params = {}
        params["N_avg"]              = N_avg
        params["chis_list"]          = chis_list
        params["IF_list"]            = IF_list
        params["rotation_pulse"]     = rotation_pulse
        params["Displacement_pulse"] = Displacement_pulse
        params["mem_mode"]           = mem_mode  
        params["scale_amp_list"]     = scale_amp_list  

        pi_amp_mul = 0.0 if normalize else 1.0

        if update_prog: self.spec_displacment_calib_params = params
        
        try:
            run_time = N_avg*num_of_pnts*(self.pulse_len(self.main_readout,self.ro_pulse)+self.wait_between_seq * 4)
            print('Run time is {}s'.format(round(run_time * 1e-9)))         
        except:
            print('cant calculate time')
        
        #Create program:
        with program() as prog:
            n         = declare(int)
            I         = declare(fixed)
            Q         = declare(fixed)
            freq      = declare(int)
            S_amp = declare(fixed)
            with for_(n,0,n<N_avg,n+1):
                with for_each_(freq , IF_list.astype(int).tolist()):
                    update_frequency(self.main_qubit,freq)
                    align(self.main_qubit, mem_mode)
                    with for_each_(S_amp,scale_amp_list.tolist()):
                    
                        play(Displacement_pulse*amp(S_amp) , mem_mode )
                        align(self.main_qubit, mem_mode)
                        
                    self.cavity2qubit_0_mapping(I,Q,mem_mode = mem_mode,pi_amp_mul = pi_amp_mul )
#TODO orginize data collection
                    save(freq, 'freq')
        
        if update_prog: self.spec_displacment_calib_prog = prog
        
        return prog,params
                        
    def run_displacment_calib_spectroscopy(self,normalize = False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 is_plot_results= True,**kwargs):
        #TODO adjust normalization to work

        if not hasattr(self, 'spec_displacment_calib_prog'): raise ValueError("Idiot! You did not write spec_displacment_calib_")
     
        self.spec_displacment_calib_results = dict()
        self.spec_displacment_calib_results["freq"], self.spec_displacment_calib_results["I"], self.spec_displacment_calib_results["Q"] = self.run_spec(self.spec_displacment_calib_prog ,self.spec_displacment_calib_params["N_avg"] )        
    
        if normalize:
            prog,_ = self.load_displacment_calib_spectroscopy(**self.spec_displacment_calib_params,normalize = True,update_prog = False)
            self.spec_displacment_calib_results =  self.normalize_CrossKerr(self.spec_displacment_calib_params, self.spec_displacment_calib_results,prog = prog)
        else:
            plot_normaliztion =False        
        self.displacment_calib_plot_normaliztion = plot_normaliztion
        if is_plot_results:
            return self.plot_displacment_calib_spectroscopy(**kwargs)
        return 


    def plot_displacment_calib_spectroscopy(self,start_x_fit=None,plot_normaliztion=None,**kwargs):
        plot_normaliztion = self.displacment_calib_plot_normaliztion if plot_normaliztion is None else plot_normaliztion

        # ret = self.fit_and_plot( 'Exp(Exp)' , self.process_data(self.spec_displacment_calib_results), ti = self.spec_displacment_calib_results["freq"]  , title_str = f"T1 EM ({self.main_mem})",label = 'Fitted data',**kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        #TODO with Eliya turn in to a spectroscopy plotter
        ret = plt.figure()
        plt.plot(self.autoscale_data(self.spec_displacment_calib_params["chis_list"])[0], self.autoscale_data(self.process_data(data = [self.spec_displacment_calib_results["I"], self.spec_displacment_calib_results["Q"]])[0])[0],'-o',label = 'Measured Data')
        if plot_normaliztion:
            plt.plot(self.autoscale_data(self.spec_displacment_calib_params["chis_list"])[0], self.autoscale_data(self.process_data(data = [self.spec_displacment_calib_results["I_orig"], self.spec_displacment_calib_results["Q_orig"]])[0])[0],'+',label = 'Measured Data')
            plt.plot(self.autoscale_data(self.spec_displacment_calib_params["chis_list"])[0], self.autoscale_data(self.process_data(data = [self.spec_displacment_calib_results["I_CrossKerr"], self.spec_displacment_calib_results["Q_CrossKerr"]], is_calc_stat_error = False, is_thresh = False))[0],'o', label = 'CrossKerr')
            leg = plt.legend()
            leg.set_draggable(True)
        return ret
    
    def run_displacment_calibration(self,amp_list,#list of floats
                                chis_of_mode = None,#list of (negative) floats in Hz
                                normalize = True,
                                plot_normaliztion =False,**kwargs):
        
        if chis_of_mode is None: chis_of_mode = self.chis_of_mode
        
        for amp in amp_list:
            self.load_number_splitting_spectroscopy(chis_list = chis_of_mode,
                                                    is_sweep_EM = False,is_plot_results = False,
                                                    **kwargs)
    #%% cavity_charactaristic_function_decay

    def load_cavity_charactaristic_function_decay(self, num_of_pnts = 51, # How many points to measure, .
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    max_seq_time =  200e3,
                                    Displacement_pulse = None,
                                    disp_amp = 1.0,
                                    condisp_amp = 1.0,
                                    mem_mode=None,
                                    start_with: tuple = None, #(self.pi_pulse,self.main_qubits)
                                    update_prog = True,
                                    pi2_phase = 0.5, #phase of second pi2 pulse in units of pi 
                                    normalize = False,
                                    is_Yale= None,
                                    ConDisp_pulse= None,
                                    Cavity_detuning = 0,
                                    is_remove = False,
                                    **kwargs):
        
        max_seq_clks= int(max_seq_time // 4)
        step_size_clks =  max_seq_clks // num_of_pnts
        if mem_mode is None: mem_mode =self.main_mem
        if Displacement_pulse is None: Displacement_pulse = self.UnDisp1_pulse
        if ConDisp_pulse is None: ConDisp_pulse   = self.ConDisp1_pulse
        if is_Yale is None: is_Yale   = self.is_Yale
        # max_seq_time = int(max_seq_clks * 4)                            
        if is_remove:
            pi2_phase = pi2_phase -0.5
        # rot_phase_list = detuning/2/np.pi + pi2_phase
        # rot_phase_list = rot_phase_list.tolist()
        wt_time_list     = np.arange(0, max_seq_clks, step_size_clks)
        C_rot_phase_list = 4*Cavity_detuning*wt_time_list
        
        wt_time_list     = wt_time_list.tolist()
        C_rot_phase_list = C_rot_phase_list.tolist()
    
        #Save parameters:
        cavity_charactaristic_function_decay_params = {}
        cavity_charactaristic_function_decay_params["N_avg"]              = N_avg
        cavity_charactaristic_function_decay_params["max_seq_time"]       = max_seq_time
        cavity_charactaristic_function_decay_params["num_of_pnts"]        = num_of_pnts
        cavity_charactaristic_function_decay_params["Displacement_pulse"] = Displacement_pulse
        cavity_charactaristic_function_decay_params["ConDisp_pulse"]      = ConDisp_pulse
        cavity_charactaristic_function_decay_params["mem_mode"]           = mem_mode  
        cavity_charactaristic_function_decay_params["start_with"]         = start_with
        cavity_charactaristic_function_decay_params["is_Yale"]            = is_Yale
        cavity_charactaristic_function_decay_params["disp_amp"]           = disp_amp
        cavity_charactaristic_function_decay_params["condisp_amp"]        = condisp_amp
        cavity_charactaristic_function_decay_params["pi2_phase"]          = pi2_phase
        cavity_charactaristic_function_decay_params["Cavity_detuning"]    = Cavity_detuning
        cavity_charactaristic_function_decay_params["wt_time_list"]       = wt_time_list
        
        # cavity_charactaristic_function_decay_params["step_size_clks"]     = step_size_clks
        
        pi_amp_mul = 0.0 if normalize else 1.0
        
        if update_prog: self.cavity_charactaristic_function_decay_params=cavity_charactaristic_function_decay_params
            
        #Calculate and show expeced runtime:
        try:
            run_time = N_avg*num_of_pnts*(self.pulse_len(self.main_readout,self.ro_pulse)+max_seq_time/2+self.wait_between_seq * 4 +self.pulse_len(mem_mode,Displacement_pulse))
            print('Run time is {}s'.format(round(run_time * 1e-9)))         
        except:
            print('cant calculate time')
        
        
        #Create program:
        with program() as prog:
            wt_time       = declare(int)
            n             = declare(int)
            C_rot_phase   = declare(fixed)
            I             = declare(fixed)
            Q             = declare(fixed)
            with for_(n,0,n<N_avg,n+1):
                with for_each_((wt_time, C_rot_phase),(wt_time_list,C_rot_phase_list)):
                # with for_(wt_time, step_size_clks, wt_time <= max_seq_clks, wt_time + step_size_clks):
                    reset_frame(mem_mode,self.main_qubit)
                    
                    if not start_with is None: 
                        play(*start_with)
                        align(start_with[1],mem_mode)
                    
                    play(Displacement_pulse*amp(disp_amp) , mem_mode )
                    
                    with if_(wt_time>0):
                        frame_rotation_2pi(C_rot_phase,mem_mode)
                        wait(wt_time, mem_mode)

                    
                    align(self.main_qubit, mem_mode)
                    play(self.pi2_pulse,self.main_qubit)
                                            
                    align(self.main_qubit, mem_mode)
                    self._ConDisp_pulse(qubit = self.main_qubit, mem_mode = mem_mode, scale_amp=condisp_amp,Displacement_pulse = ConDisp_pulse,is_Yale = is_Yale,**kwargs)

                    align(self.main_qubit, mem_mode)

                    frame_rotation_2pi(pi2_phase,self.main_qubit)
                    play(self.pi2_pulse*amp(pi_amp_mul),self.main_qubit)
                    
                    align(self.main_qubit, self.main_readout)
                    self.perform_full_measurement(I,Q)

                    # save(wt_time, 't')
        
        if update_prog: self.cavity_charactaristic_function_decay_prog = prog
        
        return prog,cavity_charactaristic_function_decay_params
    def complete_cavity_charactaristic_function_decay(self,num_of_pnts = 50, # How many points to measure, .
                                    max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    # normalize = False,#if normalize is true the program will run 
                                    just_run=True,
                                    **kwargs):
       #TODO finish this to work with normalize
       raise ValueError('not ready yet')
       self.cavity_charactaristic_function_decay_results["amp"], self.cavity_charactaristic_function_decay_results["I"], self.cavity_charactaristic_function_decay_results["Q"] = self.run_prog(self.cavity_charactaristic_function_decay_prog ,self.cavity_charactaristic_function_decay_params["N_avg"],var = 'scale_amp' )        
       return self.cavity_charactaristic_function_decay_results["amp"], self.cavity_charactaristic_function_decay_results["I"], self.cavity_charactaristic_function_decay_results["Q"] 
        
        # self.cavity_charactaristic_function_decay_results["t"], self.cavity_charactaristic_function_decay_results["I"], self.cavity_charactaristic_function_decay_results["Q"] = self.run_prog(self.cavity_charactaristic_function_decay_prog ,self.cavity_charactaristic_function_decay_params["N_avg"] )        
        # return self.plot_cavity_charactaristic_function_decay(**kwargs)

    def run_cavity_charactaristic_function_decay(self,normalize = True,is_remove= False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):
        #TODO orginize to complete and run sperate
        if 0:
            self.cavity_charactaristic_function_decay_results, _, self.charactaristic_function_decay_plot_normaliztion=self.run_for_phase(self.load_cavity_charactaristic_function_decay,var_name ='t',**kwargs)
        else:
            if not hasattr(self, 'cavity_charactaristic_function_decay_prog'): raise ValueError("Idiot! You did not write charactaristic_function_decay program")
         
            self.cavity_charactaristic_function_decay_results = dict()
            if is_remove:
                self.cavity_charactaristic_function_decay_results["I_orig"], self.cavity_charactaristic_function_decay_results["Q_orig"] = self.run_prog(self.cavity_charactaristic_function_decay_prog , self.cavity_charactaristic_function_decay_params["N_avg"],var = 't' )
             
                prog,_ = self.load_cavity_charactaristic_function_decay(**self.cavity_charactaristic_function_decay_params,is_remove = True,update_prog = False)
                _, self.cavity_charactaristic_function_decay_results["I_rev"], self.cavity_charactaristic_function_decay_results["Q_rev"] = self.run_prog(prog ,self.cavity_charactaristic_function_decay_params["N_avg"],var = 't')
                
                self.cavity_charactaristic_function_decay_results["I"] = self.cavity_charactaristic_function_decay_results["I_orig"]-self.cavity_charactaristic_function_decay_results["I_rev"]
                self.cavity_charactaristic_function_decay_results["Q"] = self.cavity_charactaristic_function_decay_results["Q_orig"]-self.cavity_charactaristic_function_decay_results["Q_rev"]
                plot_normaliztion = False
            else:
                self.cavity_charactaristic_function_decay_results["I"], self.cavity_charactaristic_function_decay_results["Q"] = self.run_prog(self.cavity_charactaristic_function_decay_prog ,self.cavity_charactaristic_function_decay_params["N_avg"],var = 't' )        
                # return self.cavity_charactaristic_function_decay_results["amp"], self.cavity_charactaristic_function_decay_results["I"], self.cavity_charactaristic_function_decay_results["Q"] 
                if normalize:
                    prog,_ = self.load_cavity_charactaristic_function_decay(**self.cavity_charactaristic_function_decay_params,normalize = True,update_prog = False)
                    self.cavity_charactaristic_function_decay_results =  self.normalize_CrossKerr(self.cavity_charactaristic_function_decay_params, self.cavity_charactaristic_function_decay_results,prog = prog,var = 't')
                else:
                    plot_normaliztion =False        
            self.cavity_charactaristic_function_decay_results["t"] = 4*np.array(self.cavity_charactaristic_function_decay_params["wt_time_list"])
            self.charactaristic_function_decay_plot_normaliztion = plot_normaliztion
            # return self.cavity_charactaristic_function_decay_results
        return self.plot_cavity_charactaristic_function_decay(**kwargs)


    
    def plot_cavity_charactaristic_function_decay(self,start_x_fit=None,plot_normaliztion=None,guess_Wb=1,guess_sqew= 0.01,guess_phase = 0, guess_T=None, guess_Delta = None,guess_linear = 0,guess_offset = 0.01,txt= '',**kwargs):
        plot_normaliztion = self.charactaristic_function_decay_plot_normaliztion if plot_normaliztion is None else plot_normaliztion
        # Check what's happening here:
        #           fit_and_plot(self,fittype, I, Q, ti, N_avg,  txt=None,title_str =None, fig_num=None ):  
        prog_name = 'charactaristic_function_decay EM'
        fig_num = prog_name + ' ' + str(self.next_fig_num_by_prog_name(prog_name))
        
        Data = self.process_data(self.cavity_charactaristic_function_decay_results,is_thresh = False)
        max_Data = np.max(Data[0])
        min_Data = np.min(Data[0])
        Data_process = (Data[0] - 0.5*(max_Data+min_Data))*2.0/(max_Data-min_Data)

        Data_process = np.arccos(Data_process*0.999999999)
        fit_processed,fit_process_fidelty = self.fit_and_plot( 'ExpCos', Data_process, ti = self.cavity_charactaristic_function_decay_results["t"]  , title_str = f"charactaristic_function_decay EM ({self.main_mem})",label = 'Fitted data', fig_num = None,x_type = 't',is_calc_stat_error = False, plot = False, **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        # fit_processed,fit_process_fidelty,process_fig = self.fit_and_plot( 'ExpCos', Data_process, ti = self.cavity_charactaristic_function_decay_results["t"]  , title_str = f"charactaristic_function_decay EM ({self.main_mem})",label = 'Fitted data', fig_num = None,x_type = 't',is_calc_stat_error = False, plot = True, **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        
        # curve_fit(func, xdata = time, ydata = trace, p0 = guess, sigma = stat_error)

        fig=self.fit_and_plot( None ,self.process_data(self.cavity_charactaristic_function_decay_results), ti = self.cavity_charactaristic_function_decay_results["t"]  , title_str = f"charactaristic_function_decay EM ({self.main_mem} Raw)",label = 'Fitted data', fig_num = fig_num,x_type = 't', **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        
        if guess_T     is None: guess_T = fit_processed[0]
        if guess_Delta is None: guess_Delta = fit_processed[1]/2
        guess_Amp = max_Data if abs(max_Data) >abs(min_Data) else min_Data
        guess_Const = 0
        
        guess = [guess_Delta,guess_T,guess_phase,guess_sqew,guess_Amp,guess_Const,guess_Wb]
        try: 
            print(guess)
            is_succeed = True
        # return self.autoscale_data(self.cavity_charactaristic_function_decay_results["t"]*1e-9)[0],self.process_data(self.cavity_charactaristic_function_decay_results)
            fit,cov= curve_fit(self.coherent_decay_func_Real, xdata = self.cavity_charactaristic_function_decay_results["t"]*1e-3, ydata = self.process_data(self.cavity_charactaristic_function_decay_results)[0], p0 = guess)
            fit_fidelty = np.sqrt(np.diag(cov))
        except: 
            is_succeed = False
            print('***************************\n Could not find fit \n***************************')
            
            fit = [0,[0]]
        print(fit)
        if not is_succeed:
            _,_,process_fig = self.fit_and_plot( 'ExpCos', Data_process, ti = self.cavity_charactaristic_function_decay_results["t"]  , title_str = f"charactaristic_function_decay EM ({self.main_mem})",label = 'Fitted data', fig_num = None,x_type = 't',is_calc_stat_error = False, plot = True, **kwargs )
            # txt = txt + r'$| \beta |$' +r' =  ({0} $\pm$ {1})/$| \alpha |$ '.format(*round_value_by_error(abs(fit[0]),fit_fidelty[0]))
        else:            
            txt = txt + r'Decay time' +r' =  {0} $\pm$ {1} [$\mu$s] '.format(*round_value_by_error(fit[1],fit_fidelty[1]))
            txt = txt + '\n'+ r'frequencey' +r' =  {0} $\pm$ {1} [MHz] '.format(*round_value_by_error(fit[0],fit_fidelty[0]))
            txt = txt + '\n'+ r'Detuning' +r' =   {0} [MHz] '.format(self.cavity_charactaristic_function_decay_params["Cavity_detuning"]*1e3)
            print(txt)
            # ann = plt.annotate( r'decay time =  {0} $\pm$ {1} [us]'.format(*round_value_by_error(fit[2],fit_fidelty[2]), xy = (x_text,y_text), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 

        if not txt == '':
            ann = plt.annotate(txt, xy = (0,0), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
            ann.draggable()
        if plot_normaliztion:
            plt.plot(self.autoscale_data(self.cavity_charactaristic_function_decay_results["t"]*1e-9)[0], self.autoscale_data(self.process_data(data = [self.cavity_charactaristic_function_decay_results["I_orig"], self.cavity_charactaristic_function_decay_results["Q_orig"]])[0])[0],'+',label = 'Measured Data')
            plt.plot(self.autoscale_data(self.cavity_charactaristic_function_decay_results["t"]*1e-9)[0], self.autoscale_data(self.process_data(data = [self.cavity_charactaristic_function_decay_results["I_CrossKerr"], self.cavity_charactaristic_function_decay_results["Q_CrossKerr"]], is_calc_stat_error = False, is_thresh = False))[0],'o', label = 'Cross Kerr')
            leg = plt.legend()
            leg.set_draggable(True)
        # plt.figure()
        # y_fit = self.autoscale_data(self.coherent_decay_func_Real(self.cavity_charactaristic_function_decay_results["t"]*1e-3,*guess))[0]
        y_fit = self.autoscale_data(self.coherent_decay_func_Real(self.cavity_charactaristic_function_decay_results["t"]*1e-3,*fit))[0]
        # x_fit = self.autoscale_data(self.cavity_charactaristic_function_decay_results["t"]*1e-9)[0]                         
        x_fit = self.cavity_charactaristic_function_decay_results["t"]*1e-3                    
        plt.plot(x_fit,y_fit)
        plt.xlabel(r'time [$\mu$s]')
        #fig.show()
        plt.tight_layout()
        return fit,fit_fidelty
    def coherent_decay_func_Real(self,t,Delta,T,phase,sqew,Amp,Cons,Wb):
        return (Amp*np.cos(Wb*np.exp(-t/(2*T))*(np.cos(t*2*np.pi*Delta+phase)+sqew))+Cons)
    #%% new characteristic_function_tomography
    def load_characteristic_function_tomography(self, num_of_pnts = 51, # How many points to measure, .
                                    N_avg = 10 , # How many times each sequence is executed 
                                    seq = None,# a tuple for an inital pulse(pulse,element) or a function, function will receive **kwargs and 
                                    Displacement_pulse = None,
                                    # grid_size = (20,20), # for(n,m) grid size will be (2n+1)*(2m+1) 
                                    # step_size = 0.05,
                                    grid_X_list = None,
                                    grid_Y_list = None,
                                    is_two_mode_sweep = False,
                                    mem_mode=None,
                                    scnd_mem = None,
                                    start_with: tuple = None, #(self.pi_pulse,self.main_qubits)
                                    update_prog = True,
                                    pi2_phase = 0.5, #phase of second pi2 pulse in units of pi 
                                    normalize = False,
                                    qb_phase_correction = None,
                                    is_Yale= True,
                                    is_remove = False,
                                    is_phase = False, #are measuring the phase of the qubit b
                                    is_measure_Z = False,#only relevent when _is_calibrate_SigZ is True
                                    detuning = 0, 
                                    **kwargs):
        
        if qb_phase_correction is None: qb_phase_correction = self.ConDisp_correction
        if mem_mode is None: mem_mode =self.main_mem
        if scnd_mem is None: scnd_mem =self.scnd_mem
        
        if Displacement_pulse is None: Displacement_pulse = self.ConDisp1_pulse

        pi2_phase_list,qb_pulse_list =self.get_qb_measure_seq_list(pi2_phase=pi2_phase,is_remove= is_remove, is_phase = is_phase, is_measure_Z=is_measure_Z,normalize=normalize)
            
        if grid_X_list is None: grid_X_list = np.arange(-grid_size[0],grid_size[0]+1,1)*step_size 
        if grid_Y_list is None: grid_Y_list = np.arange(-grid_size[1],grid_size[1]+1,1)*step_size
        if max(np.abs(grid_X_list))>1.99 or  max(np.abs(grid_Y_list))>1.99:
            raise ValueError('QUA amp function hate me')

        rot_phase_list =  (grid_X_list**2)*0*detuning/2/np.pi 
        qb_phase_corr_list =0*grid_X_list**2*qb_phase_correction/2/np.pi
        warn( 'using only grid_X_list ')        
        
        rot_phase_list     = rot_phase_list.tolist()
        qb_phase_corr_list = qb_phase_corr_list.tolist()
        grid_Y_list        =  grid_Y_list.tolist()
        grid_X_list        =  grid_X_list.tolist()
        num_of_pnts        = len(grid_X_list)*len(grid_Y_list)
        # print(num_of_pnts)
        # ramsey_vaccum_N_avg = N_avg
       
        #Save parameters:
        characteristic_function_tomography_params = {}
        characteristic_function_tomography_params["N_avg"]              = N_avg
        characteristic_function_tomography_params["Displacement_pulse"] = Displacement_pulse
        characteristic_function_tomography_params["mem_mode"]           = mem_mode  
        characteristic_function_tomography_params["scnd_mem"]           = scnd_mem  
        characteristic_function_tomography_params["start_with"]         = start_with
        characteristic_function_tomography_params["is_Yale"]            = is_Yale
        characteristic_function_tomography_params["rot_phase_list"]     = rot_phase_list
        characteristic_function_tomography_params["grid_X_list"]        = grid_X_list
        characteristic_function_tomography_params["grid_Y_list"]        = grid_Y_list
        characteristic_function_tomography_params["detuning"]           = detuning
        characteristic_function_tomography_params["qb_phase_correction"]= qb_phase_correction
        characteristic_function_tomography_params["pi2_phase_list"]     = pi2_phase_list
        
        # characteristic_function_tomography_params["_is_cat_and_back"]    = _is_cat_and_back
        
        # characteristic_function_tomography_params["step_size_clks"]     = step_size_clks
        if update_prog: self.characteristic_function_tomography_params=characteristic_function_tomography_params

        assert  (not self.pulse_len(self.main_qubit,self.pi_pulse)%4), 'pi pulse length does not divide by 4'
        pi_wait_time = self.pulse_len(self.main_qubit,self.pi_pulse)//4
        #Calculate and show expeced runtime:
        try:
            run_time = len(pi2_phase_list)*N_avg*num_of_pnts*(self.pulse_len(self.main_readout,self.ro_pulse)+self.wait_between_seq * 4 +self.pulse_len(mem_mode,Displacement_pulse))
            print('Run time is {}s'.format(round(run_time * 1e-9)))         
        except:
            print('cant calculate time')
        print(pi2_phase_list)
        #Create program:
        with program() as prog:
            n             = declare(int)
            i             = declare(int)
            j             = declare(int)
            k             = declare(int)
            I             = declare(fixed)
            Q             = declare(fixed)
            I_st          = declare_stream()
            Q_st          = declare_stream()
            pi2_phase_arr = declare(fixed,value = pi2_phase_list)
            qb_pulse_arr  = declare(int,value =qb_pulse_list)
            
            rot_phase     = declare(fixed)
            rot_phase_arr = declare(fixed,value = rot_phase_list)
            qb_phasCor_arr= declare(fixed,value = qb_phase_corr_list)
            grid_X_arr = declare(fixed,value = grid_X_list)
            grid_Y_arr = declare(fixed,value = grid_Y_list)
            grid_X       = declare(fixed)
            grid_Y       = declare(fixed)
            qb_pulse      = declare(int)
            
            detuning      = declare(fixed)
            qb_phase_corr = declare(fixed)
            # qb_amp        = declare(fixed)
            # qb_amp_arr    = declare(fixed,value = pi_amp_mul_list)
            with for_(n,0,n<N_avg,n+1):
                with for_(k,0,k<len(grid_Y_list),k+1):
                    with for_(i,0,i<len(grid_X_list),i+1):
                        assign(grid_X,grid_X_arr[i])
                        assign(grid_Y,grid_Y_arr[k])
                        assign(detuning,rot_phase_arr[i])
                        assign(qb_phase_corr,qb_phasCor_arr[i])
                        with for_(j,0,j<pi2_phase_arr.length(),j+1):
                            assign(qb_pulse,qb_pulse_arr[j])
                            # assign(qb_amp, qb_amp_arr[j])
                            assign(rot_phase, detuning+pi2_phase_arr[j])
                                
                            reset_phase(self.main_qubit)
                            reset_phase(mem_mode)
                            # if self.is_two_mode_disp or self.is_two_mode_disp=='Sim': 
                            try:
                                reset_phase(scnd_mem)
                                reset_frame(mem_mode,scnd_mem,self.main_qubit)
                            # else:
                            except:
                                reset_frame(mem_mode,self.main_qubit)
                                if is_two_mode_sweep:
                                    raise ValueError('failed to reset phase of scnd mem')
                            
                            
                            if type(start_with) is tuple: 
                                play(*start_with)
                                align(start_with[1],self.main_qubit)
                            elif type(start_with) is types.FunctionType or type(start_with) is types.MethodType:
                                start_with(mem_mode = mem_mode,**kwargs)
                            else:
                                print('\n \n ******** \n no starting sequence \n ******** \n \n')
                            
                                            
                            play(self.pi2_pulse,self.main_qubit)
                                
                            # align(self.main_qubit, mem_mode)
                            if is_two_mode_sweep:
                                self._ConDisp_pulse(qubit = self.main_qubit, mem_mode = mem_mode,mem_mode2= scnd_mem, scale_amp =[grid_X,-grid_Y,grid_Y,grid_X],Displacement_pulse = Displacement_pulse,is_Yale = is_Yale,qb_phase_correction =qb_phase_corr,**kwargs) 
                            else:
                                self._ConDisp_pulse(qubit = self.main_qubit, mem_mode = mem_mode,mem_mode2= scnd_mem, scale_amp =[grid_X,-grid_Y,grid_Y,grid_X],Displacement_pulse = Displacement_pulse,is_Yale = is_Yale,qb_phase_correction =qb_phase_corr,**kwargs) 
                            # self._ConDisp_pulse(qubit = self.main_qubit, mem_mode = mem_mode,mem_mode2= scnd_mem, scale_amp =grid_X,Displacement_pulse = Displacement_pulse,is_Yale = is_Yale,qb_phase_correction =qb_phase_corr,**kwargs) 
                                                  
                            frame_rotation_2pi(rot_phase,self.main_qubit)
                            
                            align(self.main_qubit, mem_mode)
                            # play(self.pi2_pulse,self.main_qubit)
                            # frame_rotation_2pi(pi2_phase,self.main_qubit)
                            with switch_(qb_pulse,unsafe = True):
                                with case_(1): 
                                    play(self.pi2_pulse,self.main_qubit)
                                with case_(2): 
                                    play(self.pi_pulse,self.main_qubit)
                                with case_(0): 
                                    wait(pi_wait_time,self.main_qubit)
                            align(self.main_qubit, self.main_readout)
                            # self.perform_full_measurement(I,Q)#,I_output_name = I_st,Q_output_name = Q_st)
                            self.perform_full_measurement(I,Q,I_output_name = I_st,Q_output_name = Q_st)
                                
                                # save(grid_Y, 'grid_Y')
                            # save(grid_X, 'grid_X')
                    
            with stream_processing():
                I_st.save_all('I')
                Q_st.save_all('Q')

        if update_prog: self.characteristic_function_tomography_prog = prog
        # self.simulate_prog(prog)
        return prog,characteristic_function_tomography_params

    def run_characteristic_function_tomography(self,is_phase= True,#normalize = True,is_remove= False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):
        # self.run_grid_prog()
        self.characteristic_function_tomography_results,_ , self.characteristic_function_tomography_normaliztion=self.run_for_phase(self.load_characteristic_function_tomography,is_phase=is_phase,var_name =['grid_X','grid_Y'],is_grid = True,**kwargs)
        
        if is_phase: return self.plot_characteristic_function_tomography_phase(**kwargs)
        return self.plot_characteristic_function_tomography(**kwargs)
      



        
    def plot_characteristic_function_tomography_phase(self,results =None,params = None,fig_num= None,plot_normaliztion= None
                                        ,**kwargs):
        X_grid,Y_grid =np.meshgrid(self.characteristic_function_tomography_params['grid_X_list'],self.characteristic_function_tomography_params['grid_Y_list'])
        X_grid= np.array(self.characteristic_function_tomography_params['grid_X_list'])
        Y_grid =np.array(self.characteristic_function_tomography_params['grid_Y_list'])

        Data_Re = self.process_data(self.characteristic_function_tomography_results['Re'], is_calc_stat_error = False,is_thresh = False)
        print(len(Data_Re))
        Normalize = Data_Re[int(len(Data_Re)/2)]
        Data_Re = Data_Re/Normalize
        Data_Im = self.process_data(self.characteristic_function_tomography_results['Im'], is_calc_stat_error = False,is_thresh = False)/Normalize
        fig_real = self.plot_continuous_tomography(X_grid,Y_grid, Data_Re.reshape(*X_grid.shape,*Y_grid.shape),
                                        title_str = f"Real characteristic_function of ({self.main_mem})",
                                        fig_num = fig_num,**kwargs )
        self.plot_continuous_tomography(X_grid,Y_grid,Data_Im.reshape(*X_grid.shape,*Y_grid.shape),
                                        title_str = f"Imag characteristic_function of ({self.main_mem})",
                                        fig_num = fig_num,**kwargs )
        # self.fitNplot_phase(self.characteristic_function_tomography_results,
        #                            self.characteristic_function_tomography_params,ti= self.characteristic_function_tomography_results['Re']["var"], 
        #                            xlabel    = r'$amplitude$',x_type    = 'amp',
        #                            prog_name = f'charactristic function phase',
        #                            plot_normaliztion = self.characteristic_function_tomography_normaliztion,
        #                            fit_types =  ['Gaussian',None,None,'Gaussian'],**kwargs)


    #%% two modes two_modes_CharFunc_tomography
    def load_two_modes_CharFunc_tomography(self, num_of_pnts = 51, # How many points to measure, .
                                    N_avg = 10 , # How many times each sequence is executed 
                                    seq = None,# a tuple for an inital pulse(pulse,element) or a function, function will receive **kwargs and 
                                    Displacement_pulse = None,
                                    # grid_size = (20,20), # for(n,m) grid size will be (2n+1)*(2m+1) 
                                    # step_size = 0.05,
                                    grid_X_list = None,
                                    grid_Y_list = None,
                                    mem_mode=None,
                                    scnd_mem = None,
                                    start_with: tuple = None, #(self.pi_pulse,self.main_qubits)
                                    update_prog = True,
                                    pi2_phase = 0.5, #phase of second pi2 pulse in units of pi 
                                    normalize = False,
                                    qb_phase_correction = None,
                                    is_Yale= True,
                                    is_remove = False,
                                    is_phase = False, #are measuring the phase of the qubit b
                                    is_measure_Z = False,#only relevent when _is_calibrate_SigZ is True
                                    angle_1 = 0,
                                    angle_2 = None,
                                    detuning = 0, 
                                    is_two_mode_disp = True,
                                    **kwargs):
        
        if qb_phase_correction is None: qb_phase_correction = self.ConDisp_correction
        if mem_mode is None: mem_mode =self.main_mem
        if scnd_mem is None: scnd_mem =self.scnd_mem
        if angle_2 is None: angle_2 = angle_1
        
        warn('angle corection not yet apllied')
        
        if Displacement_pulse is None: Displacement_pulse = self.ConDisp1_pulse

        pi2_phase_list,qb_pulse_list =self.get_qb_measure_seq_list(pi2_phase=pi2_phase,is_remove= is_remove, is_phase = is_phase, is_measure_Z=is_measure_Z,normalize=normalize)
            
        if grid_X_list is None: grid_X_list = np.arange(-grid_size[0],grid_size[0]+1,1)*step_size 
        if grid_Y_list is None: grid_Y_list = np.arange(-grid_size[1],grid_size[1]+1,1)*step_size
        if max(np.abs(grid_X_list))>1.99 or  max(np.abs(grid_Y_list))>1.99:
            raise ValueError('QUA amp function hate me')

        rot_phase_list =  (grid_X_list**2)*0*detuning/2/np.pi 
        qb_phase_corr_list =0*grid_X_list**2*qb_phase_correction/2/np.pi
        warn( 'using only grid_X_list ')        
        
        rot_phase_list     = rot_phase_list.tolist()
        qb_phase_corr_list = qb_phase_corr_list.tolist()
        grid_Y_list        =  grid_Y_list.tolist()
        grid_X_list        =  grid_X_list.tolist()
        num_of_pnts        = len(grid_X_list)*len(grid_Y_list)
        # print(num_of_pnts)
        # ramsey_vaccum_N_avg = N_avg
       
        #Save parameters:
        two_modes_CharFunc_tomography_params = {}
        two_modes_CharFunc_tomography_params["N_avg"]              = N_avg
        two_modes_CharFunc_tomography_params["Displacement_pulse"] = Displacement_pulse
        two_modes_CharFunc_tomography_params["mem_mode"]           = mem_mode  
        two_modes_CharFunc_tomography_params["scnd_mem"]           = scnd_mem  
        two_modes_CharFunc_tomography_params["start_with"]         = start_with
        two_modes_CharFunc_tomography_params["is_Yale"]            = is_Yale
        two_modes_CharFunc_tomography_params["rot_phase_list"]     = rot_phase_list
        two_modes_CharFunc_tomography_params["grid_X_list"]        = grid_X_list
        two_modes_CharFunc_tomography_params["grid_Y_list"]        = grid_Y_list
        two_modes_CharFunc_tomography_params["detuning"]           = detuning
        two_modes_CharFunc_tomography_params["qb_phase_correction"]= qb_phase_correction
        two_modes_CharFunc_tomography_params["pi2_phase_list"]     = pi2_phase_list
        
        # two_modes_CharFunc_tomography_params["_is_cat_and_back"]    = _is_cat_and_back
        
        # two_modes_CharFunc_tomography_params["step_size_clks"]     = step_size_clks
        if update_prog: self.two_modes_CharFunc_tomography_params=two_modes_CharFunc_tomography_params

        assert  (not self.pulse_len(self.main_qubit,self.pi_pulse)%4), 'pi pulse length does not divide by 4'
        pi_wait_time = self.pulse_len(self.main_qubit,self.pi_pulse)//4
        #Calculate and show expeced runtime:
        try:
            run_time = len(pi2_phase_list)*N_avg*num_of_pnts*(self.pulse_len(self.main_readout,self.ro_pulse)+self.wait_between_seq * 4 +self.pulse_len(mem_mode,Displacement_pulse))
            print('Run time is {}s'.format(round(run_time * 1e-9)))         
        except:
            print('cant calculate time')
        print(pi2_phase_list)
        #Create program:
        with program() as prog:
            n             = declare(int)
            i             = declare(int)
            j             = declare(int)
            k             = declare(int)
            I             = declare(fixed)
            Q             = declare(fixed)
            I_st          = declare_stream()
            Q_st          = declare_stream()
            pi2_phase_arr = declare(fixed,value = pi2_phase_list)
            qb_pulse_arr  = declare(int,value =qb_pulse_list)
            
            rot_phase     = declare(fixed)
            rot_phase_arr = declare(fixed,value = rot_phase_list)
            qb_phasCor_arr= declare(fixed,value = qb_phase_corr_list)
            grid_X_arr = declare(fixed,value = grid_X_list)
            grid_Y_arr = declare(fixed,value = grid_Y_list)
            grid_X       = declare(fixed)
            grid_Y       = declare(fixed)
            qb_pulse      = declare(int)
            
            detuning      = declare(fixed)
            qb_phase_corr = declare(fixed)
            # qb_amp        = declare(fixed)
            # qb_amp_arr    = declare(fixed,value = pi_amp_mul_list)
            with for_(n,0,n<N_avg,n+1):
                with for_(k,0,k<len(grid_Y_list),k+1):
                    with for_(i,0,i<len(grid_X_list),i+1):
                        assign(grid_X,grid_X_arr[i])
                        assign(grid_Y,grid_Y_arr[k])
                        assign(detuning,rot_phase_arr[i])
                        assign(qb_phase_corr,qb_phasCor_arr[i])
                        with for_(j,0,j<pi2_phase_arr.length(),j+1):
                            assign(qb_pulse,qb_pulse_arr[j])
                            # assign(qb_amp, qb_amp_arr[j])
                            assign(rot_phase, detuning+pi2_phase_arr[j])
                                
                            reset_phase(self.main_qubit)
                            reset_phase(mem_mode)
                            # if self.is_two_mode_disp or self.is_two_mode_disp=='Sim': 
                            reset_phase(scnd_mem)
                            reset_frame(mem_mode,scnd_mem,self.main_qubit)
                        # else:
      
                            
                            if type(start_with) is tuple: 
                                play(*start_with)
                                align(start_with[1],self.main_qubit)
                            elif type(start_with) is types.FunctionType or type(start_with) is types.MethodType:
                                start_with(mem_mode = mem_mode,**kwargs)
                            else:
                                print('\n \n ******** \n no starting sequence \n ******** \n \n')
                            
                                            
                            play(self.pi2_pulse,self.main_qubit)
                                
                            # align(self.main_qubit, mem_mode)
                            self._ConDisp_pulse(qubit = self.main_qubit, mem_mode = mem_mode,mem_mode2= scnd_mem, scale_amp =grid_X,scale_amp2 = grid_Y,Displacement_pulse = Displacement_pulse,is_two_mode_disp = is_two_mode_disp,is_Yale = is_Yale,qb_phase_correction =qb_phase_corr,**kwargs) 
                            frame_rotation_2pi(rot_phase,self.main_qubit)
                            
                            align(self.main_qubit, mem_mode)
                            # play(self.pi2_pulse,self.main_qubit)
                            # frame_rotation_2pi(pi2_phase,self.main_qubit)
                            with switch_(qb_pulse,unsafe = True):
                                with case_(1): 
                                    play(self.pi2_pulse,self.main_qubit)
                                with case_(2): 
                                    play(self.pi_pulse,self.main_qubit)
                                with case_(0): 
                                    wait(pi_wait_time,self.main_qubit)
                            align(self.main_qubit, self.main_readout)
                            # self.perform_full_measurement(I,Q)#,I_output_name = I_st,Q_output_name = Q_st)
                            self.perform_full_measurement(I,Q,I_output_name = I_st,Q_output_name = Q_st)
                                
                                # save(grid_Y, 'grid_Y')
                            # save(grid_X, 'grid_X')
                    
            with stream_processing():
                I_st.save_all('I')
                Q_st.save_all('Q')

        if update_prog: self.two_modes_CharFunc_tomography_prog = prog
        # self.simulate_prog(prog)
        return prog,two_modes_CharFunc_tomography_params

    def run_two_modes_CharFunc_tomography(self,is_phase= True,#normalize = True,is_remove= False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):
        # self.run_grid_prog()
        self.two_modes_CharFunc_tomography_results,_ , self.two_modes_CharFunc_tomography_normaliztion=self.run_for_phase(self.load_two_modes_CharFunc_tomography,is_phase=is_phase,var_name =['grid_X','grid_Y'],is_grid = True,**kwargs)
        
        if is_phase: return self.plot_two_modes_CharFunc_tomography_phase(**kwargs)
        return self.plot_two_modes_CharFunc_tomography(**kwargs)
      



        
    def plot_two_modes_CharFunc_tomography_phase(self,results =None,params = None,fig_num= None,plot_normaliztion= None
                                        ,**kwargs):
        X_grid,Y_grid =np.meshgrid(self.two_modes_CharFunc_tomography_params['grid_X_list'],self.two_modes_CharFunc_tomography_params['grid_Y_list'])
        X_grid= np.array(self.two_modes_CharFunc_tomography_params['grid_X_list'])
        Y_grid =np.array(self.two_modes_CharFunc_tomography_params['grid_Y_list'])

        Data_Re = self.process_data(self.two_modes_CharFunc_tomography_results['Re'], is_calc_stat_error = False,is_thresh = False)
        # print(len(Data_Re))
        Normalize = np.max(Data_Re)
        Data_Re = Data_Re/Normalize
        Data_Im = self.process_data(self.two_modes_CharFunc_tomography_results['Im'], is_calc_stat_error = False,is_thresh = False)/Normalize
        fig_Re,ax_Re = self.plot_continuous_tomography(X_grid,Y_grid, Data_Re.reshape(*X_grid.shape,*Y_grid.shape),
                                        title_str = f"Real two_modes_CharFunc of ({self.main_mem})",
                                        fig_num = fig_num,**kwargs )
        ax_Re.set_xlabel(self.main_mem)
        ax_Re.set_ylabel(self.scnd_mem)
        fig_img,ax_img = self.plot_continuous_tomography(X_grid,Y_grid,Data_Im.reshape(*X_grid.shape,*Y_grid.shape),
                                        title_str = f"Imag two_modes_CharFunc of ({self.main_mem})",
                                        fig_num = fig_num,**kwargs )
        ax_img.set_xlabel(self.main_mem)
        ax_img.set_ylabel(self.scnd_mem)
        # self.fitNplot_phase(self.two_modes_CharFunc_tomography_results,
        #                            self.two_modes_CharFunc_tomography_params,ti= self.two_modes_CharFunc_tomography_results['Re']["var"], 
        #                            xlabel    = r'$amplitude$',x_type    = 'amp',
        #                            prog_name = f'charactristic function phase',
        #                            plot_normaliztion = self.two_modes_CharFunc_tomography_normaliztion,
        #                            fit_types =  ['Gaussian',None,None,'Gaussian'],**kwargs)
        return fig_Re,ax_Re,fig_img,ax_img 
#%% characteristic_function_tomography   
    def load_characteristic_function_tomography_old(self, N_avg =1000,
                                    Displacement_pulse = None,
                                    seq = None,# a tuple for an inital pulse(pulse,element) or a function, function will receive **kwargs and 
                                    grid_size = (20,20), # for(n,m) grid size will be (2n+1)*(2m+1) 
                                    step_size = 0.05,
                                    grid_X = None,
                                    grid_Y = None,
                                    mem_mode=None,
                                    update_prog = True,
                                    pi2_phase = 0.5, #phase of second pi2 pulse in units of pi 
                                    is_remove = False,
                                    normalize = False,
                                    is_Yale= True,**kwargs):
        
        if mem_mode is None: mem_mode =self.main_mem
        if Displacement_pulse is None: Displacement_pulse =  self.ConDisp1_pulse
        #TODO, add check overflow
        
        if grid_X is None: grid_X = np.arange(-grid_size[0],grid_size[0]+1,1)*step_size 
        if grid_Y is None: grid_Y = np.arange(-grid_size[1],grid_size[1]+1,1)*step_size
        if max(np.abs(grid_X))>1.99 or  max(np.abs(grid_Y))>1.99:
            raise ValueError('QUA amp function hate me')
        #check overflow
        if is_remove:
            pi2_phase = pi2_phase -0.5
            
        #Save parameters:
        params = {}
        params["N_avg"]              = N_avg
        params["grid_X"]             = grid_X#cartesan
        params["grid_Y"]             = grid_Y#cartesan
        params["Displacement_pulse"] = Displacement_pulse
        params["mem_mode"]           = mem_mode  
        params["seq"]                = seq
        params["is_Yale"]            = is_Yale
        params["pi2_phase"]          = pi2_phase

        # params["step_size_clks"]     = step_size_clks
        
        pi2_amp_mul = 0.0 if normalize else 1.0
        
        if update_prog: self.characteristic_function_params = params
            
        #Calculate and show expeced runtime:
        try:
            run_time = N_avg*len(grid_X)*len(grid_Y)*(self.pulse_len(self.main_readout,self.ro_pulse)+self.wait_between_seq * 4)
            print('Run time is {}s'.format(round(run_time * 1e-9)))         
        except:
            print('cant calculate time')
        
        #Create program:
        with program() as prog:
            n       = declare(int)
            Xval    = declare(fixed)
            Yval    = declare(fixed)
            I       = declare(fixed)
            Q       = declare(fixed)
            with for_(n,0,n<N_avg,n+1):
                with for_each_(Xval,grid_X.tolist()):
                    with for_each_(Yval,grid_Y.tolist()):
                        reset_frame(self.main_qubit,mem_mode)
                        reset_phase(self.main_qubit)
                        reset_phase(mem_mode)
                        
                        if type(seq) is tuple: 
                            play(*seq)
                            align(seq[1],self.main_qubit)
                        elif type(seq) is types.FunctionType:
                            seq(mem_mode = mem_mode,**kwargs)
                        else:
                            warn('characteristic_functioning nothing')
                        
                        
                        play(self.pi2_pulse,self.main_qubit)
                            
                        align(self.main_qubit, mem_mode)
                        
    
                        self._ConDisp_pulse(qubit = self.main_qubit, mem_mode = mem_mode, scale_amp =[Xval,-Yval,Yval,Xval],Displacement_pulse = Displacement_pulse,is_Yale = is_Yale) 
    
                        frame_rotation_2pi(pi2_phase,self.main_qubit)
                        
                        align(self.main_qubit, mem_mode)
                        # frame_rotation_2pi(pi2_phase,self.main_qubit)
                        play(self.pi2_pulse*amp(pi2_amp_mul),self.main_qubit)
    
                            # align(self.main_qubit,mem_mode)
                            # play(Displacement_pulse*amp(-Xval,-Yval,-Yval,-Xval),mem_mode)
                            
                        align(self.main_qubit, self.main_readout,mem_mode)
                        self.perform_full_measurement(I,Q)
        
        if update_prog: self.characteristic_function_prog = prog
        
        return prog,params
# self.perform_full_measurement(I,Q)
   
    # def characteristic_function_seq(self,DeltaT,Displacement_pulse,Scale_displacment,mem_mode):
    #     for s_disp in Scale_displacment:
    #         play(Displacement_pulse*amp(s_disp) , mem_mode )
       
    #     align(self.main_qubit, mem_mode)
    #     with if_(~(DeltaT==0)):
    #         wait(DeltaT, self.main_qubit)

    def run_characteristic_function_tomography_old(self,is_remove= False,normalize = False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):

        if not hasattr(self, 'characteristic_function_prog'): raise ValueError("Idiot! You did not write characteristic_function program")
     
        self.characteristic_function_results = dict()
        if is_remove:
            self.characteristic_function_results["I_orig"], self.characteristic_function_results["Q_orig"] = self.run_grid_prog(self.characteristic_function_prog,self.characteristic_function_params["N_avg"],len(self.characteristic_function_params['grid_X']),len(self.characteristic_function_params['grid_Y']) )        
         
            prog,_ = self.load_characteristic_function_tomography(**self.characteristic_function_params,is_remove = True,update_prog = False)
            self.characteristic_function_results["I_rev"], self.characteristic_function_results["Q_rev"] = self.run_grid_prog(prog ,self.characteristic_function_params["N_avg"],len(self.characteristic_function_params['grid_X']),len(self.characteristic_function_params['grid_Y']) )
            
            self.characteristic_function_results["I"] = self.characteristic_function_results["I_orig"]-self.characteristic_function_results["I_rev"]
            self.characteristic_function_results["Q"] = self.characteristic_function_results["Q_orig"]-self.characteristic_function_results["Q_rev"]
            plot_normaliztion = False
        else:
            if normalize:
                raise ValueError('requires changing to run grid prog') 
            self.characteristic_function_results["I"], self.characteristic_function_results["Q"] =  self.run_grid_prog(self.characteristic_function_prog,self.characteristic_function_params["N_avg"],len(self.characteristic_function_params['grid_X']),len(self.characteristic_function_params['grid_Y']) )        
            # return self.characteristic_function_results["amp"], self.characteristic_function_results["I"], self.characteristic_function_results["Q"] 
            if normalize:
                prog,_ = self.load_characteristic_function(**self.characteristic_function_params,normalize = True,update_prog = False)
                self.characteristic_function_results =  self.normalize_CrossKerr(self.characteristic_function_params, self.characteristic_function_results,prog = prog,var = 't')
            else:
                plot_normaliztion =False        
            
        self.characteristic_function_plot_normaliztion = plot_normaliztion
        return self.plot_characteristic_function_tomography(**kwargs)

    # def plot_cavity_tomograpy(self,)
    def plot_characteristic_function_tomography_old(self,title_str='',**kwargs):
        # plot_normaliztion = self.characteristic_function_plot_normaliztion if plot_normaliztion is None else plot_normaliztion
        # Check what's happening here:
        #           fit_and_plot(self,fittype, I, Q, ti, N_avg,  txt=None,title_str =None, fig_num=None ):  
        prog_name = 'characteristic_function tomography'
        fig_num = prog_name + ' ' + str(self.next_fig_num_by_prog_name(prog_name))
        ret = self.plot_continuous_tomography(self.characteristic_function_params['grid_X'],self.characteristic_function_params['grid_Y'],
                                              self.process_data(self.characteristic_function_results, is_calc_stat_error = False,is_thresh = False), 
                                              title_str = f"characteristic_function of ({self.main_mem})",
                                              fig_num = fig_num )
        return ret        

#%% disentagle 2 cats
    def load_cavity_disentagle_2cats(self, num_of_pnts = 51, # How many points to measure, .
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    scale_amp_list = None,
                                    Displacement_pulse_first = None,
                                    Displacement_pulse_second = None,
                                    mem_mode=None,
                                    mem_mode2=None,
                                    amp_first1 =None,
                                    amp_first2 = 1.0,
                                    amp_second1 = None,
                                    amp_second2 = 1.0,
                                    start_with: tuple = None, #(self.pi_pulse,self.main_qubits)
                                    update_prog = True,
                                    pi2_phase = 0.5, #phase of second pi2 pulse in units of pi 
                                    normalize = False,
                                    qb_phase_correction = None,
                                    is_Yale= False,
                                    is_remove = False,
                                    is_phase = False,
                                    _is_calibrate_SigZ=False,
                                    detuning = 0, #only relevent when _is_calibrate_SigZ is True
                                    **kwargs):
        
        if qb_phase_correction is None: qb_phase_correction = self.ConDisp_correction
        if mem_mode is None: mem_mode ='mm1'
        if mem_mode2 is None: mem_mode2 ='mm2'
        if Displacement_pulse_first is None: Displacement_pulse_first = self.ConDisp1_pulse
        if Displacement_pulse_second is None: Displacement_pulse_second = "VFast_ConDisp1"
        
        is_sweep_second =True
        if amp_first1 is None:
            if amp_second1 is None:
                amp_first1 =1.0
            else:
                is_sweep_second = False
            
        # max_seq_time = int(max_seq_clks * 4)                            
        
        
        if scale_amp_list is None: 
            if not (num_of_pnts%2): num_of_pnts =num_of_pnts+1
            scale_amp_list = np.linspace(-1.9,1.9,num_of_pnts,dtype=float)
      
        if type(scale_amp_list) == list: scale_amp_list= np.array(scale_amp_list)
        rot_phase_list =  (scale_amp_list**2)*detuning/2/np.pi + pi2_phase
        
        qb_phase_corr_list =scale_amp_list**2*qb_phase_correction/2/np.pi
        
        
        rot_phase_list     = rot_phase_list.tolist()
        scale_amp_list     = scale_amp_list.tolist()
        qb_phase_corr_list = qb_phase_corr_list.tolist()
        num_of_pnts = len(scale_amp_list)
        # disentagle_2cats_N_avg = N_avg
        dif_pulse1_time = self.pulse_len(mem_mode,Displacement_pulse_first)-self.pulse_len(mem_mode2,Displacement_pulse_first)
        dif_pulse2_time = self.pulse_len(mem_mode,Displacement_pulse_second)-self.pulse_len(mem_mode2,Displacement_pulse_second)
     
        #Save parameters:
        cavity_disentagle_2cats_params = {}
        cavity_disentagle_2cats_params["N_avg"]              = N_avg
        cavity_disentagle_2cats_params["num_of_pnts"]        = num_of_pnts
        if is_sweep_second:
            cavity_disentagle_2cats_params["Displacement_pulse"]       = Displacement_pulse_second
            cavity_disentagle_2cats_params["Displacement_pulse_other"] = Displacement_pulse_first
        else:
            cavity_disentagle_2cats_params["Displacement_pulse"]       = Displacement_pulse_first
            cavity_disentagle_2cats_params["Displacement_pulse_other"] = Displacement_pulse_second
        cavity_disentagle_2cats_params["mem_mode"]           = mem_mode  
        cavity_disentagle_2cats_params["mem_mode2"]           = mem_mode2
        cavity_disentagle_2cats_params["start_with"]         = start_with
        cavity_disentagle_2cats_params["scale_amp_list"]     = scale_amp_list
        cavity_disentagle_2cats_params["is_Yale"]            = is_Yale
        cavity_disentagle_2cats_params["pi2_phase"]          = pi2_phase
        cavity_disentagle_2cats_params["detuning"]           = detuning
        cavity_disentagle_2cats_params["qb_phase_correction"]= qb_phase_correction
        cavity_disentagle_2cats_params["dif_pulse1_time"]    = dif_pulse1_time
        cavity_disentagle_2cats_params["dif_pulse2_time"]    = dif_pulse2_time
        cavity_disentagle_2cats_params["amp_first1"]         = amp_first1
        cavity_disentagle_2cats_params["amp_first2"]         = amp_first2
        cavity_disentagle_2cats_params["amp_second1"]        = amp_second1
        cavity_disentagle_2cats_params["amp_second2"]        = amp_second2
        
        # cavity_disentagle_2cats_params["step_size_clks"]     = step_size_clks
        
        pi_amp_mul = 0.0 if normalize else 1.0
        
        if update_prog: self.cavity_disentagle_2cats_params=cavity_disentagle_2cats_params
        
   
        #Calculate and show expeced runtime:
        try:
            run_time = N_avg*num_of_pnts*(self.pulse_len(self.main_readout,self.ro_pulse)+self.wait_between_seq * 4 +self.pulse_len(mem_mode,Displacement_pulse_first))
            print('Run time is {}s'.format(round(run_time * 1e-9)))         
        except:
            print('cant calculate time')
        
        #Create program:
        with program() as prog:
            n             = declare(int)
            I             = declare(fixed)
            Q             = declare(fixed)
            scale_amp     = declare(fixed)
            rot_phase     = declare(fixed)
            qb_phase_corr = declare(fixed)
            with for_(n,0,n<N_avg,n+1):
                with for_each_((scale_amp,rot_phase,qb_phase_corr), (scale_amp_list,rot_phase_list,qb_phase_corr_list)):
                    reset_frame(mem_mode,mem_mode2,self.main_qubit)
                    
                    if type(start_with) is tuple: 
                        play(*start_with)
                        align(start_with[1],self.main_qubit)
                    elif type(start_with) is types.FunctionType:
                        start_with(mem_mode = mem_mode,**kwargs)
                    else:
                        warn('\n ******** \n no starting seqeunce \n ******** \n')
                    
                                                        
                    play(self.pi2_pulse,self.main_qubit)
                    align(self.main_qubit, mem_mode,mem_mode2)
                   
                    if is_sweep_second:
                        play(Displacement_pulse_first*amp(amp_first1) , mem_mode )
                    else:
                        play(Displacement_pulse_first*amp(scale_amp) , mem_mode )
                    play(Displacement_pulse_first*amp(amp_first2) , mem_mode2 )
                    if is_Yale:
                        align(self.main_qubit, mem_mode,mem_mode2)
                        self._pi_pulse()#self.main_qubit,**kwargs)
                        align(self.main_qubit, mem_mode,mem_mode2)
                        if is_sweep_second:
                            play(Displacement_pulse_first*amp(-amp_first1) , mem_mode )
                        else:
                            play(Displacement_pulse_first*amp(-scale_amp) , mem_mode )                       
                        play(Displacement_pulse_first*amp(-amp_first2) , mem_mode2 )
                    
                    align(self.main_qubit, mem_mode,mem_mode2)
                    play(self.pi2_pulse*amp(0,-1,1,0),self.main_qubit)
                    align(self.main_qubit, mem_mode,mem_mode2)
                        
                    play(Displacement_pulse_second*amp(0,-amp_second2,amp_second2,0) , mem_mode2 )
                    if is_sweep_second:
                        play(Displacement_pulse_second*amp(0,-scale_amp,scale_amp,0) , mem_mode )
                    else:
                        play(Displacement_pulse_second*amp(0,-amp_second1,amp_second1,0) , mem_mode )
                        
                    if is_Yale:
                        align(self.main_qubit, mem_mode,mem_mode2)
                        self._pi_pulse()#self.main_qubit,**kwargs)
                        align(self.main_qubit, mem_mode,mem_mode2)
                        play(Displacement_pulse_second*amp(0,amp_second2,-amp_second2,0) , mem_mode2)
                        if is_sweep_second:
                            play(Displacement_pulse_second*amp(0,scale_amp,-scale_amp,0) , mem_mode )
                        else:
                            play(Displacement_pulse_second*amp(0,amp_second1,-amp_second1,0) , mem_mode )
                            
                                
                    # align(self.main_qubit, mem_mode)
                    

                    frame_rotation_2pi(rot_phase,self.main_qubit)
                    
                    align(self.main_qubit, mem_mode,mem_mode2)
                    # frame_rotation_2pi(pi2_phase,self.main_qubit)
                    play(self.pi2_pulse*amp(pi_amp_mul),self.main_qubit)
                    
                    align(self.main_qubit, self.main_readout)
                    self.perform_full_measurement(I,Q)

                    save(scale_amp, 'scale_amp')
        
        if update_prog: self.cavity_disentagle_2cats_prog = prog

        return prog,cavity_disentagle_2cats_params

    def run_cavity_disentagle_2cats(self,is_phase= True,#normalize = True,is_remove= False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):
        if is_phase:
            self.cavity_disentagle_2cats_phase_results,_ , self.cavity_disentagle_2cats_plot_normaliztion=self.run_for_phase(self.load_cavity_disentagle_2cats,var_name ='scale_amp',**kwargs)
            return self.plot_cavity_disentagle_2cats_phase(**kwargs)
      
        self.cavity_disentagle_2cats_results,_ , self.cavity_disentagle_2cats_plot_normaliztion=self.run_and_fix(self.load_cavity_disentagle_2cats,var_name ='scale_amp',**kwargs)
        return self.plot_cavity_disentagle_2cats(**kwargs)


        
    def plot_cavity_disentagle_2cats_phase(self,results =None,params = None,fig_num= None,plot_normaliztion= None
                                        ,**kwargs):
        return self.fitNplot_phase(self.cavity_disentagle_2cats_phase_results,
                                   self.cavity_disentagle_2cats_params,ti= self.cavity_disentagle_2cats_phase_results['Re']["var"], 
                                   xlabel    = r'$amplitude$',x_type    = 'amp',
                                   prog_name = f'disentagle 2 cats phase',
                                   plot_normaliztion = self.cavity_disentagle_2cats_plot_normaliztion,
                                   fit_types =  [None,None,None,None],**kwargs)

    # def run_cavity_disentagle_2cats(self,normalize = True,is_remove= False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
    #                              **kwargs):
    #     #TODO orginize to complete and run sperate

    #     if not hasattr(self, 'cavity_disentagle_2cats_prog'): raise ValueError("Idiot! You did not write disentagle_2cats program")
     
    #     self.cavity_disentagle_2cats_results = dict()
    #     if is_remove:
    #         self.cavity_disentagle_2cats_results["amp"], self.cavity_disentagle_2cats_results["I_orig"], self.cavity_disentagle_2cats_results["Q_orig"] = self.run_prog(self.cavity_disentagle_2cats_prog , self.cavity_disentagle_2cats_params["N_avg"],var = 'scale_amp' )
         
    #         prog,_ = self.load_cavity_disentagle_2cats(**self.cavity_disentagle_2cats_params,is_remove = True,update_prog = False)
    #         _, self.cavity_disentagle_2cats_results["I_rev"], self.cavity_disentagle_2cats_results["Q_rev"] = self.run_prog(prog ,self.cavity_disentagle_2cats_params["N_avg"],var = 'scale_amp' )
            
    #         self.cavity_disentagle_2cats_results["I"] = self.cavity_disentagle_2cats_results["I_orig"]-self.cavity_disentagle_2cats_results["I_rev"]
    #         self.cavity_disentagle_2cats_results["Q"] = self.cavity_disentagle_2cats_results["Q_orig"]-self.cavity_disentagle_2cats_results["Q_rev"]
    #         plot_normaliztion = False
    #     else:
    #         self.cavity_disentagle_2cats_results["amp"], self.cavity_disentagle_2cats_results["I"], self.cavity_disentagle_2cats_results["Q"] = self.run_prog(self.cavity_disentagle_2cats_prog ,self.cavity_disentagle_2cats_params["N_avg"],var = 'scale_amp' )        
    #         # return self.cavity_disentagle_2cats_results["amp"], self.cavity_disentagle_2cats_results["I"], self.cavity_disentagle_2cats_results["Q"] 
    #         if normalize:
    #             prog,_ = self.load_cavity_disentagle_2cats(**self.cavity_disentagle_2cats_params,normalize = True,update_prog = False)
    #             self.cavity_disentagle_2cats_results =  self.normalize_CrossKerr(self.cavity_disentagle_2cats_params, self.cavity_disentagle_2cats_results,prog = prog,var = 'scale_amp')
    #         else:
    #             plot_normaliztion =False        
        
    #     self.disentagle_2cats_plot_normaliztion = plot_normaliztion
    #     # return self.cavity_disentagle_2cats_results
    #     return self.plot_cavity_disentagle_2cats(**kwargs)


    
    def plot_cavity_disentagle_2cats(self,start_x_fit=None,plot_normaliztion=None,txt= '',fig_num = None,**kwargs):
        plot_normaliztion = self.disentagle_2cats_plot_normaliztion if plot_normaliztion is None else plot_normaliztion
        # Check what's happening here:
        #           fit_and_plot(self,fittype, I, Q, ti, N_avg,  txt=None,title_str =None, fig_num=None ):  
        prog_name = 'disentagle_2cats EM'
        if fig_num is None: fig_num = prog_name + ' ' + str(self.next_fig_num_by_prog_name(prog_name))
        
        plt.xlabel('amplitude scaling')
        
        fig, fit,fit_fidelty, x_text,y_text = self.fit_and_plot( 'Gaussian' , self.process_data(self.cavity_disentagle_2cats_results), ti = self.cavity_disentagle_2cats_results["amp"]  , title_str = f"disentagle_2cats EM ({self.main_mem})",label = 'Fitted data', fig_num = fig_num,x_type = 'amp', **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        # ret  p=lt.plot( self.autoscale_data(self.process_data(self.cavity_disentagle_2cats_results))#, ti = self.cavity_disentagle_2cats_results["amp"]  , title_str = f"disentagle_2cats EM ({self.main_mem})",label = 'Fitted data', fig_num = fig_num, **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        if plot_normaliztion:
            plt.plot(self.autoscale_data(self.cavity_disentagle_2cats_results["amp"])[0], self.autoscale_data(self.process_data(data = [self.cavity_disentagle_2cats_results["I_orig"], self.cavity_disentagle_2cats_results["Q_orig"]])[0])[0],'+',label = 'Measured Data')
            plt.plot(self.autoscale_data(self.cavity_disentagle_2cats_results["amp"])[0], self.autoscale_data(self.process_data(data = [self.cavity_disentagle_2cats_results["I_CrossKerr"], self.cavity_disentagle_2cats_results["Q_CrossKerr"]], is_calc_stat_error = False, is_thresh = False))[0],'o', label = 'Cross Kerr')
            leg = plt.legend()
            leg.set_draggable(True)
        if not fit is None:
            txt = txt + r'$| \alpha |$' +' =  {0}'.format(np.round(abs(1/fit[2]),4))
            # ann = plt.annotate( r'decay time =  {0} $\pm$ {1} [us]'.format(*round_value_by_error(fit[2],fit_fidelty[2]), xy = (x_text,y_text), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
        if not txt == '':
            ann = plt.annotate(txt, xy = (self.autoscale_data(self.cavity_disentagle_2cats_results["amp"])[0][0]/1.5,y_text/1.25), fontsize=self.text_size, horizontalalignment='center', verticalalignment ='top') 
            ann.draggable()
        
        plt.title(r'$\sigma $ = {0} amp = {1}'.format(self.pulse_sig(self.cavity_disentagle_2cats_params["mem_mode"] ,self.cavity_disentagle_2cats_params["Displacement_pulse_first"] ),self.pulse_amp(self.cavity_disentagle_2cats_params["mem_mode"] , self.cavity_disentagle_2cats_params["Displacement_pulse_first"] )))
        #fig.show()
        plt.tight_layout()
        return fig,fit,fit_fidelty
    #%% Cavity T1 and displacment calibration
    
    def load_cavity_T1_measurement(self, num_of_pnts = 50, # How many points to measure, .
                                    max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    Displacement_pulse = None,
                                    mem_mode=None,
                                    start_with: tuple = None, #(self.pi_pulse,self.main_qubits)
                                    update_prog = True,
                                    normalize = False,
                                    **kwargs):
        
        if mem_mode is None: mem_mode =self.main_mem
        if Displacement_pulse is None: Displacement_pulse = self.UnDisp1_pulse

        # max_seq_time = int(max_seq_clks * 4)                            
        max_seq_clks = int(max_seq_time // 4)
        step_size_clks =  max_seq_clks // num_of_pnts
        # T1_N_avg = N_avg

        self.cavity_T1_results = dict()
        #Save parameters:
        cavity_T1_params = {}
        cavity_T1_params["N_avg"]              = N_avg
        cavity_T1_params["max_seq_time"]       = max_seq_time
        cavity_T1_params["num_of_pnts"]        = num_of_pnts
        cavity_T1_params["Displacement_pulse"] = Displacement_pulse
        cavity_T1_params["mem_mode"]           = mem_mode  
        cavity_T1_params["start_with"]         = start_with
        cavity_T1_params["start_with"]         = start_with
        self.cavity_T1_results["t"]         = np.linspace(0, max_seq_time, num_of_pnts)

        # cavity_T1_params["step_size_clks"]     = step_size_clks
        
        pi_amp_mul = 0.0 if normalize else 1.0
        
        if update_prog: self.cavity_T1_params=cavity_T1_params
            
        #Calculate and show expeced runtime:
        try:
            run_time = N_avg*num_of_pnts*(max_seq_time/2+self.pulse_len(self.main_readout,self.ro_pulse)+self.wait_between_seq * 4 +self.pulse_len(mem_mode,Displacement_pulse))
            print('Run time is {}s'.format(round(run_time * 1e-9)))         
        except:
            print('cant calculate time')
        
        #Create program:
        with program() as prog:
            DeltaT  = declare(int)
            n       = declare(int)
            I       = declare(fixed)
            Q       = declare(fixed)
            with for_(n,0,n<N_avg,n+1):
                with for_(DeltaT , step_size_clks , DeltaT<=max_seq_clks , DeltaT + step_size_clks):
                    
                    if not start_with is None: 
                        play(*start_with)
                        align(start_with[1],mem_mode)
                    # Play Displacement:
                    play(Displacement_pulse, mem_mode )
                       
                    align(self.main_qubit, mem_mode)
                    
                    if not start_with is None: 
                        play(*start_with)
                        warn('this start with is not at the begining')
                        
                    self.cavity2qubit_0_mapping(I,Q, pi_amp_mul = pi_amp_mul, **kwargs )

                    save(DeltaT, 't')
        
        if update_prog: self.cavity_T1_prog = prog
        
        return prog,cavity_T1_params
# self.perform_full_measurement(I,Q)
   
    # def cavity_T1_seq(self,DeltaT,Displacement_pulse,Scale_displacment,mem_mode):
    #     for s_disp in Scale_displacment:
    #         play(Displacement_pulse*amp(s_disp) , mem_mode )
       
    #     align(self.main_qubit, mem_mode)
    #     with if_(~(DeltaT==0)):
    #         wait(DeltaT, self.main_qubit)
                
    def complete_cavity_T1_measurement(self,num_of_pnts = 50, # How many points to measure, .
                                    max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    normalize = False,#if normalize is true the program will run 
                                    just_run=True,
                                    **kwargs):
       #TODO finish this to work with normalize
        self.cavity_T1_results["t"], self.cavity_T1_results["I"], self.cavity_T1_results["Q"] = self.run_prog(self.cavity_T1_prog ,self.cavity_T1_params["N_avg"] )        
        
        self.cavity_T1_results["t"], self.cavity_T1_results["I"], self.cavity_T1_results["Q"] = self.run_prog(self.cavity_T1_prog ,self.cavity_T1_params["N_avg"] )        
        return self.plot_cavity_T1_measurement(**kwargs)

    def run_cavity_T1_measurement(self,normalize = True,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):
        #TODO orginize to complete and run sperate

        if not hasattr(self, 'cavity_T1_prog'): raise ValueError("Idiot! You did not write T1 program")
     
        # self.cavity_T1_results = dict()
        self.cavity_T1_results["I"], self.cavity_T1_results["Q"] = self.run_prog(self.cavity_T1_prog ,self.cavity_T1_params["N_avg"] )        
    
        if normalize:
            prog,_ = self.load_cavity_T1_measurement(**self.cavity_T1_params,normalize = True,update_prog = False)
            self.cavity_T1_results =  self.normalize_CrossKerr(self.cavity_T1_params, self.cavity_T1_results,prog = prog)
        else:
            plot_normaliztion =False        
        self.T1_plot_normaliztion = plot_normaliztion
        return self.plot_cavity_T1_measurement(**kwargs)


    
    def plot_cavity_T1_measurement(self,start_x_fit=None,plot_normaliztion=None,**kwargs):
        plot_normaliztion = self.T1_plot_normaliztion if plot_normaliztion is None else plot_normaliztion
        # Check what's happening here:
        #           fit_and_plot(self,fittype, I, Q, ti, N_avg,  txt=None,title_str =None, fig_num=None ):  
        prog_name = 'T1 EM'
        fig_num = next_fig_num_by_name(prog_name)
        ret = self.fit_and_plot( 'Exp(Exp)' , self.process_data(self.cavity_T1_results), ti = self.cavity_T1_results["t"]  , title_str = f"T1 EM ({self.main_mem})",label = 'Fitted data', fig_num = fig_num, **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        if plot_normaliztion:
            plt.plot(self.autoscale_data(self.cavity_T1_results["t"]*1e-9)[0], self.autoscale_data(self.process_data(data = [self.cavity_T1_results["I_orig"], self.cavity_T1_results["Q_orig"]])[0])[0],'+',label = 'Measured Data')
            plt.plot(self.autoscale_data(self.cavity_T1_results["t"]*1e-9)[0], self.autoscale_data(self.process_data(data = [self.cavity_T1_results["I_CrossKerr"], self.cavity_T1_results["Q_CrossKerr"]], is_calc_stat_error = False, is_thresh = False))[0],'o', label = 'Cross Kerr')
            leg = plt.legend()
            leg.set_draggable(True)
        return ret
    #%% Cavity T2 Ramsey
    
    def load_cavity_ramsey_measurement(self, num_of_pnts = 50, # How many points to measure, .
                                    max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    Rotation_pulse = None,
                                    Displacement_pulse = None,
                                    mem_mode = None,
                                    Scale_displacment  = [1.0],
                                    detuning = 0, #in GHz
                                    start_with: tuple = None, #(self.pi_pulse,self.main_qubits)
                                    update_prog = True,
                                    normalize = False):
        
        if Rotation_pulse is None:     Rotation_pulse = self.ConRotPi_pulse
        if mem_mode is None:           mem_mode =self.main_mem
        if Displacement_pulse is None: Displacement_pulse = self.UnDisp1_pulse
        
        pi_amp_mul = 0.0 if normalize else 1.0

        # max_seq_time = int(max_seq_clks * 4)                            
        max_seq_clks = int(max_seq_time // 4)
        step_size_clks =  max_seq_clks // num_of_pnts
        # T1_N_avg = N_avg
       
        try:
            run_time = N_avg*num_of_pnts*(max_seq_time/2+self.pulse_len(self.main_readout,self.ro_pulse)+self.wait_between_seq * 4 +self.pulse_len(self.main_qubit,Rotation_pulse))
            print('Run time is {}s'.format(round(run_time * 1e-9)))         
        except:
            print('cant calculate time')
       
        #Save parameters:
        params = {}
        params["N_avg"]              = N_avg
        params["max_seq_time"]       = max_seq_time
        params["num_of_pnts"]        = num_of_pnts
        params["detuning"]           = detuning
        params["Displacement_pulse"] = Displacement_pulse
        params['Rotation_pulse']     = Rotation_pulse
        params["Scale_displacment"]  = Scale_displacment 
        params["mem_mode"]           = mem_mode 
        params["start_with"]         = start_with
                # params["step_size_clks"]     = step_size_clks

        if update_prog: self.cavity_ramsey_params = params
        
        # ti_list = []; phase_list = []
        # for wt_time in np.linspace(step_size_clks, max_seq_clks, num_of_pnts):
        #     ti_list.append(int(0.5 * wt_time)); A1_list.append(float(np.cos(detuning * wt_time * 4 * 2 *np.pi)));
        # #Create program:
        if 1:
            with program() as prog:
                n       = declare(int)
                I       = declare(fixed)
                Q       = declare(fixed)
                DeltaT = declare(int)
    
                with for_(n,0,n<N_avg,n+1):
                    with for_(DeltaT , 0 , DeltaT<=max_seq_clks , DeltaT + step_size_clks):          
                        if not start_with is None: 
                            play(*start_with)
                            align(start_with[1],mem_mode)
                        for s_disp in Scale_displacment:
                            play(Displacement_pulse*amp(s_disp) , mem_mode )                        # frame_rotation(1*np.pi,mem_mode)
                   
                        frame_rotation(detuning*2*np.pi*Cast.to_fixed(DeltaT),mem_mode)
                        
                        with if_(~(DeltaT ==0)):
                            wait(DeltaT, mem_mode)
                        
                        for s_disp in Scale_displacment:
                            play(Displacement_pulse*amp(-1.0*s_disp) , mem_mode )
                                   
                        self.cavity2qubit_0_mapping(I,Q,pi_amp_mul = pi_amp_mul )
                       
                        save(DeltaT, 't')
                    
        if update_prog: self.cavity_ramsey_prog = prog
        
        return prog,params
    # def cavity_ramsey_seq(self,DeltaT,Displacement_pulse,Scale_displacment,mem_mode):
    #     for s_disp in Scale_displacment:
    #         play(Displacement_pulse*amp(s_disp) , mem_mode )
       
    #     align(self.main_qubit, mem_mode)
    #     with if_(~(DeltaT==0)):
    #         wait(DeltaT, self.main_qubit)

 # self.perform_full_measurement(I,Q)
    def complete_cavity_ramsey_measurement(self,num_of_pnts = 50, # How many points to measure, .
                                    max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    normalize = True,#if normalize is true the program will run 
                                    just_run=True,
                                    **kwargs):
       #TODO finish this to work with normalize
        self.cavity_ramsey_results["t"], self.cavity_ramsey_results["I"], self.cavity_ramsey_results["Q"] = self.run_prog(self.cavity_ramsey_prog ,self.cavity_ramsey_params["N_avg"] )        
        
        self.cavity_ramsey_results["t"], self.cavity_ramsey_results["I"], self.cavity_ramsey_results["Q"] = self.run_prog(self.cavity_ramsey_prog ,self.cavity_ramsey_params["N_avg"] )        
        return self.plot_cavity_ramsey_measurement(**kwargs)

    def run_cavity_ramsey_measurement(self,normalize = False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):
        #TODO orginize to complete and run sperate

        if not hasattr(self, 'cavity_ramsey_prog'): raise ValueError("Idiot! You did not write cavity_ramsey_prog program")
     
        self.cavity_ramsey_results = dict()
        self.cavity_ramsey_results["t"], self.cavity_ramsey_results["I"], self.cavity_ramsey_results["Q"] = self.run_prog(self.cavity_ramsey_prog ,self.cavity_ramsey_params["N_avg"] )        
    
        if normalize:
            prog,_ = self.load_cavity_ramsey_measurement(**self.cavity_ramsey_params,normalize = True,update_prog = False)
            self.cavity_ramsey_results =  self.normalize_CrossKerr(self.cavity_ramsey_params, self.cavity_ramsey_results,prog = prog)
        else:
            plot_normaliztion = False        
        self.ramsey_plot_normaliztion = plot_normaliztion
        return self.plot_cavity_ramsey_measurement(**kwargs)

    def plot_cavity_ramsey_measurement(self,start_x_fit=None,plot_normaliztion=None,**kwargs):
        plot_normaliztion = self.ramsey_plot_normaliztion if plot_normaliztion is None else plot_normaliztion
        # Check what's happening here:
        #           fit_and_plot(self,fittype, I, Q, ti, N_avg,  txt=None,title_str =None, fig_num=None ):  
        prog_name = 'Ramsey EM'
        fig_num = prog_name + ' ' + str(self.next_fig_num_by_prog_name(prog_name))
        ret = self.fit_and_plot( None, self.process_data(self.cavity_ramsey_results), ti = self.cavity_ramsey_results["t"]  , title_str = f"Ramsey EM ({self.main_mem})",label = 'Fitted data', fig_num = fig_num, **kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        if plot_normaliztion:
            plt.plot(self.autoscale_data(self.cavity_ramsey_results["t"]*1e-9)[0], self.autoscale_data(self.process_data(data = [self.cavity_ramsey_results["I_orig"], self.cavity_ramsey_results["Q_orig"]])[0])[0],'+',label = 'Measured Data')
            plt.plot(self.autoscale_data(self.cavity_ramsey_results["t"]*1e-9)[0], self.autoscale_data(self.process_data(data = [self.cavity_ramsey_results["I_CrossKerr"], self.cavity_ramsey_results["Q_CrossKerr"]], is_calc_stat_error = False, is_thresh = False))[0],'o', label = 'Cross Kerr')
            leg = plt.legend()
            leg.set_draggable(True)
        return ret
#%% Cavity  State revival through Qubit ramsey interfometery

#%% Number splliting spectrocsopy
    def load_number_splitting_spectroscopy(self, Delta_freq_list = np.linspace(-2e6,2e6,101), # How many points to measure, .
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    rotation_pulse = None,
                                    Scale_displacment =[1.0],
                                    Displacement_pulse = None,
                                    mem_mode=None,
                                    is_sweep_EM = False,
                                    IF_list = None,
                                    start_with: tuple = None, #(self.pi_pulse,self.main_qubits)
                                    normalize = False,
                                    update_prog = True,**kwargs):
        
        if rotation_pulse is None:     rotation_pulse = self.ConRotPi_pulse
        if mem_mode is None:           mem_mode =self.main_mem
        if Displacement_pulse is None:
            Displacement_pulse = self.ConDisp1_pulse if is_sweep_EM  else self.UnDisp1_pulse 
        
        if is_sweep_EM:
            IF_list     = Delta_freq_list + self.element_IF(mem_mode)
        else:
            IF_list     = Delta_freq_list + self.element_IF(self.main_qubit)
        
        num_of_pnts = len(IF_list)
        
        #Save parameters:
        params = {}
        params["N_avg"]              = N_avg
        params["Delta_freq_list"]    = Delta_freq_list
        params["IF_list"]            = IF_list
        # params["num_of_pnts"]        = num_of_pnts
        params["rotation_pulse"]     = rotation_pulse
        params["Displacement_pulse"] = Displacement_pulse
        params["mem_mode"]           = mem_mode  
        params["is_sweep_EM"]        = is_sweep_EM
        params["start_with"]         = start_with

        pi_amp_mul = 0.0 if normalize else 1.0

        if update_prog: self.spec_number_splitting_params= params
        
        try:
            run_time = N_avg*num_of_pnts*(self.pulse_len(self.main_readout,self.ro_pulse)+self.wait_between_seq * 4)
            print('Run time is {}s'.format(round(run_time * 1e-9)))         
        except:
            print('cant calculate time')
        
        #Create program:
        with program() as prog:
            n       = declare(int)
            I       = declare(fixed)
            Q       = declare(fixed)
            freq    = declare(int)
            with for_(n,0,n<N_avg,n+1):
                with for_each_(freq , IF_list.astype(int).tolist()):
                    
                    if not start_with is None: 
                        play(*start_with)
                        align(start_with[1],mem_mode)
                    
                    if is_sweep_EM:
                        update_frequency(mem_mode,freq)
                    else:
                        update_frequency(self.main_qubit,freq)
                        align(self.main_qubit, mem_mode)
                    
                    for s_disp in Scale_displacment:
                        play(Displacement_pulse*amp(s_disp) , mem_mode )  
                    align(self.main_qubit, mem_mode)
                    
                    if not start_with is None: 
                        play(*start_with)
                        align(start_with[1],mem_mode)
                                  
                    self.cavity2qubit_0_mapping(I,Q,mem_mode = mem_mode,pi_amp_mul = pi_amp_mul )

                    save(freq, 'freq')
        
        if update_prog: self.spec_number_splitting_prog = prog
        
        return prog,params
#TODO preform the sweep EM mode properly, this will also require a differnt normalization for cross kerr
                    # if not initial_seq is None:#need to turn to ordered dict
                    #     for element,pulse in initial_seq.items():
                    #         play(element,pulse)
                    #     align(*initial_seq.keys())
                    
    def run_number_splitting_spectroscopy(self,normalize = False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 is_plot_results= True,**kwargs):
        #TODO adjust normalization to work

        if not hasattr(self, 'spec_number_splitting_prog'): raise ValueError("Idiot! You did not write spec_number_splitting_")
     
        self.spec_number_splitting_results = dict()
        self.spec_number_splitting_results["freq"], self.spec_number_splitting_results["I"], self.spec_number_splitting_results["Q"] = self.run_spec(self.spec_number_splitting_prog ,self.spec_number_splitting_params["N_avg"] )        
    
        if normalize:
            prog,_ = self.load_number_splitting_spectroscopy(**self.spec_number_splitting_params,normalize = True,update_prog = False)
            self.spec_number_splitting_results =  self.normalize_CrossKerr(self.spec_number_splitting_params, self.spec_number_splitting_results,prog = prog, is_spec = True)
        else:
            plot_normaliztion =False        
        self.number_splitting_plot_normaliztion = plot_normaliztion
        if is_plot_results:
            return self.plot_number_splitting_spectroscopy(**kwargs)
        return


    def plot_number_splitting_spectroscopy(self,plot_normaliztion=None,prog_name = 'spectroscopy',is_thresh = None, fig_num = None,**kwargs):
        if is_thresh is None: is_thresh = self.is_thresh
        
        plot_normaliztion = self.number_splitting_plot_normaliztion if plot_normaliztion is None else plot_normaliztion

        # ret = self.fit_and_plot( 'Exp(Exp)' , self.process_data(self.spec_number_splitting_results), ti = self.spec_number_splitting_results["freq"]  , title_str = f"T1 EM ({self.main_mem})",label = 'Fitted data',**kwargs ) #            max_seq_clks = int(max_seq_time // 4)
        #TODO with Eliya turn in to a spectroscopy plotter
        if fig_num is  None: fig_num = prog_name + ' ' + str(self.next_fig_num_by_prog_name(prog_name))
        ret = plt.figure(fig_num)
        plt.plot(self.autoscale_data(self.spec_number_splitting_params["Delta_freq_list"])[0], self.autoscale_data(self.process_data(data = [self.spec_number_splitting_results["I"], self.spec_number_splitting_results["Q"]])[0])[0],'-o',label = 'Measured Data')
            
        plt.xlabel("frequencey [{}Hz]".format(self.autoscale_data(self.spec_number_splitting_params["Delta_freq_list"][0])[1]))
        Y_units_prefix =self.autoscale_data(self.process_data(data = [self.spec_number_splitting_results["I"], self.spec_number_splitting_results["Q"]])[0])[1]
        if is_thresh: plt.ylabel(r'$\langle Z \rangle$')
        elif self.which_data is 'Phase': plt.ylabel('{1} [{0}Rad]'.format(Y_units_prefix, self.which_data))
        else: plt.ylabel('{1} [{0}V]'.format(Y_units_prefix, self.which_data))
            
        if plot_normaliztion:
            plt.plot(self.autoscale_data(self.spec_number_splitting_params["Delta_freq_list"])[0], self.autoscale_data(self.process_data(data = [self.spec_number_splitting_results["I_orig"], self.spec_number_splitting_results["Q_orig"]])[0])[0],'+',label = 'Measured Data')
            plt.plot(self.autoscale_data(self.spec_number_splitting_params["Delta_freq_list"])[0], self.autoscale_data(self.process_data(data = [self.spec_number_splitting_results["I_CrossKerr"], self.spec_number_splitting_results["Q_CrossKerr"]], is_calc_stat_error = False, is_thresh = False))[0],'o', label = 'CrossKerr')
            leg = plt.legend()
            leg.set_draggable(True)
        
        return ret
    
    def run_number_splitting_measurment(self,normalize =False,**kwargs):
        self.load_number_splitting_spectroscopy(is_sweep_EM = False,**kwargs)
        self.run_number_splitting_spectroscopy(normalize=normalize,**kwargs)
        plt.show()
        plt.pause(0.5)
        x= self.autoscale_data(self.spec_number_splitting_params["Delta_freq_list"])[0]
        popt,pcov, freq,amplitudes = self.fit_multiple_gaussians(x, self.autoscale_data(self.process_data(data = [self.spec_number_splitting_results["I"], self.spec_number_splitting_results["Q"]])[0])[0])
        plt.plot(x, fit_multiple_gaussians(x,*popt) , '-')

    def run_cavity_pinopi_spectroscopy(self,normalize = False, is_thresh = None, fig_num = None, plot_normaliztion= False,**kwargs):
        if is_thresh is None: is_thresh = self.is_thresh

        self.load_number_splitting_spectroscopy(is_sweep_EM = True,start_with = None,fig_num = None,plot_normaliztion = False,**kwargs)
        self.run_number_splitting_spectroscopy(is_plot_results = False,normalize=normalize,**kwargs)

        prog_name = 'cavity_pinopi_spec'
        x = self.autoscale_data(self.spec_number_splitting_params["Delta_freq_list"])[0]
        y = self.autoscale_data(self.process_data(data = [self.spec_number_splitting_results["I"], self.spec_number_splitting_results["Q"]])[0])[0]
        if fig_num is None: fig_num = prog_name + ' ' + str(self.next_fig_num_by_prog_name(prog_name))
        ret = plt.figure(fig_num)
        plt.plot(x, y,'-o',label = 'Fitted Data_nopi')
        popt_nopi, pcov_nopi,sign_lor = self.fit_to_lorentzian(x,y)
        popt_nopi_g, pcov_nopi_g,sign_lor_g = self.fit_to_gussian(x,y)
        np.sqrt(np.diag(pcov_nopi))[3]
        print(f'{popt_nopi[3]})')#' +/- {}')
        print(f'{popt_nopi_g[1]}')#' +/- {np.sqrt(np.diag(pcov_nopi_g))[1]}')
        plt.plot(x,sign_lor*Lorentzian(x,*popt_nopi),'-', label= 'fit to nopi')
        plt.plot(x,sign_lor_g*Gaussian(x,*popt_nopi_g),'+', label= 'fit to nopi')
        plt.xlabel("frequencey [{}Hz]".format(self.autoscale_data(self.spec_number_splitting_params["Delta_freq_list"][0])[1]))
        Y_units_prefix =self.autoscale_data(self.process_data(data = [self.spec_number_splitting_results["I"], self.spec_number_splitting_results["Q"]])[0])[1]
        if self.is_thresh: plt.ylabel(r'$\langle Z \rangle$')
        elif self.which_data is 'Phase': plt.ylabel('{1} [{0}Rad]'.format(Y_units_prefix, self.which_data))
        else: plt.ylabel('{1} [{0}V]'.format(Y_units_prefix, self.which_data))
            
        if plot_normaliztion:
            plt.plot(x, self.autoscale_data(self.process_data(data = [self.spec_number_splitting_results["I_orig"], self.spec_number_splitting_results["Q_orig"]])[0])[0],'+',label = 'Measured Data_nopi')
            plt.plot(x, self.autoscale_data(self.process_data(data = [self.spec_number_splitting_results["I_CrossKerr"], self.spec_number_splitting_results["Q_CrossKerr"]], is_calc_stat_error = False, is_thresh = False))[0],'o', label = 'CrossKerr_nopi')
        self.Cavity_pinopi_results = {}
        self.Cavity_pinopi_results['noPi']  = self.spec_number_splitting_results.copy()
        plt.pause(0.1)
        self.load_number_splitting_spectroscopy(is_sweep_EM = True,start_with = (self.pi_pulse,self.main_qubit) ,**kwargs)
        self.run_number_splitting_spectroscopy(is_plot_results = False,normalize=normalize,**kwargs)
        plt.plot(x, self.autoscale_data(self.process_data(data = [self.spec_number_splitting_results["I"], self.spec_number_splitting_results["Q"]])[0])[0],'-o',label = 'Measured Data_pi')
        x = self.autoscale_data(self.spec_number_splitting_params["Delta_freq_list"])[0]
        y = self.autoscale_data(self.process_data(data = [self.spec_number_splitting_results["I"], self.spec_number_splitting_results["Q"]])[0])[0]
        popt_pi, pcov_pi,sign_lor = self.fit_to_lorentzian(x,y)
        popt_pi_g, pcov_pi_g,sign_lor_g = self.fit_to_gussian(x,y)
        print(f'{popt_pi[3]}')#' +/- {np.sqrt(np.diag(pcov_pi))[3]}')
        print(f'{popt_pi_g[1]}')#' +/- {np.sqrt(np.diag(pcov_pi_g))[1]}')
        plt.plot(x,sign_lor*Lorentzian(x,*popt_pi),'-', label= 'fit to pi')
        plt.plot(x,sign_lor_g*Gaussian(x,*popt_pi_g),'+', label= 'fit to pi')
        if plot_normaliztion:
            plt.plot(x, self.autoscale_data(self.process_data(data = [self.spec_number_splitting_results["I_orig"], self.spec_number_splitting_results["Q_orig"]])[0])[0],'+',label = 'Measured Data_pi')
            plt.plot(x, self.autoscale_data(self.process_data(data = [self.spec_number_splitting_results["I_CrossKerr"], self.spec_number_splitting_results["Q_CrossKerr"]], is_calc_stat_error = False, is_thresh = False))[0],'o', label = 'CrossKerr_pi')
        self.Cavity_pinopi_results['Pi']  = self.spec_number_splitting_results.copy()
        
        leg = plt.legend()
        leg.set_draggable(True)
        return popt_nopi,pcov_nopi,popt_nopi_g,pcov_nopi_g,popt_pi,pcov_pi,popt_pi_g,pcov_pi_g
    
    def run_number_splitting_measurment(number_gaussians = None,**kwargs):
        self.load_number_splitting_spectroscopy(is_sweep_EM = False,**kwargs)
        ret = self.run_number_splitting_spectroscopy(**kwargs)
        plt.show()
        plt.pause(0.5)
        x= lf.autoscale_data(self.spec_number_splitting_params["Delta_freq_list"])[0]
        popt,pcov, freq,amplitudes = self.fit_multiple_gaussians(x, self.autoscale_data(self.process_data(data = [self.spec_number_splitting_results["I"], self.spec_number_splitting_results["Q"]])[0])[0])
        plt.plot(x, fit_multiple_gaussians(x,*popt) , '-')
    # def fit_and_plot_lorentzian(self,x,y,fig_num**kwargs):
        
    # def fit_to_positive_lorentzian(self,x,y,guess = None):#center at maximum
    #     if guess is None:
    #         b_guess = np.abs(x[np.argmin(np.abs(y-(max(y)+min(y))/2))]-x[np.argmax(y)])   # b_guess :)
    #         a_guess = (np.max(y)-np.min(y))*b_guess**2
    #         guess =  [a_guess, b_guess, np.min(y),x[np.argmax(y)]]
    #     return curve_fit(Lorentzian,x,y,p0 =guess)
    # #TODO calculate fit and statitical error
    # def fit_to_lorentzian(self,x,y,guess = None):
    #     try:
    #        popt_p,pcov_p = self.fit_to_positive_lorentzian(x,y,guess = guess)
    #     except:
    #        popt_m,pcov_m = self.fit_to_positive_lorentzian(x,-y,guess = guess)
    #        return popt_m,pcov_m,-1
       
    #     try:
    #        popt_m,pcov_m = self.fit_to_positive_lorentzian(x,-y,guess = guess)
    #     except:
    #        return popt_p,pcov_p,1
        
    #     if max(abs(Lorentzian(x,*popt_p)-y))>max(abs(Lorentzian(x,*popt_m)+y)):
    #         return popt_m,pcov_m,-1
    #     return popt_p,pcov_p,1

    # def fit_multiple_gaussians(self, x,y, noise_level= None):
    #     if noise_level is  None:
    #         print('how many gaussians')
    #         N_gauss = input()
    #         guess = []
    #         print('what is the center farthest to the left')
    #         first_center = input()
    #         print('what is chi')
    #         chi = input()
    #         print('what is gaussian width')
    #         width = [input()]
    #         for i in range(N_gauss):
    #             print('what is gaussian center')
    #             # guess += [input()]
    #             guess += [i*chi+first_center]
    #             print('what is gaussian amp')
    #             guess +=  [input()]
    #             guess += width
    #         from scipy.optimize import  curve_fit
    #         popt, pcov = curve_fit(multiple_gaussians, x, y, p0=guess)
            
    #     frequency = []; amplitudes = []
    #     for i in range(3, len(popt)+3, 3):
    #         freq+=popt[i]
    #         amplitudes+=popt[i+1]
    #     return popt,pcov, freq,amplitudes
            
            
        
        
        
#%% Cavity wigner 
    
    def load_wigner_tomography(self, N_avg =100,
                                    Displacement_pulse = None,
                                    seq = None,# a tuple for an inital pulse(pulse,element) or a function, function will receive **kwargs and 
                                    grid_size = (20,20), # for(n,m) grid size will be (2n+1)*(2m+1) 
                                    step_size = 0.05,
                                    mem_mode=None,
                                    update_prog = True,
                                    wait_time = 13089, #pi/chi in ns
                                    normalize = False,**kwargs):
        
        if mem_mode is None: mem_mode =self.main_mem
        if Displacement_pulse is None: Displacement_pulse = self.UnDisp1_pulse
        wait_time = wait_time//4
        #TODO, add check overflow
        
        # max_seq_time = int(max_seq_clks * 4)                            
        grid_X = np.arange(-grid_size[0],grid_size[0]+1,1)*step_size
        grid_Y = np.arange(-grid_size[1],grid_size[1]+1,1)*step_size
        if max(np.abs(grid_X))>1.99 or  max(np.abs(grid_Y))>1.99:
            raise ValueError('QUA amp function hate me')
        #check overflow
        
        #Save parameters:
        params = {}
        params["N_avg"]              = N_avg
        params["grid_X"]             = grid_X#cartesan
        params["grid_Y"]             = grid_Y#cartesan
        params["Displacement_pulse"] = Displacement_pulse
        params["mem_mode"]           = mem_mode  
        params["seq"]                = seq

        # params["step_size_clks"]     = step_size_clks
        
        pi2_amp_mul = 0.0 if normalize else 1.0
        
        if update_prog: self.wigner_params = params
            
        #Calculate and show expeced runtime:
        try:
            run_time = N_avg*len(grid_X)*len(grid_Y)*(self.pulse_len(self.main_readout,self.ro_pulse)+wait_time * 4 +self.wait_between_seq * 4)
            print('Run time is {}s'.format(round(run_time * 1e-9)))         
        except:
            print('cant calculate time')
        
        #Create program:
        with program() as prog:
            Xval    = declare(fixed)
            Yval    = declare(fixed)
            n       = declare(int)
            I       = declare(fixed)
            Q       = declare(fixed)
            with for_(n,0,n<N_avg,n+1):
                with for_each_(Xval,grid_X.tolist()):
                    with for_each_(Yval,grid_Y.tolist()):
                    
                        if type(seq) is tuple: 
                            play(*seq)
                            align(seq[1],self.main_qubit)
                        elif type(seq) is types.FunctionType:
                            seq(mem_mode = mem_mode,**kwargs)
                        else:
                            warn('wignering nothing')
                        
                        play(Displacement_pulse*amp(Xval,Yval,Yval,Xval),mem_mode)
                        align(mem_mode,self.main_qubit)
                        
                        play(self.pi2_pulse*amp(pi2_amp_mul),self.main_qubit)
                        wait(wait_time)
                        play(self.pi2_pulse*amp(pi2_amp_mul),self.main_qubit)
                        
                        # align(self.main_qubit,mem_mode)
                        # play(Displacement_pulse*amp(-Xval,-Yval,-Yval,-Xval),mem_mode)
                        
                        align(self.main_qubit, self.main_readout,mem_mode)
                        self.perform_full_measurement(I,Q)
        
        if update_prog: self.wigner_prog = prog
        
        return prog,params
# self.perform_full_measurement(I,Q)
   
    # def wigner_seq(self,DeltaT,Displacement_pulse,Scale_displacment,mem_mode):
    #     for s_disp in Scale_displacment:
    #         play(Displacement_pulse*amp(s_disp) , mem_mode )
       
    #     align(self.main_qubit, mem_mode)
    #     with if_(~(DeltaT==0)):
    #         wait(DeltaT, self.main_qubit)

    def run_wigner_tomography(self,normalize = False,plot_normaliztion =True,#if normalize is true the program will run cross kerr
                                 **kwargs):

        if not hasattr(self, 'wigner_prog'): raise ValueError("Idiot! You did not write wigner program")
     
        self.wigner_results = dict()
        self.wigner_results["I"], self.wigner_results["Q"] = self.run_grid_prog(self.wigner_prog,self.wigner_params["N_avg"],len(self.wigner_params['grid_X']),len(self.wigner_params['grid_Y']) )        
        
        if normalize:
            prog,_ = self.load_wigner_tomography(**self.wigner_params,normalize = True,update_prog = False)

            plot_normaliztion = False
        else:
            if normalize:
                raise ValueError('requires changing to run grid prog')
                self.wigner_results =  self.normalize_CrossKerr(self.wigner_params, self.wigner_results,prog = prog)
            else:
                plot_normaliztion = False  
                
        self.wigner_plot_normaliztion = plot_normaliztion
        return self.plot_wigner_tomography(**kwargs)

    # def plot_cavity_tomograpy(self,)
    def plot_wigner_tomography(self,title_str='',**kwargs):
        # plot_normaliztion = self.wigner_plot_normaliztion if plot_normaliztion is None else plot_normaliztion
        # Check what's happening here:
        #           fit_and_plot(self,fittype, I, Q, ti, N_avg,  txt=None,title_str =None, fig_num=None ):  
        prog_name = 'Wigner tomography'
        fig_num = prog_name + ' ' + str(self.next_fig_num_by_prog_name(prog_name))
        ret = self.plot_continuous_tomography(self.wigner_params['grid_X'],self.wigner_params['grid_Y'],
                                              self.process_data(self.wigner_results, is_calc_stat_error = False,is_thresh = False), 
                                              title_str = f"Wigner of ({self.main_mem})",
                                              fig_num = fig_num )
        return ret
    
    

#%% Example Code:
    def run_and_fix_old(self,prog_function,var_name ='scale_amp',normalize = False,is_remove= True,plot_normaliztion =False,program_name = None,**kwargs):#if normalize is true the program will run cross kerr
        
        prog_ori,params = prog_function(**kwargs)
        
        results = dict()
                
        if is_remove:
            results['var'], results["I_orig"], results["Q_orig"] = self.run_prog(prog_ori, params["N_avg"],var = var_name )
         
            prog,_ = prog_function(**params,is_remove = True,update_prog = False)
            _, results["I_rev"], results["Q_rev"] = self.run_prog(prog ,params["N_avg"],var = var_name )
            
            results["I"] = results["I_orig"]-results["I_rev"]
            results["Q"] = results["Q_orig"]-results["Q_rev"]
            plot_normaliztion = False
        else:
            results['var'], results["I"], results["Q"] = self.run_prog(prog_ori ,params["N_avg"],var = var_name )        
            if normalize:
                prog,_ = prog_function(**params,normalize = True,update_prog = False)
                results =  self.normalize_CrossKerr(params, results,prog = prog,var = var_name)
            else:
                plot_normaliztion =False        
        
        return results,params,plot_normaliztion

    def run_for_phase_old(self,prog_function,is_use_previous = False,is_measure_Z = False,**kwargs):
        complex_results = dict()
        
        results1,params,plot_normaliztion=self.run_and_fix(prog_function,is_phase = False,**kwargs)

            
        if params['pi2_phase']%0.5:
            complex_results['Im'] = results1
            key= 'Re'
        else:
            complex_results['Re'] = results1
            key = 'Im'
            
        complex_results[key],_,_=self.run_and_fix(prog_function,update_prog = False,is_phase=True,**kwargs)
        
        if is_measure_Z: complex_results['Z'],_,_=self.run_and_fix(prog_function,update_prog = False,is_phase=True,**kwargs)
        
        return complex_results,params,plot_normaliztion
 
    def old_load_T1(self, num_of_pnts = 50, # How many points I measure
                    max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
                    N_avg = 1000 # How many times each sequence is executed 
                    ):        
        max_seq_clks= int(max_seq_time // 4)
        step_size_clks =  max_seq_clks // num_of_pnts
        self.T1_N_avg = N_avg
     
        run_time = N_avg * num_of_pnts * (max_seq_time / 2 + self.config['pulses']['meas_pulse_in']['length'] + 2 * self.config['pulses']['pi2_pulse_qb1_in']['length'] + 4 * self.wait_between_seq)
        print('Run time is {}s'.format(round(run_time * 1e-9)))
        
        with program() as prog:
            wt_time = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            with for_(n, 0, n < N_avg, n + 1):
                with for_(wt_time, step_size_clks, wt_time <= max_seq_clks, wt_time + step_size_clks):
                    
                    if self.e_to_f:  self.map_e_to_g(self.main_qubit_g_to_e)

                    play('pi2_pulse', self.main_qubit)
                    play('pi2_pulse', self.main_qubit)
                    
                    wait(wt_time, self.main_qubit)
                    
                    if self.e_to_f and self.e_to_f_map_back:  self.map_e_to_g(self.main_qubit_g_to_e)

                    align(self.main_qubit, self.main_readout)
                    measure(self.ro_pulse, self.main_readout, None,
                            demod.full('integ_w_I', I, 'out1'),
                            demod.full('integ_w_Q', Q, 'out2'))
                    wait(self.wait_between_seq, self.main_readout)  # reset time
                    save(wt_time, 't')
                    save(I, 'I')
                    save(Q, 'Q')
        
        self.cavity_T1_prog =prog
        
        
    def old_run_T1(self, 
              *args,**kwargs): 
    
        if self.cavity_T1_prog is None: raise ValueError("Idiot! You did not write T1 program")
        
        self.T1_t_res, self.T1_I_res, self.T1_Q_res = self.run_prog(self.cavity_T1_prog ,self.T1_N_avg, *args,**kwargs)
        
        return self.plot_T1()

    def old_plot_T1(self):
        try:
            return self.fit_and_plot('Exp',self.T1_I_res_accum, self.T1_Q_res_accum,self.T1_t_res,title_str = 'T1 ({})'.format(self.main_qubit), N_avg = self.T1_N_avg)
        except: 
            return self.fit_and_plot('Exp',self.T1_I_res, self.T1_Q_res,self.T1_t_res,title_str = 'T1 ({})'.format(self.main_qubit), N_avg = self.T1_N_avg)
    
    
    def old_T1_complete(self, num_of_pnts = 50, # How many points I measure
               max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
               N_avg = 1000 # How many times each sequence is executed 
               ):
        
        self.load_T1(num_of_pnts=num_of_pnts,max_seq_time= max_seq_time, N_avg=N_avg)
        
        return self.run_T1()

#%% general boson functions
    def _ConDisp_pulse(self,is_two_mode_disp = None,qubit = None, mem_mode = None,mem_mode1 = None, mem_mode2 = None, is_fixCD_cPhase = None,fix_Cangle = None, scale_amp =1.0,Displacement_pulse = None,is_Yale = None,qb_phase_correction=0,factor_amp2 =1.0, **kwargs):
        
        if is_two_mode_disp is None: is_two_mode_disp =  self.is_two_mode_disp
        
        if mem_mode1 is None: mem_mode1= mem_mode
        if mem_mode2 is None: mem_mode2= self.scnd_mem
        
        if is_two_mode_disp == 'Sim': #Two modes are displaced at the same time
                    
            self._2mode_ConDisp_pulse(qubit=qubit,is_fixCD_cPhase=is_fixCD_cPhase,fix_Cangle1=fix_Cangle,Displacement_pulse1 =Displacement_pulse,is_Yale=is_Yale,qb_phase_correction=qb_phase_correction,scale_amp1 = scale_amp,factor_amp2 =factor_amp2,**kwargs)
        elif is_two_mode_disp or is_two_mode_disp =='Sam':

            if is_two_mode_disp == 'Sam':
                print('Sam')
                self._2mode_ConDisp_pulse(qubit=qubit,is_fixCD_cPhase=is_fixCD_cPhase,mem_mode1 =mem_mode1, mem_mode2 = mem_mode2,fix_Cangle1=fix_Cangle,Displacement_pulse1 =Displacement_pulse,is_Yale=is_Yale,qb_phase_correction=qb_phase_correction,scale_amp1 = scale_amp,factor_amp2 =factor_amp2,_is_fix_sam = True,**kwargs)
            else:
                self._2mode_ConDisp_pulse(qubit=qubit,is_fixCD_cPhase=is_fixCD_cPhase,mem_mode1 =mem_mode1, mem_mode2 = mem_mode2,fix_Cangle1=fix_Cangle,Displacement_pulse1 =Displacement_pulse,is_Yale=is_Yale,qb_phase_correction=qb_phase_correction,scale_amp1 = scale_amp,factor_amp2 =0,**kwargs)
                if not factor_amp2 == 0:
                    if is_fixCD_cPhase or (is_fixCD_cPhase is None and  self.is_fixCD_cPhase):
                        # print ('fixing phase of pi pulse')
                        frame_rotation_2pi(-np.round(self.pulse_len(self.main_qubit,self.pi_pulse)/2*self.mem_chi[mem_mode1],6),mem_mode1)
                        frame_rotation_2pi(-np.round(self.pulse_len(self.main_qubit,self.pi_pulse)/2*self.mem_chi[mem_mode2],6),mem_mode2)
                    self._pi_pulse(qubit = qubit,**kwargs)
                    
                    self._2mode_ConDisp_pulse(qubit=qubit,is_fixCD_cPhase=is_fixCD_cPhase,mem_mode1 =mem_mode, mem_mode2 = mem_mode2,fix_Cangle1=fix_Cangle,Displacement_pulse1 =Displacement_pulse,is_Yale=is_Yale,qb_phase_correction=qb_phase_correction,scale_amp1 = scale_amp,factor_amp2 =factor_amp2,_factor_amp1=0,**kwargs)
        else:
            
            if qubit is None: qubit = self.main_qubit
            if mem_mode is None: mem_mode = self.main_mem
            if Displacement_pulse is None: Displacement_pulse= self.ConDisp1_pulse
            if is_Yale is None: is_Yale = self.is_Yale
            try:
                if is_fixCD_cPhase is None: is_fixCD_cPhase = self.is_fixCD_cPhase
                if  is_fixCD_cPhase and (not is_fixCD_cPhase ==  'by_dic'):
                    fix_Cangle =-(self.pulse_len(mem_mode,Displacement_pulse)+self.pulse_len(self.main_qubit,self.pi_pulse)/2)*self.mem_chi[mem_mode]
                    print('by chi')
                elif fix_Cangle is None: 
                    fix_Cangle = self.CD_cPhase_dic[mem_mode][Displacement_pulse]
                    print('by dict')
                fix_Cangle = np.round(fix_Cangle,6)
            except:
                warn(f'failed CD phase correction on {mem_mode} {Displacement_pulse}')
                is_fixCD_cPhase = False
            align(qubit, mem_mode)
            if type(scale_amp) is list:
                play(Displacement_pulse*amp(scale_amp[0],scale_amp[1],scale_amp[2],scale_amp[3]) , mem_mode )
            else:
                play(Displacement_pulse*amp(scale_amp) , mem_mode )
    
            if is_Yale:
                align(qubit, mem_mode)
                if is_fixCD_cPhase:
                    print('fixing phase of memory modes')
                    print(fix_Cangle)
                    frame_rotation_2pi(fix_Cangle,mem_mode)
    
                self._pi_pulse(qubit = qubit,**kwargs)
                align(qubit, mem_mode)
                if type(scale_amp) is list:
                    play(Displacement_pulse*amp(-scale_amp[0],-scale_amp[1],-scale_amp[2],-scale_amp[3]) , mem_mode )
                else:
                    play(Displacement_pulse*amp(-scale_amp) , mem_mode )
                
            # elif qb_phase_correction:
                
            #     frame_rotation_2pi(qb_phase_correction,qubit)
            # align(qubit, mem_mode)
            # play(self.pi2_pulse*amp(0),qubit)
            align(qubit, mem_mode)
    def _2mode_ConDisp_pulse(self,qubit = None, mem_mode1 = None, mem_mode2 = None, is_fixCD_cPhase = None,fix_Cangle1 = None,fix_Cangle2 = None, scale_amp1 =1.0,scale_amp2 = None,factor_amp2 =1.0, Displacement_pulse1= None, Displacement_pulse2= None,is_Yale = None,qb_phase_correction=0,_factor_amp1=1,_is_fix_sam = False,**kwargs):
        if qubit is None: qubit = self.main_qubit
        if mem_mode1 is None: mem_mode1 = self.main_mem
        if mem_mode2 is None: mem_mode2 = self.scnd_mem
        if Displacement_pulse1 is None: Displacement_pulse1= self.ConDisp1_pulse
        if Displacement_pulse2 is None: Displacement_pulse2= Displacement_pulse1
        if is_Yale is None: is_Yale = self.is_Yale
        # if scale_amp2 is None and type(scale_amp1) is list: scale_amp2 = [factor_amp2*scale_amp1[0],factor_amp2*scale_amp1[1],factor_amp2*scale_amp1[2],factor_amp2*scale_amp1[3]]
        # if scale_amp2 is None:  scale_amp2 = scale_amp1
        if _factor_amp1 ==0 and factor_amp2 == 0:
            return
        if scale_amp2 is None and type(scale_amp1) in [float,list,int]: scale_amp2 = scale_amp1
        print(mem_mode1)
        print(mem_mode2)
        if 1:#try:
            if is_fixCD_cPhase is None: is_fixCD_cPhase = self.is_fixCD_cPhase
            
            if  is_fixCD_cPhase and (not is_fixCD_cPhase ==  'by_dic'):
                if _is_fix_sam:
                    tot_time =(self.pulse_len(mem_mode2,Displacement_pulse2)+self.pulse_len(mem_mode1,Displacement_pulse1)+self.pulse_len(self.main_qubit,self.pi_pulse)/2)
                    if not _factor_amp1: tot_time = self.pulse_len(mem_mode2,Displacement_pulse2)+self.pulse_len(self.main_qubit,self.pi_pulse)/2
                    if not factor_amp2: tot_time  = self.pulse_len(mem_mode1,Displacement_pulse1)+self.pulse_len(self.main_qubit,self.pi_pulse)/2
                    fix_Cangle1 =-tot_time*self.mem_chi[mem_mode1]
                    fix_Cangle2  =-tot_time*self.mem_chi[mem_mode2]
                else:
                    fix_Cangle1 =-(self.pulse_len(mem_mode1,Displacement_pulse1)+self.pulse_len(self.main_qubit,self.pi_pulse)/2)*self.mem_chi[mem_mode1]
                    fix_Cangle2  =-(self.pulse_len(mem_mode2,Displacement_pulse2)+self.pulse_len(self.main_qubit,self.pi_pulse)/2)*self.mem_chi[mem_mode2]
                    if not _factor_amp1: fix_Cangle1 =-(self.pulse_len(mem_mode2,Displacement_pulse2)+self.pulse_len(self.main_qubit,self.pi_pulse)/2)*self.mem_chi[mem_mode1]
                    if not factor_amp2: fix_Cangle2  =-(self.pulse_len(mem_mode1,Displacement_pulse1)+self.pulse_len(self.main_qubit,self.pi_pulse)/2)*self.mem_chi[mem_mode2]
                print('by chi') 
                is_fixCD_cPhase = True
                
            if fix_Cangle1 is None:
                fix_Cangle1 = self.CD_cPhase_dic[mem_mode1][Displacement_pulse1]
                print('by dict')
            if fix_Cangle2 is None:
                fix_Cangle2 = self.CD_cPhase_dic[mem_mode2][Displacement_pulse2]
                print('by dict')
            
            fix_Cangle1= np.round(fix_Cangle1,6)
            fix_Cangle2= np.round(fix_Cangle2,6)
        else:#except:
            warn('failed CD phase correction')
            is_fixCD_cPhase = False
        
        # print(scale_amp1)
        align(qubit, mem_mode1, mem_mode2)
        if _factor_amp1 ==0:
            if factor_amp2 == 0: is_fixCD_cPhase = False
            print('skiping main displacment')
            
            # play(Displacement_pulse1*amp(0) , mem_mode1 )
        elif type(scale_amp1) is list:
            play(Displacement_pulse1*amp(scale_amp1[0],scale_amp1[1],scale_amp1[2],scale_amp1[3]) , mem_mode1 )
        else:
            play(Displacement_pulse1*amp(scale_amp1) , mem_mode1 )
        
        if _is_fix_sam: align(mem_mode1,mem_mode2)
        
        if factor_amp2 == 0:
            print('skiping secondary displacment')
        elif type(scale_amp2) is list:
            # print(scale_amp2)
            play(Displacement_pulse2*amp(scale_amp2[0],scale_amp2[1],scale_amp2[2],scale_amp2[3]) , mem_mode2 )
        elif scale_amp2 is None:
            # print(f'factor amp {factor_amp2} scale_amp {scale_amp1}')
            play(Displacement_pulse2*amp(scale_amp1*factor_amp2) , mem_mode2 )
        else:
            # print(f'scale_amp {scale_amp2}')
            play(Displacement_pulse2*amp(scale_amp2) , mem_mode2 )


        if is_Yale:
            align(qubit, mem_mode1,mem_mode2)
            
            if is_fixCD_cPhase:
                print(f'fixing phase of memory modes {fix_Cangle1} {fix_Cangle2}')
                
                frame_rotation_2pi(fix_Cangle1,mem_mode1)
                frame_rotation_2pi(fix_Cangle2,mem_mode2)

            self._pi_pulse(qubit = qubit)
            align(qubit,mem_mode1,mem_mode2)
            if _factor_amp1 ==0:
            # play(Displacement_pulse1*amp(0) , mem_mode1 )
                tt  =1
            elif type(scale_amp1) is list:
                play(Displacement_pulse1*amp(-scale_amp1[0],-scale_amp1[1],-scale_amp1[2],-scale_amp1[3]) , mem_mode1 )
            else:
                play(Displacement_pulse1*amp(-scale_amp1) , mem_mode1 )
           
            if _is_fix_sam: align(mem_mode1,mem_mode2)
            
            if factor_amp2 == 0:
                tt = 1
            elif type(scale_amp2) is list:
                play(Displacement_pulse2*amp(-scale_amp2[0],-scale_amp2[1],-scale_amp2[2],-scale_amp2[3]) , mem_mode2 )
            elif scale_amp2 is None:
                play(Displacement_pulse2*amp(-scale_amp1*factor_amp2) , mem_mode2 )

            else:
                # print(-scale_amp2)
                play(Displacement_pulse2*amp(-scale_amp2) , mem_mode2 )

       # elif qb_phase_correction:
            
        #     frame_rotation_2pi(qb_phase_correction,qubit)
        # align(qubit, mem_mode)
        play(self.pi2_pulse*amp(0),qubit)
        align(qubit, mem_mode1,mem_mode2)
        def _2mode_ConDisp_pulse_test(self,qubit = None, mem_mode = None, is_fixCD_cPhase = None,fix_Cangle = None, scale_amp =1,Displacement_pulse = None,is_Yale = None,qb_phase_correction=0,**kwargs):

            if qubit is None: qubit = self.main_qubit
            # if mem_mode is None: mem_mode = self.main_mem
            # if Displacement_pulse is None: Displacement_pulse= self.ConDisp1_pulse
            if is_Yale is None: is_Yale = self.is_Yale
            # try:
            #     if is_fixCD_cPhase is None: is_fixCD_cPhase = self.is_fixCD_cPhase
            #     if fix_Cangle is None: fix_Cangle = self.CD_cPhase_dic[mem_mode][Displacement_pulse]
            # except:
            #     warn(f'failed CD phase correction on {mem_mode} {Displacement_pulse}')
            #     is_fixCD_cPhase = False
            print(type(scale_amp))
            align(qubit, 'mm1_g','mm2')
            if type(scale_amp) is list:
                play('Fast_ConDisp1'*amp(scale_amp[0],scale_amp[1],scale_amp[2],scale_amp[3]) , 'mm1_g' )
                play('Fast_ConDisp1'*amp(scale_amp[0],scale_amp[1],scale_amp[2],scale_amp[3]) , 'mm2' )
            else:
                play('Fast_ConDisp1'*amp(scale_amp) , 'mm1_g' )
                play('Fast_ConDisp1'*amp(scale_amp) , 'mm2' )
    
            if is_Yale:
                align(qubit, 'mm1_g','mm2')
                if is_fixCD_cPhase or 1:
                    print('fixing phase of memory modes')
                    print(fix_Cangle)
                    frame_rotation_2pi(-0.0113,'mm1_g')
                    frame_rotation_2pi(-0.0936,'mm2')
    
                self._pi_pulse(qubit = qubit,**kwargs)
                align(qubit, 'mm1_g','mm2')
                if type(scale_amp) is list:
                    play('Fast_ConDisp1'*amp(-scale_amp[0],-scale_amp[1],-scale_amp[2],-scale_amp[3]) , 'mm1_g' )
                    play('Fast_ConDisp1'*amp(-scale_amp[0],-scale_amp[1],-scale_amp[2],-scale_amp[3]) , 'mm2' )
                else:
                    play('Fast_ConDisp1'*amp(-scale_amp) , 'mm1_g' )
                    play('Fast_ConDisp1'*amp(-scale_amp) , 'mm2' )
            # elif qb_phase_correction:
                
            #     frame_rotation_2pi(qb_phase_correction,qubit)
            # align(qubit, mem_mode)
            play(self.pi2_pulse*amp(0),qubit)
            align(qubit, 'mm1_g','mm2')

    def cavity2qubit_0_mapping(self,I,Q,mem_mod = None,**kwargs):
        
        if mem_mod is None: mem_mod = self.main_mem
        
        align(self.main_qubit, mem_mod)
        self._Con_Pi_pulse(**kwargs)

        # Meausre if qubit is at  |e>
        # self._Un_Pi_pulse()
        align(self.main_qubit, self.main_readout,mem_mod)
        
        self.perform_full_measurement(I,Q, **kwargs)
    
    def _Con_Pi_pulse(self,*args,qubit = None,pulse=None,pi_amp_mul = 1.0, **kwargs):
        
        if qubit is None: qubit = self.main_qubit
        
        if self.is_pi_pulse:
            if pulse is None: pulse= self.ConRotPi_pulse
            play(pulse*amp(pi_amp_mul), qubit)
        else:
            if pulse is None: pulse= self.ConRotPi2_pulse
            play(pulse*amp(pi_amp_mul), qubit)
            play(pulse*amp(pi_amp_mul), qubit)

    def _Un_Pi_pulse(self,*args,qubit = None,pulse=None,**kwargs):
        if qubit is None:  qubit = self.main_qubit
        
        if self.is_pi_pulse:
            if pulse is None: pulse= self.UnRotPi_pulse
            play(pulse, qubit)
        else:
            if pulse is None: pulse= self.UnRotPi2_pulse
            play(pulse, qubit)
            play(pulse, qubit)
            '''
    def load_T2_cavity_measurement(self, num_of_pnts = 50, # How many points to measure
                                    max_seq_time = 80000, # Time of longest sequence in units of nano seconds. Must be a multiple of 4
                                    N_avg = 1000 , # How many times each sequence is executed 
                                    Displacement_pulse = None,
                                    **kwargs ):#str
        
        if Displacement_pulse is None:             
             Displacement_pulse = self.UnDisp1_pulse

        # max_seq_time = int(max_seq_clks * 4)                            
        max_seq_clks = int(max_seq_time // 4)
        step_size_clks =  max_seq_clks // num_of_pnts
        # T1_N_avg = N_avg

        #Save parameters:
        self.T1_params["N_avg"]              = N_avg
        self.T1_params["step_size_clks"]     = step_size_clks
        self.T1_params["max_seq_time"]       = max_seq_time
        self.T1_params["num_of_pnts"]        = num_of_pnts
        self.T1_params["Displacement_pulse"] = Displacement_pulse
        
        #Calculate and show expeced runtime:
        run_time = self.calc_run_time(ExperimentName = "T1")
        print('Run time is {}s'.format(round(run_time * 1e-9)))         
        
        #Create program:
        with program() as prog:
            DeltaT  = declare(int)
            n       = declare(int)
            I       = declare(fixed)
            Q       = declare(fixed)
            with for_(n,0,n<N_avg,n+1):
                with for_(DeltaT , step_size_clks , DeltaT<=max_seq_clks , DeltaT + step_size_clks):

                    # Play Displacement:
                    play(Displacement_pulse , self.main_mem )

                    # Wait DeltaT time to observe T1 decay:
                    wait(DeltaT, self.main_qubit)

                    # Play conditional rotation of qubit:   X Pi pulse to excite from |g>  to |e>  conditioned on   0 photons in Cavity:
                    self.cavity2qubit_Q_mapping(I,Q , **kwargs)

                    save(DeltaT, 't')

        self.T1_prog = prog
    '''
                       
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

if False and __name__ == "__main__":
    from .Config_flute3 import get_my_config as Flute3Config

    BosonTimeDomain = Boson_TiDo_Chara( Flute3Config() , main_qubit='' ,main_mem='mm1' , main_readout='readout_res' , which_data = 'I' ,
                                         wait_between_seq = 120e3 , check_overflow = True, is_pi_pulse = False)
    BosonTimeDomain.load_cavity_T1_measurement(num_of_pnts=10 , max_seq_time=1000 , N_avg=50 , Displacement_pulse="UnDisp1")
    BosonTimeDomain.run_cavity_T1_measurement()
    BosonTimeDomain.plot_cavity_T1_measurement()
    
    
# %%
