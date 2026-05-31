# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 10:55:35 2019

@author: Shay
"""
from general_functions import *
from general_functions import VisaObject
try: import visa # Working with GPIB control (sorry Dar)
except: import pyvisa
from labrad.units import V,mV, s,ms,us,ns, Hz,kHz,MHz,GHz, Ohm, dBm, Value, deg,rad 
from matplotlib.pyplot import pause
import numpy as np
from time import sleep
from warnings import warn


class AgilentPNA(VisaObject):
    def __init__(self, NetworkAddress = 'TCPIP0K-N5232B-81186inst0INSTR', resourceManager = None):
        """Takes a string or integer. Integer is treated as a GPIB address
        Connecting to the VNA VISA driver
        Input: unicode string with VISA address
        creates instrument with name VNA
        """
        self.initialize(NetworkAddress, resourceManager = resourceManager)
        self.visa.write('CALC:PAR:MNUM 1') # Chen 16/10/19 ARRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR The horrable lab demon will get the one(s?) who deleted this!!!!!!!!!!!!!!
        
        #set the byte order to swap
        #self.dev.write(':FORM:BORD SWAP')
        #you have to define a parameter name and select it in order commands like 
        #retrieving the electrical delay to work.
        # self.dev.write('CALC1:PAR1:DEF "My_S21",S21')
        # vna.dev.write('CALC:PAR:SEL TRACE1')
        # self.dev.write('CALC:PAR1:SEL "My_S21"')
        #todo: list the measurement parameters using the CAT command, then select the default one.
        #typically named CH1_S11_1 or CH1_S11_2
        # dev.query('CALC1:PAR1:CAT?')
        # self.dev.write('CALC:PAR1:SEL "CH1_S11_1"')
        #self.visa.write('CALC:PAR1:SEL "CH1_S11_1"')
        """   
    def wait_to_complete(self,func):
        def inner(*args, **kwargs):
            x = func(*args, **kwargs)
            self.visa.write('*OPC')
            return x
        return inner
    """  
    def get_X_units(self):
        """depnding on th sweep type returns the units of the x axis"""
        sweep_type = self.get_sweep_type()
        if sweep_type == 'LIN':
            return 1*Hz#(10*(-9))*GHz
        elif sweep_type == 'CW':
            return 1*s
        elif sweep_type == 'POW':
            return 1*dBm
        elif sweep_type == 'SEGM':
            return 1*s
            
        else: 
            raise ValueError('sweep type not supported')
    def get_Y_units(self):
        frm = self.get_format()
        if frm == 'PHAS':
            return 1*deg
        if frm == 'MLOG':
            return 1*dBm
        warn('y_units not deifned')
        return 1
    def get_X_axis(self):
        """gets the x values in the apporpritate units which are displayed on  the VNA""" 
        if 0 or  not (self.sweep_type()=='LIN'):
            X_axis = self.visa.query_ascii_values('CALC:X?',container=np.array)
        else:
            # warn('x axis is caluculated manualy')
            X_axis=np.linspace(self.get_frequency_start()[self.get_X_units()],self.get_frequency_End()[self.get_X_units()],num=self.num_points())
        return X_axis*(self.get_X_units())

    def getData(self, channel=1, typ = 'Scale'): #typ is string
        """ returns the data i.e. the y_axis values of the Vna
        when the typ is 'Format'
        And the Raw data as a complex array when typ Raw"""
        if typ == 'Scale':
            return self.compensate_ElDel()
        elif typ == 'Raw':
            return self.getRawData(channel= channel)
        elif typ == 'Format':
            self.set_format_data(typ)
            return self.getFormattedData(channel= channel)
        else:
            raise ValueError(f'data type {typ} is not valid. Use <Scale> (default), <Raw> or <Format>')
    #@self.wait_to_complete
    def getRawData(self, channel = 1):
        """acquire the complex data for the current trace from the VNA"""
        #get the data
        spectrum=self.visa.query_ascii_values(':CALC{0}:DATA? SDATA'.format(channel),container=np.array)
        spectrum = spectrum[0::2] + 1j*spectrum[1::2]
        return spectrum

    def getFormattedData(self,channel = 1):
        """acquire the current trace from the VNA in the display format"""
        #get the data
        spectrum=self.visa.query_ascii_values(':CALC{0}:DATA? FDATA'.format(channel),container=np.array)
        return spectrum
    def set_format_data(self, frm):
        """supports: MLIN MLOGc PHAS UPHase 'Unwrapped phase' IMAGinary REAL POLar
        FAHRenheit CELSius 
        current        SMIT SADMittance 'Smith Admittance' SWR GDELay 'Group Delay' KELVin
ly  only the phase, real and imaginery were treated """
        if frm == 'PHAS': #the if cluse is obslute currently
            string = 'PHAS'
        elif frm == 'REAL':
            string = 'REAL'
        elif frm == 'IMAG':
            string = 'IMAG'
        else:
            string = frm
        self.visa.write('CALC:FORM {0}'.format(string))


    def power(self,p=None):
        """gets or sets the source power from the VNA in dBm"""
        if p is None:
        #get the source power
            p=float(self.visa.query(':SOUR:POW?'))*dBm
        else:
            self.visa.write(':SOUR:POW {0}'.format(in_dbm(p)))
        return p

    def powerOnOff(self,state=None):
        """gets status of source or turns on/off the source"""
        if state is None:
            state = float(self.visa.query('OUTP?'))
        elif state in [0,1,'on','off','ON','OFF','0','1']:
            #set the source state
            self.visa.write('OUTP ' + str(state))
            state = float(self.visa.query('OUTP?'))
        else:
            raise Exception('state must be 0, 1, on, off, ON, or OFF')
        #read and return the state
        return state
    def get_in_Hz(self, string):
        self.visa.write(string)
        return self.read_float()*Hz
   
    def num_points(self,n=None):
        """set or get the number of points automaticaly changes the number to be odd"""
        if n is None:
            n= int(self.visa.query(':SENS:SWE:POIN?'))
        if (n%2) ==0:
            n = n+1
        self.visa.write(':SENS:SWE:POIN {0}'.format(n))
        return n
    
    def electrical_delay(self,corr=None):
        """gets or sets the electrical delayin labrad units or in seconds"""
        if corr is None:
            #get the electrical delay
            corr = float(self.visa.query('CALC:CORR:EDEL:TIME?'))*s
        elif corr == 'AUTO':
            corr = self.auto_Elec_Delay()
        else:
            self.visa.write('CALC:CORR:EDEL:TIME {0}'.format(in_s(corr)))
        return corr
    
    def bandwidth(self,bw=None):#IF
        """get or set the IF bandwidth in labrad units or in hz"""
        if bw is None:
            bw = self.get_in_Hz(':SENS:BAND?')
        else:
            self.visa.write(':SENS:BAND {0}'.format(in_hz(bw)))
        return bw

    def CwFreq(self,f=None):
        """get or set the CW frequency in labrad units or in hz"""
        if f is None:
            f = self.get_in_Hz(':SENS:FREQ:CW?')
        else:
            self.visa.write('SENS:FREQ:CW {0}'.format(in_hz(f)))
        return f


    def s_parameters(self,channel=1,in_port= None, out_port =None ):
        """set the measurement type baesd on the string params.
        Possible measurement types, 'S11', 'S12', 'S21', 'S22', 'S41', 'S42', etc"""
        if in_port is None or out_port is None: #get the measurement type
            return  str(self.visa.query('CALC:PAR:CAT:EXT? DISP'))[11:14]
        else:
            self.visa.write("CALC{0}:PAR:MOD S{1}{2}".format(channel,in_port,out_port))
    def averages(self, av = None):     # Chen this got deleted, I did my best to fix.  
        """sets or gets the averaging on the VNA. if passed zero then averaging is turned off """
        if av is None: #get number of average counts
            av = self.visa.query(':SENS:AVER:COUN?')
        elif(av < 1 ): #turn off averaging
            self.visa.write(':SENS:AVER 0')
        else: #set the number of averages
            avString = '%d' % av
            self.visa.write(':SENS:AVER 1')
            self.visa.write(':SENS:AVER:COUN '+avString)
        return av
## automatic control of electrical delay
        
    def Exte_StFreq(self,freq =5.0*GHz):
        self.visa.write('SENS:CORR:EXT:AUTO:STAR {0}'.format(in_hz(freq)))
    def Exte_EnFreq(self,freq =7.0*GHz):
        self.visa.write('SENS:CORR:EXT:AUTO:STOP {0}'.format(in_hz(freq)))
    ##
    def compensate_ElDel(self, freq_list=None, RawData=None):
        """gets a raw data and of the pna and the x axis of frequencies and
        returns the data with a compensation for the electrical_delay"""
        
        if RawData is not None and freq_list is not None:
            return RawData*np.exp(1j*(freq_list*self.electrical_delay()*2*np.pi - np.pi*self.get_phase_offset()/180))
        if RawData is None:
                RawData = self.getRawData()
        if self.sweep_type() == 'LIN':
            if freq_list is None:
                freq_list = self.get_X_axis()
            return RawData*np.exp(1j*(freq_list*self.electrical_delay()*2*np.pi - np.pi*self.get_phase_offset()/180))
        elif self.sweep_type() == 'CW':
            return RawData*np.exp(1j*(self.CwFreq()*self.electrical_delay()*2*np.pi - np.pi*self.get_phase_offset()/180))
        else:
            return RawData*np.exp(1j*(freq_list*self.electrical_delay()*2*np.pi - np.pi*self.get_phase_offset()/180))
            
    def auto_Elec_Delay(self, freq = 6.0*GHz, span=0.50*GHz):
        temp_num_points = self.num_points()
        temp_format_typ = self.format_typ()
        temp_freq_range = self.frequencey_span()
        temp_sweep_typ = self.sweep_type()
        self.num_points(n=3001)
        self.format_typ(frm ='PHAS')
        self.sweep_type(typ ='LIN')
        self.frequencey_span(CenFreq=freq, span=span )
        sleep(self.get_sweep_time()['s'])
        self.auto_scale()
        sleep(self.get_sweep_time()['s'])
        self.set_marker_frequencey(freq= freq)
        self.visa.write('CALC:MEAS:MARK:SET DEL')
        self.auto_scale()
        sleep(min(0.1,self.get_sweep_time()['s']))
        self.visa.write('CALC:MEAS:MARK:SET DEL')
        self.auto_scale()
        sleep(self.get_sweep_time()['s'])
        self.frequencey_span(CenFreq=temp_freq_range[0], span=temp_freq_range[1])
        self.num_points(n = temp_num_points)
        self.format_typ(frm = temp_format_typ)
        self.sweep_type(temp_sweep_typ)
        sleep(self.get_sweep_time()['s'])
        self.auto_scale()
        #sleep(self.get_sweep_time()['s'])
        #return self.electrical_delay()
   ### Dual-purpose functions: get or set depending on how many parameters you pass
   ###Return first the center frequencey and as second the span
    def frequencey_span(self, CenFreq = None, span = 0):
        """sets the the frequnecney center and span accepts a list of [frequencey center, range] in labrad frequncies"""
        if CenFreq is None:
            rang = self.frequency_range()
            return  [(rang[0]+rang[1])/2.0, rang[1]-rang[0]]
        else:
            self.frequency_range(fs =[CenFreq-span/2.0,CenFreq+span/2.0])
        return [CenFreq, span]
    def power_sweep(self,rang):
        """sets the power sweep range 
        rang should be an 2X1 array with labrad units of dbm
        starts at rang[0] and ends at rang[1] regardless which of the two is larger"""
        self.set_sweep_power_start(start =rang[0])
        self.set_sweep_power_stop(stp =rang[1])
    def set_marker_frequencey(self,freq=None):
        """sets the location  the marker, currently only works for frequncies should be addapted 
        for time in the future for such as CW sweep.
        note that the marker can only be placed on a mesured point within the span, so typicly I would 
        choose and odd number of points and place the marker in the middle of the span"""
        if freq is None:
            return# This is not finished yet*self.get_X_units()         	
        else:
            self.visa.write('CALC:MEAS:MARK:X {0}'.format(in_hz(freq)))
    def set_triger_source(self, source='EXT'):
        """ sets the source of the triger accepts as source:
            EXT -external trigger
            IMM - a continues internal trigger
            MAN - manual trigger"""
        self.visa.write('TRIG:SOUR {0}'.format(source))
    def set_triger_slope(self, slp='POS'): # Chen: This works now 
        """Specifies the polarity expected by the external trigger input circuitry, accepts slp as:
                POS - (rising Edge) or High Level
                NEG -  (falling Edge) or Low Level"""
        self.visa.write('TRIG:SEQ:SLOP {0}'.format(slp))
    def set_triger_type(self, typ='EDGE'): # Chen: This now works
        """Specifies the type of EXTERNAL trigger input detection used to listen for signals on the Meas Trig IN connectors, accepts typ as:
            EDGE -  PNA responds to the rising and falling edge of a signal.
            LEV -  PNA responds to a level (HIGH or LOW)"""
        self.visa.write('TRIG:SEQ:TYPE {0}'.format(typ))
    def set_following_triger(self, typ='POIN'):
        """Sets and reads the trigger mode for the specified channel. This determines what EACH signal will trigger, accepts typ as:
            CHANl - Each trigger signal causes ALL traces in that channel to be swept.
            SWE - Each Manual or External trigger signal causes ALL traces that share a source port to be swept.
            POIN -- Each Manual or External trigger signal causes one data point to be measured.
            TRA - Allowed ONLY when SENS:SWE:GEN:POIN is enabled. Each trigger signal causes two identical measurements to be triggered separately - one trigger signal is required for each measurement. Other trigger mode settings cause two identical parameters to be measured simultaneously."""
        #self.visa.write('TRIG:CHAN:AUX:INT {0}'.format(typ))
        self.visa.write('SENS:SWE:TRIG:MODE {0}'.format(typ))
    def trig2Cha(self,state ='CURR'):
        """specifies whether a trigger signal is sent to all channels or only the current channel
           ALL - trigger signal is sent to all channels. Also sets SENS:SWEep:TRIG:POINt OFF on ALL channels.
           CURR - trigger signal is sent to only one channel at a time. With each trigger signal, the channel is incremented to the next triggerable channel."""
        self.visa.write('TRIG:SCOP {0}'.format(state))
    def frequency_range(self,fs=None):
        """set or get the frequencey range, in Hz"""
        if fs is None:
            return [self.get_frequency_start(), self.get_frequency_End()]
        else:
            self.visa.write(':SENS:FREQ:STAR {0}'.format(in_hz(fs[0])))
            self.visa.write(':SENS:FREQ:STOP {0}'.format(in_hz(fs[1])))
        return fs
    def sweep_type(self,typ =None):
        if typ is None:
            return self.get_sweep_type()
        elif typ == 'CW' or typ == 'LIN' or typ =='POW':
            self.set_sweep_type(typ)
        return typ
    def phase_offset(self,phase =None):
        if phase is None:
            return self.get_phase_offset()
        else:
            self.set_phase_offset(phase)
        return phase
    def format_typ(self,frm = None):
        if frm is None:
            return self.get_format()
        else:
            self.set_format_data(frm)
        return frm
        
    ### get value functions
    def get_frequency_start(self):
        return self.get_in_Hz(':SENS:FREQ:STAR?')
    def get_frequency_End(self):
        return self.get_in_Hz(':SENS:FREQ:STOP?')
    def get_sweep_type(self):
        #if 'LIN' in self.visa.query('SENS:SWE:TYPE?'):
        #    return 'LIN'
        #elif 'CW' in self.visa.query('SENS:SWE:TYPE?'):
            #return 'CW'
        self.visa.write('SENS:SWE:TYPE?')
        return self.read_string()[:-1]
    def get_sweep_time(self,multi = 1.1):
        self.visa.write('SENSe:SWEep:TIME?')
        return self.read_float()*multi*s
    def get_phase_offset(self):
        self.visa.write('CALC:OFFS:PHAS?')
        return self.read_float()
    def get_format(self):
        self.visa.write('CALC:FORM?')
        return self.read_string()[:-1]
    
    def auto_scale(self):
        self.visa.write('DISP:WIND:TRAC:Y:AUTO')
        
        
    ## set value functions
    def set_source_coupling(self,state = 'ON'):
        """turns Port Power Coupling ON or OFF."""
        self.visa.write('SOUR:POW:COUP {0}'.format(state))
    
    def set_sweep_power_start(self,start):
        """start = sets start power in dBm
         Sets and reads the power sweep start power value for a specific port. This allows uncoupled forward and reverse power sweep ranges. 
         Must also set SENS:SWE:TYPE POWer and SOUR:POW:COUPle OFF."""
        self.visa.write('SOUR:POW:PORT:STAR {0}'.format(labrad_value(start)))
    def set_sweep_power_stop(self,stp):
        """stp = sets stop power in dBm
        (Read-Write) Sets and reads the power sweep stop power value for a specific port. T
        his allows uncoupled forward and reverse power sweep ranges. Must also set SENS:SWE:TYPE POWer and SOUR:POW:COUPle OFF."""
        self.visa.write('SOUR:POW1:PORT:STOP {0}'.format(labrad_value(stp)))

    def set_sweep_type(self,typ):
        """ sets the sweep type of the analyzer can be:
            LINear | LOGarithmic | POWer | CW | SEGMent | PHASe"""
        self.visa.write('SENS:SWE:TYPE {0}'.format(typ))
    def set_phase_offset(self,phase):
        """sets the constant phase offset in degrees"""
        self.visa.write('CALC:OFFS:PHAS {0}'.format(phase))
    def get_sweep_time(self):
        return self.num_points()*(1./(self.bandwidth()))

    def save_state(self,savename=None):
        """save the state of the VNA. default filename is state.sta"""
        if savename is None:
            result = self.visa.write(':MMEM:STOR "state.sta"')
        else:
            result = self.visa.write(':MMEM:STOR "'+str(savename)+'.sta"')
        return result
    def load_state(self,savename=None):
        """load the state of the VNA. default filename is state.sta"""
        if savename is None:
            result = self.visa.write(':MMEM:LOAD "state.sta"')
        else:
            result = self.visa.write(':MMEM:LOAD "'+str(savename)+'.sta"')
        return result
    def Triger_to_cont(self):
        """ sets the trigger mode to continues"""
        self.visa.write('SENS:SWE:MODE CONT')
    def Triger_data(self,Dat_type='Format'): 
        self.visa.write('SENS:SWE:MODE SING')
        sleep(self.get_sweep_time()['s'])
        self.auto_scale()
        sleep(0.1)
        DATA = self.getData(typ =Dat_type)
        X_axis = self.get_X_axis()
        self.visa.write('SENS:SWE:MODE CONT')
        return X_axis, DATA
    
    def trigger_single(self):
        self.visa.write('SENS:SWE:MODE SING')
    
    def Triger_data_cont(self,Dat_type='Format',Avg=60):
        'Added by Shay 22/09/19 to average data from PNA'
        self.visa.write('SENS:SWE:MODE CONT')
        sleep(Avg*self.get_sweep_time()['s'])
        self.auto_scale()
        sleep(0.1)
        DATA = self.getData(typ =Dat_type)
        X_axis = self.get_X_axis()
        """self.visa.write(':SENS:SWE:MODE CONT')"""
        return X_axis,DATA
        

    def freq_sweep(self):
        """start a frequency sweep, wait until complete and acquire the data"""
        #get the old timeout
        timeout = self.visa.timeout
        #set the timeout to None
        self.visa.timeout = None

        # #turn on continuous initiation
        # self.visa.write(':INIT:CONT ON')
        # #set the trigger source to the bus
        # self.visa.write(':TRIG:SOUR BUS')

        # #abort any acqusitions
        # #self.visa.write(':ABOR;*WAI')
        # #trigger
        # self.visa.write(':TRIG:SING')
        # #wait for the acqusition to complete
        # self.visa.query('*OPC?')

        #will's triggering scheme
        self.visa.write('SENS1:AVER 0')
        self.visa.write('SENS1:AVER:COUN 1')
        self.visa.write('SENS:SWE:GRO:COUN 1')

        self.visa.write('SENS:SWE:TIME:AUTO ON;:OUTP ON')
        self.visa.query('*CLS;:SENS:AVER:CLE;:SENS:SWE:MODE GRO;*OPC?')
        # operation_finished = 0
        # while operation_finished == 0:
        #   operation_finished = self.visa.query('*ESR?')


        #acquire the data
        smith=self.getData()

        self.visa.write(':SENS:SWE:MODE CONT')
        #reset the initiation and triggering
        #turn on continuous initiation
        #self.visa.write(':INIT1:CONT ON')
        #set the trigger source to the bus
        #self.visa.write(':TRIG:SOUR INT')
        #reset the timeout
        self.visa.timeout = timeout

        return smith

    def reset_measure(self):
        """set the triggering to be internal with continuous initiation"""
        #reset the initiation and triggering
        #turn on continuous initiation
        
        #set the trigger source to the bus
        self.visa.write(':TRIG:SOUR IMM')
        self.visa.write(':SENS:SWE:MODE CONT')