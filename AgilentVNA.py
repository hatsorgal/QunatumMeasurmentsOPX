# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 10:56:34 2019

@author: Shay
"""

from InstrumentControl.general_functions import * 
import visa # Working with GPIB control (sorry Dar)
from labrad.units import V,mV, s,ms,us,ns, Hz,kHz,MHz,GHz, Ohm, dBm, Value
from matplotlib.pyplot import pause
import numpy as np
from time import sleep
from warnings import warn

class AgilentVNA(VisaObject):
    def getFreq(self):
        """acquire the frequency values for the current trace from the VNA"""
        spectrum=self.visa.query_ascii_values(':SENS1:FREQ:DATA?',container=np.array)
        return spectrum

    def getData(self):
        """acquire the complex data for the current trace from the VNA"""
        #get the data
        spectrum=self.visa.query_ascii_values(':CALC1:DATA:SDAT?',container=np.array)
        spectrum = spectrum[0::2] + 1j*spectrum[1::2]
        return spectrum

    def getFormattedData(self):
        """acquire the current trace from the VNA in the display format"""
        #get the data
        spectrum=self.visa.query_ascii_values(':CALC1:DATA:FDAT?',container=np.array)
        return spectrum


    def power(self,p=None):
        """gets or sets the source power from the VNA in dBm"""
        if p is None:
        #get the source power
            p=float(self.visa.query(':SOUR1:POW?'))
        else:
            #set the source power
            pString = '%0.5E' % p
            self.visa.write(':SOUR1:POW ' + pString)
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
### Settings ###
    def electrical_delay(self,corr=None):
        """gets or sets the electrical delay"""
        if corr is None:
            #get the electrical delay
            corr = float(self.visa.query(':CALC1:CORR:EDEL:TIME?'))
        else:
            corrString = '%0.3E' % corr
            self.visa.write(':CALC1:CORR:EDEL:TIME ' + corrString)
        return corr

    def bandwidth(self,bw=None):
        """get or set the IF bandwidth"""
        if bw is None:
            bw = float(self.visa.query(':SENS1:BAND?'))
        else:
            bwString= '%f' % bw
            self.visa.write(':SENS1:BAND ' + bwString)
        return bw

    def frequency(self,f=None):
        """get or set the CW frequency"""
        if f is None:
            f = self.visa.query(':SENS1:FREQ?')
        else:
            fString= '%E' % f
            self.visa.write('SENS1:FREQ? ' + fString)
        return f

    def s_parameters(self,params=None):
        """set the measurement type baesd on the string params.
        Possible measurement types, 'S11', 'S12', 'S21', 'S22', 'S41', 'S42', etc"""
        if params is None: #get the measurement type
            params =  self.visa.query(':CALC1:PAR1:DEF?')
        else:
            self.visa.write(':CALC1:PAR1:DEF ' + params)
        return params

    def averages(self,av=None):
        """sets or gets the averaging on the VNA. if passed zero then averaging is turned off """
        if av is None: #get number of average counts
            av = self.visa.query(':SENS1:AVER:COUN?')
        elif(av < 1 ): #turn off averaging
            self.visa.write(':SENS1:AVER 0')
        else: #set the number of averages
            avString = '%d' % av
            self.visa.write(':SENS1:AVER 1')
            self.visa.write(':SENS1:AVER:COUN '+avString)
        return av

    def frequency_range(self,fs=None):
        """set or get the span, in Hz"""
        if fs is None:
            start=float(self.visa.query(':SENS1:FREQ:STAR?'))
            stop=float(self.visa.query(':SENS1:FREQ:STOP?'))
            fs=[start*Hz,stop*Hz]
            print(1)
        else:
            startString = '%f' % fs[0]
            stopString = '%f' % fs[1]
            self.visa.write(':SENS1:FREQ:STAR '+ startString)
            self.visa.write(':SENS1:FREQ:STOP '+ stopString)
        return fs

    def num_points(self,n=None):
        """set or get the number of points"""
        if n is None:
            n=long(self.visa.query(':SENS:SWE:POIN?'))
        else:
            nString = '%u' % long(n)
            self.visa.write(':SENS:SWE:POIN ' + nString)
        return n

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

    def freq_sweep(self):
        """start a frequency sweep, wait until complete and acquire the data"""
        #get the old timeout
        timeout = self.visa.timeout
        #set the timeout to None
        self.visa.timeout = None

        #turn on continuous initiation
        self.visa.write(':INIT1:CONT ON')
        #set the trigger source to the bus
        self.visa.write(':TRIG:SOUR BUS')

        #abort any acqusitions
        #self.visa.write(':ABOR;*WAI')
        #trigger
        self.visa.write(':TRIG:SING')
        #wait for the acqusition to complete
        self.visa.query('*OPC?')

        #acquire the data
        smith=self.getData()

        #reset the initiation and triggering
        #turn on continuous initiation
        #self.visa.write(':INIT1:CONT ON')
        #set the trigger source to the bus
        self.visa.write(':TRIG:SOUR INT')
        #reset the timeout
        self.visa.timeout = timeout

        return smith

    def reset_measure(self):
        """set the triggering to be internal with continuous initiation"""
        #reset the initiation and triggering
        #turn on continuous initiation
        self.visa.write(':INIT1:CONT ON')
        #set the trigger source to the bus
        self.visa.write(':TRIG:SOUR INT')
    def query_float(self, string):
        self.Visa.query(string)
        return  self.read_float()