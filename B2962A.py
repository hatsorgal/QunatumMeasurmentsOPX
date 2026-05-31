# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 10:56:32 2019

@author: Shay
"""

from general_functions import *
from general_functions import VisaObject
try: import visa # Working with GPIB control (sorry Dar)
except: import pyvisa
# from labrad.gpib import GPIBDeviceWrapper, GPIBManagedServer
from labrad.units import V,mV, s,ms,us,ns, Hz,kHz,MHz,GHz, Ohm, dBm,A, mA,Value
from matplotlib.pyplot import pause
import numpy as np
from time import sleep
from warnings import warn

class B2962A(VisaObject):
    """
    Control the B2962A power source. Does not rely on Labrad except for units.
    Returns parameters with labrad units. Accepts numbers with or
    without labrad units
    """
    def __init__(self, NetworkAddress='TCPIP0::192.168.0.6::inst0::INSTR', channel = 1, resourceManager = None, ):
        """Takes a string or integer. Integer is treated as a GPIB address
        Connecting to the VNA VISA driver
        Input: unicode string with VISA address
        creates instrument with name VNA
        """
        self.initialize(NetworkAddress, resourceManager = resourceManager)
        self.channel = channel
        
    def wait_to_complete(self,func):
        def inner(*args, **kwargs):
            x = func(*args, **kwargs)
            self.visa.query('*OPC?')
            return x
        return inner
    
    def curr(self, channel = None):
        channel = channel if channel is not None else self.channel
        write_cmd_string = ''.join(str for str in ['SOUR',str(channel),':FUNC:MODE CURR'])
        self.visa.write(write_cmd_string)
        
    def volt(self, channel = None):
        channel = channel if channel is not None else self.channel
        write_cmd_string = ''.join(str for str in ['SOUR',str(channel),':FUNC:MODE VOLT'])
        self.visa.write(write_cmd_string)
        
    def protect_on(self, channel = None):
        channel = channel if channel is not None else self.channel
        write_cmd_string = ''.join(str for str in [':OUTP:PROT ON'])
        self.visa.write(write_cmd_string)
        
    def set_curr(self, current, channel = None):
        """In A"""
        channel = channel if channel is not None else self.channel
        self.curr(channel)
        write_cmd_string = ''.join(str for str in ['SOUR',str(channel),':CURR {0}'.format(current)])
        self.visa.write(write_cmd_string)
        
    def set_curr_smooth(self, current, channel = None, step_size = 5e-5, minimum_current = 1e-5):
        channel = channel if channel is not None else self.channel
        self.curr(channel)
        current_curr = self.get_curr(channel = channel)
        
        if np.abs(current_curr) < minimum_current:  # if very small current current or off
            if current < 0 : 
                self.set_curr(-minimum_current, channel = channel)
                current_curr = -minimum_current           
            else: 
                self.set_curr(minimum_current, channel = channel)
                current_curr = minimum_current  
            self.on(channel = channel)
            
        elif current_curr * current < 0: #if crossing 0
            if current_curr < 0:
                for curr_step in np.linspace(current_curr, -minimum_current,  int(np.ceil(np.abs(current_curr-(-minimum_current))/step_size))+1):
                    self.set_curr(curr_step, channel = channel)
                    sleep(0.1)
                self.set_curr(minimum_current, channel = channel)
                current_curr = minimum_current
            else:
                for curr_step in np.linspace(current_curr, minimum_current,  int(np.ceil(np.abs(current_curr-(minimum_current))/step_size))+1):
                    self.set_curr(curr_step, channel = channel)
                    sleep(0.1)
                self.set_curr(-minimum_current, channel = channel)
                current_curr = -minimum_current
                    

        for curr_step in np.linspace(current_curr, current,  int(np.ceil(np.abs(current_curr-current)/step_size))+1):
            self.set_curr(curr_step, channel = channel)
            sleep(0.1)

    def set_volt(self, channel=None, voltage=0*V):
        """In Volts"""
        channel = channel if channel is not None else self.channel
        self.volt(channel)
        write_cmd_string = ''.join(str for str in ['SOUR',str(channel),':CURR '.format(in_volts(voltage))])
        self.visa.write(write_cmd_string)
        
    def set_volt_lim(self,voltage = 0.03, channel = None ):
        channel = channel if channel is not None else self.channel
        write_cmd_string = ''.join(str for str in [':SENS:VOLT:PROT {}'.format(in_volts(voltage))])
        self.visa.write(write_cmd_string) 
        
    def on(self, channel = None):
        channel = channel if channel is not None else self.channel
        write_cmd_string = ''.join(str for str in ['OUTP',str(channel),' ON'])
        self.visa.write(write_cmd_string)
        
    def off(self, channel = None):
        channel = channel if channel is not None else self.channel
        write_cmd_string = ''.join(str for str in ['OUTP',str(channel),' OFF'])
        self.visa.write(write_cmd_string)
    
    def get_curr(self, channel = None):
        channel = channel if channel is not None else self.channel
        write_cmd_string = ''.join(str for str in [':sens:func ""curr""'])
        self.visa.write(write_cmd_string) 
        write_cmd_string = ''.join(str for str in ['MEAS?', f' (@{channel})'])
        self.visa.write(write_cmd_string)
        string = self.read_string()
        meas = [float(x) for x in string.split(',')]
        return  meas[1]
    
    def get_output_state(self, channel = None):
        channel = channel if channel is not None else self.channel
        write_cmd_string = ''.join(str for str in ['OUTP',str(channel),':STAT?'])
        self.visa.write(write_cmd_string)
        return self.read_int()

    def settings(self):
        dict = {}
        dict['Current'] = self.set_curr()
        dict['Voltage'] = self.set_volt()
        dict['State'] = self.get_output_state()
        return dict
   
