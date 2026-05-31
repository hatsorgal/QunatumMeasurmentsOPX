# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 10:50:18 2019

@author: Shay
"""
try: import visa
except: import pyvisa
from labrad.units import V,mV, s,ms,us,ns, Hz,kHz,MHz,GHz, Ohm, dBm,A,mA, Value
from matplotlib.pyplot import pause
import numpy as np
from time import sleep
from warnings import warn
import pyvisa


class VisaObject:
    def __init__(self, NetworkAddress,  resourceManager = None):
        """Takes a string or integer. Integer is treated as a GPIB address"""
        self.initialize(NetworkAddress, resourceManager = resourceManager)
        
    def initialize(self, NetworkAddress, resourceManager = None):
        """Takes a string or integer. Integer is treated as a GPIB address"""
        if type(NetworkAddress) is int:
            NetworkAddress = 'GPIB0::'+str(NetworkAddress)+'::INSTR'
        if resourceManager is None: 
            try: self.rm = visa.ResourceManager()
            except: self.rm = pyvisa.highlevel.ResourceManager()
        else: self.rm = resourceManager
        self.visa = self.rm.open_resource(NetworkAddress)

    def read_float(self):
        """After quierying a float, call this function to retrieve the requested value"""
        dat = self.visa.read()
        self.clear()
        return float(dat)
    def read_int(self):
        """After quierying an int, call this function to retrieve the requested value"""
        dat = self.visa.read()
        self.clear()
        return int(dat)
    def read_string(self):
        dat = self.visa.read()
        self.clear()
        #print(str(dat))
        return str(dat)
    def clear(self):
        """Clear the PC visa buffer. Sometimes things get out of sync and start acting wierdly"""
        self.visa.clear()

def check_config_4None(dictionary):
        """To back up settings, create previous_cfg = cfg.copy(). cfg = previous_cfg only reasigns your local pointer
        pass previous_cfg to this function and it will copy the settings into the active cfg object"""
        # global cfg # 7/11/2017: removed global variables from how this works. Now you need to pass cfg and the previous cfg to restore
        for key in dictionary.keys():
            if type(dictionary[key]) is list:
                for i in (range(len(dictionary[key]))):
                    dictionary[key][i] = String_to_other(dictionary[key][i])  
                else:    
                    dictionary[key] = String_to_other(dictionary[key])
        return dictionary

def String_to_other(Var, string = 'None', New_Var=None):
    if Var == string:
        return New_Var
    return Var
def in_mA(value):
    if type(value)==Value:
        return value['mA']
    else:
        return value
def in_Amper(value):
    if type(value)==Value:
        return value['mA']
    else:
        return value
def in_volts(value):
    """If value is a labrad Value object with units of V, mV, uV etc., convert it to a float in units of V
    otherwise, return the original int or float, which is taken to be in volts"""
    if type(value)==Value:
        return value['V']
    else:
        return value
def in_ghz(value):
    """If value is a labrad Value object with units of Hz, MHz, GHz etc., convert it to a float in units of GHz
    otherwise, return the original int or float, which is taken to be in GHz"""
    if type(value)==Value:
        return value['GHz']
    else:
        return value
def in_hz(value):
    """If value is a labrad Value object with units of Hz, MHz, GHz etc., convert it to a float in units of GHz
    otherwise, return the original int or float, which is taken to be in GHz"""
    if type(value)==Value:
        return value['Hz']
    else:
        return value
def in_s(value):
    """If value is a labrad Value object with units of s, ms, us etc., convert it to a float in units of s
    otherwise, return the original int or float, which is taken to be in s"""
    if type(value)==Value:
        return value['s']
    else:
        return value
def in_dbm(value):
    """If value is a labrad Value object with units of dB, dBm etc., convert it to a float in units of dBm
    otherwise, return the original int or float, which is taken to be in dBm"""
    if type(value)==Value:
        return value['dBm']
    else:
        return value

def in_bool(value):
    """if value is True/False, convert to 1/0"""
    if value is False:
        return 0
    elif value is True:
        return 1
def labrad_value(var): #returns the unitless value of a lab rad or regular var
    try:
        return var[var.units]
    except:
        return var
def labrad_units(var): #returns the unitless value of a lab rad or regular var
    try:
        return var.units
    except:
        return None