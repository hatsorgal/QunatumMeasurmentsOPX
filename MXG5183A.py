# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 10:56:32 2019

@author: Shay
"""

from general_functions import *
from general_functions import VisaObject
try:  import visa # Working with GPIB control (sorry Dar)
except: import pyvisa # Working with GPIB control (sorry Dar)
# from labrad.gpib import GPIBDeviceWrapper, GPIBManagedServer
from labrad.units import V,mV, s,ms,us,ns, Hz,kHz,MHz,GHz, Ohm, dBm, Value
from matplotlib.pyplot import pause
import numpy as np
from time import sleep
from warnings import warn


class MXG5183A(VisaObject):
    """
    Control the MXG5183A/B signal generator. Does not rely on Labrad except for units.
    Returns parameters with labrad units. Accepts numbers with or
    without labrad units
    """
    def __init__(self, NetworkAddress, resourceManager = None):
        """Takes a string or integer. Integer is treated as a GPIB address
        Connecting to the VNA VISA driver
        Input: unicode string with VISA address
        creates instrument with name VNA
        """
        if NetworkAddress == 'MIDDLE':
            self.initialize('TCPIP0::192.168.0.4::inst0::INSTR', resourceManager = resourceManager)
        elif NetworkAddress == 'BOTTOM':
            self.initialize('TCPIP0::192.168.0.3::inst0::INSTR', resourceManager = resourceManager)
        elif NetworkAddress =='TOP':
            self.initialize('TCPIP0::192.168.0.5::inst0::INSTR', resourceManager = resourceManager)
        elif '192' in NetworkAddress:
            self.initialize(NetworkAddress, resourceManager = resourceManager)
        else:
            raise ValueError(f"Probably wrong network address. Expected <MIDDLE>, <TOP>, <BOTTOM> or an address with 192 in it. got <{NetworkAddress}>")
    
    def wait_to_complete(self,func):
        def inner(*args, **kwargs):
            x = func(*args, **kwargs)
            self.visa.query('*OPC?')
            return x
        return inner
    
    def on(self):
        self.visa.write('OUTP:STAT ON')
    def off(self):
        self.visa.write('OUTP:STAT OFF')
    def get_output_state(self):
        self.visa.write('OUTP:STAT?')
        return self.read_int()
    
    def atten_prot(self, state):
        if state in ['On', 'ON', 'on', 1, True]:
            state = 'ON'
        elif state in ['OFF', 'off', 'Off', 0, False]:
            state = 'OFF'
        else:
            raise ValueError(f'Invalid state. Got {state}. Expected <ON> or <OFF>.')
        self.visa.write(f':SOUR:SWE:ATT:PROT:STAT {state}')
   
    """19.8.20 Eliya removed Hz from some functions"""
    def set_freq(self, freq): 
        self.visa.write('FREQ:CW '+str(in_hz(freq)))
    def get_freq(self):
        """If no units are given on the input, assumes units of GHz"""
        self.visa.write('FREQ:CW?')
        return self.read_float()
    def freq(self, val=None):
        if val is None:
            return self.get_freq()
        elif type(val) is list:
            self.set_freq_range(val)
        else:
            self.set_freq(val)
    def set_freq_start(self,pt_start=None):
        """This command sets the first frequency point in a step sweep
        pt_start is a labrad unit of ethier  or frequencey"""
        if pt_start is None:
            self.visa.write(':FREQ:START?')
            return self.read_float()
        if type(pt_start)==Value:
            self.visa.write(':FREQ:START {0}{1}'.format(pt_start[pt_start.units],pt_start.units))
        else:
            self.visa.write(f':FREQ:START {pt_start}Hz')
    def set_freq_stop(self,pt_stop=None):
        """This command sets the last frequency point in a step sweep
        pt_stop is a labrad unit of  frequencey"""
        if pt_stop is None:
           self.visa.write(':FREQ:STOP?')
           return self.read_float()
        if type(pt_stop)==Value:
            self.visa.write(':FREQ:STOP {0}{1}'.format(pt_stop[pt_stop.units],pt_stop.units))
        else:
            self.visa.write(f':FREQ:STOP {pt_stop}Hz')
    def set_freq_range(self,freq_range = None):
        """sets the frequncey center (freq_range[0]) and span (freq_range[1]) of a frequncey sweep"""
        if freq_range is None:
            return [(self.set_freq_start()+self.set_stop())/2,(self.set_freq_start()-self.set_stop())]
        if len(freq_range)>2:
            self.set_freq_start(freq_range[0])
            self.set_freq_stop(freq_range[-1])
        else:
            self.set_freq_start(freq_range[0]-freq_range[1]/2)
            self.set_freq_stop(freq_range[0]+freq_range[1]/2)

    def set_power(self, power):
        """If no units are given on the input, assumes units of dBm"""
        self.visa.write('POW:AMPL '+str(in_dbm(power)))
    def get_power(self):
        self.visa.write('POW:AMPL?')
        return self.read_float()*dBm
    
    def power(self, val=None):
        if val is None:
            return self.get_power()
        elif type(val) in [list, np.ndarray]:
            self.set_pwr_range(val)
        elif val=='on':
            self.on()
        elif val=='off':
            self.off()        
        else:
            self.set_power(val)
    def settings(self):
        dict = {}
        dict['Frequency'] = self.freq()
        dict['Power'] = self.power()
        dict['State'] = self.get_output_state()
        return dict
   
    def set_pwr_start(self,pt_start=None):
        """This command sets the first frequency point in a step sweep
        pt_start is a labrad unit of power"""
        if pt_start is None:
            self.visa.query(':POW:START?')
            return self.read_float()*dBm
        if type(pt_start) == Value:
            self.visa.write(':POW:START {0}{1}'.format(pt_start[pt_start.units],pt_start.units))
        else:
            self.visa.write(f':POW:START {pt_start}dBm')
    def set_pwr_stop(self,pt_stop=None):
        """This command sets the last frequency point in a step sweep
        pt_stop is a labrad unit of  power"""
        if pt_stop is None:
            self.visa.query(':POW:STOP?')
            return self.read_float()*dBm
        if type(pt_stop) == Value:
            self.visa.write(':POW:STOP {0}{1}'.format(pt_stop[pt_stop.units],pt_stop.units))
        else:
            self.visa.write(f':POW:STOP {pt_stop}dBm')
    def set_pwr_range(self,p_range=None):
        """sets the power center (p_range[0]) and span (p_range[1]) of a power sweep"""
        if p_range is None:
            return [(self.set_pwr_start()+self.set_pwr_stop())/2,self.set_pwr_start()+self.set_pwr_stop()]
        if len(p_range)>2:
            self.set_pwr_start(p_range[0])
            self.set_pwr_stop(p_range[-1]) 
        else:
            self.set_pwr_start(p_range[0]-(p_range[1]/2))
            self.set_pwr_stop(p_range[0]+(p_range[1]/2)) 
    def attenuator_protection(self,state =0):
        """
        This command enables protection for the mechanical attenuator by
        automatically turning on Atten Hold during frequency and/or power step
        sweeps.
        This may cause unleveled RF output to occur for certain sweep configurations.
        Disabling this attenuator protection will allow the sweep to optimally set both
        the automatic leveling control (ALC) and output attenuation at each sweep
        point.
        ON (1) This choice enables attenuator protection.
        OFF (0) This choice disables attenuator protection. When the
        attenuator protection is disabled, the step dwell time
        will be set to a minimum of 50 ms as a precaution"""
        self.visa.write(':SWE:ATT:PROT {0}'.format(state))
    def Num_SWE_PT(self,Num_of_points=None):
        """This command defines the number of step sweep points"""
        if Num_of_points is None:
            return int(self.visa.query(':SWE:POIN?'))
        self.visa.write(':SWE:POIN {0}'.format(Num_of_points))
   
    def set_SWE_SPAC(self,space = None):
        """This command enables the signal generator LIN or LOG sweep
        modes. These commands require the signal generator to be in step mode.
            The instrument uses the specified start frequency, stop frequency, and number
            of points for both linear and log sweeps."""
        if space is None:
            return self.visa.query(':SWE:SPAC?')
        self.visa.write(':SWE:SPAC {0}'.format(space))
    def sweep_mode(self, SWE_MOD= None):
        """ SET the sweep type of the MXG:
        CONST- maps to single frequency/amplitude. 
        FREQ -maps to frequency sweep.
        PWR maps to power sweep."""
        sweep_mode_dict = {'CONST':'ABOR;:SOUR:FREQ:MODE CW;:SOUR:POW:MODE FIX',
                            'FREQ':'ABOR;:SOUR:FREQ:MODE LIST;:SOUR:POW:MODE FIX',
                            'PWR':'ABOR;:SOUR:FREQ:MODE CW;:SOUR:POW:MODE LIST',
                            'PWR_FREQ':'ABOR;:SOUR:FREQ:MODE LIST;:SOUR:POW:MODE LIST'}
        if SWE_MOD is None:
            return self.visa.query(':SOUR:FREQ:MODE?;:SOUR:POW:MODE?')
        self.visa.write(sweep_mode_dict[SWE_MOD])

    def set_list_sweep(self, freqs, powers, dwells, atten_hold = True):
        if atten_hold:
            self.visa.write(':SOUR:SWE:ATT:PROT:STAT ON')
        self.visa.write(':SOUR:LIST:FREQ '+','.join([str(f) for f in freqs]))
        self.visa.write(':SOUR:LIST:POW '+','.join([str(p) for p in powers]))
        self.visa.write(':SOUR:LIST:DWEL '+','.join([str(d) for d in dwells]))
        
    def set_sweep_type(self, sweep_type):
        "Set sweep type. Can be either <list> or <step>."
        if sweep_type in ['list', 'List', 'LIST']:
            sweep_type == 'LIST'
        elif sweep_type in ['Step', 'step', "STEP"]:
            sweep_type = 'STEP'
        else:
            raise ValueError(f"Invalid sweep type. Got {sweep_type}. Expected <list> or <step>.")
        self.visa.write(f':LIST:TYPE {sweep_type}')
    def set_dwell(self, time=None):
        """This command sets the dwell time for the current list sweep points The variable is expressed in units of seconds with a 0.000001 (mS)."""
        if time is None:
            return self.visa.query('SOUR:SWEep:DWEL?')
        if type(time) == Value:
            self.visa.write('SOUR:SWE:DWEL {0}'.format(time['s']))
        else:
            self.visa.write(f'SOUR:SWE:DWEL {time}')
    """This command sets the dwell time for the current list sweep points The variable is expressed in units of seconds with a 0.000001 (mS)."""
    
    def display(self, remote= 'ON'):
        """This command enables or disables the display updating when the signal
        generator is remotely controlled."""
        self.visa.write(':DIS:REM {0}'.format(remote))

    def set_Sweep_step(self,step=None):
        """This command sets the step size for a by setp_typ to LIN or LOG step sweep in frequency
        The variable step is in labrad units of frequency, specifies by the
        variable <unit> (as Hz, kHz, MHz, or GHz)."""
        self.visa.write(':SWE:STEP: {0}{1}'.format(step[step.units],step.units))
    def EXT_trig_source(self,outport = 'TRIG2'):
        """This command selects the external trigger source. 
        With external triggering, outport is ethier TRIG or TRIG2"""
        self.visa.write(':TRIG:EXT:SOUR {0}'.format(outport))
    def Force_current_sweep(self):
        """This event command causes an armed List or Step sweep to immediately start
        without the selected trigger occurring."""
        self.visa.write('*TRG')
    
    def start_sweep(self, typ = 'CONT'):
        """ abort the previous sweep and starts a new on instead
        'CONT' should start a continuous sweep. 'SING' should start a single sweep."""
        start_sweep_dict = {'CONT':':TRIG:SOUR IMM;:ABOR;:INIT:CONT 1',
                            'SING':':TRIG:SOUR IMM;:ABOR;INIT:CONT 0;:INIT'}
        self.visa.write(start_sweep_dict[typ])
    def abort_sweep(self):
        self.visa.write(':INIT:CONT 0;:ABOR')

    def triger_source(self,SOUR = 'IMM'):
        """:TRIGger[:SEQuence]:SOURce
        BUS|IMMediate|EXTernal|INTernal|KEY|TIMer|MANual
        :TRIGger[:SEQuence]:SOURce?
        This command sets the sweep trigger source for a list or step sweep.
        BUS This choice enables GPIB triggering using the *TRG or
        GET command. The *TRG SCPI command can be used
        with any combination of GPIB, LAN, or USB. The GET
        command requires USB, GPIB, or LAN–VXI–11.
        IMMediate This choice enables immediate triggering of the sweep
        event.
        EXTernal This choice enables the triggering of a sweep event by
        an externally applied signal at the TRIG 1, TRIG 2 or
        PULSE connector (see :TRIGger:EXTernal:SOURce).
        INTernal This choice enables the triggering of a sweep event by
        an internal Pulse Video or Pulse Sync signal (see
        :TRIGger:INTernal:SOURce).
        KEY This choice enables triggering through front panel
        interaction by pressing the Trigger key.
        TIMer This choice enables the sweep trigger timer.
        MANual This choice enables manual sweep triggering."""
        self.visa.write(':LIST:TRIG:SOUR {0}'.format(SOUR))
class DummyGenerator():
    def settings(self):
        return {}
# AnritzuMG


        
if __name__ == '__main__' and False:
    print("Assuming you are on the Sisyphus computer and the AWG is GPIB 14.")
    cgen = MXG5183A('GPIB0::5::INSTR')
    qgen = MXG5183A('GPIB0::19::INSTR')