# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:35:59 2019

@author: Nir Lev

Keysight CXA Signal Analyzer n9000B VISA Controller
"""


from general_functions import *
from general_functions import VisaObject
try: import visa # Working with GPIB control (sorry Dar)
except: import pyvisa # Working with GPIB control (sorry Dar)
# from labrad.gpib import GPIBDeviceWrapper, GPIBManagedServer
from labrad.units import V,mV, s,ms,us,ns, Hz,kHz,MHz,GHz, Ohm, dBm, Value
from matplotlib.pyplot import pause
import numpy as np
from time import sleep, time
from warnings import warn


class CXA_SA(VisaObject):
    """Programming interface for the Keysight N9000B Signal Analyzer, allows
    calling of remote commands to the device.
    """
    def __init__(self, NetworkAddress = 'TCPIP::192.168.0.12::INSTR', resourceManager = None):
        """ Constructor
        :param filename: file path to YAML configuration file or a configuration dictionary
        """
        
        # initialize low-level communication.
        self.initialize(NetworkAddress, resourceManager = resourceManager)
        
        #Initialize functionality of the class
        self.Peak = _Peak(self)
        self.Freq = _Freq(self)
        self.Bandwidth = _Bandwidth(self)
        self.Marker = _Marker(self)
        
    def Amp(self, amp):
        self.visa.write(f':DISP:WIND:TRAC:Y:SCAL:RLEV {amp}')
        
    def Get_Config(self):
        """ returns a copy of the SA config"""
        return self._config.copy()
    
class _Peak(object):
    """Subclass containing Peak Search features of the Signal Analyzer
    """
    
    #Needs to be given from configuration
    SORT = ['FREQ', 'AMPL', 'DELT']
    READOUT = ['ALL', 'GTDL', 'LTDL']
    STATE = ['ON','OFF']
    
    # There definitely exists a better data pattern to accessing to a refreence 
    # to the main class's variable,
    def __init__(self,parent= CXA_SA):
        """Passing the parent CXA_SA object, in order to access the read/write 
        functionality
        """
        self._CXA_SA = parent
    
    def Set_Search_Criteria(self, mode='MAX'):
        """Set the search mode for peak search
        
        :param mode: MAX, corresponds to the Highest Peak setting -or- PARA, corresponds to the Same as “Next Peak” Criteria setting
        
        """
        self._CXA_SA.visa.write(':CALC:MARK:PEAK:SEAR:MODE ' + mode)
        
    def Threshold_Ampl(self, ampl=None):
        """set - Turns the peak threshold requirement on/off and sets the threshold value. The peak
        threshold value defines the minimum signal level (or min threshold) that the peak
        identification algorithm uses to recognize a peak.
        
        get - reads the threshold value
        
        :param ampl: Threshold amplitude, in dBm. If not set, the function queries the thershold value from the instrument
        
        """
        
        if ampl != None:
            self._CXA_SA.visa.write('CALC:MARK:PEAK:THR '+ str(ampl) + ' dBm')
        else:
            self._CXA_SA.visa.write('CALC:MARK:PEAK:THR?')            
            return self._CXA_SA.read_float()
        
    def Threshold_State(self, state=None):
        """Sets or gets the threshold state        
        :param state: threshold state - ON or OFF
        """
        if state != None:
            if state in self.STATE:
                self._CXA_SA.visa.write('CALC:MARK:PEAK:THR:STAT ' + state)
            else:
                raise Exception('state paramter receives ' + str(self.STATE) +' . Received - ' + str(state))
        else:
            self._CXA_SA.visa.write('CALC:MARK:PEAK:THR:STAT?')
            return self._CXA_SA.read_string()
    
    def Excursion(self, rel_ampl=None):
        """Turns the peak excursion requirement on/off and sets the excursion value. The value
        defines the minimum amplitude variation (rise and fall) required for a signal to be
        identified as peak. For example, if a value of
        6 dB is selected, peak search functions like the marker Next Pk Right function move
        only to peaks that rise and fall 6 dB or more.

        get - reads the excursion value
        
        :param rel_ampl: minimum peak excursion, in dBm. If not set, the function queries the minimum peak excursion value from the instrument
        
        """
        if rel_ampl != None:
            self._CXA_SA.visa.write('CALC:MARK:PEAK:EXC '+ str(rel_ampl) + ' dB')
        else:
            self._CXA_SA.visa.write('CALC:MARK:PEAK:EXC?')
            return self._CXA_SA.read_float()
        
    def Sort(self,sort=None):
        """Sets or gets the peak table sorting routine to list the peaks in order of descending
        amplitude, ascending frequency or descending “Delta to Limit” value
        
        :param sort: sorting method - FREQ - ascending frequency, AMPL - descending amplitude, DELT - descending "Delta to Limit"
        """
        if sort != None:
            if sort in self.SORT:
                self._CXA_SA.visa.write('CALC:MARK:PEAK:SORT ' + sort)
            else:
                raise Exception('sort paramter receives ' + str(self.SORT) +' . Received - ' + str(sort))
        else:
            self._CXA_SA.visa.write('CALC:MARK:PEAK:SORT?')
            return self._CXA_SA.read_string()
        
    def Readout(self,readout=None):
        """Shows up to twenty signal peaks as defined by the setting:
        All (ALL) - lists all the peaks defined by the peak criteria, in the current sort setting.
        Above Display Line (GTDLine) - lists the peaks that are greater than the defined
        display line, and that meet the peak criteria. They are listed in the current sort order.
        Below Display Line (LTDLine) - lists the peaks that are less than the defined display
        line, and that meet the peak criteria. They are listed in the current sort order.
        If the peak threshold is defined and turned on, then the peaks must meet this peak
        criteria in addition to the display line requirements
        
        :param readout: ALL, GTDL, LTDL
        """
        if readout != None:
            if readout in self.READOUT:
                self._CXA_SA.visa.write('CALC:MARK:PEAK:TABL:READ ' + readout)   
            else:
                raise Exception('readout paramter receives ' + str(self.READOUT) + '. Received - ' + str(readout))
        else:
            self._CXA_SA.visa.write('CALC:MARK:PEAK:TABL:READ?')
            return self._CXA_SA.read_string()
        
    def Num_of_Peaks(self):
        """Outputs the number of signal peaks identified. The amplitude of the peaks can then
        be queried with :TRAC:MATH:PEAK:DATA?. This command uses only Trace 1
        data.
        """
        self._CXA_SA.visa.write('TRAC:MATH:PEAK:POIN?')
        return int(self._CXA_SA.read_string())
    
    def Get_Peaks(self):
        """Will identify the peaks of trace 1 that are above the Peak Threshold (if
        Threshold is ON) and have an excursion above the Peak Excursion (if Excursion is ON).
        """
        self._CXA_SA.visa.write('TRAC:MATH:PEAK?')
        result = self._CXA_SA.read_string()
        splitted_string = result.split(',')
        freq = [float(x) for x in splitted_string[::2]]
        ampl = [float(x) for x in splitted_string[1::2]]
        return list(zip(freq,ampl))
  
    def Find_peak(self,peak_freq):
        """ Ensures that the SA can recognize the LOL peak """
        # get peaks
        peaks = self._CXA_SA.Peak.Get_Peaks()

        # Find peaks within 10 Hz of the expected LO_freq
        peak_of_interest = [peak for peak in peaks if ((peak[0] <= (peak_freq + 10)) and ((peak_freq - 10) <= peak[0]))]
    
        if peak_of_interest == None or len(peak_of_interest) == 0:
            print('ERROR - peak not found around frequency ' + str(peak_freq))
            print(str(peaks))
            raise RuntimeError('')
        
        return peak_of_interest[0]
    
class _Freq(object):
    """Subclass containing Frequency features of the Signal Analyzer
    """
    
    SCALE = ['LIN', 'LOG']
    
    def __init__(self,parent= CXA_SA):
        """Passing the parent CXA_SA object, in order to access the read/write 
        functionality
        """
        self._CXA_SA = parent
    
    def Auto_Tune(self):
        """causes the analyzer to
        change Center Frequency to the strongest signal in the tunable span of the
        analyzer, excluding the LO. It is designed to quickly get you to the most likely signal
        (s) of interest, with no signal analysis knowledge required. As such, there are no
        configurable parameters for this feature. There are only pre-selected values that
        work in most real world situations.
        
        Auto Tune performs a Preset as part of its function, so it always returns you to the
        Normal View and a preset state, although it does leave the AC/DC coupling and
        Single/Cont state unaffected.


        """
        self._CXA_SA.visa.write('FREQ:TUNE:IMM')
        
    def Zoom_Center(self, frequency=None):
        """Sets the frequency that corresponds to the horizontal center of the graticule (when
        frequency Scale Type is set to linear). While adjusting the Center Frequency the
        Span is held constant, which means that both Start Frequency and Stop Frequency
        will change.
        
        Pressing Center Freq also sets the frequency entry mode to Center/Span. In
        Center/Span mode, the center frequency and span values are displayed below the
        graticule, and the default active function in the Frequency menu is Center Freq.
        The center frequency setting is the same for all measurements within a mode, that
        is, it is Meas Global. Some modes are also able to share a Mode Global center
        frequency value. If this is the case, the Mode will have a Global Settings key in its
        
        Mode Setup menu.
        The Center Freq function sets (and queries) the Center Frequency for the currently
        selected input. If your analyzer has multiple inputs, and you select another input, the
        Center Freq changes to the value for that input. SCPI commands are available to
        directly set the Center Freq for a specific input.
       
        Center Freq is remembered as you go from input to input. Thus you can set a Center
        Freq of 10 GHz with the RF Input selected, change to BBIQ and set a Center Freq of
        20 MHz, then switch to External Mixing and set a Center Freq of 60 GHz, and when
        you go back to the RF Input the Center Freq will go back to 10 GHz; back to BBIQ and
        it is 20 MHz; back to External Mixing and it is 60 GHz.

        get - reads currently set center frequency

        :param frequency: frequnecy in MHz to center graitucle upon, as a double
        """
        if frequency != None:
            self._CXA_SA.visa.write('FREQ:CENT '+ str(frequency) + ' MHz')
        else:
            self._CXA_SA.visa.write('FREQ:CENT?')            
            return self._CXA_SA.read_float()
    
    def Scale_Type(self, scale=None):
        """Selects either linear or logarithmic scaling for the frequency axis.
        get - reads the excursion value
        
        :param rel_ampl: minimum peak excursion, in dBm. If not set, the function queries the minimum peak excursion value from the instrument
        
        """
        if scale != None:
            if scale in self.SCALE:
                self._CXA_SA.visa.write('DISP:WIND:TRAC:X:SPAC '+ scale)
            else:
                raise Exception('scale paramter receives ' + str(self.SCALE)+  '. Received - ' + str(scale))
        else:
            self._CXA_SA.visa.write('DISP:WIND:TRAC:X:SPAC?')
            return self.CXA_SA.read_float()
        
    def Span(self,span=None):
        """Sets or gets the frequency span, changing the start and end frequencies
        of the graitucle around the center frequency defined in the instrument.
        
        By setting the span value to 0 , the Span can only go as far down as 10 Hz 
        and cannot actually be set to zero. In this case, the analyzer becomes
        time domain. Any span different than 0 is frequency domain.
        
        :param span: Frequency span in MHz, as a double
        """
        if span != None:
            if span < 30:
                print('WARNING: Signal Analyzer does not seem to find peaks for span values below 30 MHz')
            self._CXA_SA.visa.write('FREQ:SPAN ' + str(span) + 'MHz')
        else:
            self._CXA_SA.visa.write('FREQ:SPAN?')
            return self._CXA_SA.read_string()
        
    def Signal_Track(self,state=None):
        """Set or gets whether the instrument is set to track the marker as the
        center frequency, changing as the marker's frequency changes. This could be
        useful when the top peak frequency varies.
        :param state: boolean, 0 or 1
        """
        if state != None:
            state_val = 1 if state else 0               
            self._CXA_SA.visa.write('CALC:MARK:TRCK ' + str(state_val))     
            
        else:
            self._CXA_SA.visa.write('CALC:MARK:TRCK?')
            return self.CXA_SA.read_string()
        
class _Bandwidth(object):
    """Subclass containing Bandwidth features of the Signal Analyzer
    """
    
    def __init__(self,parent= CXA_SA):
        """Passing the parent CXA_SA object, in order to access the read/write 
        functionality
        """
        self._CXA_SA = parent
        
    def ResBW(self,frequency=None):
        """Set or gets the bandwidth resolution
        :param frequency: Bandwidth resolution in KHz
        """
        if frequency != None:
            self._CXA_SA.visa.write('SENS:BAND:RES '+ str(frequency) + ' KHZ')
        else:
            self._CXA_SA.visa.write('SENS:BAND:RES?')            
            return self._CXA_SA.read_float()
        
    def VidBW(self,frequency=None):
        """Set or gets the video bandwidth 
        :param frequency: Video bandwidth in KHz
        """
        if frequency != None:
            self._CXA_SA.visa.write('SENS:BAND:VID '+ str(frequency) + 'KHZ')
        else:
            self._CXA_SA.visa.write('SENS:BAND:VID?')            
            return self._CXA_SA.read_float()


class _Marker(object):
    """Subclass containing Marker features of the Signal Analyzer
    """
    
    MODE = ['POS', 'DELT', 'FIX', 'OFF']
    
    def __init__(self,parent= CXA_SA):
        """Passing the parent CXA_SA object, in order to access the read/write 
        functionality
        Default mode is POS
        """
        self._CXA_SA = parent
        
    def Marker_Mode(self,marker, mode=None):
        if mode != None:
            if mode in self.MODE:
                self._CXA_SA.visa.write('CALC:MARK' + str(marker) + ':MODE '+ mode)
            else:
                raise Exception('mode paramter receives ' + str(self.MODE)+  '. Received - ' + str(mode))
        else:
            self._CXA_SA.visa.write('CALC:MARK' + str(marker) + ':MODE?')
            return self._CXA_SA.read_string()
        
    def Frequency(self, marker, frequency=None):
        """gets or sets the marker frequency in Hz. Returns 0 Hz frequency if marker mode is off
        :param frequency: Marker frequency in KHz"""
        
        if frequency != None:
            self._CXA_SA.visa.write('CALC:MARK' + str(marker) + ':X ' + str(frequency) + 'MHZ')         
        else:
            # First confirm mode
            if self.Marker_Mode() == 'OFF':
                print('Marker mode is off - cannot read marker data')
                return float(0)
            
            self._CXA_SA.visa.write('CALC:MARK' + str(marker) + ':X?')            
            return self._CXA_SA.read_float()
        
    def Get_Power(self, marker):
        """gets the marker power in dBm"""
        # First confirm mode
        if self.Marker_Mode(marker) == 'OFF':
            print('Marker mode is off - cannot read marker data')
            return float(0)
        
        self._CXA_SA.visa.write('CALC:MARK' + str(marker) + ':Y?')            
        return self._CXA_SA.read_float()