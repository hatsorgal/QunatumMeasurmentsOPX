# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 12:39:07 2021

@author: Eliya
"""

# Common:
import numpy as np
from scipy import signal, special
from warnings import warn 

# for reltive folder imports:
import pathlib, sys
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\measurementClass')
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\measurementClass')
sys.path.append(r'C:\Users\PHshayg-lab3\Documents\GitHub\MeasurementControl\ConfigurationClasses')
sys.path.append(r'C:\Users\Shay\Documents\GitHub\MeasurementControl\ConfigurationClasses')
currentDir = pathlib.Path(__file__).resolve().parent
ParentDir  = currentDir.parent
UtilsDir   = ParentDir / 'QMUtilities'
ConfigDir  = ParentDir / 'ConfigurationClasses'
sys.path.append(str(UtilsDir))
sys.path.append(str(ConfigDir))
# Our Classes:
from ConfigurationMembers import IntegrationWeights, Controller, MXG
from ConfigurationSuperMembers import PulseParams, Qubit , Readout
from ConfigurationCreator import Config

import matplotlib.pyplot as plt

    
def create_my_config():

    """====================================================================================================================================================================="""
    """===================================================================== Set All Params ================================================================================"""
    """====================================================================================================================================================================="""

    controller = "con1"
    octave_ip = '192.168.0.31' #frige 2
    octave_clock = '1000MHz'
    octave_port = '11051' #fridge 2

    # ports of each element
    I_input = 1
    Q_input = 2
    I_ro    = 1
    Q_ro    = 2
    I_qb1   = 3
    Q_qb1   = 4
    
    
    #DC offsets for the input
    DC_offset_input_I = 0.01474124298095703
    DC_offset_input_Q = 0.01474124298095703
  
    tof = 256+16 #time of flight
    SOf = 1 # scale to prevent input overflow, find something that works for your amplification setup
       
    I_phase_ro =  0.745459390398943
    Q_phase_ro =  1.168628603049243
    I_amp_ro =  0.051550029278282
    Q_amp_ro =  0.050532333715755315
        
    rot_phase = 0

    # DC Offsets for the output
    offsets = [0.0] * 11 #DC offsets 
    
    """ ro  """
    freq_ro = 6444.72e6
    IF_ro = -125e6
    LO_ro = freq_ro-IF_ro
    
    offsets[I_ro] = 0.010968
    offsets[Q_ro] = 0.000416 
    ssb_cor_phase_ro = 18.7329085 
    g_corr_ro= 0.322
    
    
    """ qb1  """
    freq_qb1 = 2.4680719203296597e9
    # freq_qb1 = 8.647e9
    # freq_qb1 = 8.4018e9
    IF_qb1 = -150e6
    LO_qb1 = freq_qb1-IF_qb1
        
    offsets[I_qb1] = -0.0038013
    offsets[Q_qb1] = 0.004832 
    ssb_cor_phase_qb1 = 19.2592397 
    g_corr_qb1= -0.156
    
    #at 15mA
    # offsets[I_qb1] = -0.007144
    # offsets[Q_qb1] = -0.008752 
    # ssb_cor_phase_qb1 = 3.1757475 
    # g_corr_qb1= -0.012
    
    
    """ pulses """
    ramp_length = 32
    rabi_length = 1000
    pi2_length = 128
    
    # qubit1
    qb1_pi2_amp =  0.022
    qb1_pi_amp  =   0.044
    qb1_DRAG_param = 0
    
    qb1_rabi_amp = 0.05
    
    
    # ro
    ro_amp       = 0.065*np.exp(1j*80/180*np.pi)
    ro_pulse_len = 10400
    
    
    """====================================================================================================================================================================="""
    """=============================================================== Configuration Creation =============================================================================="""
    """====================================================================================================================================================================="""
    
    myConfig = Config()
    myConfig.octave_ip = octave_ip
    myConfig.octave_port = octave_port
    myConfig.octave_clock = octave_clock    
    
    
    octave1 = MXG( "lo1", None,  power= 14, inst_type = 'octave', gain = 20,
            is_using_octave_externally = False, trig_mode = 'normal')
    
    octave2 = MXG( "lo2", None,  power= 14, inst_type = 'octave', gain = 20,
            is_using_octave_externally = False, trig_mode = 'normal')

    # Controller:
    myConfig.add( controller=Controller(
        analog_outputs  = offsets ,
        analog_inputs   = [0 ,  DC_offset_input_I , DC_offset_input_Q ] , # first array element is zero because OPX doesn't read it anyway.
        name            = controller,
        typ             = "opx1",
        digital_outputs = [1,3]
        )
    )
    
    
    qb1 = Qubit(
        name = "qb1",
        ICon = I_qb1,
        QCon = Q_qb1,        
        phaseCorrection  = ssb_cor_phase_qb1 ,
        gCorrection      = g_corr_qb1 ,
        PulseParamsList=[ 
            #PulseParams( pulseType , operationName       ,   amp                                       
            PulseParams(  'sin2' , "pi2_pulse"              ,qb1_pi2_amp  , length = pi2_length,    drag_param = qb1_DRAG_param),
            PulseParams(  'sin2' , "pi_pulse"              ,qb1_pi_amp  ,  length = pi2_length,   drag_param = qb1_DRAG_param),
            PulseParams(  'constant' , "rabi_pulse"             ,qb1_rabi_amp     ,   length=rabi_length ).add_ramp('up', ramp_length).add_ramp('down', ramp_length),
            PulseParams(  'constant' , "mixer_cal_pulse"        ,qb1_pi2_amp     ,   length=rabi_length ),
        ],
        freq   = freq_qb1,
        intermediateFreq  = IF_qb1 ,
        digitalInput = dict(channel = 3, delay = 87, buffer = 15),
        MXG =(octave2,'master'),
    )
    myConfig.add(qb1)
    
    
    ro = Readout(
        name = "ro",
        ICon = I_ro,
        QCon = Q_ro,        
        phaseCorrection  = ssb_cor_phase_ro,
        gCorrection      = g_corr_ro,
        PulseParamsList=[ 
            #PulseParams( pulseType , operationName       ,   amp         
            PulseParams(  'constant' , "ro_pulse"         ,   ro_amp     ,   length=ro_pulse_len , operationType="measurement",
                        integrationWeights = IntegrationWeights(SOf ,  I_phase_ro , Q_phase_ro , rot_phase , I_amp_ro , Q_amp_ro  , I_name="integW1_I" , Q_name="integW1_Q"),
                        ).add_ramp("up",ramp_length).add_ramp("down",ramp_length),
            PulseParams(  'constant' , "mixer_cal_pulse"  ,   ro_amp    ,   length=ro_pulse_len ),
        ],
        freq   = freq_ro ,
        intermediateFreq  = IF_ro ,
        time_of_flight  = tof ,
        smearing = 0 ,
        octave_rf_in_port = 1,
        digitalInput = dict(channel = 1, delay = 87, buffer = 30),
        MXG =(octave1,'master'),
    )
    myConfig.add(ro)
    
    
    return myConfig
    """====================================================================================================================================================================="""
    """========================================================================= End ======================================================================================="""
    """====================================================================================================================================================================="""


#%% Fitting older Versions with get_my_config function:
def get_my_config(myConfig = None):
    if myConfig is None:
        myConfig = create_my_config()
        

    return {
        'opx_config': myConfig.get("config"),
        'aux_config': myConfig.get("aux_config") 
    }


#%% Simple script to test and print Config:
if __name__ == "__main__":

    myConfig  = create_my_config()
    
    aux_config = myConfig.get("aux_config")
    
    # myConfig.plotPulse( "ro" , "ro_pulse" )

