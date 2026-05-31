# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 11:28:41 2025

@author: Shay
"""
import numpy as np
from qm.qua import *

def prepare_coherent(mm, is_phase = False, amp_scale = 1, **kwargs):
    play('disp' * amp(amp_scale), mm)
    align('qb1', mm)
    if is_phase: frame_rotation_2pi(0.25, mm)
    
def prepare_coherent_and_cool(mm, is_phase = False, amp_scale = 1, tido_class = None, sb_duration = None, **kwargs):
    prepare_coherent(mm, is_phase = False, amp_scale = amp_scale)
    tido_class.sideband_cool(sb_duration = sb_duration, mm = mm, **kwargs)
    
def prepare_coherent_and_wait(wait_time, mm, is_phase = False, amp_scale = 1, **kwargs):
    prepare_coherent(mm, is_phase = False, amp_scale = amp_scale)
    if wait_time!=0: 
        wait(int(wait_time//4), mm)
        align('qb1', mm, 'ro')

def prepare_condisp(is_phase = True, amp_scale = 1, is_pi = False, **kwargs):
    if is_pi:
        play('pi_pulse', 'qb1')
        align('qb1', mm)
    tido_class.play_con_disp('qb1', mm, amp_scale = amp_scale)
    align('qb1', mm)
    if is_pi:
        play('pi_pulse', 'qb1')
        align('qb1', mm)
    if is_phase: frame_rotation_2pi(0.25, mm)
    
def prepare_cat(condisp_amp, cat_amp_scale, **kwargs):
    play('pi2_pulse', 'qb1')
    align('qb1', mm)
    tido_class.play_con_disp('qb1', mm, amp_scale = cat_amp_scale)
    align('qb1', mm)
    play('pi2_pulse', 'qb1')
    align('qb1', mm)
    frame_rotation_2pi(0.25, mm)    
    tido_class.play_con_disp('qb1', mm, amp_scale = np.pi/4/cat_amp_scale/condisp_amp**2)
    frame_rotation_2pi(-0.25, mm)    
    frame_rotation_2pi(-0.25, 'qb1')    
    align('qb1', mm)
    play('pi2_pulse', 'qb1')
    align('qb1', mm, 'ro')
    
    
def sb_cool(**kwargs):
    tido_class.sideband_cool(mm=mm, mm_sideband_cooling_pulse = 'constant_sideband_cooling_pulse', 
                       sb_extra_duration = 0, sb_wait_time = 2000, sb_steady_time = 0,
                       is_ramp = True,
                       is_pi2 = True,
                       rabi_sideband_cooling_pulse = 'rabi_pulse',
                       sideband_pulse='constant_sideband_cooling_pulse',
                       ro_detuning = tido_class.results['parameters']['ro']['sideband_detuning'],
                       qb_detuning = tido_class.results['parameters']['qb1']['sideband_detuning'],
                       mm_detuning = tido_class.results['parameters'][mm]['sideband_detuning'])
    
def prepare_thermal_and_wait(wait_time, nbar, disp0, mm, **kwargs):
    prepare_thermal(nbar, disp0, mm)
    if wait_time!=0: 
        wait(int(wait_time//4), mm)
        align('qb1', mm, 'ro')
    
def prepare_thermal(nbar, disp0, mm, **kwargs):
    def bose(betaOmega):
        return(1/(np.exp(betaOmega)-1))
    betaOmega = np.log(1/nbar+1)
    
    length = 1500
    random_amps = np.zeros(length)
    for i in range(length):
        u = np.random.rand()
        x2 = np.sqrt(-np.log((1-u))*bose(betaOmega))/disp0
        random_amps[i] = x2
    # plt.figure()
    # plt.hist(random_amps)
    print('Avg. amp scale = ', np.mean(random_amps), '+-', np.std(random_amps))
    print('Avg.[(amp scale * disp0)^2] = ', np.mean((random_amps*disp0)**2))
    if np.max(random_amps)>1.99:
        raise ValueError(f"Random amp scale is too big. Got {np.round(np.max(random_amps),3)} but the OPX can't deal with more than 2.")
        
    rand_amp = declare(fixed, value = random_amps)
    rand_amp_index = Random()
        
    align('qb1', mm)
    random_ind = declare(int)
    assign(random_ind, rand_amp_index.rand_int(length-1))
    play('disp' * amp(rand_amp[random_ind]), mm)
    
    random_phase = Random()
    ph = declare(fixed)
    assign(ph, random_phase.rand_fixed())
    frame_rotation_2pi(ph, mm)

    align('qb1', mm, 'ro')
    
def prepare_thermal_and_cool(nbar, disp0, mm, tido_class, sb_duration = None, **kwargs):
    prepare_thermal(nbar, disp0, mm)
    tido_class.sideband_cool(sb_duration = sb_duration, mm = mm, **kwargs)
    
def prepare_fock_and_transfer(tido_class, create_dict, mm = 'mm2', mm_sec = 'mm1', qubit = 'qb1',
                              qb_detuning = 0, mm_detuning = 0, mm_sec_detuning = 0, qubit_init = 'minus',
                              rabi_pulse = 'rabi_coupling', mm_sideband_cooling_pulse = 'constant_sideband_cooling_pulse',
                              duration = 0, is_mid_M1 = False, **kwargs):
    prepare_fock(**create_dict)
    
    if is_mid_M1: 
        align('ro', mm, qubit)
        tido_class.perform_full_measurement(is_wait = False,**kwargs)
        if tido_class.wait_after_reset >= 16:
            wait(tido_class.wait_after_reset//4, 'ro')
            align('ro',qubit)
    
    
    align(qubit,mm,mm_sec,'ro')
    if qb_detuning!=0: update_frequency(qubit, tido_class.element_IF(qubit) + qb_detuning, keep_phase = True)
    if mm_detuning!=0: update_frequency(mm, tido_class.element_IF(mm) + mm_detuning, keep_phase = True)
    if mm_sec_detuning!=0: update_frequency(mm_sec, tido_class.element_IF(mm_sec) + mm_sec_detuning, keep_phase = True)
    
    if qubit_init == 'plus' : init_phase = 0.25
    elif qubit_init == 'minus': init_phase = -0.25
    else: raise ValueError(f"Unknown <qubit_init>. Expected <'minus'> or <'plus'>, got {qubit_init}")
    
    rabi_ramp_up_time = int(tido_class.pulse_ramp_len(qubit, rabi_pulse))
    pi2_time = int(tido_class.pulse_len(qubit, 'pi2_pulse'))
    sb_ramp_up_time = int(tido_class.pulse_ramp_len(mm, mm_sideband_cooling_pulse))
    sb_time = (duration + 2*rabi_ramp_up_time + pi2_time*2)
    duration = int(duration)

    if duration != 0:
        play(mm_sideband_cooling_pulse + '_ramp_up', mm)
        play(mm_sideband_cooling_pulse, mm, duration = sb_time//4)
        play(mm_sideband_cooling_pulse + '_ramp_down', mm)
        
        play(mm_sideband_cooling_pulse + '_ramp_up', mm_sec)
        play(mm_sideband_cooling_pulse, mm_sec, duration = sb_time//4)
        play(mm_sideband_cooling_pulse + '_ramp_down', mm_sec)
        
        wait(sb_ramp_up_time//4, 'qb1')
        play('pi2_pulse', 'qb1')
        frame_rotation_2pi(init_phase, 'qb1')
        play(rabi_pulse + '_ramp_up', 'qb1')
        play(rabi_pulse, 'qb1', duration = duration//4)
        play(rabi_pulse + '_ramp_down', qubit)
        frame_rotation_2pi(init_phase, 'qb1')
        play('pi2_pulse', 'qb1')
        align('qb1', mm, mm_sec, 'ro')
    if qb_detuning!=0: update_frequency(qubit, tido_class.element_IF(qubit), keep_phase = True)
    if mm_detuning!=0: update_frequency(mm, tido_class.element_IF(mm), keep_phase = True)
    if mm_sec_detuning!=0: update_frequency(mm_sec, tido_class.element_IF(mm_sec), keep_phase = True)
    
def prepare_fock_and_transfer_OPX(tido_class, create_dict, mm = 'mm2', mm_sec = 'mm1', qubit = 'qb1',
                              qb_detuning = 0, mm_detuning = 0, mm_sec_detuning = 0, qubit_init = 'minus',
                              rabi_pulse = 'rabi_coupling', mm_sideband_cooling_pulse = 'constant_sideband_cooling_pulse',
                              duration = 0, sb_time = 0, sb_ramp_up_time = 0, is_mid_M1 = False, **kwargs):
    prepare_fock(**create_dict)
    
    if is_mid_M1: 
        align('ro', mm, qubit)
        tido_class.perform_full_measurement(is_wait = False,is_active_reset = False,**kwargs)
        if tido_class.wait_after_reset >= 16:
            wait(tido_class.wait_after_reset//4, 'ro')
            align('ro',qubit)
    
    
    align(qubit,mm,mm_sec,'ro')
    if qb_detuning!=0: update_frequency(qubit, tido_class.element_IF(qubit) + qb_detuning, keep_phase = True)
    if mm_detuning!=0: update_frequency(mm, tido_class.element_IF(mm) + mm_detuning, keep_phase = True)
    if mm_sec_detuning!=0: update_frequency(mm_sec, tido_class.element_IF(mm_sec) + mm_sec_detuning, keep_phase = True)
    
    if qubit_init == 'plus' : init_phase = 0.25
    elif qubit_init == 'minus': init_phase = -0.25
    else: raise ValueError(f"Unknown <qubit_init>. Expected <'minus'> or <'plus'>, got {qubit_init}")
    


    play(mm_sideband_cooling_pulse + '_ramp_up', mm)
    play(mm_sideband_cooling_pulse, mm, duration = sb_time)
    play(mm_sideband_cooling_pulse + '_ramp_down', mm)
    
    play(mm_sideband_cooling_pulse + '_ramp_up', mm_sec)
    play(mm_sideband_cooling_pulse, mm_sec, duration = sb_time)
    play(mm_sideband_cooling_pulse + '_ramp_down', mm_sec)
    
    wait(sb_ramp_up_time//4, 'qb1')
    play('pi2_pulse', 'qb1')
    frame_rotation_2pi(init_phase, 'qb1')
    play(rabi_pulse + '_ramp_up', 'qb1')
    play(rabi_pulse, 'qb1', duration = duration)
    play(rabi_pulse + '_ramp_down', qubit)
    frame_rotation_2pi(init_phase, 'qb1')
    play('pi2_pulse', 'qb1')
    align('qb1', mm, mm_sec, 'ro')
    if qb_detuning!=0: update_frequency(qubit, tido_class.element_IF(qubit), keep_phase = True)
    if mm_detuning!=0: update_frequency(mm, tido_class.element_IF(mm), keep_phase = True)
    if mm_sec_detuning!=0: update_frequency(mm_sec, tido_class.element_IF(mm_sec), keep_phase = True)
        
def prepare_fock_and_cool(n, duration, mm, tido_class, sb_duration,
                          qb_detuning_fock, mm_detuning_fock,  rabi_pulse_fock = 'rabi_pulse3', **kwargs):
    prepare_fock(n, duration, tido_class, rabi_pulse = rabi_pulse_fock, mm_sideband_cooling_pulse = 'sideband_coupling_pulse',
                 qb_detuning = qb_detuning_fock, mm_detuning = mm_detuning_fock)
    if sb_duration!=0: tido_class.sideband_cool(sb_duration = sb_duration, mm = mm, **kwargs)
        
def prepare_fock(n, duration, tido_class,
                qubit = 'qb1', mm = 'mm2', 
                qb_detuning = 0, mm_detuning = 0,
                rabi_pulse = 'rabi_pulse', qubit_init = 'minus',
                mm_sideband_cooling_pulse = 'constant_sideband_cooling_pulse', 
                 **kwargs):
    align(qubit, mm, 'ro')
    if qb_detuning!=0: update_frequency(qubit, tido_class.element_IF(qubit) + qb_detuning, keep_phase = True)
    if mm_detuning!=0: update_frequency(mm, tido_class.element_IF(mm) + mm_detuning, keep_phase = True)
    
    if qubit_init == 'plus' : init_phase = 0.25
    elif qubit_init == 'minus': init_phase = -0.25
    else: raise ValueError(f"Unknown <qubit_init>. Expected <'minus'> or <'plus'>, got {qubit_init}")

    rabi_ramp_up_time = tido_class.pulse_ramp_len(qubit, rabi_pulse)
    pi2_time = tido_class.pulse_len(qubit, 'pi2_pulse')
    sb_ramp_up_time = tido_class.pulse_ramp_len(mm, mm_sideband_cooling_pulse)
    
    sb_time = np.sum(1/np.sqrt(range(1,n+1)))*duration + n * 2*rabi_ramp_up_time + pi2_time*2
    
    play(mm_sideband_cooling_pulse + '_ramp_up', mm)
    play(mm_sideband_cooling_pulse, mm, duration = sb_time//4)
    play(mm_sideband_cooling_pulse + '_ramp_down', mm)
    
    wait(sb_ramp_up_time//4, 'qb1')
    play('pi2_pulse', 'qb1')
    frame_rotation_2pi(init_phase, 'qb1')
    for i in range(1,n+1):
        play(rabi_pulse + '_ramp_up', 'qb1')
        play(rabi_pulse, 'qb1', duration = (duration/np.sqrt(i))//4)
        play(rabi_pulse + '_ramp_down', qubit)
        frame_rotation_2pi(0.5, 'qb1')
    frame_rotation_2pi(init_phase, 'qb1')
    play('pi2_pulse', 'qb1')
    align('qb1', mm, 'ro')
    if qb_detuning!=0: update_frequency(qubit, tido_class.element_IF(qubit), keep_phase = True)
    if mm_detuning!=0: update_frequency(mm, tido_class.element_IF(mm), keep_phase = True)
    
def prepare_sideband(tido_class,
                     duration = None,
                     wait_time = 0,
                    qubit = 'qb1', mm = 'mm2', 
                    mm_detuning = 0,
                    phase_2pi = 0,
                    is_ramp = True,
                    mm_sideband_cooling_pulse = 'constant_sideband_cooling_pulse', **kwargs):
    align(qubit, mm, 'ro')
    if mm_detuning!=0: update_frequency(mm, tido_class.element_IF(mm) + mm_detuning, keep_phase = True)
    
    if is_ramp: play(mm_sideband_cooling_pulse + '_ramp_up', mm)
    if duration is not None and duration != 0: 
        play(mm_sideband_cooling_pulse, mm, duration = duration//4)
    elif duration is None:
        play(mm_sideband_cooling_pulse, mm)
    if is_ramp:play(mm_sideband_cooling_pulse + '_ramp_down', mm)
    if mm_detuning!=0: update_frequency(mm, tido_class.element_IF(mm), keep_phase = True)
    if wait_time > 0: wait(int(wait_time//4),mm)
    if phase_2pi != 0: frame_rotation_2pi(phase_2pi, mm)
    align(qubit, mm, 'ro')
    
    
    
def prepare_readout(ro_pulse, ro_element, qubit, amp_scale = 1, is_pi = False, **kwargs):
    if is_pi:
        play('pi_pulse',qubit)
        align(ro_element, qubit)
    if amp_scale!=1: play(ro_pulse*amp(amp_scale), ro_element)
    else: play(ro_pulse, ro_element)
    if is_pi:
        align(ro_element, qubit)
        play('pi_pulse',qubit)
    align(ro_element, qubit)
