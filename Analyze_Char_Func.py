# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 12:09:17 2022

@author: Shay
"""

from MLE_tf import wignerMLEIterator
import numpy as np
import qutip as qt
import tf_quantum as tfq
from scipy.interpolate import interp1d, UnivariateSpline
import matplotlib
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit

#%%

def pickle_load(filepath):
    with open(f'{filepath}.pickle', 'rb') as handle:
        loaded_dict = pickle.load(handle)
    return loaded_dict
    
def pickle_save(self, to_save_dict = None, meas_name = None, foldername = None, **kwargs):
    """Use **kwargs to add to the basic dictionary of the tido class"""
    
    time_of_meas = datetime.now().strftime("%d_%m_%Y___%H_%M_%S")
    
    if to_save_dict is None:
        to_save_dict = self.to_save_dict
    to_save_dict.update({'time_of_meas': time_of_meas})
    
    to_save_dict.update(kwargs)
    
    if foldername is None:
        if self.save_folder is not None:
            foldername = self.save_folder
            if not os.path.exists(self.save_folder):                
                os.makedirs(self.save_folder)
        else:
            foldername = ''
    else:
        if not os.path.exists(self.save_folder):                
                os.makedirs(self.save_folder)
                
    if meas_name is None:
        meas_name = 'unnamed_meas'
    to_save_dict.update({'meas_name': meas_name})
    
    filename = meas_name + '_' + time_of_meas
    
    filepath = foldername + r'\\' + filename
    
    with open(f'{filepath}.pickle', 'wb') as handle:
        pickle.dump(to_save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('\n'+f'Saved data to {filepath}'+'\n')
    
#%%
matplotlib.rcParams['font.family'] = ['Cambria']
#%%
which_data = 'I'

def process_data(data, is_mean = True, is_calc_stat_error = None,
                   is_thresh = None, is_boson_threshed = False, #this is true when Data with processed and thresholed before this 
                   **kwargs):
    """Processes the data.
    If data is a list of I and Q, automatically calls determine_data(I,Q)
    Can average the data.
    Can return the statistical error.
    Can threshold the data"""
    if len(data) == 2 and type(data) is list: data = determine_data(data[0], data[1])
    if type(data) is dict: data = determine_data(data['I'], data['Q'])
    
    # if is_boson_threshed: isb
    if is_thresh is None: is_thresh = is_thresh
    if is_thresh: data = thresh_data(data, **kwargs)
    if is_calc_stat_error is None: is_calc_stat_error = is_calc_stat_error
    if is_calc_stat_error: stat_error = sp.stats.sem(data)
    
    if is_mean: data = data.mean(0)
    
    if is_calc_stat_error: return data, stat_error
    return data

def determine_data(I, Q):
    """
    takes all the data and returns the appropriate data according to which_data
    """
    if which_data == 'I': data = I
    elif which_data == 'Q': data =  Q
    elif which_data == 'Mag': data = np.sqrt(I**2+Q**2)
    elif which_data == 'Phase': data = np.arctan2(Q,I)
    else:
        raise ValueError('No such data type! slef.which_data must be one of ["I","Q","Mag","Phase"]')
    return data

def get_geo_phase_corr(amps, phases, is_plot = True):
    func = UnivariateSpline(amps, phases)
    if is_plot:
        XX = np.linspace(min(amps), max(amps), 101)
        plt.figure()
        plt.plot(amps, phases, 'o', label = 'Data')
        plt.plot(XX, func(XX), label = 'Spline fit')
        plt.legend()
        plt.xlabel('Displacement amplitude', fontsize=25)
        plt.ylabel('Phase', fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.pause(0.1)
        
    return func

def fit_to_gaussian(x, y, guess = None):
    try:
       popt_p,pcov_p = fit_to_positive_gaussian(x,y,guess = guess)
    except:
       popt_m,pcov_m = fit_to_positive_gaussian(x,-y,guess = guess)
       return popt_m,pcov_m,-1
   
    try:
       popt_m,pcov_m = fit_to_positive_gaussian(x,-y,guess = guess)
    except:
       return popt_p,pcov_p,1
    
    if max(abs(Gaussian(x,*popt_p)-y))>max(abs(fit_to_positive_gaussian(x,*popt_m)+y)):
        return popt_m,pcov_m,-1
    return popt_p, pcov_p, 1
    
def fit_to_positive_gaussian(x, y, guess = None): #center at maximum
    if guess is None:   
        a_guess = (np.max(y)-np.min(y))
        b_guess = np.abs(x[np.argmin(np.abs(y-a_guess*np.exp(-0.5)))]-x[np.argmax(y)])   # b_guess :)
        
        guess =  [a_guess, x[np.argmax(y)], b_guess, np.min(y)]
    return curve_fit(Gaussian,x,y,p0 =guess)

def Gaussian(x, amp, center, std, C):
    return amp*np.exp(-((x-center)**2)/(2*std**2))+C

def get_scale_from_ramsey(x,y):
    plt.figure()
    plt.plot(x, y)
    fit_ramsey = fit_to_gaussian(x, y)
    plt.plot(x, Gaussian(x, *fit_ramsey[0]))
    return 1/fit_ramsey[0][2]

def get_rho_from_char_func(char_func, x, y, N):
    rho = qt.Qobj(np.zeros((N,N)))
    
    for i,yy in enumerate(y):
        for j,xx in enumerate(x):
            rho+= char_func[i,j] * qt.displace(N, -xx-1j*yy)
    return rho / rho.tr()
    
#%%
folder = r'C:\Users\Shay\OneDrive - Technion\Asaf SQCL\Exp Data\Fast cat\entagled_and_Cat_and_back_sweep'

cat_and_back_file = r'\Single_CatAndBack_mm1_20_17_07_2022___22_12_34'
cavity_ramsey_file = r'\Cavity_ramsey_mm1_3_16_07_2022___10_42_22'
data_file = r'\two_Cats_Charactristic_of_mm120_18_07_2022___05_06_14'


# cat_and_back_file = r'\cat_and_back_for_measurment5_07_07_2022___16_04_51'
# cavity_ramsey_file = r'\cavity_ramsey_vaccum_for_measurment5_07_07_2022___16_00_36'
# folder = r'F:\Asaf_experment_data\Flute3_Kanbara_SI_two_cats'
# data_file = r'\Charactristic_Sequence_amp10_as_on_ground_results_5_08_07_2022___09_40_09'
# data_file = r'\Charactristic_Sequence_amp10_as_on_excited_results_5_08_07_2022___11_27_49'


# cat_and_back_file = r'\cat_and_back_for_3_02_07_2022___10_51_35'
# cavity_ramsey_file = r'\cavity_ramsey_vaccum_for_4_04_07_2022___15_52_43'
# data_file = r'\Charactristic_Sequence_amp10_as_on_ground_results_3_02_07_2022___21_37_57'


folder = r'C:\Users\Shay\OneDrive - Technion\Asaf SQCL\Exp Data\Fast cat\charactristic function of  mm2'
# cat_and_back_file = r'\CatAndBack_mm2_Fast_ConDisp1_03_08_2022___01_33_06'
cavity_ramsey_file = r'\Cavity_ramsey_mm2_Fast_ConDisp1_03_08_2022___01_27_54'
# cat_and_back_file = r'\CatAndBack_mm2_Fast_ConDisp1_02_08_2022___16_46_56'
# cavity_ramsey_file = r'\Cavity_ramsey_mm2_Fast_ConDisp1_02_08_2022___16_41_45'
data_file = r'\Cat_Charactristic_disentagled_mm2_03_08_2022___01_22_23'
data_file = r'\Cat_Charactristic_disentagled_mm2_22_07_2022___00_59_05'

# data_file = r'\Cat_Charactristic_disentagled_with_pi2Pulse_mm2_04_08_2022___00_28_06'
# data_file = r'\two_Cat_Charactristic_disentagled_mm2_05_08_2022___02_08_28'

# cat_and_back_file = r'\Single_CatAndBack_mm1_g_Fast_ConDisp1_07_08_2022___11_10_55'
# data_file = r'\two_Cat_Charactristic_disentagled_mm1_g_05_08_2022___10_50_49'

cat_and_back_file = r'\Single_CatAndBack_mm2_Fast_ConDisp1_07_08_2022___11_22_37'
data_file = r'\two_Cat_Charactristic_disentagled_mm2_05_08_2022___02_08_28'
=======
# folder = r'C:\Users\Shay\OneDrive - Technion\Asaf SQCL\Exp Data\Fast cat\charactristic function of  mm2'
# # cat_and_back_file = r'\CatAndBack_mm2_Fast_ConDisp1_03_08_2022___01_33_06'
# cavity_ramsey_file = r'\Cavity_ramsey_mm2_Fast_ConDisp1_03_08_2022___01_27_54'
# # cat_and_back_file = r'\CatAndBack_mm2_Fast_ConDisp1_02_08_2022___16_46_56'
# # cavity_ramsey_file = r'\Cavity_ramsey_mm2_Fast_ConDisp1_02_08_2022___16_41_45'
# # data_file = r'\Cat_Charactristic_disentagled_mm2_03_08_2022___01_22_23'
# # data_file = r'\Cat_Charactristic_disentagled_mm2_03_08_2022___01_22_23'
# data_file = r'\Cat_Charactristic_disentagled_with_pi2Pulse_mm2_04_08_2022___00_28_06'

data_filepath = folder +  data_file
cavity_ramsey_filepath = folder +  cavity_ramsey_file
cat_and_back_filepath = folder +  cat_and_back_file

is_npz = False
is_cavity_ramsey = False

N = 45

if is_npz:
    data = np.load(data_filepath, allow_pickle=True)
    cat_and_back = np.load(cat_and_back_filepath, allow_pickle=True)
    cavity_ramsey = np.load(cavity_ramsey_filepath, allow_pickle=True)
    


    cavity_ramsey_data = process_data(cavity_ramsey['Re'])
    if is_cavity_ramsey:
        amp_scale = get_scale_from_ramsey(cavity_ramsey['var'], cavity_ramsey_data)
    else:
        amp_scale =6.0
    
    cat_and_back_amps = amp_scale * cat_and_back['var']
    cat_and_back_phases = np.angle(process_data(cat_and_back['Re'].tolist()) + 1j * process_data(cat_and_back['Im'].tolist()), deg = False)
    geo_phase_corr_func = get_geo_phase_corr(cat_and_back_amps, cat_and_back_phases)

    var = data.get('var')
    data_shape =  (len(var[0]), len(var[1]))
    meas_amps = np.sqrt(var[0]**2+var[1]**2) * amp_scale
    if is_cavity_ramsey:
        origin_ref = process_data(cavity_ramsey['Re'].tolist()).max()
    else:
        origin_ref = process_data(data['Re'].tolist()).max()
    Re = process_data(data['Re'].tolist()).reshape(data_shape)/origin_ref
    Im = process_data(data['Im'].tolist()).reshape(data_shape)/origin_ref
    x = var[0][0] * amp_scale
    y = var[1].transpose()[0] * amp_scale
    
    
else:
    data = pickle_load(data_filepath)
    cat_and_back = pickle_load(cat_and_back_filepath)
    cavity_ramsey = pickle_load(cavity_ramsey_filepath)
    


    cavity_ramsey_data = process_data(cavity_ramsey['Re'])
    
    if is_cavity_ramsey:
        amp_scale = get_scale_from_ramsey(cavity_ramsey['var'], cavity_ramsey_data)[0]
    else:
        amp_scale = 1.0
        
    origin_ref = cavity_ramsey_data.max()
    
    cat_and_back_amps = amp_scale * cat_and_back['var']
    cat_and_back_phases = np.angle(process_data(cat_and_back['Re']) + 1j * process_data(cat_and_back['Im']), deg = False)
    geo_phase_corr_func = get_geo_phase_corr(cat_and_back_amps, cat_and_back_phases)

    var = data.get('var')
    data_shape =  (len(var[0]), len(var[1]))
    meas_amps = np.sqrt(var[0]**2+var[1]**2) * amp_scale
    if is_cavity_ramsey:
        origin_ref = cavity_ramsey_data.max()
    else:
        origin_ref = process_data(data['Re']).max()
    Re = process_data(data['Re']).reshape(data_shape)/origin_ref
    Im = process_data(data['Im']).reshape(data_shape)/origin_ref
    argmax_Re = Re[len(Re)//2].argmax(axis=-1)
    # argmax_Re = 35
    x = (var[0][0] - var[0][0][argmax_Re]) * amp_scale
    # x = var[0][0] * amp_scale
    y = var[1].transpose()[0] * amp_scale
    


# Im =(Im-Im.transpose())/2
char_func_data = (Re+1j*Im)*np.exp(-1j*geo_phase_corr_func(meas_amps)/2)
phase_offset = np.angle(char_func_data[len(x)//2,len(y)//2])
print(phase_offset*180/np.pi)
char_func_data = char_func_data * np.exp(-1j*phase_offset)

# for i in range
# char_func_data = char_func_data[i][i]-char_func_data.transpose()
# char_func_data =np.real(char_func_data)+0j*np.imag(char_func_data)


ml = wignerMLEIterator(N = N)
ml.plot_char_func(char_func_data, x, y, title = 'Measured data')
plt.pause(0.2)

alphas = ml.coor_to_complex(x,y).reshape((len(x),len(y)))

#%%
is_plot_no_MLE = False
if is_plot_no_MLE:
    rho_no_MLE = get_rho_from_char_func(char_func_data, x, y, N)
    no_MLE_char_func = ml.characteristic_function(rho_no_MLE, alphas)
    ml.plot_char_func(no_MLE_char_func, x, y, title = 'Reconst. data')
    
    plt.pause(0.2)

    a = qt.destroy(N)
    nbar = qt.expect(a.dag()*a, rho_no_MLE).real
    # nbar = qt.expect(a.dag()*a, r).real
    abar = np.sqrt(nbar)
    # abar = 1.7543
    guess_phase = 0
    
    cat_guess = (qt.coherent(N, np.exp(1j*guess_phase)*abar) + qt.coherent(N, -np.exp(1j*guess_phase)*abar)).unit()
        
    guess_char_func = ml.characteristic_function(cat_guess, alphas)
    ml.plot_char_func(guess_char_func, x, y, 'Theory cat from reconst.')
    
    fid = qt.fidelity(qt.ket2dm(cat_guess), rho_no_MLE)**2
    
    print(fid)
#%%
res = ml.run_MLE(data = char_func_data,
                            x = x,
                            y = y,
                            guess = 'rnd',
                            tolerance = 1e-12,
                            x_tolerance = 0,
                            f_relative_tolerance = 0,
                            max_iterations = 1000,
                            parallel_iterations = 20
                            )
N_used = N

plt.pause(1)





MLE_dm = tfq.tf2qt(ml.tvec2dm(res.position.numpy()))
MLE_dm.dims = [[N_used], [N_used]]

#%%

a = qt.destroy(N_used)
nbar = qt.expect(a.dag()*a, MLE_dm).real
# nbar = qt.expect(a.dag()*a, r).real
abar = np.sqrt(nbar)
guess_phase = -0.05

# coherent_guess = qt.coherent(N_used, np.exp(1j*guess_phase)*np.sqrt(nbar))
coherent_guess = qt.coherent(N_used, qt.expect(a, MLE_dm))
coherent_guess = qt.coherent(N_used, 3)

is_guess_mixed = False
exp_alpha_MLE = qt.expect(a, MLE_dm)
uncon_disp_alpha = 2*exp_alpha_MLE*0
fid_list = []
abar_list = np.linspace(0,5,101)
for abar in abar_list:
    if is_guess_mixed:
        cat_guess = (qt.ket2dm(qt.coherent(N_used, np.exp(1j*guess_phase)*abar)) + qt.ket2dm(qt.coherent(N_used, -np.exp(1j*guess_phase)*abar))).unit()
        cat_guess = qt.displace(N, uncon_disp_alpha) * cat_guess * qt.displace(N, uncon_disp_alpha).dag()
        fid = qt.fidelity(cat_guess, MLE_dm)**2
    else:
        cat_guess =  (qt.coherent(N_used, np.exp(1j*guess_phase)*abar) + qt.coherent(N_used, -np.exp(1j*guess_phase)*abar)).unit()
        cat_guess = qt.displace(N, uncon_disp_alpha) * cat_guess
        fid = qt.fidelity(qt.ket2dm(cat_guess), MLE_dm)**2
    
    # fid = qt.fidelity(qt.ket2dm(cat_guess), r)**2
    fid_list.append(fid)
# print(fid)
plt.figure()
plt.plot(abar_list, fid_list)

abar = abar_list[np.array(fid_list).argmax()]
print(f"The maximum fidelity is {np.round(max(fid_list),4)} for alpha = {np.round(abar,4)}")


if is_guess_mixed:
    cat_guess = (qt.ket2dm(qt.coherent(N_used, np.exp(1j*guess_phase)*abar)) + qt.ket2dm(qt.coherent(N_used, -np.exp(1j*guess_phase)*abar))).unit()
else:
    cat_guess = (qt.coherent(N_used, np.exp(1j*guess_phase)*abar) + qt.coherent(N_used, -np.exp(1j*guess_phase)*abar)).unit()

# cat_guess = qt.displace(N, uncon_disp_alpha) * cat_guess * qt.displace(N, uncon_disp_alpha).dag()
guess_char_func = ml.characteristic_function(cat_guess, alphas)
ml.plot_char_func(guess_char_func, x, y, 'Theory cat from MLE')
    
qt.plot_wigner(MLE_dm, )
plt.title('MLE')
qt.plot_wigner(cat_guess)
plt.title('Theory')
# ml.plot_char_func(guess_char_func - ml.MLE_char_func_data, x, y, title = 'MLE and theory diff')
#%%
for k in [0]:
    cut_values = np.zeros(len(x),dtype=('complex128')) 
    cut_values_rotated = np.zeros(len(x),dtype=('complex128')) 
    # plt.figure()
    shift_list = [0]
    for shift in shift_list:
        for i in range(len(x)-1-np.abs(shift)):
            # cut_values_rotated[i] = (char_func_data[len(x)-i-1][i])#+char_func_data[i][len(x)-i-2])/2
            if k:
                cut_values_rotated[i] = (char_func_data[len(x)-i-1-0*shift][i-1-1*shift])#+char_func_data[i][len(x)-i-2])/2
                cut_values[i] =(char_func_data[len(x)-i-1-0*shift][len(x)-i-1-0*shift])#+ char_func_data[i][i])/2
                tit = 'angled'
            else:
                cut_values_rotated[i] = (char_func_data[len(x)-i-1-shift][len(x)//2])#+char_func_data[i][len(x)-i-2])/2
                cut_values[i] =(char_func_data[len(x)//2][len(x)-i-1-shift])#[len(x)-i-1])#+ char_func_data[i][i])/2
                tit = 'constant'
        
    # cut_values_rotated/cut_values_rotated[len(x)//2]
        # cut_values/cut_values[len(x)//2]
        plt.figure()
        plt.plot(x,cut_values.real,'.-')
        plt.legend(shift_list)
        plt.plot(x,cut_values_rotated.real,'*-')
        # #%%
        plt.title(tit + ' real')
        plt.figure()
        # cut_values_rotated/cut_values_rotated[len(x)//2]
        # cut_values/cut_values[len(x)//2]
        plt.plot(x,cut_values.imag,'.-y')
        plt.plot(x,cut_values_rotated.imag,'*-g')
        
        
#%%
N = 30
alpha = 3.56/2
state = qt.coherent(N,alpha)

ml = wignerMLEIterator(N = N)
ml._construct_needed_matrices(N)
ml.plot_char_func_line(state, phase = np.pi/2, amp = 2, npts = 100)
# ml.plot_char_func(char_func_data, x, y, title = 'Measured data')