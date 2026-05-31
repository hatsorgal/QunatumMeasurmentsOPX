
import qutip as qt
import numpy as np
import scipy as sp

# OPX Qunatum Machine:
try: from qm.QuantumMachinesManager import QuantumMachinesManager
except: from qm.quantum_machines_manager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig

class OneQubitTomo():
    
    def __init__(self, qubit):
        self.qubit = qubit
        self._create_puali_mat_dict()
        self.pulse_seq_list = ['+I','+X','+x','+y','-X','-x','-y']
        
    def _create_puali_mat_dict(self):
        self.pauli_mat_dict = {
                                'I': qt.qeye(2),
                                'X': qt.sigmax(),
                                'Y': qt.sigmay(),
                                'Z': qt.sigmaz()}
        self.pauli_ops = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
        
    def create_measured_ops(self):
        self.meas_ops_list = []
        for pulse in self.pulse_seq_list:
            meas_op =   self._qubit_rot_mat(pulse) \
                        * qt.sigmaz() \
                        * self._qubit_rot_mat(pulse).dag()
            self.meas_ops_list.append(meas_op)
         
            
    #  Qubit Rotations operaotrs
    def _qubit_rot_mat(self, pulse):
        if pulse == '+I' or pulse == '-I':       return  qt.qeye(2)
        elif pulse == '+x':                      return  np.cos(np.pi/4)*qt.qeye(2)-1j*np.sin(np.pi/4)*qt.sigmax()
        elif pulse == '-x':                      return  np.cos(-np.pi/4)*qt.qeye(2)-1j*np.sin(-np.pi/4)*qt.sigmax()
        elif pulse == '+X':                      return  np.cos(np.pi/2)*qt.qeye(2)-1j*np.sin(np.pi/2)*qt.sigmax()
        elif pulse == '-X':                      return  np.cos(-np.pi/2)*qt.qeye(2)-1j*np.sin(-np.pi/2)*qt.sigmax()
        elif pulse == '+y':                      return  np.cos(np.pi/4)*qt.qeye(2)-1j*np.sin(np.pi/4)*qt.sigmay()
        elif pulse == '-y':                      return  np.cos(-np.pi/4)*qt.qeye(2)-1j*np.sin(-np.pi/4)*qt.sigmay()
        elif pulse == '+Y':                      return  np.cos(np.pi/2)*qt.qeye(2)-1j*np.sin(np.pi/2)*qt.sigmay()
        elif pulse == '-Y':                      return  np.cos(-np.pi/2)*qt.qeye(2)-1j*np.sin(-np.pi/2)*qt.sigmay()
        else: raise ValueError(f"pulse name ({pulse}) is wrong. Use the format (sign)(pulse name) where sign is +/- and pulse name is I/x/X/y/Y ")
                        
        
    def calculate_eigenvalues(self, betas, data, errors = None):
        self.beta_I = betas.mean()
        self.beta_Z = (np.diff(betas)/2)[0]
        # betas[0] = beta_g, beta[1] = beta_e
        coeffs_mat = qt.expect(self.meas_ops_list, self.pauli_ops)
        coeffs_mat = np.array(coeffs_mat)/2
        self.errors = errors
        self.data = data
        self.eig_vals = np.linalg.lstsq(coeffs_mat, (data-self.beta_I)/self.beta_Z, rcond=None)[0]
    
    #create density matrix from data
    def calculate_density_mat(self):
        self.density_mat = qt.qeye(2)/2
        for val, op in zip(self.eig_vals, self.pauli_ops):
            self.density_mat = self.density_mat + val * op / 2
        return self.density_mat
    
    def calculate_likelihood(self, t_list):
    
        rho = matrix_from_t_list(t_list)
        L = 0
        for eig_val, pauli_op in zip(self.eig_vals, self.pauli_ops):
            L += (eig_val - qt.expect(pauli_op, rho)) ** 2
        return L
        
    def calculate_likelihood_errors(self, t_list):
    
        rho = matrix_from_t_list(t_list)
        L = 0
        for i in range(len(t_list)):
            L += ((self.data[i] - qt.expect(self.meas_ops_list[i], rho))/self.errors[i]) ** 2
        return L                             
    
    def calculate_MLE_density_matrix(self, guess = 'random', errors = False, is_bounds = True):
        
        if type(guess) != list:
            if guess == 'random':
                t_list0 = 2*np.random.random(4)-1
            elif guess == 'mixed':
                t_list0 = np.identity(2).flatten()/2
            else:
                raise ValueError("Invalid guess")
        else:
            t_list0 = guess

        solver_method = 'Nelder-Mead'
        solver_method = 'Powell'
        
        if is_bounds:
            bounds = sp.optimize.Bounds(-10,10)
        else:
            bounds = sp.optimize.Bounds()
        if solver_method == 'Powell': bounds = None
            
        if errors:
            results = sp.optimize.minimize(self.calculate_likelihood_errors, t_list0, method = solver_method,  bounds = bounds, options={'gtol': 1e-11, 'disp': True,})
        else:
            results = sp.optimize.minimize(self.calculate_likelihood, t_list0, method = solver_method,  bounds = bounds, options={'gtol': 1e-15, 'disp': True})
        self.MLE_density_mat = matrix_from_t_list(results.x)
        return self.MLE_density_mat

    def calculate_concurrence(self, is_MLE = True, is_unselected = False):
        if is_MLE: rho = self.MLE_density_mat
        else: rho = self.density_mat
        
        rho_tilde = qt.tensor(qt.sigmay(),qt.sigmay()) * rho.conj() * qt.tensor(qt.sigmay(), qt.sigmay())
        lambdas = np.sqrt((rho*rho_tilde).eigenenergies())
        lambda_max = lambdas.max()
        self.concurrence = np.real(np.round_(max(0, 2 * lambda_max - sum(lambdas) ), 4))
        img_part = np.imag(np.round_(max(0, 2 * lambda_max - sum(lambdas) ), 4))
        if img_part != 0: print(f"note that the imaginary part of the concurrence {img_part} is not zero so something went wrong")
        if is_MLE and is_unselected: print(f"The unselected concurrence after MLE is {self.concurrence}")
        elif is_MLE:print(f"The concurrence after MLE is {self.concurrence}")
        else: print(f"The concurrence before MLE is {self.concurrence}")
        return self.concurrence
        
    def calculate_fidelity(self, target_state, is_MLE = True):
        if is_MLE: 
            fdlty = qt.fidelity(self.MLE_density_mat, target_state)**2
            print(f'The fidelity after MLE is {fdlty}')
        else: 
            fdlty = qt.fidelity(self.density_mat, target_state)**2
            print(f'The fidelity before MLE is {fdlty}')
            
        return fdlty
        
    def plot_density_mat(self, is_MLE = True, is_rotated = False, prettl = ''):
        if not hasattr(self, 'MLE_density_mat'): self.calculate_MLE_density_matrix()
        plot_labels = ['g', 'e']
        
        if is_MLE: 
            qt.matrix_histogram_complex(self.MLE_density_mat, plot_labels, plot_labels, 'Reconstructed density matrix (MLE)')
        else: 
            qt.matrix_histogram_complex(self.density_mat, plot_labels, plot_labels, prettl+'Reconstructed density matrix (no MLE)')
        if is_rotated:
            qt.matrix_histogram_complex(self.rotated_MLE_density_mat, plot_labels, plot_labels, prettl+'Reconstructed density matrix (no MLE)')
            
        
    def print_density_mat(self, is_MLE = True, round_digit = 3):
        if is_MLE:
            mat = self.MLE_density_mat
        else:
            mat = self.density_mat
            
        for row in mat.full():
            print(*row, sep="\t")
        
    def play_tomo_pulse(self, pulses, pulse_time):
        self.play_pulse_with_name(pulses, self.qubit, pulse_time)
        
    def declarations(self):
        self.n = declare(int)
        self.I = declare(fixed)
        self.Q = declare(fixed)
        self.I_betas = declare(fixed)
        self.Q_betas = declare(fixed)
        self.x = declare(int)
    
    def play_pulse_with_name(self, name, element, pulse_time):
        if name[1] != 'I':
            if name[0] == '-':
                play(name, element)
            elif name[0] == '+':
                play(name[1], element)
        else:
            wait(pulse_time//4, element)
            
    def rotate_density_mat(self, xy_phase, phase):
        rot = np.cos(phase/2)*qt.qeye(2) - 1j*np.sin(phase/2) * (np.cos(xy_phase) * qt.sigmax() + np.sin(xy_phase) * qt.sigmay())
        self.rotated_MLE_density_mat = rot * self.MLE_density_mat * rot.dag()
        
        
    def measure_beta_coeffs(self, ro_element, qubit, meas_func, **kwargs):
        with for_(self.x, 0, self.x<2, self.x+1):
            with switch_(self.x):
                with case_(0):
                    align(qubit, ro_element)
                with case_(1):
                    play('pi_pulse', qubit)
                    align(qubit, ro_element)
            meas_func(self.I_betas, self.Q_betas,'I_betas', 'Q_betas', ro_element = ro_element, **kwargs)
                    
        
def matrix_from_t_list(t_list):
    arr = np.zeros((2,2), dtype=complex)
    t_ind = 0
    for i in range(2):
        for j in range(i+1):
            arr[i][j] += t_list[t_ind]
            t_ind += 1 
    for i in range(1,2):
        for j in range(i):
            arr[i][j] += 1j * t_list[t_ind]
            t_ind += 1 
            
    T = qt.Qobj(arr)    
    rho = T.dag() * T / ((T.dag() * T).tr())
    rho.dims = [[2],[2]]
    return  rho


#%%
# data = np.random.random(64)
# tom = TwoQubitsTomo()
# tom.calculate_density_mat(data)
# tom.calculate_MLE_density_matrix()
# print(tom.density_mat)
# qt.matrix_histogram_complex(tom.density_mat)
# print(tom.MLE_density_mat)
