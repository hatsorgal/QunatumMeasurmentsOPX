
import qutip as qt
import numpy as np
import scipy as sp

# OPX Qunatum Machine:
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig

class TwoQubitsTomo():
    
    def __init__(self, measurement_type):
        """Measurement_type should be either "projective" or "generalized"."""
        self.measurement_type = measurement_type
        
        self.pauli_mat_dict = {
                                'I': qt.qeye(2),
                                'Z': qt.sigmaz(),
                                'X': qt.sigmax(),
                                'Y': qt.sigmay()
            }
        self._create_composite_pauli_ops()
        
    def create_measured_ops(self, betas = None):
        if self.measurement_type == 'projective':
            self.beta_II = betas.mean()
            self.beta_IZ = (betas[0] - betas[1] + betas[2] - betas[3])/4
            self.beta_ZI = (betas[0] + betas[1] - betas[2] - betas[3])/4
            self.beta_ZZ = (betas[0] - betas[1] - betas[2] + betas[3])/4
            meas_op0 = self.beta_IZ * qt.tensor(qt.qeye(2), qt.sigmaz()) \
                        + self.beta_ZI * qt.tensor(qt.sigmaz(), qt.qeye(2)) \
                        + self.beta_ZZ * qt.tensor(qt.sigmaz(), qt.sigmaz())
            self.meas_ops_list = []
            for qubit1_pulse in self.qubit_pulse_seq_list:
                for qubit2_pulse in self.qubit_pulse_seq_list:
                    meas_op =  qt.tensor(self._qubit_rot_mat(qubit1_pulse), self._qubit_rot_mat(qubit2_pulse)) \
                            *   meas_op0 \
                            *   qt.tensor(self._qubit_rot_mat(qubit1_pulse), self._qubit_rot_mat(qubit2_pulse)).dag()
                    self.meas_ops_list.append(meas_op)
        else:
             self.meas_ops_list = []
             ops0 = [qt.tensor(qt.basis(2,1).proj(), qt.basis(2,1).proj()),
                     qt.tensor(qt.basis(2,1).proj(), qt.basis(2,0).proj()),
                     qt.tensor(qt.basis(2,0).proj(), qt.basis(2,1).proj()),
                     qt.tensor(qt.basis(2,0).proj(), qt.basis(2,0).proj())]
             
             for qubit1_pulse in self.qubit_pulse_seq_list:
                 for qubit2_pulse in self.qubit_pulse_seq_list:
                     for op0 in ops0:
                         meas_op =  qt.tensor(self._qubit_rot_mat(qubit1_pulse), self._qubit_rot_mat(qubit2_pulse)) \
                                 *   op0 \
                                 *   qt.tensor(self._qubit_rot_mat(qubit1_pulse), self._qubit_rot_mat(qubit2_pulse)).dag()
                         self.meas_ops_list.append(meas_op)
                
    def _create_composite_pulse_seq_list(self, qubit_pulse_seq_list = None):
        if qubit_pulse_seq_list is None: self.qubit_pulse_seq_list = ['+I','+X','+x','+y','-X','-x','-y']
        # if qubit_pulse_seq_list is None: self.qubit_pulse_seq_list = ['+I','+x','+y']
        self.pulse_seq_list = []
        for qubit1_pulse in self.qubit_pulse_seq_list:
            for qubit2_pulse in self.qubit_pulse_seq_list:
                self.pulse_seq_list.append((qubit1_pulse, qubit2_pulse))
        
    
    def _create_composite_pauli_ops(self):
        self.composite_pauli_ops = []
        for qubit1_op_name, qubit1_op in self.pauli_mat_dict.items():
            for qubit2_op_name, qubit2_op in self.pauli_mat_dict.items():
                if qubit1_op_name is 'I' and qubit2_op_name in 'I':
                    continue
                self.composite_pauli_ops.append(qt.tensor(qubit1_op,qubit2_op))
                
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
                        
    def calculate_eigenvalues(self, data, errors = None, threshold = 1e-6):
        
        if self.measurement_type == 'generalized':
            
            coeffs_mat = qt.expect(self.meas_ops_list, self.composite_pauli_ops)
            coeffs_mat = np.array(coeffs_mat)
            coeffs_mat = coeffs_mat * 4 / (coeffs_mat**2).sum(axis=0)
            self.eig_vals = np.tensordot(np.array(coeffs_mat).transpose(), data.flatten(), axes = 1)
            self.eig_vals[np.abs(self.eig_vals) < threshold] = 0
                    
        else:
            coeffs_mat = qt.expect(self.meas_ops_list, self.composite_pauli_ops)
            self.errors = errors
            self.data = data
            if errors is not None:
                coeffs_mat = np.multiply((np.array(coeffs_mat)/4).transpose(),1/self.errors).transpose()
                self.eig_vals = np.linalg.lstsq(coeffs_mat, (data-self.beta_II)/self.errors, rcond=None)[0]
            else:
                coeffs_mat = np.array(coeffs_mat)/4
                self.eig_vals = np.linalg.lstsq(coeffs_mat, data-self.beta_II, rcond=None)[0]
    
    #create density matrix from eigenvalues
    def calculate_density_mat(self):
        self.density_mat = qt.tensor(qt.qeye(2), qt.qeye(2))/4
        for val, op in zip(self.eig_vals, self.composite_pauli_ops):
            self.density_mat = self.density_mat + val * op / 4
            
        return self.density_mat
    
    def calculate_likelihood(self, t_list):
    
        rho = matrix_from_t_list(t_list)
        L = 0
        for eig_val, pauli_op in zip(self.eig_vals, self.composite_pauli_ops):
            L += (eig_val - qt.expect(pauli_op, rho)) ** 2
        return L                                                                            
    
        
    def calculate_MLE_density_matrix(self, maxiter = 10000, errors = False, guess = 'random'):
        if guess == 'random':
            t_list0 = 2*np.random.random(16)-1
            t_list1 = 2*np.random.random(16)-1
            t_list2 = 2*np.random.random(16)-1
        elif guess == 'mixed':
            t_list0 = np.identity(4).flatten()/2
            

        solver_method = 'BFGS'
        solver_method = 'Powell'
        if errors:
            results = sp.optimize.minimize(self.calculate_likelihood_errors, t_list0, method = solver_method, options={'gtol':1e-10, 'disp': True, 'maxiter': maxiter, 'adaptive': True})
        else:
            results = sp.optimize.minimize(self.calculate_likelihood, t_list0, method = solver_method, options={'gtol':1e-10, 'disp': True, 'maxiter': maxiter, 'adaptive': True})

        self.MLE_density_mat = matrix_from_t_list(results.x)
        return self.MLE_density_mat

    def calculate_concurrence(self, is_MLE = True, is_unselected = False, is_rotated = False):
        if is_MLE: 
            if is_rotated:
                rho = self.rotated_MLE_density_mat
            else:
                rho = self.MLE_density_mat
            
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
        
    def calculate_fidelity(self, target_state, is_MLE = True, is_rotated = False):
        if is_MLE: 
            if is_rotated:
                fdlty = qt.fidelity(self.rotated_MLE_density_mat, target_state)**2
            else:
                fdlty = qt.fidelity(self.MLE_density_mat, target_state)**2
            print(f'The fidelity after MLE is {fdlty}')
        else: 
            fdlty = qt.fidelity(self.density_mat, target_state)**2
            print(f'The fidelity before MLE is {fdlty}')
            
        return fdlty
        
    def calculate_purity(self, is_MLE = True):
        if is_MLE:
            purity = (self.MLE_density_mat**2).tr()
            print(f'MLE purity = {purity}')
        else:
            purity = (self.density_mat**2).tr()
            print(f'No MLE purity = {purity}')
        return purity
        
    def plot_density_mat(self, is_MLE = True, is_rotated = False, prettl = ''):
        if not hasattr(self, 'MLE_density_mat'): self.calculate_MLE_density_matrix()
        plot_labels = ['gg', 'ge','eg', 'ee']

        
        if is_MLE: 
            qt.matrix_histogram_complex(self.MLE_density_mat, plot_labels, plot_labels, 'Reconstructed density matrix (MLE)')
        else: 
            qt.matrix_histogram_complex(self.density_mat, plot_labels, plot_labels, prettl+'Reconstructed density matrix (no MLE)')
        if is_rotated:
            qt.matrix_histogram_complex(self.rotated_MLE_density_mat, plot_labels, plot_labels, prettl+'Reconstructed density matrix (no MLE)')
            
        
    def print_density_mat(self,is_MLE = True, round_digit = 3):
        if is_MLE:
            mat = self.MLE_density_mat
        else:
            mat = self.density_mat
            
        for row in mat.full():
            print(*row, sep="\t")
        
    def play_tomo_pulses(self, pulses, pulse_time):
        self.play_pulse_with_name(pulses[0], 'qb1', pulse_time)
        self.play_pulse_with_name(pulses[1], 'qb2', pulse_time)
        
    def declarations(self):
        self.n = declare(int)
        self.I = declare(fixed)
        self.Q = declare(fixed)
        self.x = declare(int)
    
    def play_pulse_with_name(self, name, element, pulse_time):
        if name[1] != 'I':
            if name[0] == '-':
                play(name, element)
            elif name[0] == '+':
                play(name[1], element)
        else:
            wait(pulse_time//4, element)
            
    def rotate_density_mat(self, qb1_xy_phase, qb1_phase, qb2_xy_phase, qb2_phase, qb1_z_phase, qb2_z_phase):
        """Rotate around Z and then rotate around XY."""
        rot_xy1 = np.cos(qb1_phase/2)*qt.qeye(2) - 1j*np.sin(qb1_phase/2) * (np.cos(qb1_xy_phase) * qt.sigmax() + np.sin(qb1_xy_phase) * qt.sigmay())
        rot_xy2 = np.cos(qb2_phase/2)*qt.qeye(2) - 1j*np.sin(qb2_phase/2) * (np.cos(qb2_xy_phase) * qt.sigmax() + np.sin(qb2_xy_phase) * qt.sigmay())
        rot_z1 = np.cos(qb1_z_phase/2)*qt.qeye(2) - 1j * np.sin(qb1_z_phase/2) * qt.sigmaz() 
        rot_z2 = np.cos(qb2_z_phase/2)*qt.qeye(2) - 1j * np.sin(qb2_z_phase/2) * qt.sigmaz() 
        
        tot_rot = qt.tensor(rot_xy1*rot_z1, rot_xy2*rot_z2)
        self.rotated_MLE_density_mat = tot_rot * self.MLE_density_mat * tot_rot.dag()
        
def matrix_from_t_list(t_list):
    arr = np.zeros((4,4), dtype=complex)
    t_ind = 0
    for i in range(4):
        for j in range(i+1):
            arr[i][j] += t_list[t_ind]
            t_ind += 1 
    for i in range(1,4):
        for j in range(i):
            arr[i][j] += 1j * t_list[t_ind]
            t_ind += 1 
            
    T = qt.Qobj(arr)    
    rho = T.dag() * T / ((T.dag() * T).tr())
    rho.dims = [[2,2],[2,2]]
    return  rho


#%%
# data = np.random.random(64)
# tom = TwoQubitsTomo()
# tom.calculate_density_mat(data)
# tom.calculate_MLE_density_matrix()
# print(tom.density_mat)
# qt.matrix_histogram_complex(tom.density_mat)
# print(tom.MLE_density_mat)
