# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:35:43 2022

@author: -
"""
#%% load
import matplotlib
import tensorflow as tf
import numpy as np
import qutip as qt
import scipy as sp
import tf_quantum as tfq
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.python.ops.distributions.util import fill_triangular, pad, fill_triangular_inverse
import tensorflow_probability as tfp
import time
from scipy import ndimage
import os


# is_gpu = True
# if is_gpu:
#     gpu = tf.config.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)
# else:
#     # Hide GPU from visible devices
#     tf.config.set_visible_devices([], 'GPU')
dtype = tf.complex64

class wignerMLE():
    
    def __init__(self, N):
        self.N = N
        self._construct_needed_matrices(N)
        self.real_ts_len = np.sum(range(N+1))
        self.imag_ts_len = np.sum(range(N))
        self.d_ops = None
        
    def run_MLE(self,
                data, x, y,
                err = None,
                guess = 'rnd',
                d_ops_method = 'qt',
                is_inverse = True,
                tolerance = 1e-9, # Tolerance for the gradient - stops if the gradient is lower than the tolerance
                x_tolerance = 0, # Tolerance for the change in the answer - stops if the distance changed in the t-vector is lower than the x-tolerance the
                f_relative_tolerance = 0, #Tolerance for the value we optimize, in this case the difference between the measured characteristic function and the MLE characteristic function
                max_iterations = 1000,
                parallel_iterations = 20,
                title = '',
                **kwargs):
        
        self.data=tf.constant(data, dtype=dtype)
        if err is not None:
            self.err=tf.constant(err, dtype=dtype)
        else:
            self.err = None
        
        if guess == 'vac':
            tvec0 = [0]*(self.real_ts_len+self.imag_ts_len)
            tvec0[0] = 1
        elif guess == 'rnd':
            tvec0 = np.random.rand(self.real_ts_len+self.imag_ts_len)
        elif guess == 'mix':
            tvec0 = np.append(np.ones(self.N),np.zeros(self.real_ts_len+self.imag_ts_len-self.N))
        else:
            raise ValueError(f"Unkown guess: {guess} (can be <vac>, <rnd>)")
        tvec=tf.constant(tvec0, dtype=tf.float64)
        
        self.alphas = self.coor_to_complex(x,y).reshape((len(x),len(y)))
        self.d_ops = tf.squeeze(self.construct_displacement_operators(self.alphas, method = d_ops_method))
        if self.err is not None:
            self.res = tfp.optimizer.bfgs_minimize(self.diff_val_and_grad_with_err, 
                                                    tvec, 
                                                    tolerance = tolerance, x_tolerance = x_tolerance, f_relative_tolerance = f_relative_tolerance,
                                                    max_iterations = max_iterations, parallel_iterations = parallel_iterations,
                                                    )
        else:
            self.res = tfp.optimizer.bfgs_minimize(self.diff_val_and_grad, 
                                                    tvec, 
                                                    tolerance = tolerance, x_tolerance = x_tolerance, f_relative_tolerance = f_relative_tolerance,
                                                    max_iterations = max_iterations, parallel_iterations = parallel_iterations,
                                                    )
        print(f'converged: {self.res.converged}')
        print(f'failed: {self.res.failed}')
        print(f'num_of_iters: {self.res.num_iterations}')
        print(f'value: {self.res.objective_value}')
        print(f'Fock space size = {self.N}')
        
        self.MLE_rho = self.tvec2dm(self.res.position.numpy())
        self.MLE_char_func_data = self._characteristic_function_tf(self.MLE_rho, self.d_ops, self.alphas.shape)
        self.plot_char_func(self.MLE_char_func_data, x, y,
                            title = title + ' MLE')
        self.MLE_rho = tfq.tf2qt(self.MLE_rho, matrix = True)
        self.MLE_rho.dims = [[self.N], [self.N]]
        return self.res
        
    @tf.function
    def diff_val_and_grad(self, tvec):
        return tfp.math.value_and_gradient(self.diff, tvec)
    
    @tf.function
    def diff_val_and_grad_with_err(self, tvec):
        return tfp.math.value_and_gradient(self.diff_with_err, tvec)
    
    @tf.function
    def diff(self, tvec):
        return tf.cast(tf.norm(self._characteristic_function_tf(self.tvec2dm(tvec), self.d_ops, self.alphas.shape)-self.data, ord=1), dtype=tf.float64)
    
    @tf.function
    def diff_with_err(self, tvec):
        return tf.cast(tf.norm((self._characteristic_function_tf(self.tvec2dm(tvec), self.d_ops, self.alphas.shape)-self.data)/self.err, ord=1), dtype=tf.float64)
    
    @tf.function
    def tvec2dm(self, tvec):
        real_ts = tf.cast(tf.slice(tvec, [0], [self.real_ts_len]), dtype)
        imag_ts = tf.cast(tf.slice(tvec, [self.real_ts_len], [self.imag_ts_len]), dtype)
        tri_mat = fill_triangular(real_ts, upper = True) + 1j*pad(pad(fill_triangular(imag_ts, upper = True), axis=0, back=True), axis=1, front=True)
        dm = tf.matmul(tf.transpose(tri_mat, conjugate = True), tri_mat)
        return tf.math.divide(dm, tf.linalg.trace(dm))
    
    # @tf.function
    # def dm2tvec(self, rho):
    #     lu, d, perm = sp.linalg.ldl(rho.full(), lower=0)
    #     tri_mat = lu.dot(np.sqrt(d))
    #     real_ts = fill_triangular_inverse(tf.math.real(tri_mat), upper = True).numpy()
    #     imag_ts = fill_triangular_inverse(tf.math.imag(tri_mat[:len(tri_mat[0])-1,1:]), upper = True).numpy()
    #     return np.append(real_ts, imag_ts)
        
    def construct_displacement_operators(self, alphas, method = 'tf'):
        if method == 'qt':
            return self.construct_displacement_operators_qt(alphas)
        elif method == 'tf':
            return self.construct_displacement_operators_tf(alphas)
        
    @tf.function
    def construct_displacement_operators_tf(self, alphas):

        num_pts = alphas.shape[0]*alphas.shape[1]
        # Reshape amplitudes for broadcast against diagonals
        sqrt2 = tf.math.sqrt(tf.constant(2, dtype=tf.complex64))
        re_a = tf.reshape(
            sqrt2 * tf.cast(tf.math.real(alphas), dtype=tf.complex64),
            [alphas.shape[0], alphas.shape[1], 1],
        )
        im_a = tf.reshape(
            sqrt2 * tf.cast(tf.math.imag(alphas), dtype=tf.complex64),
            [alphas.shape[0], alphas.shape[1], 1],
        )
        
        # Exponentiate diagonal matrices
        expm_q = tf.linalg.diag(tf.math.exp(1j * im_a * self._eig_q))
        expm_p = tf.linalg.diag(tf.math.exp(-1j * re_a * self._eig_p))
        expm_c = tf.linalg.diag(tf.math.exp(-0.5 * re_a * im_a * self._qp_comm))
        
        # Apply Baker-Campbell-Hausdorff
        return tf.reshape(tf.cast(
            self._U_q
            @ expm_q
            @ tf.linalg.adjoint(self._U_q)
            @ self._U_p
            @ expm_p
            @ tf.linalg.adjoint(self._U_p)
            @ expm_c,
            dtype=tf.complex64,
        ), (num_pts, self.N, self.N))
    
    def construct_displacement_operators_qt(self, alphas):
        num_pts = alphas.shape[0]*alphas.shape[1]
        disp_list = []
        for alpha_list in alphas:
            for alpha in alpha_list:
                disp_list.append(tf.constant(qt.displace(self.N, alpha).full(), dtype = tf.complex64))
        return tf.reshape(tf.stack(disp_list), (num_pts, self.N, self.N))
       
    def _construct_needed_matrices(self, N):
        q = tfq.position(N)
        p = tfq.momentum(N)

        # Pre-diagonalize
        (self._eig_q, self._U_q) = tf.linalg.eigh(q)
        (self._eig_p, self._U_p) = tf.linalg.eigh(p)

        self._qp_comm = tf.linalg.diag_part(q @ p - p @ q)
        
    def _characteristic_function_psi(self, psi, d_ops, alphas_shape):
        psi = tfq.qt2tf(psi)
        num_pts = d_ops.shape[0]
        psis = tf.constant(np.array([psi] * num_pts), dtype = tf.complex64)
        C = tf.linalg.adjoint(psis) @ d_ops @ psis
        return np.squeeze(C.numpy()).reshape(alphas_shape).transpose()
    
    def _characteristic_function_rho(self, rho, d_ops, alphas_shape):
        rho = tfq.qt2tf(rho)
        num_pts = d_ops.shape[0]
        rhos = tf.constant(np.array([rho] * num_pts), dtype = tf.complex64)
        C = tf.linalg.trace(d_ops @ rhos)
        return np.squeeze(C.numpy()).reshape(alphas_shape).transpose()
    
    @tf.function
    def _characteristic_function_tf(self, tensor, d_ops, alphas_shape):
        num_pts = d_ops.shape[0]
        rhos = tf.reshape(tf.tile(tensor, [num_pts,1]), (num_pts, self.N, self.N))
        C = tf.linalg.trace(d_ops @ rhos)
        return tf.transpose(tf.reshape(C,(alphas_shape)))
    
    def get_tf_alphas(self, alphas):
        alphas_flat = alphas.flatten()
        alphas_tf = tf.constant([alphas_flat])
        return alphas_tf

    def characteristic_function(self, state, alphas, d_ops_method = 'qt'):
        
        if tf.is_tensor(alphas): alphas_tf = self.get_tf_alphas(alphas)
        else: alphas_tf = alphas
        
        d_ops = tf.squeeze(self.construct_displacement_operators(alphas_tf, method = d_ops_method))
        
        if type(state) is np.ndarray: state = qt.Qobj(state)
        
        if type(state) is qt.Qobj:
            if state.isket:
                return self._characteristic_function_psi(psi = state, d_ops = d_ops, alphas_shape = alphas.shape)
            elif state.isoper:
                return self._characteristic_function_rho(rho = state, d_ops = d_ops, alphas_shape = alphas.shape)
        elif tf.is_tensor(state):
            return self._characteristic_function_tf(tensor = state, d_ops = d_ops, alphas_shape = alphas.shape)
    
    def coor_to_complex(self,x,y):
        complex_vec = []
        for xx in x:
            for yy in y:
                complex_vec.append(xx+1j*yy)
        return np.array(complex_vec)
    
    def plot_char_func(self, data, x_list, y_list, 
                       title = '',to_roll = False, title_size = 30, label_size = 25, ticks_size = 20,
                       y_label=r'Im$(\alpha)$', x_label =r'Re$(\alpha)$',
                       ax_real = None, ax_imag = None, is_color_bar = True, **kwargs):
        try:
            delta_x = np.diff(x_list)[0]/2
            delta_y = np.diff(y_list)[0]/2
            shifted_x_list = np.append([x_list[0]-delta_x], x_list + delta_x)
            shifted_y_list = np.append([y_list[0]-delta_y], y_list + delta_y)
        except:
            shifted_x_list = x_list
            shifted_y_list = y_list
        if tf.is_tensor(data):
            data = np.squeeze(data.numpy())

        if type(to_roll) ==tuple: data = np.roll(data, to_roll) 
        # if to_roll:  np.roll(char_func_data, (np.shape(x_list)//3+0*argmax_Re,0)) 
        
        if ax_real == None: 
            fig, ax_real = plt.subplots()
        plt.title(title + ' Real', fontsize = title_size)
        im = ax_real.pcolormesh(shifted_x_list, shifted_y_list, data.real, cmap = 'seismic', vmin=-1, vmax=1)
        divider = make_axes_locatable(ax_real)
        if is_color_bar:
            cax = divider.append_axes("right", size="5%", pad=0.07)
            plt.colorbar(im, cax=cax)
            plt.yticks(fontsize = ticks_size)
        ax_real.set_aspect("equal", adjustable="box")
        plt.sca(ax_real)
        plt.ylabel(y_label, fontsize = label_size)
        plt.xlabel(x_label, fontsize = label_size)
        plt.xticks(fontsize=ticks_size)
        plt.yticks(fontsize=ticks_size)
        
        plt.tight_layout()
        
        if ax_imag == None:
            fig, ax_imag = plt.subplots()
        plt.title(title + ' Imaginary', fontsize = title_size)
        im = ax_imag.pcolormesh(shifted_x_list, shifted_y_list, data.imag, cmap = 'seismic', vmin=-1, vmax=1)
        divider = make_axes_locatable(ax_imag)
        cax = divider.append_axes("right", size="5%", pad=0.07)
        plt.colorbar(im, cax=cax)
        plt.yticks(fontsize = ticks_size)
        ax_imag.set_aspect("equal", adjustable="box")
        plt.sca(ax_imag)
        plt.ylabel(y_label, fontsize = label_size)
        plt.xlabel(x_label, fontsize = label_size)
        plt.xticks(fontsize=ticks_size)
        plt.yticks(fontsize=ticks_size)
        
        plt.tight_layout()
        
        return ax_real, ax_imag
    
   
    
    def characteristic_function_line(self, Qobj, phase, amp, shift, npts):
        """phase sets the angle of the line, amp sets the length of the line,
        shift sets the displacement of the middle of the line from the origin """
        alphas = self.create_alphas_line(phase, amp, shift, npts)
        alphas_tf = self.get_tf_alphas(alphas)
        d_ops = tf.squeeze(self.construct_displacement_operators(alphas_tf))
        if Qobj.isket:
            return self._characteristic_function_psi(psi = Qobj, d_ops = d_ops, alphas_shape = alphas.shape)
        elif Qobj.isoper:
            return self._characteristic_function_rho(rho = Qobj, d_ops = d_ops, alphas_shape = alphas.shape)
    
    def create_alphas_line(self, phase, amp, shift, npts):
        x = np.real(shift) + np.linspace(-1,1,npts) * np.cos(phase) * amp
        y = np.imag(shift) +  np.linspace(-1,1,npts) * np.sin(phase) * amp
        return x+1j*y
    
    def plot_char_func_line(self, Qobj, phase, amp, npts, complex_axis = 'real', shift = 0,
                            ax = None,
                            is_rotate_axis = False,
                            plot_amp_scale_factor = 1):
        """phase sets the angle of the line, amp sets the length of the line,
        shift sets the displacement of the middle of the line from the origin.
        complex_axis can be either <'real'>, <'imag'> or <'mag'> to plot the real, the imaginary or the magnitude of the characteristic function"""
        
        char_func_data = self.characteristic_function_line(Qobj, phase, amp, shift, npts).flatten()
        if complex_axis == 'real' : data_to_plot = char_func_data.real  
        elif complex_axis == 'imag': data_to_plot = char_func_data.imag
        elif complex_axis == 'mag': data_to_plot = np.abs(char_func_data)
        
        if ax is not None:
            if not is_rotate_axis: 
                ax.plot(np.linspace(-amp, amp, npts)/plot_amp_scale_factor, data_to_plot)
                ax.set_xlabel('amp')
                ax.set_ylabel('Characteristic Wigner')
                ax.set_title(f'Phase = {phase/np.pi}'+r'$\pi$')
            else:
                ax.plot(data_to_plot, np.linspace(-amp, amp, npts)/plot_amp_scale_factor)
                ax.set_ylabel('amp')
                ax.set_xlabel('Characteristic Wigner')
                ax.set_title(f'Phase = {phase/np.pi}'+r'$\pi$')
        else:
            plt.figure()
            if not is_rotate_axis: 
                plt.plot(np.linspace(-amp, amp, npts)/plot_amp_scale_factor, data_to_plot)
                plt.xlabel('amp')
                plt.ylabel('Characteristic Wigner')
                plt.title(f'Phase = {phase/np.pi}'+r'$\pi$')
            else: 
                plt.plot(data_to_plot, np.linspace(-amp, amp, npts)/plot_amp_scale_factor)
                plt.ylabel('amp')
                plt.xlabel('Characteristic Wigner')
                plt.title(f'Phase = {phase/np.pi}'+r'$\pi$')

    def plot_char_func_axis(self, data, x_list, y_list, ratio =True,
                       title = '', x_label =r'Re$(\gamma)$', y_label = r'Im$(\gamma)$', title_pad = 12,
                       is_real = True,to_roll = False, rotate_angle = 0,colorbar_loc = None,
                       title_size = 30, label_size = 25, ticks_size = 20,
                       ax_real = None, fig_real_num = None,colorbar_axis = None,colorbar_title = '', **kwargs):
        if not data is None:
            
            try:
                delta_x = np.diff(x_list)[0]/2
                delta_y = np.diff(y_list)[0]/2
                shifted_x_list = np.append([x_list[0]-delta_x], x_list + delta_x)
                shifted_y_list = np.append([y_list[0]-delta_y], y_list + delta_y)
            except:
                shifted_x_list = x_list
                shifted_y_list = y_list
            if tf.is_tensor(data):
                data = np.squeeze(data.numpy())
           
            if type(to_roll) ==tuple: 
                data = np.roll(data, to_roll[0],0)
                data = np.roll(data, to_roll[1],1)
                
            # if to_roll:  np.roll(char_func_data, (np.shape(x_list)//3+0*argmax_Re,0)) 
            if is_real:
                data_to_plot = data.real
            else:                
                data_to_plot = data.imag
    
            if  rotate_angle: data_to_plot = ndimage.rotate(data_to_plot, rotate_angle, reshape=False)
           
        if ax_real == None: 
            fig_real, ax_real = plt.subplots( num = fig_real_num)
        # else:##
            # fig_real = plt.figure(num = fig_real_num); ax_real =plt.subplot(ax_real)
        if not title == None:
            plt.title(title, fontsize = title_size,pad =title_pad)
        if not data is None: im = ax_real.pcolormesh(shifted_x_list, shifted_y_list, data_to_plot, cmap = 'seismic', vmin=-1, vmax=1)
        if not colorbar_axis is None:
            if type(colorbar_axis) == matplotlib.axes._axes.Axes:
                cax =colorbar_axis
            else:
                divider = make_axes_locatable(ax_real)
                cax = divider.append_axes("right", size="5%", pad=0.07)
            if colorbar_loc is None:
                plt.colorbar(im, cax=cax)          
            else:
                plt.colorbar(im, cax=cax,orientation =colorbar_loc)     
            plt.yticks(fontsize = ticks_size)
            plt.xticks(fontsize = ticks_size)
            plt.title(colorbar_title)
        if not data is None: ax_real.set_aspect("equal", adjustable="box")
        plt.sca(ax_real)
        
        # if data is None: return ax_real

        plt.ylabel(y_label, fontsize = label_size)
        plt.xlabel(x_label, fontsize = label_size)
        plt.xticks(fontsize=ticks_size)
        plt.yticks(fontsize=ticks_size)
        
        # plt.tight_layout()
        
        return  ax_real


    def plot_wigner_line(self, Qobj, phase, amp, npts, 
                            ax = None):
        
        phase_char_func = phase + np.pi/2
        char_func = self.characteristic_function_line(Qobj = Qobj, phase = phase_char_func, amp = amp/np.pi, shift = 0, npts = npts).flatten()
        char_func_x = np.linspace(-amp/np.pi, amp/np.pi, npts) * np.exp(1j*phase_char_func)
        wigner_x = np.linspace(-amp, amp, npts) * np.exp(1j*phase)
        wigner = np.zeros(npts, dtype = np.complex128)
        for i,x in enumerate(wigner_x):
            wigner[i] = np.sum(char_func * np.exp(x*char_func_x.conj()-x.conj()*char_func_x))/np.pi**2
        plt.figure()
        plt.plot(np.real(wigner_x* np.exp(-1j*phase)), wigner)
        
    def wigner(self, Qobj, x, y,
                            ax = None, method = 'tf'):
        
        alphas = self.coor_to_complex(y/np.pi,x/np.pi).reshape((len(y),len(x))).astype(np.complex64)
        char_func = self.characteristic_function(Qobj, alphas, d_ops_method = 'tf')
        
        if method == 'tf':
            betas = self.coor_to_complex(x,y).reshape((len(x),len(y))).astype(np.complex64)
            wigner = self._wigner_tf(char_func, alphas, betas)
        else:
            wigner = np.zeros([len(x),len(y)], dtype = np.complex128)
            for i,xx in enumerate(x):
                for j,yy in enumerate(y):
                    beta = xx+1j*yy
                    wigner[i,j] = np.sum(char_func * np.exp(beta*alphas.conj()-beta.conj()*alphas))/np.pi**2
        return wigner
    
    @tf.function
    def _wigner_tf(self, char_func, alphas, betas):
        npts = alphas.shape[0] * alphas.shape[1]
        char_func_tensor = tf.reshape(tf.tile(char_func, [npts,1]), (npts, alphas.shape[0], alphas.shape[1]))
        alphas_tensor = alphas
        betas_tensor = betas
        W = tf.linalg.trace(char_func_tensor * tf.math.exp(alphas_tensor * tf.math.conj(betas_tensor) - tf.math.conj(alphas_tensor) * betas_tensor))
        return tf.transpose(tf.reshape(W,(alphas.shape)))
        
    def plot_wigner(self, Qobj, x,y, ax=None, is_color_bar = True, ticks_size = 15, method = 'tf'):
        wigner = self.wigner(Qobj,x,y)
        try:
            delta_x = np.diff(x)[0]/2
            delta_y = np.diff(y)[0]/2
            shifted_x = np.append([x[0]-delta_x], x + delta_x)
            shifted_y = np.append([y[0]-delta_y], y + delta_y)
        except:
            shifted_x = x
            shifted_y = y
            
        data = np.real(wigner).astype(np.float)
        # if tf.is_tensor(data):
        #     data = np.squeeze(data.numpy())

        # if type(to_roll) ==tuple: data = np.roll(data, to_roll) 
        # if to_roll:  np.roll(char_func_data, (np.shape(x_list)//3+0*argmax_Re,0)) 
        
        if ax == None: 
            fig, ax = plt.subplots()
        im = ax.pcolormesh(shifted_x, shifted_y, data, cmap = 'seismic', vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        if is_color_bar:
            cax = divider.append_axes("right", size="5%", pad=0.07)
            plt.colorbar(im, cax=cax)
            plt.yticks(fontsize = ticks_size)
        ax.set_aspect("equal", adjustable="box")
        # plt.sca(ax)
        # plt.ylabel(y_label, fontsize = label_size)
        # plt.xlabel(x_label, fontsize = label_size)
        plt.xticks(fontsize=ticks_size)
        plt.yticks(fontsize=ticks_size)
        
class wignerMLEIterator(wignerMLE):
    
    def __init__(self, N):
        super().__init__(N)
    
    def run_sweep_focks_MLE(self, 
                            data, x, y,
                            start = 2, stop = None,
                            purity_thresh = None, likelihood_thresh = None,
                            guess = 'rnd',
                            d_ops_method = 'tf',
                            is_plot_all = True,
                            **kwargs):
        """Sweeps over the size of the Fock space. 
        Will stop if the size reaches stop or if both the change in purity and the change in likelihood is below their respective thresholds.
        See run_MLE for the possible **kwargs to be sent to the estimator"""
        
        self.purity_list = []
        self.likelihood_list = []
        self.res_list = []
        self.dm_list = []
        N = start
        
        break_condition = False
        
        while True:
            super().__init__(N)
            res = self.run_MLE(data, x, y, guess = guess, d_ops_method = d_ops_method, **kwargs)
            self.res_list.append(res)
            dm = self.tvec2dm(res.position.numpy())
            self.dm_list.append(dm)
            purity = tf.linalg.trace(dm**2)
            likelihood = res.objective_value
            self.likelihood_list.append(likelihood)
            if is_plot_all:
                self.plot_char_func(self._characteristic_function_tf(self.tvec2dm(res.position.numpy()), self.d_ops, self.alphas.shape), x, y,
                                    title = f'MLE for N = {N}')
                plt.pause(0.1)
            
            N_cond = False
            purity_cond = False
            likelihood_cond = False
            
            if stop is not None:
                if N>stop:
                    print(f'\n Maximum Fock space size reached {N}>{stop}\n')
                    N_cond = True
            else:
                N_cond = True
                
            if purity_thresh is not None:
                if len(self.purity_list) > 1:
                    delta_purity = np.abs(purity - self.purity_list[-2])
                    if delta_purity < purity_thresh:
                        print(f'\n Purity threshold reached {delta_purity} < {purity_thresh}\n')
                        purity_cond = True
            else:
                purity_cond = False
                
            if likelihood_thresh is not None:
                if len(self.likelihood_list) > 1:
                    delta_likelihood = np.abs(likelihood - self.likelihood_list[-2])
                    if  delta_likelihood < likelihood_thresh:
                        print(f'\n Likelihood threshold reached {delta_likelihood} < {likelihood_thresh}\n')
                        likelihood_cond = True
            else:
                likelihood_cond = False
                
            if N_cond or (likelihood_cond and purity_cond): 
                break
            
            N+=1
            
        return res, N
        
    #%%
if __name__ == "__main__":

    N=30
    ml = wignerMLEIterator(N)
    therm = qt.thermal_dm(N,3)
    point_size = 0.1
    x_lims = [-2,2]
    y_lims = [-2,2]
    x = np.linspace(x_lims[0],x_lims[1], int((x_lims[1]-x_lims[0])/point_size+1))
    y = np.linspace(y_lims[0],y_lims[1], int((y_lims[1]-y_lims[0])/point_size+1))
    
    alphas = ml.coor_to_complex(x,y).reshape((len(x),len(y)))
    char_func_thermal = ml.characteristic_function(therm, alphas)
    ml.plot_char_func(char_func_thermal, x, y)
    char_func_vacuum = ml.characteristic_function(qt.basis(N,0), alphas)
    ml.plot_char_func(char_func_vacuum, x, y)
    
    
    # wMLEIter.run_sweep_focks_MLE(char_func, x, y,
    #                               start = 25, stop = 35, purity_thresh = 0.1, likelihood_thresh = 0.1,
    #                               d_ops_method = 'tf')
    
    # cat = qt.ket2dm((qt.coherent(N , 3) + qt.coherent(N, -3)).unit())
    
    # tvec = [1,2,3,4,5,6,7,8,9]
    
    # dm = tfq.tf2qt(ml.tvec2dm(tvec))
    
    # print(dm)
    # tvec_re = ml.dm2tvec(dm)
    # print(tvec_re)
    
    # print(tfq.tf2qt(ml.tvec2dm(tvec_re)))
    # dm = ml.tvec2dm(tvec)
    # print(ml.tvec2dm(tvec))
    # N = 50
    # wMLE = wignerMLE(N)
    # d_ops_method = 'tf'
    
    # point_size = 0.1
    # x_lims = [-8,8]
    # y_lims = [-4,4]
    # x = np.linspace(x_lims[0],x_lims[1], int((x_lims[1]-x_lims[0])/point_size+1))
    # y = np.linspace(y_lims[0],y_lims[1], int((y_lims[1]-y_lims[0])/point_size+1))
    
    # alphas = wMLE.coor_to_complex(x,y).reshape((len(x),len(y)))
    
    
    # char_func = wMLE.characteristic_function(qt.displace(N,2)*qt.basis(N,0), alphas, d_ops_method = d_ops_method)


    # alpha = 2
    # phase = np.pi/2
    
    # cat_state = (qt.coherent(N, alpha) + np.exp(1j*phase) * qt.coherent(N, -alpha) ).unit()
    
    # cat_state = (qt.displace(N, 0.6j) * qt.coherent(N, alpha) + qt.coherent(N, -alpha) ).unit()
    
    
    
    
    
    # fig, ax_real = plt.subplots(2,2)
    # plt.suptitle(r'Real $|\alpha\rangle -i |-\alpha\rangle$')
    
    # ax_real_full = ax_real[0][0]
    # ax_real[1,1].set_visible(False)
    
    # fig, ax_imag = plt.subplots(2,2)
    # plt.suptitle(r'Imaginary $|\alpha\rangle -i |-\alpha\rangle$')
    # ax_imag_full = ax_imag[0][0]
    # ax_imag[1,1].set_visible(False)
    
    # wMLE.plot_char_func(char_func, x,y, title = 'target', ax_real = ax_real_full, ax_imag = ax_imag_full)
    
    # wMLE.plot_char_func_line(cat_state, phase = 0, amp = max(x_lims), npts = 101, ax = ax_real[1][0])
    # wMLE.plot_char_func_line(cat_state, phase = np.pi/2, amp = max(y_lims), npts = 101, ax = ax_real[0][1], is_rotate_axis = True)
    
    # wMLE.plot_char_func_line(cat_state, phase = 0, amp = max(x_lims), npts = 101, complex_axis = 'imag', ax = ax_imag[1][0])
    # wMLE.plot_char_func_line(cat_state, phase = np.pi/2, amp = max(y_lims), npts = 101, complex_axis = 'imag', ax = ax_imag[0][1], is_rotate_axis = True)
    
    # plt.pause(0.2)
    
    # #%%
    # N2 = 40
    # wMLE2 = wignerMLE(N2)
    # start_time = time.time()
    # res = wMLE2.run_MLE(char_func, x, y, guess = 'rnd', d_ops_method = d_ops_method)
    # char_func_MLE = wMLE2.characteristic_function(wMLE2.tvec2dm(res.position.numpy()), alphas, d_ops_method = d_ops_method)
    # wMLE2.plot_char_func(char_func_MLE, x, y, title = 'MLE')
    # print(f'MLE runtime is {time.time()-start_time} s')
    
    # #%%    
    # N=10
    # ml = wignerMLEIterator(N)
    # # wMLEIter.run_sweep_focks_MLE(char_func, x, y,
    # #                               start = 25, stop = 35, purity_thresh = 0.1, likelihood_thresh = 0.1,
    # #                               d_ops_method = 'tf')
    
    # cat = qt.ket2dm((qt.coherent(N , 3) + qt.coherent(N, -3)).unit())
    
    # ml.
    # N = 50
    # wMLE = wignerMLE(N)
    # mixed_cat = (qt.ket2dm(qt.coherent(N,1))+qt.ket2dm(qt.coherent(N,-1)))/2
    
    # cat = state9.ptrace((1))
    # cat = (tensor(e.proj(), qeye(N)) * state9).unit().ptrace((1))
    
    # IQ_phase = 0
    # canonical_phase = np.pi
    # meas_phase = 0
    # displace_middle = 0.0
    # shift = 0 * np.exp(1j*0)
    # alpha = 3 * np.exp(1j*IQ_phase)
    # cat = qt.displace(N, displace_middle) * ( qt.coherent(N,alpha) + np.exp(1j*canonical_phase) * qt.coherent(N,-alpha)).unit()
    # char_func = wMLE.characteristic_function(cat, alphas, d_ops_method = d_ops_method)
    # wMLE.plot_char_func(char_func, x, y)
    # wMLE.plot_char_func_line(cat, phase = meas_phase, amp = max(x_lims), shift = shift, npts = 101,
    #                           complex_axis = 'real',
    #                           plot_amp_scale_factor = 1)
    # qt.plot_wigner(cat)
    
    
    #%%
