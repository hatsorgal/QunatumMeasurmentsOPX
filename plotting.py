# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:36:41 2022

@author: Eliya
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Cambria']
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import fftpack

def plot_2D(data,
            x,
            y,
            vmax = None,
            vmin = None,
            ax = None,
            cax = None,
            ylabel = None,
            xlabel = None,
            zlabel = None,
            cmap = 'seismic',
            is_colorbar = True,
            is_shift_x = True,
            is_shift_y = True,
            is_ticks = True,
            is_equal_aspect_ratio = False,
            title = '',
            fontsize = 15,
            ticksize = None,
            is_tight = True):
    
    if ticksize is None: ticksize = fontsize
    if is_shift_x:
        delta_x = np.diff(x)[0]/2
        shifted_x = np.append([x[0]-delta_x], x + delta_x)
    else:
        shifted_x=x
    if is_shift_y:
        delta_y = np.diff(y)[0]/2
        shifted_y = np.append([y[0]-delta_y], y + delta_y)
    else:
        shifted_y=y
        
    if ax is None:
        fig, ax = plt.subplots(figsize = [10,8])
    # else:
    #     ax.clear()
    if vmax is not None:
        if vmin is not None:
            im = ax.pcolormesh(shifted_x, shifted_y, data, cmap = cmap, vmax=vmax, vmin=vmin, ec = 'face')
        else:
            im = ax.pcolormesh(shifted_x, shifted_y, data, cmap = cmap, vmax=vmax, ec = 'face')
    else:
        if vmin is not None:
            im = ax.pcolormesh(shifted_x, shifted_y, data, cmap = cmap, vmin=vmin, ec = 'face')
        else:
            im = ax.pcolormesh(shifted_x, shifted_y, data, cmap = cmap, ec = 'face')
        
    if is_colorbar: 
        divider = make_axes_locatable(ax)
        if cax is None:
            cax = divider.append_axes("right", size="5%", pad=0.07)
        plt.colorbar(im, cax=cax)
        plt.yticks(fontsize = fontsize)
        plt.sca(cax)
        plt.yticks(fontsize = ticksize)
        plt.xticks(fontsize = ticksize)
        plt.ylabel(zlabel, fontsize = fontsize)
    plt.sca(ax)
    if is_equal_aspect_ratio: ax.set_aspect("equal")
    if ylabel is not None: plt.ylabel(ylabel, fontsize = fontsize)
    if xlabel is not None: plt.xlabel(xlabel, fontsize = fontsize)
    if is_ticks:
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
    
    if title != '': plt.suptitle(title, fontsize = 18)
    if is_tight: plt.tight_layout()
    
    if is_colorbar: return ax, cax
    else: return ax, None
    
def plot_hist2d(data,
                x,
                num_of_bins = None,
                vmax = None,
                vmin = None,
                ax = None,
                xlabel = None,
                ylabel = None,
                zlabel = None,
                cmap = 'seismic',
                is_colorbar = True,
                figname = ''):
    
        
        if num_of_bins is None:
            num_of_bins = int(len(data[0]) / 10)
        bins = np.histogram(data, bins = num_of_bins)[1]
    
        hist2d = []
        for i in range(len(data[0,:])):
            hist2d.append(np.histogram(data[:,i], bins = bins)[0])
        
        if ax is None:
            fig,ax = plt.subplots(num = next_fig_num_by_name(figname + ' 2D histogram'))
        else:
            plt.sca(ax)
            
        plot_2D(data = np.array(hist2d).transpose(), x = x, y = bins, ax = ax,
                is_shift_y = False,
                xlabel = xlabel, ylabel = ylabel,
                cmap = cmap, is_colorbar = is_colorbar)
        plt.sca(ax)
        plt.plot(x,data.mean(0), 'k-')
        
    
    
def plot_fft(times=None, data=None, data_dic = None, title_str = None, fig_num = None, ax = None, lgd = '', is_filter_dc = True,
             title_size = 20, padding_factor = 1, **kwargs):
    """Assumes times is in [ns] and points are equally spread. Give padding factor>1 to smooth out the fft"""
    if ax is not None: plt.sca(ax)
    else:
        fig = plt.figure(fig_num) if fig_num is not None else plt.figure()
    
    
    if isinstance(data_dic,dict):
        data = 0
        for key, item in data_dic.items():
            times = np.array(item['times'])
            if key == 'I':
                data += np.array(item['data'], dtype = np.complex128)
            if key == 'Q':
                data += 1j*np.array(item['data'], dtype = np.complex128)
            Tfinal = times[-1]
            N = len(times)*padding_factor
            x = np.linspace(0, Tfinal, N)
        if is_filter_dc: yf = fftpack.fft(data - data.mean(),N)
        else: yf = fftpack.fft(data,N)
        # xf = np.linspace(0.0, 1.0/(2.0*(Tfinal/N)), N//2) * 1e3
        xf = np.linspace(-1.0/(2.0*(Tfinal/N)), 1.0/(2.0*(Tfinal/N)), N) * 1e3
        xf = xf = fftpack.fftfreq(N, np.diff(times)[0])
        xf = fftpack.fftshift(xf)*1e3
        yf = fftpack.fftshift(yf)
        # plt.plot(xf, 2.0/N * np.abs(yf[:N//2]),'o-', label = key)
        plt.plot(xf, 2.0/N * yf.real,'o-', label = 'I')
        plt.plot(xf, 2.0/N * yf.imag,'o-', label = 'Q')
        plt.plot(xf, 2.0/N * np.abs(yf),'o-', label = 'Amp')
    else:
        Tfinal = times[-1]
        N = len(times)*padding_factor
        x = np.linspace(0, Tfinal, N)
        if is_filter_dc: yf = fftpack.fft(data - data.mean(),N)
        else: yf = fftpack.fft(data,N)
        xf = np.linspace(-1.0/(2.0*(Tfinal/N)), 1.0/(2.0*(Tfinal/N)), N) * 1e3
        xf = xf = fftpack.fftfreq(N, np.diff(times)[0])
        xf = fftpack.fftshift(xf)*1e3
        yf = fftpack.fftshift(yf)
        # plt.plot(xf, 2.0/N * np.abs(yf[:N//2]),'o-', label = key)
        plt.plot(xf, 2.0/N * yf.real,'o-', label = 'I')
        plt.plot(xf, 2.0/N * yf.imag,'o-', label = 'Q')
        plt.plot(xf, 2.0/N * np.abs(yf),'o-', label = 'Amp')
    leg =plt.legend()
    leg.set_draggable(True)
    plt.xlabel('Frequency [MHz]')
    plt.ylabel(r'$|F(f)|$')
    if title_str is not None:
        plt.title(title_str, fontsize = title_size)
    else:
        plt.title(' FFT', fontsize = title_size)

    plt.tight_layout()
    
    
def next_fig_num_by_name(name: str):
    i = 1
    while plt.fignum_exists(name + ' ' + f'{i}'):
        i+=1
    return name + ' ' + f'{i}'

def last_fig_num_by_name(name: str):
    i = 1
    while plt.fignum_exists(name + ' ' + f'{i}'):
        i+=1
    return name + ' ' + f'{i-1}'
