# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:31:06 2025

@author: Shay
"""

from scipy.optimize import minimize
import numpy as np
from plotting import plot_2D


def parabuloid(x, y, a, b, c):
    return a + b * (x**2+y**2) + c * x * y
def parabuloid(x, y, a, b, c, x0, y0):
    return a + b * ((x-x0)**2+(y-y0)**2) + c * x * y
def parabuloid(x, y, a, b, x0, y0):
    return a + b * ((x-x0)**2+(y-y0)**2)

def curve_fit_2D(func, x, y, data, p0):
    xg, yg = np.meshgrid(x, y)
    def cost_func(p0):
        return np.sum((func(xg, yg,*p0) - data)**2)
    res = minimize(cost_func, x0=p0)
    resVariance = res.fun/(len(data)-len(p0))
    cov = res.hess_inv * resVariance
    return res.x, cov

#%%
if __name__ == '__main__':
    x = np.linspace(-2,2,101)
    y = np.linspace(-2,2,101)
    xg, yg = np.meshgrid(x, y)
    
    data = parabuloid(xg,yg,1,-1,0)+np.random.rand(xg.shape[0]*xg.shape[1]).reshape(xg.shape)*0.5
    plot_2D(data, x, y, is_shift_x = True, is_shift_y = True)
    
    p0 = [1,-1,0]
    fit,cov = curve_fit_2D(parabuloid, x, y, data, p0)
    
    plot_2D(parabuloid(xg, yg, *fit), x, y, is_shift_x = True, is_shift_y = True)
#%%
if __name__ == '__main__':
    Re, Im, x, y, Re_err, Im_err = tido.plot_char_func(a=219)
    plt.close('all')
    xg, yg = np.meshgrid(x, y)
    data = Re/Re.max()
    data = Re/Re.max()
    plot_2D(Im/Re.max(), x, y, is_shift_x = True, is_shift_y = True, cmap = 'Reds')
    ax, cax = plot_2D(data, x, y, is_shift_x = True, is_shift_y = True, cmap = 'Reds')
    ax.plot(0,0,'bx')
    
    p0 = [1,-1,0]
    p0 = [1,-1,0,0,0]
    p0 = [1,-1,0,0]
    fit,cov = curve_fit_2D(parabuloid, x, y, data, p0)
    ax, cax = plot_2D(parabuloid(xg, yg, *fit), x, y, is_shift_x = True, is_shift_y = True, cmap = 'Reds')
    ax.plot(0,0,'bx')
    nbar = -1/2-fit[1]*2
    print('nbar = {}+-{}'.format(*round_value_by_error(nbar, np.sqrt(cov[1,1]))))
    plot_2D(parabuloid(xg, yg, *fit)-data, x, y, is_shift_x = True, is_shift_y = True, cmap = 'Reds')
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(xg.flatten(), yg.flatten(), data.flatten(), '.', color = 'tab:orange')
    # ax.plot_surface(xg, yg, parabuloid(xg, yg, *fit), alpha = 0.5)
    ax.plot_wireframe(xg, yg, parabuloid(xg, yg, *fit))
    plt.show()
    
#%%

