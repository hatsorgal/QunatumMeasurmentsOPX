
"""
Created on Tue Apr 28 12:02:16 2020

@author: Eliya
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
from numpy import pi, cos, sin, exp, real, imag
from scipy.optimize import curve_fit, minimize
import msvcrt
from os.path import isfile
from time import time, sleep
import scipy.stats


#%% FIT
class sFit():

    def __init__(self, fittype, trace, time, itrace = None, error = None, is_absolute_sigma = True, bounds = (-np.inf, np.inf), t0 = None, **kwargs):
        
        if fittype == 'ExpCosiSin' and itrace is None: raise ValueError(f"Expected itrace as well (imaginary trace) for fit type <{fittype}>")
            
        self.fittype = fittype
        self.trace = trace
        self.itrace = itrace
        self.time = time
        self.t0 = t0 if t0 != None else time[0]
        dt = np.diff(time)[0]
        self.dt = np.diff(time)[0]
        self.error = error
        if error is None:
            is_absolute_sigma = False
        self.absolute_sigma = is_absolute_sigma
        
        self.sGuess = self.smart_guess(trace, self.dt)
            
        self.guess  = self.fit_guess(self.fittype, *self.sGuess)
        self.func   = self.fit_function(self.fittype)
        
        if fittype in ['c']:  
            pass
            # trans_trace = trace +abs(min(trace))+1e-5 if min(trace)<=0 else trace
            # self.trans_trace = np.log(trans_trace)-np.log(trans_trace[-1])
            
            # if fittype == 'Exp(Exp)':
            #     self.internal_sfit = sFit('Exp', self.trans_trace, time)
                
            #     if self.internal_sfit.is_succeed:
            #         self.guess[1]=-self.internal_sfit.fit_results[0]        
            #         self.guess[2]=self.internal_sfit.fit_results[1] 
    
            # elif fittype =='Exp(ExpCos)':
            #     self.internal_sfit = sFit('ExpCos', trans_trace, time)
            #     raise ValueError("finish Exp(ExpCos)") 
            
        else:
            self.internal_sfit = None
  
        if fittype == 'ExpCosiSin' or fittype == "GaussianCosiSin":
            self.trace = np.append(trace, itrace)
            self.fit_results, self.cov_results = self._fit()
            
        else:
            self.fit_results, self.cov_results = self._fit(bounds = bounds)
         
    
    def Line(self, x, a, b):
        return (a*x + b).astype(np.float64)
    def Poly2(self, x, a, b, c):
        return (a*x**2 + b*x + c).astype(np.float64)
    def x2(self, x, a, b):
        return (a*x**2+b).astype(np.float64)
    def x4(self, x, a, b,c):
        return (a*x**2+b*x**4+c).astype(np.float64)
    def x6(self, x, a, b, c, d):
        return (a*x**2+b*x**4+c*x**6+d).astype(np.float64)
    def x61(self, x, a, b, c):
        return (a*x**2+b*x**4+c*x**6+1).astype(np.float64)
    def Exp(self, t, A, gamma, offset): # 3 args
        return (A*exp(-t*gamma) + offset).astype(np.float64)
    def Cos(self, t, A, f, delta, offset): # 4 args
        return (A*cos((t-self.t0)*2*pi*f+delta)+offset).astype(np.float64)
    def ExpCos(self,  t, A, f, gamma, delta, offset): # 5 args
        return (A*exp(-t*gamma)*cos((t-self.t0)*2*pi*f+delta) + offset).astype(np.float64)
    def ExpCosiSin(self,  t, A, f, gamma, delta, offset): # 5 args
        complex_res = A*exp(-t*gamma)*exp(1j*((t-self.t0)*2*pi*f+delta)) + offset
        return np.append(complex_res.real, complex_res.imag)
    def ExpExp(self,  t, A, gamma, sigma_i, sigma_f, offset):
        return (A*exp(sigma_i*exp(-gamma*t)+sigma_f ) + offset).astype(np.float64)
    def ExpExpCos(self):
        pass

    
    
    
    def fit_guess(self,  fittype, Amp, Freq=0*1e-9, Gamma=0*1e-9, Delta=0, Offset=0, alphaSize=1):
        """Basically, this converts a bunch of parameters into a fit type. Pass as numbers, not labrad units"""
        if fittype == 'Exp':
            return [Amp, Gamma, Offset]
        elif fittype == 'Cos':
            return [np.abs(Amp), Freq, Delta, Offset]
        elif fittype == 'ExpCos':
            return [np.abs(Amp), Freq, Gamma, Delta, Offset]
        elif fittype == 'ExpCosiSin':
            return [np.abs(Amp), Freq, Gamma, Delta, Offset]
        elif fittype == 'Exp(Exp)':
            return [Amp, Gamma, 0,0, Offset]
        elif fittype == 'Exp(ExpCos)':
            return [Amp, abs(alphaSize),Freq, Gamma, Delta, Offset]
        elif fittype == 'Poly2':
            return [0, 0, Offset - Amp]
        elif fittype == 'x2':
            return [np.diff(np.diff(self.trace))[(len(self.trace)-2)//2], self.trace[np.argmin(np.abs(self.time))]]
        elif fittype == 'x4':
            return [np.diff(np.diff(self.trace))[(len(self.trace)-2)//2], 0, 0]
        elif fittype == 'x6':
            return [np.diff(np.diff(self.trace))[(len(self.trace)-2)//2], 0, 0, 0]
        
    def smart_guess(self,  trace, dt):
        [traceFFT, traceFreqs] = self.fft_trace( trace, dt)
        if self.fittype != 'Exp': 
            Freq = self.peak_freq(traceFFT, traceFreqs)
            Delta = np.angle(traceFFT[traceFreqs == Freq])[0]
        else: 
            Freq = 0
            Delta=0
        if self.fittype == 'Exp': Amp = trace[-1]-trace[0]
        elif self.fittype == 'Exp(Exp)': Amp = trace[-1]-trace[0]
        else: Amp = (max(trace)-min(trace)) / (1 - np.exp(-3))
        if np.argmax(trace) > np.argmin(trace): Amp = -Amp
        Offset = (max(trace)+min(trace))/2
        if trace.mean() < 0 and self.fittype != 'Exp(Exp)': Offset = -Offset 
        if self.fittype == 'Exp':
            Gamma = -np.log((trace[len(self.time)//4]-trace[-1])/(trace[0]-trace[-1]))/(self.time[len(self.time)//4]-self.time[0])
        else:
            Gamma = 0
        return Amp, Freq, Gamma, Delta, Offset
    
    def fit_function(self, fittype):
        """Select the fit function"""
        if fittype == 'Line':
            return self.Line
        elif fittype == 'Exp':
            return self.Exp
        elif fittype == 'Cos':
            return self.Cos
        elif fittype == 'ExpCos':
            return self.ExpCos
        elif fittype == 'ExpCosiSin':
            return self.ExpCosiSin
        elif fittype == 'Exp(Exp)':
            return self.ExpExp
        elif fittype == 'Exp(ExpCos)':
            return self.ExpExpCos
        elif fittype == 'Poly2':
            return self.Poly2
        elif fittype == 'x2':
            return self.x2
        elif fittype == 'x4':
            return self.x4
        elif fittype == 'x6':
            return self.x6
        elif fittype == 'x61':
            return self.x61
        else:
            raise ValueError("no such fittype <{fittype}>")
        
    def freq_axis(self, dt, length):
        """returns an array of frequencies"""
        return np.linspace(-1/(2*dt), 1/(2*dt) - 1/(length*dt), length)
    
    def peak_freq(self, ffttrace, freqs):
        """Find the peak frequency in an fft trace. Assumes fftshift has been applied, and ignores the DC componnent
        common usage: peak_freq(*fft_trace(trace, dt)) """
        n = len(ffttrace)
        startInd = int(n/2+1)
        if n > 4:
            amps = abs(ffttrace[startInd:n-1])
            maxval = max(amps)
            inds = np.where(amps==maxval)
            return freqs[startInd+inds[0][0]]
        else:
            return freqs[0]*1e9 # if ffttrace is too short, just return some frequency
        
    def fft_trace(self,  trace, dt):
            """fft a 1D numpy array and return the fft with proper axes in MHz units"""
            ffttrace = fftshift(fft(trace))
            freqs = self.freq_axis( dt, length = len(trace))
            return ffttrace, freqs
    
    def _fit(self, func = None, time = None, trace = None, guess = None, stat_error = None, bounds = (-np.inf, np.inf)):
        func = func if not func is None else self.func
        time = time if not time is None else self.time
        guess = guess if not guess is None else self.guess
        stat_error = stat_error if not stat_error is None else self.error
        trace = trace if not trace is None else self.trace
        
        if bounds == (-np.inf, np.inf):
            if self.fittype == 'ExpCos':
                bounds = ([0, 0, 0, -np.inf, -np.inf], np.inf)
            if self.fittype == 'Cos':
                bounds = ([0, 0, -np.inf, -np.inf], np.inf)
        try: 
            self.is_succeed = True
            return curve_fit(func, xdata = time.astype(np.float64), ydata = trace.astype(np.float64), p0 = guess, sigma = stat_error, bounds = bounds, absolute_sigma = self.absolute_sigma) 
        except RuntimeError: 
            self.is_succeed = False
            print('***************************\n Could not find fit \n***************************')
            return [0,[0]]
    
    def get_fit_results(self):
        return [self.fit_results, self.cov_results]



def fit_circle(x,y, guess = None, method = 'Nelder-Mead', tol = None):
    if tol is None:
        tol = np.mean(np.diff(x)**2+np.diff(y)**2)/10
    x = np.array(x)
    y = np.array(y)
    xmean = np.mean(x)
    ymean = np.mean(y)
    if guess is None:
        guess = np.array([xmean, ymean, np.sqrt(np.max((x-xmean)**2+(y-ymean)**2))])
        # guess = np.array([xmean, ymean, 0])
    fit = minimize(circle, guess, args = (x,y), method = method, tol = tol)
    return fit.x
    
def circle(x0y0r, x, y):
    x0,y0,r=x0y0r
    return np.sum(np.abs((x0-x)**2+(y0-y)**2-r**2))



class non_TimeDomain_fit(object):
    def __init__(self, fittype, Data, x_axis, error = None, itrace = None, is_absolute_sigma = True, bounds = (-np.inf,np.inf),  **kwargs):
        if fittype == 'GaussianCosiSin' and itrace is None: raise ValueError(f"Expected itrace as well (imaginary trace) for fit type <{fittype}>")
        if itrace is not None: self.is_complex = True
        else: self.is_complex = False
        Data = np.array(Data)
        x_axis = np.array(x_axis)
        self.fittype = fittype
        self.Data = Data
        self.x_axis = x_axis
        dx = np.diff(x_axis)[0]
        self.dx = np.diff(x_axis)[0]
        
        self.absolute_sigma = is_absolute_sigma
        self.error = error
        self.func = self.fit_function(self.fittype)  
        if fittype in ['Lorentzian','Gaussian','Symmetric_Gaussian']:
            
            # self.fit_params, self.fit_cov,self.fit_sign  = self.fit_attempt(self.fittype,**kwargs)
            arg_max = np.argmax(np.abs(Data-Data[-1]))
            center_guess = x_axis[arg_max]
            # if bounds != (-np.inf,np.inf):
            #     center_guess = 0
            amp_guess = Data[arg_max]-Data[-1]
            offset_guess = Data[-1]
            arg_std = np.argmin(np.abs(Data-Data[-1]-amp_guess*np.exp(-1/2)))
            std_guess = x_axis[arg_max]-x_axis[arg_std]
            guess = amp_guess, center_guess, std_guess, offset_guess
            if fittype == 'Lorentzian':
                arg_fwhm = np.argmin(np.abs(Data-Data[-1]-amp_guess*0.5))
                fwhm_guess = x_axis[arg_max]-x_axis[arg_fwhm]
                guess = amp_guess, std_guess, offset_guess, center_guess
                
            if fittype == 'Symmetric_Gaussian':
                guess = amp_guess, std_guess, offset_guess
            try: 
                self.is_succeed = True
                self.fit_params, self.fit_cov  = curve_fit(self.func, x_axis, Data, p0=guess, sigma=error, absolute_sigma=is_absolute_sigma, bounds = bounds)
            except Exception as e:
                print(e)
                self.is_succeed = False
                print('***************************\n Could not find fit \n***************************')
                self.fit_params, self.fit_cov = 0,0
        elif fittype == 'Line':
            a_guess = (Data[-1]-Data[0])/(x_axis[-1]-x_axis[0])
            b_guess = Data[0]-a_guess*x_axis[0]
            guess = a_guess,b_guess
            try: 
                self.is_succeed = True
                self.fit_params, self.fit_cov  = curve_fit(self.func, x_axis, Data, p0=guess, sigma=error, absolute_sigma=is_absolute_sigma)
            except Exception as e:
                print(e)
                self.is_succeed = False
                print('***************************\n Could not find fit \n***************************')
                self.fit_params, self.fit_cov = 0,0
            
        else:
            if fittype == "GaussianCosiSin":
                self.sGuess = self.smart_guess(Data+1j*itrace, self.dx)
                self.guess  = self.fit_guess(self.fittype, *self.sGuess,**kwargs )
                abs_data = np.sqrt(Data**2+itrace**2)
                arg_max = np.argmax(np.abs(abs_data-abs_data[-1]))
                center_guess = x_axis[arg_max]
                amp_guess = abs_data[arg_max]-abs_data[-1]
                offset_guess = abs_data[-1]
                arg_std = np.argmin(np.abs(abs_data-abs_data[-1]-amp_guess*np.exp(-1/2)))
                std_guess = np.abs(x_axis[arg_max]-x_axis[arg_std])
                guess = amp_guess,center_guess,std_guess,offset_guess
                try:
                    self.is_succeed = True
                    fit_gaussian,_  = curve_fit(Gaussian, x_axis, abs_data, p0=guess, absolute_sigma=is_absolute_sigma)
                    self.guess[0] = fit_gaussian[0] #Amp
                    self.guess[2] = fit_gaussian[2] #sigma
                    self.guess[3] = 0 #delta
                    self.guess[4] = fit_gaussian[1] #center
                    self.guess[5] = fit_gaussian[3] #offset
                    self.Data = np.append(Data, itrace) 
                    self.fit_params, self.fit_cov = self._fit()
                except Exception as e:
                    print(e)
                    self.is_succeed = False
                    print('***************************\n Could not find fit \n***************************')
                    self.fit_params, self.fit_cov = 0,0
            else:
                
                self.sGuess = self.smart_guess(Data, self.dx)
                self.guess  = self.fit_guess(self.fittype, *self.sGuess, **kwargs)
                self.fit_params, self.fit_cov = self._fit()
                self.fit_sign = 1
                
        
    def fit_components(self,  fittype, fit,Gamma = 0,**kwargs):
        """ """
        Amp = fit[0]
        if fittype == 'Exp':
            Freq = 0
            Gamma = fit[1]
            Delta = 0;
        elif fittype == 'Cos':
            Freq = fit[1]*1
            Gamma = 0
            Amp, Delta = self.correct_delta(Amp, fit[2])
        elif fittype == 'ExpCos':
            Freq = fit[1]
            Gamma = fit[2]
            Amp, Delta = self.correct_delta(Amp, fit[3])
        elif fittype == 'Exp(Exp)':
            Freq = 0
            Gamma = fit[2]
            Delta = 0;
        Offset = fit[len(fit)-1]
        return Amp, Freq, Gamma, Delta, Offset   
    
    def freq_axis(self, dt, length):
        """returns an array of frequencies"""
        return np.linspace(-1/(2*dt), 1/(2*dt) - 1/(length*dt), length)
    
    def peak_freq(self, ffttrace, freqs):
        """Find the peak frequency in an fft trace. Assumes fftshift has been applied, and ignores the DC componnent
        common usage: peak_freq(*fft_trace(trace, dt)) """
        
        n = len(ffttrace)
        if self.is_complex:
            startInd = 0
        else:
            startInd = int(n/2+1)
        if n > 4:
            amps = abs(ffttrace[startInd:n-1])
            maxval = max(amps)
            inds = np.where(amps==maxval)
            return freqs[startInd+inds[0][0]]
        else:
            return freqs[0] # if ffttrace is too short, just return some frequency
    def fft_trace(self,  trace, dt):
            """fft a 1D numpy array and return the fft with proper axes in MHz units"""
            ffttrace = fftshift(fft(trace))
            freqs = self.freq_axis( dt, length = len(trace))
            return ffttrace, freqs
    def correct_delta(self,  FitAmp, FitDelta):
        """If the amplitude is negative, make it positive and shift the phase by pi instead"""
        if FitAmp<0:
            FitAmp = abs(FitAmp)
            FitDelta = (FitDelta+pi)%(2*pi)
        return FitAmp, FitDelta

    def get_fit_results(self):
        return [self.fit_results, self.cov_results]


    def smart_guess(self,  trace, dt):
        [traceFFT, traceFreqs] = self.fft_trace( trace, dt)
        Freq =1*self.peak_freq(traceFFT, traceFreqs)
        Amp = np.abs(max(trace)-min(trace))
        Offset = trace.mean(0)
        Gamma = 0
        Delta = np.angle(traceFFT[traceFreqs == Freq])[0]
        Sigma = 1
        return Amp, Freq, Gamma, Sigma, Delta, Offset

    def fit_guess(self,  fittype, Amp, Freq=0*1e-9, Gamma=0*1e-9, Sigma=1, Delta=0, Offset=0, alphaSize=1):
        """Basically, this converts a bunch of parameters into a fit type. Pass as numbers, not labrad units"""
        if fittype == 'Exp':
            return [Amp, Gamma, Offset]
        elif fittype == 'Cos':
            return [Amp, Freq, Delta, Offset]
        elif fittype == 'ExpCos':
            return [Amp, Freq, Gamma, Delta, Offset]
        elif fittype == 'Exp(Exp)':
            return [Amp, alphaSize, Gamma, Offset]
        elif fittype == 'Exp(ExpCos)':
            return [Amp, alphaSize,Freq, Gamma, Delta, Offset]
        elif fittype == 'GaussianCos':
            return [Amp, Freq, Sigma, Delta, 0, Offset]
        elif fittype == 'GaussianCosiSin':
            return [Amp, Freq, Sigma, Delta, 0, Offset]
        elif fittype == 'Poly2':
            return [0, 0, Offset - Amp]
        
    def fit_function(self,  fittype):
        """Select the fit function"""
        if fittype == 'Gaussian':
            return Gaussian
        if fittype == 'Symmetric_Gaussian':
            return Symmetric_Gaussian
        if fittype == 'Lorentzian':
            return Lorentzian
        # if fittype == 'Exp':
        #     return self.Exp
        if fittype == 'Poly2':
            return self.Poly2
        if fittype == 'Cos':
            return self.Cos
        if fittype == 'Line':
            return self.Line
        if fittype == 'Line180':
            return self.Line180
        if fittype == 'ExpCos':
            return self.ExpCos
        if fittype == 'GaussianCos':
            return self.GaussianCos
        if fittype == 'GaussianCosiSin':
            return self.GaussianCosiSin
        raise  ValueError('no such fittype')
        
    def _fit(self, func = None, x_axis = None, Data = None, guess = None, stat_error = None):
        func = func if not func is None else self.func
        x_axis = x_axis if not x_axis is None else self.x_axis
        guess = guess if not guess is None else self.guess
        stat_error = stat_error if not stat_error is None else self.error
        Data = Data if not Data is None else self.Data
        # print(guess)
        try: 
            self.is_succeed = True
            return curve_fit(func, xdata = x_axis, ydata = Data, p0 = guess, sigma = stat_error, absolute_sigma = self.absolute_sigma)
        except Exception as e:
            print(e)
            self.is_succeed = False
            print('***************************\n Could not find fit \n***************************')
            return [0,[0]]
        
    def fit_to_positive_lorentzian(self,x,y,guess = None):#center at maximum
        if guess is None:   
            b_guess = np.abs(x[np.argmin(np.abs(y-(max(y)+min(y))/2))]-x[np.argmax(y)])   # b_guess :)
            a_guess = (np.max(y)-np.min(y))*b_guess**2
            guess =  [a_guess, b_guess, np.min(y),x[np.argmax(y)]]
        return curve_fit(self.Lorentzian,x,y,p0 = guess)
    #TODO calculate fit and statitical error
    
    def fit_to_lorentzian(self,x,y,guess = None):
        try:
           popt_p,pcov_p = self.fit_to_positive_lorentzian(x,y,guess = guess)
        except:
           popt_m,pcov_m = self.fit_to_positive_lorentzian(x,-y,guess = guess)
           return popt_m,pcov_m,-1
       
        try:
           popt_m,pcov_m = self.fit_to_positive_lorentzian(x,-y,guess = guess)
        except:
           return popt_p,pcov_p,1
        
        if max(abs(self.Lorentzian(x,*popt_p)-y))>max(abs(self.Lorentzian(x,*popt_m)+y)):
            return popt_m,pcov_m,-1
        return popt_p,pcov_p,1
    
    def Cos(self, t, A, f, delta, offset): # 4 args
        return A*cos(t*2*pi*f+delta)+offset
    def Line(self, x, a, b):
        return a*x + b
    def Line180(self, x, b):
        return -180*x + b
    def Poly2(self, x, a, b, c):
        return (a*(x-b)**2 + c).astype(np.float64)
    def ExpCos(self,  t, A, f, gamma, delta, offset): # 5 args
        return A*exp(-t*gamma)*cos(t*2*pi*f+delta) + offset
    def GaussianCos(self,  x, A, f, sigma, delta, center, offset): # 5 args
        return A*exp(-(x-center)**2/(2*sigma**2))*cos(x*2*pi*f+delta) + offset
    def GaussianCosiSin(self,  x, A, f, sigma, delta, center, offset): # 5 args
        complex_res = A*exp(-(x-center)**2/(2*sigma**2))*exp(1j*(x*2*pi*f+delta)) + offset
        return np.append(complex_res.real, complex_res.imag+offset)
    
    def fit_multiple_gaussians(self, x,y, noise_level= None):
        if noise_level is  None:
            print('how many gaussians')
            N_gauss = input()
            guess = []
            print('what is the center farthest to the left')
            first_center = input()
            print('what is chi')
            chi = input()
            print('what is gaussian width')
            width = [input()]
            for i in range(N_gauss):
                print('what is gaussian center')
                # guess += [input()]
                guess += [i*chi+first_center]
                print('what is gaussian amp')
                guess +=  [input()]
                guess += width
            from scipy.optimize import curve_fit
            popt, pcov = curve_fit(self.multiple_gaussians, x, y, p0=guess)
            
        frequency = []; amplitudes = []
        for i in range(3, len(popt)+3, 3):
            freq+=popt[i]
            amplitudes+=popt[i+1]
        return popt,pcov, freq,amplitudes
    def multiple_gaussians(self,x,a,b,c, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            ctr = params[i]
            amp = params[i+1]
            wid = params[i+2]
            y = y + amp * np.exp( -((x - ctr)/wid)**2)
        return y + a*x**2+b*x+c            
    
def Lorentzian(x,a,b,c,d):
    return (a*b**2/((x-d)**2+b**2) + c)
def Gaussian(x,amp,center,std,C):
    return amp*exp(-((x-center)**2)/(2*std**2))+C
def Symmetric_Gaussian(x,amp,std,C):
    return amp*exp(-((x)**2)/(2*std**2))+C

# def _fit(self, func = None, time = None, trace = None, guess = None, stat_error = None):
#     func = func if not func is None else self.func
#     time = time if not time is None else self.time
#     guess = guess if not guess is None else self.guess
#     stat_error = stat_error if not stat_error is None else self.error
#     trace = trace if not trace is None else self.trace
    
#     try: 
#         self.is_succeed = True
#         return curve_fit(func, xdata = time, ydata = trace, p0 = guess, sigma = stat_error)
#     except: 
#         self.is_succeed = False
#         print('***************************\n Could not find fit \n***************************')
#         return [0,[0]]
"""  
Credits:
=======

[1] Yvonne Y. Gao, Multi-Cavity Operations in Circuit Quantum Electrodynamics 

"""

