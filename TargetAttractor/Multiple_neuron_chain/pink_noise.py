from __future__ import division
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from matplotlib import mlab
import sys
import random
from numpy.fft import fft, fftshift, ifft, ifftshift
from scipy.fftpack import fft, fftfreq
import math

dt = 0.01 
T = 30
N=50000

#t = np.linspace(0, T, T/dt)
#x = 0.4 * np.cos(2*np.pi*f*t) + np.cos(2*np.pi*f1*t)

if N%2:
    M = N+1  
else:
    M = N 

#generate white noise
x = np.random.rand(1, M) # burada random sayilar buluyorsun ama list of list seklinde
x = x[0]                 # [[1,2,3,4,5,6]] olacagina [1,2,3,4,5,6] ya cevirdim, en temel boyut hatasi buydu
                         # boyutu 1500 yerine 1 gosteriyordu cunku liste icinde 1 tane liste var
#FFT
X = fft(x) 
# prepare a vector for 1/f multiplication
NumUniquePts = M/2.0 + 1.0
n = np.arange(1.0,NumUniquePts) # NumUniquePts hesaplarken zaten float yapiyor gerek yok dt.float kismina
n = sqrt(n) 
'''
multiplicate the left half of the spectrum so the power spectral density
is proportional to the frequency by factor 1/f, i.e. the
amplitudes are proportional to 1/sqrt(f)
'''
NumUniquePts = int(NumUniquePts) # ama float olarak hesaplandigindan indis olarak kullanamiyorsun o yuzden integer a cevirdim
#print (M)
#quit()

X[:NumUniquePts-1] = X[:NumUniquePts-1]/n
'''
# prepare a right half of the spectrum - a copy of the left one,
# except the DC component and Nyquist frequency - they are unique

# [:NumUniquePts-1] array in ilk yarisini [NumUniquePts-1:] ikinci yarisini gosteriyor
# bizim durumumuzda [0:1500] ve [1500:3000] seklinde ikiye ayrildi array
'''

X[NumUniquePts-1:] = X[NumUniquePts-2:-1:1] - 1j*X[NumUniquePts-2:-1:1] # np.complex tekrardan yazman gerekiyor
# IFFT
y = ifft(X) 

# prepare output vector y
k=np.arange(1,N+1)
y = (y[k-1]).real

# ensure unity standard deviation and zero mean value
y = y - mean(y) 
yrms = math.sqrt(mean(np.power(y,2))) 
y = y/yrms 

### power spectrum###

freq = fftfreq(x.size, d=dt)
# Only keep positive frequencies.
keep = freq>=0
y = y[keep]
z=y/n
freq = freq[keep]
'''
ax1 = plt.subplot(111)
ax1.semilogx(freq,z)
ax1.set_xlim(1,30)
ax1.set_ylim(0,0.1)
plt.xlabel('frequency')
plt.ylabel('Power Spectrum')
#plt.show()
'''
