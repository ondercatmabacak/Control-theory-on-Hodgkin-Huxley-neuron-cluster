#!/usr/bin/env python3 
from __future__ import division
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import sys
import scipy.fftpack as fft
import scipy.integrate as integrate
import scipy.special as special
from sympy import powsimp

#computing of resting potential with Nerst FOR HH1 and HH2 
Naext1=440
Naint1=50
Kext1=20
Kint1=400
Clext1=560
Clint1=150
PK1=1
PNa1=0.04
PCl1=0.45

Vrest1=58*np.log10(((PK1*Kext1)+(PNa1*Naext1)+(PCl1*Clint1))/((PK1*Kint1)+(PNa1*Naint1)+(PCl1*Clext1)))

Naext2=440
Naint2=50
Kext2=20
Kint2=400
Clext2=560
Clint2=150
PK2=1
PNa2=0.04
PCl2=0.45

Vrest2=58*np.log10(((PK2*Kext2)+(PNa2*Naext2)+(PCl2*Clint2))/((PK2*Kint2)+(PNa2*Naint2)+(PCl2*Clext2))) 

# simulation time 
T     = 30   # ms
dt    = 0.01 # ms
time  = np.arange(0,T,dt)

# constant parameters FOR HH1 and HH2 
Cm=1             
gNamax1=120
gNamax2=120      
gKmax1=36
gKmax2=36         
gLmax1=0.3
gLmax2=0.3
EK1 = -12
ENa1=115
EL1=10.6
EK2 = -12
ENa2=115
EL2=10.6

V2          = np.zeros((len(time))) # mV
V2[1]       = Vrest2
a_m2        = np.zeros((len(time)))
a_n2        = np.zeros((len(time)))
a_h2        = np.zeros((len(time)))
b_m2        = np.zeros((len(time)))
b_n2        = np.zeros((len(time)))
b_h2        = np.zeros((len(time)))
m2          = np.zeros((len(time)))
n2          = np.zeros((len(time)))
h2          = np.zeros((len(time)))
gNa2        = np.zeros((len(time)))
gK2         = np.zeros((len(time)))
INa2        = np.zeros((len(time)))
IK2         = np.zeros((len(time)))
IL2         = np.zeros((len(time)))
F2          = np.zeros((len(time)))
I2          = np.zeros((len(time)))
IK2_dot     = np.zeros((len(time)))
INa2_dot    = np.zeros((len(time)))
F2_dot      = np.zeros((len(time)))
I2_dot      = np.zeros((len(time)))

INa2_dot  = gNamax2*(((120/(25-V2 )+(20*V2 /3)*(25-V2 ))*(np.exp((225-14*V2 )/90) - np.exp(-V2 /18)))*np.power((1+(40*(np.exp((225-14*V2 )/90) - np.exp(-V2 /18)))/(25-V2 )),-4)*np.power((1+100/(7*(np.exp((60-3*V2 )/20) - np.exp(-V2 /20)))),-1)*np.power((1+(40*(np.exp((225-14*V2 )/90) - np.exp(-V2 /18)))/(25-V2 )),-3)*(-10*V2 *((np.exp((225-14*V2 )/90) - np.exp(-V2 /20)))/(7*np.power((np.exp((60-3*V2 )/20)-np.exp(-V2 /20)),2))))*(V2 -ENa2)+gNamax2*np.power(m2 ,3)*h2 *(V2  - ENa2)

K = powsimp(INa2_dot)

show ()
