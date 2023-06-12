#!/usr/bin/env python3 
from __future__ import division
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import sys

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

# simulation time 
T     = 30   # ms
dt    = 0.01 # ms
time  = np.arange(0,T,dt)

# constant parameters FOR HH1 and HH2 
Cm=1             
gNamax1=120
gKmax1=36
gLmax1=0.3
EK1 = -12
ENa1=115
EL1=10.6

# set the initial states FOR HH1

V1          = np.zeros((len(time))) # mV
V1[1]       = Vrest1
a_m1        = np.zeros((len(time)))
a_n1        = np.zeros((len(time)))
a_h1        = np.zeros((len(time)))
b_m1        = np.zeros((len(time)))
b_n1        = np.zeros((len(time)))
b_h1        = np.zeros((len(time)))
m1          = np.zeros((len(time)))
n1          = np.zeros((len(time)))
h1          = np.zeros((len(time)))
gNa1        = np.zeros((len(time)))
gK1         = np.zeros((len(time)))
INa1        = np.zeros((len(time)))
IK1         = np.zeros((len(time)))
IL1         = np.zeros((len(time)))
Iapp1       = np.zeros((len(time)))
IC1         = np.zeros((len(time)))

alpha=1.0
r=10.0
#Vst=70
Vst= 1.0*np.cos(2.0*time + 3.0)
#Vst= 1.0*np.exp(-(np.power((time-2.0),2))/(np.power(3.0,2)))

for i in range(1,len(time)-1):
    Vst[0] = 0
    Vst[i-1] = int(i-1)
    a_n1[i]=0.01*((10-V1[i]) / (np.exp((10-V1[i])/10)-1) ) 
    b_n1[i]=0.125*np.exp(-V1[i]/80) 
    a_m1[i]=0.1*((25-V1[i]) /(np.exp((25-V1[i])/10)-1) ) 
    b_m1[i]=4*np.exp(-V1[i]/18) 
    a_h1[i]=0.07*np.exp(-V1[i]/20) 
    b_h1[i]=1/(np.exp((30-V1[i])/10)+1) 
    
    n1[i]=a_n1[i]/(a_n1[i]+b_n1[i]) 
    m1[i]=a_m1[i]/(a_m1[i]+b_m1[i]) 
    h1[i]=a_h1[i]/(a_h1[i]+b_h1[i]) 
    
    m1[i+1]= dt*(a_m1[i]*(1 - m1[i]) - b_m1[i]*m1[i])
    h1[i+1]= dt*(a_h1[i]*(1 - h1[i]) - b_h1[i]*h1[i])
    n1[i+1]= dt*(a_n1[i]*(1 - n1[i]) - b_n1[i]*n1[i])

    gK1[i]=gKmax1*np.power(n1[i+1],4) 
    gNa1[i]=(np.power(m1[i+1],3))*gNamax1*h1[i+1] 
    INa1[i]=(V1[i]-ENa1)*gNa1[i] 
    IK1[i]=(V1[i]-EK1)*gK1[i] 
    IL1[i]=(V1[i]-EL1)*gLmax1 
    
    Iapp1[i]=-r*(V1[i]-Vst[i])/Cm 

    IC1[i]=Iapp1[i]-IK1[i]-INa1[i]-IL1[i] 
    V1[i+1]=V1[i]+dt*IC1[i]/Cm 

figure()  

plt.subplot(4,2,1)
plt.plot(time,V1, label='V1') 
ylabel('V1(mV)') 
xlabel('time')
ylim(-30.0,30.0)
xlim(0.0,30.0)
plt.grid()
plt.hold(True) 

plt.subplot(4,2,2)
plt.plot(time,Iapp1, label='Iapp1') 
ylabel('Applied Current (mA)') 
xlabel('time')
xlim(0.0,30.0)
ylim(-15.0,15.0)
plt.grid()
plt.hold(True) 

plt.subplot(4,2,3)
plt.plot(time,IC1, label='IC1') 
ylabel('Control Current (mA)') 
xlabel('time')
xlim(0.0,30.0)
ylim(-15.0,15.0)
plt.grid()
show()
