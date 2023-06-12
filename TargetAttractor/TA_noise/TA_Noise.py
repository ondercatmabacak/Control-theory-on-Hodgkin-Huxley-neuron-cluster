#!/usr/bin/env python3 
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import *
import scipy as sp
import random
import scipy.integrate as integrate
from scipy.integrate import quad
import scipy.special as special
import sympy as sym
import pink_noise
import sys
from pylab import *
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

# set the initial states FOR HH1 and HH2 
# For HH1
alpha=1.0
TAU=0.01
A=10.0
#Vst=70

a=2.0
b=3.0
c=4.0
d=5.0

Vst         = a*np.cos(b*time + c) + b*np.cos(c*time + d) + c*np.cos(d*time + a) + d*np.cos(a*time + b) 

Vst_dot     = -a*b*np.sin(b*time + c) - b*c*np.sin(c*time + d) - c*d*np.sin(d*time + a) + d*a*np.sin(a*time + b)

Vst_dot_dot = -np.power(a,2)*b*np.cos(b*time + c) - b*np.power(c,2)*np.cos(c*time + d) - c*np.power(d,2)*np.cos(d*time + a) + d*np.power(a,2)*np.cos(a*time + b)
#Vst= a*np.exp(-(np.power((time-b),c))/(np.power(d,2)))

'''
plt.plot(time,Vst)
plt.plot(time,Vst_dot)
plt.plot(time,Vst_dot_dot)
show()
quit()
'''
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
F1          = np.zeros((len(time)))
I1          = np.zeros((len(time)))

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
PN          = np.zeros((len(time)))
PNV         = np.zeros((len(time)))
PNn         = np.zeros((len(time)))
PNm         = np.zeros((len(time)))
PNh         = np.zeros((len(time)))
delta       = np.zeros((len(time)))
V_out       = np.zeros((len(time)))
V_out_dot   = np.zeros((len(time)))

length = len(pink_noise.z)
epsilon=np.linspace(0,1.01,length)
PN = 2*epsilon*(pink_noise.z-1/2)
PNV= PN*1.0
PNm= PN*1.0
PNn= PN*1.0
PNh= PN*1.0

#Vst=70
#Vst= a*np.cos(w*time + f)
#Vst= a*np.exp(-(np.power((time-b),2))/(np.power(c,2)))

#Vst_dot = 0
#Vst_dot= -a*w*np.sin(w*time+f)
#Vst_dot= -(2*a)*(time-b)/(np.power(c,2))*np.exp(-(np.power((time-b),2))/(np.power(c,2)))

#Vst_dot_dot = 0
#Vst_dot_dot= -a*np.power(w,2)*np.cos(w*time+f)
#Vst_dot_dot= (-((2*a)/np.power(c,2) + np.power((2*a)*(time-b)/np.power(c,2),2)*np.exp(-(np.power((time-b),2))/np.power(c,2))))

'''
def integrand(time):
    return abs(V1[i] - Vst[i]) / Vst[i]
I = quad(integrand, 10, 20, args=(a,b,c,f,w))
'''



for i in range(1,len(time)-1):
    
    a_n1[i]=0.01*((10-V1[i]) / (np.exp((10-V1[i])/10)-1) ) 
    b_n1[i]=0.125*np.exp(-V1[i]/80) 
    a_m1[i]=0.1*((25-V1[i]) /(np.exp((25-V1[i])/10)-1) ) 
    b_m1[i]=4*np.exp(-V1[i]/18) 
    a_h1[i]=0.07*np.exp(-V1[i]/20) 
    b_h1[i]=1/(np.exp((30-V1[i])/10)+1) 
    
    m1[i+1]= dt*(a_m1[i]*(1 - m1[i]) - b_m1[i]*m1[i]) + PNm[i] 
    h1[i+1]= dt*(a_h1[i]*(1 - h1[i]) - b_h1[i]*h1[i]) + PNh[i]
    n1[i+1]= dt*(a_n1[i]*(1 - n1[i]) - b_n1[i]*n1[i]) + PNn[i]

    gK1[i] =gKmax1*np.power(n1[i],4) 
    gNa1[i]=(np.power(m1[i],3))*gNamax1*h1[i] 
    INa1[i]=(V1[i]-ENa1)*gNa1[i] 
    IK1[i] =(V1[i]-EK1)*gK1[i] 
    IL1[i] =(V1[i]-EL1)*gLmax1 
    
    a_n2[i]=.01*((10-V2[i]) / (np.exp((10-V2[i])/10)-1) ) 
    b_n2[i]=.125*np.exp(-V2[i]/80) 
    a_m2[i]=.1*((25-V2[i]) /(np.exp((25-V2[i])/10)-1) ) 
    b_m2[i]=4*np.exp(-V2[i]/18) 
    a_h2[i]=.07*np.exp(-V2[i]/20) 
    b_h2[i]=1/(np.exp((30-V2[i])/10)+1) 

    m2[i+1]= dt*(a_m2[i]*(1 - m2[i]) - b_m2[i]*m2[i]) + PNm[i]
    h2[i+1]= dt*(a_h2[i]*(1 - h2[i]) - b_h2[i]*h2[i]) + PNh[i]
    n2[i+1]= dt*(a_n2[i]*(1 - n2[i]) - b_n2[i]*n2[i]) + PNn[i]

    gK2[i] = gKmax2*np.power(n2[i],4) 
    gNa2[i]= np.power(m2[i],3)*gNamax2*h2[i] 
    
    INa2[i]= (V2[i]-ENa2)*gNa2[i] 
    IK2[i] = (V2[i]-EK2)*gK2[i] 
    IL2[i] = (V2[i]-EL2)*gLmax2 
    
    F2[i]     = INa2[i] + IK2[i] + IL2[i]
    I2[i]     = Cm*(Vst_dot[i] - 1/TAU*(V2[i] - Vst[i])-F2[i])
    V2[i+1]   = V2[i] + dt*Cm*(Vst_dot[i] -1/TAU*(V2[i] - Vst[i]))
    '''
    IK2_dot[i]  = gKmax2*(400/(8*V2[i+1])*(np.exp(1-7*V2[i]/80)-np.exp(V2[i]/80))+400/8*(10-V2[i])*((7*V2[i+1]/80)*np.exp(1-7*V2[i]/80)+V2[i+1]/80*np.exp(V2[i]/80)))*np.power((1+100/(8*(10-V2[i]))*(np.exp(1-7*V2[i]/80) - np.exp(V2[i]/80))),-5)*(V2[i]-EK2)+gKmax2*np.power(n2[i],4)*(V2[i+1] - EK2)
    
    INa2_dot[i] = gNamax2*(((120/(25-V2[i+1])+(20*V2[i+1]/3)*(25-V2[i]))*(np.exp((225-14*V2[i])/90) - np.exp(-V2[i]/18)))*np.power((1+(40*(np.exp((225-14*V2[i])/90) - np.exp(-V2[i]/18)))/(25-V2[i])),-4)*np.power((1+100/(7*(np.exp((60-3*V2[i])/20) - np.exp(-V2[i]/20)))),-1)*np.power((1+(40*(np.exp((225-14*V2[i])/90) - np.exp(-V2[i]/18)))/(25-V2[i])),-3)*(-10*V2[i+1]*((np.exp((225-14*V2[i])/90) - np.exp(-V2[i]/20)))/(7*np.power((np.exp((60-3*V2[i])/20)-np.exp(-V2[i]/20)),2))))*(V2[i]-ENa2)+gNamax2*np.power(m2[i],3)*h2[i]*(V2[i+1] - ENa2)
    
    F2_dot[i] = IK2_dot[i] + INa2_dot[i] + IL2[i]
    I2_dot[i] = Cm*(Vst_dot_dot[i] - F2_dot[i] - 1/TAU*(V2[i+1] - Vst_dot[i]))
    '''
    #time      = Symbol('time')
    
    F2_dot[i] = (F2[i]-F2[i-1])/dt #F2[0]=0 TANIMLAMAN LAZIM !!!
    #V_out_dot= I2_dot[i] / A
    
    V_out[i]     = I2[i]/A
    #V_out_dot[i]= sp.diff(V_out[i],time)
    V_out_dot[i] = (V_out[i+1] - V_out[i])/dt
    F1[i]        = gNa1[i]*(V1[i] - ENa1) + gK1[i]*(V1[i] - EK1) + gLmax1*(V1[i] - EL1)
    I1[i]        = Cm*(V_out_dot[i] -F1[i] - 1/TAU*(V1[i] - V_out[i]))
    V1[i+1]      = V1[i] + dt*Cm*(Vst_dot[i] -1/TAU*(V1[i] - V_out[i])) 
    
    delta[i]= abs(V2[i]-Vst[i]) ### ACHIEVABILITY & ERROR ANALYSIS
    
nDELTA     = np.sum(delta)
'''
'GÜFÜ BURADA İNTEGRALİ NETTİ' ?
'''
ndelta_bar = abs(nDELTA*dt/(time[20]-time[10]))
DELTA      = np.linspace(0, nDELTA,length)        
delta_bar  = np.linspace(0,ndelta_bar,length)
#print(V1)
#quit()


# applying current, 
figure()
subplot2grid((4,1), (0, 1))
plt.plot(time,I1, label='Iapp1') 
ylabel('Iapp1 (mA)') 
xlabel('I1 (mA)') 
title('Vst = a*np.cos(b*time + c) + b*np.cos(c*time + d) + c*np.cos(d*time + a) + d*np.cos(a*time + b)')
ylim(-120.0,120.0)
plt.grid()
plt.hold(True)

subplot2grid((4,1), (2, 1))
plt.plot(time,I2, label='Iapp2') 
ylabel('I2 (mA)') 
xlabel('time (ms)') 
ylim(-40.0,40.0)
plt.grid()

#plot voltage 

figure()
subplot2grid((6,1), (0, 1))
plt.plot(time,V1, label='V1') 
ylabel('V1 (ms)')
xlabel('time(ms)') 
title('Vst= a*np.cos(b*time + c) + b*np.cos(c*time + d) + c*np.cos(d*time + a) + d*np.cos(a*time + b)')
ylim(-20.0,20.0)
plt.grid()
plt.hold(True)

subplot2grid((6,1), (2, 1))
plt.plot(time,V2, label='V2') 
ylabel('V2 (ms)')
xlabel('time(ms)')
ylim(-20.0,20.0)
plt.grid()
plt.hold(True)

subplot2grid((6,1), (4, 1))
plt.plot(time,Vst, label='V*')
ylabel('Vstar')
xlabel('time(ms)') 
ylim(-20.0,20.0)
xlim (0,30.0)
plt.grid()
# Achievability & Error analysis

figure()
subplot2grid((4,1), (0, 1))
plt.plot(time,delta, label='0.1') 
ylabel('delta')
xlabel('time(ms)') 
title('Vst=a*np.cos(b*time + c) + b*np.cos(c*time + d) + c*np.cos(d*time + a) + d*np.cos(a*time + b)')
ylim(-0.3,0.3)
plt.grid()
plt.hold(True)

subplot2grid((4,1), (2, 1))
plt.plot(epsilon,delta_bar, label='delta bar') 
ylabel('delta_bar')
xlabel('epsilon')
ylim(0,20)
xlim(0.0,1.0)
plt.grid()
show()
