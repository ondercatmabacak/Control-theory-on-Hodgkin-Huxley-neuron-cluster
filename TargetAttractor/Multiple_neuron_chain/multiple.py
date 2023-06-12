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
 
#computing of resting potential with Nerst 
Naext=440 
Naint=50 
Kext=20 
Kint=400 
Clext=560 
Clint=150 
PK=1 
PNa=0.04 
PCl=0.45 
 
Vrest=58*np.log10(((PK*Kext)+(PNa*Naext)+(PCl*Clint))/((PK*Kint)+(PNa*Naint)+(PCl*Clext))) 
 
# simulation time  
T     = 30   # ms 
dt    = 0.01 # ms 
time  = np.arange(0,T,dt) 
 
# constant parameters FOR HH1 and HH2  
Cm    =1 
gNamax=120 
gKmax =36 
gLmax =0.3 
EK    =-12 
ENa   =115 
EL    =10.36 
 
# initial states 
 
''' 
erey=np.arange(0,1,0.1) 
 
delta_bar   = np.zeros((len(erey))) 
DELTA       = np.zeros((len(erey))) 
epsilon     = np.zeros((len(erey))) 
delta       = np.zeros((len(erey),len(time))) 
''' 
V1          = np.zeros((len(time))) # mV 
V1_dot   = np.zeros((len(time))) # mV 
V1_dot_dot   = np.zeros((len(time))) # mV 
V1_st       = np.zeros((len(time))) # mV 
V1_st_dot   = np.zeros((len(time))) # mV 
V1_st_dot_dot   = np.zeros((len(time))) # mV 
V1[1]       = Vrest 
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
F1_dot      = np.zeros((len(time))) 
F1_dot_dot  = np.zeros((len(time))) 
F1[0]       = 0.0 
F1[-1]      = 0.0 
I1_dot      = np.zeros((len(time))) 
I1_st       = np.zeros((len(time))) 
I1_st_dot   = np.zeros((len(time))) 
I1_st_dot_dot   = np.zeros((len(time))) 
IC          = np.zeros((len(time))) 
IT1         = np.zeros((len(time))) 
V2          = np.zeros((len(time))) # mV 
V2_dot   = np.zeros((len(time))) # mV 
V2_dot_dot   = np.zeros((len(time))) # mV 
V2_st       = np.zeros((len(time))) # mV 
V2_st_dot   = np.zeros((len(time))) # mV 
V2_st_dot_dot   = np.zeros((len(time))) # mV 
V2[1]       = Vrest 
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
F2_dot      = np.zeros((len(time))) 
F2_dot_dot  = np.zeros((len(time))) 
F2[0]       = 0.0 
F2[-1]      = 0.0 
I2_dot      = np.zeros((len(time))) 
I2_st       = np.zeros((len(time))) 
I2_st_dot   = np.zeros((len(time))) 
I2_st_dot_dot   = np.zeros((len(time))) 
IC          = np.zeros((len(time))) 
IT2         = np.zeros((len(time))) 
V3          = np.zeros((len(time))) # mV 
V3_dot   = np.zeros((len(time))) # mV 
V3_dot_dot   = np.zeros((len(time))) # mV 
V3[1]       = Vrest 
a_m3        = np.zeros((len(time))) 
a_n3        = np.zeros((len(time))) 
a_h3        = np.zeros((len(time))) 
b_m3        = np.zeros((len(time))) 
b_n3        = np.zeros((len(time))) 
b_h3        = np.zeros((len(time))) 
m3          = np.zeros((len(time))) 
n3          = np.zeros((len(time))) 
h3          = np.zeros((len(time))) 
gNa3        = np.zeros((len(time))) 
gK3         = np.zeros((len(time))) 
INa3        = np.zeros((len(time))) 
IK3         = np.zeros((len(time))) 
IL3         = np.zeros((len(time))) 
F3          = np.zeros((len(time))) 
I3          = np.zeros((len(time))) 
F3_dot      = np.zeros((len(time))) 
F3_dot_dot  = np.zeros((len(time))) 
F3[0]       = 0.0 
F3[-1]      = 0.0 
I3_dot      = np.zeros((len(time))) 
I3_st       = np.zeros((len(time))) 
I3_st_dot   = np.zeros((len(time))) 
I3_st_dot_dot   = np.zeros((len(time))) 
IC          = np.zeros((len(time))) 
IT3         = np.zeros((len(time))) 
PN          = np.zeros((len(time))) 
Ifb         = np.zeros((len(time))) 
PNV         = np.zeros((len(time))) 
PNn         = np.zeros((len(time))) 
PNm         = np.zeros((len(time))) 
PNh         = np.zeros((len(time))) 
delta       = np.zeros((len(time))) 
 
#epsilon=0.01 
length = len(pink_noise.z) 
epsilon=np.linspace(0,1.01,length) 
PN = 2*epsilon*(pink_noise.z-1/2) 
PNV= PN*1.5 
PNm= PN*1.0 
PNn= PN*1.0 
PNh= PN*1.0 
    
#A=10.0 
TAU=0.1
alpha=1.0 
#Vst=70 
a=1.0 
b=2.0 
c=3.0 
d=4.0 
Vst=a*np.cos(b*time + c) + b*np.cos(c*time + d) + c*np.cos(d*time + a) + d*np.cos(a*time + b)  
Vst_dot     = -a*b*np.sin(b*time + c) - b*c*np.sin(c*time + d) - c*d*np.sin(d*time + a) + d*a*np.sin(a*time + b) 
 
Vst_dot_dot = -a*np.power(b,2)*np.cos(b*time + c) - b*np.power(c,2)*np.cos(c*time + d) - c*np.power(d,2)*np.cos(d*time + a) + d*np.power(a,2)*np.cos(a*time + b) 
 
Vst_dot_dot_dot = a*np.power(b,3)*np.sin(b*time + c) + b*np.power(c,3)*np.sin(c*time + d) + c*np.power(d,3)*np.sin(d*time + a) + d*np.power(a,3)*np.sin(a*time + b) 
 
#Vst= a*np.exp(-(np.power((time-b),c))/(np.power(d,2))) 
for i in range(1,len(time)-1): 
 
    a_n1[i]=0.01*((10.0-V1[i]) / (np.exp(1.0-0.1*V1[i])-1.0) )  
    b_n1[i]=0.125*np.exp(-V1[i]/80.0)  
    a_m1[i]=0.1*((25.0-V1[i]) /(np.exp(2.5-0.1*V1[i])-1.0) )  
    b_m1[i]=4.0*np.exp(-V1[i]/18.0)  
    a_h1[i]=0.07*np.exp(-V1[i]/20.0)  
    b_h1[i]=1.0/(np.exp(3.0-V1[i])+1.0)  
    
    n1[i]=a_n1[i]/(a_n1[i]+b_n1[i])  
    m1[i]=a_m1[i]/(a_m1[i]+b_m1[i])  
    h1[i]=a_h1[i]/(a_h1[i]+b_h1[i])  
    
    gK1[i]=gKmax*np.power(n1[i+1],4)  
    gNa1[i]=(np.power(m1[i+1],3))*gNamax*h1[i+1]  
    
    INa1[i]=(V1[i]-ENa)*gNa1[i]  
    IK1[i]=(V1[i]-EK)*gK1[i]  
    IL1[i]=(V1[i]-EL)*gLmax  
    
    F1[i]=INa1[i] + IK1[i] + IL1[i] 
    F1_dot[i] = (F1[i]-F1[i-1])/dt 
    F1_dot_dot[i] = (F1[i]-2*F1[i-1]+F1[i-2])/np.power(dt,2) 
    
    m1[i+1]= dt*(a_m1[i]*(1.0 - m1[i]) - b_m1[i]*m1[i])+PNm[i] 
    h1[i+1]= dt*(a_h1[i]*(1.0 - h1[i]) - b_h1[i]*h1[i])+PNh[i] 
    n1[i+1]= dt*(a_n1[i]*(1.0 - n1[i]) - b_n1[i]*n1[i])+PNn[i] 
    V1[i+1]= V1[i] + dt*Cm*(V1_st_dot[i] -1.0/TAU*(V1[i] - V1_st[i])) 

    a_n2[i]=0.01*((10.0-V2[i]) / (np.exp(1.0-0.1*V2[i])-1.0) )  
    b_n2[i]=0.125*np.exp(-V2[i]/80.0)  
    a_m2[i]=0.1*((25.0-V2[i]) /(np.exp(2.5-0.1*V2[i])-1.0) )  
    b_m2[i]=4.0*np.exp(-V2[i]/18.0)  
    a_h2[i]=0.07*np.exp(-V2[i]/20.0)  
    b_h2[i]=1.0/(np.exp(3.0-0.1*V2[i])+1.0)  
    
    n2[i]=a_n2[i]/(a_n2[i]+b_n2[i])  
    m2[i]=a_m2[i]/(a_m2[i]+b_m2[i])  
    h2[i]=a_h2[i]/(a_h2[i]+b_h2[i])  
    V2[i+1]= V2[i] + dt*Cm*(V2_st_dot[i] -1/TAU*(V2[i] - V2_st[i])) 

    gK2[i]=gKmax*np.power(n2[i+1],4)  
    gNa2[i]=(np.power(m2[i+1],3))*gNamax*h2[i+1]  
    
    INa2[i]=(V2[i]-ENa)*gNa2[i]  
    IK2[i]=(V2[i]-EK)*gK2[i]  
    IL2[i]=(V2[i]-EL)*gLmax  
    
    F2[i]=INa2[i] + IK2[i] + IL2[i] 
    F2_dot[i] = (F2[i]-F2[i-1])/dt 
    F2_dot_dot[i] = (F2[i]-2*F2[i-1]+F2[i-2])/np.power(dt,2) 
    
    m2[i+1]= dt*(a_m2[i]*(1.0 - m2[i]) - b_m2[i]*m2[i])+PNm[i] 
    h2[i+1]= dt*(a_h2[i]*(1.0 - h2[i]) - b_h2[i]*h2[i])+PNh[i] 
    n2[i+1]= dt*(a_n2[i]*(1.0 - n2[i]) - b_n2[i]*n2[i])+PNn[i] 
    V3[i+1]= V3[i] + dt*Cm*(Vst_dot[i] -1/TAU*(V3[i] - Vst[i])) 

    a_n3[i]=0.01*((10.0-V3[i]) / (np.exp(1.0-0.1*V3[i])-1.0) )  
    b_n3[i]=0.125*np.exp(-V3[i]/80.0)  
    a_m3[i]=0.1*((25.0-V3[i]) /(np.exp(2.5-0.1*V3[i])-1.0) )  
    b_m3[i]=4.0*np.exp(-V3[i]/18.0)  
    a_h3[i]=0.07*np.exp(-V3[i]/20.0)
    b_h3[i]=1.0/(np.exp(3.0-0.1*V3[i])+1.0)  
    
    n3[i]=a_n3[i]/(a_n3[i]+b_n3[i])  
    m3[i]=a_m3[i]/(a_m3[i]+b_m3[i])  
    h3[i]=a_h3[i]/(a_h3[i]+b_h3[i])  
    
    gK3[i]=gKmax*np.power(n3[i+1],4)  
    gNa3[i]=(np.power(m3[i+1],3))*gNamax*h3[i+1]  
    
    INa3[i]=(V3[i]-ENa)*gNa3[i]  
    IK3[i]=(V3[i]-EK)*gK3[i]  
    IL3[i]=(V3[i]-EL)*gLmax  
    
    F3[i]=INa3[i] + IK3[i] + IL3[i] 
    F3_dot[i] = (F3[i]-F3[i-1])/dt 
    F3_dot_dot[i] = (F3[i]-2*F3[i-1]+F3[i-2])/np.power(dt,2) 
    
    m3[i+1]= dt*(a_m3[i]*(1.0 - m3[i]) - b_m3[i]*m3[i])+PNm[i] 
    h3[i+1]= dt*(a_h3[i]*(1.0 - h3[i]) - b_h3[i]*h3[i])+PNh[i] 
    n3[i+1]= dt*(a_n3[i]*(1.0 - n3[i]) - b_n3[i]*n3[i])+PNn[i] 
    
    Ifb[i]=alpha*(V3[i] - Vrest) 
     
    I3[i]=alpha*(V2[i] - Vrest)
    I2[i]=alpha*(V1[i] - Vrest)
     
    V1_dot[i]=V1_st_dot[i] -(1.0/TAU)*(V1[i]-V1_st[i]) 
    V2_dot[i]=V2_st_dot[i] -(1.0/TAU)*(V2[i]-V2_st[i]) 
    V3_dot[i]=Vst_dot[i] -(1.0/TAU)*(V3[i]-Vst[i]) 
     
    I3_dot[i]=alpha*(V2_dot[i]) 
    I2_dot[i]=alpha*(V1_dot[i]) 
     
    V3_dot_dot[i]=Vst_dot_dot[i] -(1.0/TAU)*(V3_dot[i]-Vst_dot[i]) 
    V2_dot_dot[i]=V2_st_dot_dot[i] -(1.0/TAU)*(V2_dot[i]-V2_st_dot[i]) 
 
    I3_st[i]= Vst_dot[i] - (1.0/TAU)*(V3[i]-Vst[i]) + (1.0/Cm)*(F3[i]) 
    I3_st_dot[i]= Vst_dot_dot[i] - (1.0/TAU)*(V3[i+1]-Vst_dot[i]) + (1.0/Cm)*(F3_dot[i])
    I3_st_dot_dot[i]= Vst_dot_dot_dot[i] - (1.0/TAU)*(V3_dot_dot[i]-Vst_dot_dot[i]) + (1.0/Cm)*(F1_dot_dot[i]) 
     
    V2_st[i]=I3_st[i]/alpha +Vrest
    V2_st_dot[i]=I3_st_dot[i]/alpha 
    V2_st_dot_dot[i]=I3_st_dot_dot[i]/alpha 
     
    I2_st[i]= Vst_dot[i] - (1.0/TAU)*(V2[i]-Vst[i]) + (1.0/Cm)*(F2[i]) 
    I2_st_dot[i]= Vst_dot_dot[i] - (1.0/TAU)*(V2[i+1]-Vst_dot[i]) + (1.0/Cm)*(F2_dot[i]) 
    I2_st_dot_dot[i]= Vst_dot_dot_dot[i] - (1.0/TAU)*(V2_dot_dot[i]-Vst_dot_dot[i]) + (1.0/Cm)*(F2_dot_dot[i]) 
     
    V1_st[i]=I2_st[i]/alpha +Vrest
    V1_st_dot[i]=I2_st_dot[i]/alpha 
     
    IC[i]= Vst_dot[i] - (1.0/TAU)*(V1[i]-V1_st[i]) + (1.0/Cm)*(F1[i]) 
    I1[i]=IC[i]+Ifb[i] 
     
    IT1[i]= I1[i] - F1[i]
    V1[i+1]=V1[i]+dt*IT1[i]/Cm + PNV[i] 
    
    IT2[i]= I2[i] - F2[i]
    V2[i+1]=V2[i]+dt*IT2[i]/Cm + PNV[i] 
    
    IT3[i]= I3[i] - F3[i]
    V3[i+1]=V3[i]+dt*IT3[i]/Cm + PNV[i] 
    
    delta[i]= abs(V3[i]-Vst[i])

### ACHIEVABILITY & ERROR ANALYSIS

nDELTA     = np.sum(delta)/(len(time)) ### mean absolute error
ndelta_bar = nDELTA*dt/(time[30]-time[0])
DELTA = np.linspace(0, nDELTA,length)
delta_bar=np.linspace(0,ndelta_bar,length)
#print(V1)
#quit()

figure()  

plt.subplot(2,2,1)
plt.plot(time,V1, label='V1') 
ylabel('V1(mV)') 
xlabel('time')
#ylim(-3.0,5.0)
xlim(0.0,30.0)
plt.grid()
#plt.hold(True) 

plt.subplot(2,2,2)
plt.plot(time,V2, label='V2') 
ylabel('V2(mV)') 
xlabel('time')
xlim(0.0,30.0)
#ylim(-8.0,12.0)
plt.grid()
#plt.hold(True) 

plt.subplot(2,2,3)
plt.plot(time,V3, label='V3') 
ylabel('V3(mV)') 
xlabel('time')
xlim(0.0,30.0)
#ylim(-8.0,12.0)
plt.grid()
#plt.hold(True) 

plt.subplot(2,2,4) 
plt.plot(time,Vst, label='Vst') 
ylabel('V* (mV)') 
xlabel('time')
xlim(0.0,30.0)
#ylim(-8.0,12.0)
plt.grid()

figure()  

plt.subplot(3,1,1)
plt.plot(time,I1, label='I1') 
ylabel('I1(mA)') 
xlabel('time')
ylim(-100.0,100.0)
xlim(0.0,30.0)
plt.grid()
#plt.hold(True) 

plt.subplot(3,1,2)
plt.plot(time,I2, label='I2') 
ylabel('I2(mA)') 
xlabel('time')
xlim(0.0,30.0)
ylim(-30.0,200.0)
plt.grid()
#plt.hold(True) 

plt.subplot(3,1,3) 
plt.plot(time,I3, label='I3') 
ylabel('I3 (mA)') 
xlabel('time')
xlim(0.0,30.0)
#ylim(0.0,60.0)
plt.grid()

figure()
plt.subplot(2,1,1)
plt.plot(time,delta, label='0.1')
ylabel('delta')
xlabel('time(ms)') 
#ylim(0.0,0.25)
xlim(0.0,30.0)
plt.grid()
#plt.hold(True)

subplot(2,1,2)
plt.plot(epsilon,delta_bar, label='delta bar')
ylabel('delta_bar')
xlabel('epsilon')
#ylim(0.0,15.0)
xlim(0.0,1.0)
plt.grid()
show()
