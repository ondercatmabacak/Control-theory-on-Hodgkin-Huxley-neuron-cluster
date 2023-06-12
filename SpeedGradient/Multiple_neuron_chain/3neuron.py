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
gNamax=120
gKmax=36
gLmax=0.3
EK = -12
ENa=115
EL=10.6

length = len(pink_noise.z) 
epsilon=np.linspace(0,1.01,length) 
PN = 2*epsilon*(pink_noise.z-1/2) 
PNV= PN*1.0 
PNm= PN*1.0 
PNn= PN*1.0 
PNh= PN*1.0 

# set the initial states FOR HH1 and HH2 
# For HH1
V1          = np.zeros((len(time))) # mV
V1_st       = np.zeros((len(time))) # mV
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
I1          = np.zeros((len(time)))
IC1         = np.zeros((len(time)))

V2          = np.zeros((len(time))) # mV
V2_st       = np.zeros((len(time))) # mV
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
I2          = np.zeros((len(time)))
I2_st       = np.zeros((len(time)))
IC2         = np.zeros((len(time)))

V3          = np.zeros((len(time))) # mV
V3[1]       = Vrest2
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
I3          = np.zeros((len(time)))
I3_st       = np.zeros((len(time)))
IC3         = np.zeros((len(time)))
delta       = np.zeros((len(time)))

alpha=1.0
r=3.0
#Vst=70
Vst= 1.0*np.cos(2.0*time + 3.0)
#Vst= 7.0*np.exp(-(np.power((time-5.0),2))/(np.power(4.0,2)))

#figure()
#plot(time,Vst,'r','LineWidth',2)
#ylabel('Vst')

for i in range(1,len(time)):
    I3_st[i]=(-r*(V3[i]-Vst[i]))/Cm 
    V2_st[i]=I3_st[i]/alpha +Vrest2
    I2_st[i]=-(r/Cm)*(V2[i] - V2_st[i])
    V1_st[i]=I2_st[i]/alpha +Vrest1

    I1[i]=-(r/Cm)*(V1[i] - V1_st[i]) 

    a_n1[i]=0.01*((10-V1[i]) / (np.exp((10-V1[i])/10)-1) ) 
    b_n1[i]=0.125*np.exp(-V1[i]/80) 
    a_m1[i]=0.1*((25-V1[i]) /(np.exp((25-V1[i])/10)-1) ) 
    b_m1[i]=4*np.exp(-V1[i]/18) 
    a_h1[i]=0.07*np.exp(-V1[i]/20) 
    b_h1[i]=1/(np.exp((30-V1[i])/10)+1) 

    n1[i]=a_n1[i]/(a_n1[i]+b_n1[i]) 
    m1[i]=a_m1[i]/(a_m1[i]+b_m1[i]) 
    h1[i]=a_h1[i]/(a_h1[i]+b_h1[i]) 

    gK1[i]=gKmax*np.power(n1[i+1],4) 
    gNa1[i]=(np.power(m1[i+1],3))*gNamax*h1[i+1] 
    INa1[i]=(V1[i]-ENa)*gNa1[i] 
    IK1[i]=(V1[i]-EK)*gK1[i] 
    IL1[i]=(V1[i]-EL)*gLmax 
    
    IC1[i]=I1[i]-IK1[i]-INa1[i]-IL1[i] 

    V1[i+1]=V1[i]+dt*IC1[i]/Cm 
    m1[i+1]= dt*(a_m1[i]*(1 - m1[i]) - b_m1[i]*m1[i])
    h1[i+1]= dt*(a_h1[i]*(1 - h1[i]) - b_h1[i]*h1[i])
    n1[i+1]= dt*(a_n1[i]*(1 - n1[i]) - b_n1[i]*n1[i])

    I2[i]=alpha*(V1[i]-Vrest1)

    a_n2[i]=.01*((10-V2[i]) / (np.exp((10-V2[i])/10)-1) ) 
    b_n2[i]=.125*np.exp(-V2[i]/80) 
    a_m2[i]=.1*((25-V2[i]) /(np.exp((25-V2[i])/10)-1) ) 
    b_m2[i]=4*np.exp(-V2[i]/18) 
    a_h2[i]=.07*np.exp(-V2[i]/20) 
    b_h2[i]=1/(np.exp((30-V2[i])/10)+1) 
    
    n2[i]=a_n2[i]/(a_n2[i]+b_n2[i]) 
    m2[i]=a_m2[i]/(a_m2[i]+b_m2[i]) 
    h2[i]=a_h2[i]/(a_h2[i]+b_h2[i]) 

    gK2[i]=gKmax*np.power(n2[i+1],4) 
    gNa2[i]=np.power(m2[i+1],3)*gNamax*h2[i+1] 
    INa2[i]=(V2[i]-ENa)*gNa2[i] 
    IK2[i]=(V2[i]-EK)*gK2[i] 
    IL2[i]=(V2[i]-EL)*gLmax     
    
    IC2[i]=I2[i]-IK2[i]-INa2[i]-IL2[i] 
    
    V2[i+1]=V2[i]+dt*IC2[i]/Cm 
    m2[i+1]= dt*(a_m2[i]*(1 - m2[i]) - b_m2[i]*m2[i])
    h2[i+1]= dt*(a_h2[i]*(1 - h2[i]) - b_h2[i]*h2[i])
    n2[i+1]= dt*(a_n2[i]*(1 - n2[i]) - b_n2[i]*n2[i])

    I3[i]=alpha*(V2[i]-Vrest2)

    a_n3[i]=.01*((10-V3[i]) / (np.exp((10-V3[i])/10)-1) ) 
    b_n3[i]=.125*np.exp(-V3[i]/80) 
    a_m3[i]=.1*((25-V3[i]) /(np.exp((25-V3[i])/10)-1) ) 
    b_m3[i]=4*np.exp(-V3[i]/18) 
    a_h3[i]=.07*np.exp(-V3[i]/20) 
    b_h3[i]=1/(np.exp((30-V3[i])/10)+1) 
    
    n3[i]=a_n3[i]/(a_n3[i]+b_n3[i]) 
    m3[i]=a_m3[i]/(a_m3[i]+b_m3[i]) 
    h3[i]=a_h3[i]/(a_h3[i]+b_h3[i]) 
    
    gK3[i]=gKmax*np.power(n3[i+1],4) 
    gNa3[i]=np.power(m3[i+1],3)*gNamax*h3[i+1] 
    INa3[i]=(V3[i]-ENa)*gNa3[i] 
    IK3[i]=(V3[i]-EK)*gK3[i] 
    IL3[i]=(V3[i]-EL)*gLmax 

    IC3[i]=I3[i]-IK3[i]-INa3[i]-IL3[i] 

    V3[i+1]=V3[i]+dt*IC3[i]/Cm 
    m3[i+1]= dt*(a_m3[i]*(1 - m3[i]) - b_m3[i]*m3[i])
    h3[i+1]= dt*(a_h3[i]*(1 - h3[i]) - b_h3[i]*h3[i])
    n3[i+1]= dt*(a_n3[i]*(1 - n3[i]) - b_n3[i]*n3[i])

    delta[i]= abs(V3[i]-Vst[i])
print(I1)
quit()
### ACHIEVABILITY & ERROR ANALYSIS

nDELTA     = np.sum(delta)
ndelta_bar = nDELTA*dt/(time[30]-time[0])
DELTA = np.linspace(0, nDELTA,length)
delta_bar=np.linspace(0,ndelta_bar,length)

# applying current, 
'''
figure ()
plt.subplot(4,2,1)
plt.plot(time,a_h1, label='a_h1')
ylabel('ah1')
xlabel('time(ms)') 
#ylim(0.0,2.0)
xlim (0,30.0)
plt.hold(True)

plt.subplot(4,2,2)
plt.plot(time,a_m1, label='V*')
ylabel('am1')
xlabel('time(ms)') 
#ylim(-45.0,-35.0)
xlim (0,30.0)
plt.hold(True)

plt.subplot(4,2,3)
plt.plot(time,a_n1, label='V*')
ylabel('an1')
xlabel('time(ms)') 
#ylim(0.0,2.0)
xlim (0,30.0)
plt.hold(True)

plt.subplot(4,2,4)
plt.plot(time,b_h1, label='V*')
ylabel('bh1')
xlabel('time(ms)') 
#ylim(-45.0,-35.0)
xlim (0,30.0)
plt.hold(True)

plt.subplot(4,2,5)
plt.plot(time,b_m1, label='V*')
ylabel('bm1')
xlabel('time(ms)') 
#ylim(0.0,2.0)
xlim (0,30.0)
plt.hold(True)

plt.subplot(4,2,6)
plt.plot(time,b_n1, label='V*')
ylabel('bn1')
xlabel('time(ms)') 
#ylim(0.0,2.0)
xlim (0,30.0)
show()

quit()

'''
figure()  

plt.subplot(4,2,1)
plt.plot(time,np.log10(I1), label='Iapp1') 
ylabel('I1 (mA)') 
xlabel('time (ms)') 
#ylim(-100.0,100.0)
plt.hold(True)

plt.subplot(4,2,2)
plt.plot(time,np.log10(I2_st), label='I2_st') 
ylabel('I2 (mA)') 
xlabel('time (ms)') 
#ylim(-800.0,0.0)
plt.hold(True)

plt.subplot(4,2,3)
plt.plot(time,np.log10(I3_st), label='I2_st') 
ylabel('I3 (mA)') 
xlabel('time (ms)') 
#ylim(-10.0,10.0)
plt.hold(True)

plt.subplot(4,2,4)
plt.plot(time,np.log10(V1), label='V1') 
ylabel('V1 (mV)')
xlabel('time(ms)') 
#ylim(-800.0,0.0)
plt.hold(True)

plt.subplot(4,2,5)
plt.plot(time,np.log10(V2), label='V2') 
ylabel('V2 (mV)')
xlabel('time(ms)')
#ylim(-50.0,50.0)
plt.hold(True)

plt.subplot(4,2,6)
plt.plot(time,np.log10(V3), label='V2') 
ylabel('V3 (mV)')
xlabel('time(ms)')
#ylim(-45.0,-35.0)
plt.hold(True)

plt.subplot(4,2,7)
plt.plot(time,np.log10(Vst), label='V*')
ylabel('V_star (mV)')
xlabel('time(ms)') 
#ylim(-45.0,-35.0)
xlim (0,30.0)
plt.hold(True)

figure()
plt.subplot(2,1,1)
plt.plot(time,np.log10(delta), label='0.1')
ylabel('delta')
xlabel('time(ms)') 
#ylim(0.0,0.2)
xlim(0.0,30.0)
plt.grid()
plt.hold(True)

subplot(2,1,2)
plt.plot(epsilon,np.log10(delta_bar), label='delta bar')
ylabel('delta_bar')
xlabel('epsilon')
#ylim(0.0,15.0)
xlim(0.0,1.0)
plt.grid()
show()

