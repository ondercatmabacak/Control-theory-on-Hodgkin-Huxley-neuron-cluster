#!/usr/bin/env python3 
from __future__ import division
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import sys
## Functions for 2nd Neuron

### channel activity ###

#V2 = np.arange(-50,151) # mV
#figure()
#plot(V2, m2_inf(V2), V2, h2_inf(V2), V2, n2_inf(V2))
#legend(('m','h','n'))
#title('Steady state values of ion channel gating variables')
#ylabel('Magnitude')
#xlabel('Voltage (mV)')

## setup parameters and state variables

T     = 80   # ms
dt    = 0.01 # ms
time  = np.arange(0,T,dt)


## HH Parameters

Naext1=440.0
Naint1=50.0
Kext1=20.0
Kint1=400.0
Clext1=560.0
Clint1=150.0

PK1=1.0
PNa1=0.04
PCl1=0.45

Naext2=440.0
Naint2=50.0
Kext2=20.0
Kint2=400.0
Clext2=560.0
Clint2=150.0
PK2=1.0
PNa2=0.04
PCl2=0.45


Cm      = 1      # uF/cm2
gbar_Na = 120    # mS/cm2
gbar_K  = 36     # mS/cm2
gbar_l  = 0.3    # mS/cm2
E_Na    = 115    # mV
E_K     = -12    # mV
E_l     = 10.613 # mV
B=70             # mV
Gamma=0.1
A=1.0
a=5.0
f=1.0
w=3.0
b=7.0
c=3.0

V1_rest=58 * np.log10(((PK1 * Kext1) + (PNa1 * Naext1) + (PCl1 * Clint1)) / ((PK1 * Kint1) + (PNa1 * Naint1) + (PCl1 * Clext1)))
V2_rest=58 * np.log10(((PK2 * Kext2) + (PNa2 * Naext2)+(PCl2 * Clint2)) / ((PK2 * Kint2) + (PNa2 * Naint2) + (PCl2 * Clext2)))


Vm2             = np.zeros((len(time))) # mV
Vm2[0]          = V2_rest
G2              = np.zeros((len(time)))
I2              = np.zeros((len(time))) 
F2_dot          = np.zeros((len(time)))
G2_dot          = np.zeros((len(time)))
I2_dot          = np.zeros((len(time))) 
V_star          = np.zeros((len(time)))
V_star_dot      = np.zeros((len(time)))
V_star_dot_dot  = np.zeros((len(time)))
a_m2            = np.zeros((len(time)))
a_n2            = np.zeros((len(time)))
a_h2            = np.zeros((len(time)))
b_m2            = np.zeros((len(time)))
b_n2            = np.zeros((len(time)))
b_h2            = np.zeros((len(time)))
m2              = np.zeros((len(time)))
n2              = np.zeros((len(time)))
h2              = np.zeros((len(time)))
g_Na2           = np.zeros((len(time)))
g_K2            = np.zeros((len(time)))

#V_star=B
V_star= a*np.cos(w*time+f)
#V_star= a*np.exp(-(np.power((time-b),2))/(np.power(c,2)))

#print(V_star)
#sys.exit()

#V_star_dot = 0
V_star_dot= -a*w*np.sin(w*time+f)
#V_star_dot= -(2*a)*(time-b)/(np.power(c,2))*np.exp(-(np.power((time-b),2))/(np.power(c,2)))

a_m2[0]  = 0.1*(-Vm2[0] + 25)/(np.exp((-Vm2[0] + 25)/10) - 1)
a_n2[0]  = 0.01*(-Vm2[0] + 10)/(np.exp((-Vm2[0] + 10)/10) - 1)
a_h2[0]  = 0.07*np.exp(-Vm2[0]/20)

b_m2[0]  = 4*np.exp(-Vm2[0]/18)
b_n2[0]  = 0.125*np.exp(-Vm2[0]/80)
b_h2[0]  = 1/(np.exp((-Vm2[0] + 30)/10) + 1)

m2[0] =  a_m2[0] / (a_m2[0] + b_m2[0])
n2[0] = (a_n2[0] * (b_h2[0] + a_h2[0]) - (a_n2[0] * b_h2[0])) / ((a_n2[0] + b_n2[0]) * ((a_h2[0] + b_h2[0]) - (a_n2[0] *b_h2[0])))
h2[0] = (b_h2[0] * (a_n2[0] + b_n2[0]) - (a_n2[0] * b_h2[0])) / ((a_n2[0] + b_n2[0]) * (a_h2[0] + b_h2[0]) - (a_n2[0] * b_h2[0]))


## Simulate Model

for i in range(1,len(time)-1):
  V_star[0]   = 0
  #V_star[i-1] = int(i-1)
  a_m2[i-1]   = 0.1*(-Vm2[i-1] + 25)/(np.exp((-Vm2[i-1] + 25)/10) - 1)
  a_n2[i-1]   = 0.01*(-Vm2[i-1] + 10)/(np.exp((-Vm2[i-1] + 10)/10) - 1)
  a_h2[i-1]   = 0.07*np.exp(-Vm2[i-1]/20)

  b_m2[i-1]   = 4*np.exp(-Vm2[i-1]/18)
  b_n2[i-1]   = 0.125*np.exp(-Vm2[i-1]/80)
  b_h2[i-1]   = 1/(np.exp((-Vm2[i-1] + 30)/10) + 1)

  m2[i-1]     = a_m2[i-1] / (a_m2[i-1] + b_m2[i-1])
  n2[i-1]     = (a_n2[i-1] * (b_h2[i-1] + a_h2[i-1]) - (a_n2[i-1] * b_h2[i-1])) / ((a_n2[i-1] + b_n2[i-1]) * ((a_h2[i-1] + b_h2[i-1]) - (a_n2[i-1] * b_h2[i-1])))
  h2[i-1]     = (b_h2[i-1] * (a_n2[i-1] + b_n2[i-1]) - (a_n2[i-1] * b_h2[i-1])) / ((a_n2[i-1] + b_n2[i-1]) * ((a_h2[i-1] + b_h2[i-1]) - (a_n2[i-1] * b_h2[i-1])))

  g_Na2[i-1]  = gbar_Na*(np.power(m2[i-1],3))*h2[i-1]
  g_K2[i-1]   = gbar_K*(np.power(n2[i-1],4))
  g_l         = gbar_l
  
  I2[i-1]     = -Gamma*(1/Cm)*(Vm2[i-1] - V_star[i-1]) 
  #G2[i-1]    = np.power(((1/2)*(Vm2[i-1] - V_star[i-1])),2)
  F2_dot[i-1] = Vm2[i]*(g_Na2[i-1] + g_K2[i-1] + g_l)

  #G2_dot[i-1]= (Vm2[i-1] - V_star[i-1])*(Vm2[i] - V_star_dot[i])
  I2_dot[i-1] = -(Gamma/Cm)*(Vm2[i] - V_star_dot[i-1])


  g_Na2[i]   = gbar_Na*(np.power(m2[i],3))*h2[i]
  g_K2[i]    = gbar_K*(np.power(n2[i],4))
  
  n2[i]      = n2[i-1]+dt*(a_n2[i-1]*(1-n2[i-1])-b_n2[i-1]*n2[i-1])
  m2[i]      = m2[i-1]+dt*(a_m2[i-1]*(1-m2[i-1])-b_m2[i-1]*m2[i-1])
  h2[i]      = h2[i-1]+dt*(a_h2[i-1]*(1-h2[i-1])-b_h2[i-1]*h2[i-1])

  I2[i]      = -Gamma*(1/Cm)*(Vm2[i] - V_star[i-1]) 
  Vm2[i]     = Vm2[i-1] +I2[i] - (g_Na2[i] * (Vm2[i-1] - E_Na) + g_K2[i] * (Vm2[i-1] - E_K) + g_l * (Vm2[i-1] - E_l)) / Cm *dt 

  G2[i]      = np.power(((1/2)*(Vm2[i-1] - V_star[i-1])),2)
  F2_dot[i]  = Vm2[i]*(g_Na2[i] + g_K2[i] + g_l)

  G2_dot[i]  = (Vm2[i-1] - V_star[i-1])*(Vm2[i] - V_star_dot[i])
  I2_dot[i]  = -(Gamma/Cm)*(Vm2[i] - V_star_dot[i])

## Functions for 1st Neuron


### channel activity ###

V1 = np.arange(-50,151) # mV
#figure()
#plot(V1, m1_inf(V1), V1, h1_inf(V1), V1, n1_inf(V1))
#legend(('m','h','n'))
#title('Steady state values of ion channel gating variables')
#ylabel('Magnitude')
#xlabel('Voltage (mV)')

## setup parameters and state variables

T     = 100    # ms
dt    = 0.1 # ms
time  = np.arange(0,T+dt,dt)

## HH Parameters

Vm1          = np.zeros((len(time))) # mV
Vm1[0]       = V1_rest
I1           = np.zeros((len(time))) 
V_out        = np.zeros((len(time)))
V_out_dot    = np.zeros((len(time)))
F1           = np.zeros((len(time)))
G1           = np.zeros((len(time)))
G1_dot       = np.zeros((len(time)))
G1_dot_I1dot = np.zeros((len(time)))
a_m1         = np.zeros((len(time)))
a_n1         = np.zeros((len(time)))
a_h1         = np.zeros((len(time)))
b_m1         = np.zeros((len(time)))
b_n1         = np.zeros((len(time)))
b_h1         = np.zeros((len(time)))
m1           = np.zeros((len(time)))
n1           = np.zeros((len(time)))
h1           = np.zeros((len(time)))
g_Na1        = np.zeros((len(time)))
g_K1         = np.zeros((len(time)))

a_m1[0]  = 0.1*(-Vm1[0] + 25)/(np.exp((-Vm1[0] + 25) / 10) - 1)
a_n1[0]  = 0.01*(-Vm1[0] + 10)/(np.exp((-Vm1[0] + 10)/10) - 1)
a_h1[0]  = 0.07*np.exp(-Vm1[0]/20)

b_m1[0]  = 4*np.exp(-Vm1[0]/18)
b_n1[0]  = 0.125*np.exp(-Vm1[0]/80)
b_h1[0]  = 1/(np.exp((-Vm1[0] + 30)/10) + 1)

m1[0]    = a_m1[0] / (a_m1[0] + b_m1[0])

n1[0]    = (a_n1[0] * (b_h1[0] + a_h1[0]) - (a_n1[0] * b_h1[0])) / ((a_n1[0] + b_n1[0]) * ((a_h1[0] + b_h1[0]) - (a_n1[0] *b_h1[0])))

h1[0]    = (b_h1[0] * (a_n1[0] + b_n1[0]) - (a_n1[0] * b_h1[0])) / ((a_n1[0] + b_n1[0]) * (a_h1[0] + b_h1[0]) - (a_n1[0] * b_h1[0]))


## Simulate Model

for i in range(1,len(time)-1):
  
  V_star[0]  = 0
  V_star[i-1]= int(i-1)
  a_m1[i-1]  = 0.1*(-Vm1[i-1] + 25)/(np.exp((-Vm1[i-1] + 25) / 10) - 1)
  a_n1[i-1]  = 0.01*(-Vm1[i-1] + 10)/(np.exp((-Vm1[i-1] + 10)/10) - 1)
  a_h1[i-1]  = 0.07*np.exp(-Vm1[i-1]/20)

  b_m1[i-1]  = 4*np.exp(-Vm1[i-1]/18)
  b_n1[i-1]  = 0.125*np.exp(-Vm1[i-1]/80)
  b_h1[i-1]  = 1/(np.exp((-Vm1[i-1] + 30)/10) + 1)

  m1[i-1]    = a_m1[i-1] / (a_m1[i-1] + b_m1[i-1])
  n1[i-1]    = (a_n1[i-1] * (b_h1[i-1] + a_h1[i-1]) - (a_n1[i-1] * b_h1[i-1])) / ((a_n1[i-1] + b_n1[i-1]) * ((a_h1[i-1] + b_h1[i-1]) - (a_n1[i-1] * b_h1[i-1])))
  h1[i-1]    = (b_h1[i-1] * (a_n1[i-1] + b_n1[i-1]) - (a_n1[i-1] * b_h1[i-1])) / ((a_n1[i-1] + b_n1[i-1]) * (a_h1[i-1] + b_h1[i-1]) - (a_n1[i-1] * b_h1[i-1]))

  g_Na1[i-1] = gbar_Na*(np.power(m1[i-1],3))*h1[i-1]
  g_K1[i-1]  = gbar_K*(np.power(n1[i-1],4))
  g_l        = gbar_l

  I1[i-1]    = -Gamma*(1/Cm)*(Vm1[i-1] + (Gamma / A)*(1/Cm)*(Vm2[i-1] - V_star[i-1]) )

  V_out[i-1] = I2[i-1]/A
  
  Vm1[i]     = Vm1[i-1] + ((-Gamma*(1/Cm)*(Vm1[i-1] + (Gamma / A)*(1/Cm)*(Vm2[i-1] - V_star[i-1]) )) - (g_Na1[i-1] * (Vm1[i-1] - E_Na) + g_K1[i-1] * (Vm1[i-1] - E_K) + g_l * (Vm1[i-1] - E_l))) / Cm * dt 

  n1[i]      = n1[i-1]+dt*(a_n1[i-1]*(1-n1[i-1])-b_n1[i-1]*n1[i-1])
  m1[i]      = m1[i-1]+dt*(a_m1[i-1]*(1-m1[i-1])-b_m1[i-1]*m1[i-1])
  h1[i]      = h1[i-1]+dt*(a_h1[i-1]*(1-h1[i-1])-b_h1[i-1]*h1[i-1])
  
  g_Na1[i]   = gbar_Na*(np.power(m1[i],3))*h1[i]
  g_K1[i]    = gbar_K*(np.power(n1[i],4))
  
  I1[i]      = -Gamma*(1/Cm)*(Vm1[i-1] + (Gamma / A)*(1/Cm)*(Vm2[i-1] - V_star[i-1]) )

#print(V_star)
#sys.exit()

  
## plot membrane potential trace
figure()
# subplot2grid((4,1), (0, 1))
plt.plot(time,Vm1)
ylabel('Membrane Potential (mV)')
ylim(-15.0,110.0)

# subplot2grid((4,1), (2, 1))
plot(time,I1)
xlabel("time (ms)")
ylabel('i ($\\mu$ A/cm^2)')
ylim(-15,10.0)
yticks(np.linspace(0, 10, 3), ('0', '5', '10') )

show()
