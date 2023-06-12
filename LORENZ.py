#!/usr/bin/env python3 
from __future__ import division
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

## Functions for 2nd Neuron

## HH Parameters
a=1.0
b=1.0
c=1.0
A=1.0
B=1.0
T=1.0
f=1.0
w=1.0
#V1=-70  
#V2=-70

C_m      = 1      # uF/cm2
gbar_Na = 120.0    # mS/cm2
gbar_K  = 36.0     # mS/cm2
gbar_l  = 0.3    # mS/cm2
E_Na    = 115    # mV
E_K     = -12    # mV
E_l     = 10.613 # mV

alpha_m2 = 0.1*(-V2 + 25)/(np.exp((-V2 + 25)/10) - 1)
alpha_n2 = 0.01*(-V2 + 10)/(np.exp((-V2 + 10)/10) - 1)
alpha_h2 = 0.07*np.exp(-V2/20)

beta_m2  = 4*np.exp(-V2/18)
beta_n2  = 0.125*np.exp(-V2/80)
beta_h2  = 1/(np.exp((-V2 + 30)/10) + 1)

m2       = alpha_m2/(alpha_m2+beta_m2)
n2       = (alpha_n2*(beta_h2+alpha_h2) - (alpha_n2 * beta_h2))/((alpha_n2+beta_n2)*((alpha_h2 + beta_h2)-(alpha_n2 * beta_h2)))
h2       = beta_h2*(alpha_n2+beta_n2)-(alpha_n2*beta_h2)/((alpha_n2+beta_n2)*(alpha_h2+beta_h2)-(alpha_n2*beta_h2))

g_Na2 = gbar_Na*(np.power(m2,3))*h2
g_K2  = gbar_K*(np.power(n2,4))
g_l  = gbar_l

m2_dot= (alpha_m2*(1 - m2) - beta_m2 * m2)
h2_dot= (alpha_h2*(1 - h2) - beta_h2 * h2)
n2_dot= (alpha_n2*(1 - n2) - beta_n2 * n2)

V_star = B
#V_star= a*np.cos(wt+f)
#V_star= a*np.exp(-(np.power((t-b),2))/(np.power(c,2)))

V_star_dot = 0
#V_star_dot= -a*w*np.sin(wt+f)
#V_star_dot= -(2*a)*(t-b)/(np.power(c,2))*np.exp(-(np.power((t-b),2))/(np.power(c,2)))

V_star_dot_dot = 0
#V_star_dot_dot= -a*np.power(w,2)*np.cos(wt+f)
#V_star_dot_dot= (-((2*a)/np.power(c,2) + np.power((2*a)*(t-b)/np.power(c,2),2)*np.exp(-(np.power((t-b),2))/np.power(c,2))))


F2=g_Na2*(V2 - E_Na) + g_K2*(V2 - E_K) + g_l*(V2 - E_l)

G2     = V2 - V_star
I2     = C_m*(V_star_dot - 1/T*(G2)-F2)
V2_dot = C_m*(V_star_dot -1/T*(G2))
F2_dot = V2_dot*(g_Na2 + g_K2 + g_l)

G2_dot = V2_dot - V_star_dot
I2_dot = C_m*(V_star_dot_dot - F2_dot - 1/T*G2_dot)

## Functions for 1st Neuron
# K channel


alpha_m1 = 0.1*(-V1 + 25)/(np.exp((-V1 + 25)/10) - 1)
alpha_n1 = 0.01*(-V1 + 10)/(np.exp((-V1 + 10)/10) - 1)
alpha_h1 = 0.07*np.exp(-V1/20)


beta_m1  = 4*np.exp(-V1/18)
beta_n1  = 0.125*np.exp(-V1/80)
beta_h1  = 1/(np.exp((-V1 + 30)/10) + 1)

m1       = alpha_m1/(alpha_m1+beta_m1)
n1       = (alpha_n1)*((beta_h1+alpha_h1) - alpha_n1 * beta_h1)/((alpha_n1+beta_n1)*((alpha_h1 + beta_h1)-(alpha_n1 * beta_h1)))
h1       = (beta_h1*(alpha_n1+beta_n1) - (alpha_n1*beta_h1))/((alpha_n1+beta_n1)*(alpha_h1+beta_h1) - (alpha_n1*beta_h1))

g_Na1 = gbar_Na*(np.power(m1,3))*h1
g_K1  = gbar_K*(np.power(n1,4))


F1=g_Na1*(V1 - E_Na) + g_K1*(V1 - E_K) + g_l*(V1 - E_l)
V_out = I2/A
V_out_dot = I2_dot / A

V1_dot = C_m*(V_star_dot -1/T*(V1 - V_out)) 
G1     = V1 - V_out
I1     = C_m*(V_out_dot -F1 - 1/T*(V1 - V_out))
G1_dot = V1_dot - V_out_dot

## plot membrane potential trace
figure()
plot(I1,V1)
x = [10, 20, 30, 40, 50, 60, 70, 80]
labels = [' ', ' ', ' ', ' ', ' ', ' ', ' ',' ']
plt.xticks(x, labels)
ylabel('Membrane Potential (mV)')
ylim(-15.0,110.0)
plt.show()
