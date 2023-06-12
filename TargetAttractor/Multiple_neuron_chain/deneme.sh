#!/bin/bash
start=1
finish=3
output=multiple.py
printf "#!/usr/bin/env python3 \n">>$output
printf "from __future__ import division \n">>$output
printf "import numpy as np \n">>$output
printf "import matplotlib.pyplot as plt \n">>$output
printf "from matplotlib import * \n">>$output
printf "import scipy as sp \n">>$output
printf "import random \n">>$output
printf "import scipy.integrate as integrate \n">>$output
printf "from scipy.integrate import quad \n">>$output
printf "import scipy.special as special \n">>$output
printf "import sympy as sym \n">>$output
printf "import pink_noise \n">>$output
printf "import sys \n">>$output
printf "from pylab import * \n">>$output
printf " \n">>$output
printf "#computing of resting potential with Nerst \n">>$output
printf "Naext=440 \n">>$output
printf "Naint=50 \n">>$output
printf "Kext=20 \n">>$output
printf "Kint=400 \n">>$output
printf "Clext=560 \n">>$output
printf "Clint=150 \n">>$output
printf "PK=1 \n">>$output
printf "PNa=0.04 \n">>$output
printf "PCl=0.45 \n">>$output
printf " \n">>$output
printf "Vrest=58*np.log10(((PK*Kext)+(PNa*Naext)+(PCl*Clint))/((PK*Kint)+(PNa*Naint)+(PCl*Clext))) \n">>$output
printf " \n">>$output
printf "# simulation time  \n">>$output
printf "T     = 30   # ms \n">>$output
printf "dt    = 0.01 # ms \n">>$output
printf "time  = np.arange(0,T,dt) \n">>$output
printf " \n">>$output
printf "# constant parameters FOR HH1 and HH2  \n">>$output
printf "Cm    =1 \n">>$output
printf "gNamax=120 \n">>$output
printf "gKmax =36 \n">>$output
printf "gLmax =0.3 \n">>$output
printf "EK    =-12 \n">>$output
printf "ENa   =115 \n">>$output
printf "EL    =10.6 \n">>$output
printf " \n">>$output
printf "# initial states \n">>$output
printf " \n">>$output
printf "''' \n">>$output
printf "erey=np.arange(0,1,0.1) \n">>$output
printf " \n">>$output
printf "delta_bar   = np.zeros((len(erey))) \n">>$output
printf "DELTA       = np.zeros((len(erey))) \n">>$output
printf "epsilon     = np.zeros((len(erey))) \n">>$output
printf "delta       = np.zeros((len(erey),len(time))) \n">>$output
printf "''' \n">>$output
for ((j = $start; j <= $finish; j++))
do
printf "V$j          = np.zeros((len(time))) # mV \n">>$output
printf "V$((j))_dot   = np.zeros((len(time))) # mV \n">>$output
printf "V$((j))_dot_dot   = np.zeros((len(time))) # mV \n">>$output
printf "V$((j))_st       = np.zeros((len(time))) # mV \n">>$output
printf "V$((j))_st_dot   = np.zeros((len(time))) # mV \n">>$output
printf "V$((j))_st_dot_dot   = np.zeros((len(time))) # mV \n">>$output
printf "V$j[1]       = Vrest \n">>$output
printf "a_m$j        = np.zeros((len(time))) \n">>$output
printf "a_n$j        = np.zeros((len(time))) \n">>$output
printf "a_h$j        = np.zeros((len(time))) \n">>$output
printf "b_m$j        = np.zeros((len(time))) \n">>$output
printf "b_n$j        = np.zeros((len(time))) \n">>$output
printf "b_h$j        = np.zeros((len(time))) \n">>$output
printf "m$j          = np.zeros((len(time))) \n">>$output
printf "n$j          = np.zeros((len(time))) \n">>$output
printf "h$j          = np.zeros((len(time))) \n">>$output
printf "gNa$j        = np.zeros((len(time))) \n">>$output
printf "gK$j         = np.zeros((len(time))) \n">>$output
printf "INa$j        = np.zeros((len(time))) \n">>$output
printf "IK$j         = np.zeros((len(time))) \n">>$output
printf "IL$j         = np.zeros((len(time))) \n">>$output
printf "F$j          = np.zeros((len(time))) \n">>$output
printf "I$j          = np.zeros((len(time))) \n">>$output
printf "F$((j))_dot      = np.zeros((len(time))) \n">>$output
printf "F$((j))_dot_dot  = np.zeros((len(time))) \n">>$output
printf "F$((j))[0]       = 0.0 \n">>$output
printf "F$((j))[-1]      = 0.0 \n">>$output
printf "I$((j))_dot      = np.zeros((len(time))) \n">>$output
printf "I$((j))_st       = np.zeros((len(time))) \n">>$output
printf "I$((j))_st_dot   = np.zeros((len(time))) \n">>$output
printf "I$((j))_st_dot_dot   = np.zeros((len(time))) \n">>$output
printf "IC          = np.zeros((len(time))) \n">>$output
printf "IT$j         = np.zeros((len(time))) \n">>$output
done
printf "PN          = np.zeros((len(time))) \n">>$output
printf "Ifb         = np.zeros((len(time))) \n">>$output
printf "PNV         = np.zeros((len(time))) \n">>$output
printf "PNn         = np.zeros((len(time))) \n">>$output
printf "PNm         = np.zeros((len(time))) \n">>$output
printf "PNh         = np.zeros((len(time))) \n">>$output
printf "delta       = np.zeros((len(time))) \n">>$output
printf " \n">>$output
printf "epsilon=0.01 \n">>$output
printf "length = len(pink_noise.z) \n">>$output
printf "epsilon=np.linspace(0,1.01,length) \n">>$output
printf "PN = 2*epsilon*(pink_noise.z-1/2) \n">>$output
printf "PNV= PN*1.0 \n">>$output
printf "PNm= PN*1.0 \n">>$output
printf "PNn= PN*1.0 \n">>$output
printf "PNh= PN*1.0 \n">>$output
printf "    \n">>$output
printf "A=10.0 \n">>$output
printf "TAU=0.05 \n">>$output
printf "alpha=1.0 \n">>$output
printf "#Vst=70 \n">>$output
printf "a=2.0 \n">>$output
printf "b=3.0 \n">>$output
printf "c=4.0 \n">>$output
printf "d=5.0 \n">>$output
printf "Vst=a*np.cos(b*time + c) + b*np.cos(c*time + d) + c*np.cos(d*time + a) + d*np.cos(a*time + b)  \n">>$output
printf "Vst_dot     = -a*b*np.sin(b*time + c) - b*c*np.sin(c*time + d) - c*d*np.sin(d*time + a) + d*a*np.sin(a*time + b) \n">>$output
printf " \n">>$output
printf "Vst_dot_dot = -a*np.power(b,2)*np.cos(b*time + c) - b*np.power(c,2)*np.cos(c*time + d) - c*np.power(d,2)*np.cos(d*time + a) + d*np.power(a,2)*np.cos(a*time + b) \n">>$output
printf " \n">>$output
printf "Vst_dot_dot_dot = a*np.power(b,3)*np.sin(b*time + c) + b*np.power(c,3)*np.sin(c*time + d) + c*np.power(d,3)*np.sin(d*time + a) + d*np.power(a,3)*np.sin(a*time + b) \n">>$output
printf " \n">>$output
printf "#Vst= a*np.exp(-(np.power((time-b),c))/(np.power(d,2))) \n">>$output
printf "for i in range(1,len(time)-1): \n">>$output
printf " \n">>$output
for ((j = $start; j <= $finish; j++))
do
printf "    a_n$j[i]=0.01*((10-V$j[i]) / (np.exp((10-V$j[i])/10)-1) )  \n">>$output
printf "    b_n$j[i]=0.125*np.exp(-V$j[i]/80)  \n">>$output
printf "    a_m$j[i]=0.1*((25-V$j[i]) /(np.exp((25-V$j[i])/10)-1) )  \n">>$output
printf "    b_m$j[i]=4*np.exp(-V$j[i]/18)  \n">>$output
printf "    a_h$j[i]=0.07*np.exp(-V$j[i]/20)  \n">>$output
printf "    b_h$j[i]=1/(np.exp((30-V$j[i])/10)+1)  \n">>$output
printf "    \n">>$output
printf "    n$j[i]=a_n$j[i]/(a_n$j[i]+b_n$j[i])  \n">>$output
printf "    m$j[i]=a_m$j[i]/(a_m$j[i]+b_m$j[i])  \n">>$output
printf "    h$j[i]=a_h$j[i]/(a_h$j[i]+b_h$j[i])  \n">>$output
printf "    \n">>$output
printf "    gK$j[i]=gKmax*np.power(n$j[i+1],4)  \n">>$output
printf "    gNa$j[i]=(np.power(m$j[i+1],3))*gNamax*h$j[i+1]  \n">>$output
printf "    \n">>$output
printf "    INa$j[i]=(V$j[i]-ENa)*gNa$j[i]  \n">>$output
printf "    IK$j[i]=(V$j[i]-EK)*gK$j[i]  \n">>$output
printf "    IL$j[i]=(V$j[i]-EL)*gLmax  \n">>$output
printf "    \n">>$output
printf "    F$j[i]=INa$j[i] + IK$j[i] + IL$j[i] \n">>$output
printf "    F$((j))_dot[i] = (F$((j))[i]-F$((j))[i-1])/dt \n">>$output 
printf "    F$((j))_dot_dot[i] = (F$((j))[i]-2*F$((j))[i-1]+F$((j))[i-2])/np.power(dt,2) \n">>$output
printf "    \n">>$output
printf "    m$j[i+1]= dt*(a_m$j[i]*(1 - m$j[i]) - b_m$j[i]*m$j[i])+PNm[i] \n">>$output
printf "    h$j[i+1]= dt*(a_h$j[i]*(1 - h$j[i]) - b_h$j[i]*h$j[i])+PNh[i] \n">>$output
printf "    n$j[i+1]= dt*(a_n$j[i]*(1 - n$j[i]) - b_n$j[i]*n$j[i])+PNn[i] \n">>$output
printf "    \n">>$output
done
printf "    Ifb[i]=alpha*(V$((finish))[i] - Vrest) \n">>$output
printf "     \n">>$output
    for ((j = $finish; j > $start; j--))
    do
    printf "    I$j[i]=alpha*(V$((j-1))[i] - Vrest)\n">>$output
    done
    printf "     \n">>$output
        for ((j = $start; j<= $finish; j++))
        do
        printf "    V$((j))_dot[i]=V$((j))_st_dot[i] -(1/TAU)*(V$((j))[i]-V$((j))_st[i]) \n">>$output
        done
        printf "     \n">>$output
            for ((j = $finish; j > $start; j--))
            do
            printf "    I$((j))_dot[i]=alpha*(V$((j-1))_dot[i]) \n">>$output
            done
            printf "     \n">>$output
                for ((j = $finish; j > $start; j--))
                do
                printf "    V$((j))_dot_dot[i]=V$((j))_st_dot_dot[i] -(1/TAU)*(V$((j))_dot[i]-V$((j))_st_dot[i]) \n">>$output
                done
                printf " \n">>$output
printf "    I$((finish))_st[i]= Cm*Vst_dot[i] - (Cm/TAU)*(V$((finish))[i]-Vst[i]) + (F$((finish))[i]) \n">>$output
printf "    I$((finish))_st_dot[i]= Cm*Vst_dot_dot[i] - (Cm/TAU)*(V$((finish))[i+1]-Vst_dot[i]) + (F$((fini         sh))_dot[i]) \n">>$output
printf "    I$((finish))_st_dot_dot[i]= Cm*Vst_dot_dot_dot[i] - (Cm/TAU)*(V$((finish))_dot_dot[i]-Vst_dot_dot[i]) + (F$((j))_dot_dot[i]) \n">>$output
printf "     \n">>$output
                    for ((j = $finish-1; j > $start; j--))
                    do
                    printf "    V$((j))_st[i]=I$((j+1))_st[i]/alpha +Vrest\n">>$output 
                    printf "    V$((j))_st_dot[i]=I$((j+1))_st_dot[i]/alpha \n">>$output
                    printf "    V$((j))_st_dot_dot[i]=I$((j+1))_st_dot_dot[i]/alpha \n">>$output
                    printf "     \n">>$output
                    printf "    I$((j))_st[i]= Vst_dot[i] - (1/TAU)*(V$((j))[i]-Vst[i]) + (1/Cm)*(F$((j))[i]) \n">>$output
                    printf "    I$((j))_st_dot[i]= Vst_dot_dot[i] - (1/TAU)*(V$((j))[i+1]-Vst_dot[i]) + (1/Cm)*(F$((j))_dot[i]) \n">>$output
                    printf "    I$((j))_st_dot_dot[i]= Vst_dot_dot_dot[i] - (1/TAU)*(V$((j))_dot_dot[i]-Vst_dot_dot[i]) + (1/Cm)*(F$((j))_dot_dot[i]) \n">>$output
                    printf "     \n">>$output
                    done
printf "    V$((start))_st[i]=I$((start))_st[i]/alpha +Vrest\n">>$output 
printf "    V$((start))_st_dot[i]=I$((start+1))_st_dot[i]/alpha \n">>$output
printf "     \n">>$output
printf "    IC[i]= Cm*Vst_dot[i] - (Cm/TAU)*(V$((start))[i]-V$((start))_st[i]) + (F$((start))[i]) \n">>$output
printf "    I$((start))[i]=IC[i]+Ifb[i] \n">>$output
printf "     \n">>$output
        for ((j = $start; j<=$finish; j++))
        do
        printf "    IT$j[i]= I$j[i] - F$j[i]\n">>$output
        printf "    V$j[i+1]=V$j[i]+dt*IT$j[i]/Cm + PNV[i] \n">>$output
        printf "    \n">>$output
        done
        printf "    delta[i]= abs(V$finish[i]-V$((finish))_st[i])\n">>$output
printf "\n">>$output
printf "### ACHIEVABILITY & ERROR ANALYSIS\n">>$output
printf "\n">>$output    
printf "nDELTA     = np.sum(delta)\n">>$output
printf "ndelta_bar = nDELTA*dt/(time[30]-time[0])\n">>$output
printf "DELTA = np.linspace(0, nDELTA,length)\n">>$output        
printf "delta_bar=np.linspace(0,ndelta_bar,length)\n">>$output
printf "\n">>$output    
printf "#print(V1)\n">>$output
printf "#quit()\n">>$output
printf "\n">>$output    
printf "figure()  \n">>$output
printf "\n">>$output
printf "plt.subplot(2,2,1)\n">>$output
printf "plt.plot(time,V1, label='V1') \n">>$output
printf "ylabel('V1(mV)') \n">>$output
printf "xlabel('time')\n">>$output
printf "#ylim(-3.0,5.0)\n">>$output
printf "xlim(0.0,30.0)\n">>$output
printf "plt.grid()\n">>$output
printf "plt.hold(True) \n">>$output
printf "\n">>$output
    for ((j = 2; j <= $finish; j++))
    do
    printf "plt.subplot(2,2,$j)\n">>$output
    printf "plt.plot(time,V$j, label='V$j') \n">>$output
    printf "ylabel('V$j(mV)') \n">>$output
    printf "xlabel('time')\n">>$output
    printf "xlim(0.0,30.0)\n">>$output
    printf "#ylim(-8.0,12.0)\n">>$output
    printf "plt.grid()\n">>$output
    printf "plt.hold(True) \n">>$output
    printf "\n">>$output
    done
printf "plt.subplot(2,2,$((finish+1))) \n">>$output
printf "plt.plot(time,Vst, label='Vst') \n">>$output
printf "ylabel('V* (mV)') \n">>$output
printf "xlabel('time')\n">>$output
printf "xlim(0.0,30.0)\n">>$output
printf "#ylim(-8.0,12.0)\n">>$output
printf "plt.grid()\n">>$output
printf "\n">>$output
printf "figure()  \n">>$output
printf "\n">>$output
printf "plt.subplot(2,2,1)\n">>$output
printf "plt.plot(time,I1, label='I1') \n">>$output
printf "ylabel('I1(mA)') \n">>$output
printf "xlabel('time')\n">>$output
printf "#ylim(-500.0,500.0)\n">>$output
printf "xlim(0.0,30.0)\n">>$output
printf "plt.grid()\n">>$output
printf "plt.hold(True) \n">>$output
printf "\n">>$output
    for ((j = 2; j < $finish; j++))
    do
    printf "plt.subplot(2,2,$j)\n">>$output
    printf "plt.plot(time,I$j, label='I$j') \n">>$output
    printf "ylabel('I$j(mA)') \n">>$output
    printf "xlabel('time')\n">>$output
    printf "xlim(0.0,30.0)\n">>$output
    printf "#ylim(-30.0,50.0)\n">>$output
    printf "plt.grid()\n">>$output
    printf "plt.hold(True) \n">>$output
    printf "\n">>$output
    done
printf "plt.subplot(2,2,$finish) \n">>$output
printf "plt.plot(time,I$finish, label='I$finish') \n">>$output
printf "ylabel('I$finish (mA)') \n">>$output
printf "xlabel('time')\n">>$output
printf "xlim(0.0,30.0)\n">>$output
printf "#ylim(-30.0,50.0)\n">>$output
printf "plt.grid()\n">>$output
printf "\n">>$output
printf "figure()\n">>$output
printf "plt.subplot(2,1,1)\n">>$output
printf "plt.plot(time,delta, label='0.1')\n">>$output 
printf "ylabel('delta')\n">>$output
printf "xlabel('time(ms)') \n">>$output
printf "title('Vst=a*np.cos(b*time + c) + b*np.cos(c*time + d) + c*np.cos(d*time + a) + d*np.cos(a*time + b)')\n">>$output
printf "#ylim(0.0,0.25)\n">>$output
printf "xlim(0.0,30.0)\n">>$output
printf "plt.grid()\n">>$output
printf "plt.hold(True)\n">>$output
printf "\n">>$output
printf "subplot(2,1,2)\n">>$output
printf "plt.plot(epsilon,delta_bar, label='delta bar')\n">>$output 
printf "ylabel('delta_bar')\n">>$output
printf "xlabel('epsilon')\n">>$output
printf "#ylim(0.0,15.0)\n">>$output
printf "xlim(0.0,1.0)\n">>$output
printf "plt.grid()\n">>$output
printf "show()\n">>$output
chmod +x multiple.py
./multiple.py
