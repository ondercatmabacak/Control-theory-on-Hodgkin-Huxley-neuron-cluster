#!/usr/bin/env python3 
import sympy as sp
'''
a_m=0.1*((25- sp.power(t,2) )/(sp.exp((25- sp.power(t,2) )/10)-1)) 
b_m=4*sp.exp(- sp.power(t,2) /18) 
a_h=0.07*sp.exp(- sp.power(t,2) /20) 
b_h=1/(sp.exp((30- sp.power(t,2) )/10)+1) 
a_n=0.01 * ((10- sp.power(t,2) )/(sp.exp((10- sp.power(t,2) )/10)-1)) 
b_n=0.125*sp.exp(- sp.power(t,2) /80) 


m  =  a_m  / (a_m + b_m )
n  =  a_n  * (b_n + a_n )
h  =  b_h  * (a_h + b_h )

#m  =  (a_m  / (a_m + b_m ))*(1- sp.exp(t*(a_m + b_m )))
#n  =  (a_n  * (b_n + a_n ))*(1- sp.exp(t*(a_m + b_m )))
#h  =  (b_h  * (a_h + b_h ))*(1- sp.exp(t*(a_m + b_m )))
'''
x   = sp.Symbol('x')

def f(x):
    return 0.01 * ((10- sp.power(x,2) )/(sp.exp((10- sp.power(x,2) )/10)-1))/(0.01 * ((10- sp.power(x,2) )/(sp.exp((10- sp.power(x,2) )/10)-1)) + 0.125* sp.exp (- sp.power(x,2) /80))

g = [f(x).diff(x)]
print('grad f(x)', g)
