#!/usr/bin/env python3 
from __future__ import division
import sympy as sp
import numpy as np

'''
a=2.0
b=3.0
c=4.0

time  = Symbol('time')
y     = a*sp.cos(b*time + c)
yprime=sp.diff(y,time)

print(yprime)
quit()
'''
from scipy.misc import derivative

def f(x):
    return 3*x**2-1

def d(x):
    return (derivative(f,x))

print(d(1.0))

