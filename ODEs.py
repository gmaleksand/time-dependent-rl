from jax import jit
from jax.numpy import sin, cos, tan, sqrt, cosh, tanh, split, hstack, pi, exp

@jit
def cosh_fixed(t, z, Fx):
    p, x = split(z,2)
    dp =  -tanh(x)/cosh(x)+Fx
    dx = p
    f = hstack((dp,dx))
    return f

@jit
def cosh_var_height(t, z, Fx):
    p, x = split(z,2)
    dp =  -tanh(x)/cosh(x)*(1+0.5*sin(2*t))+Fx
    dx = p
    f = hstack((dp,dx))
    return f

@jit
def cosh_var_width(t, z, Fx):
    p, x = split(z,2)
    a = 1+0.2*sin(sqrt(2)*t)
    dp =  -tanh(x/a)/cosh(x/a)/a+Fx
    dx = p
    f = hstack((dp,dx))
    return f

@jit
def cosh_var(t, z, Fx):
    p, x = split(z,2)
    A = 1+0.5*sin(2*t)
    a = 1+0.2*sin(sqrt(2)*t)
    dp =  -A*tanh(x/a)/cosh(x/a)/a+Fx
    dx = p
    f = hstack((dp,dx))
    return f

@jit
def double_well_fixed(t, z, Fx):
    p, x = split(z,2)
    dp = 2*x-4*(x**3)+Fx
    dx = p
    f = hstack((dp,dx))
    return f

@jit
def double_well_var(t, z, Fx):
    p, x = split(z,2)
    a = 1+0.9*sin(0.5*t)
    dp = a*(2*x-4*(x**3))+Fx
    dx = p
    f = hstack((dp,dx))
    return f

"""
    2D potential:
    pr = dL/dr = m r'
    ptheta = dL/dtheta = m r^2 theta'
    
    H = pr^2/2m + ptheta^2/2(mr^2) - A sin^2(r/r0) sin^2(theta + omega*t)
    
    dr = dH/dpr     dtheta = dH/dptheta
    dpr = -dH/dr    dptheta = -dH/dtheta
"""

@jit
def field2D_fixed(t, z, F_dir):
    pr, ptheta, r, theta = split(z,4)
    dr = pr
    dtheta = ptheta/(r**2+0.01)
    dpr = -sin(2*r)*(sin(theta))**2 + ptheta**2/(r**3+0.01)+0.05*sin(F_dir)  #Force decreased 10 times
    dptheta = -(sin(r)**2)*sin(2*theta)+0.05*cos(F_dir)*r
    f = hstack((dpr,dptheta,dr,dtheta))
    return f

@jit
def field2D_var(t, z, F_dir):
    pr, ptheta, r, theta = split(z,4)
    dr = pr
    dtheta = ptheta/(r**2+0.01)
    dpr = -sin(2*r)*(sin(theta+0.4*t))**2 + ptheta**2/(r**3+0.01)+0.5*sin(F_dir)
    dptheta = -(sin(r)**2)*sin(2*theta+0.8*t)+0.5*cos(F_dir)*r
    f = hstack((dpr,dptheta,dr,dtheta))
    return f

@jit
def field2D_fixed_without_shortcut(t, z, F_dir):
    pr, ptheta, r, theta = split(z,4)
    dr = pr
    dtheta = ptheta/(r**2+0.01)
    dpr = -sin(2*r)*0.5 + ptheta**2/(r**3+0.01)+0.05*sin(F_dir)
    dptheta = 0.05*cos(F_dir)*r
    f = hstack((dpr,dptheta,dr,dtheta))
    return f

@jit
def field2D_fixed_thin_shortcut(t, z, F_dir):
    pr, ptheta, r, theta = split(z,4)
    dr = pr
    dtheta = ptheta/(r**2+0.01)
    dpr = -sin(2*r)*(1/cosh(0.1*tan(theta))) + ptheta**2/(r**3+0.01)+0.5*sin(F_dir)
    dptheta = 0.1*(sin(r)**2)*tanh(0.1*tan(theta))/(cosh(0.1*tan(theta))*cos(theta)**2)+0.5*cos(F_dir)*r
    f = hstack((dpr,dptheta,dr,dtheta))
    return f

@jit
def field2D_var_thin_shortcut(t, z, F_dir):
    pr, ptheta, r, theta = split(z,4)
    dr = pr
    dtheta = ptheta/(r**2+0.01)
    dpr = -sin(2*r)*(1/cosh(0.1*tan(theta))) + ptheta**2/(r**3+0.01)+0.5*sin(F_dir)
    dptheta = 0.1*(sin(r)**2)*tanh(0.1*tan(theta))/(cosh(0.1*tan(theta))*cos(theta)**2)+0.5*cos(F_dir)*r
    f = hstack((dpr,dptheta,dr,dtheta))
    return f

@jit
def field2D_fixed_v2(t, z, F_dir):
    pr, ptheta, r, theta = split(z,4)
    dr = pr
    dtheta = ptheta/(r**2+0.01)
    dpr = ptheta**2/(r**3+0.01) -sin(2*r)*(1-exp(-10*tan(theta)**2)) + 0.5*sin(F_dir)
    dptheta = 20*(sin(r)**2)*exp(-10*tan(theta)**2)*tan(theta)/cos(theta)**2 + 0.5*cos(F_dir)*r
    f = hstack((dpr,dptheta,dr,dtheta))
    return f