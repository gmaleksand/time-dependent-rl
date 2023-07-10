from jax import jit
from jax.numpy import sin, sqrt, cosh, tanh, split, hstack
import jax.numpy as jnp


@jit
def cosh_fixed(t, z, Fx):
    p, x = split(z,2)
    dp =  -tanh(x)/cosh(x)+Fx
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