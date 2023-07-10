from jax import jit
from jax.numpy import sin, split, hstack
import jax.numpy as jnp


@jit
def double_well_fixed(t, z, Fx):
    p, x = split(z,2)
    dp = 4*(x-x**3)+Fx
    dx = p
    f = hstack((dp,dx))
    return f

@jit
def double_well_var(t, z, Fx):
    p, x = split(z,2)
    a = 1+0.9*sin(0.5*t)
    dp = 4*a*(x-x**3)+Fx
    dx = p
    f = hstack((dp,dx))
    return f