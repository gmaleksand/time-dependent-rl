from jax import jit
from jax.numpy import sin, cos, tan, sqrt, cosh, tanh, split, hstack, pi, exp
import jax.numpy as jnp

@jit
def field2D(t, z, F_dir):                                                        # H = pr^2/2m + ptheta^2/2(mr^2)
    pr, ptheta, r, theta = split(z,4)                                                   #     - A sin^2(r/r0) ( 1- e^- (k sin^2(theta+omega*t) )
    s = sin(theta+t)
    drag = .5
    dr = pr
    dtheta = jnp.clip(jnp.nan_to_num(ptheta/r),a_min=-10,a_max=10)
    dpr = jnp.clip(jnp.nan_to_num(ptheta**2/r),a_min=-10,a_max=10) - 2*(jnp.abs(r)*sin(2*r)+sin(r)**2*jnp.sign(r))*(1-exp(-10*s**2)) + sin(F_dir) - drag*pr
    dptheta = -jnp.sign(r)*(sin(r)**2)*exp(-10*s**2)*40*s*cos(theta+t) + cos(F_dir) - drag*ptheta
    f = hstack((dpr,dptheta,dr,dtheta))
    return f
