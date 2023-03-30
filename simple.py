import numpy as np
import time
import os
from environments import fixed_env


#Initial parameters
N_MC = 2
t = 40.
deltat = 0.2
reward_equation="(-4*(x-np.sqrt(2)/2)**2 - p**2) - 10*(1-done_now)"    #double well
#reward_equation="np.abs(x)" #cosh potential

env = fixed_env(N_MC=N_MC,
                reward_equation=reward_equation,   #Default reward equation
                t=t,
                deltat = deltat
)
N_states = env.N_states
N_actions = env.N_actions
n_time_steps = env.n_time_steps-1
rewards = np.empty((N_MC,n_time_steps))
state = np.zeros((N_MC,N_states))
returns = np.zeros((N_MC, n_time_steps))
seed = 0
rng = np.random.default_rng(seed)


start_time = time.time()

directory = 'data/simple&z0=random&t={:d}'.format(int(start_time))
os.mkdir(directory)

with open(directory+'/details.txt','w') as f:       # Write all relevant parameters to a txt file
    f.write('Simple test (no RL control)')
    f.write('MC = {:d}\n'.format(N_MC))
    #f.write('Timesteps = {:.1f}\n'.format(n_time_steps))
    #f.write('Duration = {:d}\n'.format(env.duration))
    #f.write('Differential equation: jax_vector\n')


for epoch in range(1):
    start_time = time.time()
    #env.change_initial_state([rng.uniform(low=-5,high=5,size=N_MC),rng.uniform(low=-np.pi,high=np.pi,size=N_MC)], multiple_initial_states=False)
    env.reset()                             #Choose the first state
    timestep = 0
    dones = np.zeros(N_MC)
    while timestep<n_time_steps:
                action = np.where(np.sign(state[:,0])*np.sign(state[:,1])<0,1,0)
                #action = np.where(np.abs(state[:,0])<1.5,-np.sign(state[:,0]),-np.sign(state[:,0])*np.sign(state[:,2])) + 1 #actions: 0:left, 1:stop, 2:right
                state, rewards[:,timestep], dones = env.step(action,dones)
                timestep += 1
    returns[:,:] = np.cumsum(rewards[:,::-1], axis=1)[:,::-1]
    print("Mean return:",np.mean(returns[:,0]),"Time:",time.time()-start_time)
    print("Successful trajectories: {:d}/{:d}\n".format(np.sum(dones),N_MC))


    

    probs = np.zeros((20,20,3))
    #before_plot = time.time()
    env.Render((directory,epoch,returns), probs=probs, trajectory_numbers_to_show = range(N_MC))
    #print(f'{time.time()-before_plot:.2f} s')

