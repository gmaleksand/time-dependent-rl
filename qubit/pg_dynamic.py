from jax import grad , jit, random, lax, value_and_grad
import jax.numpy as jnp
import numpy as np
from jax.example_libraries import optimizers, stax #from jax.experimental import optimizers, stax (use with the old version of JAX)
from jax. tree_util import tree_flatten
import time
from matplotlib import pyplot as plt
import os
from environments import *
import pickle
from openpyxl import load_workbook

#Initial parameters
N_epochs = 4002
N_MC = 512

env = qubit_env(n_time_steps = 60, N_MC = N_MC)
K = 1
T0 = 0.
Tdecay = 100.0
seed = 0
begin_time = int(time.time())

#Initializing the neural network

rng = np.random.default_rng(seed)
jrng = random.PRNGKey(seed)

N_states = env.N_states
N_actions = env.N_actions
n_time_steps = env.n_time_steps
temperature = T0

#Hyperparameters of the neural network

step_size = 0.001
b1 = .9 
b2 = .999
eps = 1e-8
l2_param = 0.001
discount_rate = 1
layer_1 = 512
layer_2 = 256
layer_3 = 64

optimizer_name='adam'
if optimizer_name=='sgd':
    opt_init, opt_update, get_params = optimizers.sgd(step_size)
if optimizer_name=='adam':
    opt_init, opt_update, get_params = optimizers.adam(step_size, b1=b1, b2=b2, eps=eps)



initialize_params, predict = stax.serial(
                                            stax.Dense(layer_1), stax.elementwise(jnp.sin),
                                            stax.Dense(layer_2), stax.Relu,
                                            stax.Dense(layer_3), stax.Relu,
                                            stax.Dense(N_actions), stax.LogSoftmax
                                        )

output_shape, initial_params = initialize_params(jrng, (-1,n_time_steps,N_states+1))

rewards = np.empty((N_MC,n_time_steps))

opt_state = opt_init(initial_params)
state = np.zeros((N_MC,N_states), dtype=np.float32)
statelist = np.zeros((N_MC, n_time_steps,N_states+1))
state_and_time = np.zeros((N_MC, N_states+1))
actionlist = np.zeros((N_MC, n_time_steps),dtype=int)
returns = np.zeros((N_MC, n_time_steps))
returns_mask = np.ones((N_MC, n_time_steps), dtype=bool)      #array which will contain masks
mean_return = np.zeros(N_epochs)
min_return = np.zeros(N_epochs)
max_return = np.zeros(N_epochs)
entropylist = np.zeros(N_epochs)             #array which will save the entropy of the policy
successful_trajectories = np.zeros(N_epochs) #array which will save the number of successful trajectories
Tlist = np.zeros(N_epochs)
Tlist = T0*np.exp(-np.arange(N_epochs)/Tdecay)
#Tlist[800:] = T0*np.exp(-800/Tdecay)




def cost(params, oldparams, batch, temperature, eps=0.1):

    #states, actions, returns, returns_mask = batch      #added masks to batch
    states, actions, returns = batch

    preds = predict(params, states)
    oldpreds = predict(oldparams, states)

    #baseline = jnp.nan_to_num(jnp.true_divide(returns.sum(axis=0),returns_mask.sum(axis=0)))
    baseline = jnp.mean(returns,axis=0)

    preds_trajectory = jnp.take_along_axis(preds, jnp.expand_dims(actions, axis=2), axis=2).squeeze()
    #"""     PPO
    oldpreds_trajectory = lax.stop_gradient(jnp.take_along_axis(oldpreds, jnp.expand_dims(actions, axis=2), axis=2).squeeze())
    A = (returns-baseline)                                                                 #Advantage
    imp = jnp.exp(preds_trajectory-oldpreds_trajectory)                                    #Importance sampling ratio
    #c = -jnp.mean(jnp.sum(jnp.minimum(imp*A,jnp.clip(imp,1-eps,1+eps)*A), axis=1))         #Simple cost
    c = -jnp.mean(jnp.mean(jnp.minimum(imp*A,jnp.clip(imp,1-eps,1+eps)*A), axis=1)) -jnp.mean(temperature*entropy(preds)) + l2reg(params, l2_param)
    #"""    END PPO
    """     policy gradient"""
    #c = -jnp.mean(jnp.mean(preds_trajectory * lax.stop_gradient(returns_mask*(returns - baseline)), axis=1)) -jnp.mean(temperature*entropy(preds)) + l2reg(params, l2_param)     #masked pg
    #c = -jnp.mean(jnp.sum(preds_trajectory*(returns-baseline), axis=1)) + l2reg(params, l2_param)         #policy gradient
    return c

def entropy(preds):
    H = -jnp.sum(preds*preds, axis=2)
    return H


def true_entropy(preds):
    H = -jnp.sum(preds*jnp.exp(preds), axis=2)
    return H

def l2reg(params, l2_param):
    return l2_param*jnp.sum(jnp.array([jnp.sum(jnp.abs(theta)**2) for theta in tree_flatten(params)[0] ]))

@jit #speed-up
def update(i, opt_state, oldparams, batch, temperature):
    current_params = get_params(opt_state)
    grad_params = grad(cost)(current_params, oldparams, batch, temperature)
    return opt_update(i, grad_params, opt_state)


def render():                    #Training curve

    plt.rc('font', size=15)        #Resize font
    plt.figure(figsize=(19.2,10.8))
    episodes=list(range(N_epochs))
    plt.plot(episodes[:-1],mean_return[:-1],label='Mean return')
    #plt.fill_between(episodes[:-1],min_return[:-1],max_return[:-1], color='k', alpha=0.25)
    plt.xlabel('Epoch')
    plt.ylabel('Mean return')
    plt.grid(True)
    plt.title('Returns per epoch')
    #plt.legend()

    #Save plot
    plt.savefig('train_curve.png')
    plt.close('all')
    #""" Successful trajectories plot
    plt.figure(figsize=(19.2,10.8))
    plt.plot(episodes,successful_trajectories,label='Successful trajectories')
    plt.xlabel('Epoch')
    plt.ylabel('Successful trajectories')
    plt.savefig('successful_trajectories.png')
    plt.close('all')
    #"""
    ### Agent entropy curve
    plt.figure(figsize=(19.2,10.8))
    plt.plot(episodes,entropylist,label='Policy entropy')
    plt.xlabel('Epoch')
    plt.ylabel('Entropy')
    plt.savefig('entropy.png')
    plt.close('all')


    ### Loss on which the agent is learning
    plt.figure(figsize=(19.2,10.8))
    plt.xlabel("Epoch")
    plt.ylabel("Learning loss")
    plt.plot(episodes[:-1],-2*entropylist[:-1]*Tlist[:-1]+mean_return[:-1]/n_time_steps)
    plt.plot(episodes[:-1],mean_return[:-1]/n_time_steps)
    plt.savefig('loss.png')
    plt.close('all')


#Vectorized random choice
#https://stackoverflow.com/questions/47722005/vectorizing-numpy-random-choice-for-given-2d-array-of-probabilities-along-an-a
def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(rng.random(a.shape[1-axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)


epoch = 0

env.plot_decay_probability()

# Start training
while epoch < N_epochs:

    start_time = time.time()

    if epoch%250 == 0:                          #Simulation backup
        save_total_state()

    nnparams = get_params(opt_state)        #Get the current policy in policy matrix



    env.reset()                             #Choose the first state
    dones = np.zeros(N_MC)
    for timestep in range(env.n_time_steps):
        state_and_time[:,1:] = env.state
        state_and_time[:,0]  = timestep
        statelist[:,timestep,:] = state_and_time
        p = np.exp(predict(nnparams, state_and_time))
        action = random_choice_prob_index(p)    #Vectorized random choice
        _ , rewards[:,timestep], dones = env.step(action, dones)
        actionlist[:,timestep] = action



    #returns[:,:] = jnp.cumsum(rewards[:,::-1], axis=1)[:,::-1]                     #non-discounted cumulative sum of the rewards
    discounts = jnp.power(discount_rate,jnp.arange(n_time_steps))
    returns[:,:] = jnp.cumsum((rewards*discounts)[:,::-1], axis=1)[:,::-1]/discounts    #discounted cumulative sum (+3.5 s for 800 calls, N_MC = 512, 200 timesteps)
    #print(returns)

    batch = statelist, actionlist, returns#, returns_mask                     #added masks to batch

    oldparams = get_params(opt_state)
    if epoch !=N_epochs-1:  #Do not learn in the test epoch
        for u in range(K):                          #K=1 for policy gradient
            opt_state = update(epoch*K+u, opt_state, oldparams, batch, Tlist[epoch])

    #The following calculations are needed for recording and display
    mean_return[epoch] = np.mean(returns[:,-1])
    min_return[epoch] = np.min(returns[:,-1])
    max_return[epoch] = np.max(returns[:,-1])
        
    nnparams = get_params(opt_state)
    preds = predict(nnparams, statelist)
    entropylist[epoch] = jnp.mean(true_entropy(preds))
    

    if epoch%10 == 0:
        #Print results of the epoch
        epoch_time = time.time()-start_time
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time) )
        print("Returns: {:0.4f} mean; {:0.4f} min; {:0.4f} max".format(mean_return[epoch],min_return[epoch], max_return[epoch]) )
        print("Successful trajectories: {:.1f}/{:d}\n".format(np.sum(dones),N_MC))
        successful_trajectories[epoch] = np.sum(dones)

    if epoch==N_epochs-1:
        env.Render((epoch,rewards), trajectory_numbers_to_show=[36, 74, 93, 104]) # these trajectories were chosen so that they pass through different parts of the Bloch sphere.
        render()
        

    epoch += 1

print('\a')     #Alert when the program has finished