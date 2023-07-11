import numpy as np
import jax.numpy as jnp
from scipy.linalg import expm
import matplotlib.pyplot as plt
from qutip import Bloch
import pickle
from colorsys import hsv_to_rgb

def arccos_abs(x):
    return np.arccos(np.clip(np.abs(x),-1,1))

class qubit_env():
    
    def __init__(self, n_time_steps, N_MC, seed=0):    #Initialize the environment.

        self.n_time_steps = n_time_steps
        self.N_MC = N_MC
        total_time = 1*np.pi
        dt = total_time/n_time_steps
        self.dt = dt
        # Pauli matrices
        Id =  np.array([[1. , 0. ],
                        [0. , 1. ]])

        s_x = np.array([[0. , 1. ],
                        [1. , 0. ]])

        s_y = np.array([[0. ,-1.j],
                        [1.j, 0. ]])

        s_z = np.array([[1.,  0. ],
                        [0., -1. ]])
        # Actions
        self.N_actions = 7
        self.actions = np.array([expm(-1j*dt*Id),
                        expm(-1j*dt*s_x),
                        expm(-1j*dt*s_y),
                        expm(-1j*dt*s_z),
                        expm( 1j*dt*s_x),
                        expm( 1j*dt*s_y),
                        expm( 1j*dt*s_z)])

        # Target state
        self.S_terminal = np.array([0.,0.])     # North Pole (phi = 0, theta = 0)
        self.psi_terminal = self.rl_to_qubit(self.S_terminal)
        self.cap_size = 1e-2    # Tolerance

        self.N_states = 2

        self.set_seed(seed)
        self.reset()

    #Works for a single qubit
    def qubit_to_rl(self,psi):          # |ψ> = e^(iα)(cos(θ/2), e^(iφ)sin(θ/2))
        alpha = np.angle(psi[0])        # Global phase
        psi0 = np.exp(-1j*alpha)*psi    # Remove the global phase from psi
        theta = 2.0*np.arccos(psi0[0]).real  #ψ0[0] = cos(θ/2)
        phi = np.angle(psi0[1])              #ψ0[1] = e^(iφ)*sin(θ/2)
        return np.array([theta,phi])

    #Works for a single qubit
    def rl_to_qubit(self,s):
        theta, phi = s
        return np.array([                np.cos(.5*theta),
                          np.exp(1j*phi)*np.sin(.5*theta)])

    def qubits_to_rl(self,psi):    # returns (N_MC, 2) vector of states
        """
        alpha = np.angle(psi[:,0])
        psi0 = np.zeros((self.N_MC,2), dtype=np.complex_)
        theta = np.zeros(self.N_MC)
        phi = np.zeros(self.N_MC)
        
        psi0 = np.exp(-1j*alpha)[:,np.newaxis]*psi    # Remove the global phase from psi
        theta = 2.0*np.arccos(psi0[:,0]).real         #psi0[0] = cos(theta/2)
        phi = np.angle(psi0[:,1])                     #psi0[1] = e^(i*phi)*sin(theta/2)
        """
        phi = np.angle(np.nan_to_num(psi[:,1]/psi[:,0], nan=0))           # |ψ> = e^(iα)(cos(θ/2), e^(iφ)sin(θ/2))
        theta = 2.0*arccos_abs(psi[:,0])
        #"""
        """
        print(theta)
        print(2.0*np.arccos(np.abs(psi[:,0])))
        print(phi)
        print(np.angle(psi[:,1]/psi[:,0]))
        """
        return np.array([theta,phi]).T

    def rls_to_qubits(self,s):  # returns (N_MC, 2) vector of qubits
        theta, phi = s.T
        return np.array([                np.cos(.5*theta),          # |ψ> = (cos(θ/2), e^(iφ)sin(θ/2))
                          np.exp(1j*phi)*np.sin(.5*theta)]).T

    def decay_probability(self, psi):
        #"""
        thetas = 2.0*arccos_abs(psi[:,0])
        modified_phis = (np.angle(psi[:,1]) - np.angle(psi[:,0]) + (self.time_step/self.n_time_steps)*2.5*np.pi)%(2*np.pi)-np.pi
        """ Time-dependent barrier"""
        #phis = (np.angle(psi[:,1]) - np.angle(psi[:,0]) + self.time_step/self.n_time_steps*(2*np.pi))%(2*np.pi)-np.pi
        #print(phis)
        #phis = (np.angle(np.nan_to_num(psi[:,1]/psi[:,0])))#%(2*np.pi)-np.pi
        #print(phis)
        return 0.5*np.exp(-20*(thetas-np.pi/2)**2)*(1-np.exp(-modified_phis**2))
        #return 0
        #return np.exp(-thetas**2)*(1-np.exp(-10*phis**2))

    def step(self, action, done_before): #Performs one step in the environemnt.
        self.psi = np.einsum('ijk,ik->ij',self.actions[action],self.psi)      # apply gate (i indexes the Monte Carlo samples)
        # dissipate
        # ground_state: phi = 0,  theta = pi: psi = (0,1)
        psi_ground = np.array([np.zeros(self.N_MC),np.ones(self.N_MC)] , dtype= np.complex_).T
        self.psi = np.where(self.rng.random(self.N_MC) < self.decay_probability(self.psi), psi_ground.T, self.psi.T).T

        self.state = self.qubits_to_rl(self.psi)                     # compute rl state
        reward = np.zeros(self.N_MC)
        done = np.zeros(self.N_MC)
        
        reward = np.abs(self.psi_terminal.conj().dot(self.psi.T).T)**2
        #done = np.where(np.abs(reward-1.0)<self.cap_size,1,0)               # check if state is terminal
        done_now = np.where(np.abs(reward)<self.cap_size,1,0) #mark the dissipated states
        done = np.logical_or(done_now, done_before)
        #print(self.state[0], reward[0], done[0])
        self.time_step += 1
        self.state_history[self.time_step,:,:] = self.state[:self.N_MC,:]
        return self.state, reward, done


    def set_seed(self,seed=0):  #Sets the seed of the RNG.
        self.rng = np.random.default_rng(seed)

    def reset(self, random = True): #Resets the environment to its initial values.
        self.time_step = 0
        if random:  # Generate a random point on the Bloch sphere
            theta = 0.8*np.pi+0.2*np.pi*self.rng.random(self.N_MC)
            phi = 2*np.pi*self.rng.random(self.N_MC)
        else:       # Start from the south pole
            theta = np.pi*np.ones(self.N_MC)
            phi = np.zeros(self.N_MC)

        self.state = np.array([theta,phi]).T
        self.state_history = np.zeros((self.n_time_steps+1, self.N_MC, 2))
        self.state_history[0,:,:] = self.state[:self.N_MC,:]
        self.psi = self.rls_to_qubits(self.state)

    def change_initial_state(self, new_initial_state):
        self.state = new_initial_state
        self.psi = self.rls_to_qubits(self.state)


    def Render(self, label, trajectory_numbers_to_show = [0]):
        
        b = Bloch()
        b.make_sphere()
        epoch,rewards = label

        thetas = self.state_history[:,:,0]
        phis = self.state_history[:,:,1]
        xs = np.sin(thetas)*np.cos(phis)
        ys = np.sin(thetas)*np.sin(phis)
        zs = np.cos(thetas)


        for i in trajectory_numbers_to_show:
            b.add_points([xs[:,i],ys[:,i],zs[:,i]])

        #Save plot
        b.save('trajectories_{}.png'.format(epoch))
        b.clear()
        #"""

    def Render_rot(self,label,trajectory_numbers_to_show =[0]):
        b = Bloch()
        b.make_sphere()
        directory,epoch,rewards = label

        thetas = self.state_history[:,:,0]
        phis = self.state_history[:,:,1]+ (np.expand_dims(np.arange(self.n_time_steps+1),axis=1)/self.n_time_steps)*2.5*np.pi
        xs = np.sin(thetas)*np.cos(phis)
        ys = np.sin(thetas)*np.sin(phis)
        zs = np.cos(thetas)
        for i in trajectory_numbers_to_show:
            print(thetas[0,i])
            b.add_points([xs[:,i],ys[:,i],zs[:,i]])
        b.save('rot.png')
        b.clear()


    def plot_decay_probability(self, directory):
        n = 40
        thetamesh, phimesh = np.mgrid[slice(0, np.pi+np.pi/n, np.pi/n),slice(-np.pi, np.pi+2*np.pi/n, 2*np.pi/n)]
        n += 1 # the array created is in fact (n+1)x(n+1)
        rls = np.transpose([thetamesh, phimesh],(1,2,0))

        zmesh = np.zeros((n,n))
        for i in range(n):
            zmesh[i,:] = self.decay_probability(self.rls_to_qubits(rls[i,:]))

        #"""
        fig, ax1 = plt.subplots(figsize=(19.2,10.8))
        ax1.pcolormesh(thetamesh, phimesh, zmesh, vmin=-.5 , vmax=.5 ,cmap=plt.colormaps['PiYG'])                    # 2D static environment
        plt.xlabel('$theta$')
        plt.ylabel('$phi$')
        #Save plot
        file = '/dissipation.png'
        plt.savefig(directory+file)
        plt.close('all')
