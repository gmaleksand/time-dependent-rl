import numpy as np
import jax.numpy as jnp
import odes
from scipy.integrate import ode
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle


class var_env():
    
    def __init__(self, N_MC=128, seed=0, Fpower=0.5, t=10., deltat = 0.1):    #Initialize the environment.

        # defining states and actions
        self.N_states = 6   # (pr,ptheta,r,sin(theta),cos(theta),t)
        self.N_actions = 8 # (0, pi/4, pi/2, ...)
        #self.actions = np.array([0, 1])     #0 - left, 1 - right
        self.Fpower = Fpower

        #initializing differential equation

        self.set_seed(seed)
        # Constants
        self.solver = ode(odes.field2D)
        self.solver.set_integrator('dop853', rtol=1e-3, atol=1e-6)            #Use the scipy ODE integrator (default:dop853)

        #self.x0 = -np.sqrt(2)/2*np.ones(N_MC)    # Initial state for double-well
        #self.x0 = np.zeros(N_MC)    # Initial state for cosh potential
        #self.p0 = np.zeros(N_MC)
        #self.r0 = np.zeros(N_MC)
        #self.theta0 = np.zeros(N_MC)
        self.r0 = self.rng.rayleigh(scale=np.pi/8, size=N_MC)
        self.theta0 = self.rng.uniform(low=0., high=2*np.pi, size=N_MC)
        #self.pr0 = np.zeros(N_MC)
        #self.ptheta0 = np.zeros(N_MC)
        self.pr0 = self.rng.normal(loc=0.,scale=.2,size=N_MC)
        self.ptheta0 = self.rng.normal(loc=0.,scale=.2,size=N_MC)

        self.z0 = np.zeros((4,N_MC))
        self.z0[0,:] = self.pr0
        self.z0[1,:] = self.ptheta0
        self.z0[2,:] = self.r0
        self.z0[3,:] = self.theta0

        self.N_MC = N_MC

        # Initial values
        self.t0 = 0
        self.t1 = t
        self.n_time_steps = int(t/deltat)+1
        # Time runs from t0 to t1, with resolution deltat
        self.t = np.linspace(self.t0+deltat, self.t1, self.n_time_steps)  

        #Array that records all trajectories for visualizations
        self.sol = np.empty((self.n_time_steps,4,N_MC))


        self.kicks = np.zeros((self.n_time_steps,4))   #Enable for visualizations

        self.reset()



    def step(self, action, done_before): #Performs one step in the environemnt.

        self.Fdir = np.pi/4*action            #Turn the RL-determined action to force

        """ Recording kicks is only required if we use arrows in the plot
        self.kicks[self.current_step,0] = self.sol[self.current_step-1,0,0]
        self.kicks[self.current_step,1] = self.sol[self.current_step-1,1,0]
        self.kicks[self.current_step,2] = self.Fx[0]/10
        self.kicks[self.current_step,3] = 0.
        #"""
        self.solver.set_f_params(self.Fdir)
        if not self.solver.successful():
            raise Exception("Unsuccessful integration")
        
        self.solver.integrate(self.t[self.current_step])
        pr, ptheta, r, theta = np.split(self.solver.y,4)        #x is an array of momentums, y is an array of angles
        self.sol[self.current_step,0,:] = pr
        self.sol[self.current_step,1,:] = ptheta
        self.sol[self.current_step,2,:] = r
        self.sol[self.current_step,3,:] = theta
        
        done_now = np.logical_and(np.logical_and(r>3*np.pi-0.1, r<3*np.pi+0.1), np.abs(pr)<0.1)     #(Mask) Condition for finishing the trajectory //2D
        if self.current_step>1:
            done = np.logical_or(done_now, done_before)                         #(Mask) Determine if the trajectory has finished
        else:
            done = done_now            


        self.current_step += 1

        """
        self.state[:,0] = pr
        self.state[:,1] = ptheta
        self.state[:,2] = r
        self.state[:,3] = np.sin(theta)
        self.state[:,4] = np.cos(theta)
        #self.state[:,2] = self.t[self.current_step-1]*2*np.pi/0.5  # the oscillations of the potential have angular frequency 0.5
        self.state[:,5] = np.sin(self.t[self.current_step-1])  # Direction of the shortcut
        #"""
        """
        self.state[:,0] = self.t[self.current_step-1]
        self.state[:,1] = pr
        self.state[:,2] = ptheta
        self.state[:,3] = np.abs(r)
        self.state[:,4] = np.sin(theta)
        self.state[:,5] = np.cos(theta)
        #"""
        #"""
        self.state[:,0] = self.t[self.current_step-1]
        self.state[:,1] = theta
        self.state[:,2] = pr
        self.state[:,3] = ptheta
        self.state[:,4] = r
        reward = -np.abs(r-3*np.pi)
        return self.state, reward, done

    def set_seed(self,seed=0):  #Sets the seed of the RNG.
        self.rng = np.random.default_rng(seed)

    def reset(self): #Resets the environment to its initial values.


        """
        self.r0 = self.rng.rayleigh(scale=np.pi/4)
        self.theta0 = self.rng.uniform(low=0.,high=2*np.pi)
        self.z0[2,:] = self.r0
        self.z0[3,:] = self.theta0
        """

        self.current_step = 0
        self.sol[0,:,:] = self.z0


        self.Fdir=0.0

        self.solver.set_f_params(self.Fdir)  #Solver with control
        self.solver.set_initial_value(self.z0.flatten(), self.t0)         # Set the initial value
        self.state = np.zeros((self.N_MC,self.N_states))    #3 when using sin and cos
        self.state[:,0] = self.z0[0,:]
        self.state[:,1] = self.z0[1,:]
        self.state[:,2] = self.z0[2,:]
        self.state[:,3] = self.z0[3,:]
        return self.state

    def change_initial_state(self, z0, multiple_initial_states = False):
        if multiple_initial_states:
            self.z0 = z0.copy()
        else:
            self.z0[0,:] = z0[0] #pr0
            self.z0[1,:] = z0[1] #ptheta0
            self.z0[2,:] = z0[2] #r0
            self.z0[3,:] = z0[3] #theta0
        self.reset()

    def Render(self, label, probs, critic_preds = None, use_insets = True, trajectory_numbers_to_show = [0, 50], show_arrows = False):
        plt.rc('font', size=15)        #Resize font
        epoch, returns = label
        fig, ax1 = plt.subplots(figsize=(19.2,10.8))
        
        """
        self.sol[:,1,:] = (self.sol[:,1,:]+np.pi) % (2*np.pi)-np.pi     #All angles are now between (-pi,pi)
        if self.usemasks:                                               #Send the states of a finished trajectory to (0,0).
            self.sol[:,0,:] = self.sol[:,0,:]*(1-self.donelist)
            self.sol[:,1,:] = self.sol[:,1,:]*(1-self.donelist)
        """

        total_trajecs = np.max(trajectory_numbers_to_show)+1
        colors = [np.minimum(np.linspace(0,2,num=total_trajecs),np.linspace(2,0,num=total_trajecs)),# red:  0 - 1 - 0
                  np.maximum(np.linspace(-1,1,num=total_trajecs),0),                                # green:0 - 0 - 1
                  np.maximum(np.linspace(1,-1,num=total_trajecs),0)]                                # blue: 1 - 0 - 0
        colors = list(map(tuple,np.swapaxes(colors,0,1)))               #Make the colors an array of tuples: [(,,),...,(,,)]


        r = self.sol[:,2,:]
        theta = self.sol[:,3,:]
        xs = r*np.cos(theta)
        ys = r*np.sin(theta)

        for i in trajectory_numbers_to_show:
            ax1.scatter(xs[:,i], ys[:,i], s=1.0, color=colors[i], label='trajectory {}'.format(i))                   #Create a scatter plot of the trajectories shown.
            ax1.plot(   xs[0,i], ys[0,i], color=colors[i], label='initial {}'.format(i), marker='o')                # Large point at the initial state
            ax1.plot(  xs[-1,i],ys[-1,i], color=colors[i], label='final {}'.format(i),   marker='^')      # Large point at the final state
                
        if show_arrows:
            for step in range(self.n_time_steps):   #Add arrows
                #if not self.donelist[step*self.duration,0]:                                   #(Mask) arrows to be shown
                plt.arrow(self.kicks[step,1], self.kicks[step,0], self.kicks[step,2], self.kicks[step,3], head_width=0.03, width=0.001 )
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        ax1.grid(True)
        plt.title('Mean return:{:.1f}'.format(np.mean(returns[:,0])))
        #ax1.legend()
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
        """
        if self.usemasks:
            ellipse = Ellipse((0, 0), width=np.arccos(1-np.exp(-4)) * 2, height=np.sqrt(2*np.exp(-4)) * 2, edgecolor='blue', facecolor='none')
            ax1.add_patch(ellipse)
        """
        if use_insets:
            left, bottom, width, height = [0.15, 0.65, 0.2, 0.2]        #Subplot with theta(t)
            ax2 = fig.add_axes([left, bottom, width, height])
            for i in trajectory_numbers_to_show:
                ax2.scatter(self.t,(theta[:,i]+np.pi)%(2*np.pi)-np.pi, s=1.0, color=colors[i],label='$theta$ {}'.format(i))
            ax2.set_xlabel('$t$')
            ax2.set_ylabel('$theta$')
            ax2.set_ylim([-np.pi, np.pi])


            left, bottom, width, height = [0.65, 0.65, 0.2, 0.2]        #Subplot with r(t)
            ax3 = fig.add_axes([left, bottom, width, height])
            for i in trajectory_numbers_to_show:
                ax3.scatter(self.t,np.abs(r[:,i]), s=1.0, color=colors[i],label='r {}'.format(i))
            ax3.set_xlabel('$t$')
            ax3.set_ylabel('$r$')
            ax3.set_ylim([0, 20])


            left, bottom, width, height = [0.65, 0.15, 0.2, 0.2]        #Subplot with return(timestep)
            ax4 = fig.add_axes([left, bottom, width, height])
            timestep_array = np.arange(self.n_time_steps)
            for i in trajectory_numbers_to_show:
                ax4.scatter(timestep_array, returns[i,:], s=1.0,color=colors[i],label='return {}'.format(i))
            ax4.set_xlabel('timestep')
            ax4.set_ylabel('reward')

            """
            left, bottom, width, height = [0.15, 0.15, 0.2, 0.2]        #Subplot with strategy
            ax5 = fig.add_axes([left, bottom, width, height])
            ax5.set_xlabel('$x$')
            ax5.set_ylabel('$p$')
            
            y = np.arange(-10, 10, 1)
            x = np.arange(-np.pi, np.pi, np.pi/10)

            z = np.empty((20,20))
            z = probs[:,:,0]
            
            ax5.pcolormesh(x, y, z, shading='auto')

            if not (critic_preds is None):          #Show only if critic_preds are given as a parameter
                left, bottom, width, height = [0.40, 0.65, 0.2, 0.2]        #Subplot with critic values
                ax6 = fig.add_axes([left, bottom, width, height])

                ax6.set_xlabel('$\\theta$')
                ax6.set_ylabel('$p$')
                ax6.grid(True)
                

                y = np.arange(-1, 1, 0.1)
                x = np.arange(-1.5, 1.5, 0.15)
                z = np.empty((20,20))
                z = critic_preds.squeeze()
                valuesplot = ax6.pcolormesh(x, y, z, shading='auto')
                plt.colorbar(valuesplot, ax=ax6)
                #ax6.colorbar()
            """

            left, bottom, width, height = [0.40, 0.15, 0.2, 0.2]        #Histogram of the returns
            ax7 = fig.add_axes([left, bottom, width, height])
            ax7.hist(returns[:,0],bins='auto')


        #Save plot
        plt.savefig('phase_portrait.png')


        plt.close('all')

    def save_phase_portrait(self, label, probs):
        directory, epoch, returns = label
        phasefile = open(directory+'/{}_phase_portrait.txt'.format(epoch), 'wb')
        pickle.dump([self.t, self.sol, self.kicks, returns, probs],phasefile)
        phasefile.close()


    def Render_best(self, label, probs, critic_preds = None, use_insets = True, show_arrows = False):
        _, _, returns = label
        trajectory_numbers_to_show = np.argmax(returns[:,0])
        self.Render(label, probs, critic_preds = critic_preds, use_insets = use_insets, trajectory_numbers_to_show = [trajectory_numbers_to_show], show_arrows = show_arrows)

class fixed_env():
    
    def __init__(self, N_MC=128, seed=0, Fpower=0.5, t=10., deltat = 0.1):    #Initialize the environment.

        # defining states and actions
        self.N_states = 5   # (pr,ptheta,r,sin(theta),cos(theta))
        self.N_actions = 8 # (0, pi/4, pi/2, ..., do_nothing)
        #self.actions = np.array([0, 1])     #0 - left, 1 - right
        self.Fpower = Fpower

        #initializing differential equation

        self.set_seed(seed)
        # Constants
        self.solver = ode(odes.field2D)
        self.solver.set_integrator('dop853', rtol=1e-3, atol=1e-6)            #Use the scipy ODE integrator (default:dop853)

        self.r0 = self.rng.rayleigh(scale=np.pi/8, size=N_MC)
        self.theta0 = self.rng.uniform(low=0., high=2*np.pi, size=N_MC)
        self.pr0 = self.rng.normal(loc=0.,scale=.2,size=N_MC)
        self.ptheta0 = self.rng.normal(loc=0.,scale=.2,size=N_MC)

        self.z0 = np.zeros((4,N_MC))
        self.z0[0,:] = self.pr0
        self.z0[1,:] = self.ptheta0
        self.z0[2,:] = self.r0
        self.z0[3,:] = self.theta0

        self.N_MC = N_MC

        # Initial values
        self.t0 = 0
        self.t1 = t
        self.n_time_steps = int(t/deltat)+1
        # Time runs from t0 to t1, with resolution deltat
        self.t = np.linspace(self.t0+deltat, self.t1, self.n_time_steps)  

        #Array that records all trajectories for visualizations
        self.sol = np.empty((self.n_time_steps,4,N_MC))


        self.kicks = np.zeros((self.n_time_steps,4))   #Enable for visualizations

        self.reset()



    def step(self, action, done_before): #Performs one step in the environemnt.

        #self.Fr = np.where(action==8, 0,np.sin(action*np.pi/4))            # allow empty action
        #self.Ftheta = np.where(action==8,0, np.cos(action*np.pi/4))        # allow empty action
        self.Fdir = np.pi/4*action            #Turn the RL-determined action to force

        """ Recording kicks is only required if we use arrows in the plot
        self.kicks[self.current_step,0] = self.sol[self.current_step-1,0,0]
        self.kicks[self.current_step,1] = self.sol[self.current_step-1,1,0]
        self.kicks[self.current_step,2] = self.Fx[0]/10
        self.kicks[self.current_step,3] = 0.
        #"""
        self.solver.set_f_params(self.Fdir)
        #self.solver.set_f_params(self.Fr,self.Ftheta)                     # allow empty action
        if not self.solver.successful():
            raise Exception("Unsuccessful integration")
        
        self.solver.integrate(self.t[self.current_step])
        pr, ptheta, r, theta = np.split(self.solver.y,4)        #x is an array of momentums, y is an array of angles
        #print(r[0])
        self.sol[self.current_step,0,:] = pr
        self.sol[self.current_step,1,:] = ptheta
        self.sol[self.current_step,2,:] = r
        self.sol[self.current_step,3,:] = theta
        
        done_now = np.logical_and(np.logical_and(r>3*np.pi-0.1, r<3*np.pi+0.1), np.abs(pr)<0.1)     #(Mask) Condition for finishing the trajectory //2D
        if self.current_step>1:
            done = np.logical_or(done_now, done_before)                         #(Mask) Determine if the trajectory has finished
        else:
            done = done_now            



        self.current_step += 1

        self.state[:,0] = pr
        self.state[:,1] = ptheta
        self.state[:,2] = r
        self.state[:,3] = np.sin(theta)
        self.state[:,4] = np.cos(theta)
        #print(x)
        #exit()
        reward = -np.abs(r-3*np.pi)
        #print(reward)
        #exit()
        return self.state, reward, done

    def set_seed(self,seed=0):  #Sets the seed of the RNG.
        self.rng = np.random.default_rng(seed)

    def reset(self): #Resets the environment to its initial values.

        """ Additional feature: randomize the initial phase 
        randomized_phase = self.rng.random()*2*np.pi/0.4
        self.t0 += randomized_phase
        self.t1 += randomized_phase
        self.t = np.linspace(self.t0, self.t1, self.n_time_steps)
        """

        self.current_step = 0
        self.sol[0,:,:] = self.z0

        self.Fdir = 0.
        #self.Fr = 0.
        #self.Ftheta = 0.
        self.solver.set_f_params(self.Fdir)  #Solver with control
        #self.solver.set_f_params(self.Fr,self.Ftheta)  #Solver with control
        self.solver.set_initial_value(self.z0.flatten(), self.t0)         # Set the initial value
        self.state = np.zeros((self.N_MC,self.N_states))    #3 when using sin and cos
        self.state[:,0] = self.z0[0,:]
        self.state[:,1] = self.z0[1,:]
        self.state[:,2] = self.z0[2,:]
        self.state[:,3] = np.sin(self.z0[3,:])
        self.state[:,4] = np.cos(self.z0[3,:])
        return self.state


    def change_initial_state(self, z0, multiple_initial_states = False):
        if multiple_initial_states:
            self.z0 = z0.copy()
        else:
            self.z0[0,:] = z0[0] #pr0
            self.z0[1,:] = z0[1] #ptheta0
            self.z0[2,:] = z0[2] #r0
            self.z0[3,:] = z0[3] #theta0
        self.reset()

    def Render(self, label, probs, critic_preds = None, use_insets = True, trajectory_numbers_to_show = [0, 50], show_arrows = False):
        plt.rc('font', size=15)        #Resize font
        directory, epoch, returns = label
        #plt.figsize=(19.2,10.8)
        fig, ax1 = plt.subplots(figsize=(19.2,10.8))
        

        ymesh, xmesh = np.mgrid[slice(-20, 20.1, 0.1),slice(-20, 20.1, 0.1)]
        #zmesh = np.sin(np.sqrt(ymesh**2+xmesh**2))**2*ymesh**2/(ymesh**2+xmesh**2)        # Add colors to the potential
        zmesh = np.sin(np.sqrt(ymesh**2+xmesh**2))**2*(1-np.exp(-10*ymesh**2/(ymesh**2+xmesh**2)))
        ax1.pcolormesh(xmesh, ymesh, zmesh, vmin=-2 , vmax=2 ,cmap=plt.colormaps['PiYG'])                    # 2D static environment


        """
        self.sol[:,1,:] = (self.sol[:,1,:]+np.pi) % (2*np.pi)-np.pi     #All angles are now between (-pi,pi)
        if self.usemasks:                                               #Send the states of a finished trajectory to (0,0).
            self.sol[:,0,:] = self.sol[:,0,:]*(1-self.donelist)
            self.sol[:,1,:] = self.sol[:,1,:]*(1-self.donelist)
        """

        total_trajecs = np.max(trajectory_numbers_to_show)+1
        colors = [np.minimum(np.linspace(0,2,num=total_trajecs),np.linspace(2,0,num=total_trajecs)),# red:  0 - 1 - 0
                  np.maximum(np.linspace(-1,1,num=total_trajecs),0),                                # green:0 - 0 - 1
                  np.maximum(np.linspace(1,-1,num=total_trajecs),0)]                                # blue: 1 - 0 - 0
        colors = list(map(tuple,np.swapaxes(colors,0,1)))               #Make the colors an array of tuples: [(,,),...,(,,)]


        r = self.sol[:,2,:]
        theta = self.sol[:,3,:]
        xs = r*np.cos(theta)
        ys = r*np.sin(theta)

        for i in trajectory_numbers_to_show:
            ax1.scatter(xs[:,i], ys[:,i], s=1.0, color=colors[i], label='trajectory {}'.format(i))                   #Create a scatter plot of the trajectories shown.
            ax1.plot(   xs[0,i], ys[0,i], color=colors[i], label='initial {}'.format(i), marker='o')                # Large point at the initial state
            ax1.plot(  xs[-1,i],ys[-1,i], color=colors[i], label='final {}'.format(i),   marker='^')      # Large point at the final state
                
        if show_arrows:
            for step in range(self.n_time_steps):   #Add arrows
                #if not self.donelist[step*self.duration,0]:                                   #(Mask) arrows to be shown
                plt.arrow(self.kicks[step,1], self.kicks[step,0], self.kicks[step,2], self.kicks[step,3], head_width=0.03, width=0.001 )
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        ax1.grid(True)
        plt.title('Mean return:{:.1f}'.format(np.mean(returns[:,0])))
        #ax1.legend()
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
        """
        if self.usemasks:
            ellipse = Ellipse((0, 0), width=np.arccos(1-np.exp(-4)) * 2, height=np.sqrt(2*np.exp(-4)) * 2, edgecolor='blue', facecolor='none')
            ax1.add_patch(ellipse)
        """
        if use_insets:
            left, bottom, width, height = [0.15, 0.65, 0.2, 0.2]        #Subplot with theta(t)
            ax2 = fig.add_axes([left, bottom, width, height])
            for i in trajectory_numbers_to_show:
                ax2.scatter(self.t,(theta[:,i]+np.pi)%(2*np.pi)-np.pi, s=1.0, color=colors[i],label='$theta$ {}'.format(i))
            ax2.set_xlabel('$t$')
            ax2.set_ylabel('$theta$')
            ax2.set_ylim([-np.pi, np.pi])


            left, bottom, width, height = [0.65, 0.65, 0.2, 0.2]        #Subplot with r(t)
            ax3 = fig.add_axes([left, bottom, width, height])
            for i in trajectory_numbers_to_show:
                ax3.scatter(self.t,np.abs(r[:,i]), s=1.0, color=colors[i],label='r {}'.format(i))
            ax3.set_xlabel('$t$')
            ax3.set_ylabel('$r$')
            ax3.set_ylim([0, 20])


            left, bottom, width, height = [0.65, 0.15, 0.2, 0.2]        #Subplot with return(timestep)
            ax4 = fig.add_axes([left, bottom, width, height])
            timestep_array = np.arange(self.n_time_steps)
            for i in trajectory_numbers_to_show:
                ax4.scatter(timestep_array, returns[i,:], s=1.0,color=colors[i],label='return {}'.format(i))
            ax4.set_xlabel('timestep')
            ax4.set_ylabel('reward')

            """
            left, bottom, width, height = [0.15, 0.15, 0.2, 0.2]        #Subplot with strategy
            ax5 = fig.add_axes([left, bottom, width, height])
            ax5.set_xlabel('$x$')
            ax5.set_ylabel('$p$')
            
            y = np.arange(-1, 1, 0.1)
            x = np.arange(-1.5, 1.5, 0.15)
                
            z = np.empty((20,20))
            z = probs[:,:,0]

            ax5.pcolormesh(x, y, z, shading='auto')
            
            if not (critic_preds is None):          #Show only if critic_preds are given as a parameter
                left, bottom, width, height = [0.40, 0.65, 0.2, 0.2]        #Subplot with critic values
                ax6 = fig.add_axes([left, bottom, width, height])

                ax6.set_xlabel('$\\theta$')
                ax6.set_ylabel('$p$')
                ax6.grid(True)
                

                y = np.arange(-10, 10, 1)
                x = np.arange(-np.pi, np.pi, np.pi/10)
                z = np.empty((20,20))
                z = critic_preds.squeeze()
                valuesplot = ax6.pcolormesh(x, y, z, shading='auto')
                plt.colorbar(valuesplot, ax=ax6)
                #ax6.colorbar()
            
            """
            left, bottom, width, height = [0.40, 0.15, 0.2, 0.2]        #Histogram of the returns
            ax7 = fig.add_axes([left, bottom, width, height])
            ax7.hist(returns[:,0],bins='auto')


        #Save plot
        file = '/epoch={}.png'.format(epoch)
        plt.savefig(directory+file)


        plt.close('all')

    def save_phase_portrait(self, label, probs):
        directory, epoch, returns = label
        phasefile = open(directory+'/{}_phase_portrait.txt'.format(epoch), 'wb')
        pickle.dump([self.t, self.sol, self.kicks, returns, probs],phasefile)
        phasefile.close()

    def Render_best(self, label, probs, critic_preds = None, use_insets = True, show_arrows = False):
        _, _, returns = label
        trajectory_numbers_to_show = np.argmax(returns[:,0])
        self.Render(label, probs, critic_preds = critic_preds, use_insets = use_insets, trajectory_numbers_to_show = [trajectory_numbers_to_show], show_arrows = show_arrows)