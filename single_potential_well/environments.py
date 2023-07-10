import numpy as np
import jax.numpy as jnp
import odes
from scipy.integrate import ode
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse




class flexible_env():
    
    def __init__(self, 
                N_MC=128,
                seed=0, 
                Fpower=0.1,
                t=10.,
                deltat = 0.1
                ):    #Initialize the environment.

        # defining states and actions
        self.N_states = 3  
        """
            The state should be:
            (x,p,t) in most settings;
            (t,x,p) in pg_flexible_time;
            (x,p, cos1(t), cos2(t)) if we give explicit information about the time-dependence to the agent
        """
        self.N_actions = 2
        self.actions = np.array([0, 1])     #0 - left, 1 - right
        self.Fpower = Fpower

        #initializing differential equation

        self.set_seed(seed)
        # Constants
        self.solver = ode(odes.cosh_var) # cosh_fixed has no time dependence
        self.solver.set_integrator('dop853', rtol=1e-3, atol=1e-6)            #Use the scipy ODE integrator (default:dop853)

        #self.x0 = -np.sqrt(2)/2*np.ones(N_MC)    # Initial state for double-well
        self.x0 = np.zeros(N_MC)    # Initial state for cosh potential
        self.p0 = np.zeros(N_MC)

        self.z0 = np.zeros((2,N_MC))
        self.z0[1,:] = self.x0
        self.z0[0,:] = self.p0

        self.N_MC = N_MC

        # Initial values
        self.t0 = 0
        self.t1 = t
        self.n_time_steps = int(t/deltat)+1
        # Time runs from t0 to t1, with resolution deltat
        self.t = np.linspace(self.t0, self.t1, self.n_time_steps)  

        #Array that records all trajectories for visualizations
        self.sol = np.empty((self.n_time_steps,2,N_MC))


        self.kicks = np.zeros((self.n_time_steps,4))   #Enable for visualizations

        self.reset()



    def step(self, action, done_before): #Performs one step in the environemnt.

        self.Fx = self.Fpower*(2*action-1)            #Turn the RL-determined action to force

        #""" Recording kicks is only required if we use arrows in the plot
        self.kicks[self.current_step,0] = self.sol[self.current_step-1,0,0]
        self.kicks[self.current_step,1] = self.sol[self.current_step-1,1,0]
        self.kicks[self.current_step,2] = self.Fx[0]/10
        self.kicks[self.current_step,3] = 0.
        #"""
        self.solver.set_f_params(self.Fx)
        if not self.solver.successful():
            raise Exception("Unsuccessful integration")
        
        self.solver.integrate(self.t[self.current_step])
        p, x = np.split(self.solver.y,2)        #x is an array of momentums, y is an array of angles
        self.sol[self.current_step,0,:] = p
        self.sol[self.current_step,1,:] = x
        
        #done_now = np.where(4*(x-np.sqrt(2)/2)**2 + p**2 < 0.05 , True, False)     #(Mask) Condition for finishing the trajectory
        done_now = np.where(np.abs(x) > 5 , True, False)     #(Mask) Condition for finishing the trajectory //cosh environment
        if self.current_step>1:
            done = np.logical_or(done_now, done_before)                         #(Mask) Determine if the trajectory has finished
        else:
            done = done_now            


        self.current_step += 1

        """
        self.state[:,0] = p
        self.state[:,1] = x
        #self.state[:,2] = self.t[self.current_step-1]*2*np.pi/0.5  # the oscillations of the potential have angular frequency 0.5
        self.state[:,2] = np.sin(0.5*self.t[self.current_step-1])  # double-well / cosh height
        self.state[:,3] = np.sin(np.sqrt(2)*self.t[self.current_step-1])  # cosh width
        """
        self.state[:,0] = self.t[self.current_step-1]   # Add the time to the state without explicit information about the time dependence of the environment
        self.state[:,1] = p
        self.state[:,2] = x

        #print(x)
        #exit()
        reward = np.abs(x)
        #print(reward)
        #exit()
        #print(np.shape(self.state))
        return self.state, reward, done

    def set_seed(self,seed=0):  #Sets the seed of the RNG.
        self.rng = np.random.default_rng(seed)

    def reset(self): #Resets the environment to its initial values.

        self.current_step = 1
        self.sol[0,:,:] = self.z0


        self.Fx=0.0

        self.solver.set_f_params(self.Fx)  #Solver with control
        self.solver.set_initial_value(self.z0.flatten(), self.t0)         # Set the initial value
        self.state = np.zeros((self.N_MC,self.N_states))    #3 when using sin and cos
        self.state[:,0] = self.z0[0,:]
        self.state[:,1] = self.z0[1,:]
        return self.state

    def change_initial_state(self, z0, multiple_initial_states = False):
        if multiple_initial_states:
            self.z0 = z0.copy()
        else:
            self.z0[0,:] = z0[0] #x0
            self.z0[1,:] = z0[1] #p0
        self.reset()

    def Render(self, label, probs, critic_preds = None, use_insets = True, trajectory_numbers_to_show = [0, 50], show_arrows = False):
        plt.rc('font', size=15)        #Resize font
        epoch, returns = label
        #plt.figsize=(19.2,10.8)
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


        for i in trajectory_numbers_to_show:
            ax1.scatter(self.sol[:,1,i],self.sol[:,0,i], s=1.0, color=colors[i], label='trajectory {}'.format(i))                   #Create a scatter plot of the trajectories shown.
            ax1.plot(self.sol[0,1,i],self.sol[0,0,i] , color=colors[i], label='initial {}'.format(i), marker='o')                # Large point at the initial state
            ax1.plot(self.sol[-1,1,i],self.sol[-1,0,i] , color=colors[i],label='final {}'.format(i), marker='^')      # Large point at the final state
        
        if show_arrows:
            for step in range(self.n_time_steps):   #Add arrows
                #if not self.donelist[step*self.duration,0]:                                   #(Mask) arrows to be shown
                plt.arrow(self.kicks[step,1], self.kicks[step,0], self.kicks[step,2], self.kicks[step,3], head_width=0.05, width=0.001 )
        plt.xlabel('$x$')
        plt.ylabel('$p$')
        ax1.grid(True)
        plt.title('Mean return:{:.1f}'.format(np.mean(returns[:,0])))
        #ax1.legend()
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        """
        if self.usemasks:
            ellipse = Ellipse((0, 0), width=np.arccos(1-np.exp(-4)) * 2, height=np.sqrt(2*np.exp(-4)) * 2, edgecolor='blue', facecolor='none')
            ax1.add_patch(ellipse)
        """
        if use_insets:
            left, bottom, width, height = [0.15, 0.65, 0.2, 0.2]        #Subplot with x(t)
            ax2 = fig.add_axes([left, bottom, width, height])
            for i in trajectory_numbers_to_show:
                ax2.scatter(self.t,self.sol[:,1,i], s=1.0, color=colors[i],label='theta {}'.format(i))
            ax2.set_xlabel('$t$')
            ax2.set_ylabel('$x$')
            ax2.set_ylim([-np.pi, np.pi])


            left, bottom, width, height = [0.65, 0.65, 0.2, 0.2]        #Subplot with p(t)
            ax3 = fig.add_axes([left, bottom, width, height])
            for i in trajectory_numbers_to_show:
                ax3.scatter(self.t,self.sol[:,0,i], s=1.0, color=colors[i],label='p {}'.format(i))
            ax3.set_xlabel('$t$')
            ax3.set_ylabel('$p$')
            ax3.set_ylim([-3, 3])


            left, bottom, width, height = [0.65, 0.15, 0.2, 0.2]        #Subplot with return(timestep)
            ax4 = fig.add_axes([left, bottom, width, height])
            timestep_array = np.arange(self.n_time_steps-1)
            for i in trajectory_numbers_to_show:
                ax4.scatter(timestep_array, returns[i,:], s=1.0,color=colors[i],label='return {}'.format(i))
            ax4.set_xlabel('timestep')
            ax4.set_ylabel('return')


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
            

            left, bottom, width, height = [0.40, 0.15, 0.2, 0.2]        #Histogram of the returns
            ax7 = fig.add_axes([left, bottom, width, height])
            ax7.hist(returns[:,0],bins='auto')


        #Save plot
        plt.savefig('phase_portrait_{}'.format(epoch))


        plt.close('all')



    def Render_best(self, label, probs, critic_preds = None, use_insets = True, show_arrows = False):
        _, _, returns = label
        trajectory_numbers_to_show = np.argmax(returns[:,0])
        self.Render(label, probs, critic_preds = critic_preds, use_insets = use_insets, trajectory_numbers_to_show = [trajectory_numbers_to_show], show_arrows = show_arrows)


class var_env():
    
    def __init__(self, 
                N_MC=128,
                seed=0, 
                Fpower=0.1,
                t=10.,
                deltat = 0.1
                ):    #Initialize the environment.

        # defining states and actions
        self.N_states = 4
        """
            The state should be:
            (x,p,t) in most settings;
            (t,x,p) in pg_flexible_time;
            (x,p, cos1(t), cos2(t)) if we give explicit information about the time-dependence to the agent
        """
        self.N_actions = 2
        self.actions = np.array([0, 1])     #0 - left, 1 - right
        self.Fpower = Fpower

        #initializing differential equation

        self.set_seed(seed)
        # Constants
        self.solver = ode(odes.cosh_var) # cosh_fixed has no time dependence
        self.solver.set_integrator('dop853', rtol=1e-3, atol=1e-6)            #Use the scipy ODE integrator (default:dop853)

        #self.x0 = -np.sqrt(2)/2*np.ones(N_MC)    # Initial state for double-well
        self.x0 = np.zeros(N_MC)    # Initial state for cosh potential
        self.p0 = np.zeros(N_MC)

        self.z0 = np.zeros((2,N_MC))
        self.z0[1,:] = self.x0
        self.z0[0,:] = self.p0

        self.N_MC = N_MC

        # Initial values
        self.t0 = 0
        self.t1 = t
        self.n_time_steps = int(t/deltat)+1
        # Time runs from t0 to t1, with resolution deltat
        self.t = np.linspace(self.t0, self.t1, self.n_time_steps)  

        #Array that records all trajectories for visualizations
        self.sol = np.empty((self.n_time_steps,2,N_MC))


        self.kicks = np.zeros((self.n_time_steps,4))   #Enable for visualizations

        self.reset()



    def step(self, action, done_before): #Performs one step in the environemnt.

        self.Fx = self.Fpower*(2*action-1)            #Turn the RL-determined action to force

        #""" Recording kicks is only required if we use arrows in the plot
        self.kicks[self.current_step,0] = self.sol[self.current_step-1,0,0]
        self.kicks[self.current_step,1] = self.sol[self.current_step-1,1,0]
        self.kicks[self.current_step,2] = self.Fx[0]/10
        self.kicks[self.current_step,3] = 0.
        #"""
        self.solver.set_f_params(self.Fx)
        if not self.solver.successful():
            raise Exception("Unsuccessful integration")
        
        self.solver.integrate(self.t[self.current_step])
        p, x = np.split(self.solver.y,2)        #x is an array of momentums, y is an array of angles
        self.sol[self.current_step,0,:] = p
        self.sol[self.current_step,1,:] = x
        
        #done_now = np.where(4*(x-np.sqrt(2)/2)**2 + p**2 < 0.05 , True, False)     #(Mask) Condition for finishing the trajectory
        done_now = np.where(np.abs(x) > 5 , True, False)     #(Mask) Condition for finishing the trajectory //cosh environment
        if self.current_step>1:
            done = np.logical_or(done_now, done_before)                         #(Mask) Determine if the trajectory has finished
        else:
            done = done_now            


        self.current_step += 1

        #"""
        self.state[:,0] = p
        self.state[:,1] = x
        #self.state[:,2] = self.t[self.current_step-1]*2*np.pi/0.5  # the oscillations of the potential have angular frequency 0.5
        self.state[:,2] = np.sin(0.5*self.t[self.current_step-1])  # double-well / cosh height
        self.state[:,3] = np.sin(np.sqrt(2)*self.t[self.current_step-1])  # cosh width
        """
        self.state[:,0] = self.t[self.current_step-1]   # Add the time to the state without explicit information about the time dependence of the environment
        self.state[:,1] = p
        self.state[:,2] = x
        """
        #print(x)
        #exit()
        reward = np.abs(x)
        #print(reward)
        #exit()
        #print(np.shape(self.state))
        return self.state, reward, done

    def set_seed(self,seed=0):  #Sets the seed of the RNG.
        self.rng = np.random.default_rng(seed)

    def reset(self): #Resets the environment to its initial values.

        self.current_step = 1
        self.sol[0,:,:] = self.z0


        self.Fx=0.0

        self.solver.set_f_params(self.Fx)  #Solver with control
        self.solver.set_initial_value(self.z0.flatten(), self.t0)         # Set the initial value
        self.state = np.zeros((self.N_MC,self.N_states))    #3 when using sin and cos
        self.state[:,0] = self.z0[0,:]
        self.state[:,1] = self.z0[1,:]
        return self.state

    def change_initial_state(self, z0, multiple_initial_states = False):
        if multiple_initial_states:
            self.z0 = z0.copy()
        else:
            self.z0[0,:] = z0[0] #x0
            self.z0[1,:] = z0[1] #p0
        self.reset()

    def Render(self, label, probs, critic_preds = None, use_insets = True, trajectory_numbers_to_show = [0, 50], show_arrows = False):
        plt.rc('font', size=15)        #Resize font
        epoch, returns = label
        #plt.figsize=(19.2,10.8)
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


        for i in trajectory_numbers_to_show:
            ax1.scatter(self.sol[:,1,i],self.sol[:,0,i], s=1.0, color=colors[i], label='trajectory {}'.format(i))                   #Create a scatter plot of the trajectories shown.
            ax1.plot(self.sol[0,1,i],self.sol[0,0,i] , color=colors[i], label='initial {}'.format(i), marker='o')                # Large point at the initial state
            ax1.plot(self.sol[-1,1,i],self.sol[-1,0,i] , color=colors[i],label='final {}'.format(i), marker='^')      # Large point at the final state
        
        if show_arrows:
            for step in range(self.n_time_steps):   #Add arrows
                #if not self.donelist[step*self.duration,0]:                                   #(Mask) arrows to be shown
                plt.arrow(self.kicks[step,1], self.kicks[step,0], self.kicks[step,2], self.kicks[step,3], head_width=0.05, width=0.001 )
        plt.xlabel('$x$')
        plt.ylabel('$p$')
        ax1.grid(True)
        plt.title('Mean return:{:.1f}'.format(np.mean(returns[:,0])))
        #ax1.legend()
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        """
        if self.usemasks:
            ellipse = Ellipse((0, 0), width=np.arccos(1-np.exp(-4)) * 2, height=np.sqrt(2*np.exp(-4)) * 2, edgecolor='blue', facecolor='none')
            ax1.add_patch(ellipse)
        """
        if use_insets:
            left, bottom, width, height = [0.15, 0.65, 0.2, 0.2]        #Subplot with x(t)
            ax2 = fig.add_axes([left, bottom, width, height])
            for i in trajectory_numbers_to_show:
                ax2.scatter(self.t,self.sol[:,1,i], s=1.0, color=colors[i],label='theta {}'.format(i))
            ax2.set_xlabel('$t$')
            ax2.set_ylabel('$x$')
            ax2.set_ylim([-np.pi, np.pi])


            left, bottom, width, height = [0.65, 0.65, 0.2, 0.2]        #Subplot with p(t)
            ax3 = fig.add_axes([left, bottom, width, height])
            for i in trajectory_numbers_to_show:
                ax3.scatter(self.t,self.sol[:,0,i], s=1.0, color=colors[i],label='p {}'.format(i))
            ax3.set_xlabel('$t$')
            ax3.set_ylabel('$p$')
            ax3.set_ylim([-3, 3])


            left, bottom, width, height = [0.65, 0.15, 0.2, 0.2]        #Subplot with return(timestep)
            ax4 = fig.add_axes([left, bottom, width, height])
            timestep_array = np.arange(self.n_time_steps-1)
            for i in trajectory_numbers_to_show:
                ax4.scatter(timestep_array, returns[i,:], s=1.0,color=colors[i],label='return {}'.format(i))
            ax4.set_xlabel('timestep')
            ax4.set_ylabel('return')


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
            

            left, bottom, width, height = [0.40, 0.15, 0.2, 0.2]        #Histogram of the returns
            ax7 = fig.add_axes([left, bottom, width, height])
            ax7.hist(returns[:,0],bins='auto')


        #Save plot
        plt.savefig('phase_portrait_{}.png'.format(epoch))


        plt.close('all')



    def Render_best(self, label, probs, critic_preds = None, use_insets = True, show_arrows = False):
        _, _, returns = label
        trajectory_numbers_to_show = np.argmax(returns[:,0])
        self.Render(label, probs, critic_preds = critic_preds, use_insets = use_insets, trajectory_numbers_to_show = [trajectory_numbers_to_show], show_arrows = show_arrows)




class pseudovar_env():
    
    def __init__(self, 
                N_MC=128,
                seed=0, 
                Fpower=0.1,
                t=10.,
                deltat = 0.1
                ):    #Initialize the environment.

        # defining states and actions
        self.N_states = 2   # (x,p)
        self.N_actions = 2
        self.actions = np.array([0, 1])     #0 - left, 1 - right
        self.Fpower = Fpower

        #initializing differential equation

        self.set_seed(seed)
        # Constants
        self.solver = ode(odes.cosh_var) # cosh_fixed has no time dependence
        self.solver.set_integrator('dop853', rtol=1e-3, atol=1e-6)            #Use the scipy ODE integrator (default:dop853)

        #self.x0 = -np.sqrt(2)/2*np.ones(N_MC)    # Initial state for double-well
        self.x0 = np.zeros(N_MC)    # Initial state for cosh potential
        self.p0 = np.zeros(N_MC)

        self.z0 = np.zeros((2,N_MC))
        self.z0[1,:] = self.x0
        self.z0[0,:] = self.p0

        self.N_MC = N_MC

        # Initial values
        self.t0 = 0
        self.t1 = t
        self.n_time_steps = int(t/deltat)+1
        # Time runs from t0 to t1, with resolution deltat
        self.t = np.linspace(self.t0, self.t1, self.n_time_steps)  

        #Array that records all trajectories for visualizations
        self.sol = np.empty((self.n_time_steps,2,N_MC))


        self.kicks = np.zeros((self.n_time_steps,4))   #Enable for visualizations

        self.reset()



    def step(self, action, done_before): #Performs one step in the environemnt.

        self.Fx = self.Fpower*(2*action-1)            #Turn the RL-determined action to force

        #""" Recording kicks is only required if we use arrows in the plot
        self.kicks[self.current_step,0] = self.sol[self.current_step-1,0,0]
        self.kicks[self.current_step,1] = self.sol[self.current_step-1,1,0]
        self.kicks[self.current_step,2] = self.Fx[0]/10
        self.kicks[self.current_step,3] = 0.
        #"""
        self.solver.set_f_params(self.Fx)
        if not self.solver.successful():
            raise Exception("Unsuccessful integration")
        
        self.solver.integrate(self.t[self.current_step])
        p, x = np.split(self.solver.y,2)        #x is an array of momentums, y is an array of angles
        self.sol[self.current_step,0,:] = p
        self.sol[self.current_step,1,:] = x
        
        #done_now = np.where(4*(x-np.sqrt(2)/2)**2 + p**2 < 0.05 , True, False)     #(Mask) Condition for finishing the trajectory
        done_now = np.where(np.abs(x) > 5 , True, False)     #(Mask) Condition for finishing the trajectory //cosh environment
        if self.current_step>1:
            done = np.logical_or(done_now, done_before)                         #(Mask) Determine if the trajectory has finished
        else:
            done = done_now            



        self.current_step += 1

        self.state[:,0] = p
        self.state[:,1] = x
        #print(x)
        #exit()
        reward = np.abs(x)
        #print(reward)
        #exit()
        return self.state, reward, done

    def set_seed(self,seed=0):  #Sets the seed of the RNG.
        self.rng = np.random.default_rng(seed)

    def reset(self): #Resets the environment to its initial values.

        self.current_step = 1
        self.sol[0,:,:] = self.z0


        self.Fx=0.0

        self.solver.set_f_params(self.Fx)  #Solver with control
        self.solver.set_initial_value(self.z0.flatten(), self.t0)         # Set the initial value
        self.state = np.zeros((self.N_MC,self.N_states))    #3 when using sin and cos
        self.state[:,0] = self.z0[0,:]
        self.state[:,1] = self.z0[1,:]
        return self.state


    def change_initial_state(self, z0, multiple_initial_states = False):
        if multiple_initial_states:
            self.z0 = z0.copy()
        else:
            self.z0[0,:] = z0[0] #x0
            self.z0[1,:] = z0[1] #p0
        self.reset()

    def Render(self, label, probs, critic_preds = None, use_insets = True, trajectory_numbers_to_show = [0, 50], show_arrows = False):
        plt.rc('font', size=15)        #Resize font
        epoch, returns = label
        #plt.figsize=(19.2,10.8)
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


        for i in trajectory_numbers_to_show:
            ax1.scatter(self.sol[:,1,i],self.sol[:,0,i], s=1.0, color=colors[i], label='trajectory {}'.format(i))                   #Create a scatter plot of the trajectories shown.
            ax1.plot(self.sol[0,1,i],self.sol[0,0,i] , color=colors[i], label='initial {}'.format(i), marker='o')                # Large point at the initial state
            ax1.plot(self.sol[-1,1,i],self.sol[-1,0,i] , color=colors[i],label='final {}'.format(i), marker='^')      # Large point at the final state
        
        if show_arrows:
            for step in range(self.n_time_steps):   #Add arrows
                #if not self.donelist[step*self.duration,0]:                                   #(Mask) arrows to be shown
                plt.arrow(self.kicks[step,1], self.kicks[step,0], self.kicks[step,2], self.kicks[step,3], head_width=0.03, width=0.001 )
        plt.xlabel('$x$')
        plt.ylabel('$p$')
        ax1.grid(True)
        plt.title('Mean return:{:.1f}'.format(np.mean(returns[:,0])))
        #ax1.legend()
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        """
        if self.usemasks:
            ellipse = Ellipse((0, 0), width=np.arccos(1-np.exp(-4)) * 2, height=np.sqrt(2*np.exp(-4)) * 2, edgecolor='blue', facecolor='none')
            ax1.add_patch(ellipse)
        """
        if use_insets:
            left, bottom, width, height = [0.15, 0.65, 0.2, 0.2]        #Subplot with x(t)
            ax2 = fig.add_axes([left, bottom, width, height])
            for i in trajectory_numbers_to_show:
                ax2.scatter(self.t,self.sol[:,1,i], s=1.0, color=colors[i],label='theta {}'.format(i))
            ax2.set_xlabel('$t$')
            ax2.set_ylabel('$x$')
            ax2.set_ylim([-np.pi, np.pi])


            left, bottom, width, height = [0.65, 0.65, 0.2, 0.2]        #Subplot with p(t)
            ax3 = fig.add_axes([left, bottom, width, height])
            for i in trajectory_numbers_to_show:
                ax3.scatter(self.t,self.sol[:,0,i], s=1.0, color=colors[i],label='p {}'.format(i))
            ax3.set_xlabel('$t$')
            ax3.set_ylabel('$p$')
            ax3.set_ylim([-3, 3])


            left, bottom, width, height = [0.65, 0.15, 0.2, 0.2]        #Subplot with return(timestep)
            ax4 = fig.add_axes([left, bottom, width, height])
            timestep_array = np.arange(self.n_time_steps-1)
            for i in trajectory_numbers_to_show:
                ax4.scatter(timestep_array, returns[i,:], s=1.0,color=colors[i],label='return {}'.format(i))
            ax4.set_xlabel('timestep')
            ax4.set_ylabel('return')


            left, bottom, width, height = [0.15, 0.15, 0.2, 0.2]        #Subplot with strategy
            ax5 = fig.add_axes([left, bottom, width, height])
            ax5.set_xlabel('$x$')
            ax5.set_ylabel('$p$')
            
            y = np.arange(-1, 1, 0.1)
            x = np.arange(-1.5, 1.5, 0.15)
                
            z = np.empty((20,20))
            z = probs[:,:,0]

            cmesh = ax5.pcolormesh(x, y, z, shading='auto', vmin=0, vmax = 1, )
            plt.colorbar(cmesh)

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
            

            left, bottom, width, height = [0.40, 0.15, 0.2, 0.2]        #Histogram of the returns
            ax7 = fig.add_axes([left, bottom, width, height])
            ax7.hist(returns[:,0],bins='auto')


        #Save plot
        file = '/epoch={}.png'.format(epoch)
        plt.savefig('phase_portrait_{}.png'.format(epoch))


        plt.close('all')


    def Render_best(self, label, probs, critic_preds = None, use_insets = True, show_arrows = False):
        _, _, returns = label
        trajectory_numbers_to_show = np.argmax(returns[:,0])
        self.Render(label, probs, critic_preds = critic_preds, use_insets = use_insets, trajectory_numbers_to_show = [trajectory_numbers_to_show], show_arrows = show_arrows)



class fixed_env():
    
    def __init__(self, 
                N_MC=128,
                seed=0, 
                Fpower=0.1,
                t=10.,
                deltat = 0.1
                ):    #Initialize the environment.

        # defining states and actions
        self.N_states = 2   # (x,p)
        self.N_actions = 2
        self.actions = np.array([0, 1])     #0 - left, 1 - right
        self.Fpower = Fpower

        #initializing differential equation

        self.set_seed(seed)
        # Constants
        self.solver = ode(odes.cosh_fixed) # cosh_fixed has no time dependence
        self.solver.set_integrator('dop853', rtol=1e-3, atol=1e-6)            #Use the scipy ODE integrator (default:dop853)

        #self.x0 = -np.sqrt(2)/2*np.ones(N_MC)    # Initial state for double-well
        self.x0 = np.zeros(N_MC)    # Initial state for cosh potential
        self.p0 = np.zeros(N_MC)

        self.z0 = np.zeros((2,N_MC))
        self.z0[1,:] = self.x0
        self.z0[0,:] = self.p0

        self.N_MC = N_MC

        # Initial values
        self.t0 = 0
        self.t1 = t
        self.n_time_steps = int(t/deltat)+1
        # Time runs from t0 to t1, with resolution deltat
        self.t = np.linspace(self.t0, self.t1, self.n_time_steps)  

        #Array that records all trajectories for visualizations
        self.sol = np.empty((self.n_time_steps,2,N_MC))


        self.kicks = np.zeros((self.n_time_steps,4))   #Enable for visualizations

        self.reset()



    def step(self, action, done_before): #Performs one step in the environemnt.

        self.Fx = self.Fpower*(2*action-1)            #Turn the RL-determined action to force

        #""" Recording kicks is only required if we use arrows in the plot
        self.kicks[self.current_step,0] = self.sol[self.current_step-1,0,0]
        self.kicks[self.current_step,1] = self.sol[self.current_step-1,1,0]
        self.kicks[self.current_step,2] = self.Fx[0]/10
        self.kicks[self.current_step,3] = 0.
        #"""
        self.solver.set_f_params(self.Fx)
        if not self.solver.successful():
            raise Exception("Unsuccessful integration")
        
        self.solver.integrate(self.t[self.current_step])
        p, x = np.split(self.solver.y,2)        #x is an array of momentums, y is an array of angles
        self.sol[self.current_step,0,:] = p
        self.sol[self.current_step,1,:] = x
        
        #done_now = np.where(4*(x-np.sqrt(2)/2)**2 + p**2 < 0.05 , True, False)     #(Mask) Condition for finishing the trajectory
        done_now = np.where(np.abs(x) > 5 , True, False)     #(Mask) Condition for finishing the trajectory //cosh environment
        if self.current_step>1:
            done = np.logical_or(done_now, done_before)                         #(Mask) Determine if the trajectory has finished
        else:
            done = done_now            



        self.current_step += 1

        self.state[:,0] = p
        self.state[:,1] = x
        #print(x)
        #exit()
        reward = np.abs(x)
        #print(reward)
        #exit()
        return self.state, reward, done

    def set_seed(self,seed=0):  #Sets the seed of the RNG.
        self.rng = np.random.default_rng(seed)

    def reset(self): #Resets the environment to its initial values.

        self.current_step = 1
        self.sol[0,:,:] = self.z0


        self.Fx=0.0

        self.solver.set_f_params(self.Fx)  #Solver with control
        self.solver.set_initial_value(self.z0.flatten(), self.t0)         # Set the initial value
        self.state = np.zeros((self.N_MC,self.N_states))    #3 when using sin and cos
        self.state[:,0] = self.z0[0,:]
        self.state[:,1] = self.z0[1,:]
        return self.state


    def change_initial_state(self, z0, multiple_initial_states = False):
        if multiple_initial_states:
            self.z0 = z0.copy()
        else:
            self.z0[0,:] = z0[0] #x0
            self.z0[1,:] = z0[1] #p0
        self.reset()

    def Render(self, label, probs, critic_preds = None, use_insets = True, trajectory_numbers_to_show = [0, 50], show_arrows = False):
        plt.rc('font', size=15)        #Resize font
        epoch, returns = label
        #plt.figsize=(19.2,10.8)
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


        for i in trajectory_numbers_to_show:
            ax1.scatter(self.sol[:,1,i],self.sol[:,0,i], s=1.0, color=colors[i], label='trajectory {}'.format(i))                   #Create a scatter plot of the trajectories shown.
            ax1.plot(self.sol[0,1,i],self.sol[0,0,i] , color=colors[i], label='initial {}'.format(i), marker='o')                # Large point at the initial state
            ax1.plot(self.sol[-1,1,i],self.sol[-1,0,i] , color=colors[i],label='final {}'.format(i), marker='^')      # Large point at the final state
        
        if show_arrows:
            for step in range(self.n_time_steps):   #Add arrows
                #if not self.donelist[step*self.duration,0]:                                   #(Mask) arrows to be shown
                plt.arrow(self.kicks[step,1], self.kicks[step,0], self.kicks[step,2], self.kicks[step,3], head_width=0.03, width=0.001 )
        plt.xlabel('$x$')
        plt.ylabel('$p$')
        ax1.grid(True)
        plt.title('Mean return:{:.1f}'.format(np.mean(returns[:,0])))
        #ax1.legend()
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        """
        if self.usemasks:
            ellipse = Ellipse((0, 0), width=np.arccos(1-np.exp(-4)) * 2, height=np.sqrt(2*np.exp(-4)) * 2, edgecolor='blue', facecolor='none')
            ax1.add_patch(ellipse)
        """
        if use_insets:
            left, bottom, width, height = [0.15, 0.65, 0.2, 0.2]        #Subplot with x(t)
            ax2 = fig.add_axes([left, bottom, width, height])
            for i in trajectory_numbers_to_show:
                ax2.scatter(self.t,self.sol[:,1,i], s=1.0, color=colors[i],label='theta {}'.format(i))
            ax2.set_xlabel('$t$')
            ax2.set_ylabel('$x$')
            ax2.set_ylim([-np.pi, np.pi])


            left, bottom, width, height = [0.65, 0.65, 0.2, 0.2]        #Subplot with p(t)
            ax3 = fig.add_axes([left, bottom, width, height])
            for i in trajectory_numbers_to_show:
                ax3.scatter(self.t,self.sol[:,0,i], s=1.0, color=colors[i],label='p {}'.format(i))
            ax3.set_xlabel('$t$')
            ax3.set_ylabel('$p$')
            ax3.set_ylim([-3, 3])


            left, bottom, width, height = [0.65, 0.15, 0.2, 0.2]        #Subplot with return(timestep)
            ax4 = fig.add_axes([left, bottom, width, height])
            timestep_array = np.arange(self.n_time_steps-1)
            for i in trajectory_numbers_to_show:
                ax4.scatter(timestep_array, returns[i,:], s=1.0,color=colors[i],label='return {}'.format(i))
            ax4.set_xlabel('timestep')
            ax4.set_ylabel('return')


            left, bottom, width, height = [0.15, 0.15, 0.2, 0.2]        #Subplot with strategy
            ax5 = fig.add_axes([left, bottom, width, height])
            ax5.set_xlabel('$x$')
            ax5.set_ylabel('$p$')
            
            y = np.arange(-1, 1, 0.1)
            x = np.arange(-1.5, 1.5, 0.15)
                
            z = np.empty((20,20))
            z = probs[:,:,0]

            cmesh = ax5.pcolormesh(x, y, z, shading='auto', vmin=0, vmax = 1, )
            plt.colorbar(cmesh)

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
            

            left, bottom, width, height = [0.40, 0.15, 0.2, 0.2]        #Histogram of the returns
            ax7 = fig.add_axes([left, bottom, width, height])
            ax7.hist(returns[:,0],bins='auto')


        #Save plot
        plt.savefig('phase_portrait_{}.png'.format(epoch))


        plt.close('all')


    def Render_best(self, label, probs, critic_preds = None, use_insets = True, show_arrows = False):
        _, _, returns = label
        trajectory_numbers_to_show = np.argmax(returns[:,0])
        self.Render(label, probs, critic_preds = critic_preds, use_insets = use_insets, trajectory_numbers_to_show = [trajectory_numbers_to_show], show_arrows = show_arrows)