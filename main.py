"""
==================================
Inverted pendulum animation class
==================================

Adapted from the double pendulum problem animation.
https://matplotlib.org/examples/animation/double_pendulum_animated.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal as MVN
from scipy.interpolate import interp1d
import seaborn as sns
import pdb
import time


class EnvAnimate:
    '''
    Initialize Inverted Pendulum Animation Settings
    '''
    def __init__(self):       
        pass
        
    def load_random_test_trajectory(self,):
        # Random trajectory for example
        self.theta = np.linspace(-np.pi, np.pi, self.t.shape[0])
        self.u = np.zeros(self.t.shape[0])

        self.x1 = np.sin(self.theta)
        self.y1 = np.cos(self.theta)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2, -2, 2])

        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.8, '', transform=self.ax.transAxes)
        pass
    
    '''
    Provide new rollout trajectory (theta and control input) to reanimate
    '''
    def load_trajectory(self, theta, u):
        """
        Once a trajectory is loaded, you can run start() to see the animation
        ----------
        theta : 1D numpy.ndarray
            The angular position of your pendulum (rad) at each time step
        u : 1D numpy.ndarray
            The control input at each time step
            
        Returns
        -------
        None
        """
        self.theta = theta
        self.x1 = np.sin(theta)
        self.y1 = np.cos(theta)
        self.u = u
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2,-2, 2])
        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)
        pass
    
    # region: Animation
    # Feel free to edit (not necessarily)
    def init(self):
        self.line.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.time_text

    def _update(self, i):
        thisx = [0, self.x1[i]]
        thisy = [0, self.y1[i]] 
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (self.t[i], self.theta[i], self.u[i]))
        return self.line, self.time_text

    def start(self):
        print('Starting Animation')
        print()
        # Set up plot to call animate() function periodically
        self.ani = FuncAnimation(self.fig, self._update, frames=range(len(self.x1)), interval=25, blit=True, init_func=self.init, repeat=False)
        plt.show()
    # endregion: Animation


class InvertedPendulum(EnvAnimate):
    def __init__(self):
        EnvAnimate.__init__(self,)
        # Change this to match your discretization
        # Usually, you will need to load parameters including
        # constants: dt, vmax, umax, n1, n2, nu
        # parameters: a, b, σ, k, r, γ
        
        self.dt =tau
        self.t = np.arange(0.0, 100, self.dt)
        self.load_random_test_trajectory()
        pass
        
    
    def cost(self,X,u): # Cost Function
        return (1-np.exp(k*(np.cos(X[0])-1))+ (0.5*r*u*u) )
    
    def wrap(self, angles): # Angle Wrapping
        n= int(np.abs((angles/np.pi)))
        if(n%2==0):
            angles=angles-(n*(np.pi)*np.sign(angles))
        else:
            angles= angles-(n+1)*np.pi*(np.sign(angles))
        return angles
    
    def f_x_u(self,X,u): # Motion Model
        return np.vstack(( X[1], (a*np.sin(X[0]))-(b*X[1])+u))
    
    def loc1(self,x1,x2,len_x1,len_x2): # Just a auxilliary Function, not used currently
        l=(x2*len_x1)+(x1)
        return l

    def Build_MDP(self, x1,x2,U,cov1,state_space): # Function to build the MDP
        print('Building the MDP Model')
        
        len_x1=len(x1)
        len_x2=len(x2)
        nx=len_x1*len_x2 # Number of States
        nu=len(U) # Number of actions
        P=np.zeros((nx,nx,nu),dtype=np.float16) # Initialize transition Matrix
        L=np.zeros((nx,nu))

        # Build MDP
        e=0
        for theta in (x1):
            for vel in (x2):
                for k,action in enumerate(U):
                    curr_state=np.array([theta,vel]) # Get Current State
                    x_next= curr_state.reshape(2,1) +(self.f_x_u(curr_state,action)*tau) # Get the next state mean
                    # Trim the velocity, if it goes out of bound
                    if(x_next[1]>v_max): 
                        x_next[1]=v_max
                    if(x_next[1]<-v_max):
                        x_next[1]=-v_max
                    x_next[0]=self.wrap(x_next[0])  # Wrap the angles

                    P[e,:,k]=MVN.pdf(state_space,mean=x_next.T[0],cov=cov1) # Fit a gaussian on the next state mean
                    P[e,:,k]=P[e,:,k]/np.sum(P[e,:,k]) # Normalize and Store
                    L[ e,k]=(self.cost(curr_state,action)*tau) # Building the cost matrix
                e=e+1
        return P,L

    def value_iteration(self,P,L,x1,x2,gamma1):
        # Value Iteration
        print('Value Iteration in Progress ...')
        len_x1=len(x1)
        len_x2=len(x2)
        nx=len_x1*len_x2 # Number of States
        gamma=gamma1
        v=np.zeros((nx,1))
        H=np.zeros((nx,len(U)))

        itr=1
        while True:
            v_old=v
            for i in range(len(U)):
                H[:,i]=(L[:,i]+(gamma* (P[:,:,i]) @ v).reshape(1,nx))
                # sns.heatmap(P[:,:,i])
                # plt.plot(H[:,i])
                # plt.show()
                #  print(H)
            policy=np.argmin(H,axis=1) # Extract the Policy
            v=(np.min(H,axis=1)).reshape(nx,1) # update Value function
            
            # sns.heatmap(v.reshape(len_x2,len_x1),yticklabels=np.round(x2,2),xticklabels=np.round(x1,2))
            # plt.show()
    
            if(np.sum((v-v_old)**2) <0.000001): # Check if the value function converges
                break
            itr=itr+1
        print(itr)

        # sns.heatmap(v.reshape(len_x1,len_x2).T,yticklabels=np.round(x2,2),xticklabels=np.round(x1,2))
        # plt.show()

        V_x, VI_policy = v, policy
        return V_x, VI_policy,
    
    def policy_iteration(self,P,L,x1,x2,U,gamma):

        # Policy Iteration
        print('Policy Iteration in Progress ...')
        len_x1=len(x1)
        len_x2=len(x2)
        nx=len_x1*len_x2
        nu=len(U)
        policy=np.ones((nx,1))*np.round((nu+1)/2) # Define a random Policy
        PP= np.zeros((nx,nx))
        LL= np.zeros((nx,1))
        H=np.zeros((nx,nu))
        v=np.zeros((nx,1))

        itr=1
        while True:
            v_old=v
        
            for i in np.arange(nx):
                PP[i,:]=P[i,:,int(policy[i]) ]
                LL[i]=L[i,int(policy[i])]

            v=np.linalg.inv(np.eye(nx)-(gamma*(PP)))@LL # Get intermediate value of value fuction

            for j in range(len(U)):
                H[:,j]=(L[:,j]+(gamma*P[:,:,j]@ v).reshape(nx))

            policy=np.argmin(H,axis=1) # Extract Policy
            v=(np.min(H,axis=1)).reshape(nx,1) # Update the value Funtion
            # sns.heatmap(v.reshape(len_x1,len_x2).T,yticklabels=np.round(x2,2),xticklabels=np.round(x1,2))
            # plt.show()
            if(np.sum((v-v_old)**2) <0.00001): # Check if the value function converges
                break
                
            itr=itr+1
        print(itr)
        # sns.heatmap(v.reshape(len_x1,len_x2).T,linewidths=0.5,yticklabels=np.round(x2,2),xticklabels=np.round(x1,2))
        # plt.show()
        V_x, PI_policy = v, policy
        return V_x, PI_policy,
    
    def generate_trajectory(self, in_loc, init_state, policy, t,states,tau,P,n_states,U):
        print('Generating Trajectory')
        loc=in_loc  # Make a copy of initial state location on state space matrix
        angle=[]
        ctrl=[]
        while(t>0):  
            t-=tau
            theta =states[loc,0] # Get the angle corresponding to the state
            angle.append(theta) # Save to angle lisy

            u1= policy[loc] # Get the policy corresponding to the state
            ctrl.append(U[u1]) # Get the control input Magnitude
            
            # Get the next state space location using the Transiton Probability matrix and update the loc variable.
            loc=np.random.choice(range(n_states),1,p=(P[loc,:,u1]))[0] 
        
        # Interpolate to Continuous Space
        f=interp1d(range(len(ctrl)),ctrl,kind='cubic',fill_value="extrapolate")
        ctrl_cont=np.linspace(0,len(ctrl),1000)

        f2=interp1d(range(len(angle)),angle,kind='cubic',fill_value="extrapolate")
        angle_cont=np.linspace(0,len(angle),1000)
        
        theta,u= f2(angle_cont),f(ctrl_cont)
        return theta, u 
    

if __name__ == '__main__':

    #-----------Parameters----------------------

    tau=0.2 
    v_max= 3  # Maximum Value of Velocity
    u_max= 3  # Maximum Value of Control Input

    n_u=31 # Control Space Discreetization
    n1= 31  #Angle Space Discreetization
    n2= 31  #Velocity Space Discreetization

    # Pendulum Parameters
    k=1
    r=0.01 
    a=1 
    b=0.01 

    gamma=0.9 # Discount Factor
    cov=np.diag([0.01,0.01])*(tau) # Noise Covariance 

    init_state=np.array([-np.pi,0]) # Initialize Starting State
    
    inv_pendulum = InvertedPendulum()
    x1=np.linspace(-np.pi,np.pi,n1) # Theta
    x2=np.linspace(-v_max,v_max,n2) # velocity
    U=np.linspace(-u_max,u_max,n_u)
    
    len_x1=len(x1)
    len_x2=len(x2)

    # Initializaing State Space------------
    state_space=np.zeros(( (n1*n2),2))
    
    e=0
    for theta in x1:
        for vel in x2:
            state_space[e,0]=theta
            state_space[e,1]=vel
            e=e+1
    # -------------------------------------------

    # Finding nearest state to given Initial Position of pendulum
    E=np.sum((state_space-init_state)**2,axis=1) 
    in_loc=np.argmin(E)
    
    init_state=state_space[in_loc,:]

    P,L=inv_pendulum.Build_MDP(x1,x2,U,cov,state_space) # BUilding the MDP, Transition Probabilities
    
    VI_V, VI_policy = inv_pendulum.value_iteration(P,L,x1,x2,gamma) # Performing Value Iteration
    theta_vi, u_vi = inv_pendulum.generate_trajectory(in_loc,init_state,VI_policy,10,state_space,tau,P,(n1*n2),U) # Generating Trajectory
    inv_pendulum.load_trajectory(theta_vi, u_vi) # Animate
    inv_pendulum.start()

    PI_V, PI_policy = inv_pendulum.policy_iteration(P,L,x1,x2,U,gamma) # Performing Value Iteration
    theta_pi, u_pi = inv_pendulum.generate_trajectory(in_loc,init_state,PI_policy,10,state_space,tau,P,(n1*n2),U) # Generating Trajectory
    inv_pendulum.load_trajectory(theta_pi, u_pi) # Animate
    inv_pendulum.start()

    # Plotting all relevent Data
    plt.figure()
    plt.subplot(121)
    sns.heatmap(VI_V.reshape(len_x1,len_x2).T,yticklabels=np.round(x2,2),xticklabels=np.round(x1,2))   
    plt.xlabel('Angle')
    plt.ylabel('Angular Velocity')
    plt.title('Value Iteration')
    plt.subplot(122)
    sns.heatmap(PI_V.reshape(len_x1,len_x2).T,yticklabels=np.round(x2,2),xticklabels=np.round(x1,2))
    plt.xlabel('Angle')
    plt.ylabel('Angular Velocity')
    plt.title('Policy Iteration')
    plt.suptitle('tau: ' + str(tau)+ ' v_max: ' + str(v_max)+' u_max: '+ str(u_max)+' n_u: '+str(n_u)+' n_theta: '+str(n1)+' n_vel: '+str(n2)+' k: '+ str(k)+ ' r: '+str(r)+' a: '+str(a)+' b: '+str(b)+' Gamma: '+str(gamma)+'\n Cov:'+str(cov))
    plt.show()

    plt.figure()
    plt.subplot(121)
    sns.heatmap(U[VI_policy].reshape(len_x1,len_x2).T)   
    # plt.xlabel('Angle')
    # plt.ylabel('Angular Velocity')
    plt.title('Value Iteration (Policy Plot)')
    plt.subplot(122)
    sns.heatmap(U[PI_policy].reshape(len_x1,len_x2).T)
    # plt.xlabel('Angle')
    # plt.ylabel('Angular Velocity')
    plt.title('Policy Iteration (Policy Plot)')
    plt.suptitle('tau: ' + str(tau)+ ' v_max: ' + str(v_max)+' u_max: '+ str(u_max)+' n_u: '+str(n_u)+' n_theta: '+str(n1)+' n_vel: '+str(n2)+' k: '+ str(k)+ ' r: '+str(r)+' a: '+str(a)+' b: '+str(b)+' Gamma: '+str(gamma)+'\n Cov:'+str(cov))
    plt.show()

    plt.subplot(221)
    plt.plot(u_vi)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Control input')
    plt.title('Value Iteration')
    plt.subplot(223)
    plt.plot(theta_vi)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Angle')
    

    plt.subplot(222)
    plt.plot(u_pi)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Control input')
    plt.title('Policy Iteration')
    plt.subplot(224)
    plt.plot(theta_pi)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.suptitle('Angle and control input plot w.r.t.Time. Initial State: '+ str(init_state))
    plt.show()

        
    


