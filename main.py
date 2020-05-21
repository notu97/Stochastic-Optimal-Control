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

#-----------Parameters----------------------

tau=0.1
# n2=8 # x2 discreet

n_u=10 # Num of controlls workred: 10

v_max=10
u_max=10
goal=np.zeros((0,0))

n1=151 # x1 discreet angle worked: 101
n2=41 # round ((tau*(2*v_max))/((2*np.pi)/n1))+1 # x2 discreet. worked= 31
k=0.1 # damping ratio
r=1 # 
a=1 # g/L
b=0.1 # k/m

gamma=0.1
sigma=np.array([[1],[1]])
dist=0.001
init_state=np.array([np.pi,0])

# FLAGS to select the type of algorithm. Set the choice to 1 and the other to 0
VI=0 # Value Iteration (VI)
PI=1 # Policy Iteration (PI)

#---------------------------------

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
        
        self.dt = 0.1
        self.t = np.arange(0.0, 100, self.dt)
        self.load_random_test_trajectory()
        pass
        
    # TODO: Implement your own VI and PI algorithms
    # Feel free to edit/delete these functions
    # def l_xu(self, x1, x2, u):
    #     # Stage cost
    #     return
    
    # def f_x_u(self,X,u):
    #     return np.vstack(( X[1], (a*np.sin(wrapp(X[0]) ))-(b*X[1])+u))
    # def f_xu(self, x1, x2, u):
    #     # Motion model
    #     return
    
    def cost(self,X,u):
        return (1-np.exp(k*(np.cos(X[0]-1)))+ (0.5*r*u*u) )

    def wrapp(self, angles):
        n=int(abs(angles/ (np.pi) ))+1
        angle=(-(n*np.pi - angles) * (angles >= np.pi)) + ((n*np.pi+angles)*(angles <= -np.pi))
        return angle 
    
    def f_x_u(self,X,u):
        return np.vstack(( X[1], (a*np.sin(self.wrapp(X[0]) ))-(b*X[1])+u))
    
    def loc1(self,x1,x2,len_x1,len_x2):
        l=(x2*len_x1)+(x1)
        return l

    def Build_MDP(self, x1,x2,U):
        print('Building the MDP Model')
        len_x1=len(x1)
        len_x2=len(x2)
        nx=len_x1*len_x2 # Number of States
        nu=len(U) # Number of actions
        P=np.zeros((nx,nx,nu), dtype=np.float16)
        L=np.zeros((nx,nu))

        # Build MDP
        e=0
        for i,x_2 in enumerate(x2):
            for j,x_1 in enumerate(x1):
                # print('State',[x_1,x_2])
                for k,u in enumerate(U):
                    x_next= np.array([[x_1],[x_2]])+(self.f_x_u([x_1,x_2],u)*tau)           
                    gauss=MVN(x_next.T[0],cov) # add noise
                    vals=gauss.pdf(states)
                    ind=np.where(vals>0.001)
                    temp=np.zeros(np.shape(vals))

                    temp[ind[0],ind[1]]=((vals[ind[0],ind[1]])/(np.sum(vals[ind[0],ind[1]])))            
                    P[e,:,k]=temp.flatten() # Transition matrix
                    L[ int(self.loc1(j,i,len_x1,len_x2)),k]=self.cost([x_1,x_2],u) 
                e=e+1
                # print('-----------------------------------------------------------------')          
        goal_ind=np.where( (np.logical_and(( np.round(states[:,:,0])==0),( np.round(states[:,:,1])==0)))==True) 
        # print(goal_ind) 
        goal_loc=self.loc1(goal_ind[1][0],goal_ind[0][0],len_x1,len_x2)
        L[goal_loc]-=1

        return P,L

    def value_iteration(self,P,L,x1,x2,gamma1):
        # Value Iteration
        print('Value Iteration in Progress ...')
        len_x1=len(x1)
        len_x2=len(x2)
        nx=len_x1*len_x2 # Number of States
        # nu=len(U) # Number of actions

        # a,b=np.shape(L)
        # isgoal=int(nx/2) #loc()  #int(sz/2) #(((int(b/2))*b))+int(a/2)
        gamma=gamma1
        v=np.zeros((nx,1))
        # policy=np.ones((nx,1))*( round((len(U)+1)/2) )
        H=np.zeros((nx,len(U)))

        itr=1
        while True:
            # print(itr)
            #     print(v)
            v_old=v
            for i in range(len(U)):
                H[:,i]=(L[:,i]+(gamma*P[:,:,i]@ v).reshape(nx))
            #     print(H)
            policy=np.argmin(H,axis=1)
            v=(np.min(H,axis=1)).reshape(nx,1)
            #     v[isgoal]=0
            # print(np.max(np.abs(v-v_old)))
            if(np.max(np.abs(v-v_old))<1e-100):
                break
            itr=itr+1

        plt.figure()
        plt.imshow(v.reshape(len_x2,len_x1))
        plt.show()
        V_x, VI_policy = v, policy
        return V_x, VI_policy,
    
    def policy_iteration(self,P,L,x1,x2,U,gamma):

        # Policy Iteration
        print('Policy Iteration in Progress ...')
        len_x1=len(x1)
        len_x2=len(x2)
        nx=len_x1*len_x2
        nu=len(U)
        policy=np.ones((nx,1))*np.round((nu+1)/2)
        PP= np.zeros((nx,nx))
        LL= np.zeros((nx,1))
        H=np.zeros((nx,nu))
        v=np.zeros((nx,1))

        itr=1
        while True:
            v_old=v
            # print(itr)
            for i in np.arange(nx):
        #         print('itr: ',i)
        #         print(int(policy[i]))
                PP[i,:]=P[i,:,int(policy[i]) ]
                LL[i]=L[i,int(policy[i])]

            v=np.linalg.inv(np.eye(nx)-(gamma*(PP)))@LL

            for j in range(len(U)):
                H[:,j]=(L[:,j]+(gamma*P[:,:,j]@ v).reshape(nx))

            policy=np.argmin(H,axis=1)
            v=(np.min(H,axis=1)).reshape(nx,1)
            # print(np.max(np.abs(v-v_old)))
            
            if(np.max(np.abs(v-v_old))<1e-100):
                break
                
            itr=itr+1
        plt.figure()
        plt.imshow(v.reshape(len_x2,len_x1))
        plt.show()
        V_x, PI_policy = v, policy
        return V_x, PI_policy,
    
    def generate_trajectory(self, in_loc, init_state, policy, t):
        print('Generating Trajectory')

        u=policy[in_loc]
        ctrl=np.array([u])

        X= np.array([[np.pi],[0]])+ (self.f_x_u(init_state,u)*tau)
        # X[0]=self.wrapp(X[0])
        angle=np.array([X[0]])

        # X1=np.array([np.pi,0])
        c=0
        while c<100: #(X[0]!=0 ):
        #     print(X)
            E=((states[:,:,0]-X[0])**2+(states[:,:,1]-X[1])**2)
            temp=np.argmin(E)
            u1= policy[temp]
            ctrl=np.append(ctrl,u1)
            
            ind=np.unravel_index(np.argmin(E, axis=None), E.shape)    
            X= states[ind[0],ind[1],:].reshape(2,1)+(self.f_x_u(states[ind[0],ind[1],:],u1)*tau)
            X[0]=self.wrapp(X[0])
            angle=np.append(angle,X[0])
            # print(X)
            c=c+1

        f_u = interp1d(np.arange(len(ctrl)), ctrl,kind='cubic')
        f_theta = interp1d(np.arange(len(angle)), angle, kind='cubic')
        # U[policy]
        w=np.linspace(1,10,1000)
        # plt.plot(w,f_u(w))
        # plt.grid()
        # plt.figure()
        # plt.plot(w,f_theta(w))
        # plt.grid()
        # plt.show()
        # plt.plot(range(len(policy)),U[policy])


        theta, u = (f_theta(w)), (f_u(w))
        return theta, u
    

if __name__ == '__main__':
    inv_pendulum = InvertedPendulum()
    # print(tau)
    x1=np.linspace(-np.pi,np.pi,n1)
    x2=np.linspace(-v_max,v_max,n2)
    U=np.linspace(-u_max,u_max,n_u)
    # v=np.zeros((n1,n2))
    cov= (sigma @ sigma.T)*tau+dist*np.eye(2)

    len_x1=len(x1)
    len_x2=len(x2)

    X1,X2=np.meshgrid(x1,x2) # x axis --> x2, y axis --> x1
    states = np.dstack((X1, X2))

    init_ind=np.where( (np.logical_and((states[:,:,0]==init_state[0]),(states[:,:,1]==init_state[1])))==True)
    in_loc=inv_pendulum.loc1(init_ind[1][0],init_ind[0][0],len_x1,len_x2)


    P,L=inv_pendulum.Build_MDP(x1,x2,U)
    if(VI):
        VI_V, VI_policy = inv_pendulum.value_iteration(P,L,x1,x2,gamma)
        theta, u = inv_pendulum.generate_trajectory(in_loc,init_state,VI_policy,0)
    if(PI):
        PI_V, PI_policy = inv_pendulum.policy_iteration(P,L,x1,x2,U,gamma)
        theta, u = inv_pendulum.generate_trajectory(in_loc,init_state,PI_policy,0)
    
    # print(type(theta),type(u))
    # print(theta)
    # print(len(u))
    inv_pendulum.load_trajectory(theta, u)
    inv_pendulum.start()

    # inv_pendulum.start()
    ######## Using example functions to get the animation ########
    # inv_pendulum = InvertedPendulum(*params)
    # PI_V, PI_policy = inv_pendulum.policy_iteration()
    # theta, u = inv_pendulum.generate_trajectory()
    # theta=0

    # u=np.load('control1.npy')
    
    # theta=np.load('angle1.npy')

    # inv_pendulum.load_trajectory(theta, u)
    # inv_pendulum.start()
    
    ######## TODO: Implement functions for visualization ########
    #############################################################
    
    


