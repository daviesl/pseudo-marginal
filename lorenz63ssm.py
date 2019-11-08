import numpy as np
import math
import time
import numba
from numba import jit, float64
from pmmcmc import *
from mdlinalg import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import sys

theta0 = np.array([10.,28.,2.667])
#theta0 = np.array([10.,10,2.667])

class Lorenz63Abstract(ItoProcess):
    @classmethod
    def who(cls):
        return "Lorenz63"
    @classmethod
    def default_theta(cls):
        global theta0
        return theta0
    @classmethod
    def X_size(self):
        return 3
    @classmethod
    def theta_size(self):
        return 3
    @classmethod
    def y_dim(cls):
        # TODO make this DRY with obs_map.ndim
        return 2
    @staticmethod
    @numba.jit
    def drift(X_k,y_J,theta):
        '''
        Given:
           x, y, z: a point of interest in three dimensional space
           s, r, b: parameters defining the lorenz attractor
        Returns:
           x_dot, y_dot, z_dot: values of the lorenz attractor's partial
               derivatives at the point x, y, z
        '''
        x = X_k[:,0]
        y = X_k[:,1]
        z = X_k[:,2]
        s=theta[0]
        r=theta[1]
        b=theta[2]
        X_k_dot = np.zeros_like(X_k)
        X_k_dot[:,0] = s*(y - x) 
        X_k_dot[:,1] = r*x - y - x*z 
        X_k_dot[:,2] = x*y - b*z 
        return X_k_dot

    @classmethod
    def obsinterval(cls):
        return 40
    @classmethod
    def obserr(cls):
        return 0.1
    @classmethod
    def delta_t(cls):
        return 0.001
    @classmethod
    def observationCovariance(cls):
        return np.array([[cls.obserr()**2,0],[0,cls.obserr()**2]])
    @classmethod
    def obs_map(cls):
        return np.array([[1., 0.],
                        [0., 0.],
                        [0., 1.]],dtype=np.float64)
    @staticmethod
    @numba.jit("float64[:](float64[:])")
    def transformThetatoParameters(theta):
        #return np.array([10,28,2.667])
        global theta0
        target = theta0 #np.array([10.,28.,2.667])
        #target = np.array([20.,40.,10.])
        lower = target * 0.2
        upper = target + (target - lower)
        #uniform
        return theta*(upper-lower) + lower
        #Normal
        #return np.power(1.4,theta*2) + np.array([10,28,2.667]) - 1.
        #return theta + np.array([10,28,2.667])
    @staticmethod
    @numba.jit("float64[:](float64[:])")
    def transformParameterstoTheta(nt):
        global theta0
        target = theta0 
        lower = target * 0.2
        upper = target + (target - lower)
        return (nt - lower)/(upper-lower)
    @staticmethod
    @numba.jit 
    def transformXtoState(X):
        return np.column_stack((X[:,0]*20.0,X[:,1]*20.0,X[:,2]*5.+0.8))
    @staticmethod
    @numba.jit 
    def transformStatetoX(tr):
        return np.column_stack((tr[:,0]/20.0,tr[:,1]/20.0,(tr[:,2]-0.8)/5.))
    @staticmethod
    @numba.jit
    def observationEquation(Xs): 
        dim = Xs.shape[0]
        Ys = np.zeros((dim,2))
        Ys[:,0] = Xs[:,0] 
        Ys[:,1] = Xs[:,2] 
        return Ys
    @classmethod
    def obseqn_with_noise(cls,Xs):
        cov = cls.observationCovariance()
        dim = cov.shape[0]
        Ys = cls.observationEquation(Xs) + np.random.multivariate_normal(np.zeros(dim),cov,size=Xs.shape[0])
        return Ys
    @classmethod
    def initialState(cls):
        return np.array([0.,1.,1.05])
    @classmethod
    def synthetic(cls,dt=0.001,num_steps=10000,th0=theta0):
        x0=cls.initialState()
        obsinterval = cls.obsinterval()
        # Need one more for the initial values
        Xs = np.empty((num_steps + 1,cls.X_size()))
        Ys = np.empty((num_steps + 1,cls.y_dim()))
        
        # Set initial values
        Xs[0,:]=x0
        
        # Step through "time", calculating the partial derivatives at the current point
        # and using them to estimate the next point
        for i in range(num_steps):
            X_dot = cls.drift(Xs[i,:].reshape((1,cls.X_size())),0,th0).reshape((cls.X_size(),))
            Xs[i + 1,:] = Xs[i,:] + (X_dot * dt)+ mdla_dottrail2x1_broadcast(spsd_sqrtm(cls.diffusion(Xs[i,:],0,th0)),np.random.normal(0,np.sqrt(dt),cls.X_size())) 
        Ys = cls.obseqn_with_noise(Xs[::obsinterval])#,obserr_mat)
        return (Xs,Ys)
    @classmethod
    def plot_synthetic(cls,Xs,Ys,num_steps):
        ff,ax=plt.subplots()
        #ax.plot(Xs[:,0],'r',Xs[:,1],'g',Xs[:,2],'b',lw=1)
        ax.plot(Xs[:,0],'r',label="x",lw=1)
        ax.plot(Xs[:,1],'g',label="y",lw=1)
        ax.plot(Xs[:,2],'b',label="z",lw=1)
        ax.plot(np.arange(0,num_steps+1,cls.obsinterval()),Ys[:,0],'r+',np.arange(0,num_steps+1,cls.obsinterval()),Ys[:,1],'b+',label="Observed",lw=1)
        ax.legend()
        plt.show()
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(Ys[:,0],Ys[:,1],linewidth=0.5)
        ax1.set_title("Observed Data")
        ax1.set_xlabel("X Axis")
        ax1.set_ylabel("Z Axis")
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot(Xs[:,0], Xs[:,1], Xs[:,2], lw=0.5)
        ax2.set_xlabel("X Axis")
        ax2.set_ylabel("Y Axis")
        ax2.set_zlabel("Z Axis")
        ax2.set_title("State")
        plt.show()
    @classmethod
    def plot_traces(cls,chain_length,estparams,ml):
        global theta0
        tS=np.ones(chain_length)*theta0[0]
        tR=np.ones(chain_length)*theta0[1]
        tB=np.ones(chain_length)*theta0[2]
    
        print("Estimated Parameters {}".format(estparams))
        print("Marginal likelihoods {}".format(ml))
        fig,((ax1,hax1),(ax2,hax2),(ax3,hax3),(ax4,hax4))= plt.subplots(4,2)
        ax1.plot(estparams[:,0],'b',label="Est S",linewidth=1)
        ax2.plot(estparams[:,1],'g',label="Est R",linewidth=1)
        ax3.plot(estparams[:,2],'r',label="Est B",linewidth=1)
        ax1.plot(tS,'b',label="True S",linewidth=1)
        ax2.plot(tR,'g',label="True R",linewidth=1)
        ax3.plot(tB,'r',label="True B",linewidth=1)
        ax1.set_title("Estimated Parameters S")
        ax2.set_title("Estimated Parameters R")
        ax3.set_title("Estimated Parameters B")
        ax4.plot(ml[:],'y',linewidth=1)
        ax4.set_title("Log Marginal Likelihood")
        b = 20
        hax1.hist(estparams[:,0],bins=b,color='b',orientation="horizontal")
        hax2.hist(estparams[:,1],bins=b,color='g',orientation="horizontal")
        hax3.hist(estparams[:,2],bins=b,color='r',orientation="horizontal")
        hax4.hist(ml,color='y',bins=b,orientation="horizontal")
        plt.tight_layout()
        plt.show()
