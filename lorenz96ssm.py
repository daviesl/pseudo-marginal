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

theta0 = np.array([8.])
om = np.zeros((36,3),dtype=np.float64)
om[0,0]=1
om[36//2,1]=1
om[36-1,2]=1

class Lorenz96Abstract(ItoProcess):
    @classmethod
    def who(cls):
        return "Lorenz96"
    @classmethod
    def default_theta(cls):
        global theta0
        return cls.transformParameterstoTheta(theta0)
    @classmethod
    def X_size(self):
        return 36
    @classmethod
    def theta_size(self):
        return 1
    @classmethod
    def y_dim(cls):
        # TODO make this DRY with obs_map.ndim
        return 3
    @classmethod
    def obsinterval(cls):
        return 40
    @classmethod
    #@numba.jit
    def obserr(cls):
        return 0.1
    @classmethod
    def delta_t(cls):
        return 0.001
    @classmethod
    def observationCovariance(cls):
        return np.eye(cls.y_dim())*(cls.obserr()**2)
    @classmethod
    def obs_map(cls):
        global om
        return om
    @staticmethod
    @numba.jit("float64[:](float64[:])")
    def transformThetatoParameters(theta):
        global theta0
        target = theta0 
        lower = target * 0.2
        upper = target + (target - lower)
        return theta*(upper-lower) + lower
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
        return X*10.0
    @staticmethod
    @numba.jit 
    def transformStatetoX(tr):
        return tr*0.1
    @staticmethod
    @numba.jit
    def drift(X_k,y_J,theta):
        """
        Returns dX/dt
        vectorised
        """
        N = X_k.shape[-1] 
        F = theta[0]
        # compute state derivatives
        d = np.zeros(X_k.shape)
        # first the 3 edge cases: i=1,2,N
        d[:,0] = (X_k[:,1] - X_k[:,N-2]) * X_k[:,N-1] - X_k[:,0]
        d[:,1] = (X_k[:,2] - X_k[:,N-1]) * X_k[:,0]- X_k[:,1]
        d[:,N-1] = (X_k[:,0] - X_k[:,N-3]) * X_k[:,N-2] - X_k[:,N-1]
        # then the general case
        for i in range(2, N-1):
            d[:,i] = (X_k[:,i+1] - X_k[:,i-2]) * X_k[:,i-1] - X_k[:,i]
        # add the forcing term
        d += F
        # return the state derivatives
        return d
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
        Ys = cls.obseqn_with_noise(Xs[::obsinterval,:])#,obserr_mat)
        return (Xs,Ys)
    @classmethod
    def plot_synthetic(cls,Xs,Ys,num_steps):
        ff,ax=plt.subplots()
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
    @staticmethod
    @numba.jit
    def observationEquation(Xs):
        global om
        return mdla_dottrail2x1_broadcast(om.T,Xs) # fixme don't use global
    @classmethod
    def obseqn_with_noise(cls,Xs):
        cov = cls.observationCovariance()
        dim = cov.shape[0]
        Ys = cls.observationEquation(Xs) + np.random.multivariate_normal(np.zeros(dim),cov,size=Xs.shape[0])
        return Ys
    @classmethod
    def plot_traces(cls,chain_length,estparams,ml):
        global theta0
        tS=np.ones(chain_length)*theta0[0]
    
        print("Estimated Parameters {}".format(estparams))
        print("Marginal likelihoods {}".format(ml))
        fig,((ax1,hax1),(ax4,hax4))= plt.subplots(4,2)
        ax1.plot(estparams[:,0],'b',label="Est F",linewidth=1)
        ax1.plot(tS,'b',label="True F",linewidth=1)
        ax1.set_title("Estimated Parameters F")
        ax4.plot(ml[:],'y',linewidth=1)
        ax4.set_title("Log Marginal Likelihood")
        b = 20
        hax1.hist(estparams[:,0],bins=b,color='b',orientation="horizontal")
        hax4.hist(ml,color='y',bins=b,orientation="horizontal")
        plt.tight_layout()
        plt.show()
