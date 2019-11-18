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

nP = 3 # number of populations
nO = 1 # number of oscillators per population
nT = nO*nP # total oscillators
nY = nT//2 # number of observed
# coupling parameters        
@numba.jit
def coupling_param_index(idx):
    a= [int((nP*(nP-1)/2) - (nP-min(r,c))*((nP-min(r,c))-1)/2 + max(r,c))  for r in range(nP) for c in range(nP)] # nP^2 to nP(nP+1)//2 mapping
    return a[idx]
# fixed frequencies
dt = 0.001
#omega = np.tile(3.*2.**np.arange(nO)-2,nP)#*dt # loosely mapped to delta (0.5-3.5), theta(3.5-7), alpha(7-13) and beta(13-30) ranges for 4 oscillators
omega = np.array([-.5, 0., 0.5])#np.array([28., 19., 11.])
print('omega = {}'.format(omega))

#theta0 = np.arange(1, nP*(nP+1)//2+1)*0.1
#theta0[1::2] *= -1
theta0 = np.array([0.0, -0.3, -0.1, 0.0, -0.9, 0.0])*-5

om = np.zeros((nT,nY),dtype=np.float64)
# define observed mapping
for i in range(nY):
    om[(i*nT//nY),0]=1

@numba.jit
def kuramoto_uncoupled_dynamics(X_k,F):
    return X_k*0.+F
    
@numba.jit
def kuramoto_coupling(X_k,i,j):
    # sine coupling
    return np.sin(X_k[:,i]-X_k[:,j])


class KuramotoAbstract(ItoProcess):
    @classmethod
    def who(cls):
        return "Kuramoto"
    @classmethod
    def default_theta(cls):
        global theta0
        return cls.transformParameterstoTheta(theta0)
    @classmethod
    def X_size(self):
        global nT
        return nT
    @classmethod
    def theta_size(self):
        #global nP
        #return nP*(nP+1)//2
        global theta0
        return theta0.shape[0]
    @classmethod
    def y_dim(cls):
        global nY
        return nY
    @classmethod
    def obsinterval(cls):
        return 40
    @classmethod
    #@numba.jit
    def obserr(cls):
        return 0.1
    @classmethod
    def delta_t(cls):
        global dt
        return dt
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
        m = np.max(np.abs(target))
        return theta*m + target
    @staticmethod
    @numba.jit("float64[:](float64[:])")
    def transformParameterstoTheta(nt):
        global theta0
        target = theta0 
        m = np.max(np.abs(target))
        return (nt - target)/m
    @staticmethod
    @numba.jit 
    def transformXtoState(X):
        #return np.fmod(X*2*math.pi,2*math.pi)
        return X
    @staticmethod
    @numba.jit 
    def transformStatetoX(tr):
        #return np.fmod(tr/(2*math.pi),1)
        return tr
    @staticmethod
    @numba.jit
    def diffusion(X_k,y_J,theta):
        dim=X_k.shape[-1]
        return np.eye(dim) * 0.01
    @staticmethod
    @numba.jit
    def drift(X_k,y_J,theta):
        """
        General Kuramoto ODE of m'th harmonic order.
        Returns dX/dt
        vectorised
            w -- iterable frequency
            k -- 3D coupling matrix, unless 1st order
        """
        global nO,nP,nT,omega # number of oscillators per population, number of populations
        rnO = 1./nO # reciprocal
        #print("X_k({}) = {}".format(X_k.shape,X_k))
        #print("theta({}) = {}".format(theta.shape,theta))
        #X_dot = np.zeros(X_k.shape)
        #N = nO*nP # total number of oscillators. TODO assert X_k.shape[1] = N+1
        # define the coupling options, whether it be S=N(N+1)/2 (all-to-all triangular matrix terms)
        # or something simpler.
        # list all pairs as numpy indices tuples, total list length N.
        # For all-to-all coupling, assume coupling is bidirectional, 
        # i.e. upper-triangular adjacency matrix
        #S = nP(nP+1)//2 # population-to-population
        #F = theta[S:S+nT] # Fixed dynamics (bifurcation parameters?) 
        #couples_col = [ col for row in range(nP) for col in range(row,nP)] # defined as columns of X_k
        #couples_row = [ row for row in range(nP) for col in range(row,nP)] # defined as columns of X_k
        #for i in range(S):
        #    X_dot += coupling_params[i]*kuramoto_coupling(X_k,couples_row[i],couples_col[i]))

        coupling_params = theta #[:S] # coupling parameters


        X_dot = kuramoto_uncoupled_dynamics(X_k,omega)
        for p in range(nP):
            for o in range(nO):
                for tau in range(nP):
                    for j in range(nO):
                        a=p*nO+o
                        b=tau*nO+j
                        theta_i=p*nP+tau
                        #print("p*nO+o={}".format(p*nO+o))
                        #print("p*tau={}".format(p*tau))
                        #print("cpi[i]={}".format(coupling_param_index[p*tau]))
                        #print("X_dot[:,p*nO+o]={}".format(X_dot[:,p*nO+o]))
                        X_dot[:,a] += rnO * coupling_params[coupling_param_index(theta_i)] * kuramoto_coupling(X_k,b,a)

        return X_dot

        ##phase kuramoto
        #w, k = X_k[:,0], X_k[:,1]
        #yt = y_J[:,None]
        #dy = y_J-yt
        #phase = w.astype(self.dtype)
        #for m, _k in enumerate(k):
        #    phase += np.sum(_k*np.sin((m+1)*dy),axis=1)

        #return phase

        ##frequency kuramoto
        #w, k = arg
        #n_osc = len(w)
    
        #if len(y)>1:
        #    coupling = lambda i, j: k[i][j]*np.sin(y[j]-y[i])
        #    R = lambda i: np.random.normal(0,1)*0.00001 # Additive noise
        #    out = [w[i] + R(i) + np.sum([coupling(i,j) for j in range(n_osc) if i!=j]) for i in range(n_osc)]
        #    #~ out = [w[i] + np.sum([coupling(i,j) for j in range(n_osc) if i!=j]) for i in range(n_osc)]
        #else:
        #    out = w[0]
        #return out

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
            Xs[i + 1, :] = cls.transformXtoState(cls.transformStatetoX(Xs[i + 1,:]))
        Ys = cls.obseqn_with_noise(Xs[::obsinterval,:])#,obserr_mat)
        return (Xs,Ys)
    @classmethod
    def plot_synthetic(cls,Xs,Ys,num_steps):
        ff,ax=plt.subplots()
        # do colours?
        for i in range(Xs.shape[1]):
            ax.plot(Xs[:,i],label="X_{}".format(i),lw=1)
        # todo fix y colours by using obsmap matrix on np.arange(X_size()) indices
        for i in range(Ys.shape[1]):
            ax.plot(np.arange(0,num_steps+1,cls.obsinterval()),Ys[:,i],'+',label="Y_{}".format(i),lw=1)
        ax.legend()
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
        td = cls.theta_size()   
        print("Estimated Parameters {}".format(estparams))
        print("Marginal likelihoods {}".format(ml))
        fig,axes= plt.subplots(td+1,2)
        b = 20
        for i in range(yd):
            axes[i,0].plot(estparams[:,i],label="Est theta_{}".format(i),linewidth=1)
            axes[i,0].plot(tS,label="True theta_{}".format(i),linewidth=1)
            axes[i,0].set_title("Estimated Parameters theta_{}".format(i))
            axes[i,1].hist(estparams[:,i],bins=b,orientation="horizontal")
        axes[td,0].plot(ml[:],'y',linewidth=1)
        axes[td,0].set_title("Log Marginal Likelihood")
        axes[td,1].hist(ml,bins=b,orientation="horizontal")
        plt.tight_layout()
        plt.show()
