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

#class Lorenz63Abstract(ItoProcess):
#class Lorenz63Abstract(ModifiedDiffusionBridge):
#class Lorenz63Abstract(ResidualBridge):
class Lorenz63Abstract(LindstromBridge):
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
    #@numba.jit
    def obserr(cls):
        return 0.1
    @classmethod
    def delta_t(cls):
        return 0.001
    #@staticmethod
    #@numba.jit("f8[:,:]()")
    @classmethod
    def observationCovariance(cls):
        return np.array([[cls.obserr()**2,0],[0,cls.obserr()**2]])
    #@staticmethod
    #@numba.jit("f8[:,:]()")
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
        #target = np.array([20.,40.,10.])
        global theta0
        target = theta0 #np.array([10.,28.,2.667])
        lower = target * 0.2
        upper = target + (target - lower)
        return (nt - lower)/(upper-lower)
    @staticmethod
    @numba.jit #("float64[:][:](float64[:][:])")
    def transformXtoState(X):
        return np.column_stack((X[:,0]*20.0,X[:,1]*20.0,X[:,2]*5.+0.8))
    @staticmethod
    @numba.jit #("float64[:][:](float64[:][:])")
    def transformStatetoX(tr):
        return np.column_stack((tr[:,0]/20.0,tr[:,1]/20.0,(tr[:,2]-0.8)/5.))
    @staticmethod
    @numba.jit #("float64[:][:](float64[:][:],float64[:][:])")
    def xTPx(x,P):
        return np.sum(np.dot(x,P)*x,axis=1)
    @classmethod
    def synthetic(cls,dt=0.001,num_steps=10000,x0=0.,y0=1.,z0=1.05,xW=1.,yW=1.,zW=1.,xO=1.,yO=1.,zO=1.):
        obsinterval = cls.obsinterval()
        # Need one more for the initial values
        Xs = np.empty((num_steps + 1,3))
        Ys = np.empty((num_steps + 1,2))
        
        # Set initial values
        Xs[0,0], Xs[0,1], Xs[0,2] = (x0,y0,z0)
        
        # Step through "time", calculating the partial derivatives at the current point
        # and using them to estimate the next point
        for i in range(num_steps):
            #x_dot, y_dot, z_dot = lorenz(Xs[i,0], Xs[i,1], Xs[i,2])
            X_dot = cls.drift(Xs[i,:])
            Xs[i + 1,0] = Xs[i,0] + (X_dot[i,0] * dt)+ xW*np.random.normal(0,np.sqrt(dt)) 
            Xs[i + 1,1] = Xs[i,1] + (X_dot[i,1] * dt)+ yW*np.random.normal(0,np.sqrt(dt))
            Xs[i + 1,2] = Xs[i,2] + (X_dot[i,2] * dt)+ zW*np.random.normal(0,np.sqrt(dt))
        Ys = obseqn_with_noise(Xs[::obsinterval])#,obserr_mat)
        obserr_mat = cls.observationCovariance() #np.array([[obserr**2,0],[0,obserr**2]])
        return (Xs,Ys)
    @classmethod
    def plot_synthetic(cls,Xs,Ys):
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
    @staticmethod
    @numba.jit
    def observationEquation(Xs): #,sigma=np.array([obserr,obserr])):
        dim = Xs.shape[0]
        Ys = np.zeros((dim,2))
        Ys[:,0] = Xs[:,0] #+ np.random.normal(0,np.sqrt(sigma[0]),dim)
        Ys[:,1] = Xs[:,2] #+ np.random.normal(0,np.sqrt(sigma[1]),dim)
        return Ys
    @classmethod
    def obseqn_with_noise(cls,Xs):#,cov=np.array([[obserr**2,0],[0,obserr**2]])):
        cov = cls.observationCovariance()
        dim = cov.shape[0]
        Ys = cls.observationEquation(Xs) + np.random.multivariate_normal(np.zeros(dim),cov,size=Xs.shape[0])
        return Ys
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

if __name__ == '__main__':
    timestr = time.strftime("%Y%m%d-%H%M%S")

    argctr = 1
    print(sys.argv)
    actionflag = sys.argv[argctr]
    argctr += 1

    num_steps = 6400 #3200 #12800 #12800 #3200 #12800 # 1600
    X0_ = np.array([0,1,1.05])
    X0_ = X0_[np.newaxis,:]
    #X0_mu = Lorenz63Abstract.transformStatetoX(np.array([[0,1,1.05],[0,1,1.05]]))
    X0_mu = Lorenz63Abstract.transformStatetoX(X0_)
    print("X0_mu = {}".format(X0_mu))

    if actionflag == 't' or actionflag == 'r':
        if (len(sys.argv) > 2):
            X = np.load(sys.argv[argctr])
            argctr += 1
            Y = np.load(sys.argv[argctr])
            argctr += 1
            num_steps = X.shape[0] #3200 #12800 # 1600
        else:
            X,Y = Lorenz63Abstract.synthetic(dt=Lorenz63Abstract.delta_t(),num_steps=num_steps)
            np.save("synthetic_X_{}".format(timestr),X)
            np.save("synthetic_Y_{}".format(timestr),Y)
        print("Y = {}".format(Y))

        n=200 #2048 #1024 #8192 #1024 #16384 #2048 #512
        chain_length=1000

        pf = stateFilter(Lorenz63Abstract(),Y,n)

        # run pmmh
        #sampler = pmpfl(innov_lindstrom_bridge,innov_lindstrom_bridge,lh,Y,3,3,n)
        #sampler = pmpfl(innov,innov_lindstrom_bridge,lh,Y,3,3,n)
        #sampler = pmpfl(innov_lindstrom_bridge,lh,Y,3,3,n)
        #sampler = pmpfl(innov_diffusion_bridge,lh,Y,3,3,n)
        #sampler = pmpfl(innov_lindstrom_bridge,propagate_noisefree,lh,Y,3,3,n)
        #sampler = pmpfl(innov_lindstrom_residual_bridge,lh,Y,3,3,n)
        #sampler = pmpfl(innov,propagate_noisefree,lh,Y,3,3,n)
        #sampler = pmpfl(innov,locally_opt_proposal_lindstrom_bridge,lh,Y,3,3,n)
        #sampler = pmpfl(innov_residual_bridge,innov,lh,Y,3,3,n)
        #sampler = pmpfl(innov,innov_lindstrom_residual_bridge,lh,Y,3,3,n)
        #sampler = pmpfl(innov,innov,lh,Y,3,3,n)

    if actionflag == 't':
        Lorenz63Abstract.plot_synthetic(X,Y)
        # Assert transformation is correct
        XtrX = Lorenz63Abstract.transformXtoState(Lorenz63Abstract.transformStatetoX(X))
        print("X = {}, XtrX = {}".format(X,XtrX))
        assert((np.abs(X - XtrX) < 0.00001).all())
        # print likelihood of true solution
        T = Y.shape[0]
        #testX = np.zeros((T,n,3))
        #testX[0,:] = X0_
        log_lh = np.zeros(T)
        for i in range(0,T):
           #testX[i,:] = Lorenz63Abstract.transformXtoState(innov(Lorenz63Abstract.transformStatetoX(testX[i,:]),np.array([10.,28.,2.667])))
           log_lh[i] = pf.lh(Lorenz63Abstract.transformStatetoX(X[np.newaxis,i*Lorenz63Abstract.obsinterval(),:]),Y[np.newaxis,i,:],np.zeros(3)) # last arg theta unused
           #print("log lh at i={} is {}".format(i,log_lh[i]))
           print("i={}, loglh = {}, X[i,:]={}, Y[i,:]={}".format(i,log_lh[i],Lorenz63Abstract.transformStatetoX(X[np.newaxis,i*Lorenz63Abstract.obsinterval(),:]),Y[i,:]))

        print("synthetic observations T={} have log lh sum = {}".format(T,log_lh.sum()))
        fig = plt.figure()
        plt.plot(log_lh)
        plt.show()
        #fig = plt.figure()
        #plt.plot(testX[:,:,0])
        #plt.show()
        ml_test = pf.test_filter(chain_length,X0_mu[0,:],Lorenz63Abstract.transformParameterstoTheta(theta0))
        ml_test_log_mean = logsumexp(ml_test) - math.log(chain_length)
        ml_test_log_var = 2. * ml_test_log_mean + np.log(np.sum((np.exp(ml_test-ml_test_log_mean)-1)**2)) - math.log(chain_length)
        
        
        print("Log Marginal likelihood: Mean = {} Std Dev = {}".format(ml_test.mean(),ml_test.std()))
        print("Log Non-log: Mean = {} Std Dev = {}".format(ml_test_log_mean,ml_test_log_var * 0.5))
    
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.plot(ml_test,'y',linewidth=1)
        ax2.boxplot(ml_test)
        plt.show()

    elif actionflag == 'r':
        plot_synthetic(X,Y)
        #pcov_in = 2. * np.array([[ 6.53672845e-07,  1.80943850e-06, -2.23494164e-06],
        #           [ 1.80943850e-06,  5.00872523e-06, -6.18656485e-06],
        #           [-2.23494164e-06, -6.18656485e-06,  7.64138236e-06]])
        burnin = 200
        initial_run = 400
        esttheta,ml,ar,pcov_out = sampler.run_pmmh(initial_run,X0_mu[0,:],np.array([0.,0.,0.]),Lorenz63Abstract.transformParameterstoTheta(theta0)) #,pcov0=pcov_in)
        pcov_in = np.eye(3) * 0.0000001
        thetamean = np.mean(esttheta[burnin:,:],axis=0)
        covnorm = 1./(initial_run-burnin-1)
        for k in range(burnin,initial_run):
            pcov_in += np.outer(esttheta[k,:]-thetamean,esttheta[k,:]-thetamean)*covnorm
        print("P Covariance for second run = {}".format(pcov_in))

        estparams,ml,ar,pcov_out = sampler.run_pmmh(chain_length,X0_mu[0,:],np.array([0.,0.,0.]),Lorenz63Abstract.transformParameterstoTheta(theta0),pcov0=pcov_in)
        print("Acceptance rate = {}".format(1.*ar.sum()/chain_length))
    
        for i in range(estparams.shape[0]):
            estparams[i,:]=Lorenz63Abstract.transformThetatoParameters(estparams[i,:])
        np.save("estparams_{}".format(timestr),estparams)
        np.save("ml_{}".format(timestr),ml)
        plot_traces(chain_length,estparams,ml)
    elif actionflag == 'p':
        chain_length = int(sys.argv[argctr])
        argctr += 1
        burnin = int(sys.argv[argctr])
        argctr += 1
        estparams = np.load(sys.argv[argctr])
        argctr += 1
        ml = np.load(sys.argv[argctr])
        argctr += 1
        X = np.load(sys.argv[argctr])
        argctr += 1
        Y = np.load(sys.argv[argctr])
        argctr += 1
        plot_synthetic(X,Y)
        new_cl = chain_length - burnin
        plot_traces(chain_length-burnin,estparams[burnin:],ml[burnin:])
