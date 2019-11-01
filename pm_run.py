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
from lorenz63ssm import Lorenz63Abstract, theta0

class Lorenz63(LindstromBridge,Lorenz63Abstract):
    pass


if __name__ == '__main__':
    timestr = time.strftime("%Y%m%d-%H%M%S")

    argctr = 1
    print(sys.argv)
    actionflag = sys.argv[argctr]
    argctr += 1

    num_steps = 6400 #3200 #12800 #12800 #3200 #12800 # 1600
    X0_ = np.array([0,1,1.05])
    X0_ = X0_[np.newaxis,:]
    #X0_mu = Lorenz63.transformStatetoX(np.array([[0,1,1.05],[0,1,1.05]]))
    X0_mu = Lorenz63.transformStatetoX(X0_)
    print("X0_mu = {}".format(X0_mu))

    if actionflag == 't' or actionflag == 'r':
        if (len(sys.argv) > 2):
            X = np.load(sys.argv[argctr])
            argctr += 1
            Y = np.load(sys.argv[argctr])
            argctr += 1
            num_steps = X.shape[0] #3200 #12800 # 1600
        else:
            X,Y = Lorenz63.synthetic(dt=Lorenz63.delta_t(),num_steps=num_steps)
            np.save("synthetic_X_{}".format(timestr),X)
            np.save("synthetic_Y_{}".format(timestr),Y)
        print("Y = {}".format(Y))

        n=200 #2048 #1024 #8192 #1024 #16384 #2048 #512
        chain_length=1000

        pf = stateFilter(Lorenz63(),Y,n)

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
        sampler = parameterEstimator(pf)

    if actionflag == 't':
        Lorenz63.plot_synthetic(X,Y,num_steps=num_steps)
        # Assert transformation is correct
        XtrX = Lorenz63.transformXtoState(Lorenz63.transformStatetoX(X))
        print("X = {}, XtrX = {}".format(X,XtrX))
        assert((np.abs(X - XtrX) < 0.00001).all())
        # print likelihood of true solution
        T = Y.shape[0]
        #testX = np.zeros((T,n,3))
        #testX[0,:] = X0_
        log_lh = np.zeros(T)
        for i in range(0,T):
           #testX[i,:] = Lorenz63.transformXtoState(innov(Lorenz63.transformStatetoX(testX[i,:]),np.array([10.,28.,2.667])))
           log_lh[i] = pf.lh(Lorenz63.transformStatetoX(X[np.newaxis,i*Lorenz63.obsinterval(),:]),Y[np.newaxis,i,:],np.zeros(3)) # last arg theta unused
           #print("log lh at i={} is {}".format(i,log_lh[i]))
           print("i={}, loglh = {}, X[i,:]={}, Y[i,:]={}".format(i,log_lh[i],Lorenz63.transformStatetoX(X[np.newaxis,i*Lorenz63.obsinterval(),:]),Y[i,:]))

        print("synthetic observations T={} have log lh sum = {}".format(T,log_lh.sum()))
        fig = plt.figure()
        plt.plot(log_lh)
        plt.show()
        #fig = plt.figure()
        #plt.plot(testX[:,:,0])
        #plt.show()
        ml_test = pf.test_filter(chain_length,X0_mu[0,:],Lorenz63.transformParameterstoTheta(theta0))
        ml_test_log_mean = logsumexp(ml_test) - math.log(chain_length)
        ml_test_log_var = 2. * ml_test_log_mean + np.log(np.sum((np.exp(ml_test-ml_test_log_mean)-1)**2)) - math.log(chain_length)
        
        
        print("Log Marginal likelihood: Mean = {} Std Dev = {}".format(ml_test.mean(),ml_test.std()))
        print("Log Non-log: Mean = {} Std Dev = {}".format(ml_test_log_mean,ml_test_log_var * 0.5))
    
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.plot(ml_test,'y',linewidth=1)
        ax2.boxplot(ml_test)
        plt.show()

    elif actionflag == 'r':
        Lorenz63.plot_synthetic(X,Y,num_steps)
        #pcov_in = 2. * np.array([[ 6.53672845e-07,  1.80943850e-06, -2.23494164e-06],
        #           [ 1.80943850e-06,  5.00872523e-06, -6.18656485e-06],
        #           [-2.23494164e-06, -6.18656485e-06,  7.64138236e-06]])
        burnin = 200
        initial_run = 400
        esttheta,ml,ar,pcov_out = sampler.run_pmmh(initial_run,X0_mu[0,:],np.array([0.,0.,0.]),Lorenz63.transformParameterstoTheta(theta0)) #,pcov0=pcov_in)
        pcov_in = np.eye(3) * 0.0000001
        thetamean = np.mean(esttheta[burnin:,:],axis=0)
        covnorm = 1./(initial_run-burnin-1)
        for k in range(burnin,initial_run):
            pcov_in += np.outer(esttheta[k,:]-thetamean,esttheta[k,:]-thetamean)*covnorm
        print("P Covariance for second run = {}".format(pcov_in))

        estparams,ml,ar,pcov_out = sampler.run_pmmh(chain_length,X0_mu[0,:],np.array([0.,0.,0.]),Lorenz63.transformParameterstoTheta(theta0),pcov0=pcov_in)
        print("Acceptance rate = {}".format(1.*ar.sum()/chain_length))
    
        for i in range(estparams.shape[0]):
            estparams[i,:]=Lorenz63.transformThetatoParameters(estparams[i,:])
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
