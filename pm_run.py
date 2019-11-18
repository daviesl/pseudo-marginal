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
from lorenz63ssm import Lorenz63Abstract
from lorenz96ssm import Lorenz96Abstract
from kuramotossm import KuramotoAbstract

#class Lorenz63(LindstromBridge,Lorenz63Abstract):
#    pass
#class Lorenz96(LindstromBridge,Lorenz96Abstract):
#    pass
class Lorenz96(Lorenz96Abstract):
    pass
class Lorenz63(Lorenz63Abstract):
    pass
class Kuramoto(KuramotoAbstract):
    pass




if __name__ == '__main__':
    timestr = time.strftime("%Y%m%d-%H%M%S")

    argctr = 1
    print(sys.argv)

    modelflag = sys.argv[argctr].lower()
    argctr += 1
    
    if modelflag == 'lorenz96':
        class ModelClass(Lorenz96):
            pass
    elif modelflag == 'kuramoto':
        class ModelClass(Kuramoto):
            pass
    else: # Lorenz63
        class ModelClass(Lorenz63):
            pass

    print("Model being used is {}".format(ModelClass.who()))
    synthetic_name = "{}_synthetic".format(ModelClass.who())
    theta0=ModelClass.transformThetatoParameters(ModelClass.default_theta())

    actionflag = sys.argv[argctr]
    argctr += 1

    num_steps = 6400 #3200 #12800 #12800 #3200 #12800 # 1600
    #X0_ = ModelClass.initialState() #np.array([0,1,1.05])
    #X0_ = X0_[np.newaxis,:]
    #X0_mu = ModelClass.transformStatetoX(np.array([[0,1,1.05],[0,1,1.05]]))

    if actionflag == 't' or actionflag == 'r':
        if (len(sys.argv) > argctr):
            X = np.load(sys.argv[argctr])
            argctr += 1
            Y = np.load(sys.argv[argctr])
            argctr += 1
            num_steps = X.shape[0] #3200 #12800 # 1600
        else:
            X,Y = ModelClass.synthetic(dt=ModelClass.delta_t(),num_steps=num_steps)
            np.save("{}_X_{}".format(synthetic_name,timestr),X)
            np.save("{}_Y_{}".format(synthetic_name,timestr),Y)
        print("X(shape {}) = {}".format(X.shape,X))
        print("Y(shape {}) = {}".format(Y.shape,Y))

        X0_mu = ModelClass.transformStatetoX(X[np.newaxis,0,...])
        print("X0_mu = {}".format(X0_mu))
        print("Theta = {}".format(theta0))

        n=8192 #2048 #1024 #8192 #1024 #16384 #2048 #512
        chain_length=100

        #pf = stateFilter(ModelClass(),Y,n)
        #pf = ESSPartiallyAdaptedParticleFilter(ModelClass(),Y,n)
        pf = auxiliaryParticleFilter(ModelClass(),Y,n)

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
        ModelClass.plot_synthetic(X,Y,num_steps=num_steps)
        # Assert transformation is correct
        XtrX = ModelClass.transformXtoState(ModelClass.transformStatetoX(X))
        print("X = {}, XtrX = {}".format(X,XtrX))
        assert((np.abs(X - XtrX) < 0.00001).all())
        # print likelihood of true solution
        T = Y.shape[0]
        #testX = np.zeros((T,n,3))
        #testX[0,:] = X0_
        log_lh = np.zeros(T)
        for i in range(0,T):
           #testX[i,:] = ModelClass.transformXtoState(innov(ModelClass.transformStatetoX(testX[i,:]),np.array([10.,28.,2.667])))
           trx_=ModelClass.transformStatetoX(X[np.newaxis,i*ModelClass.obsinterval(),:])
           try_=Y[np.newaxis,i,:]
          
           log_lh[i] = pf.lh(trx_,try_,ModelClass.default_theta()) # last arg theta unused
           #print("log lh at i={} is {}".format(i,log_lh[i]))
           print("i={}, loglh = {}, X[i,:]={}, Y[i,:]={}".format(i,log_lh[i],ModelClass.transformStatetoX(X[np.newaxis,i*ModelClass.obsinterval(),:]),Y[i,:]))

        print("synthetic observations T={} have log lh sum = {}".format(T,log_lh.sum()))
        fig = plt.figure()
        plt.plot(log_lh)
        plt.show()
        #fig = plt.figure()
        #plt.plot(testX[:,:,0])
        #plt.show()
        ml_test = pf.test_filter(chain_length,X0_mu[0,:],ModelClass.transformParameterstoTheta(theta0))
        ml_test_log_mean = logsumexp(ml_test) - math.log(chain_length)
        ml_test_log_var = 2. * ml_test_log_mean + np.log(np.sum((np.exp(ml_test-ml_test_log_mean)-1)**2)) - math.log(chain_length)
        
        
        print("Log Marginal likelihood: Mean = {} Std Dev = {}".format(ml_test.mean(),ml_test.std()))
        print("Log Non-log: Mean = {} Std Dev = {}".format(ml_test_log_mean,ml_test_log_var * 0.5))
    
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.plot(ml_test,'y',linewidth=1)
        ax2.boxplot(ml_test)
        plt.show()

    elif actionflag == 'r':
        ModelClass.plot_synthetic(X,Y,num_steps)
        #pcov_in = 2. * np.array([[ 6.53672845e-07,  1.80943850e-06, -2.23494164e-06],
        #           [ 1.80943850e-06,  5.00872523e-06, -6.18656485e-06],
        #           [-2.23494164e-06, -6.18656485e-06,  7.64138236e-06]])
        burnin = 200
        initial_run = 400
        esttheta,ml,ar,pcov_out = sampler.run_pmmh(initial_run,X0_mu[0,:],np.array([0.,0.,0.]),ModelClass.transformParameterstoTheta(theta0)) #,pcov0=pcov_in)
        pcov_in = np.eye(3) * 0.0000001
        thetamean = np.mean(esttheta[burnin:,:],axis=0)
        covnorm = 1./(initial_run-burnin-1)
        for k in range(burnin,initial_run):
            pcov_in += np.outer(esttheta[k,:]-thetamean,esttheta[k,:]-thetamean)*covnorm
        print("P Covariance for second run = {}".format(pcov_in))

        estparams,ml,ar,pcov_out = sampler.run_pmmh(chain_length,X0_mu[0,:],np.array([0.,0.,0.]),ModelClass.transformParameterstoTheta(theta0),pcov0=pcov_in)
        print("Acceptance rate = {}".format(1.*ar.sum()/chain_length))
    
        for i in range(estparams.shape[0]):
            estparams[i,:]=ModelClass.transformThetatoParameters(estparams[i,:])
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
