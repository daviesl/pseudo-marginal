import numpy as np
import math
import time
import numba
from numba import jit, float64
from pmmcmc import pmpfl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import sys

class lorenz_63_ssm(object):
    def __init__(self,dt=0.01,num_steps=10000,Xsigma=1.,Ysigma=1.,X0=np.array([0,1,1.05])):
        print("Dummy")
    def innov(self,X,theta):
        print("Dummy")
    def lh(sefl,X,y,theta):
        print("Dummy")
    def synthetic(self):
        print("Dummy")

obsinterval = 40
obserr = 0.1
dt = 0.001
@numba.jit
def obserr_mat():
    return np.array([[obserr**2,0],[0,obserr**2]])
@numba.jit
def obs_map():
    return np.array([[1, 0],
                    [0, 1]])



@numba.jit
def gopy(x,y,sigma=1.5,omega=0.5*(math.sqrt(5)-1)):
    x_dot = 2. * sigma * np.tanh(x) * np.cos(2.*math.pi*y)
    y_dot = np.mod(y + omega,1.)
    return (x_dot,y_dot)

#@numba.jit #("float64[:][:](float64[:][:],float64[:])")
def innov_diffusion_bridge(X_k,y_J,theta):
    """
    THe weight that we use for this modified diffusion bridge proposal 
    require the computation of both the proposal q and transition p densitites
    for w_i=\frac{p(y^i_t|x^i_t}p(x^i_t|x^i_{t-1})}{q(x^i_t|x^i_{t-1})}
    """
    global obsinterval
    global dt 
    trth=theta2tr(theta)
    trX=X2tr(X_k)
    Xnext=np.zeros_like(trX)
    Xnext[:]=trX[:]
    Xshape0 = X_k.shape[0]
    logpqratio=np.zeros(Xshape0)
    for i in range(obsinterval): # TODO make this every 40th DRY
        dk = (obsinterval - i) * dt
        P_k = np.linalg.inv(np.eye(2)*dk + obserr_mat()) # updated precision matrix
        P_k3 = np.dot(np.dot(obs_map(),P_k),obs_map().T)
        psi_mdb = np.eye(2) - P_k3 * dt
        # lorenz model
        (xdot,ydot) = gopy(Xnext[:,0],Xnext[:,1],trth[0],trth[1])
        a_t = np.column_stack((xdot,ydot)) # partitioned to observed and latent
        mu_mdb = a_t + np.dot(np.dot(P_k3,obs_map()),( y_J - np.dot(Xnext + a_t*dk,obs_map())).T).T
        dW_t = np.random.normal(0,1,X_k.shape)
        x_mu_mdb = Xnext + mu_mdb*dt
        x_mu_em = Xnext + a_t*dt
        Xnext[:] = x_mu_mdb + np.dot(np.sqrt(dt*psi_mdb),dW_t.T).T
        logpqratio[:] += logmvnorm_vectorised(Xnext,x_mu_em,np.eye(2)*dt) - logmvnorm_vectorised(Xnext,x_mu_mdb,psi_mdb*dt)
    rtXnext=tr2X(Xnext)
    return rtXnext, logpqratio
        
#@numba.jit("float64[:](float64[:],float64[:],float64[:][:])")
#def logmvnorm_vectorised(X,mu,cov):
#    return -0.5 * math.log(2. * math.pi) - 0.5 * math.log(np.linalg.det(cov)) - 0.5 * np.dot(np.dot((X - mu).T,np.linalg.inv(cov)),(X-mu))

@numba.jit #("float64[:](float64[:][:],float64[:][:],float64[:][:])")
def logmvnorm_vectorised(X,mu,cov):
    return -0.5 * math.log(2. * math.pi) - 0.5 * math.log(np.linalg.det(cov)) - 0.5 * np.sum(np.dot(X - mu,np.linalg.inv(cov)) * (X-mu), axis=1)


@numba.jit #("float64[:][:](float64[:][:],float64[:])")
def innov(X,y_J,theta):
    global obsinterval
    global dt #=0.001 # TODO make DRY
    trth=theta2tr(theta)
    trX=X2tr(X)
    Xnext=np.zeros_like(trX)
    Xnext[:]=trX[:]
    for i in range(obsinterval): # TODO make this every 40th DRY
        (xdot,ydot) = gopy(Xnext[:,0],Xnext[:,1],trth[0],trth[1])
        Xnext[:,0]=Xnext[:,0]+(xdot*dt) + np.random.normal(0,np.sqrt(dt),X.shape[0])
        Xnext[:,1]=Xnext[:,1]+(ydot*dt) + np.random.normal(0,np.sqrt(dt),X.shape[0])
    rtXnext=tr2X(Xnext)
    return rtXnext, np.zeros(X.shape[0])


@numba.jit("float64[:](float64[:])")
def theta2tr(theta):
    target = np.array([1.5,0.5*(math.sqrt(5.)-1.)])
    lower = target * 0.2
    upper = target + (target - lower)
    return theta*(upper-lower) + lower

@numba.jit("float64[:](float64[:])")
def tr2theta(nt):
    target = np.array([1.5,0.5*(math.sqrt(5.)-1.)])
    lower = target * 0.2
    upper = target + (target - lower)
    return (nt - lower)/(upper-lower)


@numba.jit #("float64[:][:](float64[:][:])")
def X2tr(X):
    return np.column_stack((X[:,0]*5.,X[:,1]+0.5))

@numba.jit #("float64[:][:](float64[:][:])")
def tr2X(tr):
    return np.column_stack((tr[:,0]/5.0,tr[:,1]-0.5))

@numba.jit #("float64[:][:](float64[:][:],float64[:][:])")
def xTPx(x,P):
    return np.sum(np.dot(x,P)*x,axis=1)

# log likelihood
# TODO add parameters
@numba.jit #("float64[:](float64[:][:],float64[:],float64[:],float64[:][:])")
def lh(X,y,theta,cov=np.array([[obserr**2,0],[0,obserr**2]])):
    # what is the scale of the observation noise? Get dimension of Y
    dim=y.shape[-1]
    #print("Dim  == {}".format(dim))
    trX=X2tr(X)
    y_star = obseqn(trX)
    d = y_star-y
    dd = xTPx(d,np.linalg.inv(cov))
    # chris says we need to compute the normalising constant
    log_lh_un = (-0.5*(dd)) 
    #print("DD = {}".format(log_lh_un))
    norm_const = - (dim*0.5) * np.log(2*np.pi) - 0.5 * np.log(np.linalg.det(cov))
    #norm_const = - (dim*0.5) * np.log(2*np.pi) - 0.5 * np.log(np.linalg.det(cov))
    #print("Norm const = {}".format(norm_const))
    #print("x = {}, trx = {}, y = {}, ystar = {}, y-ystar = {}, log_lh_un = {}, norm const = {}".format(X, trX,y,y_star,d, log_lh_un, norm_const))
    #print(log_lh_un + norm_const)
    return log_lh_un + norm_const

# write pseudocode in latex for chris





def synthetic(dt=0.001,num_steps=10000,x0=0.,y0=1.,xW=1.,yW=1.,xO=1.,yO=1.):
    global obsinterval
    # Need one more for the initial values
    Xs = np.empty((num_steps + 1,2))
    Ys = np.empty((num_steps + 1,2))
    
    # Set initial values
    Xs[0,0], Xs[0,1] = (x0,y0)
    
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot = gopy(Xs[i,0], Xs[i,1])
        Xs[i + 1,0] = Xs[i,0] + (x_dot * dt)+ xW*np.random.normal(0,np.sqrt(dt)) 
        Xs[i + 1,1] = Xs[i,1] + (y_dot * dt)+ yW*np.random.normal(0,np.sqrt(dt))
    Ys = obseqn_with_noise(Xs[::obsinterval],np.array([[obserr**2,0],[0,obserr**2]]))
    return (Xs,Ys)

def plot_synthetic(Xs,Ys):
    ff,ax=plt.subplots()
    #ax.plot(Xs[:,0],'r',Xs[:,1],'g',Xs[:,2],'b',lw=1)
    ax.plot(Xs[:,0],'r',label="x",lw=1)
    ax.plot(Xs[:,1],'g',label="y",lw=1)
    ax.plot(np.arange(0,num_steps+1,obsinterval),Ys[:,0],'r+',np.arange(0,num_steps+1,obsinterval),Ys[:,1],'g+',label="Observed",lw=1)
    ax.legend()
    plt.show()
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(Ys[:,0],Ys[:,1],linewidth=0.5)
    ax1.set_title("Observed Data")
    ax1.set_xlabel("X Axis")
    ax1.set_ylabel("Y Axis")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(Xs[:,0], Xs[:,1], lw=0.5)
    ax2.set_xlabel("X Axis")
    ax2.set_ylabel("Y Axis")
    ax2.set_title("State")
    plt.show()

@numba.jit
def obseqn(Xs): #,sigma=np.array([obserr,obserr])):
    dim = Xs.shape[0]
    Ys = np.zeros((dim,2))
    Ys[:,0] = Xs[:,0]
    Ys[:,1] = Xs[:,1]
    return Ys

def obseqn_with_noise(Xs,cov=np.array([[obserr**2,0],[0,obserr**2]])):
    dim = cov.shape[0]
    Ys = obseqn(Xs) + np.random.multivariate_normal(np.zeros(dim),cov,size=Xs.shape[0])
    return Ys

def plot_traces(chain_length,estparams,ml):
    tS=np.ones(chain_length)*10
    tR=np.ones(chain_length)*28
    tB=np.ones(chain_length)*2.667
    
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

    num_steps = 12800 #12800 #3200 #12800 # 1600
    X0_ = np.array([0,1,1.05])
    X0_ = X0_[np.newaxis,:]
    #X0_mu = tr2X(np.array([[0,1,1.05],[0,1,1.05]]))
    X0_mu = tr2X(X0_)
    print("X0_mu = {}".format(X0_mu))



    if actionflag == 't' or actionflag == 'r':
        if (len(sys.argv) > 2):
            X = np.load(sys.argv[argctr])
            argctr += 1
            Y = np.load(sys.argv[argctr])
            argctr += 1
            num_steps = X.shape[0] #3200 #12800 # 1600
        else:
            X,Y = synthetic(dt=dt,num_steps=num_steps)
            np.save("synthetic_X_{}".format(timestr),X)
            np.save("synthetic_Y_{}".format(timestr),Y)
        print("Y = {}".format(Y))

        n=1024 #8192 #1024 #16384 #2048 #512
        chain_length=1000

        # run pmmh
        sampler = pmpfl(innov_diffusion_bridge,lh,Y,2,2,n)
        #sampler = pmpfl(innov,lh,Y,2,2,n)

    if actionflag == 't':
        plot_synthetic(X,Y)
        # Assert transformation is correct
        XtrX = X2tr(tr2X(X))
        print("X = {}, XtrX = {}".format(X,XtrX))
        assert((np.abs(X - XtrX) < 0.00001).all())
        # print likelihood of true solution
        T = Y.shape[0]
        #testX = np.zeros((T,n,3))
        #testX[0,:] = X0_
        log_lh = np.zeros(T)
        for i in range(0,T):
           #testX[i,:] = X2tr(innov(tr2X(testX[i,:]),np.array([10.,28.,2.667])))
           log_lh[i] = lh(tr2X(X[np.newaxis,i*obsinterval,:]),Y[np.newaxis,i,:],np.zeros(2)) # last arg theta unused
           #print("log lh at i={} is {}".format(i,log_lh[i]))
           print("i={}, loglh = {}, X[i,:]={}, Y[i,:]={}".format(i,log_lh[i],tr2X(X[np.newaxis,i*obsinterval,:]),Y[i,:]))

        print("synthetic observations T={} have log lh sum = {}".format(T,log_lh.sum()))
        fig = plt.figure()
        plt.plot(log_lh)
        plt.show()
        #fig = plt.figure()
        #plt.plot(testX[:,:,0])
        #plt.show()
        ml_test = sampler.test_particlefilter(chain_length,X0_mu[0,:],tr2theta(np.array([1.5,0.5*(math.sqrt(5.)-1)])))
    
        print("Log Marginal likelihood: Mean = {} Std Dev = {}".format(ml_test.mean(),ml_test.std()))
    
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
        esttheta,ml,ar,pcov_out = sampler.run_pmmh(initial_run,X0_mu[0,:],np.array([0.,0.]),tr2theta(np.array([1.5,0.5*(math.sqrt(5.)-1)]))) #,pcov0=pcov_in)
        pcov_in = np.eye(2) * 0.0000001
        thetamean = np.mean(esttheta[burnin:,:],axis=0)
        covnorm = 1./(initial_run-burnin-1)
        for k in range(burnin,initial_run):
            pcov_in += np.outer(esttheta[k,:]-thetamean,esttheta[k,:]-thetamean)*covnorm
        print("P Covariance for second run = {}".format(pcov_in))

        estparams,ml,ar,pcov_out = sampler.run_pmmh(chain_length,X0_mu[0,:],np.array([0.,0.]),tr2theta(np.array([1.5,0.5*(math.sqrt(5.)-1)])),pcov0=pcov_in)
        print("Acceptance rate = {}".format(1.*ar.sum()/chain_length))
    
        for i in range(estparams.shape[0]):
            estparams[i,:]=theta2tr(estparams[i,:])
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
