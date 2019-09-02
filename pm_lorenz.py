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


@jit #("UniTuple(float64[:],3)(float64[:],float64[:],float64[:],float64,float64,float64)")
def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s*(y - x) 
    y_dot = r*x - y - x*z 
    z_dot = x*y - b*z 
    return x_dot, y_dot, z_dot

obsinterval = 40
obserr = 0.1
dt = 0.001
@numba.jit("f8[:,:]()")
def obserr_mat():
    global obserr
    return np.array([[obserr**2,0],[0,obserr**2]])
@numba.jit("f8[:,:]()")
def obs_map():
    return np.array([[1., 0.],
                    [0., 0.],
                    [0., 1.]],dtype=np.float64)


@numba.jit
def innov_lindstrom_residual_bridge(X_k,y_J,theta):
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
    _N_ = X_k.shape[0]
    dim = X_k.shape[1]
    eta=np.zeros((_N_,dim,obsinterval+1))
    #eta[:,:,0]=trX[:,:]
    last_eta = np.zeros_like(trX)
    last_eta[:]=trX[:]
    logpqratio=np.zeros(_N_)
    beta_all = np.eye(3) # process noise on all variables
    beta_xx = np.eye(2) # process noise on observed variables TODO make F'beta F
    gamma = 0.1
    #print ("X_k.shape = {}, eta.shape = {}, last_eta.shape = {}".format(X_k.shape,eta.shape,last_eta.shape))
    # compute eta, that is the SDE without the stochastic part.
    for i in range(obsinterval+1): # TODO make this every 40th DRY
        (xdot,ydot,zdot) = lorenz(last_eta[:,0],last_eta[:,1],last_eta[:,2],trth[0],trth[1],trth[2])
        a_t = np.column_stack((xdot,ydot,zdot)) 
        #print("a_t.shape = {}".format(a_t.shape))
        eta[:,:,i] = last_eta + a_t*dt
        last_eta[:] = eta[:,:,i]
    
    for i in range(obsinterval): # TODO make this every 40th DRY
        dk = (obsinterval - i) * dt
        dk1 = (obsinterval - (i+1)) * dt
        C = (gamma/dt) * beta_xx 
        P_k = np.linalg.inv(beta_xx*dk + C*(dk1**2) + obserr_mat()) # updated precision matrix
        #P_k = np.linalg.inv(beta_xx*dk + obserr_mat()) # updated precision matrix
        r = Xnext - eta[:,:,i] # eta[...,:]
        chord = (eta[...,i+1]-eta[...,i])/dt
        P_k3 = np.dot(np.dot(obs_map(),P_k),obs_map().T)
        psi_mdb = beta_all - P_k3 * dt
        # lorenz model
        (xdot,ydot,zdot) = lorenz(Xnext[:,0],Xnext[:,1],Xnext[:,2],trth[0],trth[1],trth[2])
        a_t = np.column_stack((xdot,ydot,zdot)) 
        mu_rb = a_t + np.dot(np.dot(P_k3,obs_map()),( y_J - np.dot(eta[...,obsinterval-1] + r + (a_t-chord)*dk,obs_map())).T).T
        dW_t = np.random.normal(0,1,X_k.shape)
        x_mu_rb = Xnext + mu_rb*dt
        x_mu_em = Xnext + a_t*dt
        Xnext[:] = x_mu_rb + np.dot(np.sqrt(dt*psi_mdb),dW_t.T).T
        logpqratio[:] += logmvnorm_vectorised(Xnext,x_mu_em,beta_all*dt) - logmvnorm_vectorised(Xnext,x_mu_rb,psi_mdb*dt)
    rtXnext=tr2X(Xnext)
    return rtXnext, logpqratio

@numba.jit #("float64[:][:](float64[:][:],float64[:])")
def innov_residual_bridge(X_k,y_J,theta):
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
    _N_ = X_k.shape[0]
    dim = X_k.shape[1]
    eta=np.zeros((_N_,dim,obsinterval+1))
    #eta[:,:,0]=trX[:,:]
    last_eta = np.zeros_like(trX)
    last_eta[:]=trX[:]
    logpqratio=np.zeros(_N_)
    #print ("X_k.shape = {}, eta.shape = {}, last_eta.shape = {}".format(X_k.shape,eta.shape,last_eta.shape))
    # compute eta, that is the SDE without the stochastic part.
    for i in range(obsinterval+1): # TODO make this every 40th DRY
        (xdot,ydot,zdot) = lorenz(last_eta[:,0],last_eta[:,1],last_eta[:,2],trth[0],trth[1],trth[2])
        a_t = np.column_stack((xdot,ydot,zdot)) 
        #print("a_t.shape = {}".format(a_t.shape))
        eta[:,:,i] = last_eta + a_t*dt
        last_eta[:] = eta[:,:,i]
    
    for i in range(obsinterval): # TODO make this every 40th DRY
        dk = (obsinterval - i) * dt
        r = Xnext - eta[:,:,i] # eta[...,:]
        chord = (eta[...,i+1]-eta[...,i])/dt
        P_k = np.linalg.inv(np.eye(2)*dk + obserr_mat()) # updated precision matrix
        P_k3 = np.dot(np.dot(obs_map(),P_k),obs_map().T)
        psi_mdb = np.eye(3) - P_k3 * dt
        # lorenz model
        (xdot,ydot,zdot) = lorenz(Xnext[:,0],Xnext[:,1],Xnext[:,2],trth[0],trth[1],trth[2])
        a_t = np.column_stack((xdot,ydot,zdot)) 
        mu_rb = a_t + np.dot(np.dot(P_k3,obs_map()),( y_J - np.dot(eta[...,obsinterval-1] + r + (a_t-chord)*dk,obs_map())).T).T
        dW_t = np.random.normal(0,1,X_k.shape)
        x_mu_rb = Xnext + mu_rb*dt
        x_mu_em = Xnext + a_t*dt
        Xnext[:] = x_mu_rb + np.dot(np.sqrt(dt*psi_mdb),dW_t.T).T
        logpqratio[:] += logmvnorm_vectorised(Xnext,x_mu_em,np.eye(3)*dt) - logmvnorm_vectorised(Xnext,x_mu_rb,psi_mdb*dt)
    rtXnext=tr2X(Xnext)
    return rtXnext, logpqratio

@numba.jit #("float64[:][:](float64[:][:],float64[:])")
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
        #psi_mdb_old = np.eye(3) - np.column_stack((np.vstack((P_k,np.zeros((1,2)))),np.zeros(3))) * dt
        psi_mdb = np.eye(3) - P_k3 * dt
        # lorenz model
        (xdot,ydot,zdot) = lorenz(Xnext[:,0],Xnext[:,1],Xnext[:,2],trth[0],trth[1],trth[2])
        a_t = np.column_stack((xdot,ydot,zdot)) # partitioned to observed and latent
        mu_mdb = a_t + np.dot(np.dot(P_k3,obs_map()),( y_J - np.dot(Xnext + a_t*dk,obs_map())).T).T
        dW_t = np.random.normal(0,1,X_k.shape)
        x_mu_mdb = Xnext + mu_mdb*dt
        x_mu_em = Xnext + a_t*dt
        Xnext[:] = x_mu_mdb + np.dot(np.sqrt(dt*psi_mdb),dW_t.T).T
        logpqratio[:] += logmvnorm_vectorised(Xnext,x_mu_em,np.eye(3)*dt) - logmvnorm_vectorised(Xnext,x_mu_mdb,psi_mdb*dt)
    rtXnext=tr2X(Xnext)
    return rtXnext, logpqratio

@numba.jit("f8[:,:](f8[:,:],f8[:,:])")
def doubledot(outer,inner):
    return np.dot(np.dot(outer,inner),outer.T)

@numba.jit #("float64[:][:](float64[:][:],float64[:])")
def innov_lindstrom_bridge(X_k,y_J,theta):
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
    beta_all = np.eye(3) # process noise on all variables
    beta_xx = np.eye(2) # process noise on observed variables TODO make F'beta F
    gamma = 0.5
    for i in range(obsinterval): # TODO make this every 40th DRY
        dk = (obsinterval - i) * dt
        dk1 = (obsinterval - (i+1)) * dt
        C = (gamma/dt) * beta_xx 
        P_k = np.linalg.inv(beta_xx*dk + C*(dk1**2) + obserr_mat()) # updated precision matrix
        P_k3 = doubledot(obs_map(),P_k) #np.dot(np.dot(obs_map(),P_k),obs_map().T)
        #psi_mdb_old = np.eye(3) - np.column_stack((np.vstack((P_k,np.zeros((1,2)))),np.zeros(3))) * dt
        psi_mdb = beta_all - P_k3 * dt
        # lorenz model
        (xdot,ydot,zdot) = lorenz(Xnext[:,0],Xnext[:,1],Xnext[:,2],trth[0],trth[1],trth[2])
        a_t = np.column_stack((xdot,ydot,zdot)) # partitioned to observed and latent
        mu_mdb = a_t + np.dot(np.dot(P_k3,obs_map()),( y_J - np.dot(Xnext + a_t*dk,obs_map())).T).T
        dW_t = np.random.normal(0,1,X_k.shape)
        x_mu_mdb = Xnext + mu_mdb*dt
        x_mu_em = Xnext + a_t*dt
        Xnext[:] = x_mu_mdb + np.dot(np.sqrt(dt*psi_mdb),dW_t.T).T
        logpqratio[:] += logmvnorm_vectorised(Xnext,x_mu_em,beta_all*dt) - logmvnorm_vectorised(Xnext,x_mu_mdb,psi_mdb*dt)
    rtXnext=tr2X(Xnext)
    return rtXnext, logpqratio

#@numba.jit("float64[:](float64[:],float64[:],float64[:][:])")
#def logmvnorm_vectorised(X,mu,cov):
#    return -0.5 * math.log(2. * math.pi) - 0.5 * math.log(np.linalg.det(cov)) - 0.5 * np.dot(np.dot((X - mu).T,np.linalg.inv(cov)),(X-mu))

@numba.jit #("float64[:](float64[:][:],float64[:][:],float64[:][:])")
def logmvnorm_vectorised(X,mu,cov):
    return -0.5 * math.log(2. * math.pi) - 0.5 * math.log(np.linalg.det(cov)) - 0.5 * np.sum(np.dot(X - mu,np.linalg.inv(cov)) * (X-mu), axis=1)

#    """
#    TODO assert last dimension of X is same as cov
#         if dim of X is 1, do X.T cov^-1 X
#         else                 X cov^-1 X.T
#    """
#    if len(X.shape) > 1:
#        dd = np.sum(np.dot(X - mu,np.linalg.inv(cov)) * (X-mu), axis=1)
#    else:
#        dd = np.dot(np.dot((X - mu).T,np.linalg.inv(cov)),(X-mu))
#    return -0.5 * math.log(2. * math.pi) - 0.5 * math.log(np.linalg.det(cov)) - 0.5 * dd

    
@numba.jit
def propagate_noisefree(X,y_J,theta):
    global obsinterval
    global dt #=0.001 # TODO make DRY
    trth=theta2tr(theta)
    trX=X2tr(X)
    Xnext=np.zeros_like(trX)
    Xnext[:]=trX[:]
    for i in range(obsinterval): # TODO make this every 40th DRY
        (xdot,ydot,zdot) = lorenz(Xnext[:,0],Xnext[:,1],Xnext[:,2],trth[0],trth[1],trth[2])
        Xnext[:,0]=Xnext[:,0]+(xdot*dt) 
        Xnext[:,1]=Xnext[:,1]+(ydot*dt)
        Xnext[:,2]=Xnext[:,2]+(zdot*dt)
    rtXnext=tr2X(Xnext)
    return rtXnext, np.zeros(X.shape[0])

@numba.jit #("float64[:][:](float64[:][:],float64[:])")
def innov(X,y_J,theta):
    global obsinterval
    global dt #=0.001 # TODO make DRY
    trth=theta2tr(theta)
    trX=X2tr(X)
    Xnext=np.zeros_like(trX)
    Xnext[:]=trX[:]
    #print("Xnext_pre = {}".format(Xnext))
    #print("Xnext_pre std dev = {}".format(np.std(Xnext,axis=0)))
    for i in range(obsinterval): # TODO make this every 40th DRY
        (xdot,ydot,zdot) = lorenz(Xnext[:,0],Xnext[:,1],Xnext[:,2],trth[0],trth[1],trth[2])
        Xnext[:,0]=Xnext[:,0]+(xdot*dt) + np.random.normal(0,np.sqrt(dt),X.shape[0])
        Xnext[:,1]=Xnext[:,1]+(ydot*dt) + np.random.normal(0,np.sqrt(dt),X.shape[0])
        Xnext[:,2]=Xnext[:,2]+(zdot*dt) + np.random.normal(0,np.sqrt(dt),X.shape[0])
        #print("Xnext = {}".format(Xnext))
        #print("Xnext mean = {}".format(np.mean(Xnext,axis=0)))
        #print("Xnext std dev = {}".format(np.std(Xnext,axis=0)))
    rtXnext=tr2X(Xnext)
    return rtXnext, np.zeros(X.shape[0])


@numba.jit("float64[:](float64[:])")
def theta2tr(theta):
    #return np.array([10,28,2.667])
    target = np.array([10.,28.,2.667])
    #target = np.array([20.,40.,10.])
    lower = target * 0.2
    upper = target + (target - lower)
    #uniform
    return theta*(upper-lower) + lower
    #Normal
    #return np.power(1.4,theta*2) + np.array([10,28,2.667]) - 1.
    #return theta + np.array([10,28,2.667])

def tr2theta(nt):
    #target = np.array([20.,40.,10.])
    target = np.array([10.,28.,2.667])
    lower = target * 0.2
    upper = target + (target - lower)
    return (nt - lower)/(upper-lower)


@numba.jit #("float64[:][:](float64[:][:])")
def X2tr(X):
    return np.column_stack((X[:,0]*20.0,X[:,1]*20.0,X[:,2]*5.+0.8))

@numba.jit #("float64[:][:](float64[:][:])")
def tr2X(tr):
    return np.column_stack((tr[:,0]/20.0,tr[:,1]/20.0,(tr[:,2]-0.8)/5.))

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





def synthetic(dt=0.001,num_steps=10000,x0=0.,y0=1.,z0=1.05,xW=1.,yW=1.,zW=1.,xO=1.,yO=1.,zO=1.):
    global obsinterval
    # Need one more for the initial values
    Xs = np.empty((num_steps + 1,3))
    Ys = np.empty((num_steps + 1,2))
    
    # Set initial values
    Xs[0,0], Xs[0,1], Xs[0,2] = (x0,y0,z0)
    
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(Xs[i,0], Xs[i,1], Xs[i,2])
        Xs[i + 1,0] = Xs[i,0] + (x_dot * dt)+ xW*np.random.normal(0,np.sqrt(dt)) 
        Xs[i + 1,1] = Xs[i,1] + (y_dot * dt)+ yW*np.random.normal(0,np.sqrt(dt))
        Xs[i + 1,2] = Xs[i,2] + (z_dot * dt)+ zW*np.random.normal(0,np.sqrt(dt))
    Ys = obseqn_with_noise(Xs[::obsinterval],np.array([[obserr**2,0],[0,obserr**2]]))
    return (Xs,Ys)

def plot_synthetic(Xs,Ys):
    ff,ax=plt.subplots()
    #ax.plot(Xs[:,0],'r',Xs[:,1],'g',Xs[:,2],'b',lw=1)
    ax.plot(Xs[:,0],'r',label="x",lw=1)
    ax.plot(Xs[:,1],'g',label="y",lw=1)
    ax.plot(Xs[:,2],'b',label="z",lw=1)
    ax.plot(np.arange(0,num_steps+1,obsinterval),Ys[:,0],'r+',np.arange(0,num_steps+1,obsinterval),Ys[:,1],'b+',label="Observed",lw=1)
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

@numba.jit
def obseqn(Xs): #,sigma=np.array([obserr,obserr])):
    dim = Xs.shape[0]
    Ys = np.zeros((dim,2))
    Ys[:,0] = Xs[:,0] #+ np.random.normal(0,np.sqrt(sigma[0]),dim)
    Ys[:,1] = Xs[:,2] #+ np.random.normal(0,np.sqrt(sigma[1]),dim)
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

    num_steps = 3200 #12800 #12800 #3200 #12800 # 1600
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

        n=2048 #1024 #8192 #1024 #16384 #2048 #512
        chain_length=1000

        # run pmmh
        #sampler = pmpfl(innov_lindstrom_bridge,innov_lindstrom_bridge,lh,Y,3,3,n)
        #sampler = pmpfl(innov,innov_lindstrom_bridge,lh,Y,3,3,n)
        #sampler = pmpfl(innov_lindstrom_bridge,lh,Y,3,3,n)
        #sampler = pmpfl(innov_diffusion_bridge,lh,Y,3,3,n)
        #sampler = pmpfl(innov_residual_bridge,propagate_noisefree,lh,Y,3,3,n)
        #sampler = pmpfl(innov_lindstrom_residual_bridge,lh,Y,3,3,n)
        sampler = pmpfl(innov,propagate_noisefree,lh,Y,3,3,n)
        #sampler = pmpfl(innov_residual_bridge,innov,lh,Y,3,3,n)
        #sampler = pmpfl(innov,innov_lindstrom_residual_bridge,lh,Y,3,3,n)
        #sampler = pmpfl(innov,innov,lh,Y,3,3,n)

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
           log_lh[i] = lh(tr2X(X[np.newaxis,i*obsinterval,:]),Y[np.newaxis,i,:],np.zeros(3)) # last arg theta unused
           #print("log lh at i={} is {}".format(i,log_lh[i]))
           print("i={}, loglh = {}, X[i,:]={}, Y[i,:]={}".format(i,log_lh[i],tr2X(X[np.newaxis,i*obsinterval,:]),Y[i,:]))

        print("synthetic observations T={} have log lh sum = {}".format(T,log_lh.sum()))
        fig = plt.figure()
        plt.plot(log_lh)
        plt.show()
        #fig = plt.figure()
        #plt.plot(testX[:,:,0])
        #plt.show()
        ml_test = sampler.test_particlefilter(chain_length,X0_mu[0,:],tr2theta(np.array([10.,28.,2.667])))
    
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
        esttheta,ml,ar,pcov_out = sampler.run_pmmh(initial_run,X0_mu[0,:],np.array([0.,0.,0.]),tr2theta(np.array([10.,28.,2.667]))) #,pcov0=pcov_in)
        pcov_in = np.eye(3) * 0.0000001
        thetamean = np.mean(esttheta[burnin:,:],axis=0)
        covnorm = 1./(initial_run-burnin-1)
        for k in range(burnin,initial_run):
            pcov_in += np.outer(esttheta[k,:]-thetamean,esttheta[k,:]-thetamean)*covnorm
        print("P Covariance for second run = {}".format(pcov_in))

        estparams,ml,ar,pcov_out = sampler.run_pmmh(chain_length,X0_mu[0,:],np.array([0.,0.,0.]),tr2theta(np.array([10.,28.,2.667])),pcov0=pcov_in)
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
