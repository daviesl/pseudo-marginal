import numpy as np
from pmmcmc import pmpfl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

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

def innov(X,theta):
    global obsinterval
    dt=0.001 # TODO make DRY
    trth=tr_theta(theta)
    trX=X2tr(X)
    Xnext=np.zeros_like(trX)
    Xnext[:]=trX[:]
    for i in range(obsinterval): # TODO make this every 40th DRY
        (xdot,ydot,zdot) = lorenz(Xnext[:,0],Xnext[:,1],Xnext[:,2],trth[0],trth[1],trth[2])
        Xnext[:,0]=Xnext[:,0]+(xdot*dt) + np.random.normal(0,np.sqrt(dt),X.shape[0])
        Xnext[:,1]=Xnext[:,1]+(ydot*dt) + np.random.normal(0,np.sqrt(dt),X.shape[0])
        Xnext[:,2]=Xnext[:,2]+(zdot*dt) + np.random.normal(0,np.sqrt(dt),X.shape[0])
    rtXnext=tr2X(Xnext)
    return rtXnext

def tr_theta(theta):
    #return np.array([10,28,2.667])
    #uniform
    return theta*np.array([18.,36.,5]) + np.array([10,28,2.667]) - np.array([9,18,0.667])
    #Normal
    #return np.power(1.4,theta*2) + np.array([10,28,2.667]) - 1.
    #return theta + np.array([10,28,2.667])

def X2tr(X):
    # normal(0,1)
    #return np.column_stack((X[:,0]*20.0,X[:,1]*20,np.exp(X[:,2])*5+0.8))
    return np.column_stack((X[:,0]*20.0,X[:,1]*20,X[:,2]*5+0.8))
    # uniform(0,1)
    # return 100*(X-0.5)

def tr2X(tr):
    # normal(0,1)
    #return tr/20.0
    return np.column_stack((tr[:,0]/20.0,tr[:,1]/20,(tr[:,2]-0.8)/5))
    # uniform(0,1)
    # return X/100+0.5

# log likelihood
# TODO add parameters
def lh(X,y):
    # what is the scale of the observation noise? Assuming 1 in all x,y,z
    dim=y.shape[0]
    trX=X2tr(X)
    simy = obseqn(trX,sigma=0)
    d = (simy-y)*(1/np.sqrt(1))  # TODO find a way to DRY this obs error
    dd = np.sum(d**2,axis=1)
    return (-0.5*(dd)) #- np.log(np.sqrt((2*np.pi)**dim))


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
    Ys = obseqn(Xs[::obsinterval])
    return (Xs,Ys)

def plot_synthetic(Xs,Ys):
    ff,ax=plt.subplots()
    ax.plot(Xs[:,0],'r',Xs[:,1],'g',Xs[:,2],'b',lw=1)
    ax.plot(np.arange(0,num_steps+1,obsinterval),Ys[:,0],'r+',np.arange(0,num_steps+1,obsinterval),Ys[:,1],'b+',lw=1)
    plt.show()
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(Ys[:,0],Ys[:,1],linewidth=0.5)
    ax1.set_title("Observed Data")
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(Xs[:,0], Xs[:,1], Xs[:,2], lw=0.5)
    ax2.set_xlabel("X Axis")
    ax2.set_ylabel("Y Axis")
    ax2.set_zlabel("Z Axis")
    ax2.set_title("State")
    plt.show()

def obseqn(Xs,sigma=1.):
    dim = Xs.shape[0]
    Ys = np.zeros((dim,2))
    Ys[:,0] = Xs[:,0] + np.random.normal(0,np.sqrt(sigma),dim)
    Ys[:,1] = Xs[:,2] + np.random.normal(0,np.sqrt(sigma),dim)
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
    hax1.hist(estparams[:,0],color='b',orientation="horizontal")
    hax2.hist(estparams[:,1],color='g',orientation="horizontal")
    hax3.hist(estparams[:,2],color='r',orientation="horizontal")
    hax4.hist(ml,color='y',orientation="horizontal")
    plt.show()

if __name__ == '__main__':
    dt = 0.001
    num_steps = 12800
    X0_mu = tr2X(np.array([[0,1,1.05],[0,1,1.05]]))
    X,Y = synthetic(dt=dt,num_steps=num_steps)
    print(Y)
    
    n=2000
    chain_length=1000

    # run pmmh
    sampler = pmpfl(innov,lh,Y,3,3,n)

    if True:
        plot_synthetic(X,Y)
        ml_test = sampler.test_particlefilter(chain_length,X0_mu[0,:])
    
        print("Log Marginal likelihood: Mean = {} Std Dev = {}".format(ml_test.mean(),ml_test.std()))
    
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.plot(ml_test,'y',linewidth=1)
        ax2.boxplot(ml_test)
        plt.show()

    if False:
        plot_synthetic(X,Y)
        estparams,ml,ar = sampler.run_pmmh(chain_length,X0_mu[0,:],np.array([0.,0.,0.]))

        print("Acceptance rate = {}".format(1.*ar.sum()/chain_length))
    
        for i in range(estparams.shape[0]):
            estparams[i,:]=tr_theta(estparams[i,:])
        plot_traces(chain_length,estparams,ml)
