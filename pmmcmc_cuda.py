from numba import cuda 
import numpy as np
from scipy.stats import norm
import math
import sys



# a pseudo-marginal based sampler for a particle filter
# the innovation function will be generalised
class pmpfl(object):
    def __init__(self,innov,lh,y_all,X_size,theta_size,n):
        """
        innov : innovation function, progress X(t) given X(t-1) and theta
        yall : list of data vectors y_1,...,y_T
        X_size : size of state variable X vector
        theta_size : size of parameter X vector
        n : particle cloud size

        instance variables
        self.X_all : X_0,...,X_T of nxX_size arrays
        self.X_ancestry : indices at each time step 
                          of parent at previous time step
        self.w_all : importance weights- NORMALISED
        """
        # TODO make the below all 3D numpy arrays. Make theta a 3D array to account for "particles"
        assert(n>0)
        T = y_all.shape[0]+1
        self.X_size = X_size
        self.y_all = y_all
        self.theta_size = theta_size
        self.X_all = np.zeros((T,n,X_size))
        self.X_ancestry = np.tile(np.arange(n),(T,1))
        self.w_all = np.ones((T,n))/n
        self.T = T
        self.n = n
        self.innov = innov
        self.lh = lh
    def test_particlefilter_cpu(self,num_runs,X0):
        theta=np.zeros((self.theta_size))
        logml_chain=np.zeros((num_runs))
        for j in range(num_runs):
            self.X_all[0,:] = X0
            log_ml = np.zeros(self.T)
            for i in range(1,self.T):
                log_ml[i] = particlefilter_cpu(self.y_all[i-1,:],self.X_all[i,:],self.X_all[i-1,:],theta[:],self.X_ancestry[i,:],self.w_all[i,:],self.n,self.lh,self.innov)
            logml_chain[j] = log_ml.sum() # product of all marignal likelihoods
        return logml_chain
            
        
    def run_pmmh(self,mcmc_chain_size,X0_mu,X0_sigma):
        #seq = zip(self.X_all,self.y_all,self.X_ancestry, self.w_all)
        # init priors, data is unseen at t=0
        theta = np.zeros((mcmc_chain_size,self.theta_size))
        ar = np.zeros(mcmc_chain_size)
        logml_chain=np.ones((mcmc_chain_size))
        logml_chain[:]=np.finfo(float).min
        # don't store all of the priors, just the last one
        prior_j=0
        prior_j_1=0
        for j in range(mcmc_chain_size):
            if j>0:
                #propose theta?
                theta[j,:]=theta[j-1,:]+np.random.normal(0,0.2,self.theta_size) # assuming the innov function scales the parameters appropriately.
            # use a uniform prior
            prior_j = float(np.all(theta[j,:] <= 1) * np.all(theta[j,:] >= 0))
            #prior_j = norm.pdf(theta[j,:]).prod()
            #print("Prior[j] = {}".format(prior_j))
            # within support of prior? If not, continue.
            if prior_j == 0:
                theta[j,:] = theta[j-1,:] 
                ar[j]=0
                if j>0:
                    logml_chain[j]=logml_chain[j-1]
                continue
            
            # init priors of state variables. Innovation fn must translate from [0,1] to correct range.
            #self.X_all[0,:] = np.random.uniform(0,1,(self.n,self.X_size))
            # init priors of state variables. Innovation fn must translate from N(0,1) to correct range.
            self.X_all[0,:] = np.random.normal(X0_mu,X0_sigma,(self.n,self.X_size))
            #print(self.X_all[0,:])
            #print("Init prior to unif(0,1)")
            #print(self.X_all[0,:])
            log_ml = np.zeros(self.T)
            for i in range(1,self.T):
                log_ml[i] = particlefilter_cpu(self.y_all[i-1,:],self.X_all[i,:],self.X_all[i-1,:],theta[j,:],self.X_ancestry[i,:],self.w_all[i,:],self.n,self.lh,self.innov)
            # compute marginal likelihood of particle filter
            logml_chain[j] = log_ml.sum() # product of all marignal likelihoods
            ##print("1_Marginal likelihood at {} is {}".format(j, ml_j))
            if j > 0:
                #acceptance ratio
                #log_a = min(0,logml_chain[j]+np.log(prior_j_1)-logml_chain[j-1]-np.log(prior_j)) # assuming normal priors on parameters
                log_a = min(0,logml_chain[j]+np.log(prior_j_1)-logml_chain[j-1]-np.log(prior_j)) # assuming normal priors on parameters
                # compute unif(0,1)
                log_u = np.log(np.random.uniform(0,1))
                if log_a>=log_u:
                    logml_chain[j-1] = logml_chain[j]
                    # keep the proposed theta and X_all
                    ar[j]=1
                else:
                    # backtrack to the last theta and state X_all TODO do I need to store last state?
                    theta[j,:] = theta[j-1,:] 
                    ar[j]=0
            else:
                logml_chain[j-1] = logml_chain[j]
                prior_j_1 = prior_j
            print("Theta at {} is {} {} {}".format(j,theta[j,0],theta[j,1],theta[j,2]))
            print("Marginal likelihood at {} is {}".format(j, logml_chain[j]))
        return (theta,logml_chain,ar)

#    def run_pmgibbs(self,mcmc_chain_size):
#        # ancestry trace the final target distrubution particles and sum the weights for each
        
def particlefilter_cpu(yi,Xi,Xi_1,theta,Xianc,wi,n,lh,innov):
    Xi[:] = innov(Xi_1,theta) # TODO the theta here assumes a static timestep. We'll need to record it somehow in the pmpfl main function.
    loglh = lh(Xi,yi)
    if not np.isfinite(loglh).all():
        print("weights not finite")
        nanidx = np.argwhere(~np.isfinite(loglh))
        print (nanidx.shape,loglh[nanidx].shape,np.squeeze(Xi[nanidx,:]).shape)
        print (np.hstack((nanidx,wi[nanidx],np.squeeze(Xi[nanidx,:]),np.squeeze(Xi_1[nanidx,:]))))
        sys.exit(0)
    #loglh = np.nan_to_num(loglh)
    logml = logsumexp_cpu(loglh) - np.log(n)
    #print("log marginal likelihood = {}".format(logml))
    wi[:] = np.exp(loglh - logsumexp_cpu(loglh)) # normalise weights
    if wi.sum() == 0:
        print("Divide by zero sum of weights")
        #sys.exit(0)
    #print("n = {}, ess = {}".format(n,1./sum(wi**2)))
    if 1./sum(wi**2) < 0.5*n:
        #print("Resampling, ess = {}".format(1./sum(wi**2)))
        Xianc[:]=resample_cpu(wi)
        Xi[:]=Xi[Xianc,:]
    return logml

def resample_cpu(weights):
    n = weights.shape[0]
    indices = np.zeros_like(weights)
    #C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    C = np.cumsum(weights)
    u0, j = np.random.uniform(), 0
    for i in range(n):
        u = (u0+i)/n
        while u > C[j]:
            j+=1
        indices[i] = j
    return indices

def logsumexp_cpu(ns):
    m = np.max(ns)
    ds = ns - m
    sumOfExp = np.exp(ds).sum()
    return m + np.log(sumOfExp)

#@cuda.jit(device=True)
#def logsumexp_gpu(ns):
#    m = np.max(ns)
#    return m + np.log(np.exp(ns-m).sum())

@cuda.jit(device=True)
def log_gpu(x,out):
    start = numba.cuda.grid(1)
    stride = numba.cuda.gridsize(1)
    for i in range(start, x.shape[0], stride):
        out[i] = math.log(x[i])

@cuda.jit
def logsumexp_gpu(ns,out):
    start = numba.cuda.grid(1)
    stride = numba.cuda.gridsize(1)
    m=np.max(ns)
    tmp = 0.0
    for i in range(start, ns.shape[0], stride):
        tmp = tmp + math.exp(ns[i] - m)
    out = math.log(out) + m

@numba.vectorize(['float32(float32, float32, float32)'], target='cuda')
def gaussian(x, x0, sigma):
    return math.exp(-((x - x0) / sigma)**2 / 2) / SQRT_TWOPI / sigma

@numba.vectorize(['float32(float32)'], target='cuda')
def log_vg(x):
    return math.log(x)

def pmmcmc(n=1000,alpha=0.5):
    vec    = np.zeros(n)
    x      = 0
    oldlik = noisydnorm(x)
    vec[1] = x
    for i in range(2,n):
        innov = np.random.uniform(1,-alpha,alpha)
        can   = x+innov
        lik   = noisydnorm(can)
        aprob = lik/oldlik
        u     = np.random.uniform(1)
        if u < aprob:
            x      = can
            oldlik = lik
        vec[i] = x
    return vec

def noisydnorm(z):
    return norm(z)*np.random.exponential(1,1)
