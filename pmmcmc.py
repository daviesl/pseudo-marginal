import numpy as np
import numba
from scipy.stats import norm
#from scipy.misc import logsumexp
import sys

def logsumexp(ns):
    max = np.max(ns)
    ds = ns - max
    sumOfExp = np.exp(ds).sum()
    return max + np.log(sumOfExp)

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
        self.w_all = np.ones((T,n)) * (1./n)
        self.T = T
        self.n = n
        self.innov = innov
        self.lh = lh
    def test_particlefilter(self,num_runs,X0,theta0):
        theta=theta0 #np.zeros((self.theta_size))
        logml_chain=np.zeros((num_runs))
        for j in range(num_runs):
            self.X_all[0,:] = X0
            #print("X_all[0,:] = {}".format(self.X_all[0,:]))
            log_ml = np.zeros(self.T)
            for i in range(1,self.T):
                log_ml[i] = self.particlefilter(self.y_all[i-1,:],self.X_all[i,:],self.X_all[i-1,:],theta[:],self.X_ancestry[i,:],self.w_all[i-1,:],self.w_all[i,:],self.n,self.lh,self.innov)
                #print("X_all[{},:] = {}".format(i,self.X_all[i,:]))
                #print("X_all[{},:] = {}, y={}".format(i,self.X_all[i,:],self.y_all[i-1,:]))
                #print("y={}".format(self.y_all[i-1,:]))
                #print("log_ml[{}] = {}".format(i,log_ml[i]))
            #print("X_all[{},:] = {}".format(self.T-1,self.X_all[self.T-1,:]))
            #print("log_ml[{}] = {}".format(1,log_ml[1]))
            #print("log_ml[{}] = {}".format(self.T-1,log_ml[self.T-1]))
            logml_chain[j] = log_ml.sum() # product of all marignal likelihoods
            print("log_ml.sum() = {}".format(logml_chain[j]))
        return logml_chain
            
        
    def run_pmmh(self,mcmc_chain_size,X0_mu,X0_sigma,theta0=None,pcov0=None):
        #seq = zip(self.X_all,self.y_all,self.X_ancestry, self.w_all)
        # init priors, data is unseen at t=0
        if theta0 is not None:
            theta = np.tile(theta0,(mcmc_chain_size,1))
        else:
            theta = np.ones((mcmc_chain_size,self.theta_size)) * 0.5
        ar = np.zeros(mcmc_chain_size)
        logml_chain=np.ones((mcmc_chain_size))
        logml_chain[:]=np.finfo(float).min
        # don't store all of the priors, just the last one
        prior_j=0
        prior_j_1=1.0
        initprop = 0.0001
        if pcov0 is not None:
            pcov = pcov0
        else:
            pcov = np.eye(self.theta_size) * initprop
        for j in range(mcmc_chain_size):
            if j>0:
                thetamean = np.mean(theta[:j,:],axis=0)
                ar_k = np.sum(ar[:j])*1./max(1,j-1)
                if j < 2 or ar_k == 0:
                    covnorm = 1.
                    #pcov = np.eye(self.theta_size) * 0.0001
                    propscale = 1.0
                else:
                    propscale = np.exp((2*self.theta_size)*(ar_k-0.44))
                    propscale = min(10.0,max(0.0001,propscale))
                    covnorm = (1./(j-1))
                    #pcov[:]=0
                    pcov = np.eye(self.theta_size) * 0.0000001 # condition
                    # update cov
                    for k in range(j):
                        pcov += np.outer(theta[k,:]-thetamean,theta[k,:]-thetamean)*covnorm
                # decompose 
                #evals,evecs = np.linalg.eig(pcov)
                # propose multivariate random normal
                prop = np.random.multivariate_normal(np.zeros(self.theta_size),pcov*propscale)
                print("Proposing theta += {}, propscale = {}, ar_k = {}, pcov = {}".format(prop, propscale, ar_k, pcov))
                theta[j,:]=theta[j-1,:]+prop
                #propose theta using normal dist ar= 0.328
                #theta[j,:]=theta[j-1,:]+np.random.normal(0,0.05,self.theta_size) # assuming the innov function scales the parameters appropriately.
                #propose theta using unif indep ar=0.499
                #theta[j,:]=np.random.uniform(0,1,self.theta_size) # assuming the innov function scales the parameters appropriately.
                
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
                log_ml[i] = self.particlefilter(self.y_all[i-1,:],self.X_all[i,:],self.X_all[i-1,:],theta[j,:],self.X_ancestry[i,:],self.w_all[i-1,:],self.w_all[i,:],self.n,self.lh,self.innov)
            # compute marginal likelihood of particle filter
            logml_chain[j] = log_ml.sum() # product of all marignal likelihoods
            ##print("1_Marginal likelihood at {} is {}".format(j, ml_j))
            if j > 0:
                #acceptance ratio
                #log_a = min(0,logml_chain[j]+np.log(prior_j_1)-logml_chain[j-1]-np.log(prior_j)) # assuming normal priors on parameters
                #log_a = min(0,logml_chain[j]+np.log(prior_j_1)-logml_chain[j-1]-np.log(prior_j)) # assuming normal priors on parameters
                log_a = min(0,logml_chain[j]+np.log(prior_j)-logml_chain[j-1]-np.log(prior_j_1)) # assuming normal priors on parameters
                # compute unif(0,1)
                log_u = np.log(np.random.uniform(0,1))
                print("Log a = {}, log_u = {}, accepting = {}".format(log_a,log_u,log_a>=log_u))
                if log_a>=log_u:
                    # keep the proposed theta and X_all
                    ar[j]=1
                else:
                    # backtrack to the last theta and state X_all TODO do I need to store last state?
                    theta[j,:] = theta[j-1,:] 
                    logml_chain[j] = logml_chain[j-1]
                    ar[j]=0
            else:
                logml_chain[j-1] = logml_chain[j]
                prior_j_1 = prior_j
            print("Theta at {} is {} {} {}".format(j,theta[j,0],theta[j,1],theta[j,2]))
            print("Marginal likelihood at {} is {}".format(j, logml_chain[j]))
        return (theta,logml_chain,ar,pcov*propscale)

#    def run_pmgibbs(self,mcmc_chain_size):
#        # ancestry trace the final target distrubution particles and sum the weights for each
    @classmethod
    def particlefilter(cls,yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov):
        if False:
            return cls.bootstrapparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov)
        elif True:
            return cls.experimentalparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov)

    @classmethod
    def bootstrapparticlefilter(cls,yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov):
        # Always resample, and at the start because innov() will disperse according to random effects in SDE
        wi_1_norm = np.exp(np.log(wi_1) - logsumexp(np.log(wi_1)))
        Xianc[:]=resample(wi_1_norm)
        #print("Resample count = {}".format(np.unique(Xianc[:]).shape[0]))
        #print("Resample indices = {}".format(Xianc[:]))
        Xi_1_rs=Xi_1[Xianc[:],:]
        #wi[:]=wi[Xianc]
        Xi[:] = innov(Xi_1_rs,theta)
        loglh = lh(Xi,yi,theta)
        #print("Max llh={}".format(np.max(loglh)))
        if not np.isfinite(loglh).all():
            print("log likelihoods not finite")
            sys.exit(0)
        log_wi_next = loglh - np.log(n) # non-normalised, assumes previous weights are 1./n
        logsumexp_log_wi_next = logsumexp(log_wi_next)
        if not np.isfinite(wi_1).all():
            print("Infinite entries in wi_1: {}".format(np.argwhere(~np.isfinite(wi_1))))
            sys.exit(0)
        log_wi_norm = log_wi_next - logsumexp_log_wi_next # normalise weights
        wi_norm = np.exp(log_wi_norm)
        wi[:] = np.exp(log_wi_next)
        if not np.isfinite(wi).all():
            print("log_wi_next: {}".format(log_wi_next))
            print("Infinite entries of wi: {}".format(wi[~np.isfinite(wi)]))
            print("Infinite entries of wi: {}".format(np.argwhere(~np.isfinite(wi))))
            sys.exit(0)
        if not np.isfinite(logsumexp_log_wi_next):
            print("Divide by zero sum of weights. Log sum exp(log_wi_next) = {}".format(logsumexp_log_wi_next))
            sys.exit(0)
        logml = logsumexp_log_wi_next - np.log(n)
        return logml

    @classmethod
    def experimentalparticlefilter(cls,yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov):
        Xi[:] = innov(Xi_1,theta) # TODO the theta here assumes a static timestep. We'll need to record it somehow in the pmpfl main function.
        loglh = lh(Xi,yi,theta)
        if not np.isfinite(loglh).all():
            print("log likelihoods not finite")
            nanidx = np.argwhere(~np.isfinite(loglh))
            print (nanidx.shape,loglh[nanidx].shape,np.squeeze(Xi[nanidx,:]).shape)
            print (np.hstack((nanidx,wi[nanidx],np.squeeze(Xi[nanidx,:]),np.squeeze(Xi_1[nanidx,:]))))
            sys.exit(0)
        #loglh = np.nan_to_num(loglh)
        log_wi_next = np.log(wi_1) + loglh # non-normalised
        logsumexp_log_wi_next = logsumexp(log_wi_next)
        #print("log marginal likelihood = {}".format(logml))
        if not np.isfinite(wi_1).all():
            print("Infinite entries of wi_1: {}".format(wi_1[~np.isfinite(wi_1).all()]))
        log_wi_norm =  log_wi_next - logsumexp_log_wi_next # normalise weights
        wi_norm = np.exp(log_wi_norm)
        wi[:] = np.exp(log_wi_next)
        if not np.isfinite(wi).all():
            print("Infinite entries of wi: {}".format(wi[~np.isfinite(wi).all()]))
        if not np.isfinite(logsumexp_log_wi_next):
            print("Divide by zero sum of weights. Log sum exp(log_wi_next) = {}".format(logsumexp_log_wi_next))
            print("sum log wi = {} ".format(log_wi_next.sum()))
            print("Infinite entries of loglh: {}".format(loglh[~np.isfinite(loglh).all()]))
            print("Zero entries of wi_1: {}".format(wi_1[wi_1==0]))
            print("Infinite entries of wi_1: {}".format(wi_1[~np.isfinite(wi_1).all()]))
            print("Infinite entries of wi: {}".format(wi[~np.isfinite(wi).all()]))
            print("Zero entries of wi: {}".format(wi[wi==0]))
            print("wi_1: {}".format(wi_1[:]))
            print("wi: {}".format(wi[:]))
            sys.exit(0)
        #print("n = {}, ess = {}".format(n,1./sum(wi**2)))
        if 1./np.exp(logsumexp(2*log_wi_norm)) < 0.5*n:
            #print("Resampling, ess = {}".format(1./sum(wi**2)))
            Xianc[:]=resample(wi_norm)
            Xi[:]=Xi[Xianc,:]
            #wi[:]=wi[Xianc]
            #logsumexp_log_wi_next = logsumexp(np.log(wi))
            #logsumexp_log_wi_next = logsumexp(np.log(wi[Xianc])) # use resampled particle weights for marginal likelihood
            wi[:]=1./n
        #logml = logsumexp(loglh) - np.log(n)
        logml = logsumexp_log_wi_next - np.log(n)
        return logml

    #def temp_debug(self):
        #print(loglh)

@numba.jit
def resample(weights):
    n = weights.shape[0]
    indices = np.zeros_like(weights)
    #C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    C = np.cumsum(weights) * n
    u0 = np.random.uniform(0,1)
    j = 0
    for i in range(n):
        u = u0+i
        while u > C[j]:
            j+=1
        indices[i] = j
    return indices
    

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

# TODO a particle cloud for state space models
class sspc(object):
    def __init__(self,T,X_size,theta_size,n):
        self.theta_size = theta_size
        self.X_all = np.zeros((T,n,X_size))
        self.X_ancestry = np.tile(np.arange(n),(T,1))
        self.w_all = np.ones((T,n))/n
        self.T = T
        self.n = n
