import numpy as np
import numba
from scipy.stats import norm
#from scipy.misc import logsumexp
import sys
from mdlinalg import *

# TODO make a base class for a particle filter. DONE
# Then make a class for a pseudo marginal estimator DONE
#      should this PM estimator class return model class likelihood? Only with SMC^2?
# TODO 6/11/2019 need to take square root of process noise covariance matrix returned by diffusion().

class stateSpaceModel(object):
    """
    defaults: One parameter, one state variable.
    """
    @classmethod
    def X_size(cls):
        return 1
    @classmethod
    def theta_size(cls):
        return 1
    @classmethod
    def y_dim(cls):
        return 1
    @classmethod
    def initialState(cls):
        return np.ones(cls.X_size())
    #def get_innov(self):
    #    """
    #    TODO: return a handle to the innov fn for numba JIT
    #    """
    #    return self.class_innov
    #def innov(self,X_k,y_J,theta):
    #    return self.class_innov(X_k,y_J,theta)
    #@staticmethod
    #@numba.jit
    #def class_innov(X_k,y_J,theta):
    #    """
    #    Increment state X to next time point defined by 
    #        observation interval
    #    Computes stochastic terms.
    #    """
    #    _N_ = X_k.shape[0]
    #    dim = X_k.shape[1]
    #    dW_t = np.random.normal(0,1,(_N_,dim))
    #    X_k1 = X_k + self.A(theta)*self.dt + self.B(theta)*math.sqrt(self.dt)
    #    return X_k1, np.ones(_N_) # (next X, logpqratio)

class ItoProcess(stateSpaceModel):
    """
    The Ito Process encapsulates all information regarding the
        state variable relationships.
    It also encapsulates how the likelihood is computed.
        This decision was to simplify the bridge proposal code
        design. 
    Hence the design is a class that directly inherits from 
        ItoProcess would be the base process with drift, diffusion
        and loglikelihood defined.
    Then any child class of the above would be a bridge proposal.
    OR
    Perhaps the direct inheritance should be the bridge proposal
    And then subsequent classes that inherit should be the process
        itself with drift, diffusion and loglikelihood defined.
    """
    @classmethod
    def obserr(cls):
        return 1.0
    @classmethod
    def obsinterval(cls):
        return 40
    @classmethod
    def delta_t(cls):
        return 0.001
    @classmethod
    def observationCovariance(cls):
        return np.eye(cls.y_dim()) * (cls.obserr ** 2)
    @staticmethod
    @numba.jit
    def drift(X_k,y_J,theta):
        return np.ones(X_k.shape)
    @staticmethod
    @numba.jit
    def diffusion(X_k,y_J,theta):
        # TODO rename to volatility
        dim=X_k.shape[-1]
        #return np.ones(X_k.shape+X_k.shape[-1:])*np.eye(dim)
        return np.eye(dim)
    @staticmethod
    @numba.jit
    def transformXtoState(X):
        """
        Override this method to transform the X variables defined on
        [0,1] to the appropriate state variables for the problem
        """
        return X 
    @staticmethod
    @numba.jit
    def transformStatetoX(state):
        return state
    @classmethod
    def generateDeterministicDrift(cls):
        drift = cls.drift
        delta_t = cls.delta_t()
        X2tr = cls.transformXtoState
        tr2X = cls.transformStatetoX
        obsinterval = cls.obsinterval()
        theta2tr = cls.transformThetatoParameters
        tr2theta = cls.transformParameterstoTheta
        @numba.jit
        def EMDriftClosure(X_k,y_J,theta):
            """
            Returns Euler Maruyama innovation of the state variables
                integrated over obsinterval timesteps of length delta_t.
            """
            next_state = X2tr(X_k)
            trth = theta2tr(theta)
            for i in range(obsinterval):
                next_state += drift(next_state,y_J,trth)*delta_t
            return tr2X(next_state),np.zeros(X_k.shape[0]) 
        return EMDriftClosure
    @classmethod
    def generateInnovation(cls):
        #a lambda function closure of the below eulermaruyama with drift and diffusion
        # A quirk of Numba requires that every fn and variable in the
        # below closure must not be part of an object, i.e. we need
        # to be able to lazy evaluate it without the object/class present.
        # The solution is to "pull out" every variable and fn below
        # to the local scope. All functions that are numba.jit must be
        # declared static.
        drift = cls.drift
        diffusion = cls.diffusion
        delta_t = cls.delta_t()
        X2tr = cls.transformXtoState
        tr2X = cls.transformStatetoX
        obsinterval = cls.obsinterval()
        theta2tr = cls.transformThetatoParameters
        tr2theta = cls.transformParameterstoTheta
        @numba.jit
        def EMInnovClosure(X_k,y_J,theta):
            """
            Returns Euler Maruyama innovation of the state variables
                integrated over obsinterval timesteps of length delta_t.
            """
            next_state = X2tr(X_k)
            trth = theta2tr(theta)
            for i in range(obsinterval):
                dW_t = np.random.normal(0,np.sqrt(delta_t),X_k.shape)
                next_state += drift(next_state,y_J,trth)*delta_t + mdla_dottrail2x1_broadcast(spsd_sqrtm(diffusion(next_state,y_J,trth)),dW_t)
            return tr2X(next_state),np.zeros(X_k.shape[0]) 
        return EMInnovClosure
    @classmethod
    def generateLogLikelihood(cls):
        # As above with the innovation closure
        # pull out things like observation error etc.
        dim = cls.y_dim()
        cov = cls.observationCovariance()
        obseqn = cls.observationEquation
        X2tr = cls.transformXtoState
        tr2X = cls.transformStatetoX
        @numba.jit
        def loglhClosure(X,y,theta):
            """
            TODO: write support for state/time dependent obs cov
            """
            trX=X2tr(X)
            y_star = obseqn(trX)
            d = y_star-y
            dd = xTPx(d,np.linalg.inv(cov))
            log_lh_un = (-0.5*(dd)) 
            norm_const = - (dim*0.5) * np.log(2*np.pi) - 0.5 * np.log(np.linalg.det(cov))
            return log_lh_un + norm_const
        return loglhClosure
            
# Extend ItoProcess class for all bridge processes
# Define the model class separately or extend ItoProcess.
# Then use multiple inheritance for the model-bridge class.

class ModifiedDiffusionBridge(ItoProcess):
    @classmethod
    def generateInnovation(cls):
        drift = cls.drift
        diffusion = cls.diffusion
        delta_t = cls.delta_t()
        X2tr = cls.transformXtoState
        tr2X = cls.transformStatetoX
        obsinterval = cls.obsinterval()
        obserr_mat = cls.observationCovariance()
        obs_map = cls.obs_map()
        theta2tr = cls.transformThetatoParameters
        tr2theta = cls.transformParameterstoTheta
        @numba.jit(nopython=True)
        def MDBInnovClosure(X_k,y_J,theta):
            next_state = X2tr(X_k)
            trth = theta2tr(theta)
            Xshape0 = X_k.shape[0]
            logpqratio=np.zeros(Xshape0)
            for i in range(obsinterval): 
                a_t = drift(next_state,y_J,trth) 
                beta_all = diffusion(next_state,y_J,trth)
                beta_xx = doublemdla_dottrail2x2_broadcast(obs_map.T,beta_all)
                dW_t = np.random.normal(0,np.sqrt(delta_t),X_k.shape)
                # MDB specific quantities
                dk = (obsinterval - i) * delta_t
                P_k = mdla_invtrail2d(beta_xx*dk + obserr_mat) # updated precision matrix
                C_P_k_C = doublemdla_dottrail2x2_broadcast(obs_map,P_k)
                psi_mdb = beta_all - C_P_k_C * delta_t
                # Compute MDB drift
                mu_mdb = a_t + mdla_dottrail2x1_broadcast(C_P_k_C, mdla_dottrail2x1_broadcast(obs_map,(y_J - mdla_dottrail2x1_broadcast(obs_map.T,(next_state + a_t*dk)))))
                x_mu_mdb = next_state + mu_mdb*delta_t
                # Compute EM drift for comparison
                x_mu_em = next_state + a_t*delta_t
                next_state[:] = x_mu_mdb + mdla_dottrail2x1_broadcast(spsd_sqrtm(psi_mdb),dW_t)
                logpqratio[:] += logmvnorm_vectorised(next_state,x_mu_em,beta_all*delta_t) - logmvnorm_vectorised(next_state,x_mu_mdb,psi_mdb*delta_t)
            return tr2X(next_state), logpqratio
        return MDBInnovClosure
   

class LindstromBridge(ItoProcess):
    @classmethod
    def generateInnovation(cls):
        drift = cls.drift
        diffusion = cls.diffusion
        delta_t = cls.delta_t()
        X2tr = cls.transformXtoState
        tr2X = cls.transformStatetoX
        obsinterval = cls.obsinterval()
        obserr_mat = cls.observationCovariance()
        obs_map = cls.obs_map()
        theta2tr = cls.transformThetatoParameters
        tr2theta = cls.transformParameterstoTheta
        @numba.jit#(nopython=True)
        def LindstromInnovClosure(X_k,y_J,theta):
            next_state = X2tr(X_k)
            trth = theta2tr(theta)
            Xshape0 = X_k.shape[0]
            logpqratio=np.zeros(Xshape0)
            gamma = 0.5
            for i in range(obsinterval): 
                dW_t = np.random.normal(0,np.sqrt(delta_t),X_k.shape)
                dk = (obsinterval - i) * delta_t
                dk1 = (obsinterval - (i+1)) * delta_t
                a_t = drift(next_state,y_J,trth) 
                beta_all = diffusion(next_state,y_J,trth)
                beta_xx = doublemdla_dottrail2x2_broadcast(obs_map.T,beta_all)
                # Convex mixture
                C = (gamma/delta_t) * beta_xx 
                # MDB specific quantities
                P_k = mdla_invtrail2d(beta_xx*dk + C*(dk1**2) + obserr_mat) # updated precision matrix
                C_P_k_C = doublemdla_dottrail2x2_broadcast(obs_map,P_k)
                psi_mdb = beta_all - C_P_k_C * delta_t
                # Compute MDB drift
                mu_mdb = a_t + mdla_dottrail2x1_broadcast(C_P_k_C, mdla_dottrail2x1_broadcast(obs_map,(y_J - mdla_dottrail2x1_broadcast(obs_map.T,(next_state + a_t*dk)))))
                x_mu_mdb = next_state + mu_mdb*delta_t
                # Compute EM drift for comparison
                x_mu_em = next_state + a_t*delta_t
                next_state[:] = x_mu_mdb + mdla_dottrail2x1_broadcast(spsd_sqrtm(psi_mdb),dW_t)
                logpqratio[:] += logmvnorm_vectorised(next_state,x_mu_em,beta_all*delta_t) - logmvnorm_vectorised(next_state,x_mu_mdb,psi_mdb*delta_t)
            return tr2X(next_state), logpqratio
        return LindstromInnovClosure

class ResidualBridge(ItoProcess):
    @classmethod
    def generateInnovation(cls):
        drift = cls.drift
        diffusion = cls.diffusion
        delta_t = cls.delta_t()
        X2tr = cls.transformXtoState
        tr2X = cls.transformStatetoX
        obsinterval = cls.obsinterval()
        obserr_mat = cls.observationCovariance()
        obs_map = cls.obs_map()
        theta2tr = cls.transformThetatoParameters
        tr2theta = cls.transformParameterstoTheta
        @numba.jit(nopython=True)
        def RBInnovClosure(X_k,y_J,theta):
            next_state = X2tr(X_k)
            trth = theta2tr(theta)
            Xshape0 = X_k.shape[0]
            Xshape1 = X_k.shape[1]
            eta=np.zeros(X_k.shape+(obsinterval+1,))
            last_eta = np.zeros_like(next_state)
            logpqratio=np.zeros(Xshape0)
            gamma = 0.5
            for i in range(obsinterval+1):
                eta[:,:,i] = last_eta + drift(last_eta,y_J,trth)*delta_t
                last_eta[:] = eta[:,:,i]
            for i in range(obsinterval): 
                dW_t = np.random.normal(0,np.sqrt(delta_t),X_k.shape)
                dk = (obsinterval - i) * delta_t
                r = next_state - eta[:,:,i] # eta[...,:]
                chord = (eta[...,i+1]-eta[...,i])/delta_t
                a_t = drift(next_state,y_J,trth) 
                beta_all = diffusion(next_state,y_J,trth)
                beta_xx = doublemdla_dottrail2x2_broadcast(obs_map.T,beta_all)
                # MDB specific quantities
                P_k = mdla_invtrail2d(beta_xx*dk + obserr_mat) # updated precision matrix
                C_P_k_C = doublemdla_dottrail2x2_broadcast(obs_map,P_k)
                psi_mdb = beta_all - C_P_k_C * delta_t
                # Compute MDB drift
                mu_rb = a_t + mdla_dottrail2x1_broadcast(C_P_k_C, mdla_dottrail2x1_broadcast(obs_map,( y_J - mdla_dottrail2x1_broadcast(obs_map.T,eta[...,obsinterval-1] + r + (a_t-chord)*dk))))
                x_mu_rb = next_state + mu_rb*delta_t
                # Compute EM drift for comparison
                x_mu_em = next_state + a_t*delta_t
                next_state[:] = x_mu_rb + mdla_dottrail2x1_broadcast(spsd_sqrtm(psi_mdb),dW_t)
                logpqratio[:] += logmvnorm_vectorised(next_state,x_mu_em,beta_all*delta_t) - logmvnorm_vectorised(next_state,x_mu_rb,psi_mdb*delta_t)
            return tr2X(next_state), logpqratio
        return RBInnovClosure


class stateFilter(object):
    """
    The base class for a particle filter which return an unbiased
        estimate of the likelihood with the state variables marginalised
        (integrated out).
    Inputs
        1) the state space model object (usually the ito process)
        2) N = number of particles
        3) Y = the data time series to be assimilated
        4) Theta = the parameters that are input into the ito process
    """
    def __init__(self,ssm,y_all,n):
        T = y_all.shape[0]+1
        self.y_all = y_all
        self.T = T
        self.n = n
        # rest of ssm defn
        self.ssm = ssm
        self.innov = self.ssm.generateInnovation()
        self.propnf = self.ssm.generateDeterministicDrift()
        self.lh = self.ssm.generateLogLikelihood()
        # state variable result, ancestry and weights. TODO refactor to result class
        self.X_all = np.zeros((T,n,self.ssm.X_size()))
        self.X_ancestry = np.tile(np.arange(n),(T,1))
        self.w_all = np.ones((T,n)) * (1./n)
    def test_filter(self,num_runs,X0,theta0):
        theta=theta0 #np.zeros((self.theta_size))
        logml_chain=np.zeros((num_runs))
        for j in range(num_runs):
            self.X_all[0,:] = X0
            #print("X_all[0,:] = {}".format(self.X_all[0,:]))
            log_ml = np.zeros(self.T)
            for i in range(1,self.T):
                log_ml[i] = self.filter_step(self.y_all[i-1,:],self.X_all[i,:],self.X_all[i-1,:],theta[:],self.X_ancestry[i,:],self.w_all[i-1,:],self.w_all[i,:],self.n,self.lh,self.innov,self.propnf)
                #print("X_all[{},:] = {}".format(i,self.X_all[i,:]))
                #print("X_all[{},:] = {}, y={}".format(i,self.ssm.transformXtoState(self.X_all[i,:]),self.y_all[i-1,:]))
                #print("y={}".format(self.y_all[i-1,:]))
                #print("log_ml[{}] = {}".format(i,log_ml[i]))
            #print("X_all[{},:] = {}".format(self.T-1,self.X_all[self.T-1,:]))
            #print("log_ml[{}] = {}".format(1,log_ml[1]))
            #print("log_ml[{}] = {}".format(self.T-1,log_ml[self.T-1]))
            logml_chain[j] = log_ml.sum() # product of all marignal likelihoods
            print("log_ml.sum() = {}".format(logml_chain[j]))
        return logml_chain
    def init_state_prior(self,X0_mu,X0_sigma):
        """
        TODO refactor to remove mu and sigma from this fn. Set in statefilter class/obj instead.
             phase 2 TODO: make the priors user-determined.
        """
        #self.sf.X_all[0,:] = np.random.uniform(0,1,(self.sf.n,self.sf.X_size))
        # init priors of state variables. Innovation fn must translate from N(0,1) to correct range.
        self.X_all[0,:] = np.random.normal(X0_mu,X0_sigma,(self.n,self.ssm.X_size()))
        #print(self.sf.X_all[0,:])
        #print("Init prior to unif(0,1)")
        #print(self.sf.X_all[0,:])
    def generateRunFilter(self):
        # copy obj scope to local for closure
        y_all = self.y_all
        X_all = self.X_all
        X_ancestry = self.X_ancestry
        w_all = self.w_all
        T = self.T
        n = self.n
        lh = self.lh
        innov = self.innov
        propnf = self.propnf
        fs = self.filter_step
        @numba.jit
        def run_filter_for_params(theta):
            log_ml = np.zeros(T)
            for i in range(1,T):
                # TODO add support for aux pf
                log_ml[i] = fs(y_all[i-1,:],X_all[i,:],X_all[i-1,:],theta,X_ancestry[i,:],w_all[i-1,:],w_all[i,:],n,lh,innov,propnf)
            # compute marginal likelihood of particle filter
            return log_ml.sum()
        return run_filter_for_params
    @staticmethod
    @numba.jit
    def filter_step(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf):
        """ default is bootstrap """
        # Resample
        Xianc[:]=resample(wi_1,n)
        Xi_1_rs=Xi_1[Xianc[:],:]
        # Propagate
        Xi[:],logpqratio = innov(Xi_1_rs,yi,theta)
        loglh = lh(Xi,yi,theta)
        # Compute weights
        log_wi = loglh + logpqratio 
        logsumexp_log_wi = logsumexp(log_wi)
        wi[:] = np.exp(log_wi - logsumexp_log_wi)
        return logsumexp_log_wi - np.log(n)

# The rest of the filters
class auxiliaryParticleFilter(stateFilter):
    @staticmethod
    @numba.jit
    def filter_step(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf):
        # Note the way Xianc ancestry are stored, it is on the parent state Xi_1
        # Compute the posterior integral p(y_t | x_{t-1})
        Xi_bar, lpqr = propnf(Xi_1,yi,theta)
        loglh_nf = lh(Xi_bar,yi,theta) 
        log_v_1=np.log(wi_1) + loglh_nf + lpqr
        v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
        Xianc[:]=resample(v_1_norm,n) # j_i, the new jth parent for the ith particle , should be Xianc_1
        Xi_1_j=Xi_1[Xianc,:]
        wi_1_j=wi_1[Xianc]
        log_v_1_j=log_v_1[Xianc]
        log_v_1n_j=np.log(v_1_norm[Xianc])
        Xi[:],logpqratio=innov(Xi_1_j,yi,theta)
        loglh=lh(Xi,yi,theta)
        log_wi=loglh+logpqratio+np.log(wi_1_j)-log_v_1_j
        wi[:]=np.exp(log_wi-logsumexp(log_wi))
        return logsumexp(log_wi) + logsumexp(log_v_1) - np.log(n)

class ESSPartiallyAdaptedParticleFilter(stateFilter):
    @staticmethod
    @numba.jit
    def filter_step(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf):
        # Note the way Xianc ancestry are stored, it is on the parent state Xi_1
        # Compute the posterior integral p(y_t | x_{t-1})
        Xi_bar, lpqr = propnf(Xi_1,yi,theta)
        loglh_nf = lh(Xi_bar,yi,theta) 
        if True:
            ess_v = np.exp(-logsumexp(2*np.log(wi_1)))
            for epsilon in range(20):
                log_v_1=np.log(wi_1) + (ess_v/n) * loglh_nf
                v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
                ess_v = np.exp(-logsumexp(2*np.log(v_1_norm)))
        else:
            ess_v = 0.5 * n
            log_v_1=np.log(wi_1) + (ess_v/n) * loglh_nf
            v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
        Xianc[:]=resample(v_1_norm,n) # j_i, the new jth parent for the ith particle , should be Xianc_1
        Xi_1_j=Xi_1[Xianc,:]
        wi_1_j=wi_1[Xianc]
        log_v_1_j=log_v_1[Xianc]
        log_nf_j=loglh_nf[Xianc]
        Xi[:],logpqratio=innov(Xi_1_j,yi,theta)
        loglh=lh(Xi,yi,theta)
        log_wi=loglh+logpqratio+np.log(wi_1_j)-log_v_1_j
        wi[:]=np.exp(log_wi-logsumexp(log_wi))
        ess_w = np.exp(-logsumexp(2*np.log(wi)))
        return logsumexp(log_wi) + logsumexp(log_v_1) - np.log(n)

            
class parameterEstimator(object):            
    def __init__(self,sf):
        self.sf = sf
        self.T = sf.T
        self.n = sf.n
        self.theta_size = sf.ssm.theta_size()
        self.runfilter = self.sf.generateRunFilter()
    def run_pmmh(self,mcmc_chain_size,X0_mu,X0_sigma,theta0=None,pcov0=None):
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
                    propscale = (2.38)**2/self.theta_size
                else:
                    propscale = (2.38)**2/self.theta_size
                    covnorm = (1./(j-1))
                    pcov = np.eye(self.theta_size) * 0.0000001 # condition
                    # update cov
                    for k in range(j):
                        pcov += np.outer(theta[k,:]-thetamean,theta[k,:]-thetamean)*covnorm
                # decompose 
                # propose multivariate random normal
                prop = np.random.multivariate_normal(np.zeros(self.theta_size),pcov*propscale)
                #print("Proposing theta += {}, propscale = {}, ar_k = {}, pcov = {}".format(prop, propscale, ar_k, pcov))
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
            self.sf.init_state_prior(X0_mu,X0_sigma)
            # compute marginal likelihood of particle filter
            logml_chain[j] = self.runfilter(theta[j,:]) 
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
                    # keep the proposed theta 
                    ar[j]=1
                else:
                    # backtrack to the last theta 
                    theta[j,:] = theta[j-1,:] 
                    logml_chain[j] = logml_chain[j-1]
                    ar[j]=0
            else:
                logml_chain[j-1] = logml_chain[j]
                prior_j_1 = prior_j
            print("Theta at {} is {} {} {}".format(j,theta[j,0],theta[j,1],theta[j,2]))
            print("Marginal likelihood at {} is {}".format(j, logml_chain[j]))
        return (theta,logml_chain,ar,pcov)
        

# a pseudo-marginal based sampler for a particle filter
# the innovation function will be generalised
class pmpfl(object):
    def __init__(self,innov,propnf,lh,y_all,X_size,theta_size,n):
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
        self.propnf = propnf
        self.lh = lh
    def test_particlefilter(self,num_runs,X0,theta0):
        theta=theta0 #np.zeros((self.theta_size))
        logml_chain=np.zeros((num_runs))
        for j in range(num_runs):
            self.X_all[0,:] = X0
            #print("X_all[0,:] = {}".format(self.X_all[0,:]))
            log_ml = np.zeros(self.T)
            for i in range(1,self.T):
                # TODO add support for aux pf
                log_ml[i] = self.particlefilter(self.y_all[i-1,:],self.X_all[i,:],self.X_all[i-1,:],theta[:],self.X_ancestry[i,:],self.w_all[i-1,:],self.w_all[i,:],self.n,self.lh,self.innov,self.propnf)
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
                    #propscale = 1.0
                    propscale = (2.38)**2/self.theta_size
                else:
                    #propscale = np.exp((2*self.theta_size)*(ar_k-0.44))
                    #propscale = min(10.0,max(0.0001,propscale))
                    propscale = (2.38)**2/self.theta_size
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
                # TODO add support for aux pf
                log_ml[i] = self.particlefilter(self.y_all[i-1,:],self.X_all[i,:],self.X_all[i-1,:],theta[j,:],self.X_ancestry[i,:],self.w_all[i-1,:],self.w_all[i,:],self.n,self.lh,self.innov,self.propnf)
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
        return (theta,logml_chain,ar,pcov)

#    def run_pmgibbs(self,mcmc_chain_size):
#        # ancestry trace the final target distrubution particles and sum the weights for each
    @classmethod
    def particlefilter(cls,yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf):
        if True:
            return cls.bootstrapparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov)
        elif False:
            return cls.experimentalparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov)
        elif False:
            return cls.auxilliaryparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf)
        elif False:
            return cls.meanauxilliaryparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf)
        elif False:
            return cls.improvedauxilliaryparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf)
        elif False:
            return cls.essauxilliaryparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf)
        elif False:
            return cls.fullessauxilliaryparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf)
        elif False:
            return cls.alphaauxilliaryparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf)
        elif False:
            return cls.garbageauxilliaryparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf)
        elif False:
            return cls.onkauxilliaryparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf)
        elif False:
            return cls.sqrtauxilliaryparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf)

    @staticmethod
    #@numba.jit
    def onkauxilliaryparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf):
        # Note the way Xianc ancestry are stored, it is on the parent state Xi_1
        # Compute the posterior integral p(y_t | x_{t-1})
        Xi_bar, lpqr = propnf(Xi_1,yi,theta)
        loglh_nf = lh(Xi_bar,yi,theta) 
        # resample multinomially for u weights
        #log_u_1=np.log(wi_1) + loglh_nf + lpqr
        log_u_1=loglh_nf + lpqr
        u_1_norm=np.exp(log_u_1-logsumexp(log_u_1))
        #Xi_bar_indices = resample(u_1_norm,n) # todo change to multinomial
        #Xi_bar_indices = np.full(n,np.argmax(u_1_norm),dtype=np.int32)
        #Xi_bar_star, lpqr_star  = propnf(Xi_1,yi,theta,Xi_bar_indices)
        Xi_bar_star, lpqr_star  = propnf(Xi_1,yi,theta,np.log(u_1_norm))
        #Xi_bar_star, lpqr_star  = propnf(Xi_1,yi,theta)
        loglh_nf_star = lh(Xi_bar_star,yi,theta) 
        #loglh_nf_star = loglh_nf[Xi_bar_indices]
        log_v_1=np.log(wi_1) + loglh_nf_star + lpqr_star
        v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
        Xianc[:]=resample(v_1_norm,n) # j_i, the new jth parent for the ith particle , should be Xianc_1
        Xi_1_j=Xi_1[Xianc,:]
        wi_1_j=wi_1[Xianc]
        log_v_1_j=log_v_1[Xianc]
        log_v_1n_j=np.log(v_1_norm[Xianc])
        Xi[:],logpqratio=innov(Xi_1_j,yi,theta)
        loglh=lh(Xi,yi,theta)
        log_wi=loglh+logpqratio+np.log(wi_1_j)-log_v_1_j
        wi[:]=np.exp(log_wi-logsumexp(log_wi))
        return logsumexp(log_wi) + logsumexp(log_v_1) - np.log(n)

    @classmethod
    def sqrtauxilliaryparticlefilter(cls,yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf):
        """
        Always flatten the likelihood in the look ahead by ^(1/2)
        """
        Xi_bar, lpqr = propnf(Xi_1,yi,theta)
        loglh_nf = lh(Xi_bar,yi,theta) 
        log_v_1=np.log(wi_1) + 0.5*(loglh_nf + lpqr)
        v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
        Xianc[:]=resample(v_1_norm,n) # j_i, the new jth parent for the ith particle , should be Xianc_1
        Xi_1_j=Xi_1[Xianc,:]
        wi_1_j=wi_1[Xianc]
        log_v_1_j=log_v_1[Xianc]
        log_v_1n_j=np.log(v_1_norm[Xianc])
        Xi[:],logpqratio=innov(Xi_1_j,yi,theta)
        loglh=lh(Xi,yi,theta)
        log_wi=loglh+logpqratio+np.log(wi_1_j)-log_v_1_j
        wi[:]=np.exp(log_wi-logsumexp(log_wi))
        return logsumexp(log_wi) + logsumexp(log_v_1) - np.log(n)

    @classmethod
    def auxilliaryparticlefilter(cls,yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf):
        # Note the way Xianc ancestry are stored, it is on the parent state Xi_1
        # Compute the posterior integral p(y_t | x_{t-1})
        Xi_bar, lpqr = propnf(Xi_1,yi,theta)
        loglh_nf = lh(Xi_bar,yi,theta) 
        log_v_1=np.log(wi_1) + loglh_nf + lpqr
        v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
        Xianc[:]=resample(v_1_norm,n) # j_i, the new jth parent for the ith particle , should be Xianc_1
        Xi_1_j=Xi_1[Xianc,:]
        wi_1_j=wi_1[Xianc]
        log_v_1_j=log_v_1[Xianc]
        log_v_1n_j=np.log(v_1_norm[Xianc])
        Xi[:],logpqratio=innov(Xi_1_j,yi,theta)
        loglh=lh(Xi,yi,theta)
        log_wi=loglh+logpqratio+np.log(wi_1_j)-log_v_1_j
        wi[:]=np.exp(log_wi-logsumexp(log_wi))
        #return logsumexp(loglh+logpqratio)-np.log(n)
        #return logsumexp(log_wi)-np.log(n)
        #return logsumexp(log_wi) - logsumexp(np.log(wi_1)) + logsumexp(log_v_1) - np.log(n)
        return logsumexp(log_wi) + logsumexp(log_v_1) - np.log(n)

    @staticmethod
    #@numba.jit
    def fullessauxilliaryparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf):
        # Note the way Xianc ancestry are stored, it is on the parent state Xi_1
        # Compute the posterior integral p(y_t | x_{t-1})
        Xi_bar, lpqr = propnf(Xi_1,yi,theta)
        loglh_nf = lh(Xi_bar,yi,theta) 
        if True:
            ess_v = np.exp(-logsumexp(2*np.log(wi_1)))
            for initer in range(100):
                log_v_1=np.log(wi_1) + (ess_v/n) * loglh_nf
                v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
                ess_v = np.exp(-logsumexp(2*np.log(v_1_norm)))
                #print("Above ESS = {} for iter {}".format(ess_v/n, initer))
            print("Iterated ESS/N = {}".format(ess_v/n))
            print("log(gl/gu)^2 = {}".format(2*(loglh_nf.min() - loglh_nf.max())))
            print("Comparison ESS = {} for 1/Nsum(wi_1^2)".format(np.exp(-logsumexp(2*np.log(wi_1))-np.log(n))))
            print("Comparison ESS = {} for 1/Nsum(gi^2)".format(np.exp(-logsumexp(2*(loglh_nf-logsumexp(loglh_nf)))-np.log(n))))
        start_ess = (np.exp(-logsumexp(2*np.log(wi_1))), 1, 0,ess_v)
        for outiter in range(len(start_ess)): 
            ess_w = start_ess[outiter] #np.exp(-logsumexp(2*np.log(wi_1)))
            for initer in range(10):
                log_v_1=np.log(wi_1) + (ess_w/n) * loglh_nf
                v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
                ess_v = np.exp(-logsumexp(2*np.log(v_1_norm)))
                print("Proxy v_ESS/N = {} for iters {} {}".format(ess_v/n, initer, outiter))
                Xianc[:]=resample(v_1_norm,n) # j_i, the new jth parent for the ith particle , should be Xianc_1
                Xi_1_j=Xi_1[Xianc,:]
                wi_1_j=wi_1[Xianc]
                log_v_1_j=log_v_1[Xianc]
                Xi[:],logpqratio=innov(Xi_1_j,yi,theta)
                loglh=lh(Xi,yi,theta)
                log_wi=loglh+logpqratio+np.log(wi_1_j)-log_v_1_j#-normf
                wi[:]=np.exp(log_wi-logsumexp(log_wi))
                ess_w = np.exp(-logsumexp(2*np.log(wi)))
                print("Final w_ESS/N = {} for wi at iters {} {}".format(ess_w/n,initer, outiter))
        return logsumexp(log_wi) + logsumexp(log_v_1) - np.log(n)

    @staticmethod
    #@numba.jit
    def garbageauxilliaryparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf):
        # Note the way Xianc ancestry are stored, it is on the parent state Xi_1
        # Compute the posterior integral p(y_t | x_{t-1})
        Xi_bar, lpqr = propnf(Xi_1,yi,theta)
        loglh_nf = lh(Xi_bar,yi,theta) 
        log_g_j = np.zeros_like(loglh_nf)
        log_g_j[:] = 2*(loglh_nf + lpqr) # sqrt of sqr # should np.copy()
        Xianc[:] = range(n)
        L = 1 # number of times g_j has been approximated
        if True:
            temps = 1000
            ntemp = 1. / temps
            max_ess_v = 0
            argmax_ess_v = 0
            log_v_1 = np.log(wi_1)
            v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
            for temp in range(temps):
                log_v_1=np.log(v_1_norm) + ntemp * 0.5 * log_g_j
                v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
                ess_v = np.exp(-logsumexp(2*np.log(v_1_norm)))
                if ess_v < n*0.5:
                    print("Resampling L = {}, ess_v/n = {}".format(L,ess_v/n))
                    # resample
                    new_j=resample(v_1_norm,n) # j_i, the new jth parent for the ith particle , should be Xianc_1
                    Xianc[:] = Xianc[new_j]
                    log_g_j[:] = log_g_j[new_j]
                    v_1_norm[:] = 1./n
                    # simulate forward again?
                    Xi_bar_, lpqr_ = propnf(Xi_1[Xianc[:]],yi,theta)
                    loglh_nf_ = lh(Xi_bar_,yi,theta) 
                    log_g_j[:] = logsumexp_pair(log_g_j + np.log(L),2*(loglh_nf_ + lpqr_)) - np.log(L+1)
                    L += 1
                    # now run Gibbs. The variables are the indices which we will sample from 
                    # full conditionals are ...?
                if temp % 100 == 0:
                    print("L = {}, ess_v/n = {}".format(L,ess_v/n))
                #print("Iterated ESS = {} for iter {}".format(ess_v/n, temp))
            #set final weights
            log_v_1=np.log(wi_1[Xianc]) + 0.5 * log_g_j
            v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
            ess_v = np.exp(-logsumexp(2*np.log(v_1_norm)))
            print("ESS_V/N = {}".format(ess_v/n))
            print("Comparison ESS = {} for 1/Nsum(wi_1^2)".format(np.exp(-logsumexp(2*np.log(wi_1))-np.log(n))))
        #Xianc[:]=resample(v_1_norm,n) # j_i, the new jth parent for the ith particle , should be Xianc_1
        Xi_1_j=Xi_1[Xianc,:]
        wi_1_j=wi_1[Xianc]
        log_v_1_j=log_v_1[Xianc]
        log_nf_j=loglh_nf[Xianc]
        Xi[:],logpqratio=innov(Xi_1_j,yi,theta)
        loglh=lh(Xi,yi,theta)
        log_wi=loglh+logpqratio+np.log(wi_1_j)-log_v_1_j
        wi[:]=np.exp(log_wi-logsumexp(log_wi))
        ess_w = np.exp(-logsumexp(2*np.log(wi)))
        print("ESS_W/N = {} for final wi".format(ess_w/n))
        return logsumexp(log_wi) + logsumexp(log_v_1) - np.log(n)

    @staticmethod
    @numba.jit
    def alphaauxilliaryparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf):
        # Note the way Xianc ancestry are stored, it is on the parent state Xi_1
        # Compute the posterior integral p(y_t | x_{t-1})
        Xi_bar, lpqr = propnf(Xi_1,yi,theta)
        loglh_nf = lh(Xi_bar,yi,theta) 
        if True:
            temps = 100
            ntemp = 1. / temps
            max_ess_v = 0
            argmax_ess_v = 0
            for epsilon in range(temps):
                log_v_1=np.log(wi_1) + (ntemp * epsilon) * loglh_nf
                v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
                ess_v = np.exp(-logsumexp(2*np.log(v_1_norm)))
                if ess_v > max_ess_v:
                    argmax_ess_v = epsilon
                    max_ess_v = ess_v
                #print("Iterated ESS = {} for iter {}".format(ess_v/n, epsilon))
            log_v_1=np.log(wi_1) + (ntemp * argmax_ess_v) * loglh_nf
            v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
            ess_v = np.exp(-logsumexp(2*np.log(v_1_norm)))
            #print("Iterated ESS = {}".format(ess_v/n))
            #print("log(gl/gu)^2 = {}".format(2*(loglh_nf.min() - loglh_nf.max())))
            #print("Comparison ESS = {} for 1/Nsum(wi_1^2)".format(np.exp(-logsumexp(2*np.log(wi_1))-np.log(n))))
            #print("Comparison ESS = {} for 1/Nsum(gi^2)".format(np.exp(-logsumexp(2*(loglh_nf-logsumexp(loglh_nf)))-np.log(n))))
            #print("Comparison ESS = {} for 1/Nsum((wi_1gi)^2)".format(np.exp(-logsumexp(2*((np.log(wi_1)+loglh_nf)-logsumexp(np.log(wi_1)+loglh_nf)))-np.log(n))))
        Xianc[:]=resample(v_1_norm,n) # j_i, the new jth parent for the ith particle , should be Xianc_1
        Xi_1_j=Xi_1[Xianc,:]
        wi_1_j=wi_1[Xianc]
        log_v_1_j=log_v_1[Xianc]
        log_nf_j=loglh_nf[Xianc]
        Xi[:],logpqratio=innov(Xi_1_j,yi,theta)
        loglh=lh(Xi,yi,theta)
        log_wi=loglh+logpqratio+np.log(wi_1_j)-log_v_1_j
        wi[:]=np.exp(log_wi-logsumexp(log_wi))
        ess_w = np.exp(-logsumexp(2*np.log(wi)))
        #print("ESS/N = {} for final wi".format(ess_w/n))
        return logsumexp(log_wi) + logsumexp(log_v_1) - np.log(n)

    @staticmethod
    @numba.jit
    def essauxilliaryparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf):
        # Note the way Xianc ancestry are stored, it is on the parent state Xi_1
        # Compute the posterior integral p(y_t | x_{t-1})
        Xi_bar, lpqr = propnf(Xi_1,yi,theta)
        loglh_nf = lh(Xi_bar,yi,theta) 
        #loglh_nf = lh(propnf(Xi_1,yi,theta),yi,theta)
        #ess = np.exp(-logsumexp(2*np.log(wi_1)))
        #print("ESS = {}".format(ess))
        #log_v_1=np.log(wi_1) + loglh_nf
        #log_v_1=np.log(wi_1) + (ess/n) * loglh_nf
        if True:
            #ess_v = 0.
            #for epsilon in range(10):
            #    log_v_1=np.log(wi_1) + (ess_v/n) * loglh_nf
            #    v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
            #    ess_v = np.exp(-logsumexp(2*np.log(v_1_norm)))
            #    print("Below ESS = {} for iter {}".format(ess_v, epsilon))
            #ess_v = 1. * n
            ess_v = np.exp(-logsumexp(2*np.log(wi_1)))
            #ess_v = max(np.exp(-logsumexp(2*np.log(wi_1))),np.exp(-logsumexp(2*(loglh_nf-logsumexp(loglh_nf)))))
            #ess_v = 0
            #ess_v = np.exp(-logsumexp(2*(loglh_nf-logsumexp(loglh_nf))))
            for epsilon in range(20):
                log_v_1=np.log(wi_1) + (ess_v/n) * loglh_nf
                #log_v_1=np.log(wi_1) + (1-ess_v/n) * loglh_nf
                v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
                ess_v = np.exp(-logsumexp(2*np.log(v_1_norm)))
                #print("Iterated ESS = {} for iter {}".format(ess_v/n, epsilon))
            #print("Iterated ESS = {}".format(ess_v/n))
            #print("log(gl/gu)^2 = {}".format(2*(loglh_nf.min() - loglh_nf.max())))
            #ess_v = 1. * n
            #ess_v = np.exp(-logsumexp(2*np.log(wi_1)))
            #for epsilon in range(100):
            #    log_v_1=(ess_v/n) * np.log(wi_1) + (ess_v/n) * loglh_nf
            #    v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
            #    ess_v = np.exp(-logsumexp(2*np.log(v_1_norm)))
            #    print("Full power ESS = {} for iter {}".format(ess_v, epsilon))
            #print("Comparison ESS = {} for 1/Nsum(wi_1^2)".format(np.exp(-logsumexp(2*np.log(wi_1))-np.log(n))))
            #print("Comparison ESS = {} for 1/Nsum(gi^2)".format(np.exp(-logsumexp(2*(loglh_nf-logsumexp(loglh_nf)))-np.log(n))))
            #print("Comparison ESS = {} for 1/Nsum((wi_1gi)^2)".format(np.exp(-logsumexp(2*((np.log(wi_1)+loglh_nf)-logsumexp(np.log(wi_1)+loglh_nf)))-np.log(n))))
        else:
            ess_v = 0.5 * n
            log_v_1=np.log(wi_1) + (ess_v/n) * loglh_nf
            v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
            #ess_v = np.exp(-logsumexp(2*np.log(v_1_norm)))
        Xianc[:]=resample(v_1_norm,n) # j_i, the new jth parent for the ith particle , should be Xianc_1
        Xi_1_j=Xi_1[Xianc,:]
        wi_1_j=wi_1[Xianc]
        # secret sauce, I don't know how it works.
        normf = logsumexp(loglh_nf) - logsumexp((1-ess_v/n)*loglh_nf) # sum(g)/sum(g^alpha) , alpha = ess_v/n
        #print("logsum(lh_nf) = {}, normf = {}".format(logsumexp(loglh_nf),normf))
        #log_v_1 += normf
        log_v_1_j=log_v_1[Xianc]
        log_nf_j=loglh_nf[Xianc]
        #log_v_1_r_j=np.log(wi_1_j) + loglh_nf[Xianc]
        #log_v_1n_j=np.log(v_1_norm[Xianc])
        Xi[:],logpqratio=innov(Xi_1_j,yi,theta)
        loglh=lh(Xi,yi,theta)
        #normf = logsumexp(log_nf_j) - logsumexp((1-ess_v/n)*log_nf_j) # sum(g)/sum(g^alpha) , alpha = ess_v/n
        log_wi=loglh+logpqratio+np.log(wi_1_j)-log_v_1_j#-normf
        wi[:]=np.exp(log_wi-logsumexp(log_wi))
        ess_w = np.exp(-logsumexp(2*np.log(wi)))
        #print("ESS = {} for final wi".format(ess_w))
        #return logsumexp(log_wi)-np.log(n)
        return logsumexp(log_wi) + logsumexp(log_v_1) - np.log(n)

    #@numba.jit
    @staticmethod
    def improvedauxilliaryparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf):
        """
        Elvira, requires pointwise evaluable transition densities
        """
        # Note the way Xianc ancestry are stored, it is on the parent state Xi_1
        # Compute the posterior integral p(y_t | x_{t-1})
        Xi_bar = propnf(Xi_1,yi,theta)
        Xi_alt,lpqalt = innov(Xi_1,yi,theta)
        var_all = (Xi_alt-Xi_bar)**2
        var = np.sum((1./n)*var_all,axis=0)
        #print("Var = {}".format(var))
        loglh_nf = lh(Xi_bar,yi,theta)
        #print("initial wi_1 = {}".format(np.log(wi_1[:100])))
        #numer = logmultdiff(np.log(wi_1),Xi_bar,Xi_alt,var)
        #denom = logmultdiff(np.zeros(n),Xi_bar,Xi_alt,var)# - np.log(n)
        numer = logmultdiff2(np.log(wi_1),Xi_bar,Xi_alt,var_all)
        denom = logmultdiff2(np.zeros(n),Xi_bar,Xi_alt,var_all)# - np.log(n)
        log_v_1=loglh_nf + numer - denom
        #log_v_1=loglh_nf + np.log(wi_1) 
        tmp = numer-denom
        #print("Numer/denom - max = {}".format(tmp - tmp.max()))
        v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
        Xianc[:]=resample(v_1_norm,n) # j_i, the new jth parent for the ith particle , should be Xianc_1
        Xi_1_j=Xi_1[Xianc,:]
        wi_1_j=wi_1[Xianc]
        log_v_1_j=log_v_1[Xianc]
        #print("wi_1_j = {}".format(np.log(wi_1_j[:100])))
        #print("Xianc = {}".format(Xianc[:100]))
        # idea. for p(x_t|p_{t-1}) in final weight calc, use reciprocal of count of incidence of the Xianc entry for that m.
        #print("log_v_1_j = {}".format(log_v_1_j[:100]))
        u, fromi, indices, counts = np.unique(Xianc,return_index=True,return_inverse=True,return_counts=True)
        pXiXi_1 = np.log(1./counts[indices])
        Xi[:],logpqratio=innov(Xi_1_j,yi,theta)
        Xi[fromi,...] = Xi_alt[u,...]
        loglh=lh(Xi,yi,theta)
        Xi_bar_j = Xi_bar[Xianc]
        Xi_alt_j = Xi_alt[Xianc]
        var_j_all = (Xi_alt_j-Xi_bar_j)**2
        var_j = np.sum((1./n)*var_j_all,axis=0)
        #fnumer = logmultdiff(np.log(wi_1_j),Xi,Xi,var)
        #fdenom = logmultdiff(log_v_1_j,Xi,Xi,var)
        #fnumer = logmultdiff(np.log(wi_1) + pXiXi_1,Xi,Xi_bar,var) 
        #fdenom = logmultdiff(log_v_1 + pXiXi_1,Xi,Xi_bar,var) 
        #fnumer = logmultdiff(np.log(wi_1_j) + pXiXi_1,Xi,Xi_bar_j,var_j) 
        #fdenom = logmultdiff(log_v_1_j + pXiXi_1,Xi,Xi_bar_j,var_j) 
        # Next thing to try... don't resample???? Or are we just cancelling this out with the pXiXi_1 weight?
        fnumer = logmultdiff2(np.log(wi_1_j) + pXiXi_1,Xi,Xi_alt_j,var_j_all) 
        fdenom = logmultdiff2(log_v_1_j + pXiXi_1,Xi,Xi_alt_j,var_j_all) 
        #log_wi=loglh+logpqratio+logmultdiff(np.log(wi_1_j),Xi,Xi,var)-logmultdiff(log_v_1_j,Xi,Xi,var)
        log_wi=loglh+logpqratio+fnumer-fdenom
        #ftmp = fnumer-fdenom
        #print("Final Numer/denom - max = {}".format(ftmp - ftmp.max()))
        #print("Final Numer/denom - apf = {}".format(ftmp - (np.log(wi_1_j)-log_v_1_j)))
        #log_wi=loglh+logpqratio+np.log(wi_1_j)-log_v_1_j
        wi[:]=np.exp(log_wi-logsumexp(log_wi))
        #print("final wi = {}".format(wi[:100]))
        return logsumexp(log_wi)-np.log(n)
    
    @classmethod
    def meanauxilliaryparticlefilter(cls,yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov,propnf):
        # Note the way Xianc ancestry are stored, it is on the parent state Xi_1
        # Compute the posterior integral p(y_t | x_{t-1})
        M = 100
        Xi_1_map = np.zeros((Xi_1.shape[0],Xi_1.shape[1],M)) # we compute the "maximum likelihood" 
        loglh_nf = np.zeros((Xi_1.shape[0],M)) # we compute the "maximum likelihood" 
        for mi in range(M):
            Xi_1_map[...,mi], lpqr = propnf(Xi_1,yi,theta)
            loglh_nf[:,mi] = lh(Xi_1_map[...,mi],yi,theta)
        #log_v_1=np.log(wi_1) + logsumexp_mat(loglh_nf,axis=1) - math.log(M)
        log_v_1=np.log(wi_1) + lh(np.mean(Xi_1_map,axis=2),yi,theta)
        #log_v_1=np.log(wi_1) + loglh_nf.max(axis=1)
        v_1_norm=np.exp(log_v_1-logsumexp(log_v_1))
        Xianc[:]=resample(v_1_norm,n) # j_i, the new jth parent for the ith particle , should be Xianc_1
        Xi_1_j=Xi_1[Xianc,:]
        wi_1_j=wi_1[Xianc]
        log_v_1_j=log_v_1[Xianc]
        Xi[:],logpqratio=innov(Xi_1_j,yi,theta)
        loglh=lh(Xi,yi,theta)
        log_wi=loglh+logpqratio+np.log(wi_1_j)-log_v_1_j
        wi[:]=np.exp(log_wi-logsumexp(log_wi))
        return logsumexp(log_wi)-np.log(n)

    @classmethod
    def debugbootstrapparticlefilter(cls,yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov):
        if not np.isfinite(wi_1).all():
            print("Infinite entries in wi_1: {}".format(np.argwhere(~np.isfinite(wi_1))))
            sys.exit(0)
        # Resample
        wi_1_norm = np.exp(np.log(wi_1) - logsumexp(np.log(wi_1))) # should be normalised
        Xianc[:]=resample(wi_1_norm,n)
        Xi_1_rs=Xi_1[Xianc[:],:]
        #wi[:]=wi[Xianc]
        # Propagatge
        Xi[:],logpqratio = innov(Xi_1_rs,yi,theta)
        loglh = lh(Xi,yi,theta)
        if not np.isfinite(loglh).all():
            print("log likelihoods not finite")
            print("Infinite entries of loglh: {}".format(loglh[~np.isfinite(loglh)]))
            sys.exit(0)
        # Compute weights
        log_wi_next = loglh + logpqratio #- np.log(n) # non-normalised, assumes previous weights are 1./n
        # trying something slightly different
        max_weight = np.max(log_wi_next)
        wi_next_conditioned = np.exp(log_wi_next - max_weight)
        sum_weights = wi_next_conditioned.sum()
        wi_norm = wi_next_conditioned / sum_weights
        logsumexp_log_wi_next = max_weight + np.log(sum_weights)
        #logsumexp_log_wi_next = logsumexp(log_wi_next)
        #log_wi_norm = log_wi_next - logsumexp_log_wi_next # normalise weights
        #wi_norm = np.exp(log_wi_norm)
        wi[:] = np.exp(log_wi_next)
        if not np.isfinite(wi).all():
            print("logpqratio: {}".format(logpqratio))
            print("log_wi_next: {}".format(log_wi_next))
            print("Infinite entries of wi: {}".format(wi[~np.isfinite(wi)]))
            print("Infinite entries of wi: {}".format(np.argwhere(~np.isfinite(wi))))
            sys.exit(0)
        if not np.isfinite(logsumexp_log_wi_next):
            print("Divide by zero sum of weights. Log sum exp(log_wi_next) = {}".format(logsumexp_log_wi_next))
            sys.exit(0)
        logml = logsumexp_log_wi_next - np.log(n)
        return logml

    @staticmethod
    @numba.jit
    def bootstrapparticlefilter(yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov):
        # Resample
        Xianc[:]=resample(wi_1,n)
        Xi_1_rs=Xi_1[Xianc[:],:]
        # Propagate
        Xi[:],logpqratio = innov(Xi_1_rs,yi,theta)
        loglh = lh(Xi,yi,theta)
        # Compute weights
        log_wi = loglh + logpqratio 
        logsumexp_log_wi = logsumexp(log_wi)
        wi[:] = np.exp(log_wi - logsumexp_log_wi)
        return logsumexp_log_wi - np.log(n)

    @classmethod
    def experimentalparticlefilter(cls,yi,Xi,Xi_1,theta,Xianc,wi_1,wi,n,lh,innov):
        Xi[:],logpqratio = innov(Xi_1,yi,theta) # TODO the theta here assumes a static timestep. We'll need to record it somehow in the pmpfl main function.
        loglh = lh(Xi,yi,theta)
        if not np.isfinite(loglh).all():
            print("log likelihoods not finite")
            nanidx = np.argwhere(~np.isfinite(loglh))
            print (nanidx.shape,loglh[nanidx].shape,np.squeeze(Xi[nanidx,:]).shape)
            print (np.hstack((nanidx,wi[nanidx],np.squeeze(Xi[nanidx,:]),np.squeeze(Xi_1[nanidx,:]))))
            sys.exit(0)
        #loglh = np.nan_to_num(loglh)
        log_wi_next = np.log(wi_1) + loglh + logpqratio # non-normalised
        logsumexp_log_wi_next = logsumexp(log_wi_next)
        #print("log marginal likelihood = {}".format(logml))
        if not np.isfinite(wi_1).all():
            print("Infinite entries of wi_1: {}".format(wi_1[~np.isfinite(wi_1).all()]))
        log_wi_norm =  log_wi_next - logsumexp_log_wi_next # normalise weights
        wi_norm = np.exp(log_wi_norm)
        wi[:] = np.exp(log_wi_next) # np.exp(log_wi_next)
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
            Xianc[:]=resample(wi_norm,n)
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

@numba.jit("f8[:](f8[:],f8[:,:],f8[:,:],f8[:,:])")
def logmultdiff2(w,x0,x,v):
    r = np.zeros_like(w)
    for i in range(x.shape[0]):
        #print("(x-x0[i])**2/v = {}".format((x[:100]-x0[i,...])**2/v))
        #print("w - 0.5 * sum(x[i]-x0[i])**2/v = {}".format(w[i] - 0.5 * np.sum((x[i,...]-x0[i,...])**2/v)))
        #print("x0[i]={},x0[i,...]={}".format(x0[i],x0[i,...]))
        r[i] = logsumexp(w - 0.5 * np.sum((x - x0[i,...])**2/v[i,...]))
    return r

#@numba.jit
@numba.jit("f8[:](f8[:],f8[:,:],f8[:,:],f8[:])")
def logmultdiff(w,x0,x,v):
    r = np.zeros_like(w)
    for i in range(x.shape[0]):
        #print("(x-x0[i])**2/v = {}".format((x[:100]-x0[i,...])**2/v))
        #print("w - 0.5 * sum(x[i]-x0[i])**2/v = {}".format(w[i] - 0.5 * np.sum((x[i,...]-x0[i,...])**2/v)))
        #print("x0[i]={},x0[i,...]={}".format(x0[i],x0[i,...]))
        r[i] = logsumexp(w - 0.5 * np.sum((x - x0[i,...])**2/v))
    return r

@numba.jit("i4[:](f8[:],i4)")
def resample(weights,n):
    #n = weights.shape[0]
    indices = np.zeros_like(weights,dtype=np.int32)
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
