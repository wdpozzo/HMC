import numpy as np
#import numpy
import ray
from raynest.proposal import Proposal
from scipy.stats import multivariate_normal
from tqdm import tqdm
from collections import deque
from scipy.special import logsumexp
import os
import h5py
import ray
from raynest.nest2pos import autocorrelation, acl
#
#@ray.remote
class NUTS:
    """
    HamiltonianMonteCarlo acceptance rule
    for :obj:`raynest.proposal.HamiltonianProposal`
    """
    
    def __init__(self,
                 model,
                 dt          = 3e-2,
                 mass_matrix = None,
                 rng         = None,
                 max_storage = None,
                 output      = '.',
                 verbose     = 0):

        self.verbose        = verbose
        self.model          = model
        self.output         = output
        self.dt             = dt
        self.prior_bounds   = model.bounds
        
        if rng == None:
            self.rng            = np.random.default_rng()
        else:
            self.rng            = rng
        
        self.max_storage        = max_storage
        self.samples            = deque(maxlen = self.max_storage) # the list of samples from the mcmc chain

        if mass_matrix is None:
            self.mass_matrix = np.diag(np.ones(len(self.model.bounds))) #/np.abs(b[1]-b[0])
        else:
#            # determine the nature of the matrix, is it a callabe function or a numpy array
#            if callable(mass_matrix):
#                self.mass_matrix = mass_matrix()
#            else:
#                self.mass_matrix = mass_matrix
            self.mass_matrix = mass_matrix
        
        self.inverse_mass_matrix  = np.linalg.inv(self.mass_matrix)
        self.logdet               = np.linalg.slogdet(self.mass_matrix)[1]
        self.momenta_distribution = multivariate_normal(cov=self.mass_matrix, seed = self.rng)
        self.step_tuning = DualAveragingStepSize(initial_step_size=self.dt)

    def kinetic_energy(self, p):
        return -0.5*np.dot(p.T,np.dot(self.inverse_mass_matrix,p))-0.5*self.logdet + 0.5*len(p)*np.log(2*np.pi)
        
    def sample(self, q0, N=1000, position=0):
    
        chain = np.empty((2*N, len(q0)))  # Preallocate storage for samples
        sub_accepted, sub_counter = 0, 1

        if self.verbose:
            progress_bar_sampling = tqdm(total=N, desc=f"Sampling {position}", position=position)

        while sub_accepted < N:
        
            p0 = self.momenta_distribution.rvs()
            logP = self.model.log_posterior(q0) - self.kinetic_energy(p0)
            logu = logP - self.rng.exponential()

            q_l, q_r = q0.copy(), q0.copy()
            p_l, p_r = p0.copy(), p0.copy()

            j, s, n = 0, 1, 1

            while s == 1:
                v = self.rng.choice((-1, 1))

                if v == -1:
                    p_l, q_l, _, _, qprime, nprime, sprime = self.build_tree(p_l, q_l, logu, v, j, self.dt)
                else:
                    _, _, p_r, q_r, qprime, nprime, sprime = self.build_tree(p_r, q_r, logu, v, j, self.dt)

                if sprime:
                    alpha = min(1, nprime / n)
                    
                    if self.rng.uniform() < alpha:
#                        print("accepted", sub_counter, len(chain), sub_accepted)
                        q0[:] = qprime  # Avoid extra copying
                        chain[sub_accepted] = q0  # Store sample directly in preallocated array
                        sub_accepted += 1

                        if self.verbose:
                            progress_bar_sampling.update(1)

                n += nprime
                delta_q = q_r - q_l
                s = sprime * (np.dot(delta_q, p_l) > 0) * (np.dot(delta_q, p_r) > 0)
                j += 1

            self.acceptance = sub_accepted / (sub_counter+sub_accepted)

            if self.verbose and sub_counter % 10 == 0:  # Update less frequently for efficiency
                progress_bar_sampling.set_postfix({"acceptance rate": f"{self.acceptance:.3f}"})
            
            sub_counter += 1
            
        samples = chain[:N]  # Only keep accepted samples
        ACL = np.array([acl(samples[:, i], c=5) for i in range(chain.shape[1])])

        thinning = max(int(max(ACL)), 1)
        print(f"thinning = {thinning}")
        thinning = 1
        self.save_output(samples[::thinning, :])
        return samples[::thinning, :]
    
    def reset_samples(self):
        self.samples = deque(maxlen = self.max_storage)
        return []
    
    def sample_covariance(self, chain):
        
        D = len(self.model.bounds)
        N = len(chain)
        cov_array = np.zeros((D,N))

        if D == 1:
            name = chain[0].names[0]
            covariance = np.atleast_2d(np.var([chain[j][name] for j in range(N)]))
            mean       = np.atleast_1d(np.mean([chain[j][name] for j in range(N)]))
        else:
            for i,name in enumerate(chain[0].names):
                for j in range(N): cov_array[i,j] = chain[j][name]
            covariance = np.cov(cov_array)
            mean       = np.mean(cov_array,axis=1)
        del chain
        return covariance

    def save_output(self, samples):

        print("Saving output in {0}".format(self.output))
        h = h5py.File(os.path.join(self.output,"hmc.h5"),'w')
        grp = h.create_group("combined")
        dt = np.dtype({'names':self.model.names,
                       'formats':[ (float)]*len(self.model.names)})
        rec_arr = np.rec.array(samples,dtype=dt)
        grp.create_dataset("posterior_samples", data = rec_arr)
        h.close()

    def leap_frog(self, dt, p0, q0):
        p = p0.copy()
        q = q0.copy()

        grad_q = self.model.gradient(q)
        p += 0.5 * dt * grad_q  # First half-step momentum update
        q += dt * p  # Full-step position update

        # Reflect q against bounds
        bounds = np.array([np.array(v, dtype=np.float64) for v in self.prior_bounds.values()])
        lower_bounds, upper_bounds = bounds.T
        over_upper = q > upper_bounds
        under_lower = q < lower_bounds

        reflect_factor = np.where(over_upper | under_lower, -1.0, 1.0)
        q = np.where(q < lower_bounds, 2*lower_bounds-q, q)#np.clip(q, lower_bounds, upper_bounds)  # Clip instead of multiple conditions
        q = np.where(q > upper_bounds, 2*upper_bounds-q, q)
        
        p *= reflect_factor  # Flip momentum for out-of-bound coordinates

        # Final momentum update
        grad_q = self.model.gradient(q)
        p += 0.5 * dt * grad_q

        return p, q

    def build_tree(self, p, q, logu, v, j, dt):
#        print("j = ",j, "logu = ",logu)
        if j == 0:
            # Base case: Take one leapfrog step in the direction of v
            pprime, qprime = self.leap_frog(v*dt, p, q)
            logH = self.model.log_posterior(qprime)-self.kinetic_energy(pprime)
#            print("base level ",pprime, qprime, logH, logu, logu <= logH, logH > logu - 1000)
            nprime = int(logu <= logH)
            sprime = int(logH > logu - 1000)
            return pprime, qprime, pprime, qprime, qprime, nprime, sprime
        
        else:
            # Recursion: Build the left and right subtrees
            pprime_l, qprime_l, pprime_r, qprime_r, qprime, nprime, sprime = self.build_tree(p, q, logu, v, j-1, dt)

            if sprime:
#                print("recursing j =",j, sprime)
                if v == -1:
                    pprime_l, qprime_l, _, _, qpprime, npprime, spprime = self.build_tree(pprime_l, qprime_l, logu, v, j-1, dt)
                else:
                    _, _, pprime_r, qprime_r, qpprime, npprime, spprime = self.build_tree(pprime_r, qprime_r, logu, v, j-1, dt)
                
#                print("nprime = {} npprime = {}".format(nprime,npprime))
                if self.rng.uniform() < npprime/max(nprime+npprime,1):
                    qprime = qpprime
                
                delta_q = qprime_r-qprime_l
                sprime = spprime*(np.dot(delta_q,pprime_l)>=0)*(np.dot(delta_q,pprime_r)>=0)
                nprime = nprime + npprime
            return pprime_l, qprime_l, pprime_r, qprime_r, qprime, nprime, sprime

    def rmhmc_hamilton(self, p, q):
        """Compute dq/dt and dp/dt for Riemannian HMC."""
        G = self.model.metric(q)
        G_inv = jnp.linalg.inv(G)
        Gamma = self.model.christoffel_symbols(q)

        dq_dt = jnp.dot(G_inv, p)
        dp_dt = -self.model.gradient(q)

        # Christoffel correction term
        for i in range(q.shape[0]):
            dp_dt -= 0.5 * jnp.dot(p, jnp.dot(Gamma[i], p))

        return dq_dt, dp_dt
        
    def rmhmc_leapfrog(p, q, dt):
        """Single RMHMC leapfrog step."""
        q_new, p_new = q, p

        for _ in range(3):  # Implicit update (usually need fixed-point iteration)
            dq, dp = self.rmhmc_hamilton(q_new, p)
            q_new = q + step_size * dq
            p_new = p + step_size * dp

        return p_new, q_new
    
class DualAveragingStepSize:
    
    def __init__(self, initial_step_size, target_accept=0.5, gamma=0.05, t0=10.0, kappa=0.75):
    
        self.mu = np.log(10 * initial_step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0

    def update(self, p_accept):
        # Running tally of absolute error. Can be positive or negative. Want to be 0.
        self.error_sum += self.target_accept - p_accept

        # This is the next proposed (log) step size. Note it is biased towards mu.
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)

        # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
        eta = self.t ** -self.kappa

        # Smoothed average step size
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step

        # This is a stateful update, so t keeps updating
        self.t += 1

        # Return both the noisy step size, and the smoothed step size
        return np.exp(log_step), np.exp(self.log_averaged_step)
        
if __name__ == "__main__":

    from scipy.stats import norm
    from raynest.nest2pos import autocorrelation, acl
    import jax
    import jax.numpy as jnp
    from jax import grad
    from functools import partial
    
    class TestModel:
        
        def __init__(self, n, b):
            self.names  = n
            self.bounds = b
        
        def log_prior(self, q):
            return 0.0
        
        def log_likelihood(self, q):
            return -0.5*np.sum(q**2)
        
        def log_posterior(self, q):
            return self.log_prior(q)+self.log_likelihood(q)

        def potential(self, q):
            return -self.log_posterior(q)
        
        def gradient(self, q):
            return -q
     
#    ray.init()
    
    dimension = 20
    names = ["{}".format(i) for i in range(dimension)]
    bounds = [[-10,10] for _ in names]
    
    n_threads  = 1
    n_samps    = 1e4
    n_train    = 1e3
    e_train    = 0
    adapt_mass = 0
    verbose    = 1
    n_bins     = int(np.sqrt(n_samps))
    
    rng       = [np.random.default_rng(1111+j) for j in range(n_threads)]

    M         = TestModel(names, bounds)
    HMC       = [NUTS(M, rng = rng[j], verbose = verbose) for j in range(n_threads)]
    print(HMC)
    samples   = np.concatenate([H.sample(rng[j].uniform(-10,10,dimension),
                          N=int(n_samps//n_threads),
                          position=j)
                 for j,H in enumerate(HMC)])
    
#    import matplotlib.pyplot as plt
#    from corner import corner
#    corner(samples,
#                     labels=names,
#                     quantiles=[0.05, 0.5, 0.95], truths = None,
#                     show_titles=True, title_kwargs={"fontsize": 12}, smooth2d=1.0)
#    
#    plt.savefig("corner.pdf",bbox_inches='tight')
    
#    ray.shutdown()
