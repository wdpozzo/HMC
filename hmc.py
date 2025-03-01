import numpy as np
#import numpy
from raynest.nest2pos import autocorrelation, acl
import ray
from raynest.proposal import Proposal
from scipy.stats import multivariate_normal
from tqdm import tqdm
from collections import deque
from scipy.special import logsumexp
import os
import h5py
import ray
#
#@ray.remote
class HamiltonianMonteCarlo:
    """
    HamiltonianMonteCarlo acceptance rule
    for :obj:`raynest.proposal.HamiltonianProposal`
    """
    
    def __init__(self,
                 model,
                 proposal,
                 rng = None,
                 max_storage = None,
                 output = '.',
                 verbose = 0):

        self.verbose        = verbose
        self.model          = model
        self.output         = output
        
        if rng == None:
            self.rng            = np.random.default_rng()
        else:
            self.rng            = rng
        
        self.max_storage        = max_storage
        self.samples            = deque(maxlen = self.max_storage) # the list of samples from the mcmc chain
        
        self.mass_matrix = np.diag(np.ones(len(self.model.bounds))) #/np.abs(b[1]-b[0])
        self.proposal = proposal(self.model, self.rng, self.mass_matrix)
        self.step_tuning = DualAveragingStepSize(initial_step_size=self.proposal.dt)

    def sample(self, oldparam, n = 1000, t = 0, t_epochs = 3, mass_estimate = 0, position = 0):
        
        chain = []
        
        sub_accepted = 0
        sub_counter  = 1
        
        for training in range(t_epochs):
            
            t_samples = []
            if self.verbose:
                progress_bar_tuning   = tqdm(total = int(t), desc = "Tuning {} - cycle {}".format(position, training+1), position=position)
            
            for j in range(int(t)):
                
                newparam     = self.proposal.get_sample(oldparam.copy())

                if self.proposal.log_J > np.log(self.rng.uniform()):
                
                    oldparam        = newparam.copy()
                    # append the sample to the array of samples
                    t_samples.append(oldparam)
                    sub_accepted   += 1
                
                self.acceptance     = float(sub_accepted)/float(sub_counter)
                self.proposal.dt, _ = self.step_tuning.update(self.acceptance)
                if self.verbose:
                    pbar_updates = {'dt':f"{self.proposal.dt:.4f}"}
                    progress_bar_tuning.set_postfix(pbar_updates)
                    progress_bar_tuning.update(1)
                
                sub_counter += 1
            
            if mass_estimate == 1:
            
                self.inverse_mass_matrix = self.sample_covariance(t_samples)
                print("mass matrix =", np.linalg.inv(self.inverse_mass_matrix))
                self.proposal.update_mass_matrix(self.inverse_mass_matrix)
            _, self.proposal.dt = self.step_tuning.update(self.acceptance)
            
            # compute the ACL on the max_storage set of samples, choosing a window length of 5*tau
            if self.verbose: progress_bar_tuning.close()
    
        if self.verbose:
            progress_bar_sampling = tqdm(total = int(n), desc = "Sampling {}".format(position), position=position)
        
        while len(chain) < n:

            newparam     = self.proposal.get_sample(oldparam.copy())

            if self.proposal.log_J > np.log(self.rng.uniform()):
                
                oldparam        = newparam.copy()
                # append the sample to the array of samples
                chain.append(oldparam)
                
                if self.verbose:
                    progress_bar_sampling.update(1)
                sub_accepted   += 1
                
            self.acceptance     = float(sub_accepted)/float(sub_counter)
            
            if self.verbose:
                pbar_updates = {'acceptance rate': f"{self.acceptance:.3f}"}
                progress_bar_sampling.set_postfix(pbar_updates)
            sub_counter += 1
        
        if self.verbose:
            progress_bar_sampling.close()
    
        samples = np.array([x.values for x in chain])
        ACL = [acl(samples[:,i], c=5) for i in range(samples.shape[1])]
#        print("autocorrelation lengths = {}".format(ACL))
        thinning = int(max(ACL))
        print("thinning = {}".format(thinning))
        self.save_output(samples[::thinning])
        return chain[::thinning]
    
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
            
class HamiltonianProposal(Proposal):

    def __init__(self,
                 model,
                 rng,
                 mass_matrix):
        """
        Initialises the class with the kinetic
        energy and the :obj:`raynest.Model.potential`.
        """
        self.model                  = model
        self.rng                    = rng
        self.T                      = self.kinetic_energy
        self.V                      = self.model.potential
        self.gradient               = self.model.gradient
        self.prior_bounds           = self.model.bounds
        self.dimension              = len(self.prior_bounds)
        self.mass_matrix            = mass_matrix
        
        self.inverse_mass_matrix    = np.linalg.inv(self.mass_matrix)
        self.inverse_mass           = np.diag(self.inverse_mass_matrix)
        _, self.logdeterminant      = np.linalg.slogdet(self.mass_matrix)

        self.dt                     = 1e-3*self.dimension**0.25
        self.leaps                  = 100
        self.maxleaps               = 1000
        self.DEBUG                  = 0
        self.trajectories           = []
        self.set_momenta_distribution()

    def update_mass_matrix(self, inverse_mass_matrix):
        self.inverse_mass_matrix    = inverse_mass_matrix
        self.mass_matrix            = np.linalg.inv(self.inverse_mass_matrix)
        self.inverse_mass           = np.diag(self.inverse_mass_matrix)
        _, self.logdeterminant      = np.linalg.slogdet(self.mass_matrix)
        self.set_momenta_distribution()
        
    def set_momenta_distribution(self):
        """
        update the momenta distribution using the
        mass matrix (precision matrix of the ensemble).
        """
        self.momenta_distribution = multivariate_normal(cov=self.mass_matrix)
    
    def unit_normal(self, q):
        v = self.gradient(q)
        return v/np.linalg.norm(v)

    def force(self, q):
        """
        return the gradient of the potential function as numpy ndarray
        
        Parameters
        ----------
        q : :obj:`raynest.parameter.LivePoint`
            position

        Returns
        ----------
        dV: :obj:`numpy.ndarray` gradient evaluated at q
        """
        g = self.gradient(q)
        return np.array(g)#np.array([g[n] for n in q.names])
        
    def kinetic_energy(self,p):
        """
        kinetic energy part for the Hamiltonian.
        Parameters
        ----------
        p : :obj:`numpy.ndarray`
            momentum

        Returns
        ----------
        T: :float: kinetic energy
        """
        return 0.5 * np.dot(p,np.dot(self.inverse_mass_matrix,p))-self.logdeterminant

    def hamiltonian(self, p, q):
        """
        Hamiltonian.
        Parameters
        ----------
        p : :obj:`numpy.ndarray`
            momentum
        q : :obj:`raynest.parameter.LivePoint`
            position
        Returns
        ----------
        H: :float: hamiltonian
        """
        return self.T(p) + self.V(q)

class LeapFrog(HamiltonianProposal):
    """
    Leap frog integrator proposal for an unconstrained
    Hamiltonian Monte Carlo step
    """
    def get_sample(self, q0, *args):
        """
        Propose a new sample, starting at q0

        Parameters
        ----------
        q0 : :obj:`raynest.parameter.LivePoint`
            position

        Returns
        ----------
        q: :obj:`raynest.parameter.LivePoint`
            position
        """
        # generate a canonical momentum
        p0 = np.atleast_1d(self.momenta_distribution.rvs())
        initial_energy = self.hamiltonian(p0,q0)
        # evolve along the trajectory
        q, p, r = self.evolve_trajectory(p0, q0, *args)
        # minus sign from the definition of the potential
        final_energy   = self.hamiltonian(p, q)
        if r == 1:
            self.log_J = -np.inf
        else:
            self.log_J = min(0.0, initial_energy-final_energy)
        return q

    def evolve_trajectory(self, p0, q0, *args):
        """
        Hamiltonian leap frog trajectory subject to the
        hard boundary defined by the parameters prior bounds.
        https://arxiv.org/pdf/1206.1901.pdf

        Parameters
        ----------
        p0 : :obj:`numpy.ndarray`
            momentum
        q0 : :obj:`raynest.parameter.LivePoint`
            position

        Returns
        ----------
        p: :obj:`numpy.ndarray` updated momentum vector
        q: :obj:`raynest.parameter.LivePoint`
            position
        """
        # Updating the momentum a half-step
        p = p0 - 0.5 * self.dt * self.force(q0)
        q = q0.copy()
        
        for i in range(self.leaps):

            # do a step
            for j,k in enumerate(q.names):
                u,l = self.prior_bounds[j][1], self.prior_bounds[j][0]
                q[k] += self.dt * p[j] * self.inverse_mass[j]
                # check and reflect against the bounds
                # of the allowed parameter range
                while q[k] <= l or q[k] >= u:
                    if q[k] > u:
                        q[k] = u - (q[k] - u)
                        p[j] *= -1
                    if q[k] < l:
                        q[k] = l + (l - q[k])
                        p[j] *= -1

            F = self.force(q)
            # take a full momentum step
            p += - self.dt * F
        # Do a final update of the momentum for a half step
        p += - 0.5 * self.dt * F

        return q, -p, 0

class NUTS(HamiltonianProposal):
    
    def leap(self, dt, p0, q0):
        p = p0.copy()
        q = q0.copy()

        # Update position q using momentum p
        q.values += dt * p * self.inverse_mass

  # Reflect q against bounds
        lower_bounds, upper_bounds = np.array(self.prior_bounds).T
        #print('lower bounds', lower_bounds)
        #print('upper_bounds', upper_bounds)
        over_upper = q.values > upper_bounds
        under_lower = q.values < lower_bounds
        #print('before_reflection', q.values)
        #print('over upper', over_upper,)
        #print('under_lower', under_lower)
        q.values = np.where(over_upper,  upper_bounds - (q.values-upper_bounds), q.values)
        q.values = np.where(under_lower,  lower_bounds +(lower_bounds-q.values), q.values)
        #print("after reflection", q.values)
        #print("\n")

        # Reflect momentum for out-of-bound coordinates
        p = np.where(over_upper | under_lower, -p, p)

        # Update momentum using the force #WE ARE CARRYING OVER A MINUS SIGN!!!!
        F = self.force(q)
        p += dt * F

        return p, q, self.hamiltonian(p, q)
    
    def build_tree(self, p0, q0):
        maxdepth = 10
        q_r, q_l = q0.copy(), q0.copy()
        p_r, p_l = p0.copy(), p0.copy()

        all_qs = []
        all_ps = []
        all_H = []

        for _ in range(2 ** (maxdepth - 1)):
            # Randomly choose forward or backward direction
            v = 1 if self.rng.uniform() < 0.5 else -1
            dt = v * abs(self.dt)

            if v == 1:
                p_r, q_r, H_r = self.leap(dt, p_r, q_r)
                all_qs.append(q_r)
                all_ps.append(p_r)
                all_H.append(H_r)
            else:
                p_l, q_l, H_l = self.leap(dt, p_l, q_l)
                all_qs.append(q_l)
                all_ps.append(p_l)
                all_H.append(H_l)

            # Check for termination
            delta = q_r.values - q_l.values
            if np.dot(delta, p_l) < 0 or np.dot(delta, p_r) < 0:
                break

        # Normalize weights and sample from trajectory
        H = np.array(all_H)
        weights = np.exp(H - logsumexp(H))
        idx = self.rng.choice(len(H), p=weights)

        return -all_ps[idx], all_qs[idx], H[idx]

    def get_sample(self, q0):
        """
        Propose a new sample, starting at q0

        Parameters
        ----------
        q0 : :obj:`raynest.parameter.LivePoint`
            position

        Returns
        ----------
        q: :obj:`raynest.parameter.LivePoint`
            position
        """
        # generate a canonical momentum
        p0 = np.atleast_1d(self.momenta_distribution.rvs())
        initial_energy = self.hamiltonian(p0, q0)
        # evolve along the trajectory
        p, q, final_energy = self.build_tree(p0, q0)
        # minus sign from the definition of the potential
        self.log_J = min(0.0, initial_energy-final_energy)

        return q

class DualAveragingStepSize:
    
    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10.0, kappa=0.75):
    
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

    from raynest.model import Model
    from scipy.stats import norm
    from raynest.nest2pos import autocorrelation, acl
    import jax
    import jax.numpy as jnp
    from jax import grad
    from functools import partial


    class TestModel(Model):
        
        def __init__(self, n, b):
            self.names  = n
            self.bounds = b
        
        def log_prior(self, q):
            return 0.0
        
        def log_likelihood(self, q):
            return -0.5*np.sum(q.values**2)
        
        def log_posterior(self, q):
            return self.log_prior(q)+self.log_likelihood(q)

        def potential(self, q):
            return -self.log_posterior(q)
        
        def gradient(self, q):
            return -q.values
        

    class GaussianMultiModal(Model):

        def __init__(self, n, b):
            self.names  = n
            self.bounds = b
        
        def log_prior(self, q):
            return 0.0
        
        @partial(jax.jit, static_argnums=(0,))
        def gauss_func(self, q):
            
            input =jnp.array([q])
            mean = jnp.array([4., 4.])
            #return jnp.log((jnp.exp(-jnp.sum(input**2))))#+ (jnp.exp(-jnp.sum((mean-input)**2))))
            return -0.5*jnp.sum(input**2)
        
        def log_likelihood(self, q):
 
            mean = jnp.array([4., 4.])
            input = q.values
            return self.gauss_func(input)
           
        
        def log_posterior(self, q):
            return self.log_prior(q)+self.log_likelihood(q)

        def potential(self, q):
            return -self.log_posterior(q)
        
        def gradient(self, q):
            #grad1 = grad(self.gauss_func)
            x = q.values
            #return np.array(grad1(x))
            return -q.values
        


    
     
    #ray.init()
    
    dimension = 2
    names = ["{}".format(i) for i in range(dimension)]
    bounds = [[-5,5] for _ in names]
    
    n_threads  = 1
    n_samps    = 1e5
    n_train    = 1e4
    e_train    = 1
    adapt_mass = 1
    verbose    = 1
    n_bins     = int(np.sqrt(n_samps))
    
    rng       = [np.random.default_rng(1111+j) for j in range(n_threads)]

    M         = TestModel(names, bounds)
    Kernel    = NUTS
    HMC       = [HamiltonianMonteCarlo(M, Kernel, rng = rng[j], verbose = verbose) for j in range(n_threads)]
    
    samples   = [H.sample(M.new_point(rng = rng[j]),
                          n=n_samps//n_threads,
                          t_epochs=e_train,
                          t=n_train,
                          mass_estimate = adapt_mass,
                          position=j)
                 for j,H in enumerate(HMC)]
    
    x = []
    for s in samples:
        for v in s:
            x.append(v)
    
    v = np.column_stack([[xi[n] for xi in x] for n in names])
    
    import matplotlib.pyplot as plt
    from corner import corner
    corner(v,
                     labels=names,
                     quantiles=[0.05, 0.5, 0.95], truths = None,
                     show_titles=True, title_kwargs={"fontsize": 12}, smooth2d=1.0)
    
    plt.savefig("corner.pdf",bbox_inches='tight')
    
#    ray.shutdown()