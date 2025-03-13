import numpy as np
import jax.numpy as jnp
#import numpy
import ray
#from raynest.proposal import Proposal
from scipy.stats import multivariate_normal
from tqdm import tqdm
from functools import partial
from collections import deque
from scipy.special import logsumexp
import os
import sys
import h5py
import jax
import ray
from raynest.nest2pos import autocorrelation, acl

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

@jax.jit
def make_positive_definite(A):
    A = (A + A.T) / 2  # Ensure symmetry
    eigenvalues_, eigenvectors = jnp.linalg.eigh(A)
    
    # Replace non-positive eigenvalues with a small positive number
    eigenvalues = jnp.abs(eigenvalues_)
    
    # Reconstruct the matrix
    A_positive = eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T
    
    return A_positive

@jax.jit
def kinetic_energy(p, inverse_mass_matrix):
    return 0.5*jnp.dot(p.T,jnp.dot(inverse_mass_matrix,p))

@partial(jax.jit, static_argnums = (0))
def compute_mass_matrix(hessian, q):
    mass_matrix = -hessian(q)
    inverse_mass_matrix = jnp.linalg.inv(make_positive_definite(mass_matrix))
    det = jnp.linalg.det(mass_matrix)
    return mass_matrix, inverse_mass_matrix, det

@partial(jax.jit, static_argnums = (3))
def hamiltonian(p, q, inverse_mass_matrix, log_probability):
    return kinetic_energy(p, inverse_mass_matrix) - log_probability(q)

@partial(jax.jit, static_argnums = (0))
def generalized_leap_frog(log_probability, step_size, p0, q0, inverse_mass_matrix_0):
    
    f_max = 1
    p = p0.copy()
    q = q0.copy()
    
    nablaH = jax.grad(hamiltonian)
    hessV  = jax.hessian(log_probability)
    DH = nablaH(p, q, inverse_mass_matrix_0, log_probability)
    
    for f in range(f_max):
        p -= 0.5 * step_size * DH
        DH = nablaH(p, q, inverse_mass_matrix_0, log_probability)

    gradH_p = jnp.dot(inverse_mass_matrix_0, p)
    gradHprime_p = gradH_p.copy()
    
    for f in range(f_max):
        q += step_size * (gradHprime_p + gradH_p)/2
        _, inverse_mass_matrix, _ = compute_mass_matrix(hessV, q)
        gradHprime_p = jnp.dot(inverse_mass_matrix,p)

    gradH_q = nablaH(p, q, inverse_mass_matrix, log_probability)
    p -= 0.5 * step_size * gradH_q

    return p, q, inverse_mass_matrix
    
def build_tree(p, q, inverse_metric, logu, v, j, dt, log_probability, rng):
#        print("j = ",j, "logu = ",logu)
    if j == 0:
        # Base case: Take one leapfrog step in the direction of v
#            print("before leap frog",p, q)
        pprime, qprime, inverse_metric = generalized_leap_frog(log_probability, dt, p, q, inverse_metric)
#            print("after leap frog",pprime, qprime)
        logH = log_probability(qprime) - kinetic_energy(pprime, inverse_metric)
#            print("base level ",pprime, qprime, logH, logu, logu <= logH, logH > logu - 1000)
        nprime = int(logu <= logH)
        sprime = int(logH > logu - 1000)
#            print("leaf in the tree =",pprime, qprime, pprime, qprime, qprime, nprime, sprime)
        return pprime, qprime, inverse_metric, pprime, qprime, inverse_metric, qprime, nprime, sprime
    
    else:
        # Recursion: Build the left and right subtrees
        pprime_l, qprime_l, inverse_metric_l, pprime_r, qprime_r, inverse_metric_r, qprime, nprime, sprime = build_tree(p, q, inverse_metric, logu, v, j-1, dt, log_probability, rng)

        if sprime:
#                print("recursing j =",j, sprime)
            if v == -1:
                pprime_l, qprime_l, inverse_metric_l, _, _, _, qpprime, npprime, spprime = build_tree(pprime_l, qprime_l, inverse_metric_l, logu, v, j-1, dt, log_probability, rng)
            else:
                _, _, _, pprime_r, qprime_r, inverse_metric_r, qpprime, npprime, spprime = build_tree(pprime_r, qprime_r, inverse_metric_r, logu, v, j-1, dt, log_probability, rng)
            
#                print("nprime = {} npprime = {}".format(nprime,npprime))
            if rng.uniform() < npprime/max(nprime+npprime,1):
                qprime = qpprime
            
            delta_q = qprime_r-qprime_l
            cond1  = np.dot(delta_q.T,pprime_l)>=0
            cond2  = np.dot(delta_q.T,pprime_r)>=0
            sprime = spprime*(np.logical_and(cond1,cond2))
            nprime = nprime + npprime
        return pprime_l, qprime_l, inverse_metric_l, pprime_r, qprime_r, inverse_metric_r, qprime, nprime, sprime

def run_rmhmc(q0, n_steps, n_leaps, step_size, log_probability, *args, **kwargs):
    
    n_train = n_steps//10
    ps = np.zeros((n_steps,q0.shape[0]))
    qs = np.zeros_like(ps)
    gs = np.zeros((n_steps,q0.shape[0],q0.shape[0]))
    counter = 0

    from tqdm import tqdm

    _, inverse_mass_matrix_0, _ = compute_mass_matrix(jax.hessian(log_probability),q0)
    print("initial metric estimate = {}".format(inverse_mass_matrix_0))
    pbar = tqdm(total = n_steps)
    
#    tuner = DualAveragingStepSize(step_size, target_accept=0.9, gamma=0.05, t0=10.0, kappa=0.75)
    
    i = 0
    
    while i < n_steps:
    
        counter += 1
        p0 = jnp.dot(np.linalg.cholesky(inverse_mass_matrix_0).T,rng.normal(size=q0.shape[0]))
        
        p_ = p0
        q_ = q0
        H0 = hamiltonian(p0, q0, inverse_mass_matrix_0, logp)
        
        for _ in range(n_leaps):
            p_, q_, g_ = generalized_leap_frog(logp, step_size, p_, q_, inverse_mass_matrix_0)
        
        H     = hamiltonian(p_, q_, inverse_mass_matrix_0, logp)
        alpha = min(0.0,H0-H)

        if alpha > np.log(rng.uniform()):
            ps[i], qs[i], gs[i] = p_, q_, g_
            p0, q0, inverse_mass_matrix_0 =  p_, q_, g_
            i += 1
            pbar.update(1)
            
        acceptance = i/counter
        pbar.set_postfix({"acceptance":acceptance})

#        if counter < n_train:
#            step_size, _ = tuner.update(acceptance)
#            pbar.set_postfix({"step size tuning": f"{step_size:.3e}"})
#        
#        if counter == n_train:
#            _, step_size = tuner.update(acceptance)
    
    qs = qs[int(len(qs)/2):]
    thinning = int(max([acl(q) for q in qs.T]))
    print("ACL = {}".format(thinning))
    qs = qs[::thinning]
    
    x = np.linspace(-5,5,101)
    y = np.linspace(-5,5,101)
    Z = np.array([log_probability(np.array([xi,yi])) for yi in y for xi in x]).reshape(x.shape[0],y.shape[0])

    X, Y = np.meshgrid(x,y)
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(qs[:,0],qs[:,1],'o-',alpha=0.5,lw=0.3)
    ax.axvline(data[0])
    ax.axhline(data[1])
    C = ax.contour(X, Y, Z, 10)
    fig.colorbar(C)
    plt.show()

    return qs

def run_nuts_rmhmc(q0, n_steps, step_size, log_probability, *args, **kwargs):
    
    n_train = n_steps//5
    qs = np.zeros((n_steps+1,q0.shape[0]))
    counter = 0

    from tqdm import tqdm

    _, inverse_metric_0, _ = compute_mass_matrix(jax.hessian(log_probability),q0)
    print("initial metric estimate = {}".format(inverse_metric_0))
    pbar = tqdm(total = n_steps)
    tuner = DualAveragingStepSize(step_size, target_accept=0.654, gamma=0.05, t0=10.0, kappa=0.75)

    accepted = 0
    
    while accepted < n_steps:
    
        p0 = np.dot(np.linalg.cholesky(inverse_metric_0).T,rng.normal(size=q0.shape[0]))
        logP = log_probability(q0) - kinetic_energy(p0, inverse_metric_0)
        logu = logP - rng.exponential()

        q_l, q_r = q0.copy(), q0.copy()
        p_l, p_r = p0.copy(), p0.copy()
        inverse_metric_l, inverse_metric_r = inverse_metric_0.copy(), inverse_metric_0.copy()
        j, s, n = 0, 1, 1

        while s == 1:
            v = rng.choice((-1, 1))

            if v == -1:
                p_l, q_l, inverse_metric_l, _, _, _, qprime, nprime, sprime = build_tree(p_l, q_l, inverse_metric_l, logu, v, j, step_size, log_probability, rng)
            else:
                _, _, _, p_r, q_r, inverse_metric_r, qprime, nprime, sprime = build_tree(p_r, q_r, inverse_metric_r, logu, v, j, step_size, log_probability, rng)

            if sprime:
            
                alpha = min(1, nprime / n)
                
                if rng.uniform() < alpha:
                    q0[:] = qprime  # Avoid extra copying
                    qs[accepted] = q0
                    pbar.update(1)
                    accepted += 1

            n += nprime
            delta_q = q_r - q_l
            s = sprime * (np.dot(delta_q, p_l) > 0) * (np.dot(delta_q, p_r) > 0)
            j += 1
        
        counter += 1
        acceptance = accepted / counter
        pbar.set_postfix({"acceptance rate": f"{acceptance:.3f}"})
        if counter < n_train:
            step_size, _ = tuner.update(acceptance)
            pbar.set_postfix({"step size tuning": f"{step_size:.3e}"})

        if counter == n_train:
            _, step_size = tuner.update(acceptance)
    
    qs = qs[int(len(qs)/2):]
    thinning = int(max([acl(q) for q in qs.T]))
    print("ACL = {}".format(thinning))
    qs = qs[::thinning]
    
    x = np.linspace(-5,5,101)
    y = np.linspace(-5,5,101)
    Z = np.array([log_probability(np.array([xi,yi])) for yi in y for xi in x]).reshape(x.shape[0],y.shape[0])

    X, Y = np.meshgrid(x,y)
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(qs[:,0],qs[:,1],'o-',alpha=0.5,lw=0.3)
    ax.axvline(data[0])
    ax.axhline(data[1])
    C = ax.contour(X, Y, Z, 10)
    fig.colorbar(C)
    plt.show()
    
    return qs


if __name__=="__main__":
    
    rng = np.random.default_rng(seed = 222)
    n_steps = 20000
    n_leaps = 20
    q0 = rng.uniform(-5,5,size=2)
    
    from scipy.stats import random_correlation
    
    eigs        = rng.uniform(1,50,len(q0))
    eigs        = np.array(len(q0)*eigs/np.sum(eigs))
    cov         = random_correlation.rvs(eigs, random_state=rng)
    inv_cov     = np.linalg.inv(cov)

    
    step_size = 1.
    data  = rng.uniform(-5,5,size=2)
    
    def log_posterior(q, data, inv_cov):
        r = (data - q)
        return -0.5*jnp.dot(r.T,jnp.dot(inv_cov,r))
    
    
    logp = jax.jit(partial(log_posterior, data=data, inv_cov=inv_cov))
    
    run_nuts_rmhmc(q0, n_steps, step_size, logp)
