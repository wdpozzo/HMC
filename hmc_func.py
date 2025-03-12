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
import h5py
import jax
import ray
from raynest.nest2pos import autocorrelation, acl

#@jax.jit
def make_positive_definite(A):
    A = (A + A.T) / 2  # Ensure symmetry
    eigenvalues_, eigenvectors = jnp.linalg.eigh(A)
    
    # Replace non-positive eigenvalues with a small positive number
    eigenvalues = jnp.abs(eigenvalues_)
    
    # Reconstruct the matrix
    A_positive = eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T
    
    return A_positive

#@jax.jit
def kinetic_energy(p, inverse_mass_matrix):
    return 0.5*jnp.dot(p.T,jnp.dot(inverse_mass_matrix,p))

#@partial(jax.jit, static_argnums = (0))
def compute_mass_matrix(hessian, q):
    mass_matrix = -hessian(q)
    inverse_mass_matrix = make_positive_definite(jnp.linalg.inv(mass_matrix))
    det = jnp.linalg.det(mass_matrix)
    return mass_matrix, inverse_mass_matrix, det

@partial(jax.jit, static_argnums = (3))
def hamiltonian(p, q, inverse_mass_matrix, log_probability):
    return kinetic_energy(p, inverse_mass_matrix) - log_probability(q)

@partial(jax.jit, static_argnums = (0,1))
def generalized_leap_frog(log_probability, step_size, p0, q0):
    
    f_max = 3
    p = p0.copy()
    q = q0.copy()
    
    nablaH = jax.grad(hamiltonian)
    hessV  = jax.hessian(log_probability)

    _, inverse_mass_matrix, _ = compute_mass_matrix(hessV, q)

    for f in range(f_max):
        DH = -nablaH(p, q, inverse_mass_matrix, log_probability)
        p -= 0.5 * step_size * DH

    _, inverse_mass_matrix, _ = compute_mass_matrix(hessV, q)
    gradH_p = jnp.dot(inverse_mass_matrix, p)
    gradHprime_p = gradH_p.copy()
    
    for f in range(f_max):
        q += step_size * (gradHprime_p + gradH_p)/2
        _, inverse_mass_matrix, _ = compute_mass_matrix(hessV, q)
        gradH_q = -nablaH(p, q, inverse_mass_matrix, log_probability)
        gradHprime_p = jnp.dot(inverse_mass_matrix,p)

    gradH_q = -nablaH(p, q, inverse_mass_matrix, log_probability)
    p -= 0.5 * step_size * gradH_q

    return p, q, inverse_mass_matrix
    
def build_tree(p, q, logu, v, j, dt, log_probability, rng):
#        print("j = ",j, "logu = ",logu)
    if j == 0:
        # Base case: Take one leapfrog step in the direction of v
#            print("before leap frog",p, q)
        pprime, qprime, inverse_metric = generalized_leap_frog(v*dt, p, q)
#            print("after leap frog",pprime, qprime)
        logH = log_posterior(qprime) - kinetic_energy(pprime, inverse_metric)
#            print("base level ",pprime, qprime, logH, logu, logu <= logH, logH > logu - 1000)
        nprime = int(logu <= logH)
        sprime = int(logH > logu - 1000)
#            print("leaf in the tree =",pprime, qprime, pprime, qprime, qprime, nprime, sprime)
        return pprime, qprime, pprime, qprime, qprime, nprime, sprime
    
    else:
        # Recursion: Build the left and right subtrees
        pprime_l, qprime_l, pprime_r, qprime_r, qprime, nprime, sprime = build_tree(p, q, logu, v, j-1, dt, log_probability)

        if sprime:
#                print("recursing j =",j, sprime)
            if v == -1:
                pprime_l, qprime_l, _, _, qpprime, npprime, spprime = build_tree(pprime_l, qprime_l, logu, v, j-1, dt, log_probability)
            else:
                _, _, pprime_r, qprime_r, qpprime, npprime, spprime = build_tree(pprime_r, qprime_r, logu, v, j-1, dt, log_probability)
            
#                print("nprime = {} npprime = {}".format(nprime,npprime))
            if rng.uniform() < npprime/max(nprime+npprime,1):
                qprime = qpprime
            
            delta_q = qprime_r-qprime_l
            cond1  = np.dot(delta_q.T,np.dot(inverse_mass_matrix,pprime_l))>=0
            cond2  = np.dot(delta_q.T,np.dot(inverse_mass_matrix,pprime_r))>=0
            sprime = spprime*(np.logical_and(cond1,cond2))
            nprime = nprime + npprime
        return pprime_l, qprime_l, pprime_r, qprime_r, qprime, nprime, sprime

if __name__=="__main__":
    
    rng = np.random.default_rng(seed = 222)
    n_steps = 100000
    n_leaps = 10
    q0 = rng.uniform(-5,5,size=2)
    
    from scipy.stats import random_correlation
    
    eigs        = rng.uniform(1,50,len(q0))
    eigs        = np.array(len(q0)*eigs/np.sum(eigs))
    cov         = random_correlation.rvs(eigs, random_state=rng)
    inv_cov     = np.linalg.inv(cov)

    
    step_size = .1
    data  = rng.uniform(-5,5,size=2)
    
    def log_posterior(q, data, inv_cov):
        r = (data - q)
        return -0.5*jnp.dot(r.T,jnp.dot(inv_cov,r))
    
    
    logp = partial(log_posterior, data=data, inv_cov=inv_cov)
    
    ps = np.zeros((n_steps,q0.shape[0]))
    qs = np.zeros_like(ps)
    gs = np.zeros((n_steps,q0.shape[0],q0.shape[0]))

    from tqdm import tqdm

    _, inverse_mass_matrix_0, _ = compute_mass_matrix(jax.hessian(logp),q0)

    pbar = tqdm(total = n_steps)
    
    i = 0
    
    while i < n_steps:
    
        p0 = np.dot(np.linalg.cholesky(inverse_mass_matrix_0).T,rng.normal(size=q0.shape[0]))
        
        for _ in range(n_leaps):
            p_, q_, g_ = generalized_leap_frog(logp, step_size, p0, q0)
            
        alpha = min(0.0,hamiltonian(p0, q0, inverse_mass_matrix_0, logp)-hamiltonian(p_, q_, g_, logp))
#        print(alpha, hamiltonian(p0, q0, inverse_mass_matrix_0, logp)-hamiltonian(p_, q_, g_, logp))
        if alpha > np.log(rng.uniform()):
            ps[i], qs[i], gs[i] = p_, q_, g_
            p0, q0, inverse_mass_matrix_0 = ps[i], qs[i], gs[i]
            i += 1
            pbar.update(1)
    
    
    qs = qs[int(len(qs)/2):]
    thinning = int(max([acl(q) for q in qs.T]))
    print("ACL = {}".format(thinning))
    qs = qs[::thinning]
    x = np.linspace(-5,5,101)
    y = np.linspace(-5,5,101)
    Z = np.array([logp(np.array([xi,yi])) for yi in y for xi in x]).reshape(x.shape[0],y.shape[0])

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
