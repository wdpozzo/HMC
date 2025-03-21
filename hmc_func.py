import ray
from ray.util.queue import Queue
ray.init()

import numpy as np
import jax.numpy as jnp
from scipy.stats import multivariate_normal
from tqdm import tqdm
from functools import partial
from scipy.special import logsumexp
import os
import h5py
import jax
from raynest.nest2pos import autocorrelation, acl
from jaxopt import AndersonAcceleration, FixedPointIteration

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

def find_reasonable_time_step(q0, log_probability, rng):
    """
    algorithm 4 in https://sites.stat.columbia.edu/gelman/research/published/nuts.pdf
    """
    
    step_size = 1.0
    _, inv_m0, _ = compute_mass_matrix(jax.hessian(log_probability), q0)
    p0 = jnp.dot(np.linalg.cholesky(inv_m0).T,rng.normal(size=q0.shape[0]))
    p, q, inv_m  = implicit_midpoint(p0, q0, log_probability, step_size)
    
    log_mh_ratio = (log_probability(q0)-kinetic_energy(p0, inv_m0)) - \
                   (log_probability(q)-kinetic_energy(p, inv_m))
                
    def condition(a, log_mh_ratio):
        return a*log_mh_ratio > -a*np.log(2)
    
    if condition(1.0, log_mh_ratio):
        a = 1.0
    else:
        a = -1.0

    while condition(a, log_mh_ratio):
        step_size = step_size*2**a
        print(step_size)
        p, q, inv_m  = implicit_midpoint(p0, q0, log_probability, step_size)
        print(p, q, inv_m)
        log_mh_ratio = (log_probability(q0)-kinetic_energy(p0, inv_m0)) - \
                       (log_probability(q)-kinetic_energy(p, inv_m))
        print(a,log_mh_ratio)
        
    return step_size

@jax.jit
def softabs_lambda(lambdas, alpha):
    """
    Compute the SoftAbs regularized eigenvalues.
    Args:
        lambdas: Eigenvalues of the Hessian.
        alpha: SoftAbs smoothing parameter.
    
    Returns:
        Regularized eigenvalues.
    """
    return lambdas / jnp.tanh(alpha * lambdas)

@jax.jit
def softabs_metric(H, alpha=1e-1):
    """
    Compute the SoftAbs metric tensor given a potential energy function U.
    
    Args:
        U: Potential energy function U(q).
        q: Position variable (state in phase space).
        alpha: SoftAbs regularization parameter (controls smoothness).
    
    Returns:
        SoftAbs metric g(q).
    """

    # Eigen decomposition of the Hessian
    lambdas, V = jnp.linalg.eigh(H)  # H = V D V^T, where D is diagonal of eigenvalues

    # Apply SoftAbs function to eigenvalues
    soft_lambdas = softabs_lambda(lambdas, alpha)

    # Reconstruct metric: g(q) = V Λ_soft V^T
    G = V @ jnp.diag(soft_lambdas) @ V.T

    return G
    
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

@jax.jit
def symmetrise(A):
    return (A + A.T) / 2

@partial(jax.jit, static_argnums = (0))
def compute_mass_matrix(hessian, q):
    mass_matrix = symmetrise(-hessian(q))#+1e-6*jnp.eye(q.shape[0])
    sign, logdet = jnp.linalg.slogdet(mass_matrix)
#    jax.debug.print("sign = {sign}", sign=sign)
    
    def softabs_case(_):
        return softabs_metric(mass_matrix)

    def identity_case(_):
        return mass_matrix
        
    mass_matrix = jax.lax.cond(sign < 0.0, softabs_case, identity_case, None)
    inverse_mass_matrix = jnp.linalg.inv(mass_matrix)#make_positive_definite()
    logdet = jnp.linalg.slogdet(mass_matrix)[1]
    
    return mass_matrix, inverse_mass_matrix, logdet

@partial(jax.jit, static_argnums = (2))
def hamiltonian(p, q, log_probability):
    _, inverse_metric, logdet = compute_mass_matrix(jax.hessian(log_probability), q)
    return kinetic_energy(p, inverse_metric) + 0.5*logdet - log_probability(q)

@partial(jax.jit, static_argnums = (2))
def implicit_midpoint(p0, q0, log_probability, step_size):

    nablaHq = jax.grad(hamiltonian, argnums=1)
    nablaHp = jax.grad(hamiltonian, argnums=0)
    
    def equations_of_motion(z):
        p, q = jnp.split(z, 2)
        eq1 = p0 + step_size * nablaHq(0.5*(p+p0), 0.5*(q+q0), log_probability)
        eq2 = q0 - step_size * nablaHp(0.5*(p+p0), 0.5*(q+q0), log_probability)
        return jnp.concatenate([eq1, eq2])
        
    z_initial = jnp.concatenate([p0, q0])
#    fpi = AndersonAcceleration(fixed_point_fun=equations_of_motion,
#                               history_size=5,
#                               ridge=1e-6,
#                               tol=1e-5)
    fpi = FixedPointIteration(fixed_point_fun=equations_of_motion)
                               
#    sol = fixed_point_iter(equations_of_motion, z_initial)#, maxiter=10, history_size=5)
    sol = fpi.run(z_initial).params
    p_next, q_next = jnp.split(sol, 2)
    _, inv_m, _ = compute_mass_matrix(jax.hessian(log_probability), q_next)
    return p_next, q_next, inv_m

@partial(jax.jit, static_argnums = (0))
def generalized_leap_frog(log_probability, step_size, p0, q0, inverse_mass_matrix_0):
    
    f_max = 1
    p = p0.copy()
    q = q0.copy()
    
    nablaH = jax.grad(hamiltonian)
    hessV  = jax.hessian(log_probability)
    DH = nablaH(p, q, log_probability)
    
    for f in range(f_max):
        p -= 0.5 * step_size * DH
        DH = nablaH(p, q, log_probability)

    gradH_p = jnp.dot(inverse_mass_matrix_0, p)
    gradHprime_p = gradH_p.copy()
    
    for f in range(f_max):
        q += step_size * (gradHprime_p + gradH_p)/2
        _, inverse_mass_matrix, _ = compute_mass_matrix(hessV, q)
        gradHprime_p = jnp.dot(inverse_mass_matrix,p)

    gradH_q = nablaH(p, q, log_probability)
    p -= 0.5 * step_size * gradH_q

    return p, q, inverse_mass_matrix
    
def build_tree(p, q, inverse_metric, logu, v, j, dt, log_probability, rng):
#        print("j = ",j, "logu = ",logu)
    if j == 0:
        # Base case: Take one leapfrog step in the direction of v
#            print("before leap frog",p, q)
        """
        pprime, qprime, inverse_metric = generalized_leap_frog(log_probability, dt, p, q, inverse_metric)
        """
        pprime, qprime, inverse_metric = implicit_midpoint(p, q, log_probability, dt)
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

#@ray.remote
def run_rmhmc(q0, n_steps, n_leaps, step_size, log_probability, rng, *args, **kwargs):
    
    n_train = n_steps//2
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
        
        
#        print(i,inverse_mass_matrix_0)
        counter += 1
        p0 = jnp.dot(np.linalg.cholesky(inverse_mass_matrix_0).T,rng.normal(size=q0.shape[0]))
        
        p_ = p0
        q_ = q0
        H0 = hamiltonian(p0, q0, log_probability)
        
        for k in range(n_leaps):
#            p_, q_, g_ = generalized_leap_frog(logp, step_size, p_, q_, inverse_mass_matrix_0)
#            print("pre - leap ",k,"p:",p_,"q:",q_,"invM:",compute_mass_matrix(jax.hessian(log_probability),q_)[1])
            p_, q_, g_ = implicit_midpoint(p_, q_, log_probability, step_size)
#            print("post - leap ",k,"p:",p_,"q:",q_,"invM:",g_)
        
        H     = hamiltonian(p_, q_, log_probability)
        alpha = min(0.0,H0-H)

        if alpha > np.log(rng.uniform()):
            ps[i], qs[i], gs[i] = p_, q_, g_
#            print("accepting",i, ps[i], qs[i], gs[i])
            
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
    
    qs = qs[n_train:]

    return qs

@ray.remote
def run_nuts_rmhmc(q0, n_steps, step_size, log_probability, rng, queue, *args, **kwargs):
    
    n_train = np.minimum(n_steps//10,5000)
    print("training length =", n_train)

    qs = np.zeros((2*n_steps,q0.shape[0]))
    counter = 0

    

    _, inverse_metric_0, _ = compute_mass_matrix(jax.hessian(log_probability),q0)
    print("initial point = {}".format(q0))
    print("initial metric estimate = {}".format(inverse_metric_0))
    print("determinant =", jnp.linalg.slogdet(inverse_metric_0))
    
    tuner = DualAveragingStepSize(step_size, target_accept=0.5, gamma=0.1, t0=10.0, kappa=0.5)

    p_sharp_l = jnp.zeros_like(q0)
    p_sharp_r = jnp.zeros_like(q0)
    accepted = 0
    acceptance = 0.0
    
    while accepted < n_steps:
    
        p0 = jnp.dot(jnp.linalg.cholesky(inverse_metric_0).T,rng.normal(size=q0.shape[0]))
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
                p_sharp_l += p_l
            else:
                _, _, _, p_r, q_r, inverse_metric_r, qprime, nprime, sprime = build_tree(p_r, q_r, inverse_metric_r, logu, v, j, step_size, log_probability, rng)
                p_sharp_r += p_r
                
            if sprime:
            
                alpha = min(1, nprime / n)
                
                if rng.uniform() < alpha:
                    q0 = qprime.copy()
                    queue.put(q0)
                    accepted += 1
#                    yield q0
            
            n += nprime
            s = sprime * (jnp.dot(p_sharp_l, p_l) > 0) * (jnp.dot(p_sharp_r, p_r) > 0)
            j += 1
            
        counter += 1
        acceptance = accepted / counter
#        pbar.set_postfix({"acceptance rate": f"{acceptance:.3f}"})
        
        if counter < n_train:
            step_size, _ = tuner.update(acceptance)
#            pbar.set_postfix({"step size tuning": f"{step_size:.3e}"})

        if counter == n_train:
            _, step_size = tuner.update(acceptance)
    
#    qs = qs[n_train:n_steps]
#    
#    return qs

def test_integrator(p0, q0, steps, dt, logp, inv_m):

    qs = np.zeros((steps,q0.shape[0]))
    qs_i = np.zeros((steps,q0.shape[0]))
    ps = np.zeros((steps,q0.shape[0]))
    ps_i = np.zeros((steps,q0.shape[0]))
    
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from tqdm import tqdm
    from geometric_utils import radius_of_curvature
    
    fig1 = plt.figure()
    
    ax4 = fig1.add_subplot(224)
    ax3 = fig1.add_subplot(223)
    ax2 = fig1.add_subplot(221)
    ax  = fig1.add_subplot(222)
    
    p, q = p0.copy(), q0.copy()
    p_i, q_i = p0.copy(), q0.copy()

    ax.scatter(q0[0], q0[1], color='k', marker = '+', s=128)
    ax2.scatter(p0[0], p0[1], color='k', marker = '+', s=128)
    
    colors = cm.RdBu(np.linspace(0, 1, steps))
    
    for i in tqdm(range(steps)):
        p, q, inverse_metric = generalized_leap_frog(logp, dt, p, q, inv_m)
        p_i, q_i, inverse_metric_i = implicit_midpoint(p_i, q_i, logp, dt)
        ps[i], qs[i] = p, q
        ps_i[i], qs_i[i] = p_i, q_i
#        ax.scatter(qs[i,0], qs[i,1], -logp(qs[i]), color=colors[i], marker = 's')
        ax.scatter(qs_i[i,0], qs_i[i,1], -logp(qs_i[i]), color=colors[i], marker = 'o')
#        ax2.scatter(ps[i,0], ps[i,1], -logp(qs[i]), color=colors[i], marker = 's')
        ax2.scatter(ps_i[i,0], ps_i[i,1], -logp(qs_i[i]), color=colors[i], marker = 'o')

#    ax.plot(qs[:,0], qs[:,1], color='k', lw=0.5, linestyle='dashed', label = 'GLP')
    ax.plot(qs_i[:,0], qs_i[:,1], color='k', lw=0.5, linestyle='solid', label = 'IM')
#    ax2.plot(ps[:,0], ps[:,1], color='k', lw=0.5, linestyle='dashed', label = 'GLP')
    ax2.plot(ps_i[:,0], ps_i[:,1], color='k', lw=0.5, linestyle='solid', label = 'IM')
    
    nbins = 101
    x, y = np.linspace(10,50,nbins), np.linspace(0.1,1.0,nbins)
    Z    = np.zeros((nbins,nbins))
    R    = np.zeros((nbins,nbins))
    K    = np.zeros((nbins,nbins))
    
    g    = lambda x: -jax.hessian(logp)(x)
    
    for i in tqdm(range(nbins)):
        for j in range(nbins):
            params = np.hstack((x[i],y[j]))
            Z[i,j] = -logp(params)
            _, invM, logdet = compute_mass_matrix(g,params)
            K[i,j] = jnp.exp(logdet)#radius_of_curvature(g,params)
            R[i,j] = jnp.log(radius_of_curvature(g,params))
#            print("{} {} x1 = {} x2 = {} g_inv = {} r = {}".format(i,j,x[i],y[j], compute_mass_matrix(g,params)[1],R[i,j]))

    X, Y = np.meshgrid(x, y)

    C = ax.contour(X, Y, Z.T, 256, alpha = 0.5, cmap=cm.coolwarm)
    C = ax3.pcolormesh(X, Y, R.T, alpha = 0.5, cmap=cm.coolwarm)
    fig1.colorbar(C, label = "log curvature")
    C = ax4.pcolormesh(X, Y, K.T, alpha = 0.5, cmap=cm.coolwarm)
    

    
#    ax.plot(qs_i[:,0],qs_i[:,1],'o-', color='green', label = "IM")
    fig1.legend()
    
    plt.show()

if __name__=="__main__":
    
    dim = 2
    n_processes = 6
    rng = [np.random.default_rng(seed = 11+j) for j in range(n_processes)]
    n_steps = 2000
    n_leaps = 200
    step_size = 0.03
    
    q0 = jnp.array([-3.,2.])#rng[0].uniform(-5,5,size=dim)#
    
    from scipy.stats import random_correlation
    
    eigs        = rng[0].uniform(1,100,len(q0))
    eigs        = np.array(len(q0)*eigs/np.sum(eigs))
    cov         = random_correlation.rvs(eigs, random_state=rng[0])
    inv_cov     = np.linalg.inv(cov)
    data        = rng[0].uniform(-5,5,size=dim)
    
    def log_posterior(q, data, inv_cov):
        r = (data - q)
        return -0.5*jnp.dot(r.T,jnp.dot(inv_cov,r))
    
    eigs1        = rng[0].uniform(1,100,len(q0))
    eigs1        = np.array(len(q0)*eigs/np.sum(eigs))
    cov1         = random_correlation.rvs(eigs, random_state=rng[0])
    inv_cov1     = np.linalg.inv(cov)
    data1        = jnp.array([2.0,2.0])
    
    eigs2        = rng[0].uniform(1,10,len(q0))
    eigs2        = np.array(len(q0)*eigs/np.sum(eigs))
    cov2         = random_correlation.rvs(eigs, random_state=rng[0])
    inv_cov2     = np.linalg.inv(cov2)
    data2        = jnp.array([-2.0,-2.0])

    from jax.scipy.special import logsumexp
    
    def log_posterior_mixture(q, data1, inv_cov1, data2, inv_cov2):
        w  = 0.1
        r1 = (data1 - q)
        r2 = (data2 - q)
        p1 =  -0.5*jnp.dot(r1.T,jnp.dot(inv_cov1,r1))
        p2 =  -0.5*jnp.dot(r2.T,jnp.dot(inv_cov2,r2))
        return logsumexp(jnp.array([p1,p2]), b=jnp.array([w,1-w]))

    logp = jax.jit(partial(log_posterior_mixture, data1=data1, inv_cov1=inv_cov1, data2=data2, inv_cov2=inv_cov2))
#    logp = jax.jit(partial(log_posterior, data = data, inv_cov = inv_cov))
#    _, inverse_metric_0, logdet = compute_mass_matrix(jax.hessian(logp),q0)
#    print(inverse_metric_0, np.linalg.eig(inverse_metric_0))
#    p0 = np.dot(np.linalg.cholesky(inverse_metric_0).T,rng[0].normal(size=q0.shape[0]))
#    p0 = jnp.array([0.0,0.0])
    
#    test_integrator(p0, q0, n_leaps, step_size, logp, inverse_metric_0)
#    exit()

    from tqdm import tqdm
    
    queue = Queue()
    
    chains = [run_nuts_rmhmc.remote(rng[j].uniform(-5,5,size=dim), n_steps, step_size, logp, rng[j], queue) for j in range(n_processes)]
                  #ray.get([)
    
    pbar = tqdm(total = n_steps*n_processes)
    qs = np.zeros((n_steps*n_processes, dim))
    for i in range(n_steps*n_processes):
        qs[i] = queue.get()
        pbar.update(1)

    thinning = int(max([acl(q) for q in qs.T]))
    
    if thinning < 1:
        thinning = 1

    print("ACL = {}".format(thinning))
    qs = qs[::thinning]
    
    print("indipendent samples = ",qs.shape[0])
    
    import matplotlib.pyplot as plt
    
    x = np.linspace(-5.,5.,200)
    y = np.linspace(-5.,5.,200)
    Z = np.array([logp(np.array([xi,yi])) for yi in y for xi in x]).reshape(x.shape[0],y.shape[0])

    X, Y = np.meshgrid(x,y)

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(data1[0],data1[1], 'o', color='r', zorder=100)
    ax.plot(data2[0],data2[1], 'o', color='r', zorder=101)
    
    C = ax.contour(X, Y, Z, 32)
    ax.plot(qs[:,0],qs[:,1],markersize=2,color='green',marker='o',alpha=0.5)
    fig.colorbar(C)
    fig.savefig("likelihood.pdf", bbox_inches='tight')
    
    
    from corner import corner
    corner(qs,
                        labels=None,
                        quantiles=[0.05, 0.5, 0.95], truths = data,
                        show_titles=True, title_kwargs={"fontsize": 12}, smooth2d=1.0)
    
    plt.savefig("corner.pdf",bbox_inches='tight')
    
    plt.show()
