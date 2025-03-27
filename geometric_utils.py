import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums = (0))
def christoffel_symbols(metric, coords):
    """
    Compute Christoffel symbols of the second kind from the metric tensor.
    
    Args:
        metric: Function returning the metric tensor g_{μν}.
        coords: Coordinate variables [x1, x2, ...].
    
    Returns:
        Γ^λ_{μν} (3D tensor).
    """
    g = metric(coords)  # Metric tensor g_{μν}
    g_inv = jnp.linalg.inv(g)  # Inverse metric g^{μν}
    
    def metric_deriv(i, j, k):
        """ Compute ∂_k g_{ij}. """
        return jax.grad(lambda x: metric(x)[i, j])(coords)[k]

    dim = len(coords)
    Gamma = jnp.zeros((dim, dim, dim))

    for l in range(dim):
        for m in range(dim):
            for n in range(dim):
                term1 = metric_deriv(m, n, l)
                term2 = metric_deriv(m, l, n)
                term3 = metric_deriv(n, l, m)
                Gamma = Gamma.at[l, m, n].set(
                    0.5 * jnp.sum(g_inv[l, :] * (term1 + term2 - term3))
                )

    return Gamma

@partial(jax.jit, static_argnums = (0))
def riemann_tensor(metric, coords):
    """
    Compute the Riemann curvature tensor R^ρ_{σμν}.
    
    Args:
        metric: Function returning the metric tensor g_{μν}.
        coords: Coordinate variables [x1, x2, ...].
    
    Returns:
        R^ρ_{σμν} (4D tensor).
    """
    Gamma = christoffel_symbols(metric, coords)
    dim = len(coords)
    R = jnp.zeros((dim, dim, dim, dim))

    def gamma_deriv(r, s, mu, nu):
        return jax.grad(lambda x: christoffel_symbols(metric, x)[r, s, mu])(coords)[nu]

    for rho in range(dim):
        for sigma in range(dim):
            for mu in range(dim):
                for nu in range(dim):
                    term1 = gamma_deriv(rho, sigma, mu, nu)
                    term2 = gamma_deriv(rho, sigma, nu, mu)
                    term3 = jnp.sum(Gamma[rho, :, mu] * Gamma[:, sigma, nu])
                    term4 = jnp.sum(Gamma[rho, :, nu] * Gamma[:, sigma, mu])
                    R = R.at[rho, sigma, mu, nu].set(term1 - term2 + term3 - term4)

    return R

@partial(jax.jit, static_argnums = (0))
def ricci_scalar(metric, coords):
    """
    Compute the Ricci scalar R from the metric tensor.
    
    Args:
        metric: Function returning the metric tensor g_{μν}.
        coords: Coordinate variables [x1, x2, ...].
    
    Returns:
        Ricci scalar R.
    """
    R = riemann_tensor(metric, coords)
    g = metric(coords)
    g_inv = jnp.linalg.inv(g)
    
    Ricci = jnp.einsum("rurm->um", R)  # R_μν = R^ρ_{μρν}
    Ricci_scalar = jnp.sum(g_inv * Ricci)  # R = g^μν R_μν

    return Ricci_scalar

@partial(jax.jit, static_argnums = (0))
def radius_of_curvature(metric, coords):
    """
    Compute the radius of curvature from the Ricci scalar.
    
    Args:
        metric: Function returning the metric tensor g_{μν}.
        coords: Coordinate variables [x1, x2, ...].
    
    Returns:
        Radius of curvature ρ.
    """
    R = ricci_scalar(metric, coords)
    return 1.0 / jnp.sqrt(jnp.abs(R))

if __name__=="__main__":
    
    def potential(q):
        return jnp.sum(q**2)
    
    nablaV = jax.grad(potential)
    hess   = jax.hessian(potential)
    
    q      = jnp.array([2.0,2.0])
    print(radius_of_curvature(hess,q))
    
