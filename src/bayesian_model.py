import pymc as pm
import numpy as np
import pytensor.tensor as pt

def run_bayesian_mmm(X, y, ps_scores=None):
    """Robust Bayesian MMM with Elastic Net prior and stable weighting"""
    n, p = X.shape
    
    with pm.Model() as model:
        # ====== Improved Elastic Net Prior ======
        # Separate L1 and L2 components
        beta = pm.Normal("beta", mu=0, sigma=1, shape=p)
        l1_strength = pm.HalfNormal("l1_strength", 0.5)
        
        # Elastic Net penalty (added to logp)
        pm.Potential(
            "elastic_net_penalty",
            -l1_strength * pt.sum(pt.abs(beta)) - 0.5 * pt.sum(beta**2)
        )
        
        # ====== Stable Intercept and Noise ======
        alpha = pm.Normal("alpha", mu=y.mean(), sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        
        # ====== Robust Weight Handling ======
        mu = alpha + pt.dot(X, beta)
        
        if ps_scores is not None:
            # Clip and normalize weights safely
            weights = np.clip(ps_scores, 1e-5, 1-1e-5)
            weights = weights / np.mean(weights)
            
            # Stabilized likelihood
            pm.Normal(
                "obs",
                mu=mu,
                sigma=sigma/pt.sqrt(weights),
                observed=y
            )
        else:
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        
        # ====== Improved Sampling ======
        trace = pm.sample(
            draws=2000,
            tune=2000,
            chains=2,
            target_accept=0.95,
            compute_convergence_checks=True,
            return_inferencedata=True
        )
    
    return model, trace