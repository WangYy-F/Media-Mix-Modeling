import pymc as pm
import numpy as np

def bayesian_ab_test(control, variant):
    """Bayesian A/B test with effect size estimation"""
    with pm.Model() as model:
        # Priors
        mu_control = pm.Normal("mu_control", mu=np.mean(control), sigma=10)
        mu_variant = pm.Normal("mu_variant", mu=np.mean(variant), sigma=10)
        sigma = pm.HalfNormal("sigma", 10)
        
        # Likelihoods
        pm.Normal("control_obs", mu_control, sigma, observed=control)
        pm.Normal("variant_obs", mu_variant, sigma, observed=variant)
        
        # Uplift calculations
        delta = pm.Deterministic("delta", mu_variant - mu_control)
        pm.Deterministic("uplift", (mu_variant - mu_control) / mu_control)
        
        # Sampling
        trace = pm.sample(2000, tune=1000, chains=2, return_inferencedata=True)
        
    return model, trace