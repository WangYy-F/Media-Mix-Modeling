import pymc as pm
import numpy as np

def calculate_propensity_scores(df, treatment_var, confounders):
    """Bayesian logistic regression for propensity scores"""
    X = df[confounders].values
    treatment = df[treatment_var].values
    
    with pm.Model() as model:
        # Priors
        beta = pm.Normal("beta", 0, 1, shape=X.shape[1])
        alpha = pm.Normal("alpha", 0, 1)
        
        # Linear model
        logit_p = alpha + pm.math.dot(X, beta)
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        
        # Likelihood
        pm.Bernoulli("obs", p, observed=treatment)
        
        # Inference
        trace = pm.sample(1000, tune=1000, chains=2, return_inferencedata=True)
        
    return trace.posterior["p"].mean(("chain", "draw")).values