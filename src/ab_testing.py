import pymc as pm
import pandas as pd

def compute_propensity_score(df, treatment_col, features):
    import pymc as pm
    import aesara.tensor as at

    with pm.Model() as model:
        beta = pm.Normal('beta', mu=0, sigma=1, shape=len(features))
        logits = at.dot(df[features], beta)
        treatment = pm.Bernoulli('treatment', logit_p=logits, observed=df[treatment_col])

        trace = pm.sample(1000, tune=1000, cores=2)

    return trace