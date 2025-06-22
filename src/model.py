import pymc as pm
import numpy as np

def build_mmm_model(data, media_cols, sales_col):
    with pm.Model() as model:
        intercept = pm.Normal('intercept', mu=0, sigma=1)

        coefs = pm.Normal('coefs', mu=0, sigma=1, shape=len(media_cols))
        sigma = pm.HalfNormal('sigma', sigma=1)

        mu = intercept + pm.math.dot(data[media_cols], coefs)

        sales_obs = pm.Normal(sales_col, mu=mu, sigma=sigma, observed=data[sales_col])

    return model
