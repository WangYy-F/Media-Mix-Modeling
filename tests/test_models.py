import pytest
import numpy as np
import pandas as pd
from src import data_simulation, bayesian_model, propensity_score, ab_testing

def test_data_generation():
    df = data_simulation.generate_data(n=100)
    assert df.shape == (100, 6)
    assert 'sales' in df.columns
    assert df['sales'].min() > 0

def test_bayesian_model():
    X = np.random.randn(100, 3)
    y = np.random.randn(100)
    model, trace = bayesian_model.run_bayesian_mmm(X, y)
    assert 'beta' in trace.posterior
    assert len(trace.posterior.chain) == 2
    assert len(trace.posterior.draw) == 2000

def test_propensity_scores():
    df = pd.DataFrame({
        'treatment': np.random.binomial(1, 0.5, 100),
        'confounder1': np.random.randn(100),
        'confounder2': np.random.randn(100)
    })
    ps = propensity_score.calculate_propensity_scores(
        df, 'treatment', ['confounder1', 'confounder2']
    )
    assert ps.shape == (100,)
    assert (ps >= 0).all() and (ps <= 1).all()

def test_ab_testing():
    control = np.random.normal(100, 10, 1000)
    treatment = np.random.normal(110, 10, 1000)
    model, trace = ab_testing.bayesian_ab_test(control, treatment)
    uplift = trace.posterior['uplift'].mean().item()
    assert uplift > 5  # Should detect positive uplift

def test_model_with_weights():
    X = np.random.randn(100, 2)
    y = np.random.randn(100)
    weights = np.random.uniform(0.5, 1.5, 100)
    model, trace = bayesian_model.run_bayesian_mmm(X, y, weights)
    assert 'sigma' in trace.posterior