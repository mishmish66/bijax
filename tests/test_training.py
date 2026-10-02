"""End-to-end: the example flow should learn the two-moons density.

Maximum-likelihood training should drive the mean negative log-likelihood well
below the standard-Gaussian baseline (~2.84 nats on the standardized data).
"""

from jax import numpy as jnp
from jax import random as jr
from made_spline_flow import make_flow, sample, train, two_moons

GAUSSIAN_NLL = 2.84  # mean NLL of a unit Gaussian on the standardized data
TARGET_NLL = 2.3  # comfortably better than the Gaussian baseline


def test_example_flow_learns_two_moons():
    data = two_moons(jr.key(42), 2000)
    _, init_nll, final_nll = train(
        make_flow(jr.key(0)), data, steps=600, lr=5e-3, batch=256
    )
    assert final_nll < init_nll - 0.5, "flow did not learn (no improvement)"
    assert final_nll < TARGET_NLL, f"flow final NLL {final_nll:.3f} too high"
    assert final_nll < GAUSSIAN_NLL, "flow no better than a Gaussian"


def test_example_samples_are_finite_points_in_the_data_space():
    xs = sample(make_flow(jr.key(0)), jr.key(1), 64)
    assert xs.shape == (64, 2)
    assert jnp.all(jnp.isfinite(xs))
