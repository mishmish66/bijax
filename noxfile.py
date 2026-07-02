"""Test sessions for bijax.

Run everything with `nox`, or a single session with e.g.
`nox -s "tests-3.12(jax='latest')"`. Uses uv as the install backend for speed.

The matrix crosses supported Python versions with two jax pins: the declared
floor (`jax>=0.8.0`) and the latest release. Testing the floor guards against
using APIs newer than we claim to support (e.g. `jax.enable_x64`).
"""

import nox

nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True

PYTHON_VERSIONS = ["3.11", "3.12", "3.13"]
JAX_FLOOR = "0.8.0"

TEST_DEPS = ["pytest", "pytest-cov", "pytest-xdist", "optax"]


def _run_tests(session: nox.Session, jax_pin: str) -> None:
    session.install(".", jax_pin, *TEST_DEPS)
    session.run("pytest", "-n", "auto", *session.posargs)


# Latest jax across every supported Python.
@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("jax", ["latest"])
def tests(session: nox.Session, jax: str) -> None:
    _run_tests(session, "jax")


# Declared floor, exercised on the oldest supported Python only.
@nox.session(python=PYTHON_VERSIONS[0])
@nox.parametrize("jax", ["floor"])
def tests_min(session: nox.Session, jax: str) -> None:
    _run_tests(session, f"jax=={JAX_FLOOR}")
