from collections.abc import Callable

from jaxtyping import Array, Float, Key


def condition(
    conditioner: Callable[..., Float[Array, "n p"]],
    x: Float[Array, " d"],
    c: Float[Array, " c"] | None,
    rng: Key[Array, ""] | None,
    shape: tuple[int, int],
) -> Float[Array, "n p"]:
    """Call ``conditioner(x, c, rng=rng)`` and check the output shape."""
    params = conditioner(x, c, rng=rng)
    if params.shape != shape:
        msg = f"conditioner output {params.shape} but needed {shape}"
        raise ValueError(msg)
    return params
