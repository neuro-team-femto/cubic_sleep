import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.tensor import TensorVariable
from pytensor.tensor.random.op import RandomVariable
from pymc.distributions.distribution import Distribution, SymbolicRandomVariable
from pymc.logprob.abstract import _logprob
import warnings
from pymc.distributions.continuous import Normal
from collections.abc import Callable
from pymc.distributions.shape_utils import (
    _change_dist_size,
    change_dist_size,
    get_support_shape,
    get_support_shape_1d,
    rv_size_is_none,
)
from pytensor.graph.basic import Node, ancestors
from pymc.util import check_dist_not_registered
from pymc.logprob.basic import logp

# 1) Create a new SymbolicRandomVariable subclass that behaves like PyMC’s EulerMaruyamaRV
#    but includes tspan in its inputs and uses a logp that is time-dependent.


class tv_EulerMaruyamaRV(SymbolicRandomVariable):
    """A placeholder used to specify a log-likelihood for a EulerMaruyama sub-graph."""

    dt: float
    sde_fn: Callable
    tspan: pt.TensorVariable
    _print_name = ("EulerMaruyama", "\\operatorname{EulerMaruyama}")

    def __init__(self, *args, dt: float, tspan,sde_fn: Callable, **kwargs):
        self.dt = dt
        self.sde_fn = sde_fn
        self.tspan = tspan
        super().__init__(*args, **kwargs)

    @classmethod
    def rv_op(cls, init_dist, steps, sde_pars, dt, sde_fn,tspan,size=None):
        # We don't allow passing `rng` because we don't fully control the rng of the components!
        # if noise_rng is None:
        noise_rng = pytensor.shared(np.random.default_rng())

        # Init dist should have shape (*size,)
        if size is not None:
            batch_size = size
        else:
            batch_size = pt.broadcast_shape(*sde_pars, init_dist)
        init_dist = change_dist_size(init_dist, batch_size)

        # Create OpFromGraph representing random draws from SDE process
        def step(*prev_args):
            # Each time I arrive here, I am in a new time-step, so be
            t,prev_y, *prev_sde_pars, rng = prev_args
            f, g = sde_fn(prev_y,t,*prev_sde_pars)
            mu = prev_y + dt * f
            sigma = pt.sqrt(dt) * g
            next_rng, next_y = Normal.dist(mu=mu, sigma=sigma, rng=rng).owner.outputs
            return next_y, {rng: next_rng}

        y_t, innov_updates = pytensor.scan(
            fn=step,
            outputs_info=[init_dist],
            non_sequences=[*sde_pars, noise_rng],
            n_steps=steps,
            sequences = tspan,
            strict=True,
        )
        (noise_next_rng,) = tuple(innov_updates.values())

        sde_out = pt.concatenate([init_dist[None, ...], y_t], axis=0).dimshuffle(
            (*range(1, y_t.ndim), 0)
        )

        return tv_EulerMaruyamaRV(
            inputs=[init_dist, steps, *sde_pars, noise_rng],
            outputs=[noise_next_rng, sde_out],
            dt=dt,
            tspan = tspan,
            sde_fn=sde_fn,
            extended_signature=f"(),(s),{','.join('()' for _ in sde_pars)},[rng]->[rng],(t)",
        )(init_dist, steps, *sde_pars, noise_rng)

    def update(self, node: Node):
        """Return the update mapping for the noise RV."""
        return {node.inputs[-1]: node.outputs[0]}

class tv_EulerMaruyama(Distribution):
    r"""
    Stochastic differential equation discretized with the Euler-Maruyama method.

    Parameters
    ----------
    dt : float
        time step of discretization
    sde_fn : callable
        function returning the drift and diffusion coefficients of SDE
    sde_pars : tuple
        parameters of the SDE, passed as ``*args`` to ``sde_fn``
    init_dist : unnamed_distribution, optional
        Scalar distribution for initial values. Distributions should have shape (*shape[:-1]).
        If not, it will be automatically resized. Defaults to pm.Normal.dist(0, 100, shape=...).

        .. warning:: init_dist will be cloned, rendering it independent of the one passed as input.
    """

    rv_type = tv_EulerMaruyamaRV
    rv_op = tv_EulerMaruyamaRV.rv_op

    def __new__(cls, name, dt, sde_fn, *args, steps=None, **kwargs):
        dt = pt.as_tensor_variable(dt)
        steps = get_support_shape_1d(
            support_shape=steps,
            shape=None,  # Shape will be checked in `cls.dist`
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            # tspan = kwargs.get("tspan"),
            support_shape_offset=1,
        )
        #
        return super().__new__(cls, name, dt, sde_fn, *args, steps=steps, **kwargs)

    @classmethod
    def dist(cls, dt, sde_fn, sde_pars,tspan,*, init_dist=None, steps=None,**kwargs):
        steps = get_support_shape_1d(
            support_shape=steps, shape=kwargs.get("shape", None), support_shape_offset=1
        )
        if steps is None:
            raise ValueError("Must specify steps or shape parameter")
        steps = pt.as_tensor_variable(steps, dtype=int, ndim=0)

        dt = pt.as_tensor_variable(dt)
        sde_pars = [pt.as_tensor_variable(x) for x in sde_pars]

        if init_dist is not None:
            if not isinstance(init_dist, TensorVariable) or not isinstance(
                init_dist.owner.op, RandomVariable | SymbolicRandomVariable
            ):
                raise ValueError(
                    f"Init dist must be a distribution created via the `.dist()` API, "
                    f"got {type(init_dist)}"
                )
            check_dist_not_registered(init_dist)
            if init_dist.owner.op.ndim_supp > 0:
                raise ValueError(
                    "Init distribution must have a scalar support dimension, ",
                    f"got ndim_supp={init_dist.owner.op.ndim_supp}.",
                )
        else:
            warnings.warn(
                "Initial distribution not specified, defaulting to "
                "`Normal.dist(0, 100, shape=...)`. You can specify an init_dist "
                "manually to suppress this warning.",
                UserWarning,
            )
            init_dist = Normal.dist(1, 0.001, shape=sde_pars[0].shape)
            # tspan = kwargs.get("tspan")
        return super().dist([init_dist, steps, sde_pars, dt, sde_fn,tspan], **kwargs)

@_change_dist_size.register(tv_EulerMaruyamaRV)
def change_eulermaruyama_size(op, dist, new_size, expand=False):
    if expand:
        old_size = dist.shape[:-1]
        new_size = tuple(new_size) + tuple(old_size)

    init_dist, steps, *sde_pars, _ = dist.owner.inputs
    return tv_EulerMaruyama.rv_op(
        init_dist,
        steps,
        sde_pars,
        dt=op.dt,
        sde_fn=op.sde_fn,
        size=new_size,
    )

@_logprob.register(tv_EulerMaruyamaRV)
def eulermaruyama_logp(op, values, init_dist, steps, *sde_pars_noise_arg, **kwargs):
    (x,) = values
    tspan = op.tspan
    # noise arg is unused, but is needed to make the logp signature match the rv_op signature
    *sde_pars, _ = sde_pars_noise_arg
    # sde_fn is user provided and likely not broadcastable to additional time dimension,
    # since the input x is now [..., t], we need to broadcast each input to [..., None]
    # below as best effort attempt to make it work
    sde_pars_broadcast = [x[..., None] for x in sde_pars]
    xtm1 = x[..., :-1]
    xt = x[..., 1:]
    tspan_tm1 = tspan[:-1]
    f, g = op.sde_fn(xtm1,tspan_tm1,*sde_pars_broadcast)
    # ds_ratio = pt.as_tensor_variable(20)
    mu = xtm1 + op.dt*f
    sigma = pt.sqrt(op.dt) * g
    # Compute and collapse logp across time dimension
    sde_logp = pt.sum(logp(Normal.dist(mu, sigma), xt), axis=-1)
    init_logp = logp(init_dist, x[..., 0])
    return init_logp + sde_logp