"""Compatibility shim: brax 0.14.2 calls jax.device_put_replicated, which was
removed in jax 0.11. The Intel oneAPI plugin pins jax/jaxlib to 0.11.x, so
downgrading jax is not an option on this stack. Restore the removed API with
the documented drop-in replacement (stack leaves over devices, shard via
NamedSharding — every device ends up holding a full copy, same semantics).

Import this module BEFORE importing brax.training; it only affects the current
process, site-packages stays untouched.
"""

import numpy as np
import jax
from jax import numpy as jp


def _device_put_replicated(pytree, devices):
    n = len(devices)
    mesh = jax.sharding.Mesh(np.asarray(devices), ("_rep"))
    spec = jax.sharding.PartitionSpec("_rep")
    sharding = jax.sharding.NamedSharding(mesh, spec)
    return jax.device_put(jax.tree.map(lambda x: jp.stack([x] * n), pytree), sharding)


if not hasattr(jax, "device_put_replicated"):
    jax.device_put_replicated = _device_put_replicated
