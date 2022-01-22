# going through the flax design and making some initial changes

# flax notes

from dataclasses import dataclass
import jax
from jax import numpy as jnp, random, lax, jit
from flax import linen as nn
from flax.core.scope import Scope
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module, compact
import flax.linen as linen
import numpy as np
from dense import Dense
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict


from pprint import pprint
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.deprecated import nn
from flax.deprecated.nn import initializers
from dense import Dense
from flax.linen import Module
import jax
from jax import lax, numpy as jnp, random


def Normalize(x, axis=0, eps=1e-8):
    """normalize tensor along axis with eps for nan protection"""
    x = x - jnp.mean(x, axis=axis, keepdims=True)
    x = x / jnp.sqrt(jnp.mean(jnp.square(x), axis=axis, keepdims=True) + eps)
    return x


def LinearExplicit(Dense):
    """Linear layer with express feature config (instead of via shape inference)"""
    in_features: Optional[int] = None

    def setup(
        self,
    ):
        #  run fake batch to init
        self.__call__(
            jnp.zeroes(
                (
                    1,
                    self.in_features,
                )
            )
        )


class ffn(linen.Module):
    def setup(
        self,
    ):
        self.fc1 = LinearExplicit(in_features=1024, features=512)
        self.fc2 = LinearExplicit(in_features=512, features=10)
        # act_fn = nn.swish
        pprint(self.fc2.variables)

    def __call__(self, x):
        output = self.fc1(x)
        activations = nn.swish(output)
        result = self.fc2(activations)
        return result


rngkey = jax.random.PRNGKey(2022)
init_vars = ffn().init({"params": rngkey}, jnp.ones(10, 1024))
pprint(init_vars)

