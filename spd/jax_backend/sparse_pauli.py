import numpy as np
import jax
import jax.numpy as jnp

from ..core import BaseSparsePauliGradientOp, BaseSparsePauliOp
from . import utils


@jax.tree_util.register_pytree_node_class
class SparsePauliOp(BaseSparsePauliOp):
    def __init__(self, xz_array: jnp.ndarray, c_array: jnp.ndarray):
        self.xz_array = xz_array
        self.c_array = c_array

    def tree_flatten(self):
        return ((self.xz_array, self.c_array), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __iter__(self):
        return iter((self.xz_array, self.c_array))

    def __str__(self):
        return utils.sparse_pauli_op_to_str(self)

    __repr__ = __str__

    def get_size(self) -> int:
        return self.c_array.size

    def get_norm_square(self):
        return jnp.sum(jnp.abs(self.c_array) ** 2)

    def get_expectation_value(self, basis: str = "0"):
        n_words = self.xz_array.shape[1] // 2
        if basis in ["0", "Z"]:
            mask = jnp.all(self.xz_array[:, :n_words] == 0, axis=1)
        elif basis in ["+", "X"]:
            mask = jnp.all(self.xz_array[:, n_words:] == 0, axis=1)
        else:
            raise NotImplementedError(f"Expectation value in basis {basis} not implemented.")

        exp_val = jnp.sum(self.c_array[mask])
        return jnp.real(exp_val)

    def get_pauli_weight_distribution(self) -> dict[int, int]:
        distribution = {}
        pop = np.bitwise_count
        xz_rows = np.asarray(self.xz_array)

        for xz in xz_rows:
            n_words = xz.shape[0] // 2
            weight = int(pop(xz[:n_words] | xz[n_words:]).astype(np.int32).sum())
            distribution[weight] = distribution.get(weight, 0) + 1

        return distribution

    # Backward-compatible alias for existing callers.
    def get_Pauli_weight_distribution(self) -> dict[int, int]:
        return self.get_pauli_weight_distribution()


@jax.tree_util.register_pytree_node_class
class SparsePauliGradientOp(BaseSparsePauliGradientOp):
    def __init__(self, xz_array: jnp.ndarray, c_array: jnp.ndarray, grad_c_array: jnp.ndarray):
        self.xz_array = xz_array
        self.c_array = c_array
        self.grad_c_array = grad_c_array

    def tree_flatten(self):
        return ((self.xz_array, self.c_array, self.grad_c_array), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __iter__(self):
        return iter((self.xz_array, self.c_array, self.grad_c_array))

    def __str__(self):
        return utils.sparse_pauli_grad_op_to_str(self)

    __repr__ = __str__

    def get_size(self) -> int:
        return self.c_array.size

    def get_norm_square(self):
        return jnp.sum(jnp.abs(self.c_array) ** 2)
