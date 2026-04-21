import numpy as np
import jax
import jax.numpy as jnp

from ..core import BaseSparsePauliGradientOp, BaseSparsePauliOp
from . import utils


@jax.tree_util.register_pytree_node_class
class SparsePauliOp(BaseSparsePauliOp):
    def __init__(self, xz_array: jnp.ndarray, c_array: jnp.ndarray):
        self.xz_array = jnp.asarray(xz_array)
        self.c_array = utils.as_real_array(c_array)

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

    def get_operator_stabilizer_entropy(self, alpha: float = 1) -> float:
        probabilities = jnp.abs(self.c_array) ** 2
        probabilities /= jnp.sum(probabilities)
        if alpha == 1:
            return -jnp.sum(probabilities * jnp.log(probabilities + 1e-12))
        else:
            return (1 / (1 - alpha)) * jnp.log(jnp.sum(probabilities ** alpha) + 1e-12)

    # alias for existing callers.
    def get_OSE(self, alpha: float = 1) -> float:
        return self.get_operator_stabilizer_entropy(alpha)

    def translate(self, x: int, system_size: int):
        half_words = self.xz_array.shape[1] // 2
        translated_x = utils.translate_packed_uint_rows_prefix_right(
            self.xz_array[:, :half_words],
            x,
            system_size,
        )
        translated_z = utils.translate_packed_uint_rows_prefix_right(
            self.xz_array[:, half_words:],
            x,
            system_size,
        )
        translated_xz = jnp.concatenate([translated_x, translated_z], axis=1)
        return self.__class__(translated_xz, self.c_array)

    def __add__(self, other):
        if other == 0:
            return self.__class__(self.xz_array, self.c_array)
        if not isinstance(other, SparsePauliOp):
            return NotImplemented
        from .kernels import merge_, next_pow2, slice_to_size_c_arr, slice_to_size_x_arr

        xz_array, c_array, valid_count = merge_(
            self.xz_array,
            self.c_array,
            other.xz_array,
            other.c_array,
            0.0,
        )
        slice_size = max(1, int(next_pow2(valid_count)))
        xz_array = slice_to_size_x_arr(xz_array, slice_size)
        c_array = slice_to_size_c_arr(c_array, slice_size)
        return self.__class__(xz_array, c_array)

    def __sub__(self, other):
        if not isinstance(other, SparsePauliOp):
            return NotImplemented
        return self + ((-1.0) * other)

    def __mul__(self, scalar):
        scalar = utils.real_scalar(scalar)
        if np.isclose(np.asarray(scalar), 0.0):
            return self.__class__(
                jnp.zeros((0, self.xz_array.shape[1]), dtype=self.xz_array.dtype),
                utils.as_real_array([]),
            )
        return self.__class__(self.xz_array, scalar * self.c_array)

    def __rmul__(self, scalar):
        return self * scalar


@jax.tree_util.register_pytree_node_class
class SparsePauliGradientOp(BaseSparsePauliGradientOp):
    def __init__(self, xz_array: jnp.ndarray, c_array: jnp.ndarray, grad_c_array: jnp.ndarray):
        self.xz_array = jnp.asarray(xz_array)
        self.c_array = utils.as_real_array(c_array)
        self.grad_c_array = utils.as_real_array(grad_c_array)

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

    def get_operator_stabilizer_entropy(self, alpha: float = 1) -> float:
        probabilities = jnp.abs(self.c_array) ** 2
        probabilities /= jnp.sum(probabilities)
        if alpha == 1:
            return -jnp.sum(probabilities * jnp.log(probabilities + 1e-12))
        else:
            return (1 / (1 - alpha)) * jnp.log(jnp.sum(probabilities ** alpha) + 1e-12)

    # alias for existing callers.
    def get_OSE(self, alpha: float = 1) -> float:
        return self.get_operator_stabilizer_entropy(alpha)

    def __add__(self, other):
        if other == 0:
            return self.__class__(self.xz_array, self.c_array, self.grad_c_array)
        if not isinstance(other, SparsePauliGradientOp):
            return NotImplemented
        from .kernels import (
            merge_val_grad_,
            next_pow2,
            slice_to_size_c_arr,
            slice_to_size_x_arr,
        )

        xz_array, c_array, grad_c_array, valid_count = merge_val_grad_(
            self,
            other,
            0.0,
        )
        slice_size = max(1, int(next_pow2(valid_count)))
        xz_array = slice_to_size_x_arr(xz_array, slice_size)
        c_array = slice_to_size_c_arr(c_array, slice_size)
        grad_c_array = slice_to_size_c_arr(grad_c_array, slice_size)
        return self.__class__(
            xz_array,
            c_array,
            grad_c_array,
        )

    def __radd__(self, other):
        if other == 0:
            return self.__class__(self.xz_array, self.c_array, self.grad_c_array)
        return self.__add__(other)

    def __mul__(self, scalar):
        scalar = utils.real_scalar(scalar)
        if np.isclose(np.asarray(scalar), 0.0):
            empty_xz = jnp.zeros((0, self.xz_array.shape[1]), dtype=self.xz_array.dtype)
            empty_c = utils.as_real_array([])
            return self.__class__(empty_xz, empty_c, empty_c)
        return self.__class__(self.xz_array, scalar * self.c_array, scalar * self.grad_c_array)

    def __rmul__(self, scalar):
        return self * scalar
