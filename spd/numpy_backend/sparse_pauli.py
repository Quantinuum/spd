import numpy as np

from ..core import BaseSparsePauliGradientOp, BaseSparsePauliOp
from . import utils


class SparsePauliOp(dict, BaseSparsePauliOp):
    def __setitem__(self, key, value):
        super().__setitem__(key, utils.as_real_scalar(value))

    def __str__(self):
        return utils.sparse_pauli_op_to_str(self)

    __repr__ = __str__

    def get_size(self) -> int:
        return len(self)

    def get_norm_square(self):
        vals = np.fromiter(self.values(), dtype=utils.get_real_dtype())
        return np.linalg.norm(vals) ** 2

    def get_expectation_value(self, basis: str = "0"):
        exp_val = 0
        n_words = len(next(iter(self))) // 2
        if basis in ["0", "Z"]:
            for packed, coeff in self.items():
                xz_array = np.asarray(packed)
                if np.all(xz_array[:n_words] == 0):
                    exp_val += coeff
        elif basis in ["+", "X"]:
            for packed, coeff in self.items():
                xz_array = np.asarray(packed)
                if np.all(xz_array[n_words:] == 0):
                    exp_val += coeff
        else:
            raise ValueError(f"Unsupported basis: {basis}")

        return exp_val

    def get_pauli_weight_distribution(self) -> dict[int, int]:
        distribution = {}
        pop = np.bitwise_count

        for packed in self.keys():
            xz = np.asarray(packed)
            n_words = xz.shape[0] // 2
            weight = int(pop(xz[:n_words] | xz[n_words:]).astype(np.int32).sum())
            distribution[weight] = distribution.get(weight, 0) + 1

        return distribution

    # Backward-compatible alias for existing callers.
    def get_Pauli_weight_distribution(self) -> dict[int, int]:
        return self.get_pauli_weight_distribution()


class SparsePauliGradientOp(dict, BaseSparsePauliGradientOp):
    def __setitem__(self, key, value):
        super().__setitem__(key, utils.as_real_pair(value))

    def __str__(self):
        return utils.sparse_pauli_grad_op_to_str(self)

    __repr__ = __str__

    def get_size(self) -> int:
        return len(self)

    def get_norm_square(self):
        vals = np.fromiter((value_grad[0] for value_grad in self.values()), dtype=utils.get_real_dtype())
        return np.sum(vals ** 2, dtype=utils.get_real_dtype())
