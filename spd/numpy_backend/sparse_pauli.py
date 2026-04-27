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

    def dot(self, other):
        if not isinstance(other, SparsePauliOp):
            raise TypeError("other must be a NumPy SparsePauliOp.")

        if len(self) <= len(other):
            return sum(coeff * other[key] for key, coeff in self.items() if key in other)

        return sum(self[key] * coeff for key, coeff in other.items() if key in self)

    def inner_product(self, other):
        return self.dot(other)

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

    def get_pauli_weight_distribution(self) -> dict[int, float]:
        distribution = {}
        pop = np.bitwise_count

        for packed, coeff in self.items():
            xz = np.asarray(packed)
            n_words = xz.shape[0] // 2
            weight = int(pop(xz[:n_words] | xz[n_words:]).astype(np.int32).sum())
            distribution[weight] = distribution.get(weight, 0.0) + float(np.abs(coeff) ** 2)

        return distribution

    def get_pauli_weight_counts(self) -> dict[int, int]:
        counts = {}
        pop = np.bitwise_count

        for packed in self.keys():
            xz = np.asarray(packed)
            n_words = xz.shape[0] // 2
            weight = int(pop(xz[:n_words] | xz[n_words:]).astype(np.int32).sum())
            counts[weight] = counts.get(weight, 0) + 1

        return counts

    def get_pauli_weight_count(self) -> dict[int, int]:
        return self.get_pauli_weight_counts()

    # Backward-compatible alias for existing callers.
    def get_Pauli_weight_distribution(self) -> dict[int, float]:
        return self.get_pauli_weight_distribution()

    def get_operator_stabilizer_entropy(self, alpha: float = 1.) -> float:
        vals = np.fromiter(self.values(), dtype=utils.get_real_dtype())
        probabilities = np.abs(vals) ** 2 / np.sum(np.abs(vals) ** 2)
        if alpha == 1:
            return -np.sum(probabilities * np.log(probabilities + 1e-12))
        else:
            return 1 / (1 - alpha) * np.log(np.sum(probabilities ** alpha) + 1e-12)

    # alias for existing callers.
    def get_OSE(self, alpha: float = 1.) -> float:
        return self.get_operator_stabilizer_entropy(alpha)

    def translate(self, x: int, system_size: int):
        result = self.__class__()
        for packed, coeff in self.items():
            packed_array = np.asarray(packed)
            half_words = packed_array.shape[0] // 2
            translated = np.concatenate(
                (
                    utils.translate_packed_uint_row_prefix_right(
                        packed_array[:half_words],
                        x,
                        system_size,
                    ),
                    utils.translate_packed_uint_row_prefix_right(
                        packed_array[half_words:],
                        x,
                        system_size,
                    ),
                )
            )
            result[tuple(np.asarray(translated).tolist())] = coeff
        return result

    def __add__(self, other):
        if other == 0:
            return self.__class__(self)
        if not isinstance(other, SparsePauliOp):
            return NotImplemented

        result = SparsePauliOp()
        for key, value in self.items():
            result[key] = value
        for key, value in other.items():
            new_value = result.get(key, 0.0) + value
            if np.isclose(new_value, 0.0):
                result.pop(key, None)
            else:
                result[key] = new_value
        return result

    def __sub__(self, other):
        if not isinstance(other, SparsePauliOp):
            return NotImplemented
        return self + ((-1.0) * other)

    def __mul__(self, scalar):
        scalar = utils.as_real_scalar(scalar)
        result = SparsePauliOp()
        if np.isclose(scalar, 0.0):
            return result
        for key, value in self.items():
            result[key] = scalar * value
        return result

    def __rmul__(self, scalar):
        return self * scalar


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

    def get_operator_stabilizer_entropy(self, alpha: float = 1.) -> float:
        vals = np.fromiter((value_grad[0] for value_grad in self.values()), dtype=utils.get_real_dtype())
        probabilities = np.abs(vals) ** 2 / np.sum(np.abs(vals) ** 2)
        if alpha == 1:
            return -np.sum(probabilities * np.log(probabilities + 1e-12))
        else:
            return 1 / (1 - alpha) * np.log(np.sum(probabilities ** alpha) + 1e-12)

    def to_spo(self):
        result = SparsePauliOp()
        for key, value_grad in self.items():
            result[key] = value_grad[0]
        return result

    # alias for existing callers.
    def get_OSE(self, alpha: float = 1.) -> float:
        return self.get_operator_stabilizer_entropy(alpha)

    def __add__(self, other):
        if other == 0:
            return self.__class__(self)
        if not isinstance(other, SparsePauliGradientOp):
            return NotImplemented

        result = SparsePauliGradientOp()
        for key, value in self.items():
            result[key] = value
        for key, value in other.items():
            coeff = result.get(key, (0.0, 0.0))[0] + value[0]
            grad = result.get(key, (0.0, 0.0))[1] + value[1]
            if np.isclose(coeff, 0.0) and np.isclose(grad, 0.0):
                result.pop(key, None)
            else:
                result[key] = (coeff, grad)
        return result

    def __radd__(self, other):
        if other == 0:
            return self.__class__(self)
        return self.__add__(other)

    def __mul__(self, scalar):
        scalar = utils.as_real_scalar(scalar)
        result = SparsePauliGradientOp()
        if np.isclose(scalar, 0.0):
            return result
        for key, value in self.items():
            result[key] = (scalar * value[0], scalar * value[1])
        return result

    def __rmul__(self, scalar):
        return self * scalar
