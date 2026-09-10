"""Less frequently used object methods, loaded independently of gate kernels."""
class StateMethods:
    def dot(self, other):
        from .algebra import dot
        return dot(self, other)

    inner_product = dot

    def __add__(self, other):
        from .algebra import add
        return add(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        from . import SparsePauliOp
        from .gradient import SparsePauliGradientOp
        if not isinstance(other, SparsePauliOp) or isinstance(self, SparsePauliGradientOp) != isinstance(other, SparsePauliGradientOp):
            return NotImplemented
        return self + (-1 * other)

    def __mul__(self, scalar):
        from .algebra import scale
        return scale(self, scalar)

    __rmul__ = __mul__

    def get_pauli_weight_distribution(self):
        from .analysis import weight_maps
        return weight_maps(self)[0]

    get_Pauli_weight_distribution = get_pauli_weight_distribution

    def get_pauli_weight_counts(self):
        from .analysis import weight_maps
        return weight_maps(self, include_mass=False)[1]

    get_pauli_weight_count = get_pauli_weight_counts

    def translate(self, x, system_size):
        from .analysis import translate_state
        return translate_state(self, x, system_size)

    def __str__(self):
        from .utils import sparse_pauli_op_to_str, sparse_pauli_grad_op_to_str
        return sparse_pauli_grad_op_to_str(self) if hasattr(self, 'grad_c_array') else sparse_pauli_op_to_str(self)

    __repr__ = __str__
