from abc import ABC, abstractmethod


class ActiveIndexMetadata:
    # None is the legacy full canonical register; [] is a scalar operator.
    active_qubits = None
    system_size = None

    def set_active_qubits(self, system_size, active_qubits=None):
        """Set the fixed index universe and the original index of each column.

        None denotes the full canonical ordering; [] denotes a scalar. Storage
        width and identity padding are checked by the backend when reindexing.
        """
        if not isinstance(system_size, int) or isinstance(system_size, bool) or system_size < 1:
            raise ValueError("system_size must be a positive integer.")
        active = list(range(system_size)) if active_qubits is None else list(active_qubits)
        if any(not isinstance(q, int) or isinstance(q, bool) or not 0 <= q < system_size
               for q in active):
            raise ValueError("active_qubits must contain integer indices within system_size.")
        if len(set(active)) != len(active):
            raise ValueError("active_qubits must be unique.")
        self.system_size = system_size
        self.active_qubits = None if active_qubits is None else active
        return self

    @property
    def qubit_indices(self):
        """Resolved column mapping; requires a known system_size for canonical SPOs."""
        if self.active_qubits is not None:
            return list(self.active_qubits)
        if self.system_size is None:
            raise ValueError("system_size is required to resolve canonical SPO columns.")
        return list(range(self.system_size))

    def _check_mapping(self, other):
        left = self.active_qubits
        right = other.active_qubits
        if left is None and self.system_size is not None:
            left = list(range(self.system_size))
        if right is None and other.system_size is not None:
            right = list(range(other.system_size))
        if left != right and (self.active_qubits is not None or other.active_qubits is not None):
            raise ValueError("Operators have different active_qubits mappings.")
        if (self.system_size is not None and other.system_size is not None
                and self.system_size != other.system_size):
            raise ValueError("Operators have different system_size values.")

    def _copy_metadata_to(self, result):
        result.active_qubits = None if self.active_qubits is None else list(self.active_qubits)
        result.system_size = self.system_size
        return result


class BaseSparsePauliOp(ActiveIndexMetadata, ABC):
    @abstractmethod
    def get_size(self) -> int:
        """Return the number of Pauli terms stored in the operator."""

    @abstractmethod
    def get_norm_square(self):
        """Return the squared norm of the operator coefficients."""

    @abstractmethod
    def dot(self, other):
        """Return the coefficient dot product with another sparse Pauli operator."""

    @abstractmethod
    def get_expectation_value(self, basis: str = "0"):
        """Return the expectation value in the requested product basis."""

    @abstractmethod
    def get_pauli_weight_distribution(self) -> dict[int, float]:
        """Return squared coefficient mass keyed by Pauli weight."""

    @abstractmethod
    def get_pauli_weight_counts(self) -> dict[int, int]:
        """Return term counts keyed by Pauli weight."""

    @abstractmethod
    def get_operator_stabilizer_entropy(self, alpha: float = 1):
        """Return the operator stabilizer entropy for the given Renyi order."""

    @abstractmethod
    def translate(self, x: int, system_size: int):
        """Return a new operator translated cyclically over the physical sites."""

    @abstractmethod
    def __str__(self) -> str:
        """Return a readable string representation."""


class BaseSparsePauliGradientOp(ActiveIndexMetadata, ABC):
    @abstractmethod
    def get_size(self) -> int:
        """Return the number of Pauli terms stored in the gradient object."""

    @abstractmethod
    def get_norm_square(self):
        """Return the squared norm of the primal coefficients."""

    @abstractmethod
    def get_operator_stabilizer_entropy(self, alpha: float = 1):
        """Return the operator stabilizer entropy of the primal coefficients."""

    @abstractmethod
    def to_spo(self):
        """Return the primal coefficients as a sparse Pauli operator."""

    @abstractmethod
    def __str__(self) -> str:
        """Return a readable string representation."""
