from abc import ABC, abstractmethod


class BaseSparsePauliOp(ABC):
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
    def get_pauli_weight_distribution(self) -> dict[int, int]:
        """Return a histogram keyed by Pauli weight."""

    @abstractmethod
    def get_operator_stabilizer_entropy(self, alpha: float = 1):
        """Return the operator stabilizer entropy for the given Renyi order."""

    @abstractmethod
    def translate(self, x: int, system_size: int):
        """Return a new operator translated cyclically over the physical sites."""

    @abstractmethod
    def __str__(self) -> str:
        """Return a readable string representation."""


class BaseSparsePauliGradientOp(ABC):
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
