""" Operations are the basic units of computation in the IR """
from dataclasses import dataclass
from typing import Tuple
from abc import ABC, abstractmethod

from yarrow import FiniteFunction, Diagram

from polycirc.permutation import identity, twist, interleave, cointerleave

class Operation(ABC):
    # The type of an operation f : A → B
    # is the number of inputs A and outputs B.
    @property
    @abstractmethod
    def type(self) -> Tuple[int, int]:
        ...

    @property
    def source(self):
        return FiniteFunction.terminal(self.type[0])

    @property
    def target(self):
        return FiniteFunction.terminal(self.type[1])

    @property
    def fwd(self) -> Diagram:
        ...

    @property
    def rev(self) -> Diagram:
        ...

    @property
    def residual(self) -> Diagram:
        ...

    def diagram(self):
        xn = FiniteFunction(None, [self], dtype='object')
        return Diagram.singleton(self.source, self.target, xn)


class LinearOperation(Operation):
    def fwd(self) -> Diagram:
        return self.diagram()

    def rev(self) -> Diagram:
        return self.dagger()

    def residual(self):
        return FiniteFunction.initial(1)

    @abstractmethod
    def dagger(self) -> Diagram:
        ...

################################################################################
# Concrete operations

class Add(LinearOperation):
    @property
    def type(self):
        return (2, 1)

    def dagger(self):
        return Copy().diagram()

    def __str__(self):
        return "+"

class Negate(LinearOperation):
    @property
    def type(self):
        return (1, 1)

    def dagger(self):
        return Negate().diagram()

    def __str__(self):
        return "-"

# Constants are values of the underlying field, which we store in an
# arbitrary-precision int.
@dataclass
class Constant(LinearOperation):
    value: int

    @property
    def type(self):
        return (0, 1)

    def dagger(self):
        return Discard().diagram()

    def __str__(self):
        return f"{self.value}"

class Copy(LinearOperation):
    @property
    def type(self):
        return (1, 2)

    def dagger(self):
        return Add().diagram()

    def __str__(self):
        # NOTE: this is not used!
        return "Δ"

class Discard(LinearOperation):
    @property
    def type(self):
        return (1, 0)

    def dagger(self):
        return Constant(0).diagram()

    def __str__(self):
        # NOTE: this is not used!
        return "!"

class Sub(LinearOperation):
    @property
    def type(self):
        return (2, 1)

    def dagger(self):
        return Copy().diagram() >> (identity(1) @ Negate().diagram())

    def __str__(self):
        return "-"

class Mul(Operation):
    @property
    def type(self):
        return (2, 1)

    def residual(self):
        return self.source

    def fwd(self):
        # We haven't defined the ir copy map yet, so we have to build it
        # manually
        copy1 = Copy().diagram()
        copy2 = (copy1 @ copy1) >> cointerleave(2)
        return copy2 >> (self.diagram() @ identity(2))

    def rev(self):
        mul0    = Mul().diagram()
        mul1    = Mul().diagram()

        lhs = twist(1,1) @ Copy().diagram()
        mid = interleave(2)
        rhs = mul0 @ mul1
        return lhs >> mid >> rhs

    def __str__(self):
        return "*"

class Shr(LinearOperation):
    @property
    def type(self):
        return (2, 1)

    def __str__(self):
        return ">>"

    # TODO FIXME: straight-through estimator is a poor choice here!
    def dagger(self):
        return Copy().diagram()

################################################################################
# Equality and Comparison operations
# NOTE: All equality/comparison operations use the straight-through estimator,
# for each operation R[op] = Δ

class Eq(LinearOperation):
    @property
    def type(self):
        return (2, 1)

    def __str__(self):
        return "=="

    def dagger(self):
        return Copy().diagram()

class Gt(LinearOperation):
    @property
    def type(self):
        return (2, 1)

    def __str__(self):
        return ">"

    def dagger(self):
        return Copy().diagram()

class Geq(LinearOperation):
    @property
    def type(self):
        return (2, 1)

    def __str__(self):
        return ">="

    def dagger(self):
        return Copy().diagram()

class Lt(LinearOperation):
    @property
    def type(self):
        return (2, 1)

    def __str__(self):
        return "<"

    def dagger(self):
        return Copy().diagram()

class Leq(LinearOperation):
    @property
    def type(self):
        return (2, 1)

    def __str__(self):
        return "<="

    def dagger(self):
        return Copy().diagram()
