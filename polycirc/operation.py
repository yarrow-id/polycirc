""" Operations are the basic units of computation in the IR """
from typing import Tuple
from abc import ABC, abstractmethod

from yarrow import FiniteFunction, Diagram

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

    def diagram(self):
        xn = FiniteFunction(None, [self], dtype='object')
        return Diagram.singleton(self.source, self.target, xn)

class Add(Operation):
    @property
    def type(self):
        return (2, 1)

    def __str__(self):
        return "+"

class Sub(Operation):
    @property
    def type(self):
        return (2, 1)

    def __str__(self):
        return "-"

class Mul(Operation):
    @property
    def type(self):
        return (2, 1)

    def __str__(self):
        return "*"

class Shr(Operation):
    @property
    def type(self):
        return (2, 1)

    def __str__(self):
        return ">>"

class Eq(Operation):
    @property
    def type(self):
        return (2, 1)

    def __str__(self):
        return "=="
