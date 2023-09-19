import operator
import functools
from yarrow import Diagram, FiniteFunction

from polycirc.permutation import *
from polycirc.operation import *

################################################################################
# pointwise operations

# op : 2 → 1
def pointwise(op: Diagram, n: int) -> Diagram:
    if n == 0:
        return empty

    op_type = len(op.type[0]), len(op.type[1])
    if op_type != (2, 1):
        raise ValueError(f"pointwise: op must have type 2 → 1 but was {op_type}")
    
    i = interleave(n)
    f = Diagram.tensor_list([ op for _ in range(0, n)])
    return i >> f

# add : n × n → n
def add(n: int) -> Diagram:
    return pointwise(Add().diagram(), n)

def sub(n: int) -> Diagram:
    return pointwise(Sub().diagram(), n)

def mul(n: int) -> Diagram:
    return pointwise(Mul().diagram(), n)

def shr(n: int) -> Diagram:
    return pointwise(Shr().diagram(), n)

def copy(n: int) -> Diagram:
    return pointwise_fanout(block_size=n, copies=2)

################################################################################
# Reductions

def reduce(op: Operation, unit: Operation, n: int, flip=False) -> Diagram:
    """ Construct an ``n → 1`` reduction circuit using an arbitrary associative
    operation and unit """
    op_type = (2, 1) if not flip else (1, 2)
    unit_type = (0, 1) if not flip else (1, 0)

    if op.type != op_type:
        raise ValueError(f"reduce binop {op} should have type {op_type}, but was {op.type}")
    if unit.type != unit_type:
        raise ValueError(f"reduce unit {unit} should have type {unit_type}, but was {unit.type}")

    u = unit.diagram()
    d = op.diagram()

    if n == 0:
        return u
    if n == 1:
        return (u @ identity(1)) >> d if not flip else d >> (u @ identity(1))

    layers = []
    while n >= 2:
        quotient  = n // 2
        remainder = n % 2

        ops = Diagram.tensor_list([d] * quotient)
        ids = empty if remainder == 0 else identity(1)
        layers.append(ops @ ids)

        # the number of outputs left to reduce
        n = quotient + remainder

    # compose all the layers together
    # NOTE: this diagram can be built more efficiently; need to update
    # yarrow-diagrams to include a special case for n-fold composition.
    layers = layers if not flip else list(reversed(layers))
    initializer = identity(len(layers[0].type[0]))
    return functools.reduce(operator.rshift, layers, initializer)


def sum(n: int):
    return reduce(Add(), Constant(0), n)

def product(n: int):
    return reduce(Mul(), Constant(1), n)

# Coreduce is just reduce, but for co-binary operations with a counit (like Copy/Discard)
def coreduce(binop: Operation, unit: Operation, n: int) -> Diagram:
    return reduce(binop, unit, n, flip=True)

# fanout copies its input n times.
def fanout(n: int):
    return coreduce(Copy(), Discard(), n)

################################################################################
# Combined reduce/pointwise operations

def pointwise_fanout(block_size: int, copies: int):
    """ Copy a vector of n inputs into m vectors of size n.
    >>> pointwise_fanout(n=3, copies=2).diagram().to_function("f")(1, 2, 3) == [1, 2, 3, 1, 2, 3]
    True
    """
    # fanout for a single wire
    fan = fanout(copies)

    # tensor fan for each wire
    if block_size == 0:
        fans = empty
    else:
        fans = Diagram.tensor_list([fan]*block_size)

    # interleave outputs
    return fans >> transpose(copies, block_size)

################################################################################
# Linear algebra

# The dot product of two vectors of dimension n
def dot(n: int):
    return mul(n) >> sum(n)

# Matrix-Vector product of an n×m matrix (n×m wires) with a m-dimensional vector
def mat_mul(n: int, m: int):
    d = dot(m)

    # n dot products in parallel.
    dots = Diagram.tensor_list([d] * n) if n > 0 else empty

    lhs = identity(n * m) @ pointwise_fanout(block_size=m, copies=n)
    mid = block_interleave(num_blocks=n, block_size=m)
    rhs = dots

    return lhs >> mid >> rhs

################################################################################
# Comparisons, etc.

# Computes (x ● y) · x
# for some diagram ● : 2 → 1
def f_mul_l(op):
    return (copy(1) @ identity(1)) >> (identity(1) @ op) >> mul(1)

# computes (x ● y) · y
# for some binop ●
def f_mul_r(op):
    return (identity(1) @ copy(1)) >> (op @ identity(1)) >> mul(1)

# Diagram computing min(x,y) = (x <= y)·x + (x > y)·y
# for a single input so min1 : 1 → 1
def min():
    lop = Leq().diagram()
    rop = Gt().diagram()
    return copy(2) >> (f_mul_l(lop) @ f_mul_r(rop)) >> add(1)

def max():
    lop = Gt().diagram()
    rop = Leq().diagram()
    return copy(2) >> (f_mul_l(lop) @ f_mul_r(rop)) >> add(1)

# pointwise min
def minimum(n: int):
    return pointwise(min(), n)

# pointwise max
def maximum(n: int):
    return pointwise(max(), n)

# clip a value to [low, high]
def clip1(low: int, high: int):
    lo = Constant(low).diagram()
    hi = Constant(high).diagram()
    return (lo @ identity(1)) >> max() >> (hi @ identity(1)) >> min()

# clip n values to [low, high]
def clip(low: int, high: int, n: int):
    c = clip1(low, high)
    if n == 0:
        return empty
    return Diagram.tensor_list([c]*n)
