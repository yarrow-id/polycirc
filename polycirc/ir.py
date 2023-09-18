import numpy as np
import operator
import functools
from yarrow import Diagram, FiniteFunction, IndexedCoproduct
from polycirc.operation import *

# Initial maps for empty diagrams with no wires/operations
_xn0 = FiniteFunction(None, [], dtype='object')
_wn0 = FiniteFunction(1, [])

empty = Diagram.empty(_wn0, _xn0)

def identity(n: int):
    w = FiniteFunction.terminal(n)
    return Diagram.identity(w, _xn0)

def twist(m: int, n: int):
    wm = FiniteFunction.terminal(m)
    wn = FiniteFunction.terminal(n)
    return Diagram.twist(wm, wn, _xn0)

def interleave(n: int):
    w = FiniteFunction.terminal(n)
    # NOTE: cointerleave + dagger = interleave
    return Diagram.half_spider(FiniteFunction.cointerleave(len(w)), w + w, _xn0).dagger()

def cointerleave(n: int):
    w = FiniteFunction.terminal(n)
    # NOTE: interleave + dagger = cointerleave
    return Diagram.half_spider(FiniteFunction.interleave(len(w)), w + w, _xn0).dagger()

def block_interleave(num_blocks: int, block_size: int):
    # weave together n blocks each of length m, keeping blocks in order.
    n = num_blocks
    m = block_size

    sources = FiniteFunction(m+1, np.full(2*n, m))
    w = FiniteFunction.terminal(2*n*m)
    # cointerleave here because it's going to be daggered in the half_spider!
    f = sources.injections(FiniteFunction.cointerleave(n))
    return Diagram.half_spider(f, w, _xn0).dagger()

# Given an n×m array, compute the *transpose permutation diagram* which
def transpose(n: int, m: int):
    # use numpy to get the permutation (TODO: don't cheat :-)
    table = np.arange(m*n).reshape(m,n).T.reshape(n*m)
    f = FiniteFunction(m*n, table)
    w = FiniteFunction.terminal(m*n)
    return Diagram.half_spider(f, w, _xn0).dagger()

################################################################################
# pointwise operations

def pointwise(op, n: int) -> Diagram:
    if n == 0:
        return empty
    
    d = op.diagram()
    i = interleave(n)
    f = Diagram.tensor_list([ d for _ in range(0, n)])
    return i >> f

# add : n × n → n
def add(n: int) -> Diagram:
    return pointwise(Add(), n)

def sub(n: int) -> Diagram:
    return pointwise(Sub(), n)

def mul(n: int) -> Diagram:
    return pointwise(Mul(), n)

def shr(n: int) -> Diagram:
    return pointwise(Shr(), n)

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
