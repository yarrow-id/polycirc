from typing import List
from hypothesis import given
from hypothesis import strategies as st

from yarrow import Diagram

import polycirc.ir as ir
from polycirc.ast import diagram_to_ast

################################################################################
# Generators

MAX_VALUE = 100
MAX_ARRAY_SIZE = 100

values = st.integers(min_value=0, max_value=MAX_VALUE)
arrays = st.lists(values, max_size=MAX_ARRAY_SIZE)

################################################################################
# Pointwise diagrams

@given(n=values)
def test_pointwise_add(n):
    d = ir.add(n)
    assert len(d.type[0]) == 2*n
    assert len(d.type[1]) == n

    fun = diagram_to_ast(d, 'fun').to_function()

    # test that executing the function corresponding to the diagram gives the expected result
    xs = [1] * n
    ys = [2] * n
    rs = [3] * n
    assert fun(*xs, *ys) == rs

# Same test for add, but using shr instead.
@given(n=values)
def test_pointwise_shr(n):
    d = ir.shr(n)
    assert len(d.type[0]) == 2*n
    assert len(d.type[1]) == n

    fun = diagram_to_ast(d, 'fun').to_function()

    xs = [2] * n
    ys = [1] * n
    rs = [1] * n
    assert fun(*xs, *ys) == rs

@given(n=values)
def test_pointwise_add_symbolic(n: int):
    d = ir.add(n)
    assert len(d.type[0]) == 2*n
    assert len(d.type[1]) == n

    fun = diagram_to_ast(d, 'fun').to_function()

    # Test a symbolic version, where we rely on python's + operator doing concatenation on lists.
    xs = [ [f"x{i}"] for i in range(0, n) ]
    ys = [ [f"y{i}"] for i in range(0, n) ]
    rs = [ x+y for x, y in zip(xs, ys) ]
    assert fun(*xs, *ys) == rs

def test_identity_empty():
    lhs = ir.identity(0) 
    rhs = ir.empty
    assert lhs == rhs

################################################################################
# Reduce

@given(xs=arrays)
def test_sum(xs: List[int]):
    n = len(xs)
    d = ir.sum(n)
    assert len(d.type[0]) == n
    assert len(d.type[1]) == 1

    fun = diagram_to_ast(d, 'sum').to_function()
    assert fun(*xs) == [sum(xs)]

@given(x=values, n=values)
def test_fanout(x, n):
    d = ir.fanout(n)
    assert len(d.type[0]) == 1
    assert len(d.type[1]) == n

    fanout = diagram_to_ast(d, 'fanout').to_function()
    assert fanout(x) == [x]*n
