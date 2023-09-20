import numpy as np
from typing import List
from hypothesis import given
from hypothesis import strategies as st

from yarrow import Diagram

import polycirc.ir as ir
from polycirc.ast import diagram_to_ast

from tests.generators import blocks, values, arrays, MAX_VALUE

################################################################################
# Basic diagrams

@given(n=values)
def test_identity(n: int):
    f = diagram_to_ast(ir.identity(n), 'id').to_function()

    xs = list(range(0, n))
    assert f(*xs) == xs

@given(n=values)
def test_cointerleave(n: int):
    # "cointerleave" is a map which takes a vector of length 2n (thought of as
    # two interleaved vectors of length n) and "uninterleaves" them.
    # So
    #   cointerleave([x0, y0, x1, y1, ... xn, yn]) = [x0, x1, ..., xn, y0, y1, ... yn]
    i = diagram_to_ast(ir.cointerleave(n), 'cointerleave').to_function()

    # [0 0 1 1 ... n n ] → [ 0 1 ... n 0 1 ... n ]
    xs = [ j for i in range(0, n) for j in [f"x{i}", f"y{i}"] ]
    rs = [ f"x{i}" for i in range(0, n) ] + [ f"y{i}" for i in range(0, n) ]

    assert i(*xs) == rs

@given(n=values)
def test_interleave(n: int):
    # interleave : n + n → 2n
    # takes two vectors and interleaves them, so
    #   interleave([x0, x1, ..., xn], [y0, y1, ..., yn]) == [x0, y0, x1, y1, ...]
    i = diagram_to_ast(ir.interleave(n), 'interleave').to_function()

    xs = [ f"x{i}" for i in range(0, n) ]
    ys = [ f"y{i}" for i in range(0, n) ]
    rs = [ j for i in range(0, n) for j in [f"x{i}", f"y{i}"] ]

    assert i(*xs, *ys) == rs

# @given(nbs=blocks(map_num_blocks=lambda n: n*2))
# def test_block_interleave(nbs):
def test_block_interleave():
    nbs = (0, [[], []])
    block_size, blocks = nbs
    n = len(blocks) // 2

    i = diagram_to_ast(ir.block_interleave(num_blocks=n, block_size=block_size), 'block_interleave').to_function()
    x = np.array(blocks)

    result = i(*x.reshape(block_size*n))
    expected = [ j for i in range(0, n) for j in blocks[i] + blocks[i+(n)] ]

    assert result == expected

@given(n=st.integers(min_value=0, max_value=10), m=st.integers(min_value=0, max_value=10))
def test_transpose(n: int, m: int):
    x = np.arange(0, m*n)
    f = diagram_to_ast(ir.transpose(n, m), 'transpose').to_function()
    assert np.all(f(*x) == x.reshape(m,n).T.reshape(m*n))

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

@given(xs=arrays, copies=st.integers(min_value=0, max_value=10))
def test_pointwise_fanout(xs, copies: int):
    block_size = len(xs)
    if copies <= 0 or block_size <= 0:
        return

    d = ir.pointwise_fanout(block_size, copies)
    pf = diagram_to_ast(d, 'pointwise_fanout').to_function()

    expected = xs * copies
    actual   = pf(*xs)

    assert actual == expected

@st.composite
def mat_vec(draw):
    m = draw(st.integers(min_value=0, max_value=10))
    n = draw(st.integers(min_value=0, max_value=10))

    M = np.random.randint(0, MAX_VALUE, (n, m))
    v = np.random.randint(0, MAX_VALUE, m)

    return M, v

@given(mv=mat_vec())
def test_mat_mul(mv):
    M, v = mv
    n, m = M.shape

    d = ir.mat_mul(n, m)
    fun = diagram_to_ast(d, 'mat_mul').to_function()

    actual = fun(*M.reshape(m*n), *v)
    expected = M @ v
    assert np.all(actual == expected)

@given(x=st.integers(), y=st.integers())
def test_min(x: int, y: int):
    fn = diagram_to_ast(ir.min(), 'min').to_function()
    assert fn(x, y) == [min(x ,y)]

@given(x=st.integers(), y=st.integers())
def test_max(x: int, y: int):
    fn = diagram_to_ast(ir.max(), 'max').to_function()
    assert fn(x, y) == [max(x ,y)]

# test pointwise versions of min/max
@given(bs=blocks(num_blocks=2))
def test_minimum(bs):
    n, [x, y] = bs

    fn = diagram_to_ast(ir.minimum(n), 'minimum').to_function()
    assert np.all(fn(*x, *y) == np.minimum(x, y))

@given(bs=blocks(num_blocks=2))
def test_maximum(bs):
    n, [x, y] = bs

    fn = diagram_to_ast(ir.maximum(n), 'maximum').to_function()
    assert np.all(fn(*x, *y) == np.maximum(x, y))

@given(low=values, high=values, x=values)
def test_clip1(low, high, x):
    fn = diagram_to_ast(ir.clip1(low, high), 'clip1').to_function()
    assert fn(x) == np.clip(x, low, high)

# test pointwise clip
@given(low=values, high=values, x=arrays)
def test_clip(low, high, x):
    n = len(x)
    fn = diagram_to_ast(ir.clip(low, high, n), 'clip').to_function()
    assert np.all(fn(*x) == np.clip(x, low, high))
