import numpy as np

from hypothesis import given
from hypothesis import strategies as st

from tests.generators import blocks, mat_vec_gradients

from polycirc.ast import diagram_to_ast
from polycirc.optic import Optic, adapt_optic
import polycirc.operation as op
import polycirc.ir as ir

values = st.integers(min_value=0, max_value=1000)

def test_optic_functor_empty():
    f = Optic().map_arrow(ir.empty)
    g = adapt_optic(f)
    assert len(g.type[0]) == 0
    assert len(g.type[1]) == 0

@given(x0=values, x1=values, dy=values)
def test_optic_functor_mul(x0, x1, dy):
    mul = op.Mul().diagram()
    mul_optic = Optic().map_arrow(mul)

    adapted = adapt_optic(mul_optic)
    f = diagram_to_ast(adapted, 'mul').to_function()

    assert f(x0, x1, dy) == [x0*x1, x1*dy, x0*dy]

@given(bs=blocks(num_blocks=3))
def test_optic_functor_addN(bs):
    n, [x, y, z] = bs
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    add_optic = Optic().map_arrow(ir.add(n))
    adapted = adapt_optic(add_optic)

    ast = diagram_to_ast(adapted, f"add{n}")
    f = ast.to_function()
    assert f(*x, *y, *z) == [*(x+y), *z, *z]

@given(bs=blocks(num_blocks=2))
def test_optic_functor_negateN(bs):
    n, [x, dy] = bs

    negate_optic = Optic().map_arrow(ir.negate(n))
    adapted = adapt_optic(negate_optic)

    ast = diagram_to_ast(adapted, f"negate{n}")
    f = ast.to_function()

    actual = f(*x, *dy)
    expected = -np.array([x, dy]).reshape(2*n)
    assert np.all(actual == expected)

@given(bs=blocks(num_blocks=3))
def test_optic_functor_mulN(bs):
    n, [x0, x1, dy] = bs

    x0 = np.array(x0)
    x1 = np.array(x1)
    dy = np.array(dy)

    if n == 0:
        return

    mul_optic = Optic().map_arrow(ir.mul(n))
    adapted = adapt_optic(mul_optic)
    f = diagram_to_ast(adapted, f"mul{n}").to_function()

    assert np.all(f(*x0, *x1, *dy) == np.array([x0*x1, x1*dy, x0*dy]).reshape(n*3))

@given(Mxd=mat_vec_gradients())
def test_optic_functor_mat_mul(Mxd):
    M, x, dy = Mxd
    n, m = M.shape

    mat_mul_optic = Optic().map_arrow(ir.mat_mul(n, m))
    adapted = adapt_optic(mat_mul_optic)
    f = diagram_to_ast(adapted, f"mat_mul{n}").to_function()

    e0 = M @ x
    e1 = np.outer(dy, x)
    e2 = M.T @ dy

    expected = np.concatenate([e0.reshape(n), e1.reshape(m*n), e2.reshape(m)])

    inputs = M.reshape
    actual   = f(*M.reshape(m*n), *x.reshape(m), *dy.reshape(n))

    assert np.all(actual == expected)
