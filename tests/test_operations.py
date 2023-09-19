import pytest
from hypothesis import given
from hypothesis import strategies as st

from polycirc.ast import diagram_to_ast
from polycirc.operation import *

from tests.operations_list import ALL_OPERATIONS

@pytest.mark.parametrize("op", ALL_OPERATIONS)
def test_operation_to_diagram(op: Operation):
    # Check op source/target is compatible with type
    assert len(op.source) == op.type[0]
    assert len(op.target) == op.type[1]
    assert op.source.target == 1 # Σ₀ = 1
    assert op.target.target == 1 # Σ₀ = 1

    # check *diagram*'s source/target is the same too
    d = op.diagram()
    a, b = d.type
    assert a == op.source
    assert b == op.target

    # check diagram has just one operation xn is
    assert len(d.G.xn) == 1

@pytest.mark.parametrize("op", ALL_OPERATIONS)
def test_compile_operation(op: Operation):
    args = list(range(0, op.type[0]))

    fn = diagram_to_ast(op.diagram(), op.__class__.__name__).to_function()
    assert len(fn(*args)) == op.type[1]

@pytest.mark.parametrize("op", ALL_OPERATIONS)
def test_revs(op: Operation):
    d_fwd = op.diagram()
    d_rev = op.rev()

    A, B = d_fwd.type
    M = op.residual()
    print(op)
    assert d_rev.type == (M + B, A)

################################################################################
# Test that compiled operation functions do what they're supposed to.

# R[mul] as a function
def mul_rev(x0, x1, dy):
    return [x1*dy, x0*dy]

values = st.integers(min_value=0, max_value=100)

@given(x0=values, x1=values, dy=values)
def test_mul_rev(x0, x1, dy):
    mul_rev_compiled = diagram_to_ast(Mul().rev(), 'mul_rev').to_function()
    assert mul_rev_compiled(x0, x1, dy) == mul_rev(x0, x1, dy)

# multiply's forward map is the *lens* forward map, so mul(x,y) produces [x*y, x, y]
@given(x0=values, x1=values)
def test_mul_fwd(x0, x1):
    mul_fwd_compiled = diagram_to_ast(Mul().fwd(), 'mul_fwd').to_function()
    assert mul_fwd_compiled(x0, x1) == [x0*x1, x0, x1]
