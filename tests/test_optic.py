from hypothesis import given
from hypothesis import strategies as st

from polycirc.ast import diagram_to_ast
from polycirc.optic import Optic, adapt_optic
import polycirc.operation as op

values = st.integers(min_value=0, max_value=1000)

@given(x0=values, x1=values, dy=values)
def test_optic_functor_mul(x0, x1, dy):
    mul = op.Mul().diagram()
    mul_optic = Optic().map_arrow(mul)

    adapted = adapt_optic(mul_optic)
    f = diagram_to_ast(adapted, 'mul').to_function()

    assert f(x0, x1, dy) == [x0*x1, x1*dy, x0*dy]
