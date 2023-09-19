import ast
from polycirc.ast import Name, Assignment, Constant, BinOp, UnaryOp, Add, Negate, Copy, Mul, Sub, Eq, FunctionDefinition, diagram_to_ast

from hypothesis import given
from hypothesis import strategies as st

################################################################################
# Make an example function to test against
# It exhibits the following features:
#   - binary operations
#   - binary operations with constants
#   - equality operations
#   - setting variables to constants
#   - multiple returns
def fn_test(x0, x1):
    x2 = x0 + x1
    x3 = int(x1 == x0)
    x4 = x0 * 314 # test binops with constants
    x5 = 0 # test bare constants
    x6 = -x0
    return [x3, x2, x4, x5, x6]

# Express the same function as an AST
def make_fn_test_ast(function_name: str):
    args = [Name(f"x{i}") for i in [0,1]]
    returns = [Name(f"x{i}") for i in [3,2,4,5,6]]

    body = [
        Assignment(Name('x2'), BinOp(Name('x0'), Add(), Name('x1'))),
        Assignment(Name('x3'), BinOp(Name('x1'), Eq(), Name('x0'))),
        Assignment(Name('x4'), BinOp(Name('x0'), Mul(), Constant(314))),
        Assignment(Name('x5'), Constant(0)),
        Assignment(Name('x6'), UnaryOp(Negate(), Name('x0'))),
    ]

    return FunctionDefinition(
        function_name=function_name,
        args=args,
        body=body,
        returns=returns,
    )

# Compile the AST above to executable python code
FUNCTION_NAME = 'fn_test'
fn_test_compiled = make_fn_test_ast(FUNCTION_NAME).to_function()

################################################################################
# Test that fn_test and fn_test_compiled compute the same values

values = st.integers(min_value=0, max_value=1000)

@given(a=values, b=values)
def test_complex_function(a, b):
    assert fn_test(a, b) == fn_test_compiled(a, b)

mul_compiled = diagram_to_ast(Mul().diagram(), 'mul_compiled').to_function()

@given(a=values, b=values)
def test_diagram_to_ast(a, b):
    assert mul_compiled(a, b) == [a * b]

################################################################################
# Test copying and non-binary operations

copy_compiled = diagram_to_ast(Copy().diagram(), 'copy_compiled').to_function()
@given(a=values)
def test_copy(a):
    assert copy_compiled(a) == [a, a]

@given(a=values)
def test_constant(a):
    constant_compiled = diagram_to_ast(Constant(a).diagram(), 'constant').to_function()
    assert constant_compiled() == [a]
