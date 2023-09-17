import ast
from polycirc.ast import Name, Assignment, Constant, BinOp, Add, Mul, Sub, Eq, FunctionDefinition

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
    return x3, x2, x4, x5

# Express the same function as an AST
def make_fn_test_ast(function_name: str):
    args = [Name(f"x{i}") for i in [0,1]]
    returns = [Name(f"x{i}") for i in [3,2,4,5]]

    body = [
        Assignment(Name('x2'), BinOp(Name('x0'), Add(), Name('x1'))),
        Assignment(Name('x3'), BinOp(Name('x1'), Eq(), Name('x0'))),
        Assignment(Name('x4'), BinOp(Name('x0'), Mul(), Constant(314))),
        Assignment(Name('x5'), Constant(0)),
    ]

    return FunctionDefinition(
        function_name=function_name,
        args=args,
        body=body,
        returns=returns,
    )

# Compile the AST above to executable python code
FUNCTION_NAME = 'fn_test'
fn_ast = ast.parse(str(make_fn_test_ast(FUNCTION_NAME)))
state = {}
exec(compile(fn_ast, filename='<string>', mode='exec'), state)
fn_test_compiled = state['fn_test']

################################################################################
# Test that fn_test and fn_test_compiled compute the same values

values = st.integers(min_value=0, max_value=1000)
@given(a=values, b=values)
def test_add_sub(a, b):
    assert fn_test(a, b) == fn_test_compiled(a, b)
