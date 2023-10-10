""" Compile diagrams to abstract syntax trees (AST) and python functions.
For example:

    >>> import polycirc.ir as ir
    >>> d = ir.add(1) >> ir.negate(1) # diagram representing -(x + y)
    >>> a = diagram_to_ast(d, 'fn') # convert the diagram to an AST
    >>> fn = a.to_function() # compile the AST to a python function
    >>> fn(1, 2) # run the function
    [-3]

In more detail:

- :py:func:`diagram_to_ast` converts a ``Diagram`` to an abstract syntax tree whose top
  level node is a ``FunctionDefinition``.
- :py:class:`FunctionDefinition` is an AST with:
    - a name ...
    - a list of named inputs...
    - ... outputs
    - and a list of internal :py:class:`Assignment` statements
- A :py:class:`FunctionDefinition` ``a`` can be compiled to a callable python
  function with ``a.to_function()``.

Note that in contrast to a usual AST representation, the classes in this module
do not support recursion.
The role normally played by recursion is instead handled by the combinatorial
structure of Yarrow's ``Diagram`` datastructure, which might be thought of as a
kind of "flattened" AST.
To summarise: you shouldn't be using this module to construct programs directly.
Instead, go through the :py:mod:`polycirc.ir` module!

"""
from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import ast # python's AST module

from polycirc.operation import *
import polycirc.operation as operation
from polycirc.decompose import acyclic_decompose_operations, SingletonOp

################################################################################

class ASTNode(ABC):
    """ The base class of AST nodes """
    pass

@dataclass
class Name(ASTNode):
    """ A named variable """
    id: str

    def __str__(self):
        return self.id

@dataclass
class UnaryOp(ASTNode):
    """ Unary operations (like negation ``-x``) """
    op: Operation
    rhs: Name | Constant

    def __str__(self):
        return f"{self.op} ({self.rhs})"

@dataclass
class BinOp(ASTNode):
    """ A binary operation """
    lhs: Name | Constant
    op: Operation
    rhs: Name | Constant

    def __str__(self):
        op_expr = f"{self.lhs} {self.op} {self.rhs}"

        # NOTE: for equality operations, we need to cast the result (bool) as an int
        if type(self.op) in {Eq}:
            return f"int({op_expr})"
        return op_expr

################################################################################
# Expressions are either a BinOp or a Constant.
# Recursion is not allowed: if you want to have something like (x + (y + z))
# you need to explicitly create the intermediate variables.
class Expr(ASTNode):
    """ Expression nodes evaluate to a single value.
    Note that Expr is *not* recursive: nested operations
    like ``x0 = (x1 + x2) + x3``
    cannot be represented.
    """
    value: Name | BinOp | UnaryOp | Constant

    def __str__(self):
        return f"{self.value}"

################################################################################
# Statements

@dataclass
class Assignment(ASTNode):
    """ An Assignment sets a variable's value to an expression.
    For example ``x₀ = x₁ * 2``
    """
    lhs: Name
    rhs: Expr

    def __str__(self):
        return f"{self.lhs} = {self.rhs}"

@dataclass
class FunctionDefinition(ASTNode):
    """ The top-level node of an expression tree.
    This essentially wraps a list of assigments of expressions to variables.

    >>> from polycirc.operation import Add
    >>> name = 'foo'
    >>> x0, x1, x2 = [ Name(x) for x in ["x0", "x1", "x2"] ]
    >>> args = [x0, x1]
    >>> body = [Assignment(x2, BinOp(x0, Add(), x1))]
    >>> returns = [x2]
    >>> print(FunctionDefinition(name, args, body, returns), end='')
    def foo(x0, x1):
        x2 = x0 + x1
        return [x2]
    """
    # function name
    function_name: Name
    args: List[Name]
    body: List[Assignment]
    returns: List[Name]

    def __str__(self):
        indent = " "*4
        args_list = ", ".join(str(a) for a in self.args)
        top_line = f"def {self.function_name}({args_list}):"
        body_lines = "\n".join(indent + str(a) for a in self.body)
        returns = indent + "return [" + ", ".join(str(r) for r in self.returns) + "]"

        return f"{top_line}\n{body_lines}\n{returns}\n"

    def to_function(self, filename="<string>"):
        """ Use python's 'compile' to turn this AST into a python function """
        fn_ast = ast.parse(str(self))
        env = {}
        exec(compile(fn_ast, filename=filename, mode="exec"), env)
        return env[self.function_name]

################################################################################
# Converting diagrams to ASTs

def make_name(i: int):
    """ Turn an integer `i` into the string ``"x{i}"``

    >>> make_name(2)
    Name(id='x2')
    """
    return Name(f"x{i}")

# Convert an Operation to a list of Assignment
def op_to_assignments(op: Operation, args, coargs) -> List[Assignment]:
    """ Convert an Operation and its source and target hypernodes into a list of Assignment statements """
    arity = op.type[0]
    coarity = op.type[1]
    if len(args) != arity:
        raise ValueError("Operation {op} has arity {arity} but had {len(args)} args")
    if len(coargs) != coarity:
        raise ValueError("Operation {op} has coarity {coarity} but had {len(coargs)} coargs")

    # convert to names like x0, x1, etc.
    args = list(map(make_name, args))
    coargs = list(map(make_name, coargs))

    # Most ops become BinOps, but there are some special cases (copying,
    # discarding and unary ops)
    match type(op):
        case operation.Copy:
            return [
                Assignment(coargs[0], args[0]),
                Assignment(coargs[1], args[0]),
            ]

        case operation.Constant:
            return [
                Assignment(coargs[0], op.value)
            ]

        case operation.Discard:
            return []

        case operation.Negate:
            return [Assignment(coargs[0], UnaryOp(op, args[0]))]

        # default case is a 2 → 1 binop
        case binop:
            return [Assignment(coargs[0], BinOp(args[0], op, args[1]))]

def diagram_to_ast(d: Diagram, function_name: str) -> FunctionDefinition:
    """ Turn a yarrow Diagram into a FunctionDefinition by decomposing it acyclically """
    ops = list(acyclic_decompose_operations(d))

    args = [make_name(i) for i in d.s.table]
    body = [ a for s in ops for a in op_to_assignments(s.op, s.args, s.coargs) ]
    returns = [make_name(i) for i in d.t.table]

    return FunctionDefinition(
        function_name=function_name,
        args=args,
        body=body,
        returns=returns,
    )
