""" A generic, simplified AST class modeled on python's ast library.
    Each ASTNode's str() method returns the node represented as Python code.
"""
from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import ast # python's AST module

from polycirc.operation import *
import polycirc.operation as operation
from polycirc.decompose import acyclic_decompose_operations, SingletonOp

################################################################################

# Base class of all nodes
class ASTNode(ABC):
    pass

@dataclass
class Name:
    id: str

    def __str__(self):
        return self.id

@dataclass
class UnaryOp:
    op: Operation
    rhs: Name | Constant

    def __str__(self):
        return f"{self.op} ({self.rhs})"

@dataclass
class BinOp:
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
class Expr:
    value: Name | BinOp | UnaryOp | Constant

    def __str__(self):
        return f"{self.value}"

################################################################################
# Statements

@dataclass
class Assignment(ASTNode):
    lhs: Name
    rhs: Expr

    def __str__(self):
        return f"{self.lhs} = {self.rhs}"

@dataclass
class FunctionDefinition(ASTNode):
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
    return Name(f"x{i}")

# Convert an Operation to a list of Assignment
def op_to_assignments(op: Operation, args, coargs) -> List[Assignment]:
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
    # List[SingletonOp]
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
