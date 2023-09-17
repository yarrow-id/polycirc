""" A generic, simplified AST class modeled on python's ast library.
    Each ASTNode's str() method returns the node represented as Python code.
"""
from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass

################################################################################

# Base class of all nodes
class ASTNode(ABC):
    pass

@dataclass
class Name:
    id: str

    def __str__(self):
        return self.id

################################################################################
# All constants are values of the underlying field, which we store in an
# arbitrary-precision int.
@dataclass
class Constant:
    value: int

    def __str__(self):
        return f"{self.value}"

################################################################################
# Binary operators +, -, *

class Operator(ABC):
    pass

class Add(Operator):
    def __str__(self):
        return "+"

class Sub(Operator):
    def __str__(self):
        return "-"

class Mul(Operator):
    def __str__(self):
        return "*"

class Shr(Operator):
    def __str__(self):
        return ">>"

class Eq(Operator):
    def __str__(self):
        return "=="

@dataclass
class BinOp:
    lhs: Name | Constant
    op: Operator
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
    value: BinOp | Constant

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
        returns = indent + "return " + ", ".join(str(r) for r in self.returns)

        return f"{top_line}\n{body_lines}\n{returns}\n"
