""" A Leo backend for polycirc.ast """
from typing import List, Tuple
from dataclasses import dataclass

from yarrow import Diagram
from polycirc import ast, ir

def leo_tuple(xs: List[str]):
    """ A helper function to print tuples of Leo variables or types. """
    # Prints a bare variable x_i, or a bracketed list (x_i, x_j, ...) if len(xs) > 1
    r = ", ".join(xs)
    if len(xs) > 1:
        r = "(" + r + ")"
    return r

LEO_FUNCTION_FORMAT = """function {function_name}({args_list}) -> {return_type} {{
{assignments}
{returns}
}}"""

@dataclass
class LeoBackend:
    # e.g., u32, u64, field, etc.
    semiring: str

    def assignment(self, a: ast.Assignment):
        """ Render an assignment statement as a string in Leo """
        # NOTE: we re-use the default backend to render expressions, since it's
        # the same as python.
        return f"let {str(a.lhs)}: {self.semiring} = {str(a.rhs)}"

    def function_definition(self, f: ast.FunctionDefinition) -> str:
        indent = " "*4
        args_list = ", ".join(str(a) + f": {self.semiring}" for a in f.args)
        return_type = leo_tuple([self.semiring]*len(f.returns))
        assignments = "\n".join(indent + self.assignment(a) for a in f.body)
        returns = indent + "return " + leo_tuple([str(x) for x in f.returns])

        return LEO_FUNCTION_FORMAT.format(
            function_name = f.function_name,
            args_list = args_list,
            return_type = return_type,
            assignments = assignments,
            returns = returns)

def diagram_to_leo(d: Diagram, function_name: str, semiring: str) -> str:
    """ Print a circuit as Leo code """
    f = ast.diagram_to_ast(d, function_name)
    return LeoBackend(semiring).function_definition(f)

LEO_MODULE_FORMAT = """program {module_name}.aleo {{
{body}
}}"""

def indent(s: str, n_indent: int):
    indentation = " " * n_indent
    return '\n'.join(indentation + line for line in s.split('\n'))

def diagrams_to_leo_module(ds: List[Tuple[Diagram, str]], module_name: str, semiring: str) -> str:
    """ Export multiple diagrams as a Leo module """
    n_indent = 4
    fun_strs = [ indent(diagram_to_leo(d, n, semiring), n_indent) for d, n in ds ]
    body = "\n".join(fun_strs)
    return LEO_MODULE_FORMAT.format(module_name=module_name, body=body)

if __name__ == "__main__":
    # two example circuits
    square = ir.copy(1) >> ir.mul(1)
    sum_of_squares = (square @ square) >> ir.add(1)

    leo_module = diagrams_to_leo_module([
        (square, 'square'),
        (sum_of_squares, 'sum_of_squares'),
    ], 'examples', 'u32')

    print(leo_module)
