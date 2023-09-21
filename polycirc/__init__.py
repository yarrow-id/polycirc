from yarrow import Diagram

import polycirc.ast as ast
import polycirc.optic as optic

def compile_circuit(c: Diagram, function_name='circuit'):
    """ Compile a polynomial circuit c into a python function """
    return ast.diagram_to_ast(c, function_name).to_function()

def rdiff(c: Diagram):
    """ Transform a circuit into one which simultaneously computes its forward
    and reverse pass """
    return optic.Optic().map_arrow(c)
