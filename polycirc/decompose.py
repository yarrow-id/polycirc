from typing import List, Callable, Any, Iterator
from dataclasses import dataclass

import numpy as np
from yarrow import FiniteFunction, IndexedCoproduct, Diagram, Operations
from yarrow.finite_function import bincount
from yarrow.decompose.frobenius import frobenius_decomposition
from yarrow.numpy.layer import layer

from polycirc.operation import Operation

# The purpose of this module is basically to decompose a Diagram into a list of SingletonOp
@dataclass
class SingletonOp:
    """ A singleton operation, represented explicitly as a hypergraph """
    op: Operation
    args: List[int]
    coargs: List[int]

def decompose_wiring(d: 'Diagram'):
    # Assume 'd' is a Frobenius decomposition.
    # Then return an Operations whose s_type and t_type values are
    # actually the wi/wo maps, respectively.
    Fun = d._Fun
    Array = Fun._Array
    s_type = IndexedCoproduct(
        sources = FiniteFunction(None, bincount(d.G.xi).table),
        values = d.G.wi)
    t_type = IndexedCoproduct(
        sources = FiniteFunction(None, bincount(d.G.xo).table),
        values = d.G.wo)
    return Operations(d.G.xn, s_type, t_type)

def acyclic_decompose(d: 'Diagram'):
    # Put in convenient form
    d = frobenius_decomposition(d)

    # layer the diagram
    layering, completed = layer(d)
    is_acyclic = np.all(completed)
    if not is_acyclic:
        raise ValueError("Diagram is not acyclic")

    # extract operations
    ops = decompose_wiring(d)
    return d, ops, layering

def acyclic_decompose_operations(f: Diagram) -> Iterator[SingletonOp]:
    """ Transform an acyclic Diagram into python code """
    # Decompose an acyclic diagram (raising a ValueError if cycles exist)
    d, ops, layering = acyclic_decompose(f)

    src_ptr = np.zeros(len(ops.xn)+1, dtype='int64')
    src_ptr[1:] = np.cumsum(ops.s_type.sources.table)

    tgt_ptr = np.zeros(len(ops.xn)+1, dtype='int64')
    tgt_ptr[1:] = np.cumsum(ops.t_type.sources.table)

    args = ", ".join(f"x_{i}" for i in d.s.table)

    for op in layering.argsort().table:
        x_s = ops.s_type.values.table[src_ptr[op]:src_ptr[op+1]]
        x_t = ops.t_type.values.table[tgt_ptr[op]:tgt_ptr[op+1]]
        yield SingletonOp(d.G.xn(op), x_s, x_t)
