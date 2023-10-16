""" Serialize and deserialize circuits.

This module provides two methods ...

You can also turn diagrams into dicts and back:

Format
------

.. warning::

    TODO

.. code-block::

    TODO
"""
from typing import List
from dataclasses import dataclass, asdict, field
import json

from yarrow import Diagram
import polycirc.operation as ops
from polycirc.operation import Operation
from polycirc.decompose import acyclic_decompose_operations, SingletonOp, singleton_ops_to_diagram

################################################################################
# Exported functions

def diagram_to_dict(d: Diagram) -> dict:
    """ Serialize a diagram as a dict """
    return asdict(SerializedDiagram.from_diagram(d))

def diagram_to_json(d: Diagram) -> str:
    """ Serialize a diagram as a JSON string """
    return json.dumps(diagram_to_dict(d))

def diagram_from_dict(d: dict) -> Diagram:
    """ Deserialize a Diagram from a dict """
    sd = SerializedDiagram(
            wires=d['wires'],
            sources=d['sources'],
            targets=d['targets'],
            operations=[ SerializedOperation(**o) for o in d['operations']])
    return sd.to_diagram()

def diagram_from_json(s: str) -> Diagram:
    """ Deserialize a Diagram from a JSON string """
    return diagram_from_dict(json.loads(s))

################################################################################
# DTO objects (+ main implementation)

@dataclass
class SerializedOperation:
    """ DTO corresponding to a single operation within a hypergraph """
    operation: str | int # if operation is an int i, it's a Constant(i)
    sources: List[int]
    targets: List[int]

    def to_op_label(self) -> Operation:
        # Constant and Negate are the only two operations that don't solely
        # depend on the string label.
        if type(self.operation) == int:
            return ops.Constant(self.operation)
        elif self.operation == "-" and len(self.sources) == 1:
            return ops.Negate()
        else:
            op = self.operation
            # TODO FIXME: store a master list of operations / text representations somewhere!
            # This duplicates code from polycirc.operations!
            match op:
                case "+": return ops.Add()
                case "Δ": return ops.Copy()
                case "!": return ops.Discard()
                case "-": return ops.Sub()
                case "*": return ops.Mul()
                case ">>": return ops.Shr()
                case "=": return ops.Eq()
                case ">": return ops.Gt()
                case ">=": return ops.Geq()
                case "<": return ops.Lt()
                case "<=": return ops.Leq()
                case _:
                    raise ValueError(f"Unknown operation {op}")

    def to_singleton(self) -> SingletonOp:
        op = self.to_op_label()
        A, B = op.type
        if A != len(self.sources) or B != len(self.targets):
            raise ValueError(f"Operation labeled {op} should have type {A} → {B}, but had type {len(self.sources)} → {len(self.targets)}")

        return SingletonOp(op, self.sources, self.targets)

    @staticmethod
    def from_singleton(s: SingletonOp) -> 'SerializedOperation':
        op_label = s.op.value if type(s.op) is ops.Constant else str(s.op)
        return SerializedOperation(op_label, s.args.tolist(), s.coargs.tolist())

@dataclass
class SerializedDiagram:
    """ DTO for a complete circuit (diagram). """
    wires: int
    sources: List[int]
    targets: List[int]
    operations: List[SerializedOperation]

    @staticmethod
    def from_diagram(d: Diagram) -> 'SerializedDiagram':
        """ Serialize a diagram to a dict """
        singletons = list(acyclic_decompose_operations(d))

        ops = [ SerializedOperation.from_singleton(s) for s in singletons ]
        return SerializedDiagram(
            wires=d.wires,
            sources=d.s.table.tolist(),
            targets=d.t.table.tolist(),
            operations=ops)

    def to_diagram(self) -> Diagram:
        """ Deserialize a diagram from a dict """
        ret = singleton_ops_to_diagram(
            self.wires,
            self.sources,
            self.targets,
            [op.to_singleton() for op in self.operations])
        return ret
