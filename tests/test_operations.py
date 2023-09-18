from polycirc.operation import *
from tests.operations_list import ALL_OPERATIONS

def test_operation_to_diagram():
    for op in ALL_OPERATIONS:
        # Check op source/target is compatible with type
        assert len(op.source) == op.type[0]
        assert len(op.target) == op.type[1]
        assert op.source.target == 1 # Σ₀ = 1
        assert op.target.target == 1 # Σ₀ = 1

        # check *diagram*'s source/target is the same too
        d = op.diagram()
        a, b = d.type
        assert a == op.source
        assert b == op.target

        # check diagram has just one operation xn is 
        assert len(d.G.xn) == 1

        
