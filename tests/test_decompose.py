from polycirc.decompose import *
from tests.operations_list import ALL_OPERATIONS

def test_decompose_singletons():
    for op in ALL_OPERATIONS:
        d = op.diagram()

        # check that the op as a diagram is a singleton
        fs = list(acyclic_decompose_operations(d))
        assert len(fs) == 1
        f = fs[0]

        assert f.op == op
        assert len(f.args) == op.type[0]
        assert len(f.coargs) == op.type[1]
