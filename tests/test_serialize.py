from hypothesis import given

from tests.generators import circuits, circuit_and_inputs

from polycirc import ir, diagram_to_function
from polycirc.serialize import *

@given(dx=circuit_and_inputs())
def test_decompose_roundtrip_execution(dx: Diagram):
    """ Serializing/deserializing a diagram gives the same function """
    d, x = dx
    expected = d
    sd = SerializedDiagram.from_diagram(d)
    actual = sd.to_diagram()

    f = diagram_to_function(expected)
    g = diagram_to_function(actual)
    assert f(*x) == g(*x)

# NOTE: When we call acyclic_decompose_operations, it actually changes the
# ordering of operations, so we only test that the serialized results are the
# same, not the diagrams!
@given(d=circuits())
def test_decompose_roundtrip(d: Diagram):
    """ Ensure that serialization roundtrips """
    # If we serialize a diagram, it should be the same as...
    expected = SerializedDiagram.from_diagram(d)
    # .. if we deserialize and re-serialize.
    actual   = SerializedDiagram.from_diagram(expected.to_diagram())

    assert expected == actual


@given(d=circuits())
def test_json_roundtrip(d: Diagram):
    """ Ensure that serialization roundtrips (all the way to JSON) """
    # If we serialize a diagram, it should be the same as...
    expected = diagram_to_json(d)
    # .. if we deserialize and re-serialize.
    actual   = diagram_to_json(diagram_from_json(expected))

    assert expected == actual
