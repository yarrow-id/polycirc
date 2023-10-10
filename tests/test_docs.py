import doctest
from polycirc import ast
from polycirc import operation

def test_docs_ast():
    results = doctest.testmod(m=ast)
    assert results.failed == 0

def test_docs_operation():
    results = doctest.testmod(m=operation)
    assert results.failed == 0
