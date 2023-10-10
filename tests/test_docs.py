import doctest
from polycirc import ast

def test_docs_ast():
    results = doctest.testmod(m=ast)
    assert results.failed == 0
