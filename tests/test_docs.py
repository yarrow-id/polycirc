import doctest
from polycirc import ast
from polycirc import operation
from polycirc import ir
from polycirc import permutation
from polycirc import learner

def test_docs_ast():
    results = doctest.testmod(m=ast)
    assert results.failed == 0

def test_docs_operation():
    results = doctest.testmod(m=operation)
    assert results.failed == 0

def test_docs_ir():
    results = doctest.testmod(m=ir)
    assert results.failed == 0

def test_docs_permutation():
    results = doctest.testmod(m=permutation)
    assert results.failed == 0

def test_docs_learner():
    results = doctest.testmod(m=learner)
    assert results.failed == 0
