""" Basic permutations and maps which do not rely on operations """
import numpy as np
from yarrow import Diagram, FiniteFunction

# Initial maps for empty diagrams with no wires/operations
_xn0 = FiniteFunction(None, [], dtype='object')
_wn0 = FiniteFunction(1, [])

empty = Diagram.empty(_wn0, _xn0)

def identity(n: int):
    """ ``identity(n) : n → n`` is the circuit which returns its inputs unchanged """
    w = FiniteFunction.terminal(n)
    return Diagram.identity(w, _xn0)

def twist(m: int, n: int):
    """ ``twist(a, b) : a + b → b + a`` swaps the first ``a`` inputs with the remaining ``b`` inputs. """
    wm = FiniteFunction.terminal(m)
    wn = FiniteFunction.terminal(n)
    return Diagram.twist(wm, wn, _xn0)

def interleave(n: int):
    """ ``interleave(n) : n + n → 2n`` is the circuit taking two ``n``-dimensional inputs and interleaving their values.

    >>> from polycirc.ast import diagram_to_function
    >>> f = diagram_to_function(interleave(2))
    >>> f(1, 2, 1, 2)
    [1, 1, 2, 2]
    """
    w = FiniteFunction.terminal(n)
    # NOTE: cointerleave + dagger = interleave
    return Diagram.half_spider(FiniteFunction.cointerleave(len(w)), w + w, _xn0).dagger()

def cointerleave(n: int):
    """ ``cointerleave(n) : 2n → n`` is the inverse of ``interleave(n)``

    >>> from polycirc.ast import diagram_to_function
    >>> f = diagram_to_function(cointerleave(2))
    >>> f(1, 1, 2, 2)
    [1, 2, 1, 2]
    """
    w = FiniteFunction.terminal(n)
    # NOTE: interleave + dagger = cointerleave
    return Diagram.half_spider(FiniteFunction.interleave(len(w)), w + w, _xn0).dagger()

def block_interleave(num_blocks: int, block_size: int):
    """ ``block_interleave(num_blocks, block_size)`` is like ``interleave``, each ``n``-dimensional input vector is thought of having elements of size ``m``.

    >>> from polycirc.ast import diagram_to_function
    >>> f = diagram_to_function(interleave(2))
    >>> g = diagram_to_function(block_interleave(num_blocks=2, block_size=1))
    >>> f(1, 1, 2, 2) == g(1, 1, 2, 2)
    True
    """
    # weave together n blocks each of length m, keeping blocks in order.
    n = num_blocks
    m = block_size

    sources = FiniteFunction(m+1, np.full(2*n, m))
    w = FiniteFunction.terminal(2*n*m)
    # cointerleave here because it's going to be daggered in the half_spider!
    f = sources.injections(FiniteFunction.cointerleave(n))
    return Diagram.half_spider(f, w, _xn0).dagger()

# Given an n×m array, compute the *transpose permutation diagram* which
def transpose(n: int, m: int):
    """ Given an ``n×m``-dimensional input thought of as a 2D array, *transpose* it to get a ``m×n``-dimensional output """
    # use numpy to get the permutation (TODO: don't cheat :-)
    table = np.arange(m*n).reshape(m,n).T.reshape(n*m)
    f = FiniteFunction(m*n, table)
    w = FiniteFunction.terminal(m*n)
    return Diagram.half_spider(f, w, _xn0).dagger()
