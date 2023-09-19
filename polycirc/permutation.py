""" Basic permutations and maps which do not rely on operations """
import numpy as np
from yarrow import Diagram, FiniteFunction

# Initial maps for empty diagrams with no wires/operations
_xn0 = FiniteFunction(None, [], dtype='object')
_wn0 = FiniteFunction(1, [])

empty = Diagram.empty(_wn0, _xn0)

def identity(n: int):
    w = FiniteFunction.terminal(n)
    return Diagram.identity(w, _xn0)

def twist(m: int, n: int):
    wm = FiniteFunction.terminal(m)
    wn = FiniteFunction.terminal(n)
    return Diagram.twist(wm, wn, _xn0)

def interleave(n: int):
    w = FiniteFunction.terminal(n)
    # NOTE: cointerleave + dagger = interleave
    return Diagram.half_spider(FiniteFunction.cointerleave(len(w)), w + w, _xn0).dagger()

def cointerleave(n: int):
    w = FiniteFunction.terminal(n)
    # NOTE: interleave + dagger = cointerleave
    return Diagram.half_spider(FiniteFunction.interleave(len(w)), w + w, _xn0).dagger()

def block_interleave(num_blocks: int, block_size: int):
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
    # use numpy to get the permutation (TODO: don't cheat :-)
    table = np.arange(m*n).reshape(m,n).T.reshape(n*m)
    f = FiniteFunction(m*n, table)
    w = FiniteFunction.terminal(m*n)
    return Diagram.half_spider(f, w, _xn0).dagger()
