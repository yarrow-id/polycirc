import numpy as np
from hypothesis import given

from tests.generators import arrays, values, blocks
from polycirc.learner import gd, mse
import polycirc.optic as optic
from polycirc import diagram_to_function

@given(bs=blocks(num_blocks=2), lr=values)
def test_gd(bs, lr):
    n, [x, y] = bs
    x = np.array(x)
    y = np.array(y)

    u = gd(lr)(n)

    o = optic.adapt_optic(u)
    f = diagram_to_function(o, 'update')

    np.all(f(*x, *y) == np.concatenate([x, x - lr*y]))

@given(bs=blocks(num_blocks=2))
def test_mse(bs):
    p, [x, y] = bs
    d = mse(p)
    x = np.array(x)

    o = optic.adapt_optic(d)
    f = diagram_to_function(o, 'displacement')

    np.all(f(*x, *y) == np.concatenate([x, x - y]))
