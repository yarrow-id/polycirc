""" Circuits for gradient-based machine learning.

Suppose you have the following:

* A dataset of input/output examples ``(a, b)``, which are pairs of ``A``- and ``B``-dimensional vectors.
* An initial ``P``-dimensional parameter ``θ ∈ P``
* A circuit ``model : P + A → B`` which maps parameters and inputs to outputs

Using the :py:func:`make_learner` and :py:func:`rdiff` functions in this module,
you can obtain a circuit ``step : P + A + B → P`` mapping a parameter ``θ`` and
example datum ``(a, b)`` to an updated parameter.  Iterating the ``step``
function amounts to learning with gradient descent.

For a step-by-step explanation, see the :doc:`/user_guide`.
For a complete end-to-end example, see the
`included example <https://github.com/yarrow-id/polycirc/blob/master/examples/iris.py>`_.

For mathematical background, see :cite:t:`cfgbl`.

.. warning::
    Note that the example update/displacement maps given in this file don't work
    well for integer-valued learning.

    See the
    `examples directory <https://github.com/yarrow-id/polycirc/blob/master/examples/iris.py>`_
    for a full example of integer-valued training
    using fixed-point operations.

"""
# NOTE: the example update/displacement in this file don't work well for
# integer-valued learning.
# To see an example of custom update/displacement maps which work in the
# integer-only ZK setting, see the examples directory.
from yarrow import Diagram, FiniteFunction
import polycirc.ir as ir
import polycirc.operation as op
import polycirc.optic as optic

NBITS = 10
ONE = 2**NBITS

# Gradient-descent update
def gd(lr: int):
    """ Gradient descent update map with learning rate ``lr``.

    >>> from yarrow import Diagram
    >>> u = gd(0.01)
    >>> type(u(p = 10)) is Diagram # 10-dimensional parameter vector
    True

    """
    def gd_inner(p: int):
        fwd = ir.copy(p)
        rev = (ir.identity(p) @ ir.scale(c=lr, n=p)) >> ir.sub(p)
        return optic.make_optic(fwd, rev, residual=ir.obj(p))
    return gd_inner

# Mean-squared error displacement
def mse(b: int):
    """ Mean-squared error displacement map for ``b``-dimensional predictions.
    Computes model error as the pointwise difference ``ŷ - y`` between
    prediction ``ŷ`` and true label ``y``.
    """
    fwd = ir.copy(b)
    rev = ir.sub(b)
    return optic.make_optic(fwd, rev, residual=ir.obj(b))

def rdiff(c: Diagram):
    """ Transform a circuit into an *optic* computing the forward and reverse
    passes """
    return optic.Optic().map_arrow(c)

# Build a learner from a model optic, update, and displacement maps.
def make_learner(model_optic, update, displacement, P: int, A: int):
    """ Construct a circuit ``step : P + A + B → B + P + A`` which computes both
    model output (dimension ``B``) and new parameter vector (dimension ``P``).
    Iterate this ``step`` function to train your model.

    >>> import polycirc.ir as ir
    >>> from polycirc.ast import diagram_to_function
    >>> A = 4 # input dimension
    >>> B = 3 # output dimension
    >>> P = 3*4 # parameter dimension
    >>> model = ir.mat_mul(B, A)
    >>> f = rdiff(model) # compute derivative of model
    >>> u = gd(lr=1)(P) # update circuit
    >>> d = mse(B) # displacement circuit
    >>> step = diagram_to_function(make_learner(f, u, d, P, A))
    >>> params = [0]*P # initial params
    >>> x = [1, 2, 3, 4] # example input
    >>> y = [0, 1, 0] # example output
    >>> new_params = step(*params, *x, *y)[B:B+P] # compute parameter change: a 3×4 matrix.
    >>> new_params
    [0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0]
    """
    if len(model_optic.type[0]) != P + P + A + A:
        raise ValueError(f"model arity {model_optic.type[0]} not equal to {P} + {P} + {A} + {A}")

    id_A = optic.identity(FiniteFunction.terminal(A))
    result = (update @ id_A) >> model_optic >> displacement
    return optic.adapt_optic(result)
