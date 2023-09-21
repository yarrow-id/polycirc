# NOTE: the example update/displacement in this file don't work well for
# integer-valued learning.
# To see an example of custom update/displacement maps which work in the
# integer-only ZK setting, see the examples directory.
from yarrow import FiniteFunction
import polycirc.ir as ir
import polycirc.operation as op
import polycirc.optic as optic

NBITS = 10
ONE = 2**NBITS

# Gradient-descent update
def gd(lr: int):
    def gd_inner(p: int):
        fwd = ir.copy(p)
        rev = ir.sub(p)
        return optic.make_optic(fwd, rev, residual=ir.obj(p))
    return gd_inner

# Mean-squared error displacement
def mse(b: int):
    fwd = ir.copy(b)
    rev = ir.sub(b)
    return optic.make_optic(fwd, rev, residual=ir.obj(b))

# Build a learner from a model optic, update, and displacement maps.
def make_learner(model_optic, update, displacement, P: int, A: int):
    if len(model_optic.type[0]) != P + P + A + A:
        raise ValueError(f"model arity {model_optic.type[0]} not equal to {P} + {P} + {A} + {A}")

    id_A = optic.identity(FiniteFunction.terminal(A))
    result = (update @ id_A) >> model_optic >> displacement
    return optic.adapt_optic(result)
