from polycirc import ir
from polycirc.operation import *

ALL_OPERATIONS = [
    Add(),
    Sub(),
    Mul(),
    Shr(),
    Copy(),
    Discard(),
    Constant(0),
    Constant(1),
    Negate(),
    Eq(),
    Gt(),
    Lt(),
    Geq(),
    Leq(),
]
