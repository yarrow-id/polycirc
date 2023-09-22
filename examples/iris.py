import argparse
import numpy as np

import polycirc
from polycirc import compile_circuit, rdiff, ir, optic, learner
from polycirc.learner import make_learner

from examples.util import load_iris, model_accuracy

# This demo solves the 3-class classification problem of the Iris dataset, which
# has 4-dimensional inputs
INPUT_SIZE = 4
OUTPUT_SIZE = 3

# The model is a single fully-connected layer, so it has 4*3 = 12
# parameters to be learned.
NPARAM = OUTPUT_SIZE * INPUT_SIZE

# We'll be training in fixed-point representation with 10 bits of precision.
NBITS = 10 # Number of fixed-point bits to use
ONE = 1 << NBITS # The number 1, encoded in fixed point representation with NBITS

################################################################################
# Update/Displacement maps
#
# Start by defining "update" and "displacement" maps.
#   - For mathematical background, see this paper: https://arxiv.org/abs/2103.01931
#   - For more simple examples of update/displacement maps which work for
#     floating-point data, see polycirc.learner

# A gradient-descent-like update using fixed-point arithmetic.
def fixed_gd(lr: int):
    def fixed_gd_inner(p: int):
        # TODO: define "scale" and "constant" diagrams in polycirc.ir?
        fwd = ir.copy(p)

        scale_shift = ir.shrc(NBITS, p) >> ir.scale(lr, p) >> ir.shrc(NBITS, p)
        rev = (ir.identity(p) @ scale_shift) >> ir.sub(p)

        return optic.make_optic(fwd, rev, residual=ir.obj(p))
    return fixed_gd_inner

# Custom loss which clips model output into the range [0, 1] before comparing
# with one-hot encoded labels.
def cliploss(b: int):
    f = ir.clip(-ONE, ONE, b) >> ir.addc(ONE, b) >> ir.shrc(1, b) 
    fwd = ir.shrc(NBITS, b) >> ir.copy(b) >> (ir.identity(b) @ f)

    rev = ir.sub(b)
    return optic.make_optic(fwd, rev, residual=ir.obj(b))

################################################################################
# Constructing the model circuit

def build_model():
    # 'model' is a circuit with m*n + m inputs and n outputs, which performs a
    # matrix multiplication.
    # We're learning a simple single-layer neural network with a sigmoid-like
    # activation.
    model_circuit = ir.mat_mul(OUTPUT_SIZE, INPUT_SIZE)

    # DIFFERENTIABILITY
    # =================
    # Reverse-differentiate model_circuit using the optic algorithm described in
    # Data-Parallel Algorithms for String Diagrams
    # ( see https://arxiv.org/abs/2305.01041 )
    #
    f = rdiff(model_circuit) # : (4*3 + 4) + 3 → 3 + (4*3 + 4)

    # make_learner turns a model into a learner using a choice of update
    # (optimiser) and displacement (loss).
    u = fixed_gd(ONE >> 7)(NPARAM) # learning rate of 1/2^7
    d = cliploss(OUTPUT_SIZE)  # mean squared error loss
    step_circuit = learner.make_learner(f, u, d, NPARAM, INPUT_SIZE)

    # model_circuit takes a matrix (n*m values) and a vector (m values) and
    # produces another vector (n values).
    # step_circuit computes the same vector *plus a parameter update*.
    return model_circuit, step_circuit
 
################################################################################
# Training

def main():
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Subparser for the 'train' command
    train_parser = subparsers.add_parser("train", help="train model")
    train_parser.add_argument('--iris-data', default='data/iris.csv')

    # Subparser for the 'print' command
    print_parser = subparsers.add_parser("print", help="print model circuit as python")

    args = parser.parse_args()

    match args.command:
        case "train":
            return train(args.iris_data)
        case "print":
            # Print the forward and step circuits of the model as python functions
            fwd, rev = build_model()
            print(polycirc.ast.diagram_to_ast(fwd, 'predict'))
            print(polycirc.ast.diagram_to_ast(rev, 'step'))


def train(iris_data):
    # Load data from CSV
    print("loading data...")
    x, y = load_iris(iris_data, scale=ONE)

    # initialize params
    p = np.zeros(NPARAM, dtype=int).tolist()

    # compile the forward (fwd) and gradient (step) passes of the circuit into
    # python functions.
    predict_circuit, step_circuit = build_model()
    predict = compile_circuit(predict_circuit)
    step = compile_circuit(step_circuit)

    N = len(x)
    NUM_ITER = N * 60

    # Iterate through data in a (deterministic) schedule which speeds up
    # training a lot.
    q = np.arange(N)
    q[0::3] = np.arange(50)
    q[1::3] = np.arange(50) + 50
    q[2::3] = np.arange(50) + 100


    # Do a single step of SGD-like training.
    # NOTE: we call tolist on numpy values to get python ints, but this
    # isn't strictly necessary.
    # NOTE: 'step' produces model output, new parameters, and new data. We
    # only need the new parameters.
    for j in range(0, NUM_ITER):
        i = q[j % N]
        p = step(*p, *x[i].tolist(), *y[i].tolist())[OUTPUT_SIZE:OUTPUT_SIZE+NPARAM]
    
    # Train accuracy.
    # NOTE: we don't bother with a test or holdout set: this is just to demo
    # differentiability of the IR.
    print('final parameters', p)
    print("predicting...")
    acc = model_accuracy(predict, p, x, y)
    print(f'accuracy: {100*acc}')

if __name__ == "__main__":
    main()
