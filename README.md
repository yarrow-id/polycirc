# yarrow-polycirc: Differentiable IR for Zero-Knowledge Machine Learning

[![Documentation Status](https://readthedocs.org/projects/yarrow-polycirc/badge/?version=latest)](https://yarrow-polycirc.readthedocs.io/en/latest/?badge=latest)

`yarrow-polycirc` is a a library for representing and *differentiating*
[polynomial (arithmetic) circuits](https://www.sciencedirect.com/science/article/pii/S2352220823000469).

It's used for two things:

1. A graph-based, platform-agnostic datastructure for representing arithmetic circuits
2. *Differentiating* arithmetic circuits for verified Zero-Knowledge ML.

Install with `pip`:

    pip install yarrow-polycirc

# Polynomial Circuits

Polynomial circuits are a bit like boolean circuits,
except their wires can have values in an arbitrary semiring, not just `{0, 1}`.

Here's an example of a polynomial circuit over the semiring `Z_32`:

    x₀ ───────────┐   ┌───┐
                  └───┤   │
                      │ * ├──── y₀
          ┌───┐   ┌───┤   │
    x₁ ───┤ - ├───┘   └───┘
          └───┘

Think of this circuit as computing the expression `y₀ = x₀ * (-x₁)`.
The "dangling wires" on the left are the inputs `x₀` and `x₁`,
and on the right are the outputs `y₀`.

## Differentiability

Polycirc provides a function `rdiff` to differentiate circuits:

    from polycirc import ir, rdiff
    c = ir.dot(2) # example circuit - dot product of two 2-dimensional vectors
    rc = rdiff(c)

See [./examples/iris.py](./examples/iris.py) for an example of using `rdiff` for
machine learning.
Theoretical details on `rdiff` are in Section 10 of
[this paper](https://arxiv.org/pdf/2305.01041.pdf).

# How to use it

Use the `ir` module to build circuits in an algebraic style.
For example, let's first import the ir module:

    from polycirc import ir

Basic circuits in the IR can be combined by stacking:

    f = ir.identity(1) @ ir.negate(1)
    
Think of `f` visually like this:

    x₀ ─────────── y₀
          ┌───┐   
    x₁ ───┤ - ├─── y₁
          └───┘

We can plug the outputs of `f` into another circuit using the *composition*
operation `>>`:

    c = f >> ir.mul(1)

which looks like this:

    x₀ ───────────┐   ┌───┐
                  └───┤   │
                      │ * ├──── y₀
          ┌───┐   ┌───┤   │
    x₁ ───┤ - ├───┘   └───┘
          └───┘

We can print this program as Python code:

    from polycirc.ast import diagram_to_ast
    print(diagram_to_ast(c, 'multiply_negate'))

And we get the following:

```py
def multiply_negate(x0, x1):
    x2 = - (x1)
    x3 = x0 * x2
    return [x3]
```

# Iris demo

See the included [iris example](./examples/iris.py) which uses differentiability
of the IR to train a simple linear model
for the [iris dataset](https://archive.ics.uci.edu/dataset/53/iris).
Both *training and inference* of this model happens with polynomial circuits, so
`yarrow-polycirc` can be used for **on-chain training** of models.

First, install the example dependencies

    pip install '.[dev,example]'

Download the Iris dataset

    ./data/get-iris-data.sh

Run the example:

    python -m examples.iris train

You should see something like this:

    loading data...
    final parameters [-118, 912, -884, 408, -20, -189, 68, 28, -1407, -641, 1407, 2336]
    predicting...
    accuracy: 96.0

You can also print the iris model circuits as python code:

    > python -m examples.iris print
    def predict(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x17, x22, x27):
        x13 = x12
        x14 = x12
        x18 = x17

        ⋮
        <snipped>


# Adding a backend

If you want to add your own backend, you don't need to work with diagrams
(circuits) directly.
Instead, you can use the `polycirc.ast` module to transform a circuit to a
generic AST with `polycirc.ast.diagram_to_ast`.

    from polycirc.ast import diagram_to_ast
    c = make_some_circuit()
    diagram_to_ast(c, 'function_name')

From this generic AST, you can then convert your arithmetic circuit to your
target language.

For an in depth example, see the provided
[Leo backend example](./examples/leo.py)
and the
[docs](https://yarrow-polycirc.readthedocs.io/en/latest/user_guide.html#adding-an-ast-backend)

# Running Tests

Install dev dependencies

    pip install '.[dev]'

and run tests

    pytest
