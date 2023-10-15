User Guide
==========

Polycirc is a library for working with *differentiable arithmetic circuits*.
Here's a simple circuit that computes the sum of squares of its two inputs:


.. code-block::

          ┌───┐    ┌───┐
          │   ├────┤   │
    x₀ ───┤ Δ │    │ * ├─┐
          │   ├────┤   │ │ ┌───┐
          └───┘    └───┘ └─┤   │
                           │ + ├──── x₀² + x₁²
          ┌───┐    ┌───┐ ┌─┤   │
          │   ├────┤   │ │ └───┘
    x₁ ───┤ Δ │    │ * ├─┘
          │   ├────┤   │
          └───┘    └───┘

Each box is a primitive operation: ``Δ``, ``*``, and ``+`` are copying,
multiplication, and addition of inputs, respectively.
Notice that operations can have multiple inputs *and* multiple outputs.

Polycirc lets you do the following:

* Build and run circuits
* Save and load
* *Differentiate* circuits

Polycirc has two aims:

1. Provide a translation layer between different ZK systems (e.g., Circom and Leo)
2. Help writing zero-knowledge machine learning applications

This guide will show you how to use the main features of polycirc.


Building and Running Circuits
-----------------------------

Circuits are built from *operations*, primitive circuits with multiple inputs
and outputs which are depicted as boxes in the example above.
The list of operations supported by polycirc can be found in
the :ref:`polycirc.operation` module.

Circuits are constructed from operations using
sequential ``>>`` and parallel ``@`` composition.
For example, if we sequentially compose copying with multiplication

.. code-block:: python

   square = ir.copy(1) >> ir.mul(1)

we get the "squaring" circuit:

.. code-block::

          ┌───┐    ┌───┐
          │   ├────┤   │
    x  ───┤ Δ │    │ * ├─── x²
          │   ├────┤   │ 
          └───┘    └───┘ 

We can compose two ``square`` circuits in parallel to get a circuit which
squares both of its inputs:

.. code-block:: python

   square_both = (square @ square)

Which gives us a circuit like this:

.. code-block::

          ┌───┐    ┌───┐
          │   ├────┤   │
    x₀ ───┤ Δ │    │ * ├─── x₀²
          │   ├────┤   │ 
          └───┘    └───┘ 
                         
          ┌───┐    ┌───┐ 
          │   ├────┤   │ 
    x₁ ───┤ Δ │    │ * ├─── x₁²
          │   ├────┤   │
          └───┘    └───┘

We can now build the sum-of-squares circuit we saw earlier, and then check it
does what we expect by executing it using :py:func:`polycirc.ast.diagram_to_function`:

.. code-block:: python

   from polycirc import diagram_to_function, ir
   square = ir.copy(1) >> ir.mul(1)
   sum_of_squares = (square @ square) >> ir.add(1)
   f = diagram_to_function(sum_of_squares)
   print(f(1, 2))

The code above will print ``5``.

Save / Load
-----------

.. warning::
   TODO

Differentiability and Learning
------------------------------

The key feature of polycirc is that circuits can be *differentiated*.
Given a circuit ``c``, :py:func:`polycirc.learner.rdiff` transforms ``c`` into
a circuit which computes its *reverse derivative*.
For an end-to-end example of using this to train a linear model, see
`this example <https://github.com/yarrow-id/polycirc/blob/master/examples/iris.py>`_.

Adding an AST Backend
---------------------

Polycirc lets you "decompile" a circuit into a higher-level language like
`circom <https://circom.io/>`_ or `leo <https://developer.aleo.org/leo/>`_.

Circuits can be converted to a generic AST using the :ref:`polycirc.ast` module,
which has a built-in Python backend.
This allows you to execute circuits with :py:func:`diagram_to_function`,
but you can also print a circuit as code:

.. code-block:: python

    from polycirc import ir, ast
    square = ir.copy(1) >> ir.mul(1)
    print(ast.diagram_to_ast(square, 'fn_name'))
    # prints the following:
    #
    #  def fn_name(x0):
    #      x1 = x0
    #      x2 = x0
    #      x3 = x1 * x2
    #      return [x3]

Printing an AST gives Python code by default, so to create a backend for another
language, you will need to write an *AST backend*.

