import numpy as np

from hypothesis import given
from hypothesis import strategies as st

################################################################################
# Generators

MAX_VALUE = 100
MAX_ARRAY_SIZE = 100

values = st.integers(min_value=0, max_value=MAX_VALUE)
arrays = st.lists(values, max_size=MAX_ARRAY_SIZE)

@st.composite
def blocks(draw, num_blocks=None, block_size=None, map_num_blocks=None):
    block_size = block_size or draw(st.integers(min_value=0, max_value=10))
    num_blocks = num_blocks or draw(st.integers(min_value=0, max_value=10))

    num_blocks = num_blocks if map_num_blocks is None else map_num_blocks(num_blocks)

    # note: have to return block_size because when the result list is length 0, it's ambiguous.
    value_lists = st.lists(values, min_size=block_size, max_size=block_size)
    return block_size, [ draw(value_lists) for _ in range(0, num_blocks) ]


# Draw a random matrix M and vector v of compatible dimensions for computing M @ v.
@st.composite
def mat_vec(draw, max_dimension=10):
    m = draw(st.integers(min_value=0, max_value=max_dimension))
    n = draw(st.integers(min_value=0, max_value=max_dimension))

    M = np.random.randint(0, MAX_VALUE, (n, m))
    v = np.random.randint(0, MAX_VALUE, m)

    return M, v

# mat_vec plus gradients.
@st.composite
def mat_vec_gradients(draw):
    M, x = draw(mat_vec(max_dimension=5))
    dy = np.random.randint(0, MAX_VALUE, M.shape[0])
    return M, x, dy
