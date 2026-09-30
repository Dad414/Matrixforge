# ── __init__.py ──────────────────────────────────────────────
# Exposes the clean public API of matrixforge.
# Users import directly from matrixforge — no submodule needed.

# Core — create and inspect
from .core import (
    matrix,
    zeros,
    ones,
    identity,
    fill,
    shape,
    display,
    is_square,
    is_symmetric,
)

# Operations
from .operations import (
    add,
    subtract,
    multiply,
    element_multiply,
    transpose,
    scalar_multiply,
    scalar_add,
    determinant,
    trace,
    dot,
)

# Transformations
from .transformations import (
    flatten,
    reshape,
    slice_matrix,
    normalize,
    power,
    rotate_90,
    flip_horizontal,
    flip_vertical,
    round_matrix,
)

# Stats
from .stats import (
    row_sum, col_sum,
    row_mean, col_mean,
    row_max, row_min,
    col_max, col_min,
    total_sum, total_mean,
    total_max, total_min,
    variance, std_dev,
    correlation,
    summary,
)

# Utils
from .utils import (
    random_matrix,
    are_equal,
    from_csv_string,
    to_csv_string,
    compare,
)

__version__ = "0.1.0"
__author__  = "Your Name"