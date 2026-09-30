# ── transformations.py ───────────────────────────────────────
# Reshape, flatten, slice, normalize and transform matrices.

from .core import shape, matrix


def flatten(m):
    """
    Convert a matrix to a 1D list.
    [[1,2],[3,4]] → [1, 2, 3, 4]
    """
    return [val for row in m for val in row]


def reshape(m, new_rows, new_cols):
    """
    Reshape a matrix to new dimensions.
    Total elements must stay the same.
    e.g. 2×3 → 3×2  (6 elements either way)
    """
    rows, cols = shape(m)
    total = rows * cols

    if total != new_rows * new_cols:
        raise ValueError(
            f"Cannot reshape {rows}×{cols} ({total} elements) "
            f"into {new_rows}×{new_cols} "
            f"({new_rows * new_cols} elements)"
        )

    flat = flatten(m)
    return matrix([
        flat[i * new_cols:(i + 1) * new_cols]
        for i in range(new_rows)
    ])


def slice_matrix(m, row_start, row_end, col_start, col_end):
    """
    Extract a submatrix (slice) from a matrix.
    Like Python list slicing but for 2D.
    """
    rows, cols = shape(m)

    if row_end > rows or col_end > cols:
        raise ValueError(
            f"Slice out of bounds for matrix of shape {rows}×{cols}"
        )

    return matrix([
        [m[i][j] for j in range(col_start, col_end)]
        for i in range(row_start, row_end)
    ])


def normalize(m):
    """
    Normalize matrix values to range [0, 1].
    Formula: (x - min) / (max - min)
    Useful before machine learning or comparison.
    """
    flat    = [v for row in m for v in row]
    min_val = min(flat)
    max_val = max(flat)
    rng     = max_val - min_val

    if rng == 0:
        raise ValueError("Cannot normalize — all values are equal")

    rows, cols = shape(m)
    return matrix([
        [(m[i][j] - min_val) / rng for j in range(cols)]
        for i in range(rows)
    ])


def power(m, exp):
    """
    Raise a square matrix to an integer power.
    M^2 = M × M, M^3 = M × M × M, etc.
    M^0 = identity matrix.
    """
    from .core import identity, is_square
    from .operations import multiply

    if not is_square(m):
        raise ValueError("Matrix power only defined for square matrices")
    if not isinstance(exp, int) or exp < 0:
        raise ValueError("Exponent must be a non-negative integer")

    n      = len(m)
    result = identity(n)        # start with identity (M^0)

    for _ in range(exp):
        result = multiply(result, m)

    return result


def rotate_90(m):
    """
    Rotate matrix 90 degrees clockwise.
    Transpose then reverse each row.
    """
    from .operations import transpose
    t = transpose(m)
    return matrix([row[::-1] for row in t])


def flip_horizontal(m):
    """Flip matrix left to right — reverse each row."""
    return matrix([row[::-1] for row in m])


def flip_vertical(m):
    """Flip matrix top to bottom — reverse row order."""
    return matrix(m[::-1])


def round_matrix(m, decimals=2):
    """Round every element to given decimal places."""
    rows, cols = shape(m)
    return matrix([
        [round(m[i][j], decimals) for j in range(cols)]
        for i in range(rows)
    ])