# ── operations.py ────────────────────────────────────────────
# Matrix operations — add, subtract, multiply, transpose, etc.

from .core import shape, validate_same_shape, matrix, is_square


def add(m1, m2):
    """
    Add two matrices element by element.
    Both matrices must have the same shape.
    """
    validate_same_shape(m1, m2, "addition")
    rows, cols = shape(m1)
    return matrix([
        [m1[i][j] + m2[i][j] for j in range(cols)]
        for i in range(rows)
    ])


def subtract(m1, m2):
    """
    Subtract m2 from m1 element by element.
    Both matrices must have the same shape.
    """
    validate_same_shape(m1, m2, "subtraction")
    rows, cols = shape(m1)
    return matrix([
        [m1[i][j] - m2[i][j] for j in range(cols)]
        for i in range(rows)
    ])


def multiply(m1, m2):
    """
    Matrix multiplication (dot product).
    Rule: cols of m1 must equal rows of m2.
    Result shape: (rows of m1) × (cols of m2)

    This is NOT element-wise — it's true matrix multiplication.
    """
    r1, c1 = shape(m1)
    r2, c2 = shape(m2)

    if c1 != r2:
        raise ValueError(
            f"Cannot multiply: {r1}×{c1} by {r2}×{c2}. "
            f"Cols of m1 ({c1}) must equal rows of m2 ({r2})"
        )

    return matrix([
        [
            sum(m1[i][k] * m2[k][j] for k in range(c1))
            for j in range(c2)
        ]
        for i in range(r1)
    ])


def element_multiply(m1, m2):
    """
    Hadamard product — element-wise multiplication.
    Both matrices must have the same shape.
    Different from multiply() which is dot product.
    """
    validate_same_shape(m1, m2, "element-wise multiplication")
    rows, cols = shape(m1)
    return matrix([
        [m1[i][j] * m2[i][j] for j in range(cols)]
        for i in range(rows)
    ])


def transpose(m):
    """
    Flip rows and columns.
    Element at [i][j] moves to [j][i].
    Shape (r×c) becomes (c×r).
    """
    rows, cols = shape(m)
    return matrix([
        [m[i][j] for i in range(rows)]
        for j in range(cols)
    ])


def scalar_multiply(m, scalar):
    """
    Multiply every element by a scalar value.
    Used for scaling — e.g. multiply returns by 1.1 for 10% gain.
    """
    if not isinstance(scalar, (int, float)):
        raise TypeError("Scalar must be a number")
    rows, cols = shape(m)
    return matrix([
        [m[i][j] * scalar for j in range(cols)]
        for i in range(rows)
    ])


def scalar_add(m, scalar):
    """Add a scalar to every element."""
    rows, cols = shape(m)
    return matrix([
        [m[i][j] + scalar for j in range(cols)]
        for i in range(rows)
    ])


def determinant(m):
    """
    Calculate the determinant of a square matrix.
    Uses recursive cofactor expansion.
    det([[a,b],[c,d]]) = ad - bc
    """
    if not is_square(m):
        raise ValueError("Determinant only defined for square matrices")

    n = len(m)

    # Base cases
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]

    # Recursive cofactor expansion along first row
    det = 0
    for j in range(n):
        # Minor — matrix without row 0 and col j
        minor = [
            [m[i][k] for k in range(n) if k != j]
            for i in range(1, n)
        ]
        cofactor = ((-1) ** j) * m[0][j] * determinant(minor)
        det += cofactor

    return det


def trace(m):
    """
    Sum of diagonal elements.
    Only defined for square matrices.
    """
    if not is_square(m):
        raise ValueError("Trace only defined for square matrices")
    return sum(m[i][i] for i in range(len(m)))


def dot(v1, v2):
    """
    Dot product of two vectors (1D lists).
    Sum of element-wise products.
    """
    if len(v1) != len(v2):
        raise ValueError(
            f"Vectors must have same length: {len(v1)} vs {len(v2)}"
        )
    return sum(a * b for a, b in zip(v1, v2))