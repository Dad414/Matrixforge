# ── utils.py ─────────────────────────────────────────────────
# Helper utilities — print comparison, generate random matrix,
# convert formats, check equality.

import random
from .core import shape, matrix, display


def random_matrix(rows, cols, low=0, high=10, decimals=0):
    """
    Generate a matrix with random values.
    decimals=0 → integers, decimals=2 → floats
    """
    if decimals == 0:
        return matrix([
            [random.randint(int(low), int(high)) for _ in range(cols)]
            for _ in range(rows)
        ])
    else:
        return matrix([
            [round(random.uniform(low, high), decimals)
             for _ in range(cols)]
            for _ in range(rows)
        ])


def are_equal(m1, m2, tolerance=1e-9):
    """
    Check if two matrices are equal.
    Uses tolerance for floating point comparisons.
    """
    if shape(m1) != shape(m2):
        return False
    rows, cols = shape(m1)
    return all(
        abs(m1[i][j] - m2[i][j]) < tolerance
        for i in range(rows)
        for j in range(cols)
    )


def from_csv_string(csv_string):
    """
    Create a matrix from a CSV-formatted string.
    '1,2,3\\n4,5,6' → matrix([[1,2,3],[4,5,6]])
    """
    rows = []
    for line in csv_string.strip().split("\n"):
        row = [float(x.strip()) for x in line.split(",")]
        rows.append(row)
    return matrix(rows)


def to_csv_string(m):
    """Export matrix to CSV-formatted string."""
    return "\n".join(
        ",".join(str(val) for val in row)
        for row in m
    )


def compare(m1, m2, label1="Matrix 1", label2="Matrix 2"):
    """Display two matrices side by side for comparison."""
    display(m1, label=label1)
    display(m2, label=label2)
    print(f"Same shape:  {shape(m1) == shape(m2)}")
    print(f"Are equal:   {are_equal(m1, m2)}\n")