# ── stats.py ─────────────────────────────────────────────────
# Statistical operations on matrices.
# Row-wise, column-wise, and whole-matrix statistics.

from .core import shape, matrix
from .transformations import flatten
from .operations import transpose


def row_sum(m):
    """Sum of each row → returns a list."""
    return [sum(row) for row in m]


def col_sum(m):
    _, cols = shape(m)
    return [round(sum(m[i][j] for i in range(len(m))), 10)
            for j in range(cols)]


def row_mean(m):
    """Mean of each row → returns a list."""
    return [sum(row) / len(row) for row in m]


def col_mean(m):
    rows, cols = shape(m)
    return [round(sum(m[i][j] for i in range(rows)) / rows, 10)
            for j in range(cols)]


def row_max(m):
    """Max value in each row → returns a list."""
    return [max(row) for row in m]


def row_min(m):
    """Min value in each row → returns a list."""
    return [min(row) for row in m]


def col_max(m):
    """Max value in each column → returns a list."""
    rows, cols = shape(m)
    return [max(m[i][j] for i in range(rows)) for j in range(cols)]


def col_min(m):
    """Min value in each column → returns a list."""
    rows, cols = shape(m)
    return [min(m[i][j] for i in range(rows)) for j in range(cols)]


def total_sum(m):
    """Sum of ALL elements in the matrix."""
    return sum(flatten(m))


def total_mean(m):
    """Mean of ALL elements in the matrix."""
    flat = flatten(m)
    return sum(flat) / len(flat)


def total_max(m):
    """Maximum value in the entire matrix."""
    return max(flatten(m))


def total_min(m):
    """Minimum value in the entire matrix."""
    return min(flatten(m))


def variance(m):
    """
    Variance of all elements.
    Measures how spread out the values are.
    var = mean of squared differences from mean
    """
    flat  = flatten(m)
    mean  = sum(flat) / len(flat)
    return sum((x - mean) ** 2 for x in flat) / len(flat)


def std_dev(m):
    """
    Standard deviation — square root of variance.
    How much values deviate from the mean on average.
    """
    return variance(m) ** 0.5


def correlation(m):
    """
    Correlation matrix — how each row relates to others.
    Values range from -1 (inverse) to 1 (perfect correlation).
    Useful in finance — comparing asset returns.
    """
    rows, cols = shape(m)
    means = row_mean(m)
    stds  = [
        (sum((m[i][j] - means[i])**2 for j in range(cols)) / cols) ** 0.5
        for i in range(rows)
    ]

    result = []
    for i in range(rows):
        row = []
        for k in range(rows):
            if stds[i] == 0 or stds[k] == 0:
                row.append(0.0)
            else:
                cov = sum(
                    (m[i][j] - means[i]) * (m[k][j] - means[k])
                    for j in range(cols)
                ) / cols
                row.append(round(cov / (stds[i] * stds[k]), 4))
        result.append(row)

    return matrix(result)


def summary(m):
    """Print a full statistical summary of the matrix."""
    r, c = shape(m)
    print(f"Shape:    {r}×{c}")
    print(f"Total:    {total_sum(m):.4f}")
    print(f"Mean:     {total_mean(m):.4f}")
    print(f"Min:      {total_min(m):.4f}")
    print(f"Max:      {total_max(m):.4f}")
    print(f"Variance: {variance(m):.4f}")
    print(f"Std Dev:  {std_dev(m):.4f}")
    print(f"Row sums: {row_sum(m)}")
    print(f"Col sums: {col_sum(m)}")