# ── core.py ──────────────────────────────────────────────────
# Core module — create, display and validate matrices.
# A matrix is stored as a list of lists.
# matrix([[1,2],[3,4]]) is the main entry point.


def matrix(data, shape=None, by="row"):
    """
    Create a matrix from a list of lists, a flat list, or a dataset.

    Usage:
        # Standard — list of lists
        matrix([[1, 2], [3, 4]])

        # From flat list with shape
        matrix([1,2,3,4,5,6], shape=(2,3))          # 2×3 row-priority
        matrix([1,2,3,4,5,6], shape=(2,3), by="col") # 2×3 col-priority

        # From pandas column
        matrix(df[["Q1","Q2"]].values.tolist(), shape=(10,2))

    Parameters:
        data  : list of lists OR flat list of numbers
        shape : tuple (rows, cols) — optional
                if provided, reshapes data into that dimension
        by    : "row" → fills row by row (default)
                "col" → fills column by column
    """

    # ── Validate data type ────────────────────────────────────
    if not isinstance(data, list):
        raise TypeError("Input must be a list")

    # ── Flat list + shape provided ────────────────────────────
    if shape is not None:
        rows, cols = shape

        # Flatten if list of lists was passed with shape
        if data and isinstance(data[0], list):
            data = [val for row in data for val in row]

        # Validate all elements are numbers
        for i, val in enumerate(data):
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"Element at index {i} must be int or float. "
                    f"Got: {type(val).__name__}"
                )

        total = rows * cols

        # Pad with zeros if not enough data
        if len(data) < total:
            print(f"Warning: {total - len(data)} missing values "
                  f"padded with zeros")
            data = data + [0] * (total - len(data))

        # Trim if too much data
        if len(data) > total:
            print(f"Warning: data truncated from "
                  f"{len(data)} to {total} elements")
            data = data[:total]

        # Fill row by row
        if by == "row":
            result = [
                data[i * cols:(i + 1) * cols]
                for i in range(rows)
            ]

        # Fill column by column
        elif by == "col":
            # Initialize empty matrix
            result = [[0] * cols for _ in range(rows)]
            for idx, val in enumerate(data):
                col = idx // rows      # which column
                row = idx % rows       # which row
                result[row][col] = val

        else:
            raise ValueError(
                f"by must be 'row' or 'col'. Got: '{by}'"
            )

        return result

    # ── No shape — standard list of lists ────────────────────
    # Validate each row is a list
    for i, row in enumerate(data):
        if not isinstance(row, list):
            raise TypeError(f"Row {i} must be a list")

    # Validate all rows same length
    row_lengths = [len(row) for row in data]
    if len(set(row_lengths)) > 1:
        raise ValueError(
            f"All rows must have equal length. "
            f"Got lengths: {row_lengths}"
        )

    # Validate all elements are numbers
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"Element at [{i}][{j}] must be int or float. "
                    f"Got: {type(val).__name__}"
                )

    return data

def zeros(rows, cols):
    """Create a matrix filled with zeros."""
    return matrix([[0] * cols for _ in range(rows)])


def ones(rows, cols):
    """Create a matrix filled with ones."""
    return matrix([[1] * cols for _ in range(rows)])


def identity(n):
    """
    Create an n×n identity matrix.
    Diagonal = 1, everything else = 0.
    """
    return matrix([
        [1 if i == j else 0 for j in range(n)]
        for i in range(n)
    ])


def fill(rows, cols, value):
    """Create a matrix filled with a constant value."""
    if not isinstance(value, (int, float)):
        raise TypeError("Fill value must be a number")
    return matrix([[value] * cols for _ in range(rows)])


def shape(m):
    """
    Return (rows, cols) of a matrix.
    Like numpy's .shape — tells you the dimensions.
    """
    rows = len(m)
    cols = len(m[0]) if rows > 0 else 0
    return (rows, cols)


def display(m, label=None, decimals=2):
    """
    Pretty print a matrix with optional label.
    Aligns columns for clean output.
    """
    if label:
        print(f"\n{label}")
        print("─" * (len(label)))

    # Find the widest number for column alignment
    flat    = [val for row in m for val in row]
    max_len = max(len(f"{v:.{decimals}f}") for v in flat)

    rows, cols = shape(m)
    print(f"Shape: {rows}×{cols}")

    for i, row in enumerate(m):
        prefix = "⌈" if i == 0 else "⌊" if i == rows - 1 else "│"
        suffix = "⌉" if i == 0 else "⌋" if i == rows - 1 else "│"
        vals   = "  ".join(
            f"{v:{max_len}.{decimals}f}" for v in row
        )
        print(f"{prefix}  {vals}  {suffix}")
    print()


def is_square(m):
    """Check if matrix has equal rows and columns."""
    r, c = shape(m)
    return r == c


def is_symmetric(m):
    """
    A matrix is symmetric if M == M^T.
    Only possible for square matrices.
    """
    if not is_square(m):
        return False
    from .operations import transpose
    return m == transpose(m)


def validate_same_shape(m1, m2, operation="operation"):
    """Raise error if two matrices have different shapes."""
    if shape(m1) != shape(m2):
        raise ValueError(
            f"Shape mismatch for {operation}: "
            f"{shape(m1)} vs {shape(m2)}"
        )