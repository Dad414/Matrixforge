# ── main.py ──────────────────────────────────────────────────
# Full demo of matrixforge — finance & macro context.

import matrixforge as mf

print("=" * 50)
print("  matrixforge — demo")
print("=" * 50)

# ── Create matrices ───────────────────────────────────────────
# Asset return matrix — 3 assets, 4 quarters
returns = mf.matrix([
    [0.08,  0.12,  0.06,  0.10],   # Asset A
    [0.05,  0.07,  0.09,  0.04],   # Asset B
    [0.11, -0.02,  0.08,  0.13],   # Asset C
])
mf.display(returns, label="Asset Returns (quarterly)")

# Identity, zeros, ones
mf.display(mf.identity(3),  label="3×3 Identity")
mf.display(mf.zeros(2, 3),  label="2×3 Zeros")
mf.display(mf.ones(2, 3),   label="2×3 Ones")

# ── Operations ────────────────────────────────────────────────
A = mf.matrix([[1, 2], [3, 4]])
B = mf.matrix([[5, 6], [7, 8]])

mf.display(mf.add(A, B),        label="A + B")
mf.display(mf.subtract(A, B),   label="A - B")
mf.display(mf.multiply(A, B),   label="A × B (dot product)")
mf.display(mf.transpose(A),     label="Transpose of A")

print(f"Determinant of A: {mf.determinant(A)}")  # -2.0
print(f"Trace of A:       {mf.trace(A)}")         # 5

# Scale returns by 100 to convert to percentages
pct = mf.scalar_multiply(returns, 100)
mf.display(pct, label="Returns in %", decimals=1)

# ── Transformations ───────────────────────────────────────────
flat = mf.flatten(returns)
print(f"Flattened: {flat}")

reshaped = mf.reshape(A, 1, 4)
mf.display(reshaped, label="A reshaped 2×2 → 1×4")

mf.display(mf.power(A, 3),      label="A³")
mf.display(mf.normalize(A),     label="A normalized [0,1]")

sliced = mf.slice_matrix(returns, 0, 2, 1, 3)
mf.display(sliced, label="Sliced rows 0-1, cols 1-2")

# ── Stats ─────────────────────────────────────────────────────
print("\n── Return statistics ──")
mf.summary(returns)

print(f"Row means (avg return per asset): {mf.row_mean(returns)}")
print(f"Col means (avg return per quarter): {mf.col_mean(returns)}")
print(f"Best quarter per asset: {mf.row_max(returns)}")
print(f"Worst quarter per asset: {mf.row_min(returns)}")

# Correlation matrix — how assets move together
corr = mf.correlation(returns)
mf.display(corr, label="Asset Correlation Matrix")

# ── Utils ─────────────────────────────────────────────────────
rand = mf.random_matrix(3, 3, low=1, high=100)
mf.display(rand, label="Random 3×3 Matrix")

csv = mf.to_csv_string(A)
print(f"CSV export:\n{csv}")

restored = mf.from_csv_string(csv)
print(f"Restored from CSV — equal: {mf.are_equal(A, restored)}")