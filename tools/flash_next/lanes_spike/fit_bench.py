"""Least-squares fit of the lanes bench: layer-call us = a + b*D + c*m (+ per-column term), per format and pooled."""
import sys

rows = []
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) == 5 and p[0] in ("iq2_s", "iq2_xxs", "iq1_m"):
        rows.append((p[0], int(p[1]), int(p[2]), int(p[3]), float(p[4])))


def solve(A, y):
    # normal equations, tiny dense system
    n = len(A[0])
    M = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)] for i in range(n)]
    v = [sum(A[k][i] * y[k] for k in range(len(A))) for i in range(n)]
    for i in range(n):
        piv = M[i][i]
        for j in range(i + 1, n):
            f = M[j][i] / piv
            for k in range(i, n):
                M[j][k] -= f * M[i][k]
            v[j] -= f * v[i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (v[i] - sum(M[i][k] * x[k] for k in range(i + 1, n))) / M[i][i]
    return x


for fmt in ("iq2_s", "iq2_xxs", "iq1_m", None):
    sel = [r for r in rows if (fmt is None or r[0] == fmt) and r[3] <= 4 and not (r[1] == 1 and 1 <= r[3] <= 2)]
    A = [[1.0, r[2], r[3], r[1]] for r in sel]
    y = [r[4] for r in sel]
    a, b, c, d = solve(A, y)
    resid = [yy - (a + b * r[2] + c * r[3] + d * r[1]) for r, yy in zip(sel, y)]
    rms = (sum(e * e for e in resid) / len(resid)) ** 0.5
    print(f"{fmt or 'pooled':8s} n={len(sel):3d}  us = {a:6.1f} + {b:5.2f}*D + {c:5.1f}*miss + {d:5.1f}*C   rms {rms:5.1f}")
# T = 1 zero-copy rule points
for r in rows:
    if r[1] == 1:
        print("T=1", r)
