"""
Numerical Analysis: LU decomposition with and without partial pivoting.

We construct a 5x5 matrix that is *well conditioned* (cond(A) is small) but
whose first pivot A[0,0] = eps is deliberately tiny.  Gaussian elimination
WITHOUT pivoting therefore produces a huge multiplier l_21 = 1/eps and
catastrophic error, whereas partial pivoting (row swapping) fixes it.

Note: no scipy.linalg.lu is used anywhere -- both algorithms are implemented
from scratch with pure Python loops + NumPy.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. LU decomposition (Doolittle form), pure Python + NumPy
# ---------------------------------------------------------------------------

def lu_no_pivot(A):
    """Doolittle LU WITHOUT pivoting:  A = L @ U,  L has unit diagonal."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for k in range(n):
        if abs(U[k, k]) < 1e-14:
            print(f"    [warning] near-zero pivot U[{k},{k}] = {U[k, k]:.3e}")
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]        # multiplier l_{ik} = u_{ik}/u_{kk}
            U[i, k:] -= L[i, k] * U[k, k:]     # row_i <- row_i - l_{ik} * row_k
            U[i, k] = 0.0
    return L, U


def lu_partial_pivot(A):
    """Doolittle LU WITH partial pivoting:  P @ A = L @ U.  Returns (P, L, U)."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    for k in range(n):
        # row of the largest (in absolute value) pivot in column k
        p = int(np.argmax(np.abs(U[k:, k]))) + k
        if p != k:
            U[[k, p], :] = U[[p, k], :]        # swap rows of U
            P[[k, p], :] = P[[p, k], :]        # swap rows of P
            L[[k, p], :k] = L[[p, k], :k]      # swap already-stored multipliers
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0
    return P, L, U


def forward_sub(L, b):
    """Solve L y = b where L is unit lower-triangular (forward substitution)."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y


def back_sub(U, y):
    """Solve U x = y where U is upper-triangular (back substitution)."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def solve_lu(A, b, pivot=False):
    """Solve A x = b by LU.  pivot=False -> A = L U ; pivot=True -> P A = L U."""
    if pivot:
        P, L, U = lu_partial_pivot(A)
        y = forward_sub(L, P @ b)      # P A x = P b  =>  L U x = P b
        x = back_sub(U, y)
        return x, P, L, U
    else:
        L, U = lu_no_pivot(A)
        y = forward_sub(L, b)          # A x = L U x = b
        x = back_sub(U, y)
        return x, None, L, U


def relative_error(x, x_ref):
    """||x - x_ref||_2 / ||x_ref||_2."""
    return np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)


def build_matrix(n, eps=1e-12):
    """n x n tridiagonal matrix: 2 on the diagonal, 1 on the off-diagonals,
    but with a deliberately tiny first pivot A[0,0] = eps."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2.0
        if i + 1 < n:
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
    A[0, 0] = eps
    return A


# ---------------------------------------------------------------------------
# 2. Demo on the 5x5 matrix
# ---------------------------------------------------------------------------

def demo_5x5():
    eps = 1e-12
    n = 5
    A = build_matrix(n, eps)
    x_true = np.ones(n)                 # known exact solution
    b = A @ x_true                      # so that x = (1,1,1,1,1)^T

    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    print("=" * 70)
    print(f"5x5 demo matrix  (eps = {eps:.0e})")
    print("=" * 70)
    print("A =\n", A)
    print("b =", b)
    print(f"cond(A) = {np.linalg.cond(A):.4e}   <-- well conditioned, yet no-pivot LU fails!")

    x_ref = np.linalg.solve(A, b)       # reference solution (uses partial pivoting)
    print("\nreference x (numpy.linalg.solve) =", x_ref)

    print("\n--- (a) LU WITHOUT pivoting ---")
    x_np, _, L_np, U_np = solve_lu(A, b, pivot=False)
    print("L =\n", L_np)
    print("U =\n", U_np)
    print("x =", x_np)
    print(f"relative error vs numpy.linalg.solve = {relative_error(x_np, x_ref):.4e}")

    print("\n--- (b) LU WITH partial pivoting ---")
    x_pp, P, L_pp, U_pp = solve_lu(A, b, pivot=True)
    print("P =\n", P)
    print("L =\n", L_pp)
    print("U =\n", U_pp)
    print("x =", x_pp)
    print(f"relative error vs numpy.linalg.solve = {relative_error(x_pp, x_ref):.4e}")


# ---------------------------------------------------------------------------
# 3. Bonus: relative error vs matrix size n (5 .. 100)
# ---------------------------------------------------------------------------

def convergence_plot():
    ns = list(range(5, 101))
    err_no = []
    err_pp = []
    for n in ns:
        A = build_matrix(n)
        b = A @ np.ones(n)
        x_ref = np.linalg.solve(A, b)
        x_np, *_ = solve_lu(A, b, pivot=False)
        x_pp, *_ = solve_lu(A, b, pivot=True)
        err_no.append(relative_error(x_np, x_ref))
        err_pp.append(relative_error(x_pp, x_ref))

    plt.figure(figsize=(8, 5))
    plt.semilogy(ns, err_no, 'o-', markersize=4, label='LU without pivoting')
    plt.semilogy(ns, err_pp, 's-', markersize=4, label='LU with partial pivoting')
    plt.xlabel('matrix size n')
    plt.ylabel('relative error (log scale)')
    plt.title('Relative error vs n: no-pivot LU is unstable')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('lu_error_vs_n.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    demo_5x5()
    convergence_plot()
