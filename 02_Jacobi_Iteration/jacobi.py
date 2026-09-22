"""
Numerical Analysis: Jacobi iteration for solving a linear system A x = b.

We construct a 5x5 *strictly diagonally dominant* matrix A
(|a_ii| > sum_{j != i} |a_ij| for every row i) so that the Jacobi method is
guaranteed to converge.  The right-hand side b is built from a known exact
solution so that we can measure the accuracy of the iterative solution.

Note: no scipy.linalg or numpy.linalg.solve is used *inside* the iteration --
numpy.linalg.solve is only used afterwards to produce a reference solution
for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Matrix / right-hand-side construction
# ---------------------------------------------------------------------------

def construct_matrix():
    """Return a 5x5 strictly diagonally dominant matrix A and a right-hand
    side b built from the known exact solution x = (1, 2, 3, 4, 5)^T."""
    n = 5
    A = np.array([
        [10.0,  1.0, -1.0,  2.0,  0.0],
        [ 1.0, 12.0,  3.0, -1.0,  1.0],
        [-2.0,  1.0, 15.0,  2.0, -3.0],
        [ 1.0, -2.0,  1.0, 11.0,  2.0],
        [ 0.0,  2.0, -1.0,  3.0, 13.0],
    ], dtype=float)

    x_true = np.arange(1.0, n + 1.0)          # known exact solution
    b = A @ x_true                            # A x = b with x = (1,...,5)
    return A, b, x_true


def check_diagonal_dominance(A):
    """Verify (and print) strict diagonal dominance row by row.

    Returns True iff |a_ii| > sum_{j != i} |a_ij| for every row i.
    """
    n = A.shape[0]
    print("Diagonal dominance check (strict: |a_ii| > sum of |off-diagonal|):")
    print(f"{'row':>3} | {'|a_ii|':>8} | {'sum_j!=i |a_ij|':>17} | strict?")
    print("-" * 48)
    strictly_dominant = True
    for i in range(n):
        diag = abs(A[i, i])
        off_diag_sum = np.sum(np.abs(A[i, :])) - diag
        ok = diag > off_diag_sum
        strictly_dominant = strictly_dominant and ok
        print(f"{i:>3} | {diag:>8.3f} | {off_diag_sum:>17.3f} | {ok}")
    print("-" * 48)
    print(f"A is {'strictly' if strictly_dominant else 'NOT'} diagonally dominant.\n")
    return strictly_dominant


# ---------------------------------------------------------------------------
# 2. Jacobi iteration
# ---------------------------------------------------------------------------

def jacobi_solver(A, b, x0, tol=1e-6, max_iter=1000):
    """Solve A x = b by Jacobi iteration.

    Iteration (component form):
        x_i^{(k+1)} = ( b_i - sum_{j != i} a_ij x_j^{(k)} ) / a_ii

    which is vectorised here as
        x_new = (b - R @ x_old) / diag(A),   R = A - diag(diag(A)),
    since the two forms are mathematically identical (the diagonal term
    a_ii x_i is exactly the one excluded from the sum in the component form).

    Returns (x, errors, iters) where errors is the array of successive
    infinity-norm differences ||x_new - x_old||_inf.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)

    diag = np.diag(A)                 # diagonal entries a_ii
    R = A - np.diag(diag)             # off-diagonal part (diagonal set to 0)
    # NB: numpy's (b - R @ x) / diag automatically divides each row by a_ii,
    # matching x_i = (b_i - sum_{j!=i} a_ij x_j) / a_ii.

    errors = []
    for k in range(max_iter):
        x_new = (b - R @ x) / diag
        err = np.linalg.norm(x_new - x, ord=np.inf)   # infinity-norm
        errors.append(err)
        x = x_new
        if err < tol:
            return x, np.array(errors), k + 1

    # Did not converge within max_iter
    return x, np.array(errors), max_iter


# ---------------------------------------------------------------------------
# 3. Convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(errors):
    """Plot the infinity-norm error vs iteration number on a semilog axis."""
    # DejaVu Sans is used so that the log-axis tick labels (10^-k) render the
    # U+2212 minus glyph correctly on Windows without font-substitution warnings.
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.figure(figsize=(8, 5))
    plt.semilogy(np.arange(1, len(errors) + 1), errors, 'o-', markersize=4)
    plt.xlabel('iteration number k')
    plt.ylabel(r'$||x^{(k+1)} - x^{(k)}||_\infty$  (log scale)')
    plt.title('Jacobi iteration convergence (strictly diagonally dominant A)')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('jacobi_convergence.png', dpi=150)
    print("Saved convergence plot to 'jacobi_convergence.png'")


# ---------------------------------------------------------------------------
# 4. Main demo
# ---------------------------------------------------------------------------

def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=120)

    A, b, x_true = construct_matrix()
    print("=" * 70)
    print("Jacobi iteration on a 5x5 strictly diagonally dominant system")
    print("=" * 70)
    print("A =\n", A)
    print("b =", b)
    print("known exact solution x_true =", x_true)

    check_diagonal_dominance(A)

    x0 = np.zeros(A.shape[0])                     # initial guess
    tol = 1e-6
    max_iter = 1000
    x, errors, iters = jacobi_solver(A, b, x0, tol=tol, max_iter=max_iter)

    print(f"Converged in {iters} iterations "
          f"(final ||x^(k+1)-x^(k)||_inf = {errors[-1]:.3e} < {tol:.0e}).\n")

    x_ref = np.linalg.solve(A, b)                 # reference (direct) solution
    rel_err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)

    print("--- Jacobi approximate solution ---")
    print("x_jacobi =", x)
    print("\n--- numpy.linalg.solve (direct) reference ---")
    print("x_ref    =", x_ref)
    print(f"\nrelative error ||x - x_ref||_2 / ||x_ref||_2 = {rel_err:.3e}")

    # Convergence-rate analysis: the Jacobi iteration matrix is
    #     B = D^{-1}(L + U) = D^{-1}(D - A),
    # and its spectral radius rho(B) is the asymptotic per-iteration factor.
    D = np.diag(np.diag(A))
    B = np.linalg.inv(D) @ (D - A)
    rho = np.max(np.abs(np.linalg.eigvals(B)))
    print("\n--- convergence-rate analysis ---")
    print(f"rho(B)      = {rho:.4f}   (asymptotic error factor per iteration)")
    print(f"||B||_inf   = {np.linalg.norm(B, ord=np.inf):.4f}   "
          f"(a-priori strict-diagonal-dominance bound, must be < 1)")

    plot_convergence(errors)


if __name__ == '__main__':
    main()
