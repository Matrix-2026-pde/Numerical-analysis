"""
Numerical experiment: accelerating the Jacobi iteration.

On the *same* 5x5 strictly diagonally dominant system used in ``jacobi.py``,
we compare, under identical settings (x0 = 0, tol = 1e-6, max_iter = 1000,
infinity-norm successive-difference stopping rule):

  1. classical Jacobi            (baseline)
  2. damped / weighted Jacobi
  3. Gauss-Seidel
  4. SOR (successive over-relaxation)
  5. Anderson-accelerated Jacobi (a simplified Alternating Anderson-Jacobi,
     after Pratapa, Suryanarayana & Pask 2015)

Everything is implemented from scratch with NumPy; ``numpy.linalg.solve`` is
used only at the very end to produce the reference solution for accuracy
comparison (never inside the iterations).
"""

import numpy as np
import matplotlib.pyplot as plt

from jacobi import construct_matrix


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def spectral_radius(M):
    """Spectral radius rho(M) = max_i |lambda_i(M)|."""
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def relative_error(x, x_ref):
    return float(np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref))


def splitting(A):
    """Return D, L, U such that A = D - L - U (Golub--Van Loan convention),
    where L and U are the strictly lower / upper triangular parts."""
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    return D, L, U


def solve_fixed_point(B, c, x0, tol=1e-6, max_iter=1000):
    """Generic linear fixed-point iteration x^{k+1} = B x^{k} + c.

    Returns (x, errors, iters) with errors = ||x^(k+1) - x^(k)||_inf.
    """
    x = np.array(x0, dtype=float)
    errors = []
    for k in range(max_iter):
        x_new = B @ x + c
        err = float(np.linalg.norm(x_new - x, np.inf))
        errors.append(err)
        x = x_new
        if err < tol:
            return x, np.array(errors), k + 1
    return x, np.array(errors), max_iter


# ---------------------------------------------------------------------------
# iteration matrices  (x^{k+1} = B x^{k} + c)
# ---------------------------------------------------------------------------

def jacobi_matrices(A, b):
    D, L, U = splitting(A)
    B = np.linalg.inv(D) @ (L + U)        # = I - D^{-1} A
    c = np.linalg.inv(D) @ b
    return B, c


def damped_jacobi_matrices(A, b, omega):
    D, L, U = splitting(A)
    B_J = np.linalg.inv(D) @ (L + U)
    B = (1.0 - omega) * np.eye(A.shape[0]) + omega * B_J
    c = omega * np.linalg.inv(D) @ b
    return B, c


def gauss_seidel_matrices(A, b):
    D, L, U = splitting(A)
    M = D - L                              # lower-triangular part
    B = np.linalg.inv(M) @ U
    c = np.linalg.inv(M) @ b
    return B, c


def sor_matrices(A, b, omega):
    D, L, U = splitting(A)
    M = D - omega * L
    B = np.linalg.inv(M) @ ((1.0 - omega) * D + omega * U)
    c = omega * np.linalg.inv(M) @ b
    return B, c


def jacobi_anderson(A, b, x0, tol=1e-6, max_iter=1000, depth=3, beta=1.0):
    """Anderson-accelerated Jacobi iteration (Walker--Ni form, mixing beta).

    Applies the Anderson extrapolation to the Jacobi fixed-point map
        G(x) = D^{-1}(b - (L+U) x)
    using the last ``depth`` iterate/residual differences.  This is a
    simplified realisation of the Alternating Anderson--Jacobi idea of
    Pratapa et al. (2015), suitable for a small dense system.
    """
    d = np.diag(A)                    # 1D diagonal entries a_ii
    R = A - np.diag(d)                # off-diagonal part

    def G(x):
        return (b - R @ x) / d        # elementwise division by a_ii

    x = np.array(x0, dtype=float)
    f = G(x) - x                    # residual of the fixed-point map
    dX, dF = [], []                 # histories of x- and f-differences
    errors = []
    for k in range(max_iter):
        m_k = min(depth, len(dF))
        if m_k == 0:
            x_new = G(x)            # plain Jacobi step on the first pass
        else:
            Fmat = np.column_stack(dF[-m_k:])
            Xmat = np.column_stack(dX[-m_k:])
            gamma, *_ = np.linalg.lstsq(Fmat, f, rcond=None)
            x_new = x + beta * f - (Xmat + beta * Fmat) @ gamma
        f_new = G(x_new) - x_new
        err = float(np.linalg.norm(x_new - x, np.inf))
        errors.append(err)
        dX.append(x_new - x)
        dF.append(f_new - f)
        x, f = x_new, f_new
        if err < tol:
            return x, np.array(errors), k + 1
    return x, np.array(errors), max_iter


# ---------------------------------------------------------------------------
# main experiment
# ---------------------------------------------------------------------------

def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    plt.rcParams['font.family'] = 'DejaVu Sans'   # clean U+2212 in log ticks

    A, b, x_true = construct_matrix()
    n = A.shape[0]
    x0 = np.zeros(n)
    tol, max_iter = 1e-6, 1000
    x_ref = np.linalg.solve(A, b)

    print("=" * 78)
    print("Acceleration of the Jacobi iteration (5x5 strictly diagonally dominant)")
    print("=" * 78)

    # ---- baseline Jacobi ----
    B_j, c_j = jacobi_matrices(A, b)
    rho_j = spectral_radius(B_j)
    x_j, err_j, it_j = solve_fixed_point(B_j, c_j, x0, tol, max_iter)
    print(f"[baseline] Jacobi            rho={rho_j:.4f}  "
          f"iters={it_j:3d}  rel_err={relative_error(x_j, x_ref):.2e}")

    # ---- damped (weighted) Jacobi: scan omega ----
    omegas = np.linspace(0.05, 1.95, 39)
    damp_rho, damp_iters = [], []
    for w in omegas:
        B, c = damped_jacobi_matrices(A, b, w)
        damp_rho.append(spectral_radius(B))
        damp_iters.append(solve_fixed_point(B, c, x0, tol, max_iter)[2])
    i_damp = int(np.argmin(damp_iters))
    w_damp = omegas[i_damp]
    B_d, c_d = damped_jacobi_matrices(A, b, w_damp)
    x_d, err_d, it_d = solve_fixed_point(B_d, c_d, x0, tol, max_iter)
    print(f"[damped ] Jacobi (omega={w_damp:.2f})  rho={spectral_radius(B_d):.4f}  "
          f"iters={it_d:3d}  rel_err={relative_error(x_d, x_ref):.2e}")

    # ---- Gauss-Seidel ----
    B_gs, c_gs = gauss_seidel_matrices(A, b)
    rho_gs = spectral_radius(B_gs)
    x_gs, err_gs, it_gs = solve_fixed_point(B_gs, c_gs, x0, tol, max_iter)
    print(f"[GS     ] Gauss-Seidel       rho={rho_gs:.4f}  "
          f"iters={it_gs:3d}  rel_err={relative_error(x_gs, x_ref):.2e}")

    # ---- SOR: scan omega in (0,2) ----
    om_sor = np.linspace(0.05, 1.95, 39)
    sor_rho, sor_iters = [], []
    for w in om_sor:
        B, c = sor_matrices(A, b, w)
        sor_rho.append(spectral_radius(B))
        sor_iters.append(solve_fixed_point(B, c, x0, tol, max_iter)[2])
    i_sor = int(np.argmin(sor_iters))
    w_sor = om_sor[i_sor]
    B_s, c_s = sor_matrices(A, b, w_sor)
    x_s, err_s, it_s = solve_fixed_point(B_s, c_s, x0, tol, max_iter)
    # Young's theoretical optimal omega (valid only for consistently-ordered A)
    w_young = 2.0 / (1.0 + np.sqrt(1.0 - rho_j ** 2))
    print(f"[SOR    ] (omega={w_sor:.2f})  rho={spectral_radius(B_s):.4f}  "
          f"iters={it_s:3d}  rel_err={relative_error(x_s, x_ref):.2e}")
    print(f"          Young's formula omega* = {w_young:.4f} "
          f"(applies only to consistently-ordered matrices)")

    # ---- Anderson-accelerated Jacobi: try several depths ----
    aa_iters = {}
    for d in (1, 2, 3, 4, 5):
        x_a, err_a, it_a = jacobi_anderson(A, b, x0, tol, max_iter, depth=d)
        aa_iters[d] = (it_a, relative_error(x_a, x_ref), x_a, err_a)
    d_best = min(aa_iters, key=lambda d: aa_iters[d][0])
    it_a, rel_a, x_a, err_a = aa_iters[d_best]
    print(f"[AAJ    ] Anderson(depth={d_best})   "
          f"iters={it_a:3d}  rel_err={rel_a:.2e}")

    print("-" * 78)
    print(f"speed-up vs Jacobi (iters):  damped={it_j/it_d:.2f}x  "
          f"GS={it_j/it_gs:.2f}x  SOR={it_j/it_s:.2f}x  AAJ={it_j/it_a:.2f}x")
    print("-" * 78)

    # ---- convergence-curve comparison plot ----
    plt.figure(figsize=(8, 5.5))
    plt.semilogy(np.arange(1, len(err_j) + 1), err_j, 'o-', markersize=4,
                 label=f'Jacobi (iters={it_j})')
    plt.semilogy(np.arange(1, len(err_d) + 1), err_d, 's-', markersize=4,
                 label=f'Damped Jacobi $\\omega$={w_damp:.2f} (iters={it_d})')
    plt.semilogy(np.arange(1, len(err_gs) + 1), err_gs, '^-', markersize=4,
                 label=f'Gauss-Seidel (iters={it_gs})')
    plt.semilogy(np.arange(1, len(err_s) + 1), err_s, 'v-', markersize=4,
                 label=f'SOR $\\omega$={w_sor:.2f} (iters={it_s})')
    plt.semilogy(np.arange(1, len(err_a) + 1), err_a, 'D-', markersize=4,
                 label=f'Anderson-Jacobi depth={d_best} (iters={it_a})')
    plt.xlabel('iteration number k')
    plt.ylabel(r'$||x^{(k+1)} - x^{(k)}||_\infty$  (log scale)')
    plt.title('Convergence of Jacobi accelerations (5x5 strictly diagonally dominant)')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('jacobi_acceleration_comparison.png', dpi=150)
    print("Saved 'jacobi_acceleration_comparison.png'")

    # ---- SOR omega-scan plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(om_sor, sor_rho, 'o-', markersize=4)
    ax1.axvline(w_sor, color='r', ls='--', label=f'numeric $\\omega^*$={w_sor:.2f}')
    ax1.axvline(w_young, color='g', ls=':', label=f'Young $\\omega$={w_young:.2f}')
    ax1.set_xlabel('relaxation parameter $\\omega$')
    ax1.set_ylabel('spectral radius $\\rho(B_{SOR}(\\omega))$')
    ax1.set_title('SOR: spectral radius vs $\\omega$')
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.legend()
    ax2.plot(om_sor, sor_iters, 's-', markersize=4)
    ax2.axvline(w_sor, color='r', ls='--', label=f'numeric $\\omega^*$={w_sor:.2f}')
    ax2.set_xlabel('relaxation parameter $\\omega$')
    ax2.set_ylabel('iterations to reach tol')
    ax2.set_title('SOR: iterations vs $\\omega$')
    ax2.grid(True, ls='--', alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    plt.savefig('sor_omega_scan.png', dpi=150)
    print("Saved 'sor_omega_scan.png'")


if __name__ == '__main__':
    main()
