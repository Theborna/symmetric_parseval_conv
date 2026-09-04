"""Correctness tests for ``pnp.py``.

Run with ``python test_pnp.py`` (no pytest needed). Everything here runs on CPU
in well under a minute and checks the parts that are easy to get silently wrong:
exact adjoints, spectral norms, the solvers against closed-form solutions, the
Jacobian estimator against known linear maps, and the architectural 1-Lipschitz
property of the Parseval models.
"""

import numpy as np
import torch

import pnp

FAILURES = []


def check(name, cond, extra=''):
    status = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILURES.append(name)
    print(f'{status}  {name} {extra}')


def test_adjoints():
    print('\n== exact adjoints:  <Ax, y> == <x, A^T y> ==')
    shape = (64, 64)
    ops = [
        ('Blur/gaussian', pnp.Blur(pnp.gaussian_kernel(9, 1.6), shape)),
        ('Blur/motion', pnp.Blur(pnp.motion_kernel(9, 30.0), shape)),
        ('Inpainting', pnp.Inpainting(pnp.Inpainting.random_mask((1, 1, 64, 64), 0.4))),
        ('SuperRes x2', pnp.SuperResolution(2, pnp.gaussian_kernel(9, 2.0), shape)),
        ('SuperRes x4', pnp.SuperResolution(4, pnp.gaussian_kernel(9, 4.0), shape)),
    ]
    for name, op in ops:
        u = torch.rand(1, 1, *shape)
        v = torch.randn_like(op.A(u))
        lhs = float((op.A(u) * v).sum())
        rhs = float((u * op.AT(v)).sum())
        rel = abs(lhs - rhs) / max(abs(lhs), 1e-12)
        check(f'adjoint {name}', rel < 1e-5, f'(rel err {rel:.2e})')


def test_spectral_norms():
    print('\n== spectral norms ==')
    b = pnp.Blur(pnp.gaussian_kernel(9, 1.6), (64, 64))
    analytic = b.norm()
    power = pnp.LinearOperator.norm(b, (1, 1, 64, 64), n_iter=200)
    check('blur: analytic == power iteration', abs(analytic - power) / analytic < 1e-4,
          f'({analytic:.6f} vs {power:.6f})')
    check('blur: ||A|| <= 1 (low-pass)', analytic <= 1 + 1e-6, f'({analytic:.6f})')
    m = pnp.Inpainting(pnp.Inpainting.random_mask((1, 1, 32, 32), 0.5))
    check('inpainting: ||A|| == 1', abs(m.norm() - 1.0) < 1e-9)


def test_averaged_operator():
    print('\n== averaged denoiser ==')

    class Ortho(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q, _ = torch.linalg.qr(torch.randn(16, 16))

        def forward(self, z):
            return (z.reshape(-1, 16) @ self.q.T).reshape(z.shape)

    R = Ortho()
    a, b = torch.rand(4, 16), torch.rand(4, 16)
    lip = float((R(a) - R(b)).norm() / (a - b).norm())
    check('R is exactly 1-Lipschitz', abs(lip - 1) < 1e-4, f'({lip:.6f})')
    D = pnp.AveragedDenoiser(R, beta=0.5)
    lip_d = float((D(a) - D(b)).norm() / (a - b).norm())
    check('D_beta is nonexpansive', lip_d <= 1 + 1e-5, f'({lip_d:.6f})')


def test_jacobian_estimator():
    print('\n== Jacobian spectral norm on known linear maps ==')

    class Scale(torch.nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c

        def forward(self, z):
            return self.c * z

    for c in (0.5, 1.0, 2.0):
        est = pnp.jacobian_spectral_norm(Scale(c), torch.rand(1, 1, 16, 16), n_iter=30)
        check(f'||J|| of {c}*I', abs(est - c) < 1e-3, f'(est {est:.6f})')


def test_solvers_against_closed_form():
    print('\n== solvers vs closed-form solutions ==')
    torch.manual_seed(0)
    x_true = torch.rand(1, 1, 64, 64)
    op, y = pnp.make_problem('deblur_gauss', x_true, noise_std=0.0, seed=0)

    # CG data-consistency prox must match the direct Fourier solve.
    rho = 0.5
    H, Y = op.otf, torch.fft.fft2(y)
    x_direct = torch.fft.ifft2((H.conj() * Y) / (H.abs() ** 2 + rho)).real
    ATy = op.AT(y)
    x_cg = pnp._cg(lambda v: op.AT(op.A(v)) + rho * v, ATy, torch.zeros_like(ATy), n_iter=300)
    rel = float((x_cg - x_direct).norm() / x_direct.norm())
    check('CG prox == direct Fourier solve', rel < 1e-4, f'(rel err {rel:.2e})')

    # With D = I, PnP-PGD is gradient descent on 0.5||Ax-y||^2 and must reach the
    # least-squares solution -- but only where gradient descent can actually make
    # progress. The per-frequency contraction factor is 1 - tau|H|^2, so the
    # near-null-space frequencies converge arbitrarily slowly; restrict to the
    # well-conditioned band rather than pretend the whole spectrum has converged.
    xr, _ = pnp.solve(op, y, torch.nn.Identity(), solver='pgd', beta=0.5,
                      n_iter=4000, clamp=False, track_psnr=False)
    X_ls = torch.where(H.abs() > 1e-9, H.conj() * Y / (H.abs() ** 2 + 1e-30),
                       torch.zeros_like(Y))
    band = H.abs() >= 0.2
    rel = float((torch.fft.fft2(xr) - X_ls).abs()[..., band].norm()
                / X_ls.abs()[..., band].norm())
    check('PGD(D=I) == least-squares on |H|>=0.2', rel < 1e-3, f'(rel err {rel:.2e})')


def test_convergence_theory():
    print('\n== convergence theory ==')
    torch.manual_seed(0)
    x_true = torch.rand(1, 1, 64, 64) * 0.6 + 0.2
    op, y = pnp.make_problem('deblur_gauss', x_true, noise_std=0.01, seed=0)
    tau_max = 2.0 / pnp.operator_L(op, x_true.shape)

    class Ortho(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q, _ = torch.linalg.qr(torch.randn(16, 16))

        def forward(self, z):
            return (z.reshape(-1, 16) @ self.q.T).reshape(z.shape)

    for frac in (0.1, 0.3, 0.6, 0.9, 0.99):
        _, h = pnp.solve(op, y, Ortho(), solver='pgd', beta=0.5, tau=frac * tau_max,
                         n_iter=200, track_psnr=False)
        r = h['residual'][-1]
        check(f'nonexpansive D converges at tau/tau_max={frac}',
              np.isfinite(r) and r < 1e-3, f'(residual {r:.2e})')

    # An expansive denoiser has no guarantee -- and without clamping the
    # divergence is visible (clamping would bound it by saturation instead).
    class Expansive(torch.nn.Module):
        def forward(self, z):
            return 1.6 * z

    diverged = False
    for frac in (0.3, 0.9):
        _, h = pnp.solve(op, y, Expansive(), solver='pgd', beta=0.9, tau=frac * tau_max,
                         n_iter=150, clamp=False, track_psnr=False)
        r = h['residual'][-1]
        diverged |= (not np.isfinite(r)) or r > 1e2
    check('expansive D diverges (unclamped)', diverged)


def test_parseval_models_are_nonexpansive():
    print('\n== Parseval models: architectural 1-Lipschitz property ==')
    from parseval_cnn import MODELS

    torch.manual_seed(0)
    act = {'spline_size': 21, 'spline_range': 0.1, 'lmbda': 1e-6, 'entro': 0.1}
    x = torch.rand(1, 1, 32, 32)
    # 'dncnn' is deliberately excluded: it is registered in MODELS as an
    # unconstrained accuracy baseline (see parseval_cnn.DnCNNBaseline), not as
    # part of the 1-Lipschitz family this check verifies.
    for key in MODELS:
        if key == 'dncnn':
            continue
        depth = 2 if 'mirror' in key else 4
        net = MODELS[key]({'depth': depth, 'nb_channels': 16, 'kernel_size': 3,
                           'bias': False, 'model': key}, act).eval()
        with pnp.fast_inference(net):
            s = pnp.jacobian_spectral_norm(net, x, n_iter=25)
        check(f'{key}: ||J|| <= 1', s <= 1.02, f'({s:.4f})')

    ctrl = pnp.DnCNN(depth=5, channels=16).eval()
    for p in ctrl.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, 0, 0.25)
    s = pnp.jacobian_spectral_norm(ctrl, x, n_iter=25)
    check('unconstrained control: ||J|| > 1', s > 1.0, f'({s:.3f})')


def test_problems_end_to_end():
    print('\n== every problem kind runs end to end ==')
    from parseval_cnn import MODELS

    torch.manual_seed(0)
    act = {'spline_size': 21, 'spline_range': 0.1, 'lmbda': 1e-6, 'entro': 0.1}
    net = MODELS['symmetric_mirror']({'depth': 2, 'nb_channels': 16, 'kernel_size': 3,
                                      'bias': False, 'model': 'symmetric_mirror'}, act).eval()
    x = torch.rand(1, 1, 32, 32) * 0.6 + 0.2
    for kind in ('deblur_gauss', 'deblur_motion', 'inpaint', 'sr2', 'sr4'):
        op, y = pnp.make_problem(kind, x, noise_std=0.01, seed=0)
        with pnp.fast_inference(net):
            xr, h = pnp.solve(op, y, net, solver='pgd', beta=0.5, n_iter=20, x_true=x)
        check(f'{kind}', np.isfinite(h['residual'][-1]) and xr.shape == x.shape,
              f"(residual {h['residual'][-1]:.1e})")

    for solver in ('pgd', 'admm', 'drs'):
        op, y = pnp.make_problem('deblur_gauss', x, noise_std=0.01, seed=0)
        with pnp.fast_inference(net):
            xr, h = pnp.solve(op, y, net, solver=solver, n_iter=15, x_true=x)
        check(f'solver {solver}', np.isfinite(h['residual'][-1]))


def test_metrics():
    print('\n== metrics ==')
    a = torch.rand(2, 1, 32, 32)
    check('psnr(x, x) is large', pnp.psnr(a, a) > 100)
    check('ssim(x, x) == 1', abs(pnp.ssim(a, a) - 1) < 1e-4, f'({pnp.ssim(a, a):.6f})')
    b = a.clone()
    b[:, :, :16] = 0
    check('ssim degrades on corruption', pnp.ssim(b, a) < 0.9, f'({pnp.ssim(b, a):.4f})')


if __name__ == '__main__':
    test_adjoints()
    test_spectral_norms()
    test_averaged_operator()
    test_jacobian_estimator()
    test_metrics()
    test_solvers_against_closed_form()
    test_convergence_theory()
    test_parseval_models_are_nonexpansive()
    test_problems_end_to_end()

    print('\n' + '=' * 60)
    if FAILURES:
        print(f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES))
        raise SystemExit(1)
    print('ALL TESTS PASSED')
