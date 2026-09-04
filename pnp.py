"""Plug-and-Play (PnP) inverse problem solvers built on 1-Lipschitz denoisers.

Why this module exists
----------------------
The denoisers in ``parseval_cnn.py`` are 1-Lipschitz *by construction*
(orthogonal convolutions composed with slope-constrained linear splines). That
is exactly the property PnP convergence theory needs, and it turns an empirical
heuristic into a theorem:

1. The network :math:`R` is nonexpansive: ``||R(x) - R(y)|| <= ||x - y||``.
2. The averaged denoiser :math:`D_\\beta = (1-\\beta) I + \\beta R` is then
   :math:`\\beta`-averaged for :math:`\\beta \\in (0, 1)`
   (see ``layers/averaged.py`` / :class:`AveragedDenoiser`).
3. The gradient step :math:`G_\\tau = I - \\tau \\nabla f` for
   :math:`f(x) = \\tfrac12 \\|Ax - y\\|^2` is :math:`(\\tau L / 2)`-averaged for
   :math:`\\tau \\in (0, 2/L)`, where :math:`L = \\|A\\|_2^2`.
4. A composition of averaged operators is averaged, so
   :math:`D_\\beta \\circ G_\\tau` is averaged and the PnP-PGD iteration
   :math:`x_{k+1} = D_\\beta(G_\\tau(x_k))` is a Krasnosel'skii-Mann iteration:
   it **converges to a fixed point whenever one exists**, for *every*
   :math:`\\tau \\in (0, 2/L)` and *every* :math:`\\beta \\in (0,1)`.

No small-residual assumption is required (contrast Ryu et al., 2019, which
assumes ``R - I`` has small Lipschitz constant). Clamping to ``[0, 1]`` between
steps preserves the guarantee, since projection onto a convex set is firmly
nonexpansive and therefore averaged.

The module provides the pieces needed to *test* that claim: linear operators
with exact adjoints and spectral norms, the PnP algorithms, and diagnostics
(Jacobian spectral norm, fixed-point residual).

Conventions
-----------
Images are float tensors of shape ``(B, 1, H, W)`` with values in ``[0, 1]``.
Every :class:`LinearOperator` implements ``A`` and its exact adjoint ``AT``.
"""

import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def psnr(x, ref, data_range=1.0):
    """Mean PSNR (dB) over the batch. ``x``, ``ref`` are ``(B, C, H, W)``."""
    x = x.detach()
    ref = ref.detach()
    mse = ((x - ref) ** 2).flatten(1).mean(dim=1)
    mse = torch.clamp(mse, min=1e-12)
    return (10.0 * torch.log10(data_range ** 2 / mse)).mean().item()


def ssim(x, ref, data_range=1.0):
    """Mean SSIM over the batch, computed per image.

    Uses the standard 11x11 Gaussian-window formulation (sigma=1.5), matching
    ``skimage``/Wang et al. closely enough for reporting, with no extra
    dependency. Computed per image and then averaged -- note this differs from
    ``utils.utilities.batch_SSIM``, which compares the whole squeezed batch on
    every loop iteration and is only correct for batch size 1.
    """
    x = x.detach()
    ref = ref.detach()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    coords = torch.arange(11, dtype=x.dtype, device=x.device) - 5.0
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    window = (g[:, None] @ g[None, :]).expand(x.shape[1], 1, 11, 11).contiguous()

    def filt(t):
        return F.conv2d(t, window, padding=5, groups=t.shape[1])

    mu_x, mu_ref = filt(x), filt(ref)
    mu_x2, mu_ref2, mu_xref = mu_x ** 2, mu_ref ** 2, mu_x * mu_ref
    sigma_x = filt(x * x) - mu_x2
    sigma_ref = filt(ref * ref) - mu_ref2
    sigma_xref = filt(x * ref) - mu_xref

    num = (2 * mu_xref + c1) * (2 * sigma_xref + c2)
    den = (mu_x2 + mu_ref2 + c1) * (sigma_x + sigma_ref + c2)
    return (num / den).flatten(1).mean(dim=1).mean().item()


# --------------------------------------------------------------------------- #
# Linear operators
# --------------------------------------------------------------------------- #

class LinearOperator:
    """Base class: a linear map ``A`` with an exact adjoint ``AT``."""

    def A(self, x):
        raise NotImplementedError

    def AT(self, y):
        raise NotImplementedError

    def norm(self, shape, device='cpu', n_iter=100, seed=0):
        """Spectral norm ``||A||_2`` by power iteration on ``A^T A``."""
        g = torch.Generator(device='cpu').manual_seed(seed)
        v = torch.randn(*shape, generator=g).to(device)
        v = v / v.norm()
        s = torch.tensor(0.0, device=device)
        for _ in range(n_iter):
            w = self.AT(self.A(v))
            s = w.norm()
            if s < 1e-12:
                return 0.0
            v = w / s
        return float(s.sqrt())

    def lipschitz_grad(self, shape, device='cpu', **kw):
        """``L = ||A||_2^2``, the Lipschitz constant of ``grad f``."""
        return self.norm(shape, device=device, **kw) ** 2


def psf2otf(psf, shape, device='cpu', dtype=torch.float32):
    """Optical transfer function of ``psf`` for circular convolution on ``shape``.

    The PSF is zero-padded to ``shape`` and rolled so that its centre sits at
    index ``(0, 0)``, which makes the convolution zero-phase (no spatial shift).
    """
    kh, kw = psf.shape
    h, w = shape
    pad = torch.zeros(h, w, dtype=dtype, device=device)
    pad[:kh, :kw] = torch.as_tensor(psf, dtype=dtype, device=device)
    pad = torch.roll(pad, shifts=(-(kh // 2), -(kw // 2)), dims=(0, 1))
    return torch.fft.fft2(pad)


def gaussian_kernel(size=9, sigma=1.6):
    """Normalised 2-D Gaussian PSF."""
    c = (size - 1) / 2.0
    ax = np.arange(size) - c
    xx, yy = np.meshgrid(ax, ax, indexing='ij')
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def motion_kernel(length=9, angle=30.0):
    """Simple linear motion-blur PSF."""
    k = np.zeros((length, length), dtype=np.float32)
    c = (length - 1) / 2.0
    rad = math.radians(angle)
    for t in np.linspace(-c, c, length * 8):
        i = int(round(c + t * math.sin(rad)))
        j = int(round(c + t * math.cos(rad)))
        if 0 <= i < length and 0 <= j < length:
            k[i, j] += 1.0
    return k / k.sum()


class Blur(LinearOperator):
    """Circular (periodic) blur, diagonalised by the DFT.

    Circular boundary handling matches the cyclic padding the networks already
    use, and makes the adjoint and spectral norm exact rather than approximate.
    """

    name = 'deblurring'

    def __init__(self, psf, shape, device='cpu'):
        self.otf = psf2otf(psf, shape, device=device)
        self.shape = shape
        self.psf = psf

    def A(self, x):
        return torch.fft.ifft2(torch.fft.fft2(x) * self.otf).real

    def AT(self, y):
        return torch.fft.ifft2(torch.fft.fft2(y) * self.otf.conj()).real

    def norm(self, shape=None, device='cpu', **kw):
        # Exact: the singular values of a circular convolution are |OTF|.
        return float(self.otf.abs().max())


class Inpainting(LinearOperator):
    """Pixel-wise masking. ``A = M .* x`` with ``M`` binary, so ``||A||_2 = 1``."""

    name = 'inpainting'

    def __init__(self, mask):
        self.mask = mask

    def A(self, x):
        return x * self.mask

    def AT(self, y):
        return y * self.mask

    def norm(self, shape=None, device='cpu', **kw):
        return 1.0 if float(self.mask.max()) > 0 else 0.0

    @staticmethod
    def random_mask(shape, keep=0.4, device='cpu', seed=0):
        g = torch.Generator(device='cpu').manual_seed(seed)
        m = (torch.rand(*shape, generator=g) < keep).float()
        return m.to(device)


class SuperResolution(LinearOperator):
    """Anti-alias blur followed by decimation: ``A = S B``."""

    name = 'super-resolution'

    def __init__(self, factor, psf, shape, device='cpu'):
        self.factor = factor
        self.blur = Blur(psf, shape, device=device)

    def A(self, x):
        return self.blur.A(x)[..., ::self.factor, ::self.factor]

    def AT(self, y):
        s = self.factor
        up = torch.zeros(*y.shape[:-2], y.shape[-2] * s, y.shape[-1] * s,
                         dtype=y.dtype, device=y.device)
        up[..., ::s, ::s] = y
        return self.blur.AT(up)


# --------------------------------------------------------------------------- #
# Denoiser wrappers
# --------------------------------------------------------------------------- #

class AveragedDenoiser(nn.Module):
    """``D_beta = (1 - beta) I + beta R``.

    Mirrors ``layers/averaged.py`` but takes ``beta`` as a plain float so it can
    be swept by a hyper-parameter search without touching module state. If ``R``
    is nonexpansive then ``D_beta`` is ``beta``-averaged, which is what makes the
    PnP iteration provably convergent.
    """

    def __init__(self, R, beta=0.5):
        super().__init__()
        if not 0.0 < beta < 1.0:
            raise ValueError('beta must lie in (0, 1)')
        self.R = R
        self.beta = float(beta)

    def forward(self, x):
        return (1.0 - self.beta) * x + self.beta * self.R(x)


@contextmanager
def fast_inference(model):
    """Cache the orthogonalised convolution weights across forward passes.

    The BCOP layers re-run Bjorck orthonormalisation on *every* forward call,
    which dominates runtime when a denoiser is applied hundreds of times inside a
    PnP loop. The repo's ``streamline`` flag caches the generated kernels after
    the first pass; entering the flag also clears the cache, so the weights are
    regenerated exactly once here. Cached weights are detached, which is
    irrelevant for PnP (we never differentiate w.r.t. parameters) and harmless
    for :func:`jacobian_spectral_norm` (which differentiates w.r.t. the input).
    """
    try:
        from layers.BCOP.utils import streamline_model
    except ImportError:
        yield model
        return
    streamline_model(model, True)
    try:
        yield model
    finally:
        streamline_model(model, False)


def _jvp_double_backward(model, x, u):
    """``J u`` using two reverse-mode passes (no forward-mode AD).

    ``torch.func.jvp`` cannot be used here: ``LinearSpline`` is a custom
    ``autograd.Function`` with no forward-mode rule, and calling forward-mode AD
    through it crashes the interpreter. The double-backward identity
    ``d/dw <J^T w, u> = J u`` needs only reverse mode, which the spline does
    implement. Returns ``(J u, out, xg)`` so the caller can reuse the graph.
    """
    xg = x.clone().requires_grad_(True)
    out = model(xg)
    w = torch.zeros_like(out, requires_grad=True)
    g = torch.autograd.grad(out, xg, grad_outputs=w, create_graph=True)[0]
    Ju = torch.autograd.grad(g, w, grad_outputs=u, retain_graph=True)[0]
    return Ju, out, xg


def _jvp_finite_difference(model, x, u, eps=1e-3):
    """``J u`` by a central difference. Fallback when double-backward fails."""
    with torch.no_grad():
        return (model(x + eps * u) - model(x - eps * u)) / (2 * eps)


def jacobian_spectral_norm(model, x, n_iter=30, seed=0, tol=1e-6, method='auto'):
    """Estimate ``||J_model(x)||_2`` by power iteration on ``J^T J``.

    This is the *local* Lipschitz constant at ``x``. For a 1-Lipschitz network it
    must not exceed 1 (up to numerical slack) at any point, so sampling it over
    several inputs is a direct empirical check of the architectural constraint.

    ``method`` selects how ``J u`` is formed: ``'double'`` (exact, two reverse
    passes), ``'fd'`` (central differences), or ``'auto'`` to try the exact route
    and fall back.
    """
    model.eval()
    x = x.detach()
    g = torch.Generator(device='cpu').manual_seed(seed)
    u = torch.randn(x.shape, generator=g).to(x.device)
    u = u / u.norm()

    use_double = method in ('auto', 'double')
    sigma_prev, sigma = 0.0, 0.0
    for _ in range(n_iter):
        if use_double:
            try:
                Ju, out, xg = _jvp_double_backward(model, x, u)
                JtJu = torch.autograd.grad(out, xg, grad_outputs=Ju,
                                           retain_graph=True)[0].detach()
            except Exception:
                if method == 'double':
                    raise
                use_double = False
                continue
        else:
            Ju = _jvp_finite_difference(model, x, u)
            xg = x.clone().requires_grad_(True)
            out = model(xg)
            JtJu = torch.autograd.grad(out, xg, grad_outputs=Ju)[0].detach()

        sigma = float(JtJu.norm())
        if sigma < 1e-12:
            return 0.0
        u = JtJu / sigma
        if abs(sigma - sigma_prev) < tol * max(sigma, 1e-12):
            break
        sigma_prev = sigma
    return math.sqrt(sigma)


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #

def operator_L(op, x_shape, device='cpu'):
    """``L = ||A||_2^2``, cached on the operator (power iteration is not free)."""
    if getattr(op, '_L', None) is None:
        op._L = op.lipschitz_grad(x_shape, device=device)
    return op._L


def _init_x(op, y, init, x_shape):
    if init == 'adjoint':
        return op.AT(y).clone()
    if init == 'observation' and y.shape == x_shape:
        return y.clone()
    if init == 'zeros':
        return torch.zeros(x_shape, dtype=y.dtype, device=y.device)
    return op.AT(y).clone()


def _record(history, x, ref, x_prev, track_psnr):
    res = float((x - x_prev).norm() / max(float(x_prev.norm()), 1e-12))
    history['residual'].append(res)
    if track_psnr and ref is not None:
        history['psnr'].append(psnr(torch.clamp(x, 0, 1), ref))
    return res


def pnp_pgd(op, y, denoiser, tau=None, n_iter=100, x_true=None, init='adjoint',
            x_shape=None, clamp=True, tol=0.0, track_psnr=True, callback=None):
    """PnP proximal-gradient (forward-backward splitting).

    ``x_{k+1} = D(x_k - tau * A^T (A x_k - y))``

    With ``D`` averaged and ``tau in (0, 2/L)`` (``L = ||A||^2``) this is a
    Krasnosel'skii-Mann iteration and converges to a fixed point. ``tau``
    defaults to ``1/L``, comfortably inside the stable range.

    Returns ``(x, history)`` where ``history`` holds per-iteration ``residual``
    (relative step norm, the convergence certificate) and ``psnr``.
    """
    x_shape = x_shape or op.AT(y).shape
    L = operator_L(op, x_shape, device=y.device)
    if tau is None:
        tau = 1.0 / max(L, 1e-12)

    x = _init_x(op, y, init, x_shape)
    if clamp:
        x = x.clamp(0, 1)
    history = {'residual': [], 'psnr': [], 'tau': tau, 'L': L,
               'tau_max': 2.0 / max(L, 1e-12)}

    with torch.no_grad():
        for k in range(n_iter):
            x_prev = x
            grad = op.AT(op.A(x) - y)
            x = denoiser(x - tau * grad)
            if clamp:
                x = x.clamp(0, 1)
            res = _record(history, x, x_true, x_prev, track_psnr)
            if callback is not None:
                callback(k, x, res)
            if tol and res < tol:
                break
    return x, history


def _cg(matvec, b, x0, n_iter=12, tol=1e-8):
    """Conjugate gradient for a symmetric positive-definite ``matvec``."""
    x = x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs = float((r * r).sum())
    for _ in range(n_iter):
        if rs < tol:
            break
        Ap = matvec(p)
        denom = float((p * Ap).sum())
        if abs(denom) < 1e-20:
            break
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float((r * r).sum())
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def pnp_admm(op, y, denoiser, rho=1.0, n_iter=100, x_true=None, init='adjoint',
             x_shape=None, clamp=True, cg_iter=12, tol=0.0, track_psnr=True,
             callback=None):
    """PnP-ADMM.

    Alternates an exact-ish data-consistency prox (solved by CG, so it works for
    any operator) with the denoiser standing in for ``prox_g``.
    """
    x_shape = x_shape or op.AT(y).shape
    x = _init_x(op, y, init, x_shape)
    if clamp:
        x = x.clamp(0, 1)
    z = x.clone()
    u = torch.zeros_like(x)
    ATy = op.AT(y)

    def normal_eq(v):
        return op.AT(op.A(v)) + rho * v

    history = {'residual': [], 'psnr': [], 'rho': rho}
    with torch.no_grad():
        for k in range(n_iter):
            z_prev = z
            x = _cg(normal_eq, ATy + rho * (z - u), x, n_iter=cg_iter)
            z = denoiser(x + u)
            if clamp:
                z = z.clamp(0, 1)
            u = u + x - z
            # Track the denoised iterate against its own previous value, so the
            # residual is a genuine successive-difference on one variable.
            res = _record(history, z, x_true, z_prev, track_psnr)
            if callback is not None:
                callback(k, z, res)
            if tol and res < tol:
                break
    return (z.clamp(0, 1) if clamp else z), history


def pnp_drs(op, y, denoiser, gamma=1.0, n_iter=100, x_true=None, init='adjoint',
            x_shape=None, clamp=True, cg_iter=12, tol=0.0, track_psnr=True,
            callback=None):
    """PnP Douglas-Rachford splitting."""
    x_shape = x_shape or op.AT(y).shape
    z = _init_x(op, y, init, x_shape)
    ATy = op.AT(y)

    def normal_eq(v):
        return gamma * op.AT(op.A(v)) + v

    history = {'residual': [], 'psnr': [], 'gamma': gamma}
    x = z.clone()
    with torch.no_grad():
        for k in range(n_iter):
            x_prev = x
            x_half = _cg(normal_eq, gamma * ATy + z, x, n_iter=cg_iter)
            x = denoiser(2 * x_half - z)
            if clamp:
                x = x.clamp(0, 1)
            z = z + x - x_half
            res = _record(history, x, x_true, x_prev, track_psnr)
            if callback is not None:
                callback(k, x, res)
            if tol and res < tol:
                break
    return x, history


SOLVERS = {'pgd': pnp_pgd, 'admm': pnp_admm, 'drs': pnp_drs}


# --------------------------------------------------------------------------- #
# Problem construction
# --------------------------------------------------------------------------- #

def make_problem(kind, x_true, noise_std=0.01, device='cpu', seed=0, **kw):
    """Build ``(operator, y)`` for a degradation ``kind``.

    ``kind`` is one of ``deblur_gauss``, ``deblur_motion``, ``inpaint``, ``sr2``,
    ``sr4``. ``y = A x_true + n`` with i.i.d. Gaussian noise of std ``noise_std``.
    """
    shape = tuple(x_true.shape[-2:])
    if kind == 'deblur_gauss':
        op = Blur(gaussian_kernel(kw.get('size', 9), kw.get('sigma', 1.6)), shape, device=device)
    elif kind == 'deblur_motion':
        op = Blur(motion_kernel(kw.get('length', 9), kw.get('angle', 30.0)), shape, device=device)
    elif kind == 'inpaint':
        mask = Inpainting.random_mask(x_true.shape, keep=kw.get('keep', 0.4),
                                      device=device, seed=seed)
        op = Inpainting(mask)
    elif kind in ('sr2', 'sr4'):
        f = 2 if kind == 'sr2' else 4
        op = SuperResolution(f, gaussian_kernel(kw.get('size', 9), kw.get('sigma', 1.0 * f)),
                             shape, device=device)
    else:
        raise ValueError(f'unknown problem kind: {kind}')

    clean_y = op.A(x_true)
    g = torch.Generator(device='cpu').manual_seed(seed + 1)
    n = torch.randn(clean_y.shape, generator=g).to(device) * noise_std
    return op, clean_y + n


def solve(op, y, model, solver='pgd', beta=0.5, x_true=None, n_iter=100, **kw):
    """Convenience wrapper: wrap ``model`` as an averaged denoiser and run PnP."""
    D = AveragedDenoiser(model, beta=beta)
    fn = SOLVERS[solver]
    return fn(op, y, D, x_true=x_true, n_iter=n_iter, **kw)


# --------------------------------------------------------------------------- #
# Denoisers: loading the trained Parseval nets, and an unconstrained control
# --------------------------------------------------------------------------- #

# DnCNN now lives in layers/dncnn.py (it is also used, unrelated to PnP, as an
# accuracy baseline registered in parseval_cnn.MODELS). Re-exported here so
# `pnp.DnCNN` keeps working exactly as before -- same class, same defaults.
from layers.dncnn import DnCNN  # noqa: E402


def load_parseval(name, ckpt_path, device='cpu', config=None):
    """Rebuild a Parseval model from a training checkpoint.

    ``ckpt_path`` points at a ``.pth`` written by ``trainer.save_checkpoint``
    (keys ``state_dict`` / ``config``). The architecture is taken from the
    checkpoint's own config when present, so it always matches the weights.
    """
    from parseval_cnn import MODELS

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = config or ckpt.get('config')
    if cfg is None:
        raise ValueError(f'{ckpt_path} has no config; pass config= explicitly')
    model_key = cfg['net_params'].get('model', name)
    model = MODELS[model_key](cfg['net_params'], cfg['activation_params'])
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval()


def find_checkpoints(root='exps', prefer='best.pth'):
    """Discover ``{exp_name: checkpoint_path}`` under an experiments directory."""
    import os

    found = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) != 'checkpoints':
            continue
        pick = prefer if prefer in filenames else (
            'checkpoint.pth' if 'checkpoint.pth' in filenames else None)
        if pick:
            found[os.path.basename(os.path.dirname(dirpath))] = os.path.join(dirpath, pick)
    return dict(sorted(found.items()))


@torch.no_grad()
def denoise_psnr(model, images, sigma, seed=0):
    """Sanity check: PSNR of ``model`` denoising ``images`` at noise level ``sigma``."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(images.shape, generator=g).to(images.device) * sigma
    out = model(images + noise).clamp(0, 1)
    return psnr(out, images)


# --------------------------------------------------------------------------- #
# Hyper-parameter search
# --------------------------------------------------------------------------- #

def evaluate(model, problems, params, solver='pgd', n_iter=100, reduce='mean'):
    """Mean PSNR/SSIM of a parameter setting over a list of ``(op, y, x_true)``."""
    ps, ss = [], []
    for op, y, x_true in problems:
        x, _ = solve(op, y, model, solver=solver, n_iter=n_iter, x_true=x_true,
                     track_psnr=False, **params)
        x = x.clamp(0, 1)
        ps.append(psnr(x, x_true))
        ss.append(ssim(x, x_true))
    if reduce == 'mean':
        return float(np.mean(ps)), float(np.mean(ss))
    return ps, ss


def search_space(trial, solver, L):
    """Sample solver hyper-parameters.

    The PGD step size is parameterised as a *fraction of the theoretically
    stable range* ``(0, 2/L)``, so the search cannot leave the region where
    convergence is guaranteed. ``beta`` is the averaging (and regularisation)
    strength.
    """
    params = {'beta': trial.suggest_float('beta', 0.02, 0.98)}
    if solver == 'pgd':
        frac = trial.suggest_float('tau_frac', 0.05, 0.99)
        params['tau'] = frac * 2.0 / max(L, 1e-12)
    elif solver == 'admm':
        params['rho'] = trial.suggest_float('rho', 1e-2, 1e2, log=True)
    elif solver == 'drs':
        params['gamma'] = trial.suggest_float('gamma', 1e-2, 1e2, log=True)
    return params


def tune(model, problems, solver='pgd', n_trials=30, n_iter=100, seed=0,
         verbose=False):
    """Bayesian (TPE) search for solver hyper-parameters.

    ``problems`` should be a *validation* set, disjoint from whatever is used to
    report final numbers -- tuning and reporting on the same images would
    overstate the results. Falls back to random search if Optuna is missing.
    """
    L = operator_L(problems[0][0], problems[0][2].shape, device=problems[0][2].device)

    try:
        import optuna
        optuna.logging.set_verbosity(
            optuna.logging.INFO if verbose else optuna.logging.WARNING)

        def objective(trial):
            params = search_space(trial, solver, L)
            return evaluate(model, problems, params, solver=solver, n_iter=n_iter)[0]

        study = optuna.create_study(
            direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials)
        best = search_space(optuna.trial.FixedTrial(study.best_params), solver, L)
        return best, study.best_value, study

    except ImportError:
        rng = np.random.default_rng(seed)
        best, best_val = None, -np.inf
        for _ in range(n_trials):
            params = {'beta': float(rng.uniform(0.02, 0.98))}
            if solver == 'pgd':
                params['tau'] = float(rng.uniform(0.05, 0.99)) * 2.0 / max(L, 1e-12)
            elif solver == 'admm':
                params['rho'] = float(10 ** rng.uniform(-2, 2))
            else:
                params['gamma'] = float(10 ** rng.uniform(-2, 2))
            val = evaluate(model, problems, params, solver=solver, n_iter=n_iter)[0]
            if val > best_val:
                best, best_val = params, val
        return best, best_val, None
