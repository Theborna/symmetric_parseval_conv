"""Train every core model at several noise levels and tabulate the results.

This is the experiment driver for the paper's denoising table. It trains each
model in ``parseval_cnn.MODELS`` at each requested noise level ``sigma`` (default
5, 15, 25), records the validation PSNR/SSIM, and writes:

    <output-dir>/results.json   raw metrics for every (model, sigma) run
    <output-dir>/results.md     a Markdown table (quick to eyeball)
    <output-dir>/results.tex    a LaTeX (booktabs) table for the paper

Each run reuses the existing ``Trainer`` (so training/eval is identical to
``train.py``); only the model and sigma vary between runs. Results are saved
incrementally, so a crashed or interrupted sweep can be resumed with ``--resume``
and the table can always be regenerated from ``results.json`` with
``--tabulate-only``.

Examples
--------
    # Full sweep (4 models x 3 sigmas) on the GPU
    python experiments.py -d cuda

    # Quick smoke test: one model, few epochs
    python experiments.py -d cuda --models symmetric_mirror --epochs 1

    # Rebuild the tables from an existing results.json without retraining
    python experiments.py --tabulate-only -o exps/paper
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch

from trainer import Trainer
from parseval_cnn import MODELS, MODEL_LABELS


def set_seed(seed=0):
    """Reproducibility, mirroring train.py."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _coerce(value):
    """Turn a CLI string into an int/float/bool/None/str via JSON, else keep str."""
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def _set_dotted(config, dotted_key, value):
    """Set config['a']['b'] = value from a dotted key 'a.b' (creating dicts)."""
    keys = dotted_key.split('.')
    node = config
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def apply_overrides(config, args):
    """Apply hyper-parameter overrides (friendly flags + generic --set) in place.

    Friendly flags win over the base config; --set entries win over everything,
    so you can always reach a config key that has no dedicated flag.
    """
    friendly = {
        'net_params.depth': args.depth,
        'net_params.nb_channels': args.width,
        'net_params.kernel_size': args.kernel_size,
        'training_options.batch_size': args.batch_size,
        'training_options.epochs': args.epochs,
    }
    for dotted, val in friendly.items():
        if val is not None:
            _set_dotted(config, dotted, val)

    for item in (args.set or []):
        if '=' not in item:
            raise ValueError(f"--set expects key=value, got '{item}'")
        key, val = item.split('=', 1)
        _set_dotted(config, key.strip(), _coerce(val))

    return config


def run_one(base_config, model_name, sigma, device, output_dir):
    """Train a single (model, sigma) and return its metrics dict."""
    config = copy.deepcopy(base_config)
    config['net_params']['model'] = model_name
    config['sigma'] = sigma

    # Give each run its own experiment folder so checkpoints/logs never collide.
    config['exp_name'] = f"{model_name}_sigma_{sigma}"
    config['log_dir'] = os.path.join(output_dir, 'runs')

    set_seed(0)
    start = time.time()
    trainer = Trainer(config, device)
    metrics = trainer.train()
    metrics['minutes'] = round((time.time() - start) / 60.0, 2)
    # Provenance: record the hyper-parameters this cell was trained with.
    metrics['epochs'] = config['training_options']['epochs']
    metrics['depth'] = config['net_params']['depth']
    metrics['width'] = config['net_params']['nb_channels']
    metrics['kernel_size'] = config['net_params']['kernel_size']
    metrics['batch_size'] = config['training_options']['batch_size']
    return metrics


# --------------------------------------------------------------------------- #
# Tabulation
# --------------------------------------------------------------------------- #

def _cell(results, model, sigma, metric):
    """Return (psnr, ssim) for a run, or (None, None) if it is missing."""
    entry = results.get(model, {}).get(str(sigma))
    if entry is None:
        return None, None
    return entry.get(f'{metric}_psnr'), entry.get(f'{metric}_ssim')


def _best_psnr_per_sigma(results, models, sigmas, metric):
    """Index of the winning model (highest PSNR) for each sigma, for bolding."""
    best = {}
    for sigma in sigmas:
        vals = [(_cell(results, m, sigma, metric)[0], m) for m in models]
        vals = [(v, m) for v, m in vals if v is not None]
        best[sigma] = max(vals)[1] if vals else None
    return best


def make_markdown(results, models, sigmas, metric):
    header = ['Model'] + [f'sigma={s} (PSNR / SSIM)' for s in sigmas]
    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']
    winners = _best_psnr_per_sigma(results, models, sigmas, metric)
    for m in models:
        row = [MODEL_LABELS.get(m, m)]
        for s in sigmas:
            psnr, ssim = _cell(results, m, s, metric)
            if psnr is None:
                row.append('--')
            else:
                cell = f'{psnr:.2f} / {ssim:.4f}'
                if winners[s] == m:
                    cell = f'**{cell}**'
                row.append(cell)
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines) + '\n'


def make_latex(results, models, sigmas, metric):
    winners = _best_psnr_per_sigma(results, models, sigmas, metric)
    col_spec = 'l' + ' cc' * len(sigmas)
    lines = [
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Gaussian denoising on BSD500. PSNR (dB) / SSIM; higher is '
        'better. Best PSNR per noise level in bold.}',
        '\\label{tab:denoising}',
        f'\\begin{{tabular}}{{{col_spec}}}',
        '\\toprule',
    ]

    # Grouped sigma headers with cmidrules.
    top = ['']
    mids = []
    for i, s in enumerate(sigmas):
        top.append(f'\\multicolumn{{2}}{{c}}{{$\\sigma={s}$}}')
        c0 = 2 + 2 * i
        mids.append(f'\\cmidrule(lr){{{c0}-{c0 + 1}}}')
    lines.append(' & '.join(top) + ' \\\\')
    lines.append(''.join(mids))
    lines.append('Model & ' + ' & '.join(['PSNR & SSIM'] * len(sigmas)) + ' \\\\')
    lines.append('\\midrule')

    for m in models:
        row = [MODEL_LABELS.get(m, m)]
        for s in sigmas:
            psnr, ssim = _cell(results, m, s, metric)
            if psnr is None:
                row += ['--', '--']
            else:
                p = f'{psnr:.2f}'
                q = f'{ssim:.4f}'
                if winners[s] == m:
                    p = f'\\textbf{{{p}}}'
                row += [p, q]
        lines.append(' & '.join(row) + ' \\\\')

    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(lines)


def write_tables(results, models, sigmas, metric, output_dir):
    md = make_markdown(results, models, sigmas, metric)
    tex = make_latex(results, models, sigmas, metric)
    with open(os.path.join(output_dir, 'results.md'), 'w') as f:
        f.write(f'# Denoising results ({metric} validation metric)\n\n')
        f.write(md)
    with open(os.path.join(output_dir, 'results.tex'), 'w') as f:
        f.write(tex)
    print('\n' + md)
    print(f"Wrote results.md, results.tex and results.json to {output_dir}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(args):
    with open(args.config) as f:
        base_config = json.load(f)

    # Apply hyper-parameter overrides (width/depth/etc.) to the base config that
    # every run in this sweep inherits.
    apply_overrides(base_config, args)
    if not args.tabulate_only:
        np_ = base_config['net_params']
        to_ = base_config['training_options']
        print(f"Hyper-parameters for this sweep: depth={np_['depth']}, "
              f"width={np_['nb_channels']}, kernel_size={np_['kernel_size']}, "
              f"batch_size={to_['batch_size']}, epochs={to_['epochs']}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'results.json')

    results = {}
    if os.path.exists(results_path) and (args.resume or args.tabulate_only):
        with open(results_path) as f:
            results = json.load(f)

    if not args.tabulate_only:
        for model_name in args.models:
            results.setdefault(model_name, {})
            for sigma in args.sigmas:
                if args.resume and str(sigma) in results[model_name]:
                    print(f"[skip] {model_name} sigma={sigma} (already done)")
                    continue
                print(f"\n===== training {model_name} @ sigma={sigma} =====")
                metrics = run_one(
                    base_config, model_name, sigma, args.device,
                    args.output_dir,
                )
                results[model_name][str(sigma)] = metrics
                # Save incrementally so an interrupted sweep is recoverable.
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2, sort_keys=True)
                print(f"  -> best PSNR {metrics['best_psnr']:.2f} dB / "
                      f"SSIM {metrics['best_ssim']:.4f} "
                      f"(epoch {metrics['best_epoch']}, {metrics['minutes']} min)")

    write_tables(results, args.models, args.sigmas, args.metric, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train + tabulate the denoising models')
    parser.add_argument('-d', '--device', default='cpu', type=str, help='device to use')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='base config json (data paths, hyper-params)')
    parser.add_argument('-o', '--output-dir', default='exps/paper', type=str,
                        help='where results and per-run checkpoints go')
    parser.add_argument('--models', nargs='+', default=list(MODELS.keys()),
                        choices=list(MODELS.keys()), help='models to train')
    parser.add_argument('--sigmas', nargs='+', type=int, default=[5, 15, 25],
                        help='noise levels to train/evaluate at')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override the number of epochs from the config')

    # Hyper-parameter overrides applied to every run in the sweep.
    hp = parser.add_argument_group('hyper-parameter overrides (applied to every run)')
    hp.add_argument('--depth', type=int, default=None,
                    help='network depth (net_params.depth)')
    hp.add_argument('--width', type=int, default=None,
                    help='number of channels / width (net_params.nb_channels)')
    hp.add_argument('--kernel-size', type=int, default=None,
                    help='convolution kernel size (net_params.kernel_size)')
    hp.add_argument('--batch-size', type=int, default=None,
                    help='training batch size (training_options.batch_size)')
    hp.add_argument('--set', nargs='+', metavar='KEY=VALUE', default=None,
                    help='generic dotted-key overrides for any config field, e.g. '
                         "--set optimizer.lr_weights=1e-4 net_params.beta=0.7 "
                         'activation_params.spline_size=101 net_params.bias=true')

    parser.add_argument('--metric', choices=['best', 'final'], default='best',
                        help="report each run's best or final validation metric")
    parser.add_argument('--resume', action='store_true',
                        help='skip (model, sigma) pairs already present in results.json')
    parser.add_argument('--tabulate-only', action='store_true',
                        help='rebuild the tables from results.json without training')
    args = parser.parse_args()

    main(args)
