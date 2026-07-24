"""Train every model config in a folder at several noise levels, then tabulate.

This is the experiment driver for the paper's denoising table. Instead of one
model with CLI overrides, it reads a *folder of JSON configs* (one per model);
each config uses the same schema as ``config.json`` and additionally names the
model (``net_params.model``) and the noise levels to sweep (``sigmas``). Because
each config is independent, models may use different hyper-parameters (width,
depth, learning rates, ...).

For every config, and every sigma it requests, a run trains with the existing
``Trainer`` (training/eval identical to ``train.py``) and records the validation
PSNR/SSIM. The driver writes:

    <output-dir>/results.json   raw metrics for every (config, sigma) run
    <output-dir>/results.md     a Markdown table (quick to eyeball)
    <output-dir>/results.tex    a LaTeX (booktabs) table for the paper

Results are saved incrementally, so an interrupted sweep can be resumed with
``--resume`` and the table can be regenerated from ``results.json`` with
``--tabulate-only``.

Config folder
-------------
See ``experiment_configs/`` for examples and ``experiment_configs/_template.json``
for a documented template. Each ``*.json`` in the folder is one model; the file
name is that model's identity (its row in the table). Files whose name starts
with ``_`` are ignored, so ``_template.json`` is never run.

Examples
--------
    # Full sweep over experiment_configs/ on the GPU
    python experiments.py -d cuda

    # A subset of configs, few epochs, as a smoke test
    python experiments.py -d cuda --configs symmetric_mirror --epochs 1

    # Rebuild the tables from an existing results.json without retraining
    python experiments.py --tabulate-only -o exps/paper
"""

import argparse
import copy
import json
import os
import random
import time
from glob import glob

import numpy as np
import torch

from trainer import Trainer


def set_seed(seed=0):
    """Reproducibility, mirroring train.py."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def load_configs(config_dir, only=None):
    """Load ``*.json`` configs from a folder.

    Returns a list of ``(name, config)`` in sorted file-name order. Files whose
    name starts with ``_`` (e.g. ``_template.json``) are skipped. ``only`` is an
    optional list of names (file stems) to keep.
    """
    configs = []
    for path in sorted(glob(os.path.join(config_dir, '*.json'))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.startswith('_'):
            continue
        if only is not None and name not in only:
            continue
        with open(path) as f:
            configs.append((name, json.load(f)))
    return configs


def resolve_sigmas(config, default_sigmas):
    """Noise levels for a config: its ``sigmas`` list, else ``sigma``, else CLI."""
    if 'sigmas' in config:
        return list(config['sigmas'])
    if 'sigma' in config:
        return [config['sigma']]
    return list(default_sigmas)


def run_one(base_config, name, sigma, device, output_dir, epochs=None):
    """Train a single (config, sigma) and return its metrics dict."""
    config = copy.deepcopy(base_config)
    # Strip driver-only keys the Trainer does not expect.
    config.pop('sigmas', None)
    config.pop('label', None)
    config.pop('_comment', None)

    config['sigma'] = sigma
    if epochs is not None:
        config['training_options']['epochs'] = epochs

    # Give each run its own experiment folder so checkpoints/logs never collide.
    config['exp_name'] = f"{name}_sigma_{sigma}"
    config['log_dir'] = os.path.join(output_dir, 'runs')

    set_seed(0)
    start = time.time()
    trainer = Trainer(config, device)
    metrics = trainer.train()
    metrics['minutes'] = round((time.time() - start) / 60.0, 2)
    # Provenance: record the hyper-parameters this cell was trained with.
    metrics['model'] = config['net_params'].get('model')
    metrics['depth'] = config['net_params']['depth']
    metrics['width'] = config['net_params']['nb_channels']
    metrics['kernel_size'] = config['net_params']['kernel_size']
    metrics['epochs'] = config['training_options']['epochs']
    return metrics


# --------------------------------------------------------------------------- #
# Tabulation
# --------------------------------------------------------------------------- #

def _all_sigmas(results, names):
    """Sorted union of every sigma present across the given configs."""
    sigmas = set()
    for name in names:
        sigmas.update(int(s) for s in results.get(name, {}).get('runs', {}))
    return sorted(sigmas)


def _cell(results, name, sigma, metric):
    """Return (psnr, ssim) for a run, or (None, None) if it is missing."""
    entry = results.get(name, {}).get('runs', {}).get(str(sigma))
    if entry is None:
        return None, None
    return entry.get(f'{metric}_psnr'), entry.get(f'{metric}_ssim')


def _label(results, name):
    return results.get(name, {}).get('label') or name


def _best_psnr_per_sigma(results, names, sigmas, metric):
    """Name of the winning model (highest PSNR) for each sigma, for bolding."""
    best = {}
    for sigma in sigmas:
        vals = [(_cell(results, n, sigma, metric)[0], n) for n in names]
        vals = [(v, n) for v, n in vals if v is not None]
        best[sigma] = max(vals)[1] if vals else None
    return best


def make_markdown(results, names, sigmas, metric):
    header = ['Model'] + [f'sigma={s} (PSNR / SSIM)' for s in sigmas]
    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']
    winners = _best_psnr_per_sigma(results, names, sigmas, metric)
    for name in names:
        row = [_label(results, name)]
        for s in sigmas:
            psnr, ssim = _cell(results, name, s, metric)
            if psnr is None:
                row.append('--')
            else:
                cell = f'{psnr:.2f} / {ssim:.4f}'
                if winners[s] == name:
                    cell = f'**{cell}**'
                row.append(cell)
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines) + '\n'


def make_latex(results, names, sigmas, metric):
    winners = _best_psnr_per_sigma(results, names, sigmas, metric)
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

    for name in names:
        row = [_label(results, name)]
        for s in sigmas:
            psnr, ssim = _cell(results, name, s, metric)
            if psnr is None:
                row += ['--', '--']
            else:
                p = f'{psnr:.2f}'
                q = f'{ssim:.4f}'
                if winners[s] == name:
                    p = f'\\textbf{{{p}}}'
                row += [p, q]
        lines.append(' & '.join(row) + ' \\\\')

    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(lines)


def write_tables(results, names, metric, output_dir):
    sigmas = _all_sigmas(results, names)
    md = make_markdown(results, names, sigmas, metric)
    tex = make_latex(results, names, sigmas, metric)
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
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'results.json')

    results = {}
    if os.path.exists(results_path) and (args.resume or args.tabulate_only):
        with open(results_path) as f:
            results = json.load(f)

    if args.tabulate_only:
        names = list(results.keys())
        write_tables(results, names, args.metric, args.output_dir)
        return

    configs = load_configs(args.config_dir, only=args.configs)
    if not configs:
        raise SystemExit(f"No runnable configs found in '{args.config_dir}'.")

    names = [name for name, _ in configs]
    for name, config in configs:
        sigmas = resolve_sigmas(config, args.sigmas)
        entry = results.setdefault(name, {})
        entry['label'] = config.get('label', name)
        entry['model'] = config['net_params'].get('model')
        entry.setdefault('runs', {})
        for sigma in sigmas:
            if args.resume and str(sigma) in entry['runs']:
                print(f"[skip] {name} sigma={sigma} (already done)")
                continue
            print(f"\n===== training {name} ({entry['model']}) @ sigma={sigma} =====")
            metrics = run_one(config, name, sigma, args.device,
                              args.output_dir, epochs=args.epochs)
            entry['runs'][str(sigma)] = metrics
            # Save incrementally so an interrupted sweep is recoverable.
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, sort_keys=True)
            print(f"  -> best PSNR {metrics['best_psnr']:.2f} dB / "
                  f"SSIM {metrics['best_ssim']:.4f} "
                  f"(epoch {metrics['best_epoch']}, {metrics['minutes']} min)")

    write_tables(results, names, args.metric, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train + tabulate the denoising models')
    parser.add_argument('-d', '--device', default='cpu', type=str, help='device to use')
    parser.add_argument('--config-dir', default='experiment_configs', type=str,
                        help='folder of per-model JSON configs to run')
    parser.add_argument('-o', '--output-dir', default='exps/paper', type=str,
                        help='where results and per-run checkpoints go')
    parser.add_argument('--configs', nargs='+', default=None, metavar='NAME',
                        help='only run these config names (file stems); default: all')
    parser.add_argument('--sigmas', nargs='+', type=int, default=[5, 15, 25],
                        help="fallback noise levels for configs that don't set 'sigmas'")
    parser.add_argument('--epochs', type=int, default=None,
                        help='override the number of epochs for every config')
    parser.add_argument('--metric', choices=['best', 'final'], default='best',
                        help="report each run's best or final validation metric")
    parser.add_argument('--resume', action='store_true',
                        help='skip (config, sigma) pairs already present in results.json')
    parser.add_argument('--tabulate-only', action='store_true',
                        help='rebuild the tables from results.json without training')
    args = parser.parse_args()

    main(args)
