#!/usr/bin/env python3
"""Launch a classification experiment from any working directory."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNNERS = {
    'cifar10c': ('cifar', 'cifar10c_adapter.py'),
    'imagenetc': ('imagenet', 'imagenetc_adpt.py'),
    'imagenet3dcc': ('imagenet', 'imagenet3dc_adpt.py'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=RUNNERS, required=True)
    parser.add_argument('--config', type=Path, help='Custom YAML configuration')
    parser.add_argument('--method', choices=('cotta', 'tent', 'source', 'norm'))
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--checkpoint-dir', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true', help='Print the command without importing PyTorch or downloading data')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[], help='Additional YACS KEY VALUE pairs; place last')
    args = parser.parse_args()
    suite, entry = RUNNERS[args.dataset]
    config = args.config or ROOT / 'configs' / args.dataset / 'sapient.yaml'
    config = config.resolve()
    if not config.is_file():
        parser.error('Configuration not found: %s.' % config)
    if len(args.opts) % 2:
        parser.error('--opts expects KEY VALUE pairs')
    method = args.method or 'cotta'
    output = args.output_dir or ROOT / 'output' / args.dataset / method / ('seed%d' % args.seed)
    cmd = [sys.executable, str(ROOT / suite / entry), '--cfg', str(config),
           'DATA_DIR', str(args.data_dir.resolve()),
           'CKPT_DIR', str(args.checkpoint_dir.resolve()),
           'SAVE_DIR', str(output.resolve()), 'RNG_SEED', str(args.seed)]
    if args.method:
        cmd += ['MODEL.ADAPTATION', args.method]
    cmd += args.opts
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    return subprocess.call(cmd, cwd=ROOT / suite)


if __name__ == '__main__':
    sys.exit(main())
