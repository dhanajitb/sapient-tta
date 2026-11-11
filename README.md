# SAPIENT

Official implementation of **SAPIENT: Continual Test-time Adaptation via Lightweight plug-and-play Adapters** (TMLR 2026).

Dhanajit Brahma, Abhinav Joshi, Ashutosh Modi, and Piyush Rai

[Paper](https://openreview.net/forum?id=OKx9Vw9nZM)

## Installation

Use Python 3.10 and a CUDA-capable GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download [CIFAR-10-C](https://zenodo.org/records/2535967), [ImageNet-C](https://github.com/hendrycks/robustness), or [ImageNet3DCC](https://github.com/EPFL-VILAB/3DCommonCorruptions) and prepare the corresponding directories:

```text
data/
├── CIFAR-10-C/                # labels.npy and corruption .npy files
├── cifar-10-batches-py/       # downloaded automatically if absent
├── ImageNet-C/
│   └── gaussian_noise/5/n01440764/*.JPEG
├── ImageNet-3D/               # ImageNet3DCC
│   └── near_focus/5/n01440764/*.JPEG
└── val/                      # clean ImageNet validation set
    └── n01440764/*.JPEG
```

ImageNet uses the bundled 5,000-image subset. Pretrained weights download automatically; ImageNet uses ResNet-50 ImageNet-1K V1 weights.

## Evaluation

Run SAPIENT with the CoTTA objective on the default corruption sequence:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --dataset cifar10c --data-dir /path/to/data
CUDA_VISIBLE_DEVICES=0 python run.py --dataset imagenetc --data-dir /path/to/data
CUDA_VISIBLE_DEVICES=0 python run.py --dataset imagenet3dcc --data-dir /path/to/data
```

Configurations in `configs/` use Appendix E hyperparameters, severity 5, and one update per batch. Adaptation continues across corruptions.

| Dataset | Backbone | Batch size | Samples/corruption | Optimizer | LR | Restore probability |
|---|---|---:|---:|---|---:|---:|
| CIFAR-10-C | WideResNet-28-10 | 200 | 10,000 | Adam | 0.00125 | 0 |
| ImageNet-C | ResNet-50 | 64 | 5,000 | SGD | 0.04 | 0.001 |
| ImageNet3DCC | ResNet-50 | 64 | 5,000 | SGD | 0.03 | 0.001 |

Add `--method source` to evaluate the source baseline or `--seed N` to change the random seed.

Results are saved to `output/<dataset>/<method>/seed<seed>/`. Each run writes per-corruption error, Brier score, and NLL to `results_*.txt` and reports their corruption means in the log. Clean-data rows are excluded from these means.

## Citation

```bibtex
@article{brahma2026sapient,
  title   = {SAPIENT: Continual Test-time Adaptation via Lightweight plug-and-play Adapters},
  author  = {Brahma, Dhanajit and Joshi, Abhinav and Modi, Ashutosh and Rai, Piyush},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=OKx9Vw9nZM}
}
```

## Acknowledgments

Built on CoTTA, TENT, and RobustBench. See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and licenses.
