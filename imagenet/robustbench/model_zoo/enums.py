from enum import Enum


class BenchmarkDataset(Enum):
    cifar_10 = 'cifar10'
    imagenet = 'imagenet'
    imagenet3d = 'imagenet3d'  # imagenet 3d dataset added


class ThreatModel(Enum):
    Linf = "Linf"
    L2 = "L2"
    corruptions = "corruptions"
