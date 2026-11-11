from enum import Enum


class BenchmarkDataset(Enum):
    cifar_10 = 'cifar10'
    imagenet = 'imagenet'


class ThreatModel(Enum):
    Linf = "Linf"
    L2 = "L2"
    corruptions = "corruptions"
