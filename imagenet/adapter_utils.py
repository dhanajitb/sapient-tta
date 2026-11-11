"""Identity initialization and parameter selection for SAPIENT."""
import torch


@torch.no_grad()
def initialize_adapters(model):
    """Make each groupwise-plus-pointwise adapter an exact identity."""
    for module in model.modules():
        if not (hasattr(module, "gwc") and hasattr(module, "pwc")):
            continue
        module.gwc.weight.zero_()
        if module.gwc.bias is not None:
            module.gwc.bias.zero_()
        weight = module.pwc.weight
        if weight.shape[0] != weight.shape[1] or weight.shape[2:] != (1, 1):
            raise ValueError("Identity initialization requires a square, ungrouped 1x1 PWC")
        weight.copy_(torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype).view_as(weight))
        if module.pwc.bias is not None:
            module.pwc.bias.zero_()
    return model


def select_adapter_parameters(model, include_classifier=False):
    """Train adapters and shortcut convolutions, optionally including the classifier.

    Batch-normalization affine parameters are enabled by configure_model.
    """
    for name, param in model.named_parameters():
        param.requires_grad_(any(key in name for key in ("lhc1", "lhc2", "lhc3", "convShortcut"))
                             or (include_classifier and "fc" in name))
    return model
