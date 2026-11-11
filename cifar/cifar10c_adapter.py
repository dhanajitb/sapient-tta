from adapter_utils import initialize_adapters, select_adapter_parameters
import logging

import torch
import torch.optim as optim

from robustbench.data import load_cifar10c, load_cifar10
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import tent
import norm
import cotta

from conf import cfg, load_cfg_fom_args

import os

logger = logging.getLogger(__name__)

all_metrics = True

if all_metrics:
    from robustbench.utils import clean_metrics as metrics
else:
    from robustbench.utils import clean_accuracy as accuracy

def zero_init(model):
    return initialize_adapters(model)


def free_adapter_params(model):
    return select_adapter_parameters(model, include_classifier=True)


def evaluate(description):
    load_cfg_fom_args(description)
    if not torch.cuda.is_available():
        raise RuntimeError('Evaluation requires a CUDA GPU.')
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    base_model = zero_init(base_model)
    base_model = free_adapter_params(model=base_model)

    print("---------------Starting Test Time Adaptation-----------------")
    # print the number of trainable parameters vs total parameters in percentage
    trainables = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in base_model.parameters())
    print(
        f"Trainable params: {trainables} out of {total_params} ({trainables/total_params*100:.2f}%)"
    )

    print("--------------------------------------------------------------")

    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)

    # evaluate on each severity and type of corruption in turn
    prev_ct = "x0"
    err_dict = {}
    bri_dict = {}
    nll_dict = {}
    for severity in cfg.CORRUPTION.SEVERITY:
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # continual adaptation for all corruption
            if i_c == 0:
                try:
                    model.reset()
                    logger.info("resetting model")
                except AttributeError:
                    logger.warning("not resetting model")
            else:
                logger.warning("not resetting model")
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            # clean_metrics transfers each batch to the model device.
            if all_metrics:
                acc, bri, nll = metrics(model, x_test, y_test, cfg.TEST.BATCH_SIZE, num_classes=10, device=next(model.parameters()).device)
                err = 1. - acc
                logger.info(f"[{corruption_type}{severity}] % error : {err:.2%}, brier : {bri:.4}, nll : {nll:.4}")
                err_dict[f"{corruption_type}/{severity}"] = err
                bri_dict[f"{corruption_type}/{severity}"] = bri
                nll_dict[f"{corruption_type}/{severity}"] = nll
            else:
                acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
                err = 1. - acc
                logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")

    corruption_type = "source"
    severity = 0
    x_test, y_test = load_cifar10(cfg.CORRUPTION.NUM_EX,
                                  cfg.DATA_DIR)
    # clean_metrics transfers each batch to the model device.
    if all_metrics:
        acc, bri, nll = metrics(model, x_test, y_test, cfg.TEST.BATCH_SIZE, num_classes=10, device=next(model.parameters()).device)
        err = 1. - acc
        logger.info(f"[{corruption_type}{severity}] % error : {err:.2%}, brier : {bri:.4}, nll : {nll:.4}")
        err_dict[f"{corruption_type}/{severity}"] = err
        bri_dict[f"{corruption_type}/{severity}"] = bri
        nll_dict[f"{corruption_type}/{severity}"] = nll
    else:
        acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
        err = 1. - acc
        logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")

    corruption_keys = [key for key in err_dict if key.split('/')[0] != 'source']
    logger.info("Mean corruption error: %.2f%%; Brier: %.4f; NLL: %.4f",
                100 * sum(err_dict[key] for key in corruption_keys) / len(corruption_keys),
                sum(bri_dict[key] for key in corruption_keys) / len(corruption_keys),
                sum(nll_dict[key] for key in corruption_keys) / len(corruption_keys))

    with open(os.path.join(cfg.SAVE_DIR, "results_" + cfg.LOG_DEST),"w") as f:
        f.write("SPW,"+str(cfg.OPTIM.SPW)+"\n")
        f.write("PERC,"+str(cfg.OPTIM.PERC)+"\n")
        f.write("Error\n")
        for corr in err_dict:
            f.write(corr+","+str("{:.2%}".format(err_dict[corr]))+"\n")
        f.write("\nBrier\n")
        for corr in bri_dict:
            f.write(corr+","+str("{:.4}".format(bri_dict[corr]))+"\n")
        f.write("\nNLL\n")
        for corr in nll_dict:
            f.write(corr+","+str("{:.4}".format(nll_dict[corr]))+"\n")


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           mt_alpha=cfg.OPTIM.MT,
                           rst_m=cfg.OPTIM.RST,
                           ap=cfg.OPTIM.AP)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation.')
