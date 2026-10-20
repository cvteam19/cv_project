"""Shared LR protocol for both detectors: SGD + a three-hour cosine budget."""
import math


def budget_cosine_lr(elapsed_seconds, budget_seconds=10800.0,
                     initial_lr=0.005, minimum_lr=1e-6):
    if budget_seconds <= 0:
        raise ValueError('budget_seconds must be positive')
    progress = min(1.0, max(0.0, float(elapsed_seconds) / budget_seconds))
    return minimum_lr + (initial_lr - minimum_lr) * (1 + math.cos(math.pi * progress)) / 2


def set_budget_lr(optimizer, elapsed_seconds, budget_seconds=10800.0):
    lr = budget_cosine_lr(elapsed_seconds, budget_seconds)
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr


def build_shared_sgd(model):
    import torch
    return torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                           lr=0.005, momentum=0.9, weight_decay=0.0005,
                           nesterov=False)
