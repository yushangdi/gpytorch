#!/usr/bin/env python3

from gpytorch import Module, fast_pred_var
from math import pi, sqrt
import torch
import torch.nn.functional as F
from typing import Optional

from math import inf


def batch_simple_regret(
    x: torch.Tensor,
    model: Module,
    mc_samples: int=1000,
    seed: Optional[int] = None
) -> torch.Tensor:
    # get the old state so we can reset and prevent side-effects
    old_state = torch.random.get_rng_state()
    if seed is not None:
        torch.random.manual_seed(seed)
    # let's be paranoid
    x.requires_grad = True
    model.eval()
    with fast_pred_var():
        val = model(x).sample(mc_samples).max(0)[0].mean()
    if seed is not None:
        torch.random.set_rng_state(old_state)
    return val


def batch_probability_of_improvement(
    x: torch.Tensor,
    model: Module,
    alpha: torch.Tensor,
    mc_samples: int=1000,
    seed: Optional[int] = None,
) -> torch.Tensor:
    # get the old state so we can reset and prevent side-effects
    old_state = torch.random.get_rng_state()
    if seed is not None:
        torch.random.manual_seed(seed)
    # let's be paranoid
    x.requires_grad = True
    model.eval()
    with fast_pred_var():
        val = F.sigmoid(model(x).sample(mc_samples).max(0)[0] - alpha).mean()
    if seed is not None:
        torch.random.set_rng_state(old_state)
    return val


def batch_expected_improvement(
    x: torch.Tensor,
    model: Module,
    alpha: torch.Tensor,
    mc_samples: int=1000,
    seed: Optional[int] = None,
) -> torch.Tensor:
    # get the old state so we can reset and prevent side-effects
    old_state = torch.random.get_rng_state()
    if seed is not None:
        torch.random.manual_seed(seed)
    # let's be paranoid
    x.requires_grad = True
    model.eval()
    with fast_pred_var():
        val = (model(x).sample(mc_samples).max(0)[0] - alpha).clamp(0, inf).mean()
    if seed is not None:
        torch.random.set_rng_state(old_state)
    return val


def batch_upper_confidence_bound(
    x: torch.Tensor,
    model: Module,
    beta: float,
    mc_samples: int=1000,
    seed: Optional[int] = None,
) -> torch.Tensor:
    # get the old state so we can reset and prevent side-effects
    old_state = torch.random.get_rng_state()
    if seed is not None:
        torch.random.manual_seed(seed)
    x.requires_grad = True
    model.eval()
    with fast_pred_var():
        mvn = model(x)
        val = (
            sqrt(beta * pi / 2) * mvn.covar().zero_mean_mvn_samples(mc_samples).abs() +
            mvn._mean.view(-1, 1)
        ).max(0)[0].mean()
    if seed is not None:
        torch.random.set_rng_state(old_state)
    return val
