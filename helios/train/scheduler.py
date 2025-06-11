"""Schedulers."""

from dataclasses import dataclass

import torch
from olmo_core.optim.scheduler import Scheduler, _linear_warmup

# TODO this may break if we upgrade olmo core they change scheduler behavior


@dataclass
class PolyWithWarmup(Scheduler):
    """Polynomial learning rate schedule with a warmup."""

    warmup_steps: int | None = None
    warmup_fraction: float | None = None
    power: float = 3.3219
    t_max: int | None = None
    warmup_min_lr: float = 0.0

    def get_lr(
        self, initial_lr: float | torch.Tensor, step: int, max_steps: int
    ) -> float | torch.Tensor:
        """Get learning rate."""
        t_max = max_steps
        current = step

        t_max = t_max if self.t_max is None else self.t_max

        if self.warmup_steps is None:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        else:
            warmup = self.warmup_steps

        if current < warmup:
            return _linear_warmup(initial_lr, current, warmup, self.warmup_min_lr)
        elif current >= t_max:
            return 0
        else:
            current = current - warmup
            t_max = t_max - warmup
            return initial_lr * pow(1 - current / t_max, self.power)
