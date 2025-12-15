"""Linear warmup cosine annealing learning rate scheduler."""

import math
import warnings
from collections.abc import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Adapted from https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/optimizers/lr_scheduler.py


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule.

    Sets the learning rate of each parameter group to follow a
    linear warmup schedule between warmup_start_lr and
    base_lr followed by a cosine annealing schedule between base_lr and eta_min.

    .. warning::
        It is recommended to call :func:`.step()`
        for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an
        EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler,
        when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()`
        function before calling
        train and validation methods.

    Example:
        >>> import torch.nn as nn
        >>> from torch.optim import Adam
        >>> #
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(
        ...     optimizer,
        ...     warmup_epochs=10,
        ...     max_epochs=40,
        ... )
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)

    """

    def __init__(  # noqa: PLR0913
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """Initialize the linear warmup cosine annealing learning rate scheduler."""
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler; "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(
                    self.base_lrs,
                    self.optimizer.param_groups,
                    strict=False,
                )
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(
                    self.base_lrs,
                    self.optimizer.param_groups,
                    strict=False,
                )
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs),
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs),
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> list[float]:
        """Get closed form learning rate when epoch is passed to step function."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (base_lr - self.warmup_start_lr)
                / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs),
                )
            )
            for base_lr in self.base_lrs
        ]


class MaeLRScheduler:
    """Cosine scheduler with linear warmup designed for MAE-style training loops."""

    def __init__(  # noqa: PLR0913
        self,
        optimizer: Optimizer,
        accum_iter: int,
        lr: float | None,
        min_lr: float,
        warmup_epochs: int,
        iteration_per_epoch: int,
        max_iterations: int,
    ) -> None:
        if accum_iter < 1:
            msg = "`accum_iter` must be >= 1."
            raise ValueError(msg)
        if iteration_per_epoch <= 0:
            msg = "`iteration_per_epoch` must be a positive integer."
            raise ValueError(msg)
        if max_iterations <= 0:
            msg = "`max_iterations` must be a positive integer."
            raise ValueError(msg)
        if warmup_epochs < 0:
            msg = "`warmup_epochs` must be non-negative."
            raise ValueError(msg)
        if max_iterations < iteration_per_epoch:
            msg = "`max_iterations` must be >= `iteration_per_epoch`."
            raise ValueError(msg)

        self.optimizer = optimizer
        self.accum_iter = accum_iter
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.iteration_per_epoch = iteration_per_epoch
        self.max_iterations = max_iterations

        self.iter_num = 0
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    def get_last_lr(self) -> list[float]:
        """Return last computed learning rate."""
        return list(self._last_lr)

    def step(self) -> None:
        """Advance the scheduler by one iteration."""
        current_epoch = self.iter_num // self.iteration_per_epoch
        total_epochs = self.max_iterations // self.iteration_per_epoch
        if total_epochs <= self.warmup_epochs:
            total_epochs = max(self.warmup_epochs + 1, total_epochs)
        i_batch = self.iter_num % self.iteration_per_epoch
        epoch = current_epoch + i_batch / self.iteration_per_epoch

        if i_batch % self.accum_iter == 0:
            if epoch < self.warmup_epochs:
                lr = self.lr * epoch / max(1, self.warmup_epochs)
            else:
                cosine_progress = (epoch - self.warmup_epochs) / max(
                    1,
                    total_epochs - self.warmup_epochs,
                )
                cosine_progress = min(max(cosine_progress, 0.0), 1.0)
                lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                    1.0 + math.cos(math.pi * cosine_progress)
                )

            updated_lrs = []
            for group in self.optimizer.param_groups:
                scaled_lr = lr * group.get("lr_scale", 1.0)
                group["lr"] = scaled_lr
                updated_lrs.append(scaled_lr)
            self._last_lr = updated_lrs

        self.iter_num += 1

    def state_dict(self) -> dict[str, float | int | list[float]]:
        """Serialize scheduler state."""
        exclude = {
            "optimizer",
            "max_iterations",
            "iteration_per_epoch",
            "warmup_epochs",
            "accum_iter",
        }
        return {k: v for k, v in self.__dict__.items() if k not in exclude}

    def load_state_dict(self, state_dict: dict[str, float | int | list[float]]) -> None:
        """Restore scheduler state."""
        self.__dict__.update(state_dict)


class MaeLRSchedulerFactory:
    """Callable wrapper so Lightning CLI can configure :class:`MaeLRScheduler`."""

    def __init__(
        self,
        *,
        accum_iter: int,
        min_lr: float,
        warmup_epochs: int,
        lr: float | None = None,
        iteration_per_epoch: int | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self.accum_iter = accum_iter
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.iteration_per_epoch = iteration_per_epoch
        self.max_iterations = max_iterations

    def __call__(
        self,
        optimizer: Optimizer,
        *,
        lr: float | None = None,
        iteration_per_epoch: int | None = None,
        max_iterations: int | None = None,
    ) -> MaeLRScheduler:
        iteration_per_epoch = (
            iteration_per_epoch if iteration_per_epoch is not None else self.iteration_per_epoch
        )
        max_iterations = (
            max_iterations if max_iterations is not None else self.max_iterations
        )
        base_lr = self.lr if self.lr is not None else lr
        if base_lr is None:
            msg = (
                "MaeLRSchedulerFactory requires `lr` to be provided either at construction "
                "time or when calling the factory."
            )
            raise ValueError(msg)
        
        return MaeLRScheduler(
            optimizer=optimizer,
            accum_iter=self.accum_iter,
            lr=base_lr,
            min_lr=self.min_lr,
            warmup_epochs=self.warmup_epochs,
            iteration_per_epoch=iteration_per_epoch,
            max_iterations=max_iterations,
        )


# warmup + decay as a function
def linear_warmup_decay(
    warmup_steps: int,
    total_steps: int,
    *,
    cosine: bool = True,
    linear: bool = False,
) -> Callable[[int], float]:
    """Create a learning rate scheduler with linear warmup and optional decay."""
    if linear and cosine:
        msg = "linear and cosine cannot be True at the same time"
        raise ValueError(msg)

    def fn(step: int) -> float:
        """Calculate learning rate multiplier for given step."""
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        if not (cosine or linear):
            # no decay
            return 1.0

        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps),
        )
        if cosine:
            # cosine decay
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # linear decay
        return 1.0 - progress

    return fn
