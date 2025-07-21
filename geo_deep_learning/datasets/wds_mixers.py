"""Mixers for WebDataset."""

import logging
from collections.abc import Iterator
from typing import Any, Literal

import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


class MixedIterableDataset(IterableDataset):
    """Mixed IterableDataset."""

    def __init__(
        self,
        datasets: list[IterableDataset],
        strategy: Literal["round_robin", "uniform"] = "round_robin",
    ) -> None:
        """Initialize the MixedIterableDataset."""
        self.datasets = datasets
        self.strategy = strategy

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over the datasets."""
        iterators = [iter(d) for d in self.datasets]

        if self.strategy == "round_robin":
            return self._round_robin(iterators)
        if self.strategy == "uniform":
            return self._uniform(iterators)
        msg = f"Unsupported strategy: {self.strategy}"
        raise ValueError(msg)

    def _round_robin(
        self,
        iterators: list[Iterator[dict[str, Any]]],
    ) -> Iterator[dict[str, Any]]:
        """Round-robin strategy."""
        if not iterators:
            return

        active_iterators = list(iterators)
        sentinel = object()
        rank_str = ""
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            rank_str = f"[Rank {rank}] "
        logger.info(
            "%sStarting round_robin with %d iterators",
            rank_str,
            len(active_iterators),
        )
        batch_count = 0

        while active_iterators:
            i = 0
            while i < len(active_iterators):
                item = next(active_iterators[i], sentinel)
                if item is not sentinel:
                    yield item
                    batch_count += 1
                    if batch_count % 1000 == 0:
                        logger.debug("%sYielded %d items so far", rank_str, batch_count)
                    i += 1
                else:
                    active_iterators.pop(i)

        logger.info("%sRound_robin completed after %d items", rank_str, batch_count)

    def _uniform(
        self,
        iterators: list[Iterator[dict[str, Any]]],
    ) -> None:
        """Uniform strategy."""


def get_mixed_dataset(
    sensor_datasets: dict[str, IterableDataset],
    strategy: str,
) -> MixedIterableDataset:
    """Get a mixed dataset."""
    return MixedIterableDataset(list(sensor_datasets.values()), strategy)
