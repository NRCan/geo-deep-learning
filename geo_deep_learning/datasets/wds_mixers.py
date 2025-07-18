"""Mixers for WebDataset."""

import random
from collections.abc import Iterator
from typing import Any, Literal

from torch.utils.data import IterableDataset


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
        """Round-robin strategy without using exceptions."""
        active_iterators = list(iterators)
        sentinel = object()

        while active_iterators:
            still_active = []
            for it in active_iterators:
                item = next(it, sentinel)
                if item is not sentinel:
                    yield item
                    still_active.append(it)
            active_iterators = still_active

    def _uniform(
        self,
        iterators: list[Iterator[dict[str, Any]]],
    ) -> Iterator[dict[str, Any]]:
        """Uniform strategy."""
        iterators = list(iterators)
        while iterators:
            i = random.randint(0, len(iterators) - 1)  # noqa: S311
            try:
                yield next(iterators[i])
            except StopIteration:
                iterators.pop(i)


def get_mixed_dataset(
    sensor_datasets: dict[str, IterableDataset],
    strategy: str,
) -> MixedIterableDataset:
    """Get a mixed dataset."""
    return MixedIterableDataset(list(sensor_datasets.values()), strategy)
