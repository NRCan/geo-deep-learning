"""Mixers for WebDataset."""

import logging
from collections.abc import Iterator
from typing import Any, Literal

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
            iterator = self._round_robin(iterators)
        elif self.strategy == "uniform":
            iterator = self._uniform(iterators)
        else:
            msg = f"Unsupported strategy: {self.strategy}"
            raise ValueError(msg)
        return iterator

    def _round_robin(
        self,
        iterators: list[Iterator[dict[str, Any]]],
    ) -> Iterator[dict[str, Any]]:
        """Round-robin strategy."""
        if not iterators:
            return

        active_iterators = list(iterators)
        sentinel = object()

        while active_iterators:
            next_round_iterators = []
            for i, iterator in enumerate(active_iterators):
                try:
                    item = next(iterator, sentinel)
                    if item is not sentinel:
                        yield item
                        next_round_iterators.append(iterator)
                except Exception:  # noqa: PERF203
                    logger.exception("Error in iterator %d", i)
            if not next_round_iterators:
                break
            active_iterators = next_round_iterators

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
    return MixedIterableDataset(
        datasets=list(sensor_datasets.values()),
        strategy=strategy,
    )
