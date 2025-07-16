"""RoundRobinSampler for training."""

import logging
import math
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class RoundRobinSampler(Sampler):
    """Round Robin Sampler."""

    def __init__(
        self,
        datasets: dict[str, Dataset],
        batch_size: int,
        sensor_weighting: str = "equal",
        custom_weights: dict[str, float] | None = None,
        *,
        drop_last: bool = False,
    ) -> None:
        """Initialize RoundRobinSampler."""
        self.datasets = datasets
        self.batch_size = batch_size
        self.sensor_weighting = sensor_weighting
        self.custom_weights = custom_weights or {}
        self.drop_last = drop_last

        # Group indices by sensor
        self.sensor_indices = {}
        self.dataset_offset = 0

        for sensor_name, dataset in datasets.items():
            dataset_size = len(dataset)
            self.sensor_indices[sensor_name] = list(
                range(self.dataset_offset, self.dataset_offset + dataset_size),
            )
            self.dataset_offset += dataset_size

        # Calculate sensor weights and adjust indices
        self.sensor_weights = self._calculate_weights()
        self.adjusted_indices = self._adjust_indices()

        logger.info(
            "RoundRobinSampler initialized with %d sensors",
            len(datasets),
        )
        logger.info("Sensor weights: %s", self.sensor_weights)
        logger.info("Dataset sizes: %s", {k: len(v) for k, v in datasets.items()})

    def _calculate_weights(self) -> dict[str, int]:
        """Calculate how many batches each sensor should get per cycle."""
        if self.sensor_weighting == "equal":
            return dict.fromkeys(self.datasets.keys(), 1)

        if self.sensor_weighting == "proportional":
            # Weight by dataset size
            total_samples = sum(len(dataset) for dataset in self.datasets.values())
            weights = {}
            for sensor, dataset in self.datasets.items():
                proportion = len(dataset) / total_samples
                # Convert to integer weights (minimum 1)
                weights[sensor] = max(1, int(proportion * len(self.datasets) * 4))
            return weights

        if self.sensor_weighting == "custom":
            if not self.custom_weights:
                logger.warning(
                    "Custom weights not provided, falling back to equal weighting",
                )
                return dict.fromkeys(self.datasets.keys(), 1)

            # Convert custom weights to integer batch counts
            weights = {}
            total_weight = sum(self.custom_weights.values())
            for sensor in self.datasets:
                weight = self.custom_weights.get(sensor, 1.0)
                weights[sensor] = max(
                    1,
                    int((weight / total_weight) * len(self.datasets) * 4),
                )
            return weights
        msg = f"Unknown sensor_weighting: {self.sensor_weighting}"
        raise ValueError(msg)

    def _adjust_indices(self) -> dict[str, np.ndarray]:
        """Adjust indices based on weights to ensure balanced sampling."""
        # Find the maximum length needed across all sensors
        max_samples_needed = 0
        for sensor, weight in self.sensor_weights.items():
            samples_per_weight = len(self.sensor_indices[sensor])
            total_needed = samples_per_weight * weight
            max_samples_needed = max(max_samples_needed, total_needed)

        # Normalize to a common length per sensor
        target_length = max_samples_needed // max(self.sensor_weights.values())

        adjusted = {}
        for sensor in self.datasets:
            indices = np.array(self.sensor_indices[sensor])

            if len(indices) < target_length:
                # Replicate indices to reach target length
                repetitions = (target_length // len(indices)) + 1
                extended_indices = np.tile(indices, repetitions)[:target_length]
                adjusted[sensor] = extended_indices
            else:
                # Take subset if we have more than needed
                adjusted[sensor] = indices[:target_length]

        return adjusted

    def __iter__(self) -> Iterator[list[int]]:
        """Generate batches in round-robin fashion."""
        rng = np.random.default_rng()

        # Shuffle indices for each sensor
        sensor_batches = {}
        for sensor, indices in self.adjusted_indices.items():
            shuffled_indices = indices.copy()
            rng.shuffle(shuffled_indices)
            sensor_batches[sensor] = shuffled_indices

        # Create weighted sensor cycle
        weighted_sensors = []
        for sensor, weight in self.sensor_weights.items():
            weighted_sensors.extend([sensor] * weight)
        rng.shuffle(weighted_sensors)

        # Generate batches in round-robin fashion
        max_len = max(len(indices) for indices in sensor_batches.values())
        batch_positions = dict.fromkeys(self.datasets.keys(), 0)

        for _ in range(0, max_len, self.batch_size):
            for sensor in weighted_sensors:
                start_idx = batch_positions[sensor]
                end_idx = start_idx + self.batch_size

                if end_idx <= len(sensor_batches[sensor]):
                    batch_indices = sensor_batches[sensor][start_idx:end_idx].tolist()
                    batch_positions[sensor] = end_idx

                    if len(batch_indices) == self.batch_size or not self.drop_last:
                        yield batch_indices

    def __len__(self) -> int:
        """Estimate number of batches per epoch."""
        total_batches = 0
        for sensor, weight in self.sensor_weights.items():
            sensor_samples = len(self.adjusted_indices[sensor])
            sensor_batches = sensor_samples // self.batch_size
            total_batches += sensor_batches * weight
        return total_batches


class RoundRobinDistributedSampler(Sampler):
    """Distributed Round Robin Sampler."""

    def __init__(  # noqa: PLR0913
        self,
        datasets: dict[str, Dataset],
        batch_size: int,
        sensor_weighting: str = "equal",
        custom_weights: dict[str, float] | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
        *,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        """
        Initialize RoundRobinDistributedSampler.

        Args:
            datasets: Dict mapping sensor_name -> dataset
            batch_size: Size of each batch
            sensor_weighting: "equal", "proportional", or "custom"
            custom_weights: Custom weights for each sensor
            num_replicas: Number of processes participating in distributed training
            rank: Rank of the current process
            shuffle: Whether to shuffle data
            drop_last: Whether to drop incomplete batches

        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.sensor_weighting = sensor_weighting
        self.custom_weights = custom_weights or {}
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

        # Distributed setup
        self.num_replicas = (
            num_replicas
            if num_replicas is not None
            else torch.distributed.get_world_size()
        )
        self.rank = rank if rank is not None else torch.distributed.get_rank()

        # Group indices by sensor
        self.sensor_indices = {}
        dataset_offset = 0

        for sensor_name, dataset in datasets.items():
            dataset_size = len(dataset)
            self.sensor_indices[sensor_name] = list(
                range(dataset_offset, dataset_offset + dataset_size),
            )
            dataset_offset += dataset_size

        # Calculate weights and prepare distributed indices
        self.sensor_weights = self._calculate_weights()
        self.adjusted_indices = self._prepare_distributed_indices()

        # Calculate samples per replica
        total_samples = sum(len(indices) for indices in self.adjusted_indices.values())
        self.num_samples = math.ceil(total_samples / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

        logger.info(
            "RoundRobinDistributedSampler initialized: rank=%d, "
            "world_size=%d, samples_per_rank=%d",
            self.rank,
            self.num_replicas,
            self.num_samples,
        )

    def _calculate_weights(self) -> dict[str, int]:
        """Calculate how many batches each sensor should get per cycle."""
        if self.sensor_weighting == "equal":
            return dict.fromkeys(self.datasets.keys(), 1)

        if self.sensor_weighting == "proportional":
            total_samples = sum(len(dataset) for dataset in self.datasets.values())
            weights = {}
            for sensor, dataset in self.datasets.items():
                proportion = len(dataset) / total_samples
                weights[sensor] = max(1, int(proportion * len(self.datasets) * 4))
            return weights

        if self.sensor_weighting == "custom":
            if not self.custom_weights:
                return dict.fromkeys(self.datasets.keys(), 1)

            total_weight = sum(self.custom_weights.values())
            weights = {}
            for sensor in self.datasets:
                weight = self.custom_weights.get(sensor, 1.0)
                weights[sensor] = max(
                    1,
                    int((weight / total_weight) * len(self.datasets) * 4),
                )
            return weights

        msg = f"Unknown sensor_weighting: {self.sensor_weighting}"
        raise ValueError(msg)

    def _prepare_distributed_indices(self) -> dict[str, np.ndarray]:
        """Prepare indices for distributed training."""
        # Find max length and normalize
        max_len = max(len(indices) for indices in self.sensor_indices.values())

        adjusted = {}
        for sensor in self.datasets:
            indices = np.array(self.sensor_indices[sensor])
            if len(indices) < max_len:
                repetitions = (max_len // len(indices)) + 1
                extended_indices = np.tile(indices, repetitions)[:max_len]
                adjusted[sensor] = extended_indices
            else:
                adjusted[sensor] = indices[:max_len]

        return adjusted

    def __iter__(self) -> Iterator[list[int]]:
        """Generate batches for this rank."""
        rng = np.random.default_rng(self.epoch)

        # Prepare per-rank data for each sensor
        sensor_rank_data = {}
        for sensor, indices in self.adjusted_indices.items():
            if self.shuffle:
                rng.shuffle(indices)

            # Distribute indices across ranks
            samples_per_rank = len(indices) // self.num_replicas
            start_idx = self.rank * samples_per_rank
            end_idx = start_idx + samples_per_rank
            sensor_rank_data[sensor] = indices[start_idx:end_idx]

        # Create weighted sensor cycle
        weighted_sensors = []
        for sensor, weight in self.sensor_weights.items():
            weighted_sensors.extend([sensor] * weight)
        rng.shuffle(weighted_sensors)

        # Generate batches for this rank
        max_samples_per_rank = max(len(data) for data in sensor_rank_data.values())
        batch_positions = dict.fromkeys(self.datasets.keys(), 0)

        for _ in range(0, max_samples_per_rank, self.batch_size):
            for sensor in weighted_sensors:
                start_idx = batch_positions[sensor]
                end_idx = start_idx + self.batch_size

                if end_idx <= len(sensor_rank_data[sensor]):
                    batch_indices = sensor_rank_data[sensor][start_idx:end_idx].tolist()
                    batch_positions[sensor] = end_idx

                    if len(batch_indices) == self.batch_size or not self.drop_last:
                        yield batch_indices

    def __len__(self) -> int:
        """Return number of batches for this rank."""
        return self.num_samples // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for proper shuffling in distributed training."""
        self.epoch = epoch


def create_round_robin_sampler(
    datasets: dict[str, Dataset],
    batch_size: int,
    sensor_weighting: str = "equal",
    custom_weights: dict[str, float] | None = None,
    *,
    distributed: bool = False,
    **kwargs: object,
) -> RoundRobinSampler | RoundRobinDistributedSampler:
    """Sampler based on training mode."""
    if distributed:
        return RoundRobinDistributedSampler(
            datasets=datasets,
            batch_size=batch_size,
            sensor_weighting=sensor_weighting,
            custom_weights=custom_weights,
            **kwargs,
        )
    return RoundRobinSampler(
        datasets=datasets,
        batch_size=batch_size,
        sensor_weighting=sensor_weighting,
        custom_weights=custom_weights,
        **kwargs,
    )
