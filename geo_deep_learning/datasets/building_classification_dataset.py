"""Building Classification Dataset."""

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import rasterio as rio
import torch
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from torch.utils.data import Dataset
from torchgeo.datasets import NonGeoDataset

from geo_deep_learning.utils.tensors import normalization

logger = logging.getLogger(__name__)


@rank_zero_only
def log_dataset(split: str, patch_count: int) -> None:
    """Log dataset."""
    logger.info("Created dataset for %s split with %s patches", split, patch_count)


class BuildingClassificationDataset(Dataset):
    """
    Dataset for building classification using full image + binary mask + bbox encoding.
    
    This dataset:
    1. Takes full RGB images and segmentation masks
    2. Creates binary masks for building detection
    3. Extracts bounding boxes for each building instance
    4. Processes one building at a time with bbox encoding
    """

    def __init__(
        self,
        csv_root_folder: str,
        patches_root_folder: str,
        split: str = "trn",
        norm_stats: dict[str, list[float]] | None = None,
        target_class: int | None = None,
        min_instance_area: int = 100,
    ) -> None:
        """
        Initialize the building classification dataset.

        Args:
            csv_root_folder: The root folder where the csv files are stored
            patches_root_folder: The root folder of image and mask patches
            split: Dataset split ("trn", "val", "tst")
            norm_stats: Normalization statistics
            target_class: If specified, only extract instances of this class (1-4)
            min_instance_area: Minimum area threshold for building instances
        """
        self.csv_root_folder = csv_root_folder
        self.patches_root_folder = patches_root_folder
        self.split = split
        self.norm_stats = norm_stats
        self.target_class = target_class
        self.min_instance_area = min_instance_area
        
        # Load image-mask pairs from CSV
        self.files = self._load_files()
        
        # Pre-process to get all building instances with their bboxes
        self.all_instances = self._preprocess_instances()
        
        log_dataset(self.split, len(self.all_instances))

    def _load_files(self) -> list[dict[str, Path]]:
        """Load image and mask paths from csv files."""
        csv_path = Path(self.csv_root_folder) / f"{self.split}.csv"
        if not csv_path.exists():
            msg = f"CSV file {csv_path} not found."
            raise FileNotFoundError(msg)
        df_csv = pd.read_csv(csv_path, header=None, sep=";")
        if len(df_csv.columns) < 2:
            msg = "CSV file must contain at least two columns: image_path;mask_path"
            raise ValueError(msg)

        return [
            {
                "image": Path(self.patches_root_folder) / img,
                "mask": Path(self.patches_root_folder) / lbl,
            }
            for img, lbl in df_csv[[0, 1]].itertuples(index=False)
        ]

    def _extract_building_instances_with_bboxes(
        self,
        mask: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Extract individual building instances and their bounding boxes from segmentation mask.
        
        Args:
            mask: Segmentation mask with class labels (1-4 for buildings, 0 for background)
            
        Returns:
            List of dictionaries containing bounding box and class label information
        """
        instances = []
        
        # Determine which classes to process
        if self.target_class is not None:
            class_labels = [self.target_class]
        else:
            class_labels = range(1, 5)  # Process classes 1-4
        
        # Process each class label
        for class_label in class_labels:
            # Create binary mask for this class
            class_mask = (mask == class_label).astype(np.uint8)
            
            # Skip if no pixels of this class
            if not np.any(class_mask):
                continue
                
            # Find connected components for this class
            num_labels, labels = cv2.connectedComponents(class_mask)
            
            # Process each instance of this class
            for label_id in range(1, num_labels):
                instance_mask = (labels == label_id).astype(np.uint8)
                contours, _ = cv2.findContours(
                    instance_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    area = cv2.contourArea(largest_contour)
                    
                    # Filter out very small instances
                    if area < self.min_instance_area:
                        continue
                    
                    instances.append({
                        "bbox": (x, y, x + w, y + h),  # (x1, y1, x2, y2)
                        "area": area,
                        "label": class_label,
                        "instance_mask": instance_mask,
                    })
        
        return instances

    def _preprocess_instances(self) -> list[dict[str, Any]]:
        """Pre-process all images to extract building instances with their bboxes."""
        all_instances = []
        
        for idx, file_pair in enumerate(self.files):
            image_path = file_pair["image"]
            mask_path = file_pair["mask"]
            image_id = image_path.stem
            
            logger.info("Processing image %d: %s", idx, image_path)
            
            # Load mask
            try:
                with rio.open(mask_path) as src:
                    mask = src.read(1)
                    # Convert to signed integer to handle negative ignore_index values
                    if mask.dtype in [np.uint8, np.uint16, np.uint32]:
                        mask = mask.astype(np.int32)
            except Exception as e:
                logger.error("Error reading mask %s: %s", mask_path, e)
                continue
            
            # Extract building instances with their bboxes
            instances = self._extract_building_instances_with_bboxes(mask)
            logger.info("Found %d instances in image %s", len(instances), image_id)
            
            # Add instances with their metadata
            for i, instance in enumerate(instances):
                instance_info = {
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "image_id": image_id,
                    "instance_id": i,
                    "bbox": instance["bbox"],
                    "area": instance["area"],
                    "label": instance["label"],
                    "instance_mask": instance["instance_mask"],
                }
                all_instances.append(instance_info)
        
        # Log class distribution
        if all_instances:
            class_counts = Counter(inst["label"] for inst in all_instances)
            logger.info("Class distribution in %s dataset:", self.split)
            for class_id in sorted(class_counts.keys()):
                logger.info("  Class %d: %d instances", class_id, class_counts[class_id])
        else:
            logger.error("No instances found in any image!")
            raise ValueError("Dataset is empty - no valid building instances found")
        
        return all_instances

    def __len__(self) -> int:
        """Return the total number of building instances in the dataset."""
        return len(self.all_instances)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a single building instance with full image, binary mask, and bbox info.
        
        Returns:
            Dictionary containing:
                - 'rgb_image': RGB image (3, H, W) normalized to [0, 1]
                - 'binary_mask': Binary mask (1, H, W)
                - 'bbox': Bounding box coordinates (x1, y1, x2, y2)
                - 'label': Instance label (class 1-4)
                - Additional metadata
        """
        instance = self.all_instances[idx]
        
        # Load full RGB image
        with rio.open(instance["image_path"]) as src:
            rgb_image = src.read().astype(np.float32)  # (C, H, W)
        
        # Load full segmentation mask
        with rio.open(instance["mask_path"]) as src:
            mask = src.read(1).astype(np.float32)  # (H, W)
        
        # Create binary mask for buildings (any non-zero class)
        binary_mask = (mask > 0).astype(np.float32)  # (H, W)
        binary_mask = np.expand_dims(binary_mask, axis=0)  # (1, H, W)
        
        # Convert to tensors
        rgb_image = torch.from_numpy(rgb_image)
        binary_mask = torch.from_numpy(binary_mask)
        
        # Normalize RGB image to [0, 1] range only
        # Standardization will be applied after augmentation in the model
        rgb_image = normalization(rgb_image)
        
        # Get image name
        image_name = Path(instance["image_path"]).name
        
        # Store normalization stats for later standardization in model
        mean = torch.tensor(self.norm_stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self.norm_stats["std"], dtype=torch.float32).view(-1, 1, 1)
        
        sample = {
            "rgb_image": rgb_image,
            "binary_mask": binary_mask,
            "bbox": instance["bbox"],
            "label": torch.tensor(instance["label"] - 1, dtype=torch.long),  # Convert 1-4 to 0-3
            "image_name": image_name,
            "image_id": instance["image_id"],
            "instance_id": instance["instance_id"],
            "area": instance["area"],
            "mean": mean,
            "std": std,
        }
        
        return sample


class BuildingClassificationOversamplingDataset(BuildingClassificationDataset):
    """
    Building classification dataset with oversampling support to balance classes.
    """

    def __init__(
        self,
        csv_root_folder: str,
        patches_root_folder: str,
        split: str = "trn",
        norm_stats: dict[str, list[float]] | None = None,
        target_class: int | None = None,
        min_instance_area: int = 100,
        oversampling_strategy: str = "balance",
    ) -> None:
        """
        Initialize the building classification dataset with oversampling.
        
        Args:
            csv_root_folder: The root folder where the csv files are stored
            patches_root_folder: The root folder of image and mask patches
            split: Dataset split ("trn", "val", "tst")
            norm_stats: Normalization statistics
            target_class: If specified, only extract instances of this class
            min_instance_area: Minimum area threshold for building instances
            oversampling_strategy: Strategy for oversampling ('balance', 'max', 'min')
        """
        self.oversampling_strategy = oversampling_strategy
        super().__init__(
            csv_root_folder=csv_root_folder,
            patches_root_folder=patches_root_folder,
            split=split,
            norm_stats=norm_stats,
            target_class=target_class,
            min_instance_area=min_instance_area,
        )
        
        # Apply oversampling to balance classes
        self.balanced_instances = self._apply_oversampling()
        logger.info(
            "Dataset initialized with %d original instances, %d after oversampling",
            len(self.all_instances),
            len(self.balanced_instances),
        )

    def _apply_oversampling(self) -> list[dict[str, Any]]:
        """
        Apply oversampling to balance classes in the dataset.
        
        Returns:
            List of balanced instances with oversampling applied
        """
        # Count instances per class
        class_counts = Counter(inst["label"] for inst in self.all_instances)
        logger.info("Original class distribution:")
        for class_id in sorted(class_counts.keys()):
            logger.info("  Class %d: %d instances", class_id, class_counts[class_id])
        
        # Determine target count based on strategy
        if self.oversampling_strategy == "balance":
            # Balance to the average of all classes
            target_count = int(np.mean(list(class_counts.values())))
        elif self.oversampling_strategy == "max":
            # Balance to the maximum class count
            target_count = max(class_counts.values())
        elif self.oversampling_strategy == "min":
            # Balance to the minimum class count
            target_count = min(class_counts.values())
        else:
            target_count = int(np.mean(list(class_counts.values())))
        
        logger.info("Target instances per class: %d", target_count)
        
        # Group instances by class
        instances_by_class = defaultdict(list)
        for instance in self.all_instances:
            instances_by_class[instance["label"]].append(instance)
        
        # Apply oversampling/downsampling
        balanced_instances = []
        for class_id in sorted(instances_by_class.keys()):
            class_instances = instances_by_class[class_id]
            current_count = len(class_instances)
            
            if current_count == 0:
                continue
            
            if current_count < target_count:
                # Oversample by repeating instances
                oversample_factor = target_count // current_count
                remainder = target_count % current_count
                
                # Add full repetitions
                for _ in range(oversample_factor):
                    balanced_instances.extend(class_instances)
                
                # Add remainder instances randomly
                if remainder > 0:
                    import random
                    additional_instances = random.sample(class_instances, remainder)
                    balanced_instances.extend(additional_instances)
                
                logger.info(
                    "Class %d: %d -> %d (oversampled)",
                    class_id,
                    current_count,
                    target_count,
                )
            elif current_count > target_count:
                # Downsample by randomly selecting instances
                import random
                selected_instances = random.sample(class_instances, target_count)
                balanced_instances.extend(selected_instances)
                logger.info(
                    "Class %d: %d -> %d (downsampled)",
                    class_id,
                    current_count,
                    target_count,
                )
            else:
                # Use all instances if we have exactly the target count
                balanced_instances.extend(class_instances)
                logger.info("Class %d: %d instances (no change needed)", class_id, current_count)
        
        # Shuffle the balanced dataset
        import random
        random.shuffle(balanced_instances)
        
        # Log final distribution
        final_class_counts = Counter(inst["label"] for inst in balanced_instances)
        logger.info("Final balanced class distribution:")
        for class_id in sorted(final_class_counts.keys()):
            logger.info(
                "  Class %d: %d instances",
                class_id,
                final_class_counts[class_id],
            )
        
        return balanced_instances

    def __len__(self) -> int:
        """Return the total number of building instances in the balanced dataset."""
        return len(self.balanced_instances)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single building instance from the balanced dataset."""
        instance = self.balanced_instances[idx]
        
        # Load full RGB image
        with rio.open(instance["image_path"]) as src:
            rgb_image = src.read().astype(np.float32)  # (C, H, W)
        
        # Load full segmentation mask
        with rio.open(instance["mask_path"]) as src:
            mask = src.read(1).astype(np.float32)  # (H, W)
        
        # Create binary mask for buildings (any non-zero class)
        binary_mask = (mask > 0).astype(np.float32)  # (H, W)
        binary_mask = np.expand_dims(binary_mask, axis=0)  # (1, H, W)
        
        # Convert to tensors
        rgb_image = torch.from_numpy(rgb_image)
        binary_mask = torch.from_numpy(binary_mask)
        
        # Normalize RGB image to [0, 1] range only
        # Standardization will be applied after augmentation in the model
        rgb_image = normalization(rgb_image)
        
        # Get image name
        image_name = Path(instance["image_path"]).name
        
        # Store normalization stats for later standardization in model
        mean = torch.tensor(self.norm_stats["mean"], dtype=torch.float32)
        std = torch.tensor(self.norm_stats["std"], dtype=torch.float32)
        
        sample = {
            "rgb_image": rgb_image,
            "binary_mask": binary_mask,
            "bbox": instance["bbox"],
            "label": torch.tensor(instance["label"] - 1, dtype=torch.long),  # Convert 1-4 to 0-3
            "image_name": image_name,
            "image_id": instance["image_id"],
            "instance_id": instance["instance_id"],
            "area": instance["area"],
            "mean": mean,
            "std": std,
        }
        
        return sample


def collate_fn_building_classification(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate function for building classification dataset.
    
    Handles batching of RGB images, binary masks, and bounding boxes.
    """
    # Stack tensors
    rgb_images = torch.stack([item["rgb_image"] for item in batch])
    binary_masks = torch.stack([item["binary_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    
    # Keep bboxes as list (they're processed individually in the model)
    bboxes = [item["bbox"] for item in batch]
    
    # Create batch dictionary
    batch_dict = {
        "rgb_image": rgb_images,
        "binary_mask": binary_masks,
        "bbox": bboxes,
        "label": labels,
    }
    
    # Add metadata
    for key in ["image_name", "image_id", "instance_id", "area", "mean", "std"]:
        if key in batch[0]:
            if key in ["mean", "std"]:
                batch_dict[key] = torch.stack([item[key] for item in batch])
            else:
                batch_dict[key] = [item[key] for item in batch]
    
    return batch_dict


if __name__ == "__main__":
    csv_root_folder = ""
    patches_root_folder = csv_root_folder
    dataset = BuildingClassificationDataset(
        csv_root_folder,
        patches_root_folder,
        split="val",
    )
    print(f"Dataset size: {len(dataset)}")
