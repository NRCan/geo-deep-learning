"""Visualization tools."""

import numpy as np
import math
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

def visualize_prediction(  # noqa: PLR0913
    image: torch.Tensor,
    mask: torch.Tensor,
    prediction: torch.Tensor,
    *,
    sample_name: str | None = None,
    num_classes: int = 1,
    class_colors: list[str] | None = None,
    save_samples: bool = False,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize the input image, ground truth mask, and prediction mask side by side.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W)
        mask (torch.Tensor): Ground truth mask tensor of shape (H, W)
        prediction (torch.Tensor): Predicted mask tensor of shape (H, W)
        sample_name (str, optional): Name of the sample
        num_classes (int): Number of classes in the segmentation
        class_colors (list, optional): List of colors for each class
        save_samples (bool, optional): Whether to save the samples
        save_path (str, optional): Path to save the visualization

    Returns:
        plt.Figure: The figure containing the visualization

    """
    num_classes = num_classes + 1 if num_classes == 1 else num_classes
    image = image.cpu().numpy()
    mask = mask.squeeze(0).long().cpu().numpy()
    prediction = prediction.cpu().numpy()

    image = np.transpose(image, (1, 2, 0))
    num_channels = image.shape[-1]
    rgb_channels = 3
    if num_channels > rgb_channels:
        image = image[..., :rgb_channels]

    # Create a color map for the masks
    if class_colors is None:
        cmap = plt.cm.get_cmap("tab20")
    else:
        cmap = ListedColormap(class_colors)

    sample_name = "sample" if sample_name is None else sample_name

    if save_samples and save_path is not None:
        plt.imsave(save_path / f"{sample_name}_image.png", image)
        plt.imsave(
            save_path / f"{sample_name}_mask.png",
            mask,
            cmap=cmap,
            vmin=0,
            vmax=num_classes - 1,
        )
        plt.imsave(
            save_path / f"{sample_name}_prediction.png",
            prediction,
            cmap=cmap,
            vmin=0,
            vmax=num_classes - 1,
        )

    # Create the visualization
    plt.close("all")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes = axes.reshape(num_samples, 3) if num_samples > 1 else axes.reshape(1, 3)

    # for i in range(num_samples):
    ax_image, ax_mask, ax_output = axes

    # Plot original image
    ax_image.imshow(image)
    ax_image.set_title("Input Image")
    ax_image.axis("off")
    ax_image.text(
        0.5,
        -0.1,
        f"{sample_name}",
        transform=ax_image.transAxes,
        ha="center",
        va="top",
        wrap=True,
    )

    # Plot ground truth mask
    ax_mask.imshow(mask, cmap=cmap, vmin=0, vmax=num_classes - 1)
    ax_mask.set_title("Ground Truth Mask")
    ax_mask.axis("off")

    # Plot predicted mask
    ax_output.imshow(prediction, cmap=cmap, vmin=0, vmax=num_classes - 1)
    ax_output.set_title("Predicted Mask")
    ax_output.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close(fig)
    return fig


def get_mask_patch_size(model):
    """Get patch size based on model type."""
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'patch_size'):
        patch_size = model.patch_embed.patch_size
        # Handle both cases where patch_size is either a tuple/list or an integer
        if isinstance(patch_size, (tuple, list)):
            return patch_size[0] ** 2 * model.in_chans
        return patch_size ** 2 * model.in_chans
    elif hasattr(model, 'patch_size') and hasattr(model, 'in_chans'):
        patch_size = model.patch_size
        if isinstance(patch_size, (tuple, list)):
            return patch_size[0] ** 2 * model.in_chans
        return patch_size ** 2 * model.in_chans
    raise ValueError("Model patch size structure not recognized")


def create_visualization_mask(model, mask):
    """Create visualization mask based on model type."""
    try:
        patch_dim = get_mask_patch_size(model)
        mask = mask.detach()
        # Ensure mask has batch dimension: [num_patches] -> [1, num_patches]
        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0)  # [1, num_patches]
        # Expand mask to patch dimensions: (N, H*W, p*p*C)
        mask = mask.unsqueeze(-1).repeat(1, 1, patch_dim)  # (N, H*W, p*p*C)
        return model.unpatchify(mask)  # (N, C, H, W) - 1 is removing, 0 is keeping
    except Exception as e:
        logger.error(f"Error creating visualization mask: {str(e)}")
        raise


def visualize_pretrain(  # noqa: PLR0913
    image: torch.Tensor,
    mask: torch.Tensor,
    reconstructed: torch.Tensor,
    model: torch.nn.Module,
    *,
    sample_name: str | None = None,
    patch_size: int = 16,
    image_size: int | tuple[int, int] | None = None,
) -> plt.Figure:
    """
    Visualize the input image, mask visualization, and reconstructed image side by side.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W) or (N, C, H, W)
        mask (torch.Tensor): Mask tensor of shape (num_patches,) or (N, num_patches) where 1 means masked, 0 means visible
        reconstructed (torch.Tensor): Reconstructed image tensor of shape (C, H, W) or (N, C, H, W)
        model (torch.nn.Module): Model with unpatchify method to create visualization mask
        sample_name (str, optional): Name of the sample
        patch_size (int): Size of each patch (kept for compatibility, but inferred from model)
        image_size (int | tuple[int, int], optional): Image size. If None, inferred from image shape

    Returns:
        plt.Figure: The figure containing the visualization

    """
    # Move to CPU
    image = image.cpu()
    mask = mask.cpu()
    reconstructed = reconstructed.cpu()

    # Ensure batch dimension exists: add if missing
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # [1, C, H, W]
    if len(reconstructed.shape) == 3:
        reconstructed = reconstructed.unsqueeze(0)  # [1, C, H, W]

    # Create visualization mask
    vis_mask = create_visualization_mask(model, mask)  # [N, C, H, W]
    vis_mask = torch.einsum('nchw->nhwc', vis_mask).detach().cpu()

    # Prepare input tensor: convert to nhwc format
    x = torch.einsum('nchw->nhwc', image)  # [N, H, W, C]
    y = torch.einsum('nchw->nhwc', reconstructed)  # [N, H, W, C]

    # Create masked and reconstructed images
    im_masked = x * (1 - vis_mask)  # Masked input (visible parts only)
    im_paste = x * (1 - vis_mask) + y * vis_mask  # Reconstruction + visible parts

    sample_name = "sample" if sample_name is None else sample_name

    # Convert to numpy and scale to [0, 255] for display
    x_np = torch.clamp((x[0] * 255), 0, 255).int().numpy()
    y_np = torch.clamp((y[0] * 255), 0, 255).int().numpy()
    im_masked_np = torch.clamp((im_masked[0] * 255), 0, 255).int().numpy()
    im_paste_np = torch.clamp((im_paste[0] * 255), 0, 255).int().numpy()

    # Create the visualization
    plt.close("all")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    ax_original, ax_masked, ax_reconstruction, ax_paste = axes

    # Plot original image
    ax_original.imshow(x_np)
    ax_original.set_title("Original", fontsize=14, pad=10)
    ax_original.axis("off")

    # Plot masked image
    ax_masked.imshow(im_masked_np)
    ax_masked.set_title("Masked", fontsize=14, pad=10)
    ax_masked.axis("off")

    # Plot reconstruction
    ax_reconstruction.imshow(y_np)
    ax_reconstruction.set_title("Reconstruction", fontsize=14, pad=10)
    ax_reconstruction.axis("off")

    # Plot reconstruction + visible
    ax_paste.imshow(im_paste_np)
    ax_paste.set_title("Reconstruction + visible", fontsize=14, pad=10)
    ax_paste.axis("off")

    plt.tight_layout(pad=2.0)
    return fig


def visualize_building_classification(  # noqa: PLR0913
    image: torch.Tensor,
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    label: int,
    prediction: int | None = None,
    *,
    sample_name: str | None = None,
    class_labels: list[str] | None = None,
) -> plt.Figure:
    """
    Visualize the input image, binary mask, and image with bounding box and label.

    Args:
        image (torch.Tensor): Input RGB image tensor of shape (C, H, W)
        mask (torch.Tensor): Binary mask tensor of shape (1, H, W) or (H, W)
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
        label (int): Ground truth class label (0-indexed)
        prediction (int, optional): Predicted class label (0-indexed)
        sample_name (str, optional): Name of the sample
        class_labels (list, optional): List of class label names

    Returns:
        plt.Figure: The figure containing the visualization
    """
    # Convert to numpy
    image = image.cpu().numpy()
    mask = mask.squeeze(0).cpu().numpy() if len(mask.shape) == 3 else mask.cpu().numpy()
    
    # Convert image from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    num_channels = image.shape[-1]
    rgb_channels = 3
    if num_channels > rgb_channels:
        image = image[..., :rgb_channels]
    
    # Clamp image to [0, 1] range for visualization
    image = np.clip(image, 0, 1)
    
    # Create image with bounding box
    image_with_bbox = image.copy()
    x1, y1, x2, y2 = bbox
    
    # Get class label text - use class_labels if provided
    if class_labels is not None and len(class_labels) > 0:
        # Ensure label is within valid range
        if 0 <= label < len(class_labels):
            label_text = f"GT: {class_labels[label]}"
        else:
            label_text = f"GT: Class {label}"
    else:
        label_text = f"GT: Class {label}"
    
    if prediction is not None:
        if class_labels is not None and len(class_labels) > 0:
            # Ensure prediction is within valid range
            if 0 <= prediction < len(class_labels):
                pred_text = f"Pred: {class_labels[prediction]}"
            else:
                pred_text = f"Pred: Class {prediction}"
        else:
            pred_text = f"Pred: Class {prediction}"
        label_text = f"{label_text} | {pred_text}"
    
    sample_name = "sample" if sample_name is None else sample_name
    
    # Create the visualization
    plt.close("all")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_image, ax_mask, ax_bbox = axes
    
    # Plot original image
    ax_image.imshow(image)
    ax_image.set_title("Input Image", fontsize=14, pad=10)
    ax_image.axis("off")
    ax_image.text(
        0.5,
        -0.05,
        f"{sample_name}",
        transform=ax_image.transAxes,
        ha="center",
        va="top",
        wrap=True,
        fontsize=10,
    )
    
    # Plot binary mask
    ax_mask.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax_mask.set_title("Binary Mask", fontsize=14, pad=10)
    ax_mask.axis("off")
    
    # Plot image with bounding box
    ax_bbox.imshow(image_with_bbox)
    rect = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    ax_bbox.add_patch(rect)
    ax_bbox.set_title(f"{label_text}", fontsize=14, pad=10)
    ax_bbox.axis("off")
    
    plt.tight_layout(pad=2.0)
    return fig

