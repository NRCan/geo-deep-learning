import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_prediction(image, 
                         mask, 
                         prediction,
                         sample_name=None,
                         num_classes=1,
                         class_colors=None,
                         save_samples=False, 
                         save_path=None):
    """
    Visualize the input image, ground truth mask, and prediction mask side by side.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W)
        mask (torch.Tensor): Ground truth mask tensor of shape (H, W)
        prediction (torch.Tensor): Predicted mask tensor of shape (H, W)
        num_classes (int): Number of classes in the segmentation
        class_colors (list, optional): List of colors for each class
        save_path (str, optional): Path to save the visualization
    """
    num_classes = num_classes + 1 if num_classes == 1 else num_classes
    image = image.cpu().numpy()
    mask = mask.squeeze(0).long().cpu().numpy()
    prediction = prediction.cpu().numpy()
    
    image = np.transpose(image, (1, 2, 0))
    num_channels = image.shape[-1]
    if num_channels > 3:
        image = image[..., :3]
    
    # Create a color map for the masks
    if class_colors is None:
        cmap = plt.cm.get_cmap('tab20')
    else:
        cmap = ListedColormap(class_colors)
        
    sample_name = "sample" if sample_name is None else sample_name
    
    if save_samples and save_path is not None:
        plt.imsave(save_path / f"{sample_name}_image.png", image)
        plt.imsave(save_path / f"{sample_name}_mask.png", mask, cmap=cmap, vmin=0, vmax=num_classes-1)
        plt.imsave(save_path / f"{sample_name}_prediction.png", prediction, cmap=cmap, vmin=0, vmax=num_classes-1)
    
    # Create the visualization
    plt.close('all')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes = axes.reshape(num_samples, 3) if num_samples > 1 else axes.reshape(1, 3)
    
    # for i in range(num_samples):
    ax_image, ax_mask, ax_output = axes

    # Plot original image
    ax_image.imshow(image)
    ax_image.set_title('Input Image')
    ax_image.axis('off')
    ax_image.text(0.5,
                -0.1, 
                f"{sample_name}", 
                transform=ax_image.transAxes,
                ha='center', va='top', wrap=True)
    
    # Plot ground truth mask
    ax_mask.imshow(mask, cmap=cmap, vmin=0, vmax=num_classes-1)
    ax_mask.set_title('Ground Truth Mask')
    ax_mask.axis('off')
    
    # Plot predicted mask
    ax_output.imshow(prediction, cmap=cmap, vmin=0, vmax=num_classes-1)
    ax_output.set_title('Predicted Mask')
    ax_output.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)
    return fig