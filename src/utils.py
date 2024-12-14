import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

NUM_CLASSES = 7
DEFAULT_SMOOTH = 1e-10

def pixel_accuracy(output, mask):
    """
    Calculate the pixel-wise accuracy between the predicted output and the ground truth mask.
    
    Args:
        output (torch.Tensor): The model output logits.
        mask (torch.Tensor): The ground truth mask.
    
    Returns:
        float: Pixel-wise accuracy.
    """
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, smooth=DEFAULT_SMOOTH, n_classes=NUM_CLASSES):
    """
    Calculate the mean Intersection over Union (mIoU) for multi-class segmentation.
    
    Args:
        pred_mask (torch.Tensor): The model output logits.
        mask (torch.Tensor): The ground truth mask.
        smooth (float): Smoothing factor to avoid division by zero.
        n_classes (int): Number of classes in the segmentation task.
    
    Returns:
        float: The mean IoU across all classes.
    """
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(n_classes):  # loop through each class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # no class in the current loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()
                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

def plot_loss(history):
    """
    Plot training and validation loss over epochs.
    
    Args:
        history (dict): Dictionary containing 'train_loss' and 'val_loss' for each epoch.
    """
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, 20, 1))
    plt.legend()
    plt.grid()
    plt.show()

def plot_score(history):
    """
    Plot training and validation mean IoU over epochs.
    
    Args:
        history (dict): Dictionary containing 'train_miou' and 'val_miou' for each epoch.
    """
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['val_miou'], label='val_mIoU', marker='*')
    plt.title('Mean IoU per epoch')
    plt.ylabel('Mean IoU')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, 20, 1))
    plt.legend()
    plt.grid()
    plt.show()

def plot_acc(history):
    """
    Plot training and validation accuracy over epochs.
    
    Args:
        history (dict): Dictionary containing 'train_acc' and 'val_acc' for each epoch.
    """
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy', marker='*')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, 20, 1))
    plt.legend()
    plt.grid()
    plt.show()

def summarize_performance(history):
    """
    Summarizes the performance of the training and validation process by plotting relevant metrics.
    
    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', 'train_miou', 'val_miou', 'train_acc', 'val_acc'.
    """
    print("Final Training Loss: {:.4f}".format(history['train_loss'][-1]))
    print("Final Validation Loss: {:.4f}".format(history['val_loss'][-1]))
    print("Final Training mIoU: {:.4f}".format(history['train_miou'][-1]))
    print("Final Validation mIoU: {:.4f}".format(history['val_miou'][-1]))
    print("Final Training Accuracy: {:.4f}".format(history['train_acc'][-1]))
    print("Final Validation Accuracy: {:.4f}".format(history['val_acc'][-1]))

    plot_loss(history)
    plot_score(history)
    plot_acc(history)

def plot_images_and_masks(images, masks, predictions, num_images=2):
    """
    Plot a comparison of images, their true masks, and predicted masks.
    
    Args:
        images (list of torch.Tensor): List of images.
        masks (list of torch.Tensor): List of true segmentation masks.
        predictions (list of torch.Tensor): List of predicted masks.
        num_images (int): Number of images to plot.
    """
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    for i in range(num_images):
        axes[i, 0].imshow(to_pil_image(images[i]))
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(to_pil_image(masks[i]), cmap='gray')
        axes[i, 1].set_title('True Mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(to_pil_image(predictions[i]), cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()
