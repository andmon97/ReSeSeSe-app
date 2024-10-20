import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor

NUM_CLASSES = 6

class SegformerSegmentationModel(nn.Module):
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512", num_classes=NUM_CLASSES):
        """
        Initializes the Segformer model for semantic segmentation.

        Args:
            model_name (str): The name of the pre-trained model to load from Hugging Face.
            num_classes (int): The number of output classes for the segmentation task.
        """
        super(SegformerSegmentationModel, self).__init__()
        
        # Load the pre-trained Segformer model
        # Load the pre-trained Segformer model, ignoring size mismatch for the classifier head
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, 
            num_labels=num_classes,
            ignore_mismatched_sizes=True  # This will ignore the size mismatch in the classifier head
        )
        
        # Load the image processor
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

    def forward(self, pixel_values):
        """
        Forward pass for the segmentation model.

        Args:
            pixel_values (torch.Tensor): Input tensor of pixel values.
        
        Returns:
            torch.Tensor: Output logits for segmentation.
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def process_image(self, image):
        """
        Processes a single image using the image processor.

        Args:
            image (PIL.Image or np.ndarray): Input image to be processed.
        
        Returns:
            dict: A dictionary with 'pixel_values' suitable for model input.
        """
        return self.image_processor(images=image, return_tensors="pt")

