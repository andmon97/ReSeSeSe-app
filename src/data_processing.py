import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

NUM_CLASSES = 6

class DeepGlobeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        """
        Args:
            image_dir (str): Directory with all the satellite images.
            mask_dir (str): Directory with all the mask images.
            transform (callable, optional): Optional transform to be applied on a sample image.
            mask_transform (callable, optional): Optional transform to be applied on the mask.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_ids = [f.split('_sat.jpg')[0] for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Fetches the image and mask at index `idx` and applies the necessary transformations.
        
        Args:
            idx (int): Index of the image/mask pair.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image and mask tensors.
        """
        # Get the image and mask paths
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}_sat.jpg")
        mask_path = os.path.join(self.mask_dir, f"{image_id}_mask.png")

        # Open the image and mask using PIL
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")  # Make sure this is supposed to be RGB

        # Apply transformations to the image and mask
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Convert mask RGB to class indices (add this logic if needed)
        mask = self.rgb_to_class(mask)
        # After loading and transforming your mask to tensor
        mask.clamp_(min=0, max=NUM_CLASSES-1)

        return image, mask
    
    def rgb_to_class(self, mask):
        """
        Converts the RGB mask image to a class label map. This method assumes
        that the mask is a PIL Image and converts it to a PyTorch tensor.

        Returns:
            torch.Tensor: Mask tensor with class indices.
        """
        # Convert the mask to a NumPy array
        mask = np.array(mask)

        # Ensure the mask is in channels-last format (height, width, 3)
        if mask.shape[0] == 3 and len(mask.shape) == 3:  # If mask is in (3, height, width) format
            mask = np.transpose(mask, (1, 2, 0))  # Convert to (height, width, 3)

        # Class label map logic:
        mapping = {
            (0, 255, 255): 0,   # Urban land
            (255, 255, 0): 1,   # Agriculture land
            (255, 0, 255): 2,   # Rangeland
            (0, 255, 0): 3,     # Forest land
            (0, 0, 255): 4,     # Water
            (255, 255, 255): 5, # Barren land
            (0, 0, 0): 6        # Unknown
        }

        # Create an output mask where each pixel is replaced by the corresponding class index
        output = np.zeros(mask.shape[:2], dtype=np.uint8)  # (height, width)

        for rgb, class_index in mapping.items():
            matches = np.all(mask == rgb, axis=-1)
            output[matches] = class_index

        return torch.from_numpy(output).long()

