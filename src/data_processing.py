import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DeepGlobeDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, mask_transform=None, use_masks=True):
        """
        Args:
            image_dir (str): Directory with all the satellite images.
            mask_dir (str, optional): Directory with all the mask images, if available.
            transform (callable, optional): Optional transform to be applied on a sample image.
            mask_transform (callable, optional): Optional transform to be applied on the mask.
            use_masks (bool): Flag to indicate whether masks are available and should be used.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transforms
        self.mask_transform = mask_transform
        self.use_masks = use_masks
        self.image_ids = [f.split('_sat.jpg')[0] for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Fetches the image and mask at index `idx` and applies the necessary transformations.
        
        Args:
            idx (int): Index of the image/mask pair.
        
        Returns:
            image: Transformed image tensor.
            mask (optional): Transformed mask tensor if masks are used, otherwise None.
        """
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}_sat.jpg")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.use_masks and self.mask_dir:
            mask_path = os.path.join(self.mask_dir, f"{image_id}_mask.png")
            mask = Image.open(mask_path).convert("RGB")  # Adjust if your masks are not RGB
            if self.mask_transform:
                mask = self.mask_transform(mask)
            mask = self.rgb_to_class(mask)
            return image, mask
        else:
            return image, None  # Return None or a dummy tensor for mask if masks are not used

    def rgb_to_class(self, mask):
        """
        Converts an RGB mask image to a class label map. This method assumes
        that the mask is a PIL Image and converts it to a PyTorch tensor.
        
        Returns:
            torch.Tensor: Mask tensor with class indices.
        """
        mask = np.array(mask)  # Convert PIL image to numpy array

        # Define the mapping from RGB to class indices
        mapping = {
            (0, 255, 255): 0,   # Urban land
            (255, 255, 0): 1,   # Agriculture land
            (255, 0, 255): 2,   # Rangeland
            (0, 255, 0): 3,     # Forest land
            (0, 0, 255): 4,     # Water
            (255, 255, 255): 5, # Barren land
            (0, 0, 0): 6        # Unknown
        }
        output = np.zeros(mask.shape[:2], dtype=np.uint8)  # Initialize the class label matrix

        # Apply the mapping to the mask
        for rgb, class_index in mapping.items():
            output[(mask == rgb).all(axis=-1)] = class_index

        return torch.from_numpy(output).long()  # Convert numpy array to PyTorch tensor
