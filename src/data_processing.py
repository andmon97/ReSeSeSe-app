from PIL import Image
import os
import torch
import numpy as np
from torchvision import transforms

class DeepGlobeDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, mask_transform=None, use_masks=True):
        """
        Initializes the dataset with directory information and transformations.

        Args:
            image_dir (str): Directory with all the satellite images.
            mask_dir (str, optional): Directory with all the mask images, if available.
            transform (callable, optional): Optional transform to be applied on a sample image.
            mask_transform (callable, optional): Optional transform to be applied on a mask.
            use_masks (bool): Indicates whether masks are available and should be used.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
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
            mask = Image.open(mask_path).convert("RGB")  # Ensure conversion to RGB
            if self.mask_transform:
                mask = self.mask_transform(mask)
            
            # Convert RGB mask to class indices, and ensure it's a long tensor
            mask = self.rgb_to_class(mask)
            if not isinstance(mask, torch.LongTensor):
                mask = mask.long()  # Ensure mask is long type for loss calculation

            return image, mask
        else:
            # Return image and a dummy tensor if no mask is used
            return image, torch.tensor([], dtype=torch.long)

    def rgb_to_class(self, mask):
        """
        Converts an RGB mask image to a class label map. This method assumes
        that the mask is a PIL Image in RGB format and converts it to a PyTorch tensor of class indices.
        
        Args:
            mask (PIL.Image): Mask in RGB format.

        Returns:
            torch.Tensor: Mask tensor with class indices.
        """
        # Convert the mask from PIL to a NumPy array
        mask_np = np.array(mask)  # This should create a (height, width, channels) array

        # Check if the mask_np is in the correct shape, if not transpose it
        if mask_np.shape[-1] != 3:  # This checks if the channels are not the last dimension
            # Assuming mask_np is in (channels, height, width), transpose to (height, width, channels)
            mask_np = np.transpose(mask_np, (1, 2, 0))

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

        # Initialize a class mask with zeros
        class_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int32)

        # Assign class indices based on RGB values
        for rgb, class_index in mapping.items():
            matches = np.all(mask_np == np.array(rgb, dtype=np.uint8), axis=-1)
            class_mask[matches] = class_index

        return torch.from_numpy(class_mask).long()
