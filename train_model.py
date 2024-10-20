import torch
from torch.utils.data import DataLoader 
from torchvision import transforms
import os

from src.model import SegformerSegmentationModel
from src.train import Trainer
from src.data_processing import DeepGlobeDataset

NUM_CLASSES = 6

# Directories for images and masks
train_image_dir = 'data/raw/train/'
train_mask_dir = 'data/raw/train/'
val_image_dir = 'data/raw/valid/'
val_mask_dir = 'data/raw/valid/' 

# Normalization parameters (ImageNet statistics for pretrained models)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Define transformations for images (normalizing with ImageNet stats) and masks
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

mask_transform = transforms.ToTensor()  # Convert masks to tensors (you can add mask transformations as needed)

train_dataset = DeepGlobeDataset(image_dir=train_image_dir, mask_dir=train_mask_dir, 
                                 transform=image_transform, mask_transform=mask_transform)
val_dataset = DeepGlobeDataset(image_dir=val_image_dir, mask_dir=val_mask_dir, 
                               transform=image_transform, mask_transform=mask_transform)

# DataLoader for batching the datasets
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Initialize the Segformer model (choose model version and specify number of output classes)
model = SegformerSegmentationModel(model_name="nvidia/segformer-b0-finetuned-ade-512-512", num_classes=NUM_CLASSES)

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the device



# Initialize the Trainer with model, dataloaders, and other parameters
trainer = Trainer(model=model, dataloader_train=train_loader, dataloader_val=val_loader, device=device, num_classes=NUM_CLASSES)

# Train the model for a specified number of epochs
num_epochs = 5
history = trainer.train(num_epochs=num_epochs)

# Save the trained model to a file
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/segformer_trained.pth")
print("Training completed. Model saved as 'models/segformer_trained.pth'.")

# Optionally, you can save the training history (metrics) if needed
# For example, saving the loss/accuracy per epoch to a file
import json
with open('training_history.json', 'w') as f:
    json.dump(history, f)
print("Training history saved as 'training_history.json'.")