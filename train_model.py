import torch
import os
import json
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.model import SegformerSegmentationModel
from src.train import Trainer
from src.data_processing import DeepGlobeDataset

NUM_CLASSES = 7

# Directories for images and masks
train_image_dir = 'data/raw/train/'
train_mask_dir = 'data/raw/train/'
test_image_dir = 'data/raw/test/'

# Normalization parameters (ImageNet statistics for pretrained models)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Define transformations for the images
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Define transformations for the masks
mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Initialize datasets
full_train_dataset = DeepGlobeDataset(
    image_dir=train_image_dir,
    mask_dir=train_mask_dir,
    transform=image_transform,
    mask_transform=mask_transform,
    use_masks=True
)

# Splitting the dataset into train and validation
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Initialize DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Test dataset (without masks)
test_dataset = DeepGlobeDataset(
    image_dir=test_image_dir,
    transform=image_transform,
    use_masks=False
)
test_loader = DataLoader(test_dataset, batch_size=4)

# Initialize the Segformer model
model = SegformerSegmentationModel(model_name="nvidia/segformer-b0-finetuned-ade-512-512", num_classes=NUM_CLASSES)

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize the Trainer with model, dataloaders, and other parameters
trainer = Trainer(model=model, dataloader_train=train_loader, dataloader_val=val_loader, device=device, num_classes=NUM_CLASSES)

# Train the model for a specified number of epochs
num_epochs = 5
history = trainer.train(num_epochs=num_epochs)

# Save the trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/segformer_trained.pth")
print("Training completed. Model saved as 'models/segformer_trained.pth'.")

# Optionally, save the training history
with open('training_history.json', 'w') as f:
    json.dump(history, f)
print("Training history saved as 'training_history.json'.")

# Perform inference on the test dataset
def test_inference(model, dataloader, device):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            outputs = model(images)
            all_outputs.append(outputs.cpu())  # Assuming you want to process outputs on CPU
    return all_outputs

# Perform inference on the test dataset
test_outputs = test_inference(model, test_loader, device)
print("Inference on test data completed.")
