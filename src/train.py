import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .utils import pixel_accuracy, mIoU, plot_loss, plot_score, plot_acc
NUM_CLASSES = 6

class Trainer:
    def __init__(self, model, dataloader_train, dataloader_val, device, num_classes=NUM_CLASSES, lr=1e-4, weight_decay=1e-5):
        """
        Initialize the Trainer with the model, dataloaders, and optimization parameters.
        
        Args:
            model (nn.Module): The model to train.
            dataloader_train (DataLoader): Training dataset DataLoader.
            dataloader_val (DataLoader): Validation dataset DataLoader.
            device (torch.device): Device to use (CPU or GPU).
            num_classes (int): Number of output classes.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
        """
        self.model = model.to(device)
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.device = device
        self.num_classes = num_classes
        
        # Loss function: Cross-Entropy Loss for multi-class segmentation
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer: AdamW optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch.
        
        Args:
            epoch (int): Current epoch number.
        
        Returns:
            float: Training loss.
            float: Training pixel accuracy.
            float: Training mean IoU.
        """
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_miou = 0.0

        for batch in tqdm(self.dataloader_train, desc=f"Epoch {epoch+1} [Train]"):
            # Get inputs and labels
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Compute metrics
            running_loss += loss.item()
            running_acc += pixel_accuracy(outputs, labels)
            running_miou += mIoU(outputs, labels, n_classes=self.num_classes)
        
        avg_loss = running_loss / len(self.dataloader_train)
        avg_acc = running_acc / len(self.dataloader_train)
        avg_miou = running_miou / len(self.dataloader_train)

        return avg_loss, avg_acc, avg_miou

    def validate(self, epoch):
        """
        Validate the model on the validation dataset.
        
        Args:
            epoch (int): Current epoch number.
        
        Returns:
            float: Validation loss.
            float: Validation pixel accuracy.
            float: Validation mean IoU.
        """
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        running_miou = 0.0

        with torch.no_grad():
            for batch in tqdm(self.dataloader_val, desc=f"Epoch {epoch+1} [Validate]"):
                # Get inputs and labels
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.criterion(outputs, labels)

                # Compute metrics
                running_loss += loss.item()
                running_acc += pixel_accuracy(outputs, labels)
                running_miou += mIoU(outputs, labels, n_classes=self.num_classes)
        
        avg_loss = running_loss / len(self.dataloader_val)
        avg_acc = running_acc / len(self.dataloader_val)
        avg_miou = running_miou / len(self.dataloader_val)

        return avg_loss, avg_acc, avg_miou

    def train(self, num_epochs):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs (int): Number of epochs to train.
        
        Returns:
            dict: Training and validation loss, accuracy, and IoU history.
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_miou': [],
            'val_loss': [],
            'val_acc': [],
            'val_miou': []
        }

        for epoch in range(num_epochs):
            # Train one epoch
            train_loss, train_acc, train_miou = self.train_one_epoch(epoch)

            # Validate after every epoch
            val_loss, val_acc, val_miou = self.validate(epoch)

            # Store metrics in history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_miou'].append(train_miou)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_miou'].append(val_miou)

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train mIoU: {train_miou:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val mIoU: {val_miou:.4f}")
        
        return history
