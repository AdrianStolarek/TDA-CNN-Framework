import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class FungiDataset(Dataset):
    def __init__(self, images, labels, transform=None, is_tda=False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.is_tda = is_tda

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            if not self.is_tda:  # Raw data
                img = Image.fromarray(image.astype(np.uint8))
                image = self.transform(img)
            else:  # TDA data (6 channels)
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=(500, 500),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                image = image / 255.0
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(500, 500),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            if not self.is_tda:
                image = image / 255.0

        return image, torch.tensor(label, dtype=torch.long)


class FungiModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super(FungiModel, self).__init__()

        self.input_size = 500

        self.model = models.efficientnet_v2_s(
            weights='DEFAULT' if in_channels == 3 else None)

        if in_channels != 3:
            orig_weight = self.model.features[0][0].weight.data

            self.model.features[0][0] = nn.Conv2d(
                in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False
            )

            if in_channels > 3:
                with torch.no_grad():
                    self.model.features[0][0].weight[:,
                                                     :3, :, :].data = orig_weight

        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def get_input_size(self):
        """Zwraca oczekiwany rozmiar wejściowy modelu"""
        return (self.input_size, self.input_size)

    def forward(self, x):
        return self.model(x)


def get_transforms(data_type='raw'):
    """Get appropriate transforms for each data type"""

    if data_type == 'raw':
        train_transforms = v2.Compose([
            v2.Resize(256),
            v2.RandomResizedCrop(size=(500, 500), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomAffine(degrees=(-10, 10),
                            translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.RandomErasing(p=0.5, scale=(0.1, 0.15)),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])

        test_transforms = v2.Compose([
            v2.Resize((500, 500)),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = True
        test_transforms = True

    return train_transforms, test_transforms


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50, patience=5):
    """Train model with early stopping"""
    logs = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []
    }

    counter = 0
    best_loss = float('inf')

    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == target).sum().item()
            train_total += target.size(0)

        if scheduler:
            scheduler.step()

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                outputs = model(data)
                loss = criterion(outputs, target)

                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == target).sum().item()
                val_total += target.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        logs['train_loss'].append(train_loss)
        logs['train_acc'].append(train_acc)
        logs['val_loss'].append(val_loss)
        logs['val_acc'].append(val_acc)

        print(f'Epoch: {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]}')

        if val_loss < best_loss:
            counter = 0
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered!")
            break

    return logs


def test_model(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = criterion(outputs, target)

            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    test_loss = test_loss / total
    test_acc = correct / total

    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')

    return test_loss, test_acc


def plot_results(logs, title_suffix=""):
    """Plot training and validation curves"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(logs['train_loss'], label='Train Loss')
    plt.plot(logs['val_loss'], label='Validation Loss')
    plt.title(f'Train & Validation Loss {title_suffix}', fontsize=20)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(logs['train_acc'], label='Train Accuracy')
    plt.plot(logs['val_acc'], label='Validation Accuracy')
    plt.title(f'Train & Validation Accuracy {title_suffix}', fontsize=20)
    plt.legend()

    plt.tight_layout()

    save_path = f"training_curves{title_suffix.replace(' ', '_')}.png"
    plt.savefig(save_path)

    return save_path
