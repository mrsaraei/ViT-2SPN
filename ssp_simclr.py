import os
import random
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import resnet50
from torch.cuda.amp import autocast, GradScaler
import warnings
from medmnist.dataset import OCTMNIST
import matplotlib.pyplot as plt  

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
seed = 42
batch_size = 128
epochs = 50
learning_rate = 1e-4
momentum = 0.999
accumulation_steps = 4  # Gradient accumulation steps
output_directory = "./ssp_retinaloct_midl2025/simclr"
output_pretrained_model_path = os.path.join(output_directory, "octmnist_simclr_model.pth")

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Reproducibility
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Data Augmentation
class DualViewTransform:
    def __init__(self, augment_transform):
        self.augment_transform = augment_transform

    def __call__(self, x):
        view1 = self.augment_transform(x)
        view2 = self.augment_transform(x)
        return view1, view2

strong_augment_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dual_view_transform = DualViewTransform(strong_augment_transform)

# Load OCTMNIST Dataset
def load_octmnist_data(transform, batch_size):
    dataset = OCTMNIST(split="train", transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

unlabeled_dataloader = load_octmnist_data(transform=dual_view_transform, batch_size=batch_size)

# ResNet Backbone for SimCLR
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.flatten(x)

# SimCLR Model with ResNet50 Backbone
class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet_backbone = ResNetBackbone()
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 128)
        )

    def forward(self, x1, x2):
        feat1 = self.resnet_backbone(x1)
        feat2 = self.resnet_backbone(x2)

        proj_feat1 = self.projection_head(feat1)
        proj_feat2 = self.projection_head(feat2)

        return proj_feat1, proj_feat2

# Initialize Model, Optimizer, and Criterion
model = SimCLRModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CosineSimilarity(dim=1)
scaler = GradScaler()

# Training Function with Loss Logging
def train_self_supervised(model, dataloader, epochs, optimizer, criterion):
    model.train()
    loss_history = []  # To store loss values for each epoch
    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        for i, (views, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            view1, view2 = views
            view1, view2 = view1.to(device), view2.to(device)

            with autocast():
                proj_feat1, proj_feat2 = model(view1, view2)
                loss = -torch.mean(criterion(proj_feat1, proj_feat2)) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps

        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, linestyle='-', color='black')
    plt.title("Loss Function During Self-Supervised Pretraining")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.grid(True)
    plt.savefig('./ssp_retinaloct_midl2025/simclr/result/pretraining_loss_curv.png')
    plt.show()

# Start Training
print("Starting self-supervised pretraining with SimCLR...")
train_self_supervised(model, unlabeled_dataloader, epochs, optimizer, criterion)
torch.save(model.state_dict(), output_pretrained_model_path)
print(f"Pretrained model saved at {output_pretrained_model_path}")