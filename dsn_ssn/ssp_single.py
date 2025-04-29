import os
import random
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist.dataset import OCTMNIST
from transformers import ViTModel
from sklearn.preprocessing import StandardScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from fvcore.nn import FlopCountAnalysis
import warnings
from tqdm import tqdm  

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
use_distributed = False  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
seed = 42
batch_size = 128  
epochs = 100 
learning_rate = 1e-4
accumulation_steps = 8  
output_directory = "./ssp_retinaloct_tbme/vit2spn_tiny/dsn_ssn/"
output_pretrained_model_path = os.path.join(output_directory, "octmnist_vitspn_tiny_model.pth")

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Set random seed for reproducibility
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# CUDA Optimization
torch.backends.cudnn.benchmark = True 
torch.backends.cudnn.enabled = True 

# Save and Load Checkpoints
def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, checkpoint['epoch'], checkpoint['loss']
    return model, optimizer, 0, float('inf')

# Data Augmentation
strong_augment_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# OCTMNIST Dataset to Return Two Views
class OCTMNIST(OCTMNIST):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        view1 = strong_augment_transform(image)
        view2 = strong_augment_transform(image)
        return (view1, view2), label

# Load OCTMNIST Dataset
def load_octmnist_data(transform, batch_size):
    dataset = OCTMNIST(split="train", download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

unlabeled_dataloader = load_octmnist_data(transform=strong_augment_transform, batch_size=batch_size)

class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("WinKawaks/vit-tiny-patch16-224", output_hidden_states=True)

    def forward(self, x):
        output = self.vit(x)
        return output.hidden_states[-1].mean(dim=1)

class SingleStreamNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.online_network = ViTBackbone()
        self.target_network = ViTBackbone()

        # Freeze target network parameters
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.projection_head = nn.Sequential(
            nn.Linear(192, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 128)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, view1, view2):
        feat_online = self.online_network(view1)
        with torch.no_grad():
            feat_target = self.target_network(view2)

        online_proj_feat = self.projection_head(feat_online)
        online_pred_feat = self.prediction_head(online_proj_feat)
        target_proj_feat = self.projection_head(feat_target).detach()

        return online_pred_feat, target_proj_feat

    def update_target_network(self, momentum=0.99):
        for param, target_param in zip(self.online_network.parameters(), self.target_network.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * param.data

# Extract Online Features
def extract_online_features(model, dataloader):
    model.eval()
    combined_features = []
    labels = []

    with torch.no_grad():
        for (views, lbls) in dataloader:
            view1, view2 = views
            view1, view2 = view1.to(device), view2.to(device)
            online_feats, _ = model(view1, view2)
            combined_features.append(online_feats.cpu().numpy())
            labels.append(lbls.numpy())

    combined_features = np.concatenate(combined_features, axis=0)
    labels = np.concatenate(labels, axis=0)

# Initialize model
model = SingleStreamNetwork().to(device)
if use_distributed:
    model = DDP(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CosineSimilarity(dim=1)
scaler = GradScaler()

# Log GPU memory usage
def log_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e9  
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU Memory - Allocated: {allocated:.4f} GB, Reserved: {reserved:.4f} GB")

# Compute FLOPs (GFLOPs) 
dummy_input = torch.randn(1, 3, 224, 224).to(device)
flops = FlopCountAnalysis(model.online_network, dummy_input)
print(f"Model FLOPs: {flops.total() / 1e9:.4f} GFLOPs")

# Training Function with Loss Logging
def train_self_supervised(model, dataloader, epochs, optimizer, criterion, checkpoint_path="checkpoint.pth"):
    model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)

    model.train()
    loss_history = []  
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        for i, (views, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            view1, view2 = views
            view1, view2 = view1.to(device), view2.to(device)

            with autocast():
                online_pred_feat, target_proj_feat = model(view1, view2)
                loss = -torch.mean(criterion(online_pred_feat, target_proj_feat)) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                model.update_target_network() 
            epoch_loss += loss.item() * accumulation_steps

            # Clear CUDA memory every 10 iterations
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_epoch_loss, checkpoint_path)

# The number of parameters 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total parameters: {total_params}")

# Start Training
checkpoint_path = os.path.join(output_directory, "octmnist_vitspn_tiny_checkpoint.pth")
train_self_supervised(model, unlabeled_dataloader, epochs, optimizer, criterion, checkpoint_path)

# Save the pretrained model state without .module
torch.save(model.online_network.state_dict(), output_pretrained_model_path)  
print(f"Pretrained model saved at {output_pretrained_model_path}")