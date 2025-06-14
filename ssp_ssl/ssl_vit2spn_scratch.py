import os
import random
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import warnings
from medmnist.dataset import OCTMNIST
import matplotlib.pyplot as plt  
from transformers import ViTModel, ViTConfig
from fvcore.nn import FlopCountAnalysis
import torch.distributed as dist

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
seed = 42
batch_size = 128  
epochs = 100
learning_rate = 1e-4
momentum = 0.999
accumulation_steps = 8  
output_directory = "./ssp_retinaloct_tbme/vit2spn_tiny/scratch/"
output_pretrained_model_path = os.path.join(output_directory, "octmnist_vit2spn_tiny_scratch_model.pth")

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Reproducibility
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

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
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from epoch {epoch}, loss: {loss}")
        return model, optimizer, epoch, loss
    return model, optimizer, 0, float('inf')  

# Data Augmentation
class DualViewTransform:
    def __init__(self, augment_transform):
        self.augment_transform = augment_transform

    def __call__(self, x):
        view1 = self.augment_transform(x)
        view2 = self.augment_transform(x)
        return view1, view2

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

dual_view_transform = DualViewTransform(strong_augment_transform)

# Load OCTMNIST Dataset
def load_octmnist_data(transform, batch_size):
    dataset = OCTMNIST(split="train", transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    return dataloader

unlabeled_dataloader = load_octmnist_data(transform=dual_view_transform, batch_size=batch_size)

class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        config = ViTConfig(
            hidden_size=192,
            num_hidden_layers=12,
            num_attention_heads=3,
            intermediate_size=768,
            patch_size=16,
            image_size=224,
            output_hidden_states=True
        )
        self.vit = ViTModel(config)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        output = self.vit(x)
        x = output.hidden_states[-1]
        x = x.mean(dim=1)
        return x

class DualStreamNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.online_network_1 = ViTBackbone()
        self.online_network_2 = ViTBackbone()
        self.target_network_1 = ViTBackbone()
        self.target_network_2 = ViTBackbone()
        for param in self.target_network_1.parameters():
            param.requires_grad = False
        for param in self.target_network_2.parameters():
            param.requires_grad = False

        self.projection_head = nn.Sequential(
            nn.Linear(192 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 128)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x1, x2):
        feat1_online = self.online_network_1(x1)
        feat2_online = self.online_network_2(x2)

        with torch.no_grad():
            feat1_target = self.target_network_1(x1)
            feat2_target = self.target_network_2(x2)

        online_fused_feat = torch.cat([feat1_online, feat2_online], dim=1)
        online_proj_feat = self.projection_head(online_fused_feat)
        online_pred_feat = self.prediction_head(online_proj_feat)

        target_fused_feat = torch.cat([feat1_target, feat2_target], dim=1)
        target_proj_feat = self.projection_head(target_fused_feat).detach()

        return online_pred_feat, target_proj_feat

    def update_target_network(self):
        for param, target_param in zip(self.online_network_1.parameters(), self.target_network_1.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * param.data
        for param, target_param in zip(self.online_network_2.parameters(), self.target_network_2.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * param.data

# Initialize Model, Optimizer, and Criterion
model = DualStreamNetwork().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CosineSimilarity(dim=1)
scaler = GradScaler()

# Compute FLOPs (GFLOPs) 
dummy_input = torch.randn(1, 3, 224, 224).to(device)
flops = FlopCountAnalysis(model.online_network_1, dummy_input)
print(f"Model FLOPs: {flops.total() / 1e9:.4f} GFLOPs")

# Training Function with Loss Logging
def train_self_supervised(model, dataloader, epochs, optimizer, criterion, checkpoint_path, start_epoch=0):
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

            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_epoch_loss, checkpoint_path)

    # Plot loss curve and save model
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, linestyle='-', color='black')
    plt.title("Loss Function During Self-Supervised Pretraining")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.grid(True)
    plt.savefig('./ssp_retinaloct_tbme/vit2spn_tiny/scratch/result/ssp_loss_curv.png')
    plt.show()

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total parameters: {total_params}")

# Start Training
checkpoint_path = os.path.join(output_directory, "octmnist_vit2spn_tiny_scratch_checkpoint.pth")
model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
train_self_supervised(model, unlabeled_dataloader, epochs, optimizer, criterion, checkpoint_path, start_epoch=start_epoch)
torch.save(model.online_network_1.state_dict(), output_pretrained_model_path)  
print(f"Pretrained model saved at {output_pretrained_model_path}")