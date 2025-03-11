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
from transformers import ViTForImageClassification, ViTFeatureExtractor
from fvcore.nn import FlopCountAnalysis
from transformers import ViTModel, ViTConfig

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
accumulation_steps = 8  
output_directory = "./ssp_retinaloct_midl2025/vit2spn_tiny/"
output_pretrained_model_path = os.path.join(output_directory, "octmnist_vit2spn_tiny_model.pth")

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
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dual_view_transform = DualViewTransform(strong_augment_transform)

# Load OCTMNIST Dataset
def load_octmnist_data(transform, batch_size):
    dataset = OCTMNIST(split="train", transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

unlabeled_dataloader = load_octmnist_data(transform=dual_view_transform, batch_size=batch_size)

class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the vit-tiny model directly using ViTModel
        self.vit = ViTModel.from_pretrained("WinKawaks/vit-tiny-patch16-224", output_hidden_states=True)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Forward pass through the Vision Transformer model
        output = self.vit(x)
        x = output.hidden_states[-1] 
        x = x.mean(dim=1)  
        return x
    
# Dual Stream Network with Flattened ViT Output
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
            nn.Linear(192 * 2, 1024),  # Concatenated feature size (768 * 2)
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
model = nn.DataParallel(model)  
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
flops = FlopCountAnalysis(model.module.online_network_1, dummy_input)
print(f"Model FLOPs: {flops.total() / 1e9:.4f} GFLOPs")

# Training Function with Loss Logging
def train_self_supervised(model, dataloader, epochs, optimizer, criterion):
    model.train()
    loss_history = []  
    for epoch in range(epochs):
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
                model.module.update_target_network()  
            epoch_loss += loss.item() * accumulation_steps

            # Clear CUDA memory every 10 iterations
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}")


    # Plot loss curve and save model
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, linestyle='-', color='black')
    plt.title("Loss Function During Self-Supervised Pretraining")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.grid(True)
    plt.savefig('./ssp_retinaloct_midl2025/vit2spn_tiny/result/ssp_loss_curv.png')
    plt.show()

# The number of parameters 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total parameters: {total_params}")

# Create an instance of CustomTransformerBackbone
custom_transformer_backbone = ViTBackbone()

# Now call count_parameters
total_params = count_parameters(ViTBackbone()) 
print(f"Backbone parameters: {total_params}")

# Start Training
train_self_supervised(model, unlabeled_dataloader, epochs, optimizer, criterion)
torch.save(model.module.online_network_1.state_dict(), output_pretrained_model_path)  
print(f"Pretrained model saved at {output_pretrained_model_path}")