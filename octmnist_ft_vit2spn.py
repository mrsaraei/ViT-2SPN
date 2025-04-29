import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, roc_curve, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import INFO
from medmnist.dataset import OCTMNIST
from transformers import ViTModel
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
batch_size = 128
fine_tune_epochs = 50
k_folds = 10
subset_fraction = 0.05129415
random_seed = 42
test_subset_size = 500

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

# Load Dataset
info = INFO["octmnist"]
num_classes = len(info["label"])
train_dataset = OCTMNIST(split="train", transform=strong_augment_transform, download=True)
test_dataset = OCTMNIST(split="test", transform=strong_augment_transform, download=True)

def get_subset(dataset, fraction, seed):
    random.seed(seed)
    total_samples = len(dataset)
    indices = random.sample(range(total_samples), int(total_samples * fraction))
    return Subset(dataset, indices)

small_train_dataset = get_subset(train_dataset, subset_fraction, random_seed)
test_subset = Subset(test_dataset, random.sample(range(len(test_dataset)), test_subset_size))
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# ViT Backbone
class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("WinKawaks/vit-tiny-patch16-224", output_hidden_states=True)

    def forward(self, x):
        output = self.vit(x)
        return output.hidden_states[-1].mean(dim=1)

# Fine-tuned Model
class FineTunedModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = ViTBackbone()
        self.fc = nn.Sequential(
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

# Training Function
def fine_tune_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=50, patience=3):
    best_loss = float("inf")
    best_weights = None
    counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.squeeze().long().to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.squeeze().long().to(device)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_weights)

# AUC Computation and Curve Plotting
def compute_auc_and_plot_fold(model, loader, classes, fold):
    labels, probs = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.squeeze().long().to(device)
            outputs = model(x)
            probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    labels = np.array(labels)
    probs = np.array(probs)
    one_hot = np.eye(len(classes))[labels]

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(one_hot[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc, np.mean(list(roc_auc.values())), labels, probs

# Evaluation on Test Data (with Confusion Matrix Visualization)
def evaluate_test_data(model, loader, classes):
    labels, probs = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.squeeze().long().to(device)
            outputs = model(x)
            probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    # Compute confusion matrix
    preds = np.argmax(probs, axis=1)
    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("./ssp_retinaloct_tbme/vit2spn_tiny/result/octmnist_confusion_matrix.png")
    plt.show()
    print(classification_report(labels, preds, target_names=classes))

# Stratified K-Fold Cross-Validation with Best Model Selection
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
best_model, best_auc = None, 0
fpr_dict, tpr_dict, auc_dict = {}, {}, {}
fold_aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(small_train_dataset)), 
                                                       [small_train_dataset.dataset.labels[i] for i in small_train_dataset.indices])):
    print(f"\nFold {fold+1}/{k_folds}")
    train_ds = Subset(small_train_dataset, train_idx)
    val_ds = Subset(small_train_dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_labels = np.array([small_train_dataset.dataset.labels[i] for i in train_idx]).squeeze()
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))

    model = FineTunedModel(num_classes).to(device)
    model.backbone.load_state_dict(torch.load("./ssp_retinaloct_tbme/vit2spn_tiny/octmnist_vit2spn_tiny_model.pth"))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=3)

    fine_tune_model(model, train_loader, val_loader, optimizer, criterion, scheduler)

    fpr_dict[fold], tpr_dict[fold], auc_dict[fold], mean_auc, _, _ = compute_auc_and_plot_fold(model, val_loader, [str(i) for i in range(num_classes)], fold)
    fold_aucs.append(mean_auc)

    if mean_auc > best_auc:
        best_auc = mean_auc
        best_model = model

# Evaluate on Test Data using the Best Model
print("\nEvaluating on test data with the best model...")
evaluate_test_data(best_model, test_loader, [str(i) for i in range(num_classes)])

# === FINAL PLOT ===
mean_auc_all_folds = np.mean(fold_aucs)
std_auc_all_folds = np.std(fold_aucs)

print(f"\nMean AUC across folds: {mean_auc_all_folds:.4f}")
print(f"Standard Deviation of AUC across folds: {std_auc_all_folds:.4f}")
print(f"Best AUC across folds: {best_auc:.4f}")

# Plotting ROC Curves for All Folds
plt.figure(figsize=(10, 8))
for fold in range(k_folds):
    plt.plot(fpr_dict[fold][0], tpr_dict[fold][0], label=f"Fold {fold+1} (AUC={auc_dict[fold][0]:.4f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - All Folds (Mean AUC = {mean_auc_all_folds:.3f} ± {std_auc_all_folds:.3f})")
plt.legend()
plt.grid()
plt.savefig("./ssp_retinaloct_tbme/vit2spn_tiny/result/octmnist_roc_curve_all_folds.png")
plt.show()