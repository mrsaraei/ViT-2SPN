import os
import random
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import warnings
from transformers import ViTModel
from collections import Counter

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
batch_size = 128
fine_tune_epochs = 50
k_folds = 10
subset_size = 2000
random_seed = 42

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
dataset_path = "./datasets/ucsdoct"
full_dataset = datasets.ImageFolder(root=dataset_path, transform=strong_augment_transform)

# Count the number of samples in each class
class_counts = Counter(full_dataset.targets)
print("Original Class sizes:", dict(class_counts))

# Define the number of classes
num_classes = len(class_counts)
print(f"Number of classes: {num_classes}")

# Ensure subset_size does not exceed the number of samples in full_dataset
total_samples = len(full_dataset)
subset_size = min(subset_size, total_samples)

# Randomly select indices for the subset
subset_indices = random.sample(range(total_samples), subset_size)

# Create a subset of the full dataset
subset_dataset = Subset(full_dataset, subset_indices)

# Extract labels for the subset
labels = [full_dataset.targets[i] for i in subset_indices]

# Split dataset: 70% train, 20% validation, 10% test
train_idx, temp_idx, train_labels, temp_labels = train_test_split(
    subset_indices, labels, test_size=0.3, stratify=labels, random_state=random_seed)
val_idx, test_idx, _, _ = train_test_split(temp_idx, temp_labels, test_size=0.33, stratify=temp_labels, random_state=random_seed)

# Create subsets for train, validation, and test
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Compute class weights
all_labels = np.array(full_dataset.targets)
class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=all_labels)
base_weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# Define Loss Function
criterion = nn.CrossEntropyLoss(weight=base_weight_tensor)

# ViT Backbone Model
class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("WinKawaks/vit-tiny-patch16-224", output_hidden_states=True)

    def forward(self, x):
        output = self.vit(x)
        x = output.hidden_states[-1]
        return x.mean(dim=1)

# Fine-Tuned Model
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
def fine_tune_model(model, train_loader, val_loader, optimizer, criterion, epochs=fine_tune_epochs):
    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)
                outputs = model(x)
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()

    model.load_state_dict(best_state_dict)

# AUC Computation and Curve Plotting
def compute_auc_and_plot_fold(model, val_loader, classes):
    val_labels, val_probs = [], []
    model.eval()
    with torch.no_grad():
        for x, labels in val_loader:
            x, labels = x.to(device), labels.to(device)
            outputs = model(x)
            val_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)
    one_hot_labels = np.eye(len(classes))[val_labels]

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(one_hot_labels[:, i], val_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    mean_auc = np.mean(list(roc_auc.values()))
    return fpr, tpr, roc_auc, mean_auc

# Evaluation on Test Data
def evaluate_test_data(model, test_loader, classes):
    test_labels, test_probs = [], []
    model.eval()
    with torch.no_grad():
        for x, labels in test_loader:
            x, labels = x.to(device), labels.to(device)
            outputs = model(x)
            test_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    predictions = np.argmax(test_probs, axis=1)

    # Classification Report
    report = classification_report(test_labels, predictions, target_names=[str(i) for i in range(num_classes)])
    print(f"\nClassification Report:\n{report}")

    # Confusion Matrix
    cm = confusion_matrix(test_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(num_classes)])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("./ssp_retinaloct_tbme/vit2spn_tiny/sl_ssp/result/ucsdoct_confusion_matrix.png")
    plt.show()

# Initialize dictionaries to store results for each fold
fpr_dict = {}
tpr_dict = {}
auc_dict = {}
all_auc = []

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
best_auc = 0.0
best_model = None

for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(train_dataset)), train_labels)):
    print(f"\nFold {fold + 1}/{k_folds}")
    
    # Reinitialize model for each fold
    model = FineTunedModel(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    fine_tune_model(model, train_loader, val_loader, optimizer, criterion)

    fpr, tpr, roc_auc, mean_auc = compute_auc_and_plot_fold(model, val_loader, [str(i) for i in range(num_classes)])

    # Store results
    fpr_dict[fold] = fpr
    tpr_dict[fold] = tpr
    auc_dict[fold] = roc_auc
    all_auc.append(mean_auc)

    if mean_auc > best_auc:
        best_auc = mean_auc
        best_model = model
        print(f"Best model found for fold {fold + 1} with AUC: {mean_auc:.4f}")

# Final evaluation on test data
if best_model is not None:
    print("\nEvaluating on Test Data using the Best Model:")
    evaluate_test_data(best_model, test_loader, [str(i) for i in range(num_classes)])
else:
    print("No best model found. Evaluation cannot proceed.")

# Print average AUC for all folds
print(f"\nAverage AUC across folds: {np.mean(all_auc):.4f}")

# Final AUC and ROC plots across folds
plt.figure(figsize=(10, 8))
for fold in range(k_folds):
    plt.plot(fpr_dict[fold][0], tpr_dict[fold][0], lw=2, label=f"Fold {fold + 1} (AUC = {auc_dict[fold][0]:.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - All Folds (Mean AUC = {np.mean(all_auc):.4f})")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("./ssp_retinaloct_tbme/vit2spn_tiny/sl_ssp/result/ucsdoct_roc_curve_all_folds.png")
plt.show()