import os
import random
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import warnings
from medmnist import INFO
from medmnist.dataset import OCTMNIST

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
batch_size = 16
fine_tune_epochs = 50
k_folds = 10
subset_fraction = 0.05129415
random_seed = 42
test_subset_size = 500  

# Load Pretrained Backbone
from ssp_moco import MoCoBackbone

# Data Augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load OCTMNIST Dataset
info = INFO["octmnist"]
num_classes = len(info["label"])
labeled_dataset = OCTMNIST(split="train", transform=transform, download=True)
test_dataset = OCTMNIST(split="test", transform=transform, download=True)

# Select a Subset of the Dataset
def get_subset(dataset, fraction, seed):
    random.seed(seed)
    total_samples = len(dataset)
    subset_size = int(total_samples * fraction)
    indices = random.sample(range(total_samples), subset_size)
    return Subset(dataset, indices)

# Apply Subsetting to Training Data
small_labeled_dataset = get_subset(labeled_dataset, subset_fraction, random_seed)

# Apply Subsetting to Test Data
total_test_samples = len(test_dataset)
test_subset_indices = random.sample(range(total_test_samples), test_subset_size)
test_subset = Subset(test_dataset, test_subset_indices)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# Define FineTunedModel Class
class FineTunedModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = MoCoBackbone()
        self.fc = nn.Sequential(
            nn.Linear(2048, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

# Training Function
def fine_tune_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=fine_tune_epochs, patience=3):
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x, labels = x.to(device), labels.to(device)
            labels = labels.squeeze().long()

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
                labels = labels.squeeze().long()

                outputs = model(x)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state_dict)

# AUC Computation and Curve Plotting
def compute_auc_and_plot_fold(model, val_loader, classes, fold):
    val_labels, val_probs = [], []
    model.eval()
    with torch.no_grad():
        for x, labels in val_loader:
            x, labels = x.to(device), labels.to(device)
            labels = labels.squeeze().long()

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

# Test Evaluation
def evaluate_test_data(model, test_loader, classes):
    test_labels, test_probs = [], []
    model.eval()
    with torch.no_grad():
        for x, labels in test_loader:
            x, labels = x.to(device), labels.to(device)
            labels = labels.squeeze().long()

            outputs = model(x)
            test_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)

    predictions = np.argmax(test_probs, axis=1)
    cm = confusion_matrix(test_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(len(classes))])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("./ssp_retinaloct_midl2025/moco/result/confusion_matrix.png")
    plt.show()

    report = classification_report(test_labels, predictions, target_names=[str(i) for i in range(len(classes))])
    print(f"\nClassification Report:\n{report}")

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
best_auc = 0.0
best_model = None
fpr_dict, tpr_dict, auc_dict, all_auc = {}, {}, {}, []

for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(small_labeled_dataset)), 
                                                      [small_labeled_dataset.dataset.labels[i] for i in small_labeled_dataset.indices])):
    print(f"\nFold {fold + 1}/{k_folds}")
    train_subset = Subset(small_labeled_dataset, train_idx)
    val_subset = Subset(small_labeled_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    train_targets = np.array([small_labeled_dataset.dataset.labels[i] for i in train_idx]).squeeze()
    class_weights = compute_class_weight("balanced", classes=np.unique(train_targets), y=train_targets)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))

    model = FineTunedModel(num_classes=num_classes).to(device)
    model.backbone.load_state_dict(torch.load("./ssp_retinaloct_midl2025/moco/octmnist_moco_model.pth"))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    fine_tune_model(model, train_loader, val_loader, optimizer, criterion, scheduler)

    fpr, tpr, roc_auc, mean_auc = compute_auc_and_plot_fold(model, val_loader, [str(i) for i in range(num_classes)], fold + 1)

    fpr_dict[fold], tpr_dict[fold], auc_dict[fold] = fpr, tpr, roc_auc
    all_auc.append(mean_auc)

    if mean_auc > best_auc:
        best_auc = mean_auc
        best_model = model

# Evaluate on Test Data using the Best Model
print("\nEvaluating on Test Data using the Best Model:")
evaluate_test_data(best_model, test_loader, [str(i) for i in range(num_classes)])

# Print average AUC for all folds
print(f"\nAverage AUC across folds: {np.mean(all_auc):.4f}")

# Final AUC and ROC plots across folds
plt.figure(figsize=(10, 8))
for fold in range(k_folds):
    plt.plot(fpr_dict[fold][0], tpr_dict[fold][0], lw=2, label=f"Fold {fold + 1} (AUC = {auc_dict[fold][0]:.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - All Folds (Mean AUC = {np.mean(all_auc):.3f})")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("./ssp_retinaloct_midl2025/moco/result/roc_curve_all_folds.png")
plt.show()