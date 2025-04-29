import os
from collections import Counter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader, Dataset
from medmnist import INFO
from medmnist.dataset import OCTMNIST

# ========== Paths ==========
info = INFO["octmnist"]
num_classes = len(info["label"])  
labeled_dataset = OCTMNIST(split="train", download=True)
test_dataset = OCTMNIST(split="test", download=True)
output_dir = './ssp_retinaloct_tbme/vit2spn_tiny/preprocessing'
os.makedirs(output_dir, exist_ok=True)

# ========== Dataset Analysis ==========
classes = info["label"]
num_samples = len(labeled_dataset)

print(f"Number of classes: {num_classes}")
print(f"Number of images: {num_samples}")

# ========== Image Properties ==========
def get_image_properties(images):
    min_width, max_width = float('inf'), 0
    min_height, max_height = float('inf'), 0
    aspect_ratios = []
    widths = []
    heights = []
    file_sizes = []
    valid_images = 0

    for img in images:
        try:
            # img is already a PIL.Image
            width, height = img.size
            min_width = min(min_width, width)
            max_width = max(max_width, width)
            min_height = min(min_height, height)
            max_height = max(max_height, height)

            aspect_ratios.append(width / height)
            widths.append(width)
            heights.append(height)

            # No filename attribute, so removing file size calculation
            valid_images += 1

        except Exception as e:
            print(f"Error processing image: {e}")
    
    avg_width = np.mean(widths) if widths else 0
    avg_height = np.mean(heights) if heights else 0

    return {
        "min_width": min_width,
        "max_width": max_width,
        "avg_width": avg_width,
        "min_height": min_height,
        "max_height": max_height,
        "avg_height": avg_height,
        "min_aspect_ratio": np.min(aspect_ratios) if aspect_ratios else 0,
        "max_aspect_ratio": np.max(aspect_ratios) if aspect_ratios else 0,
        "avg_aspect_ratio": np.mean(aspect_ratios) if aspect_ratios else 0,
        "corrupted_files": valid_images != len(images),
        "widths": widths, 
        "heights": heights,  
        "aspect_ratios": aspect_ratios  
    }

# Collect images from the dataset
sample_images = [img for img, _ in labeled_dataset]

# Get properties of the images
image_properties = get_image_properties(sample_images)

# Extract the widths, heights, and aspect ratios for later visualization
widths = image_properties["widths"]
heights = image_properties["heights"]
aspect_ratios = image_properties["aspect_ratios"]

# ========== Class Distribution ==========
# Fixing the error with numpy.ndarray labels
class_counts = Counter()
for _, label in labeled_dataset:
    label = label.item() if isinstance(label, np.ndarray) else label  # Convert label to scalar if it's a numpy array
    class_counts[label] += 1  

print("\nClass distribution:")
for cls, count in class_counts.items():
    print(f"{cls}: {count} images")

# ========== Image Statistics (Mean, Std) ==========
def compute_mean_std(images):
    means = []
    stds = []
    for img in images:
        try:
            img = img.convert('RGB')
            img_array = np.array(img) / 255.0
            means.append(np.mean(img_array, axis=(0,1)))
            stds.append(np.std(img_array, axis=(0,1)))
        except Exception as e:
            print(f"Error processing image: {e}")
    mean = np.mean(means, axis=0) if means else np.array([0.0, 0.0, 0.0])
    std = np.mean(stds, axis=0) if stds else np.array([0.0, 0.0, 0.0])
    return mean, std

mean, std = compute_mean_std(sample_images)
print(f"\nDataset mean (RGB): {mean}")
print(f"Dataset std (RGB): {std}")

# ========== Visualize Some Samples ==========
def visualize_samples(dataset, num_samples=5):
    # Create the figure with a larger size for better visualization
    plt.figure(figsize=(15, 10))
    
    # Define class mapping based on desired output (AMD, CSR, NORMAL, DR, MH)
    class_mapping = {
        0: "CNV",
        1: "DME",
        2: "DRUSEN",
        3: "NORMAL"
    }
    classes = ["CNV", "DME", "DRUSEN", "NORMAL"]

    # Loop through each class in the dataset
    for i, cls in enumerate(classes):
        # Get images for the current class
        # Map class name back to index for filtering
        cls_idx = next(key for key, value in class_mapping.items() if value == cls)
        images = [img for img, label in dataset if (label.item() if isinstance(label, np.ndarray) else label) == cls_idx][:num_samples]
        
        # Loop through the images for the current class
        for j, img in enumerate(images):
            ax = plt.subplot(len(classes), num_samples, i * num_samples + j + 1)
            
            # Convert PIL Image to numpy array and display in grayscale
            img_array = np.array(img)
            if img_array.ndim == 3:  # If image is RGB, convert to grayscale
                img_array = img_array.mean(axis=2).astype(np.uint8)
            ax.imshow(img_array, cmap='gray')
            ax.axis('off')
            
            # Set title (class name) for the first image of the row
            if j == 0:
                # Display the class name to the left of the first image
                ax.text(-0.1, 0.5, cls.upper(), color='black', fontsize=18, ha='center', va='center', rotation=90, transform=ax.transAxes)
    
    # Adjust the spacing between subplots (reduce distance between columns)
    plt.subplots_adjust(wspace=0.05, hspace=0.1)  # Reduce wspace to minimize column spacing
    plt.savefig(os.path.join(output_dir, 'octmnist_sample_images.png'))

visualize_samples(labeled_dataset, num_samples=5)

# ========== Class Distribution Plot ==========
# Manually define class names
class_name_mapping = {
    0: "CNV",
    1: "DME",
    2: "DRUSEN",
    3: "NORMAL"
}

# ========== Class Distribution Plot ==========
plt.figure(figsize=(8,6))
sorted_classes = sorted(class_counts.items()) 

# Map class indices to the corresponding class names using the predefined mapping
class_names = [class_name_mapping.get(cls, f"Unknown-{cls}") for cls, _ in sorted_classes]

plt.bar(class_names, [count for _, count in sorted_classes], color='darkblue')
plt.xticks(rotation=45)
plt.title("Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.grid(axis='y')
plt.axhline(y=np.mean(list(class_counts.values())), color='r', linestyle='--', label='Mean')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'octmnist_class_distribution.png'))
# Removed plt.show() as per matplotlib guidelines

# ========== Save Dataset Summary to JSON ==========
dataset_summary = {
    "num_classes": num_classes,
    "num_images": num_samples,
    "classes": classes,
    "class_distribution": dict(class_counts),
    "dataset_mean_RGB": mean.tolist(),
    "dataset_std_RGB": std.tolist(),
    "image_properties": image_properties
}

summary_path = os.path.join(output_dir, 'octmnist_dataset_summary.json')
with open(summary_path, 'w') as f:
    json.dump(dataset_summary, f, indent=4)

print(f"\n✅ Dataset summary saved to {summary_path}")