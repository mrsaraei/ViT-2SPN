import os
from collections import Counter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json

# ========== Helper functions ==========
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))

# ========== Paths ==========
data_dir = './datasets/octird'
output_dir = './ssp_retinaloct_tbme/vit2spn_tiny/preprocessing'
os.makedirs(output_dir, exist_ok=True)

# ========== Dataset Analysis ==========

# Only include directories (class folders)
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
num_classes = len(classes)

# Count images only
num_samples = sum(
    len([img for img in os.listdir(os.path.join(data_dir, cls)) if is_image_file(img)])
    for cls in classes
)

print(f"Number of classes: {num_classes}")
print(f"Number of images: {num_samples}")

# ========== Image Properties ==========
def get_image_properties(image_paths):
    min_width, max_width = float('inf'), 0
    min_height, max_height = float('inf'), 0
    aspect_ratios = []
    file_sizes = []
    valid_images = 0
    widths = []
    heights = []

    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                min_width = min(min_width, width)
                max_width = max(max_width, width)
                min_height = min(min_height, height)
                max_height = max(max_height, height)

                aspect_ratios.append(width / height)
                file_sizes.append(os.path.getsize(img_path) / 1024) 
                widths.append(width)
                heights.append(height)

                valid_images += 1

        except Exception as e:
            print(f"Error opening {img_path}: {e}")
    
    avg_width = np.mean(widths)
    avg_height = np.mean(heights)

    return {
        "min_width": min_width,
        "max_width": max_width,
        "avg_width": avg_width,
        "min_height": min_height,
        "max_height": max_height,
        "avg_height": avg_height,
        "min_aspect_ratio": np.min(aspect_ratios),
        "max_aspect_ratio": np.max(aspect_ratios),
        "avg_aspect_ratio": np.mean(aspect_ratios),
        "avg_file_size_kb": np.mean(file_sizes),
        "corrupted_files": valid_images != len(image_paths),
        "widths": widths, 
        "heights": heights,  
        "aspect_ratios": aspect_ratios  
    }

# Collect sample image paths
sample_images = []
for cls in classes:
    cls_folder = os.path.join(data_dir, cls)
    sample_images.extend(
        os.path.join(cls_folder, img)
        for img in os.listdir(cls_folder)
        if is_image_file(img)
    )

# Get properties of the images
image_properties = get_image_properties(sample_images)

# Extract the widths, heights, and aspect ratios for later visualization
widths = image_properties["widths"]
heights = image_properties["heights"]
aspect_ratios = image_properties["aspect_ratios"]

# ========== Class Distribution ==========
class_counts = Counter()
for cls in classes:
    cls_folder = os.path.join(data_dir, cls)
    num_in_class = len([
        img for img in os.listdir(cls_folder) if is_image_file(img)
    ])
    class_counts[cls] = num_in_class

print("\nClass distribution:")
for cls, count in class_counts.items():
    print(f"{cls}: {count} images")

# ========== Image Statistics (Mean, Std) ==========

def compute_mean_std(image_paths):
    means = []
    stds = []
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img_array = np.array(img) / 255.0  
                means.append(np.mean(img_array, axis=(0,1)))
                stds.append(np.std(img_array, axis=(0,1)))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    return mean, std

mean, std = compute_mean_std(sample_images)
print(f"\nDataset mean (RGB): {mean}")
print(f"Dataset std (RGB): {std}")

# ========== Visualize Some Samples ==========

def visualize_samples(data_dir, classes, num_samples=5):
    plt.figure(figsize=(15, 10))
    
    # Loop through classes and images
    for i, cls in enumerate(classes):
        cls_folder = os.path.join(data_dir, cls)
        images = [img for img in os.listdir(cls_folder) if is_image_file(img)][:num_samples]
        
        # Display the class name to the left of the row
        for j, img_name in enumerate(images):
            img_path = os.path.join(cls_folder, img_name)
            try:
                img = Image.open(img_path)
                ax = plt.subplot(len(classes), num_samples, i * num_samples + j + 1)
                
                # Display the image
                ax.imshow(img)
                ax.axis('off')
                
                # Set title (class name) for the first image of the row
                if j == 0:
                    # Place the class name in uppercase on the left of the image
                    ax.text(-0.1, 0.5, cls.upper(), color='black', fontsize=18, ha='center', va='center', rotation=90, transform=ax.transAxes)
                
            except Exception as e:
                print(f"Error displaying {img_path}: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'octid_sample_images.png'))
    plt.show()

visualize_samples(data_dir, classes, num_samples=5)

# ========== Class Distribution Plot ==========
plt.figure(figsize=(8,6))
plt.bar(class_counts.keys(), class_counts.values(), color='darkblue')
plt.xticks(rotation=45)
plt.title("Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.grid(axis='y')
plt.axhline(y=np.mean(list(class_counts.values())), color='r', linestyle='--', label='Mean')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'octid_class_distribution.png'))
plt.show()

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

summary_path = os.path.join(output_dir, 'octid_dataset_summary.json')
with open(summary_path, 'w') as f:
    json.dump(dataset_summary, f, indent=4)

print(f"\n✅ Dataset summary saved to {summary_path}")