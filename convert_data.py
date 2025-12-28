import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
import os

# Parse arguments
img_file = sys.argv[1]
H = int(sys.argv[2])
W = int(sys.argv[3])
C = int(sys.argv[4])
output_file = sys.argv[5]
corrected = int(sys.argv[6])  # 1 or 0

# Load images
images = []
labels = []
files = sorted(os.listdir(img_file))

for file in files:
    path = os.path.join(img_file, file)
    img = Image.open(path).convert("L").resize((W, H))
    images.append(np.array(img))
    labels.append(int(file.split("-")[0]))

images = np.array(images)
labels = np.array(labels)

print(f"Loaded {len(images)} images")

# White pixel cleaning function
def clean_white_pixels(img, max_iterations=14):
    """Remove white pixels by averaging non-white neighbors"""
    img = img.astype(np.float32)
    for i in range(max_iterations):
        white_mask = (img == 255)
        if not white_mask.any():
            break
        
        # Get shifted versions for all 4 neighbors
        top = np.pad(img[:-1, :], ((1, 0), (0, 0)), mode='edge')
        bottom = np.pad(img[1:, :], ((0, 1), (0, 0)), mode='edge')
        left = np.pad(img[:, :-1], ((0, 0), (1, 0)), mode='edge')
        right = np.pad(img[:, 1:], ((0, 0), (0, 1)), mode='edge')
        print(top.shape)
        
        # Create masks for non-white neighbors
        top_valid = (top != 255)
        bottom_valid = (bottom != 255)
        left_valid = (left != 255)
        right_valid = (right != 255)
        
        # Sum valid neighbors and count them
        neighbor_sum = (top * top_valid + bottom * bottom_valid + 
                       left * left_valid + right * right_valid)
        neighbor_count = (top_valid + bottom_valid + left_valid + right_valid)
        
        # Replace white pixels where we have valid neighbors
        has_neighbors = (neighbor_count > 0) & white_mask
        img[has_neighbors] = neighbor_sum[has_neighbors] / neighbor_count[has_neighbors]
    
    return img.astype(np.uint8)

# Apply correction if requested
if corrected == 1:
    print("Applying white pixel correction to original images...")
    images_processed = np.array([clean_white_pixels(img) for img in images])
    
    # Display comparison BEFORE oversampling
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(images[0], cmap="gray")
    axes[0].set_title(f"Original - label {labels[0]}")
    axes[0].axis("off")
    
    axes[1].imshow(images_processed[0], cmap="gray")
    axes[1].set_title(f"Cleaned - label {labels[0]}")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # NOW oversample label 5 using the cleaned images
    print("\nOversampling label 5...")
    images_5 = images_processed[labels == 5]
    labels_5 = labels[labels == 5]
    
    factor = 14
    images_5_new = np.repeat(images_5, factor, axis=0)
    labels_5_new = np.repeat(labels_5, factor, axis=0)
    
    # Append to cleaned dataset
    images_final = np.concatenate([images_processed, images_5_new], axis=0)
    labels_final = np.concatenate([labels, labels_5_new], axis=0)
else:
    print("Skipping white pixel correction...")
    images_final = images
    labels_final = labels

# Count images per label
print("\nFinal dataset:")
unique_labels = np.unique(labels_final)
for lab in unique_labels:
    count = np.sum(labels_final == lab)
    print(f"Label {lab}: {count} images")

# Save to .npz file
print(f"\nSaving data to {output_file}...")
np.savez_compressed(output_file, images=images_final, labels=labels_final)
print(f"Data saved successfully! Shape: {images_final.shape}")