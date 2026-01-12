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
correct_data = int(sys.argv[6])  # 1 = correct problems, 0 = don't correct

files = sorted(os.listdir(img_file))
print(f"Found {len(files)} images")

labels = []
images = []

# Read all images
for f in files:
    labels.append(int(f[0:1]))
    img = Image.open(os.path.join(img_file, f)).convert("L").resize((W, H))
    images.append(np.array(img))

images = np.array(images)
labels = np.array(labels)

print(f"Original dataset shape: {images.shape}")

# Only correct data if correct_data == 1
if correct_data == 1:
    print("\nCorrecting data problems...")
    
    # Fix white lines (255 pixel values)
    fixes_applied = 0
    for idx, img in enumerate(images):
        msk = (img == 255)
        coords = np.where(msk)
        
        if coords[0].size < 10: 
            continue
        
        rows, cols = coords
        ur, rc = np.unique(rows, return_counts=True)
        uc, cc = np.unique(cols, return_counts=True)
        
        r = ur[np.argmax(rc)]
        c = uc[np.argmax(cc)]
        
        # Fix horizontal white line
        if rc.max() >= 10:
            fixes_applied += 1
            if 0 < r < img.shape[0] - 1:
                images[idx][r, :] = (img[r-1, :] + img[r+1, :]) / 2
        
        # Fix vertical white line
        elif cc.max() >= 10:
            fixes_applied += 1
            if 0 < c < img.shape[1] - 1:
                images[idx][:, c] = (img[:, c-1] + img[:, c+1]) / 2
    
    print(f"Fixed {fixes_applied} images with white lines")
    
    # Handle class imbalance - augment class 5
    images_5 = images[labels == 5]
    labels_5 = labels[labels == 5]
    
    factor = 14
    images_5_new = np.repeat(images_5, factor, axis=0)
    labels_5_new = np.repeat(labels_5, factor, axis=0)
    
    images_final = np.concatenate([images, images_5_new], axis=0)
    labels_final = np.concatenate([labels, labels_5_new], axis=0)
    
    print(f"Augmented class 5 by factor {factor}")
else:
    print("\nSkipping data correction (test mode)")
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