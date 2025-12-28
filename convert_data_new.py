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


files = sorted(os.listdir(img_file))
print(len(files))
labels = []
images = []
#l = []
for f in files:
    #a = int(f.split("-")[0])
    #l.append(a)
    labels.append(int(f[0:1]))
    img = Image.open(os.path.join(img_file, f)).convert("L").resize((W, H))
    images.append(np.array(img))

images = np.array(images)
labels = np.array(labels)



'''
print(mask.shape)

labels_with_255 = labels[np.any(mask, axis=(1, 2))]
unique, counts = np.unique(labels_with_255, return_counts=True)

for u, c in zip(unique, counts):
    print(f"Label {u}: {c} images contain 255")


num_images_with_255 = np.sum(np.any(mask, axis=(1, 2)))
print(num_images_with_255)
'''



fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(images[0], cmap="gray")
axes[0].set_title(f"Original - label {labels[0]}")
axes[0].axis("off")



j = 0

for img in images:
    msk = (img == 255)
    coords = np.where(msk)
    if coords[0].size <10: 
        continue

    rows, cols = coords
    
    ur, rc = np.unique(rows, return_counts=True)
    uc, cc = np.unique(cols, return_counts=True)

    r = ur[np.argmax(rc)]
    c = uc[np.argmax(cc)]

    if rc.max() >= 10:
        j += 1
        print("horizontal white line")
        if 0 < r < img.shape[0] - 1:
            img[r, :] = (img[r-1, :] + img[r+1, :]) / 2

    elif cc.max() >= 10:
        j += 1
        print("vertical white line")
        if 0 < c < img.shape[1] - 1:
            img[:, c] = (img[:, c-1] + img[:, c+1]) / 2

print(j)
images_5 = images[labels == 5]
labels_5 = labels[labels == 5]

factor = 14
images_5_new = np.repeat(images_5, factor, axis=0)
labels_5_new = np.repeat(labels_5, factor, axis=0)

images_final = np.concatenate([images, images_5_new], axis=0)
labels_final = np.concatenate([labels, labels_5_new], axis=0)

axes[1].imshow(images[0], cmap="gray")
axes[1].set_title(f"Cleaned - label {labels[0]}")
axes[1].axis("off")
plt.tight_layout()
plt.show()
    

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