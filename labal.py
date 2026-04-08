import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image

data_dir = r"C:\Users\Edward\Desktop\VGGtrain\RGB_image\dataset"

print("Exists:", os.path.exists(data_dir))

classes = sorted(os.listdir(data_dir))
print("Classes:", classes)

images = []
labels = []

for label_idx, class_name in enumerate(classes):
    class_path = os.path.join(data_dir, class_name)
    
    if not os.path.isdir(class_path):
        print("Skip non-folder:", class_path)
        continue

    files = os.listdir(class_path)
    print(class_name, "->", len(files), "files")

    for file in files:
        img_path = os.path.join(class_path, file)
        
        try:
            img = Image.open(img_path).convert("RGB").resize((32, 32))
            img = np.array(img)

            images.append(img)
            labels.append(label_idx)

        except Exception as e:
            print("Skip:", img_path, "reason:", e)

X = np.array(images) / 255.0
y = to_categorical(labels, num_classes=len(classes))

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

save_dir = r"C:\Users\Edward\Desktop\VGGtrain\RGB_image\dataset\output"
os.makedirs(save_dir, exist_ok=True)

# ===== 1. 先 train / temp =====
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ===== 2. 再切 val / test =====
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# ===== 3. 存檔 =====
np.save(os.path.join(save_dir, "X_train.npy"), X_train)
np.save(os.path.join(save_dir, "X_val.npy"), X_val)
np.save(os.path.join(save_dir, "X_test.npy"), X_test)

np.save(os.path.join(save_dir, "y_train.npy"), y_train)
np.save(os.path.join(save_dir, "y_val.npy"), y_val)
np.save(os.path.join(save_dir, "y_test.npy"), y_test)

# optional
np.save(os.path.join(save_dir, "classes.npy"), np.array(classes))

print("Saved all datasets to:", save_dir)
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)