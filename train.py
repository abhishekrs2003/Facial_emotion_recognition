import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
IMAGES_DIR = "dataset/images"   # ← your dataset path
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.pth")

IMG_SIZE    = 128
BATCH_SIZE  = 32
EPOCHS      = 50
VAL_SPLIT   = 0.2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  GPU INFO
# ─────────────────────────────────────────────
if torch.cuda.is_available():
    print(f"✅ GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
else:
    print("⚠️  No GPU found — running on CPU")
print(f"Device: {DEVICE}\n")

# ─────────────────────────────────────────────
#  EMOTION MAPPING
#  filename stem (lowercase) → class name
# ─────────────────────────────────────────────
FILENAME_TO_EMOTION = {
    "anger":     "angry",
    "contempt":  "contempt",
    "disgust":   "disgust",
    "fear":      "fear",
    "happy":     "happy",
    "neutral":   "neutral",
    "sad":       "sad",
    "surprised": "surprise",
}

EMOTIONS    = sorted(set(FILENAME_TO_EMOTION.values()))
NUM_CLASSES = len(EMOTIONS)
print(f"Classes ({NUM_CLASSES}): {EMOTIONS}\n")

# ─────────────────────────────────────────────
#  SCAN DATASET
#  Structure: images/<number>/<Emotion>.jpg
# ─────────────────────────────────────────────
input_paths = []
labels      = []
skipped     = 0

for person_folder in sorted(os.listdir(IMAGES_DIR)):
    person_path = os.path.join(IMAGES_DIR, person_folder)
    if not os.path.isdir(person_path):
        continue
    for fname in os.listdir(person_path):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem    = os.path.splitext(fname)[0].lower()
        emotion = FILENAME_TO_EMOTION.get(stem)
        if emotion is None:
            skipped += 1
            continue
        input_paths.append(os.path.join(person_path, fname))
        labels.append(EMOTIONS.index(emotion))

print(f"Total images : {len(input_paths)}")
print(f"Skipped      : {skipped}")
for i, emo in enumerate(EMOTIONS):
    print(f"  {emo:12s}: {labels.count(i)}")
print()

if len(input_paths) == 0:
    raise RuntimeError("No images found! Check your IMAGES_DIR path.")

# Save class mapping for app.py
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump({e: i for i, e in enumerate(EMOTIONS)}, f, indent=2)

# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────
class FacialEmotionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# ─────────────────────────────────────────────
#  TRANSFORMS
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
#  TRAIN / VAL SPLIT  80 / 20
# ─────────────────────────────────────────────
full_train_dataset = FacialEmotionDataset(input_paths, labels, transform=train_transform)
full_val_dataset   = FacialEmotionDataset(input_paths, labels, transform=val_transform)

train_size = int((1 - VAL_SPLIT) * len(input_paths))
val_size   = len(input_paths) - train_size

indices        = torch.randperm(len(input_paths), generator=torch.Generator().manual_seed(42)).tolist()
train_indices  = indices[:train_size]
val_indices    = indices[train_size:]

train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
val_dataset   = torch.utils.data.Subset(full_val_dataset,   val_indices)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True
)

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — 128 → 64
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2 — 64 → 32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3 — 32 → 16
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 4 — 16 → 8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        # 128px → 4× MaxPool → 8×8 feature maps
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = EmotionCNN(num_classes=NUM_CLASSES).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}\n")

# ─────────────────────────────────────────────
#  LOSS / OPTIMIZER / SCHEDULER
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, verbose=True
)

# ─────────────────────────────────────────────
#  EPOCH HELPER
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, lbls in tqdm(loader, desc="Train" if is_train else "Val  ", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            lbls   = lbls.to(DEVICE,   non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            out  = model(images)
            loss = criterion(out, lbls)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += out.argmax(1).eq(lbls).sum().item()
            total      += lbls.size(0)

    return total_loss / total, correct / total

# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────
history             = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc        = 0.0
patience_count      = 0
EARLY_STOP_PATIENCE = 10

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
    val_loss,   val_acc   = run_epoch(model, val_loader,   criterion)
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    vram = f" | VRAM {torch.cuda.memory_reserved(0)/1e9:.1f} GB" if torch.cuda.is_available() else ""
    print(
        f"Epoch [{epoch:02d}/{EPOCHS}]  "
        f"Train loss {train_loss:.4f}  acc {train_acc*100:.2f}%  |  "
        f"Val loss {val_loss:.4f}  acc {val_acc*100:.2f}%"
        + vram
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc":         val_acc,
            "classes":         EMOTIONS,
            "num_classes":     NUM_CLASSES,
            "img_size":        IMG_SIZE,
        }, MODEL_PATH)
        print(f"  ✅ Best model saved (val_acc = {val_acc*100:.2f}%)")
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= EARLY_STOP_PATIENCE:
            print(f"\n⏹  Early stopping at epoch {epoch}")
            break

# ─────────────────────────────────────────────
#  PLOT TRAINING CURVES
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history["train_acc"],  label="Train")
axes[0].plot(history["val_acc"],    label="Val")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history["train_loss"], label="Train")
axes[1].plot(history["val_loss"],   label="Val")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"))
plt.show()

print(f"\n🎉 Training complete!")
print(f"   Best val acc : {best_val_acc*100:.2f}%")
print(f"   Model saved  : {MODEL_PATH}")