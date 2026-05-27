import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE = 48
BATCH_SIZE = 128
EPOCHS = 50
NUM_CLASSES = 7

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# GPU INFO
# ─────────────────────────────────────────────
if torch.cuda.is_available():

    print(f"✅ GPU : {torch.cuda.get_device_name(0)}")

    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

else:
    print("⚠️ No CUDA GPU found — running on CPU")

print(f"Device: {DEVICE}\n")

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
train_transforms = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

test_transforms = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

# ─────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────
train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transforms
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transforms
)

# ─────────────────────────────────────────────
# DATALOADERS
# ─────────────────────────────────────────────
# num_workers=0 fixes Windows multiprocessing crash

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print(f"Classes   : {train_dataset.classes}")
print(f"Train imgs: {len(train_dataset)}")
print(f"Test imgs : {len(test_dataset)}\n")

# Save class mapping
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(train_dataset.class_to_idx, f, indent=2)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class EmotionCNN(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),

            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),

            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),

            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256 * 6 * 6, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(512, 256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

# ─────────────────────────────────────────────
# TRAIN / EVAL FUNCTION
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):

    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:

        for images, labels in tqdm(
            loader,
            desc="Train" if is_train else "Eval ",
            leave=False
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(1)

            correct += predictions.eq(labels).sum().item()

            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    model = EmotionCNN(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    print(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0

    patience_count = 0

    EARLY_STOP_PATIENCE = 10

    # ─────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = run_epoch(
            model,
            test_loader,
            criterion
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        vram_str = ""

        if torch.cuda.is_available():

            vram_used = torch.cuda.memory_reserved(0) / 1e9

            vram_str = f" | VRAM {vram_used:.1f} GB"

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train loss {train_loss:.4f} "
            f"acc {train_acc*100:.2f}% | "
            f"Val loss {val_loss:.4f} "
            f"acc {val_acc*100:.2f}%"
            + vram_str
        )

        # SAVE BEST MODEL
        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save({

                "epoch": epoch,

                "model_state": model.state_dict(),

                "optimizer_state": optimizer.state_dict(),

                "val_acc": val_acc,

                "classes": train_dataset.classes,

            }, MODEL_PATH)

            print(
                f"✅ Best model saved "
                f"(val_acc = {val_acc*100:.2f}%)"
            )

            patience_count = 0

        else:

            patience_count += 1

            if patience_count >= EARLY_STOP_PATIENCE:

                print(
                    f"\n⏹ Early stopping triggered "
                    f"at epoch {epoch}"
                )

                break

    # ─────────────────────────────────────────
    # TRAINING CURVES
    # ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_acc"], label="Train")
    axes[0].plot(history["val_acc"], label="Val")

    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")

    axes[0].legend()

    axes[1].plot(history["train_loss"], label="Train")
    axes[1].plot(history["val_loss"], label="Val")

    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")

    axes[1].legend()

    plt.tight_layout()

    curve_path = os.path.join(
        MODEL_DIR,
        "training_curves.png"
    )

    plt.savefig(curve_path)

    plt.show()

    print(f"\n🎉 Done!")

    print(
        f"Best val acc: "
        f"{best_val_acc*100:.2f}%"
    )

    print(f"Model → {MODEL_PATH}")

    print(f"Curves → {curve_path}")