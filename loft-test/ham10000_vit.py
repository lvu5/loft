import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import timm
from peft import LoraConfig, get_peft_model
from torch.cuda.amp import autocast, GradScaler

# ----------------------
# Config
# ----------------------
DATA_ROOT = "HAM10000"
IMG_DIR = os.path.join(DATA_ROOT, "images")
CSV_PATH = os.path.join(DATA_ROOT, "HAM10000_metadata.csv")

NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 3
LR = 5e-4
DEVICE = "cuda"
RANK = 8

# ----------------------
# Dataset
# ----------------------
class HAMDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {l: i for i, l in enumerate(sorted(df.dx.unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(IMG_DIR, row.image_id + ".jpg")
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row.dx]

        if self.transform:
            image = self.transform(image)

        return image, label


train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5)),
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5)),
])

df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df.dx, random_state=42
)

train_ds = HAMDataset(train_df, train_tfms)
val_ds = HAMDataset(val_df, val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=8, pin_memory=True)

# ----------------------
# Model
# ----------------------
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=NUM_CLASSES
)

# LoRA config (r = 8)
lora_cfg = LoraConfig(
    r=RANK,
    lora_alpha=RANK*2,
    target_modules=["qkv", "proj"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_cfg)
model.to(DEVICE)

print(model.print_trainable_parameters())

# ----------------------
# Optimizer / Loss
# ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=1e-4
)

scaler = GradScaler()

# ----------------------
# Training loop
# ----------------------
loss_per_iter = []
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loss_per_iter.append(loss.item())

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {total_loss/len(train_loader):.4f} "
          f"Val Acc: {acc:.2f}%")

# write loss_per_iter to a file
with open("loss_per_iter_lora_vit_r16.txt", "w") as f:
    for loss in loss_per_iter:
        f.write(f"{loss}\n")