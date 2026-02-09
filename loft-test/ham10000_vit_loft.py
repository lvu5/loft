import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import timm
from torch.cuda.amp import autocast, GradScaler
from loft import LoFTLinear, LoFTAdamW

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
RANK = 8
DEVICE = "cuda"

# ----------------------
# Dataset
# ----------------------
class HAMDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {l: i for i, l in enumerate(sorted(df.dx.unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(IMG_DIR, row.image_id + ".jpg")).convert("RGB")
        label = self.label_map[row.dx]
        return self.transform(img), label

def apply_loft(model, target_modules, rank):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_modules:
            setattr(model, name, LoFTLinear(module, rank))
        else:
            apply_loft(module, target_modules, rank)


train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df.dx, random_state=42
)

train_loader = DataLoader(
    HAMDataset(train_df, train_tfms),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

val_loader = DataLoader(
    HAMDataset(val_df, val_tfms),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

# ----------------------
# Model
# ----------------------
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=NUM_CLASSES
)

apply_loft(model, target_modules=["qkv", "proj"], rank=RANK)
model.to(DEVICE)

# Freeze everything except LoFT + head
for name, p in model.named_parameters():
    if ("U" in name) or ("V" in name) or ("head" in name):
        p.requires_grad = True
    else:
        p.requires_grad = False

# Sanity check
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable params:", trainable)

# ----------------------
# Optimizer / Loss
# ----------------------
for name, param in model.named_parameters():
    # print(name, param.requires_grad)
    if ".U" in name or ".V" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

loft_groups = []
for module in model.modules():
    if isinstance(module, LoFTLinear):
        loft_groups.append({"params": [module.U, module.V]})
# print paremeters values
# for name, param in model_loft.named_parameters():
#     # print values of parameters
#     print(f"{name}: {param.requires_grad} {param.data}")
optimizer = LoFTAdamW(loft_groups, lr=LR)
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=LR,
#     weight_decay=1e-4
# )

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
        loss_per_iter.append(loss.item())
        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total
    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss {total_loss/len(train_loader):.4f} | "
        f"Val Acc {acc:.2f}%"
    )

# write loss_per_iter to a file
with open("loss_per_iter_loft_vit_r8_loft.txt", "w") as f:
    for loss in loss_per_iter:
        f.write(f"{loss}\n")