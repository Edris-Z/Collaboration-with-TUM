# Train + Val + Test + Plots
# Datasets: Split into train / validation / test. 
# CIFAR-10 train set (50k images) is split into 45k train + 5k validation. 
# Validation set (5k images) is evaluated every epoch.
# CIFAR-10 test set (10k images) is evaluated every epoch.
# Validation set: Yes (5k images).

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba
import matplotlib.pyplot as plt
import os 

# ---------------------------
# Model
# ---------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, d_model: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        return x


class PreNorm(nn.Module):
    def __init__(self, d_model, fn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, use_fast_path=True):
        super().__init__()
        self.mamba = PreNorm(
            d_model,
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=use_fast_path),
        )
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return x + self.drop(self.mamba(x))


class VisionMamba(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_chans=3, num_classes=10,
                 d_model=256, depth=6, d_state=16, d_conv=4, expand=2, use_fast_path=True):
        super().__init__()
        assert image_size % patch_size == 0
        self.embed = PatchEmbed(in_chans, d_model, patch_size)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=use_fast_path)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks: x = blk(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


# ---------------------------
# Config & loaders
# ---------------------------
@dataclass
class Config:
    data_dir: str = "./data"
    batch_size: int = 128
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 0.05
    image_size: int = 32
    patch_size: int = 4
    d_model: int = 256
    depth: int = 6
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    num_workers: int = 2
    amp: bool = True
    val_size: int = 5000  # 5k val out of 50k train


def get_loaders(cfg: Config):
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    train_tfms = transforms.Compose([
        transforms.RandomCrop(cfg.image_size, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.15),
    ])
    eval_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=train_tfms)
    test_ds    = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=eval_tfms)

    # 45k train / 5k val
    train_len = len(full_train) - cfg.val_size
    val_len   = cfg.val_size
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_train, [train_len, val_len], generator=g)


    val_ds.dataset = datasets.CIFAR10(cfg.data_dir, train=True, download=False, transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ---------------------------
# Train / Evaluation
# ---------------------------
def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train(); loss_fn = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images); loss = loss_fn(logits, targets)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(images); loss = loss_fn(logits, targets)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1); correct += (preds == targets).sum().item(); total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); loss_fn = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images); loss = loss_fn(logits, targets)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1); correct += (preds == targets).sum().item(); total += images.size(0)
    return total_loss / total, correct / total


# ---------------------------
# Main
# ---------------------------
def main():
       
    cfg = Config()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fast_path = (device.type == "cuda")
   
    train_loader, val_loader, test_loader = get_loaders(cfg)

    model = VisionMamba(image_size=cfg.image_size, patch_size=cfg.patch_size, in_chans=3, num_classes=10,
                        d_model=cfg.d_model, depth=cfg.depth, d_state=cfg.d_state, d_conv=cfg.d_conv,
                        expand=cfg.expand, use_fast_path=use_fast_path).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print(f"Device: {device} | Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []  
    test_accs = []


    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, val_acc     = evaluate(model, val_loader, device)
        test_loss, test_acc   = evaluate(model, test_loader, device) 

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
              f"val loss {val_loss:.4f} acc {val_acc*100:.2f}% | "
              f"test loss {test_loss:.4f} acc {test_acc*100:.2f}%")


        scheduler.step()

   
    
    max_train_acc = max(train_accs) if train_accs else 0.0
    avg_train_acc = sum(train_accs) / len(train_accs) if train_accs else 0.0
    last_train_acc = train_accs[-1] if train_accs else 0.0

    max_val_acc   = max(val_accs) if val_accs else 0.0
    avg_val_acc   = sum(val_accs) / len(val_accs) if val_accs else 0.0
    last_val_acc  = val_accs[-1] if val_accs else 0.0

    max_test_acc  = max(test_accs) if test_accs else 0.0
    avg_test_acc  = sum(test_accs) / len(test_accs) if test_accs else 0.0
    last_test_acc = test_accs[-1] if test_accs else 0.0

    print(f"\n[Acc] Maximum Train Accuracy: {max_train_acc*100:.2f}%")
    print(f"[Acc] Average Train Accuracy (all epochs): {avg_train_acc*100:.2f}%")
    print(f"[Acc] Train Accuracy (last epoch): {last_train_acc*100:.2f}%")

    print(f"[Acc] Maximum Validation Accuracy: {max_val_acc*100:.2f}%")
    print(f"[Acc] Average Validation Accuracy (all epochs): {avg_val_acc*100:.2f}%")
    print(f"[Acc] Validation Accuracy (last epoch): {last_val_acc*100:.2f}%")

   
    print(f"[Acc] Maximum Test Accuracy: {max_test_acc*100:.2f}%")
    print(f"[Acc] Average Test Accuracy: {avg_test_acc*100:.2f}%")  # equals final here
    print(f"[Acc] Test Accuracy (last epoch): {last_test_acc*100:.2f}%")


 # ---------- Plots----------
    os.makedirs("plots", exist_ok=True)
    epochs = range(1, len(train_accs) + 1)

    # Loss plot: Train vs Validation
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join("plots", "loss_curve.png"))

    # Accuracy plot: Train vs Validation
    plt.figure()
    plt.plot(epochs, [a*100 for a in train_accs], label="Train Accuracy (%)")
    plt.plot(epochs, [a*100 for a in val_accs], label="Validation Accuracy (%)")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Training vs Validation Accuracy")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join("plots", "accuracy_curve.png"))

    try:
        plt.show()
    except Exception as e:
        print(f"[Info] Could not open plot windows ({e}). PNGs saved in ./plots/")

if __name__ == "__main__":
    main()
