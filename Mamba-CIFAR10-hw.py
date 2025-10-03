# Mamba CIFAR10 HW Metrics

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


import time, statistics, warnings, threading
from typing import List, Dict, Any

try:
    from thop import profile  # MACs Counter (forward)
    HAVE_THOP = True
except Exception:
    HAVE_THOP = False
    warnings.warn("[metrics] THOP not found; MACs/FLOPs will be skipped.")

try:
    import pynvml  # NVML for power/energy
    HAVE_NVML = True
except Exception:
    HAVE_NVML = False
    warnings.warn("[metrics] pynvml not found; power/energy will be skipped.")

def _cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

class PowerSampler:
    """NVML-based average power (W) and total energy (J)"""
    def __init__(self, gpu_index=0, sample_hz=50):
        self.enabled = HAVE_NVML and torch.cuda.is_available()
        self.sample_interval = 1.0 / max(1, int(sample_hz))
        self.samples_mw: List[int] = []
        self._running = False
        if self.enabled:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception as e:
                warnings.warn(f"[metrics] NVML init failed: {e}")
                self.enabled = False

    def start(self):
        if not self.enabled or self._running:
            return
        self.samples_mw.clear()
        self._running = True
        self._t0 = time.perf_counter()

        def _worker():
            next_t = time.perf_counter()
            while self._running:
                try:
                    self.samples_mw.append(pynvml.nvmlDeviceGetPowerUsage(self.handle))  # mW
                except Exception:
                    pass
                next_t += self.sample_interval
                dt = next_t - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)

        self._thr = threading.Thread(target=_worker, daemon=True)
        self._thr.start()

    def stop(self):
        if not self.enabled or not self._running:
            return
        self._running = False
        self._thr.join(timeout=2.0)
        self._t1 = time.perf_counter()

    def summary(self) -> Dict[str, float]:
        if not self.enabled or not hasattr(self, "_t1"):
            return {"avg_power_w": float("nan"), "energy_j": float("nan"), "duration_s": 0.0}
        dur = max(0.0, self._t1 - self._t0)
        if not self.samples_mw or dur == 0.0:
            return {"avg_power_w": float("nan"), "energy_j": float("nan"), "duration_s": dur}
        avg_w = (sum(self.samples_mw) / len(self.samples_mw)) / 1000.0
        return {"avg_power_w": avg_w, "energy_j": avg_w * dur, "duration_s": dur}


def count_mamba_ops(module, inputs, output):
      x = inputs[0]
      if isinstance(x, (list, tuple)):
        x = x[0]
    
      B, L, D = x.shape

      expand  = getattr(module, 'expand', 2)
      d_state = getattr(module, 'd_state', 16)
      d_conv  = getattr(module, 'd_conv', 4)

      D_inner = int(D * expand)
   
      mac_proj = B * L * (D * D_inner + D_inner * D)
      mac_conv = B * D * L * d_conv
      mac_ssm = B * L * D * d_state
      total_macs = mac_proj + mac_conv + mac_ssm
   
      import torch as _torch
      module.total_ops += _torch.DoubleTensor([float(total_macs)])

CUSTOM_THOP_OPS = {}
try:
    CUSTOM_THOP_OPS[Mamba] = count_mamba_ops
except Exception:
    pass


# Model
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


# Config & Loaders
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


    power_sample_hz: int = 50
    training_flops_multiplier: float = 3.0  # ~ fw+bw+opt ≈ 3× forward FLOPs


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



# Train / Evaluation
def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train(); loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
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
    model.eval(); loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    total, correct, total_loss = 0, 0, 0.0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images); loss = loss_fn(logits, targets)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1); correct += (preds == targets).sum().item(); total += images.size(0)
    return total_loss / total, correct / total


# Main
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
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and cfg.amp))

    print(f"Device: {device} | Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []  
    test_accs = []

    # Training Measurement
    train_gpu_time_total_s = 0.0
    train_energy_total_j   = 0.0
    train_power_time_s     = 0.0
    train_peak_mem_mb      = 0.0

    # Total samples processed during training (all epochs)
    total_train_samples = len(train_loader.dataset) * cfg.epochs
    
    for epoch in range(1, cfg.epochs + 1):

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start_evt = torch.cuda.Event(enable_timing=True); end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()

        ps_train = PowerSampler(sample_hz=cfg.power_sample_hz); ps_train.start()
        t0 = time.perf_counter()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scaler)

        _cuda_sync(device)
        t1 = time.perf_counter()
        ps_train.stop()
        p = ps_train.summary()
    
        dur = p["duration_s"] if p["duration_s"] > 0 else (t1 - t0)
        train_energy_total_j += (p["energy_j"] if p["energy_j"] == p["energy_j"] else 0.0)
        train_power_time_s   += dur

        if device.type == "cuda":
            end_evt.record(); end_evt.synchronize()
            train_gpu_time_total_s += start_evt.elapsed_time(end_evt) / 1000.0
            train_peak_mem_mb = max(train_peak_mem_mb, torch.cuda.max_memory_allocated(device) / (1024**2))

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

    # Training Metrics
    train_avg_power_w = (train_energy_total_j / train_power_time_s) if train_power_time_s > 0 else float("nan")
    train_latency_ms_per_sample = (train_gpu_time_total_s / total_train_samples) * 1000.0
    train_throughput_sps_gpu = total_train_samples / train_gpu_time_total_s

    # Inference Measurement (Test Phase)
    @torch.no_grad()
    def run_inference_measure():
        gpu_ms_list = []
        samples = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if device.type == "cuda":
            start_evt = torch.cuda.Event(enable_timing=True); end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)
            _ = model(images)
            samples += images.size(0)
        _cuda_sync(device)
        if device.type == "cuda":
            end_evt.record(); end_evt.synchronize()
            total_gpu_s = start_evt.elapsed_time(end_evt) / 1000.0
        else:
            total_gpu_s = float("nan")
        peak_mb = (torch.cuda.max_memory_allocated(device)/(1024**2)) if device.type == "cuda" else float("nan")
        return total_gpu_s, samples, peak_mb

    ps_inf = PowerSampler(sample_hz=cfg.power_sample_hz); ps_inf.start()
    t_inf0 = time.perf_counter()
    inf_gpu_time_total_s, inf_samples, inf_peak_mem_mb = run_inference_measure()
    t_inf1 = time.perf_counter()
    ps_inf.stop()
    inf_power = ps_inf.summary()
    inf_dur_s = inf_power["duration_s"] if inf_power["duration_s"] > 0 else (t_inf1 - t_inf0)
    inf_energy_j = inf_power["energy_j"]
    inf_avg_power_w = (inf_energy_j / inf_dur_s) if inf_dur_s > 0 else float("nan")

    inf_latency_ms_per_sample = (inf_gpu_time_total_s / inf_samples) * 1000.0
    inf_throughput_sps_gpu = inf_samples / inf_gpu_time_total_s

    # MACs / FLOPs
    macs_flops = {}
    try:
        if HAVE_THOP:
            model.eval()
            with torch.no_grad():
                dummy = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device)
                macs, _ = profile(model, inputs=(dummy,), verbose=False, custom_ops=CUSTOM_THOP_OPS)
                flops_fw = macs * 2.0
                macs_flops = {
                    "macs_fw": float(macs),
                    "flops_fw": float(flops_fw),
                    "flops_train_est": float(flops_fw * cfg.training_flops_multiplier),
                }
        else:
            pass
    except Exception as e:
        print(f"[END] FLOPs/MACs counting failed: {e}")

    # GPU Memory Percentage
    def gpu_mem_percent(peak_mb: float) -> float:
        if device.type != "cuda": return float("nan")
        total_bytes = torch.cuda.get_device_properties(device).total_memory
        return (peak_mb * (1024**2)) / total_bytes * 100.0

    train_mem_pct = gpu_mem_percent(train_peak_mem_mb)
    inf_mem_pct   = gpu_mem_percent(inf_peak_mem_mb)

    # Final Report
    print("\n==================== FINAL HW REPORT ====================")
    print("Training phase (overall):")
    print(f"  Execution time (GPU): {train_gpu_time_total_s:.3f} s")
    print(f"  Throughput (GPU):     {train_throughput_sps_gpu:.2f} samples/sec")
    print(f"  Latency (avg/sample): {train_latency_ms_per_sample:.3f} ms/sample")
    print(f"  Peak GPU memory:      {train_peak_mem_mb:.1f} MB ({train_mem_pct:.2f}%)")
    print(f"  Power (avg NVML):     {train_avg_power_w:.2f} W")
    print(f"  Energy (NVML):        {train_energy_total_j:.2f} J")

    print("\nTest phase – inference (overall on test loader):")
    print(f"  Inference time (GPU): {inf_gpu_time_total_s:.3f} s")
    print(f"  Throughput (GPU):     {inf_throughput_sps_gpu:.2f} samples/sec")
    print(f"  Latency (avg/sample): {inf_latency_ms_per_sample:.3f} ms/sample")
    print(f"  Peak GPU memory:      {inf_peak_mem_mb:.1f} MB ({inf_mem_pct:.2f}%)")
    print(f"  Power (avg NVML):     {inf_avg_power_w:.2f} W")
    print(f"  Energy (NVML):        {inf_energy_j:.2f} J")

    if macs_flops:
        print("\nModel complexity:")
        print(f"  MACs (forward):       {macs_flops['macs_fw']:.2e}")
        print(f"  FLOPs (forward):      {macs_flops['flops_fw']:.2e}")
        print(f"  FLOPs (train est.):   {macs_flops['flops_train_est']:.2e}")
    else:
        print("\nModel complexity: skipped (THOP unavailable).")

    # Accuracy 
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

    # Plots
    os.makedirs("plots", exist_ok=True)
    epochs = range(1, len(train_accs) + 1)

    # Loss Plot: Train vs Validation
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join("plots", "loss_curve.png"))

    # Accuracy Plot: Train vs Validation
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
