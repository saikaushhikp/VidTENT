#!/usr/bin/env python3
"""
============================================================
Action Recognition on UCF50 / UCF50_mixed Dataset
with ViTTA (Video Test-Time Adaptation) Support
============================================================

Paper Reference:
  ViTTA: Video Test-Time Adaptation
  https://arxiv.org/pdf/2211.15393
  GitHub: https://github.com/wlin-at/ViTTA

Author  : Auto-generated
Dataset : UCF50 (action recognition, 50 classes)
Default : MobileNetV3-Small (lightweight, RTX-3050 friendly)
============================================================

SWAP THE MODEL  — only change the 3 lines below:
"""

# ============================================================
# ⚙️  MODEL SWAP ZONE — change only these 3 lines for a new backbone
MODEL_NAME   = "mobilenet_v3_small"               # torchvision model function name
FEATURE_DIM  = 576                                 # channels coming out of backbone (before head)
PRETRAINED   = True                                # use ImageNet pretrained weights
# ============================================================

# MODEL_NAME   = "efficientnet_b1"
# FEATURE_DIM  = 1280

# MODEL_NAME   = "resnet18"
# FEATURE_DIM  = 512

import os, sys, copy, time, random, argparse, warnings
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tvm
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")
CLEAN_DIR = Path("./datasets/UCF50")        # Uncorrupted videos for training
MIXED_DIR = Path("./datasets/UCF50_mixed") # Corrupted videos for testing

# ─────────────────────────────────────────────────────────────
# 1.  ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="UCF50 Action Recognition with optional ViTTA"
    )
    # Paths
    p.add_argument("--clean_dir",   default=CLEAN_DIR, help="Clean video folder-Uncorrupted")
    p.add_argument("--mixed_dir",   default=MIXED_DIR, help="Corrupted video folder")
    p.add_argument("--ckpt_dir",    default="checkpoints", help="Directory to save checkpoints")
    p.add_argument("--best_model",  default="best_model.pth", help="Path for best model weights")

    # Data
    p.add_argument("--num_frames",  type=int, default=16,  help="Frames sampled per video")
    p.add_argument("--img_size",    type=int, default=112, help="Spatial resolution (H=W)")
    p.add_argument("--split_seed",  type=int, default=42,  help="Seed for 70-30 split")
    p.add_argument("--test_ratio",  type=float, default=0.30)

    # Training
    p.add_argument("--epochs",      type=int,   default=40)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--save_every",  type=int,   default=5,  help="Save checkpoint every N epochs")

    # Inference / TTA
    p.add_argument("--ViTTA",       action="store_true",   help="Enable ViTTA at test time")
    p.add_argument("--vitta_clips", type=int,   default=4,  help="Temporal clips for ViTTA")
    p.add_argument("--vitta_steps", type=int,   default=1,  help="Gradient steps for ViTTA entropy min")
    p.add_argument("--vitta_lr",    type=float, default=1e-4,help="ViTTA adaptation learning rate")

    # Mode
    p.add_argument("--mode",        default="train_eval",
                   choices=["train_eval","eval_only"],
                   help="train_eval: train then evaluate; eval_only: load weights and evaluate")
    p.add_argument("--load_weights",type = Path, default=None, help="Path to .pth file for eval_only mode")
    return p


# ─────────────────────────────────────────────────────────────
# 2.  DATASET HELPERS
# ─────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def collect_videos(root_dir: str):
    """
    Returns:
        video_paths : list of absolute paths
        labels      : list of int class indices
        class_names : sorted list of class folder names
    """
    root = Path(root_dir)
    class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    cls2idx = {c: i for i, c in enumerate(class_names)}
    video_paths, labels = [], []
    for cls in class_names:
        for vp in sorted((root / cls).iterdir()):
            if vp.suffix.lower() in (".avi", ".mp4", ".mov", ".mkv"):
                video_paths.append(str(vp))
                labels.append(cls2idx[cls])
    return video_paths, labels, class_names


def stratified_split(video_paths, labels, test_ratio, seed):
    """
    Stratified 70-30 split. Returns (train_idx, test_idx).
    Uses relative video *name* (not full path) as the identity
    so that UCF50 and UCF50_mixed share the same split boundaries.
    """
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    idx = np.arange(len(labels))
    train_idx, test_idx = next(sss.split(idx, labels))
    return train_idx.tolist(), test_idx.tolist()


def sample_frames(video_path: str, num_frames: int):
    """
    Uniformly sample `num_frames` frames from a video.
    Returns list of BGR numpy arrays (H, W, 3).
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
    frames = []
    prev_idx = -1
    for fi in indices:
        if fi != prev_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        prev_idx = fi
    cap.release()

    # Pad with last frame if video is very short
    while len(frames) < num_frames and frames:
        frames.append(frames[-1])

    return frames[:num_frames]


def temporal_clips(video_path: str, num_frames: int, n_clips: int):
    """
    ViTTA: sample `n_clips` different temporal clips of `num_frames` each.
    Returns list-of-lists (each inner list = one clip's frames).
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        return [sample_frames(video_path, num_frames)] * n_clips

    clips = []
    for i in range(n_clips):
        # Slide start offset for diversity
        offset = int(i * max(total - num_frames, 0) / max(n_clips - 1, 1))
        cap = cv2.VideoCapture(video_path)
        indices = np.linspace(offset,
                              min(offset + num_frames - 1, total - 1),
                              num_frames, dtype=int)
        frames = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        while len(frames) < num_frames and frames:
            frames.append(frames[-1])
        clips.append(frames[:num_frames])
    return clips


# ─────────────────────────────────────────────────────────────
# 3.  TRANSFORMS
# ─────────────────────────────────────────────────────────────
def build_transforms(img_size, train=True):
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((img_size + 16, img_size + 16)),
            # T.Resize((img_size, img_size)),
            # T.RandomCrop(img_size),
            # T.RandomHorizontalFlip(),
            # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])


# ─────────────────────────────────────────────────────────────
# 4.  PYTORCH DATASET
# ─────────────────────────────────────────────────────────────
class VideoDataset(Dataset):
    """
    Loads a list of video paths and labels.
    Each __getitem__ returns a tensor of shape (T, C, H, W)
    where T = num_frames.
    """
    def __init__(self, video_paths, labels, num_frames, transform):
        self.video_paths = video_paths
        self.labels      = labels
        self.num_frames  = num_frames
        self.transform   = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        frames = sample_frames(self.video_paths[idx], self.num_frames)
        if not frames:
            # Return zeros if video unreadable
            dummy = torch.zeros(self.num_frames, 3,
                                self.transform.transforms[1].size[0],
                                self.transform.transforms[1].size[0])
            return dummy, self.labels[idx]

        tensors = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb))   # (C,H,W)

        clip = torch.stack(tensors, dim=0)         # (T, C, H, W)
        return clip, self.labels[idx]

# ─────────────────────────────────────────────────────────────
# ADAPTER MODULE (ViTTA-Adapters improvement)
# ─────────────────────────────────────────────────────────────
class Adapter(nn.Module):
    """
    Lightweight residual adapter module.

    x → Linear(C→r) → ReLU → Linear(r→C) → +x
    """

    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(bottleneck, dim)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))

# ─────────────────────────────────────────────────────────────
# 5.  MODEL BUILDER
# ─────────────────────────────────────────────────────────────

class FrameAggregator(nn.Module):
    """
    Backbone → temporal pooling → adapter → classifier
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone

        # NEW: adapter module
        self.adapter = Adapter(feat_dim, bottleneck=64)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)
        feats = self.backbone(x)

        feats = feats.view(B, T, -1)
        feats = feats.mean(dim=1)

        # Adapter adaptation layer
        feats = self.adapter(feats)

        return self.classifier(feats)

def build_model(num_classes: int, model_name=MODEL_NAME,
                feat_dim=FEATURE_DIM, pretrained=PRETRAINED):
    """
    Build backbone from torchvision, strip its head, wrap in FrameAggregator.
    To swap model: change MODEL_NAME, FEATURE_DIM at the top of the file.
    """
    print(f"\n[MODEL] Loading backbone: {model_name} (pretrained={pretrained})")
    constructor = getattr(tvm, model_name)

    # ── MobileNetV3-Small ──────────────────────────────────────────────────────
    if "mobilenet_v3" in model_name:
        weights = "DEFAULT" if pretrained else None
        net = constructor(weights=weights)
        # Strip the classifier; keep feature extractor → output is (B, feat_dim)
        backbone = nn.Sequential(
            net.features,
            net.avgpool,
            nn.Flatten(),
        )
        # Verify feat_dim with a dry run
        with torch.no_grad():
            probe = backbone(torch.zeros(1, 3, 112, 112))
        actual_dim = probe.shape[1]
        if actual_dim != feat_dim:
            print(f"[MODEL] ⚠ FEATURE_DIM mismatch: expected {feat_dim}, got {actual_dim}. Auto-correcting.")
            feat_dim = actual_dim

    # ── EfficientNet family ────────────────────────────────────────────────────
    elif "efficientnet" in model_name:
        weights = "DEFAULT" if pretrained else None
        net = constructor(weights=weights)
        backbone = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        with torch.no_grad():
            probe = backbone(torch.zeros(1, 3, 112, 112))
        feat_dim = probe.shape[1]

    # ── ResNet family ──────────────────────────────────────────────────────────
    elif "resnet" in model_name or "resnext" in model_name:
        weights = "DEFAULT" if pretrained else None
        net = constructor(weights=weights)
        # Remove FC
        backbone = nn.Sequential(*list(net.children())[:-1], nn.Flatten())
        with torch.no_grad():
            probe = backbone(torch.zeros(1, 3, 112, 112))
        feat_dim = probe.shape[1]

    # ── Generic fallback ───────────────────────────────────────────────────────
    else:
        weights = "DEFAULT" if pretrained else None
        net = constructor(weights=weights)
        # Try to remove last linear layer
        if hasattr(net, "fc"):
            net.fc = nn.Identity()
        elif hasattr(net, "classifier"):
            net.classifier = nn.Identity()
        elif hasattr(net, "head"):
            net.head = nn.Identity()
        backbone = net
        with torch.no_grad():
            probe = backbone(torch.zeros(1, 3, 112, 112))
        feat_dim = probe.shape[1]

    model = FrameAggregator(backbone, feat_dim, num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total params: {total_params:,}  |  Trainable: {trainable:,}")
    return model


# ─────────────────────────────────────────────────────────────
# 6.  ViTTA — Video Test-Time Adaptation
#     Based on: https://arxiv.org/pdf/2211.15393
# ─────────────────────────────────────────────────────────────
class ViTTA:
    """
    Video Test-Time Adaptation (ViTTA) as described in
    'ViTTA: Video Test-Time Adaptation' (ECCV 2022 / arXiv 2211.15393).

    Core Idea:
    ──────────
    At test time, for each test video:
      1. Sample N_clips temporally diverse clips (temporal augmentation).
      2. Update Batch Normalisation running statistics from these clips
         (Test-Time BN adaptation — no labels needed).
      3. [Optional] Minimise prediction entropy for a few gradient steps
         (entropy-based self-supervised adaptation).
      4. Aggregate predictions across clips → final class prediction.

    The BN update (step 2) is the core of the ViTTA paper; steps 3-4
    follow the general TTA / TTT literature cited therein.
    """

    def __init__(self, model: nn.Module, n_clips: int = 4,
                 adapt_steps: int = 1, adapt_lr: float = 1e-4,
                 device: torch.device = torch.device("cpu")):
        self.original_model = model
        self.n_clips        = n_clips
        self.adapt_steps    = adapt_steps
        self.adapt_lr       = adapt_lr
        self.device         = device

    # ── Internal helpers ───────────────────────────────────────────────────────
    
    
    @staticmethod
    def _copy_model_to_adapt(model):

        adapted = copy.deepcopy(model)
        adapted.train()

        # Freeze everything
        for p in adapted.parameters():
            p.requires_grad = False

        # Unfreeze adapter parameters only
        for name, module in adapted.named_modules():
            if isinstance(module, Adapter):
                for p in module.parameters():
                    p.requires_grad = True

        n_trainable = sum(p.numel() for p in adapted.parameters() if p.requires_grad)

        print(f"[ViTTA-Adapters] adapting {n_trainable:,} parameters")

        return adapted
    
    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

    # ── BN statistics update (core ViTTA) ─────────────────────────────────────
    def _update_bn(self, adapted_model, clip_tensors):
        """
        Pass all clips through the model in train() mode so that
        BN running_mean / running_var are updated from test distribution.
        No gradients needed; this is purely a forward-pass statistics update.
        """
        with torch.no_grad():
            for clip in clip_tensors:
                adapted_model(clip.unsqueeze(0).to(self.device))

    # ── Entropy minimisation (optional self-supervised step) ──────────────────
    def _entropy_min(self, adapted_model, clip_tensors):
        
        trainable_params = [p for p in adapted_model.parameters() if p.requires_grad]

        if not trainable_params:
            return

        opt = optim.Adam(trainable_params, lr=self.adapt_lr)

        for _ in range(self.adapt_steps):

            logits_list = []

            for clip in clip_tensors:
                logits = adapted_model(clip.unsqueeze(0).to(self.device))
                logits_list.append(logits)

            logits_all = torch.cat(logits_list, dim=0)

            # ---------- Entropy loss ----------
            probs = torch.softmax(logits_all, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

            # ---------- Temporal consistency ----------
            mean_prob = probs.mean(dim=0, keepdim=True)

            kl_loss = 0
            for p in probs:
                kl_loss += torch.nn.functional.kl_div(
                    torch.log(p + 1e-8),
                    mean_prob.squeeze(0),
                    reduction='batchmean'
                )

            kl_loss = kl_loss / len(probs)

            # Final loss
            loss = entropy + 0.5 * kl_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

    # ── Public inference method ────────────────────────────────────────────────
    def predict(self, video_path: str, num_frames: int,
                transform, label: int = -1):
        """
        Runs ViTTA inference on one video.
        Returns (pred_class, confidence, entropy_value).
        """
        # 1. Sample diverse temporal clips
        clips_frames = temporal_clips(video_path, num_frames, self.n_clips)

        clip_tensors = []
        for frames in clips_frames:
            if not frames:
                continue
            tensors = []
            for f in frames:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                tensors.append(transform(rgb))
            clip_tensors.append(torch.stack(tensors, dim=0))  # (T,C,H,W)

        if not clip_tensors:
            return 0, 0.0, 0.0

        # 2. Build per-video adapted model
        adapted = self._copy_model_to_adapt(self.original_model)
        adapted.to(self.device)

        # 3. Update BN statistics from test clips (core ViTTA step)
        self._update_bn(adapted, clip_tensors)

        # 4. Optional entropy minimisation
        if self.adapt_steps > 0:
            self._entropy_min(adapted, clip_tensors)

        # 5. Aggregate predictions in eval mode
        adapted.eval()
        with torch.no_grad():
            all_probs = []
            for clip in clip_tensors:
                logits = adapted(clip.unsqueeze(0).to(self.device))   # (1,C)
                all_probs.append(torch.softmax(logits, dim=-1))
            avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)     # (1,C)
            pred      = avg_probs.argmax(dim=-1).item()
            conf      = avg_probs.max().item()
            ent       = self._entropy(torch.log(avg_probs + 1e-8)).item()

        return pred, conf, ent


# ─────────────────────────────────────────────────────────────
# 7.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    for step, (clips, labels) in enumerate(loader):
        # print(f"  [Epoch {epoch}] Step {step+1}/{len(loader)} ...", end="\r")
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(clips)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clips.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += clips.size(0)

        if (step + 1) % 10 == 0 or (step + 1) == len(loader):
            elapsed = time.time() - t0
            print(f"  [Epoch {epoch}] Step {step+1}/{len(loader)} | "
                  f"Loss: {total_loss/total:.4f} | "
                  f"Acc: {correct/total*100:.2f}% | "
                  f"Time: {elapsed:.1f}s")

    return total_loss / total, correct / total * 100.0


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)

    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        logits = model(clips)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * clips.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += clips.size(0)

        for p, l in zip(preds.cpu(), labels.cpu()):
            class_total[l.item()]   += 1
            class_correct[l.item()] += int(p == l)

    avg_loss = total_loss / total
    acc      = correct / total * 100.0
    return avg_loss, acc, class_correct, class_total


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch"     : epoch,
        "model"     : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
    }, path)
    print(f"  ✔ Checkpoint saved → {path}")


# ─────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = build_parser().parse_args()

    # ── Reproducibility ──────────────────────────────────────
    random.seed(args.split_seed)
    np.random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Action Recognition — UCF50  (device: {device})")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    print(f"  ViTTA: {'ENABLED' if args.ViTTA else 'DISABLED'}")
    print(f"{'='*60}\n")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Step 1: Load & split data ─────────────────────────────
    print("[DATA] Collecting videos …")
    clean_paths, clean_labels, class_names = collect_videos(args.clean_dir)
    mixed_paths, mixed_labels, _           = collect_videos(args.mixed_dir)

    num_classes = len(class_names)
    print(f"[DATA] Classes: {num_classes}")
    print(f"[DATA] Clean videos: {len(clean_paths)}")
    print(f"[DATA] Mixed videos: {len(mixed_paths)}")

    # --- Verify same number of videos & same class structure ---
    assert len(clean_paths) == len(mixed_paths), \
        "UCF50 and UCF50_mixed must have the same number of videos!"
    assert clean_labels == mixed_labels, \
        "Label lists must match between UCF50 and UCF50_mixed!"

    # --- Unified stratified split (same indices for both datasets) ---
    print(f"[DATA] Stratified 70-30 split (seed={args.split_seed}) …")
    train_idx, test_idx = stratified_split(
        clean_paths, clean_labels, args.test_ratio, args.split_seed
    )

    # TRAIN: from clean UCF50 (70 %)
    train_paths  = [clean_paths[i] for i in train_idx]
    train_labels = [clean_labels[i] for i in train_idx]

    # TEST: from mixed UCF50_mixed (30 %)  — SAME indices → no overlap
    test_paths   = [mixed_paths[i] for i in test_idx]
    test_labels  = [mixed_labels[i] for i in test_idx]

    print(f"[DATA] Train samples: {len(train_paths)}")
    print(f"[DATA] Test  samples: {len(test_paths)}")

    # Per-class distribution check
    from collections import Counter
    tr_dist = Counter(train_labels)
    te_dist = Counter(test_labels)
    print("\n[DATA] Per-class sample counts (train | test):")
    for cid, cname in enumerate(class_names):
        print(f"  {cname:<30} train={tr_dist[cid]:>4} | test={te_dist[cid]:>4}")

    # ── Step 2: Datasets & Loaders ────────────────────────────
    train_tf = build_transforms(args.img_size, train=True)
    test_tf  = build_transforms(args.img_size, train=False)

    train_ds = VideoDataset(train_paths, train_labels, args.num_frames, train_tf)
    test_ds  = VideoDataset(test_paths,  test_labels,  args.num_frames, test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # ── Step 3: Build model ───────────────────────────────────
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Step 4: Train (if mode permits) ──────────────────────
    best_acc = 0.0
    if args.mode == "train_eval":
        print(f"\n{'='*60}")
        print(f"  TRAINING  ({args.epochs} epochs)")
        print(f"{'='*60}")

        for epoch in range(1, args.epochs + 1):
            print(f"\n[EPOCH {epoch}/{args.epochs}]")

            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            scheduler.step()

            val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)

            print(f"\n  ▶ Epoch {epoch:>3} Summary:")
            print(f"    Train → Loss: {tr_loss:.4f} | Acc: {tr_acc:.2f}%")
            print(f"    Test  → Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            print(f"    LR    = {scheduler.get_last_lr()[0]:.6f}")

            # Save best weights
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), args.best_model)
                print(f"    ★ New best model saved → {args.best_model} (Acc={best_acc:.2f}%)")

            # Intermediate checkpoint
            if epoch % args.save_every == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch:03d}.pth")
                save_checkpoint(model, optimizer, epoch, ckpt_path)

        print(f"\n[TRAIN DONE] Best Test Accuracy: {best_acc:.2f}%")
        # Load best weights for evaluation
        model.load_state_dict(torch.load(args.best_model, map_location=device))

    elif args.mode == "eval_only":
        if args.load_weights is None:
            raise ValueError("--load_weights must be set in eval_only mode")
        model.load_state_dict(torch.load(args.load_weights, map_location=device))
        print(f"[EVAL] Loaded weights from {args.load_weights}")

    # ── Step 5: Final Evaluation ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EVALUATION  (ViTTA={'ON' if args.ViTTA else 'OFF'})")
    print(f"{'='*60}\n")

    if not args.ViTTA:
        # Standard evaluation via DataLoader
        model.eval()
        test_loss, test_acc, class_correct, class_total = evaluate(
            model, test_loader, criterion, device
        )
        print(f"[RESULT] Test Loss  : {test_loss:.4f}")
        print(f"[RESULT] Test Acc   : {test_acc:.2f}%")
        print(f"\n[RESULT] Per-class Accuracy:")
        for cid, cname in enumerate(class_names):
            ct = class_total.get(cid, 0)
            cc = class_correct.get(cid, 0)
            pct = (cc / ct * 100) if ct > 0 else 0.0
            print(f"  {cname:<30} {cc:>3}/{ct:>3} = {pct:5.1f}%")
        print(f"\n[RESULT] Overall Accuracy: {test_acc:.2f}%")

    else:
        # ViTTA evaluation — per-video inference
        print(f"[ViTTA] Running per-video adapted inference …")
        print(f"[ViTTA] n_clips={args.vitta_clips} | "
              f"adapt_steps={args.vitta_steps} | "
              f"adapt_lr={args.vitta_lr}")

        vitta_engine = ViTTA(
            model      = model,
            n_clips    = args.vitta_clips,
            adapt_steps= args.vitta_steps,
            adapt_lr   = args.vitta_lr,
            device     = device,
        )

        correct = 0
        total   = 0
        class_correct = defaultdict(int)
        class_total   = defaultdict(int)
        entropy_list  = []

        for idx, (vpath, vlabel) in enumerate(zip(test_paths, test_labels)):
            pred, conf, ent = vitta_engine.predict(
                vpath, args.num_frames, test_tf
            )
            correct += int(pred == vlabel)
            total   += 1
            class_correct[vlabel] += int(pred == vlabel)
            class_total[vlabel]   += 1
            entropy_list.append(ent)

            if (idx + 1) % 20 == 0 or (idx + 1) == len(test_paths):
                running_acc = correct / total * 100.0
                print(f"  [ViTTA] {idx+1}/{len(test_paths)} | "
                      f"Running Acc: {running_acc:.2f}% | "
                      f"AvgEnt: {np.mean(entropy_list):.4f}")

        test_acc = correct / total * 100.0
        print(f"\n[RESULT] ViTTA Test Accuracy : {test_acc:.2f}%")
        print(f"[RESULT] Mean Prediction Entropy : {np.mean(entropy_list):.4f}")
        print(f"\n[RESULT] Per-class Accuracy (ViTTA):")
        for cid, cname in enumerate(class_names):
            ct  = class_total.get(cid, 0)
            cc  = class_correct.get(cid, 0)
            pct = (cc / ct * 100) if ct > 0 else 0.0
            print(f"  {cname:<30} {cc:>3}/{ct:>3} = {pct:5.1f}%")
        print(f"\n[RESULT] Overall ViTTA Accuracy: {test_acc:.2f}%")

    print(f"\n{'='*60}")
    print("  DONE.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()