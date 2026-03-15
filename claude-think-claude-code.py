#!/usr/bin/env python3
"""
============================================================
Action Recognition on UCF50 / UCF50_mixed Dataset
============================================================

Three evaluation modes:
  1. No adaptation  (baseline)      → run without any TTA flag
  2. AM-ViTTA (our new method)      → run with --AM_ViTTA flag

Paper Reference (original ViTTA):
  ViTTA: Video Test-Time Adaptation
  https://arxiv.org/pdf/2211.15393

AM-ViTTA (our improvement over ViTTA):
  - Adaptive Momentum EMA: alpha increases as more test videos
    are seen, so early unreliable statistics are trusted less.
  - Spatial augmentation on adaptation clips: adds random crop
    and horizontal flip to give more diverse BN statistics
    beyond just temporal resampling used in original ViTTA.
  - Combined entropy + consistency loss: stabilises adaptation
    by enforcing agreement across temporal views.
  - Clean clip prediction: augmented clips used only for
    adaptation; final prediction uses clean test transform.

HOW TO RUN:
  # Run 1 — train + evaluate WITHOUT adaptation (baseline):
  python ucf50_action_recognition.py --mode train_eval

  # Run 2 — evaluate WITH our new AM-ViTTA (no retraining):
  python ucf50_action_recognition.py --mode eval_only --load_weights best_model.pth --AM_ViTTA

SWAP THE MODEL — only change the 3 lines in MODEL SWAP ZONE below.
============================================================
"""

# ============================================================
# ⚙️  MODEL SWAP ZONE — change only these 3 lines for a new backbone
MODEL_NAME  = "mobilenet_v3_small"   # torchvision model function name
FEATURE_DIM = 576                    # feature channels before head
PRETRAINED  = True                   # use ImageNet pretrained weights
# ============================================================

# MODEL_NAME  = "resnet34"
# FEATURE_DIM = 512
# PRETRAINED  = True

import os, copy, time, random, argparse, warnings
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

CLEAN_DIR = Path("./datasets/UCF50")        # Clean videos for training
MIXED_DIR = Path("./datasets/UCF50_mixed")  # Corrupted videos for testing


# ─────────────────────────────────────────────────────────────
# 1.  ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        description="UCF50 Action Recognition — Baseline vs AM-ViTTA"
    )
    # Paths
    p.add_argument("--clean_dir",    default=CLEAN_DIR,      help="Clean video folder")
    p.add_argument("--mixed_dir",    default=MIXED_DIR,      help="Corrupted video folder")
    p.add_argument("--ckpt_dir",     default="checkpoints",  help="Checkpoint save directory")
    p.add_argument("--best_model",   default="best_model.pth", help="Best model weights path")

    # Data
    p.add_argument("--num_frames",   type=int,   default=16,   help="Frames per video")
    p.add_argument("--img_size",     type=int,   default=112,  help="Spatial resolution H=W")
    p.add_argument("--split_seed",   type=int,   default=42,   help="Seed for 70-30 split")
    p.add_argument("--test_ratio",   type=float, default=0.30)

    # Training
    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--save_every",   type=int,   default=5,    help="Save checkpoint every N epochs")

    # AM-ViTTA hyperparameters
    p.add_argument("--AM_ViTTA",     action="store_true",     help="Enable AM-ViTTA (our new method)")
    p.add_argument("--vitta_clips",  type=int,   default=4,   help="Temporal clips for AM-ViTTA")
    p.add_argument("--vitta_steps",  type=int,   default=3,   help="Gradient steps for entropy minimisation")
    p.add_argument("--vitta_lr",     type=float, default=1e-4, help="AM-ViTTA adaptation learning rate")
    p.add_argument("--warmup_videos",type=int,   default=30,  help="Videos before alpha reaches alpha_max")
    p.add_argument("--alpha_min",    type=float, default=0.3, help="Starting EMA alpha (low trust in test stats)")
    p.add_argument("--alpha_max",    type=float, default=0.5, help="Final EMA alpha (more trust in test stats)")

    # Mode
    p.add_argument("--mode",         default="train_eval",
                   choices=["train_eval", "eval_only"],
                   help="train_eval: train then evaluate | eval_only: load weights and evaluate")
    p.add_argument("--load_weights", type=Path,  default=None, help=".pth file for eval_only mode")
    return p


# ─────────────────────────────────────────────────────────────
# 2.  DATASET HELPERS
# ─────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def collect_videos(root_dir: str):
    """
    Walk root_dir, collect video paths + integer labels.
    Returns (video_paths, labels, class_names).
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
    """Stratified 70-30 split. Returns (train_idx, test_idx)."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    idx = np.arange(len(labels))
    train_idx, test_idx = next(sss.split(idx, labels))
    return train_idx.tolist(), test_idx.tolist()


def sample_frames(video_path: str, num_frames: int):
    """Uniformly sample num_frames frames. Returns list of BGR arrays."""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices  = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
    frames   = []
    prev_idx = -1
    for fi in indices:
        if fi != prev_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        prev_idx = fi
    cap.release()
    while len(frames) < num_frames and frames:
        frames.append(frames[-1])
    return frames[:num_frames]


def temporal_clips(video_path: str, num_frames: int, n_clips: int):
    """
    Sample n_clips temporally diverse clips from the video.
    Returns list-of-lists (each inner list = one clip's BGR frames).
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        return [sample_frames(video_path, num_frames)] * n_clips

    clips = []
    for i in range(n_clips):
        offset = int(i * max(total - num_frames, 0) / max(n_clips - 1, 1))
        cap     = cv2.VideoCapture(video_path)
        indices = np.linspace(offset,
                              min(offset + num_frames - 1, total - 1),
                              num_frames, dtype=int)
        frames  = []
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
    """
    Standard train / test transforms.
    BUG FIX vs original code: train resize is now img_size x img_size
    (was img_size+16 which caused a train/test resolution mismatch).
    """
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),      # FIXED: was img_size+16
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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


def build_vitta_transform(img_size):
    """
    Spatial augmentation transform used by AM-ViTTA during adaptation.
    Applies random crop + flip + colour jitter so that each clip gives
    diverse spatial statistics to the BN alignment step — this goes
    beyond the purely temporal augmentation used in original ViTTA.
    """
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size + 16, img_size + 16)),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


# ─────────────────────────────────────────────────────────────
# 4.  PYTORCH DATASET
# ─────────────────────────────────────────────────────────────
class VideoDataset(Dataset):
    """
    Returns a tensor of shape (T, C, H, W) for each video.
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
            dummy = torch.zeros(self.num_frames, 3,
                                self.transform.transforms[1].size[0],
                                self.transform.transforms[1].size[0])
            return dummy, self.labels[idx]
        tensors = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb))
        clip = torch.stack(tensors, dim=0)   # (T, C, H, W)
        return clip, self.labels[idx]


# ─────────────────────────────────────────────────────────────
# 5.  MODEL BUILDER
# ─────────────────────────────────────────────────────────────
class FrameAggregator(nn.Module):
    """
    Wraps a 2-D CNN backbone for video by:
      1. Flattening (B, T, C, H, W) → (B*T, C, H, W)
      2. Extracting per-frame features via backbone
      3. Mean-pooling over T frames
      4. Classifying with a linear head
    """
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone   = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x     = x.view(B * T, C, H, W)
        feats = self.backbone(x)             # (B*T, feat_dim)
        feats = feats.view(B, T, -1).mean(dim=1)   # (B, feat_dim)
        return self.classifier(feats)        # (B, num_classes)


def build_model(num_classes: int, model_name=MODEL_NAME,
                feat_dim=FEATURE_DIM, pretrained=PRETRAINED):
    """Build backbone, strip its classification head, wrap in FrameAggregator."""
    print(f"\n[MODEL] Loading backbone: {model_name} (pretrained={pretrained})")
    constructor = getattr(tvm, model_name)

    # ── MobileNetV3 ───────────────────────────────────────────
    if "mobilenet_v3" in model_name:
        weights = "DEFAULT" if pretrained else None
        net = constructor(weights=weights)
        backbone = nn.Sequential(net.features, net.avgpool, nn.Flatten())
        with torch.no_grad():
            probe = backbone(torch.zeros(1, 3, 112, 112))
        feat_dim = probe.shape[1]

    # ── EfficientNet ──────────────────────────────────────────
    elif "efficientnet" in model_name:
        weights = "DEFAULT" if pretrained else None
        net = constructor(weights=weights)
        backbone = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        with torch.no_grad():
            probe = backbone(torch.zeros(1, 3, 112, 112))
        feat_dim = probe.shape[1]

    # ── ResNet / ResNeXt ──────────────────────────────────────
    elif "resnet" in model_name or "resnext" in model_name:
        weights = "DEFAULT" if pretrained else None
        net = constructor(weights=weights)
        backbone = nn.Sequential(*list(net.children())[:-1], nn.Flatten())
        with torch.no_grad():
            probe = backbone(torch.zeros(1, 3, 112, 112))
        feat_dim = probe.shape[1]

    # ── Generic fallback ──────────────────────────────────────
    else:
        weights = "DEFAULT" if pretrained else None
        net = constructor(weights=weights)
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

    model      = FrameAggregator(backbone, feat_dim, num_classes)
    total_p    = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total params: {total_p:,}  |  Trainable: {trainable:,}")
    return model


# ─────────────────────────────────────────────────────────────
# 6.  AM-ViTTA  — our new method
#
#  Improvements over original ViTTA:
#  (a) Adaptive Momentum EMA  — alpha grows from alpha_min to
#      alpha_max over warmup_videos test videos, so early
#      test-statistics estimates are trusted less.
#  (b) Spatial augmentation   — random crop + flip + colour
#      jitter on adaptation clips for richer BN statistics.
#  (c) Entropy + Consistency loss — entropy minimisation is
#      combined with cross-clip prediction consistency to
#      stabilise adaptation.
#  (d) Clean clip prediction  — augmented clips adapt the
#      model; final prediction uses the clean test transform
#      so augmentation noise does not hurt inference.
# ─────────────────────────────────────────────────────────────
class AM_ViTTA:
    """
    Adaptive Momentum ViTTA (AM-ViTTA).
    Our improved test-time adaptation method for action recognition.
    """

    def __init__(self, model: nn.Module,
                 n_clips: int        = 4,
                 adapt_steps: int    = 1,
                 adapt_lr: float     = 1e-4,
                 device              = torch.device("cpu"),
                 warmup_videos: int  = 50,
                 alpha_min: float    = 0.05,
                 alpha_max: float    = 0.15):

        self.original_model = model
        self.n_clips        = n_clips
        self.adapt_steps    = adapt_steps
        self.adapt_lr       = adapt_lr
        self.device         = device

        # Adaptive momentum state
        self.video_count    = 0
        self.warmup_videos  = warmup_videos
        self.alpha_min      = alpha_min
        self.alpha_max      = alpha_max

    # ── Adaptive alpha ────────────────────────────────────────
    def _get_alpha(self):
        """
        Linearly ramp alpha from alpha_min → alpha_max over
        warmup_videos videos.
        Early in testing: low alpha  → stay close to training stats.
        Later in testing: higher alpha → trust accumulated test stats more.
        """
        progress = min(self.video_count / max(self.warmup_videos, 1), 1.0)
        return self.alpha_min + progress * (self.alpha_max - self.alpha_min)

    # ── Model copy for BN-only adaptation ────────────────────
    @staticmethod
    def _copy_model_to_adapt(model):
        """Deep-copy model; freeze all params except BN affine weights."""
        adapted = copy.deepcopy(model)
        adapted.train()

        bn_param_names = set()
        for mod_name, module in adapted.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for param_name, _ in module.named_parameters(recurse=False):
                    full = f"{mod_name}.{param_name}" if mod_name else param_name
                    bn_param_names.add(full)

        if not bn_param_names:
            # Fallback: no BN layers — adapt classifier head instead
            print("[AM-ViTTA] ⚠  No BN layers — falling back to classifier head.")
            for mod_name, module in adapted.named_modules():
                if mod_name.startswith("classifier"):
                    for param_name, _ in module.named_parameters(recurse=False):
                        full = f"{mod_name}.{param_name}" if mod_name else param_name
                        bn_param_names.add(full)

        for name, param in adapted.named_parameters():
            param.requires_grad_(name in bn_param_names)

        return adapted

    # ── Entropy helper ────────────────────────────────────────
    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

    # ── (a) Adaptive BN statistics update ────────────────────
    def _update_bn_adaptive(self, adapted_model, clip_tensors):
        
        alpha = self._get_alpha()

    # Save original training stats first
        orig_means = {}
        orig_vars  = {}
        for name, mod in self.original_model.named_modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                orig_means[name] = mod.running_mean.clone()
                orig_vars[name]  = mod.running_var.clone()

    # Forward all clips to update BN stats from test data
        adapted_model.train()
        with torch.no_grad():
            for clip in clip_tensors:
                adapted_model(clip.unsqueeze(0).to(self.device))

    # Blend test stats with training stats using adaptive alpha
        for name, mod in adapted_model.named_modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                if name in orig_means:
                    mod.running_mean.copy_(
                        alpha * mod.running_mean
                        + (1.0 - alpha) * orig_means[name]
                    )
                    mod.running_var.copy_(
                        alpha * mod.running_var
                        + (1.0 - alpha) * orig_vars[name]
                    )
                
    # ── (c) Entropy + Consistency loss ───────────────────────
    def _entropy_consistency_min(self, adapted_model, clip_tensors):
        """
        Minimise entropy of predictions while enforcing consistency
        across temporal clip views (our combined loss vs original
        ViTTA which uses entropy alone).
        """
        trainable = [p for p in adapted_model.parameters() if p.requires_grad]
        if not trainable:
            return
        opt = optim.Adam(trainable, lr=self.adapt_lr)

        for _ in range(self.adapt_steps):
            logits_list = [
                adapted_model(clip.unsqueeze(0).to(self.device))
                for clip in clip_tensors
            ]
            logits_all = torch.cat(logits_list, dim=0)   # (n_clips, C)

            # Entropy loss
            ent_loss = self._entropy(logits_all)

            # Consistency loss: each clip's softmax should match the average
            avg_probs = torch.softmax(logits_all, dim=-1).mean(dim=0, keepdim=True)
            cons_loss = sum(
                torch.abs(torch.softmax(l, dim=-1) - avg_probs).sum()
                for l in logits_list
            ) / len(logits_list)

            loss = ent_loss + 0.5 * cons_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

    # ── Public inference ──────────────────────────────────────
    def predict(self, video_path: str, num_frames: int,
                test_transform, vitta_transform=None):
        """
        AM-ViTTA inference on one video.
        Returns (pred_class, confidence, entropy_value).

        vitta_transform  : spatially augmented transform used only
                           during adaptation (improvement b).
        test_transform   : clean transform used for final prediction
                           (improvement d — no augmentation noise
                           at inference time).
        """
        clips_frames = temporal_clips(video_path, num_frames, self.n_clips)

        # (b) Build spatially-augmented tensors for adaptation
        aug_tf = vitta_transform if vitta_transform is not None else test_transform
        aug_clip_tensors = []
        for frames in clips_frames:
            if not frames:
                continue
            tensors = [aug_tf(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
            aug_clip_tensors.append(torch.stack(tensors, dim=0))

        if not aug_clip_tensors:
            return 0, 0.0, 0.0

        # Adapt a fresh copy of the model
        adapted = self._copy_model_to_adapt(self.original_model)
        adapted.to(self.device)

        # (a) Adaptive BN update
        self._update_bn_adaptive(adapted, aug_clip_tensors)

        # (c) Entropy + consistency minimisation
        if self.adapt_steps > 0:
            self._entropy_consistency_min(adapted, aug_clip_tensors)

        # (d) Final prediction with CLEAN clips (no augmentation)
        clean_clip_tensors = []
        for frames in clips_frames:
            if not frames:
                continue
            tensors = [test_transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                       for f in frames]
            clean_clip_tensors.append(torch.stack(tensors, dim=0))

        adapted.eval()
        with torch.no_grad():
            all_probs = []
            for clip in clean_clip_tensors:
                logits = adapted(clip.unsqueeze(0).to(self.device))
                all_probs.append(torch.softmax(logits, dim=-1))
            avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
            pred      = avg_probs.argmax(dim=-1).item()
            conf      = avg_probs.max().item()
            ent       = self._entropy(torch.log(avg_probs + 1e-8)).item()

        # Advance adaptive momentum counter
        self.video_count += 1

        return pred, conf, ent


# ─────────────────────────────────────────────────────────────
# 7.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    for step, (clips, labels) in enumerate(loader):
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

    return total_loss / total, correct / total * 100.0, class_correct, class_total


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"  ✔ Checkpoint saved → {path}")


# ─────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = build_parser().parse_args()

    # Reproducibility
    random.seed(args.split_seed)
    np.random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine mode label for printing
    if args.AM_ViTTA:
        mode_label = "AM-ViTTA (our new method)"
    else:
        mode_label = "No Adaptation (baseline)"

    print(f"\n{'='*60}")
    print(f"  Action Recognition — UCF50  (device: {device})")
    if torch.cuda.is_available():
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    print(f"  Mode: {mode_label}")
    print(f"{'='*60}\n")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Step 1: Load & split data ─────────────────────────────
    print("[DATA] Collecting videos …")
    clean_paths, clean_labels, class_names = collect_videos(args.clean_dir)
    mixed_paths, mixed_labels, _           = collect_videos(args.mixed_dir)

    num_classes = len(class_names)
    print(f"[DATA] Classes     : {num_classes}")
    print(f"[DATA] Clean videos: {len(clean_paths)}")
    print(f"[DATA] Mixed videos: {len(mixed_paths)}")

    assert len(clean_paths) == len(mixed_paths), \
        "UCF50 and UCF50_mixed must have the same number of videos!"
    assert clean_labels == mixed_labels, \
        "Label lists must match between UCF50 and UCF50_mixed!"

    print(f"[DATA] Stratified 70-30 split (seed={args.split_seed}) …")
    train_idx, test_idx = stratified_split(
        clean_paths, clean_labels, args.test_ratio, args.split_seed
    )

    # TRAIN on clean UCF50 (70%)
    train_paths  = [clean_paths[i] for i in train_idx]
    train_labels = [clean_labels[i] for i in train_idx]

    # TEST on corrupted UCF50_mixed (30%) — same indices, no leakage
    test_paths  = [mixed_paths[i] for i in test_idx]
    test_labels = [mixed_labels[i] for i in test_idx]

    print(f"[DATA] Train samples: {len(train_paths)}")
    print(f"[DATA] Test  samples: {len(test_paths)}")

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
        shuffle=True,  num_workers=args.num_workers,
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
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Step 4: Train ─────────────────────────────────────────
    best_acc = 0.0
    if args.mode == "train_eval":
        print(f"\n{'='*60}")
        print(f"  TRAINING  ({args.epochs} epochs, lr={args.lr})")
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

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), args.best_model)
                print(f"    ★ New best saved → {args.best_model} (Acc={best_acc:.2f}%)")

            if epoch % args.save_every == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch:03d}.pth")
                save_checkpoint(model, optimizer, epoch, ckpt_path)

        print(f"\n[TRAIN DONE] Best Test Accuracy: {best_acc:.2f}%")
        model.load_state_dict(torch.load(args.best_model, map_location=device))

    elif args.mode == "eval_only":
        if args.load_weights is None:
            raise ValueError("--load_weights must be set in eval_only mode")
        model.load_state_dict(torch.load(args.load_weights, map_location=device))
        print(f"[EVAL] Loaded weights from {args.load_weights}")

    # ── Step 5: Evaluation ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EVALUATION — {mode_label}")
    print(f"{'='*60}\n")

    # ── 5a: No adaptation (baseline) ─────────────────────────
    if not args.AM_ViTTA:
        model.eval()
        test_loss, test_acc, class_correct, class_total = evaluate(
            model, test_loader, criterion, device
        )
        print(f"[RESULT] Test Loss : {test_loss:.4f}")
        print(f"[RESULT] Test Acc  : {test_acc:.2f}%")
        print(f"\n[RESULT] Per-class Accuracy (No Adaptation):")
        for cid, cname in enumerate(class_names):
            ct  = class_total.get(cid, 0)
            cc  = class_correct.get(cid, 0)
            pct = (cc / ct * 100) if ct > 0 else 0.0
            print(f"  {cname:<30} {cc:>3}/{ct:>3} = {pct:5.1f}%")
        print(f"\n[RESULT] Overall Accuracy (No Adaptation): {test_acc:.2f}%")

    # ── 5b: AM-ViTTA (our new method) ────────────────────────
    else:
        vitta_tf = build_vitta_transform(args.img_size)

        print(f"[AM-ViTTA] n_clips={args.vitta_clips} | "
              f"adapt_steps={args.vitta_steps} | "
              f"adapt_lr={args.vitta_lr}")
        print(f"[AM-ViTTA] alpha_min={args.alpha_min} | "
              f"alpha_max={args.alpha_max} | "
              f"warmup_videos={args.warmup_videos}")

        engine = AM_ViTTA(
            model          = model,
            n_clips        = args.vitta_clips,
            adapt_steps    = args.vitta_steps,
            adapt_lr       = args.vitta_lr,
            device         = device,
            warmup_videos  = args.warmup_videos,
            alpha_min      = args.alpha_min,
            alpha_max      = args.alpha_max,
        )

        correct       = 0
        total         = 0
        class_correct = defaultdict(int)
        class_total   = defaultdict(int)
        entropy_list  = []

        for idx, (vpath, vlabel) in enumerate(zip(test_paths, test_labels)):
            pred, conf, ent = engine.predict(
                vpath, args.num_frames,
                test_transform  = test_tf,
                vitta_transform = vitta_tf,
            )
            correct += int(pred == vlabel)
            total   += 1
            class_correct[vlabel] += int(pred == vlabel)
            class_total[vlabel]   += 1
            entropy_list.append(ent)

            if (idx + 1) % 20 == 0 or (idx + 1) == len(test_paths):
                running_acc = correct / total * 100.0
                print(f"  [AM-ViTTA] {idx+1}/{len(test_paths)} | "
                      f"Running Acc: {running_acc:.2f}% | "
                      f"AvgEnt: {np.mean(entropy_list):.4f} | "
                      f"alpha: {engine._get_alpha():.3f}")

        test_acc = correct / total * 100.0
        print(f"\n[RESULT] AM-ViTTA Test Accuracy      : {test_acc:.2f}%")
        print(f"[RESULT] Mean Prediction Entropy     : {np.mean(entropy_list):.4f}")
        print(f"\n[RESULT] Per-class Accuracy (AM-ViTTA):")
        for cid, cname in enumerate(class_names):
            ct  = class_total.get(cid, 0)
            cc  = class_correct.get(cid, 0)
            pct = (cc / ct * 100) if ct > 0 else 0.0
            print(f"  {cname:<30} {cc:>3}/{ct:>3} = {pct:5.1f}%")
        print(f"\n[RESULT] Overall AM-ViTTA Accuracy: {test_acc:.2f}%")

    print(f"\n{'='*60}")
    print("  DONE.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()