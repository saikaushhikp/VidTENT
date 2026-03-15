#!/usr/bin/env python3
"""
============================================================
Action Recognition on UCF50 / UCF50_mixed Dataset
with ViTTA (CVPR-2023) AND RMGA (Rhythmic Motion-Gated
Adaptation — novel TTA) Support

Extended with 3D Video Model Support:
  torchvision.models.video.r3d_18
  torchvision.models.video.mc3_18
  torchvision.models.video.r2plus1d_18
============================================================

Paper References:
  ViTTA : Video Test-Time Adaptation
          https://arxiv.org/pdf/2211.15393
  RMGA  : Rhythmic Motion-Gated Adaptation (this work)
          Spatio-Temporal Masking + Peak Anchoring for
          corruption-robust video TTA on consumer hardware.

Author  : Research implementation
Dataset : UCF50 (action recognition, 50 classes)
Default : MobileNetV3-Small (lightweight, RTX-3050 friendly)
============================================================

SWAP THE MODEL  — only change the 3 lines below:

Commands via CLI:

"""

# ============================================================
# ⚙️  MODEL SWAP ZONE — change only these 3 lines for a new backbone
#
# ── 2D CNN options (existing pipeline) ──
MODEL_NAME   = "mobilenet_v3_small"   # torchvision model function name
FEATURE_DIM  = 576                    # channels out of 2D backbone (ignored for video models)
PRETRAINED   = True                   # use ImageNet / Kinetics pretrained weights
#
# ── 3D Video model options (NEW) ──────────
# MODEL_NAME = "r3d_18"               # ResNet-3D-18      | input (B,C,T,H,W)
# MODEL_NAME = "mc3_18"               # Mixed-Conv3D-18   | input (B,C,T,H,W)
# MODEL_NAME = "r2plus1d_18"          # R(2+1)D-18        | input (B,C,T,H,W)
#
# ── Other 2D CNN options ──────────────────
# MODEL_NAME = "efficientnet_b1"      # FEATURE_DIM = 1280
# MODEL_NAME = "resnet18"             # FEATURE_DIM = 512
# ============================================================

# ── [3D SUPPORT] Registry of video model names ───────────────────────────────
# Any MODEL_NAME found in this set triggers the 3D pipeline automatically.
# Add new torchvision.models.video entries here as needed.
VIDEO_MODELS = {"r3d_18", "mc3_18", "r2plus1d_18"}

# ── [3D SUPPORT] Global flag — drives all shape-branching in the pipeline ─────
# True  → 3D video model  : dataset outputs (C, T, H, W),
#                            model forward expects (B, C, T, H, W)
# False → 2D CNN pipeline : dataset outputs (T, C, H, W),  ← unchanged
#                            model forward expects (B, T, C, H, W)
IS_VIDEO_MODEL = MODEL_NAME in VIDEO_MODELS
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, copy, time, random, argparse, warnings
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tvm
import torchvision.models.video as tvm_video   # [3D SUPPORT] video model namespace
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")
CLEAN_DIR = Path("./datasets/UCF50")         # Uncorrupted videos for training
MIXED_DIR = Path("./datasets/UCF50_mixed")   # Corrupted videos for testing


# ─────────────────────────────────────────────────────────────
# 1.  ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="UCF50 Action Recognition — 2D CNNs + 3D Video Models + ViTTA + RMGA"
    )
    # ── Paths ─────────────────────────────────────────────────
    p.add_argument("--clean_dir",    default=CLEAN_DIR, help="Clean video folder (uncorrupted)")
    p.add_argument("--mixed_dir",    default=MIXED_DIR, help="Corrupted video folder")
    p.add_argument("--ckpt_dir",     default="checkpoints", help="Directory to save checkpoints")
    p.add_argument("--best_model",   default="best_model.pth", help="Path for best model weights")

    # ── Data ──────────────────────────────────────────────────
    p.add_argument("--num_frames",   type=int,   default=16,   help="Frames sampled per video")
    p.add_argument("--img_size",     type=int,   default=112,  help="Spatial resolution (H=W)")
    p.add_argument("--split_seed",   type=int,   default=42,   help="Seed for 70-30 split")
    p.add_argument("--test_ratio",   type=float, default=0.30)

    # ── Training ──────────────────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--save_every",   type=int,   default=5,    help="Save checkpoint every N epochs")

    # ── ViTTA ─────────────────────────────────────────────────
    p.add_argument("--ViTTA",        action="store_true",      help="Enable ViTTA at test time")
    p.add_argument("--vitta_clips",  type=int,   default=4,    help="Temporal clips for ViTTA")
    p.add_argument("--vitta_steps",  type=int,   default=1,    help="Gradient steps for ViTTA entropy min")
    p.add_argument("--vitta_lr",     type=float, default=1e-4, help="ViTTA adaptation learning rate")

    # ── RMGA ──────────────────────────────────────────────────
    p.add_argument("--RMGA",         action="store_true",
                   help="Enable RMGA (Rhythmic Motion-Gated Adaptation) at test time")
    p.add_argument("--rmga_window",  type=int,   default=8,
                   help="[RMGA] Sliding temporal window size W for peak anchoring")
    p.add_argument("--rmga_steps",   type=int,   default=1,
                   help="[RMGA] Adaptation gradient steps per video")
    p.add_argument("--rmga_lr",      type=float, default=1e-4,
                   help="[RMGA] Adaptation learning rate")
    p.add_argument("--rmga_tau",     type=float, default=0.05,
                   help="[RMGA] Motion-mask threshold τ (pixel diff on normalised frames)")
    p.add_argument("--rmga_last_blocks", type=int, default=3,
                   help="[RMGA] Number of last BN blocks to adapt (selective adaptation)")
    p.add_argument("--rmga_fp16",    action="store_true",
                   help="[RMGA] Use FP16 (torch.cuda.amp) — halves VRAM on RTX-3050")
    p.add_argument("--rmga_extra_clips", type=int, default=2,
                   help="[RMGA] Extra diverse clips for initial BN-stats warm-up (like ViTTA)")

    # ── Mode ──────────────────────────────────────────────────
    p.add_argument("--mode",         default="train_eval",
                   choices=["train_eval", "eval_only"],
                   help="train_eval: train then evaluate; eval_only: load weights and evaluate")
    p.add_argument("--load_weights", type=Path, default=None,
                   help="Path to .pth file for eval_only mode")
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
    ViTTA / RMGA: sample `n_clips` different temporal clips of `num_frames` each.
    Returns list-of-lists (each inner list = one clip's BGR frames).
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        return [sample_frames(video_path, num_frames)] * n_clips

    clips = []
    for i in range(n_clips):
        offset  = int(i * max(total - num_frames, 0) / max(n_clips - 1, 1))
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
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((img_size + 16, img_size + 16)),
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

    Output tensor shape depends on `channel_first`:
      channel_first=False  →  (T, C, H, W)   [2D CNN pipeline — default]
      channel_first=True   →  (C, T, H, W)   [3D video model pipeline — NEW]

    The DataLoader then adds the batch dimension B, giving:
      2D CNN  : (B, T, C, H, W)
      3D Video: (B, C, T, H, W)

    [3D SUPPORT] The `channel_first` flag is the only change to this class.
    Everything else is unchanged.
    """
    def __init__(self, video_paths, labels, num_frames, transform,
                 channel_first: bool = False):
        self.video_paths   = video_paths
        self.labels        = labels
        self.num_frames    = num_frames
        self.transform     = transform
        # [3D SUPPORT] True → output (C, T, H, W) for video models
        self.channel_first = channel_first

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        frames = sample_frames(self.video_paths[idx], self.num_frames)
        if not frames:
            h = self.transform.transforms[1].size[0]
            dummy = torch.zeros(self.num_frames, 3, h, h)
            # [3D SUPPORT] permute dummy too so shapes stay consistent
            if self.channel_first:
                dummy = dummy.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)
            return dummy, self.labels[idx]

        tensors = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb))   # (C, H, W)

        clip = torch.stack(tensors, dim=0)         # (T, C, H, W)

        # [3D SUPPORT] Transpose to (C, T, H, W) when using a video model.
        # The .contiguous() call ensures memory layout is correct after permute.
        if self.channel_first:
            clip = clip.permute(1, 0, 2, 3).contiguous()   # (C, T, H, W)

        return clip, self.labels[idx]


# ─────────────────────────────────────────────────────────────
# 5.  MODEL BUILDER
# ─────────────────────────────────────────────────────────────

# ── 2D backbone wrapper (UNCHANGED) ──────────────────────────
class FrameAggregator(nn.Module):
    """
    Wraps a 2-D CNN backbone.
    For each video clip (B, T, C, H, W):
      - Extracts frame-level features  (B*T, feat_dim)
      - Mean-pools over T frames        (B, feat_dim)
      - Classifies                      (B, num_classes)
    Used ONLY when IS_VIDEO_MODEL is False.
    """
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone   = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x     = x.view(B * T, C, H, W)   # (B*T, C, H, W)
        feats = self.backbone(x)           # (B*T, feat_dim)
        feats = feats.view(B, T, -1)       # (B, T, feat_dim)
        feats = feats.mean(dim=1)          # (B, feat_dim)  — temporal pooling
        return self.classifier(feats)      # (B, num_classes)


# ── [3D SUPPORT] 3D video model wrapper ───────────────────────
class VideoModel3D(nn.Module):
    """
    Thin wrapper around torchvision 3D video backbones
    (r3d_18, mc3_18, r2plus1d_18).

    These models already include their own temporal modelling and a
    final FC layer. We simply replace the FC head to match num_classes.

    Forward:
      input  : (B, C, T, H, W)   ← note channel-before-time ordering
      output : (B, num_classes)

    The wrapper exists so the rest of the pipeline (ViTTA, RMGA, evaluate)
    can call model(x) uniformly regardless of backbone type.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        # The head-replaced backbone is the entire model;
        # no separate classifier is needed.
        self.backbone = backbone

    def forward(self, x):
        # x: (B, C, T, H, W)
        return self.backbone(x)   # (B, num_classes)


def build_model(num_classes: int, model_name=MODEL_NAME,
                feat_dim=FEATURE_DIM, pretrained=PRETRAINED):
    """
    Build and return the appropriate model.

    [3D SUPPORT] If model_name is in VIDEO_MODELS, loads from
    torchvision.models.video and wraps in VideoModel3D.
    Otherwise, existing 2D FrameAggregator pipeline is used unchanged.
    """
    print(f"\n[MODEL] Loading backbone : {model_name} (pretrained={pretrained})")
    print(f"[MODEL] Pipeline         : {'3D Video Model' if IS_VIDEO_MODEL else '2D CNN + FrameAggregator'}")

    # ── [3D SUPPORT] Video model branch ───────────────────────────────────────
    if model_name in VIDEO_MODELS:
        weights_arg = "DEFAULT" if pretrained else None

        # Map name → torchvision.models.video constructor
        _video_constructors = {
            "r3d_18"      : tvm_video.r3d_18,
            "mc3_18"      : tvm_video.mc3_18,
            "r2plus1d_18" : tvm_video.r2plus1d_18,
        }
        constructor = _video_constructors[model_name]
        net         = constructor(weights=weights_arg)

        # Replace the final FC layer to match num_classes.
        # All three supported models expose a top-level `.fc`.
        in_features = net.fc.in_features
        net.fc      = nn.Linear(in_features, num_classes)
        print(f"[MODEL] FC head replaced: {in_features} → {num_classes}")

        model        = VideoModel3D(net)
        total_params = sum(p.numel() for p in model.parameters())
        trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MODEL] Total params: {total_params:,}  |  Trainable: {trainable:,}")
        return model

    # ── 2D CNN branch (UNCHANGED) ──────────────────────────────────────────────
    constructor = getattr(tvm, model_name)

    if "mobilenet_v3" in model_name:
        weights  = "DEFAULT" if pretrained else None
        net      = constructor(weights=weights)
        backbone = nn.Sequential(net.features, net.avgpool, nn.Flatten())
        with torch.no_grad():
            probe    = backbone(torch.zeros(1, 3, 112, 112))
        actual_dim = probe.shape[1]
        if actual_dim != feat_dim:
            print(f"[MODEL] ⚠ FEATURE_DIM mismatch: expected {feat_dim}, "
                  f"got {actual_dim}. Auto-correcting.")
            feat_dim = actual_dim

    elif "efficientnet" in model_name:
        weights  = "DEFAULT" if pretrained else None
        net      = constructor(weights=weights)
        backbone = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        with torch.no_grad():
            probe    = backbone(torch.zeros(1, 3, 112, 112))
        feat_dim = probe.shape[1]

    elif "resnet" in model_name or "resnext" in model_name:
        weights  = "DEFAULT" if pretrained else None
        net      = constructor(weights=weights)
        backbone = nn.Sequential(*list(net.children())[:-1], nn.Flatten())
        with torch.no_grad():
            probe    = backbone(torch.zeros(1, 3, 112, 112))
        feat_dim = probe.shape[1]

    else:
        weights = "DEFAULT" if pretrained else None
        net     = constructor(weights=weights)
        if hasattr(net, "fc"):
            net.fc = nn.Identity()
        elif hasattr(net, "classifier"):
            net.classifier = nn.Identity()
        elif hasattr(net, "head"):
            net.head = nn.Identity()
        backbone = net
        with torch.no_grad():
            probe    = backbone(torch.zeros(1, 3, 112, 112))
        feat_dim = probe.shape[1]

    model        = FrameAggregator(backbone, feat_dim, num_classes)
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
    Video Test-Time Adaptation (ViTTA).

    [3D SUPPORT] The only change vs. the original is in predict():
    clip tensors are built in the correct shape for the active pipeline
    and unsqueezed to add batch dim accordingly.

    2D CNN  : clip (T, C, H, W) → unsqueeze(0) → (1, T, C, H, W)
    3D Video: clip (C, T, H, W) → unsqueeze(0) → (1, C, T, H, W)
    """

    def __init__(self, model: nn.Module, n_clips: int = 4,
                 adapt_steps: int = 1, adapt_lr: float = 1e-4,
                 device: torch.device = torch.device("cpu"),
                 is_video_model: bool = False):
        self.original_model = model
        self.n_clips        = n_clips
        self.adapt_steps    = adapt_steps
        self.adapt_lr       = adapt_lr
        self.device         = device
        # [3D SUPPORT] flag controls clip tensor shape in predict()
        self.is_video_model = is_video_model

    @staticmethod
    def _copy_model_to_adapt(model):
        """Deep-copy; unfreeze only BN affine params (γ, β)."""
        adapted = copy.deepcopy(model)
        adapted.train()

        bn_param_names = set()
        for mod_name, module in adapted.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for param_name, _ in module.named_parameters(recurse=False):
                    full_name = f"{mod_name}.{param_name}" if mod_name else param_name
                    bn_param_names.add(full_name)

        if not bn_param_names:
            print("[ViTTA] ⚠  No BatchNorm found — falling back to classifier head.")
            for mod_name, module in adapted.named_modules():
                if mod_name.startswith("classifier") or mod_name.startswith("backbone.fc"):
                    for param_name, _ in module.named_parameters(recurse=False):
                        full_name = f"{mod_name}.{param_name}" if mod_name else param_name
                        bn_param_names.add(full_name)

        for name, param in adapted.named_parameters():
            param.requires_grad_(name in bn_param_names)

        return adapted

    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

    def _update_bn(self, adapted_model, clip_tensors):
        with torch.no_grad():
            for clip in clip_tensors:
                adapted_model(clip.unsqueeze(0).to(self.device))

    def _entropy_min(self, adapted_model, clip_tensors):
        trainable_params = [p for p in adapted_model.parameters() if p.requires_grad]
        if not trainable_params:
            print("[ViTTA] ⚠  _entropy_min skipped — no trainable params.")
            return
        opt = optim.Adam(trainable_params, lr=self.adapt_lr)
        for _ in range(self.adapt_steps):
            logits_list = [
                adapted_model(clip.unsqueeze(0).to(self.device))
                for clip in clip_tensors
            ]
            logits_all = torch.cat(logits_list, dim=0)
            loss = self._entropy(logits_all)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def predict(self, video_path: str, num_frames: int,
                transform, label: int = -1):
        """
        Runs ViTTA inference on one video.
        Returns (pred_class, confidence, entropy_value).

        [3D SUPPORT] clip tensors are transposed to (C, T, H, W) when
        self.is_video_model is True, so the model receives (1, C, T, H, W).
        """
        clips_frames = temporal_clips(video_path, num_frames, self.n_clips)
        clip_tensors = []
        for frames in clips_frames:
            if not frames:
                continue
            tensors = [transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
            clip = torch.stack(tensors, dim=0)   # (T, C, H, W)

            # [3D SUPPORT] video models need (C, T, H, W)
            if self.is_video_model:
                clip = clip.permute(1, 0, 2, 3).contiguous()   # (C, T, H, W)

            clip_tensors.append(clip)

        if not clip_tensors:
            return 0, 0.0, 0.0

        adapted = self._copy_model_to_adapt(self.original_model)
        adapted.to(self.device)

        self._update_bn(adapted, clip_tensors)

        if self.adapt_steps > 0:
            self._entropy_min(adapted, clip_tensors)

        adapted.eval()
        with torch.no_grad():
            all_probs = []
            for clip in clip_tensors:
                # unsqueeze(0) adds B=1;
                # shape becomes (1,T,C,H,W) for 2D or (1,C,T,H,W) for 3D
                logits = adapted(clip.unsqueeze(0).to(self.device))
                all_probs.append(torch.softmax(logits, dim=-1))
            avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
            pred = avg_probs.argmax(dim=-1).item()
            conf = avg_probs.max().item()
            ent  = self._entropy(torch.log(avg_probs + 1e-8)).item()

        return pred, conf, ent


# ─────────────────────────────────────────────────────────────
# 7.  RMGA — Rhythmic Motion-Gated Adaptation  (novel TTA)
# ─────────────────────────────────────────────────────────────
class RMGA:
    """
    Rhythmic Motion-Gated Adaptation (RMGA).

    Two Core Innovations:
      1. Spatio-Temporal Motion Masking
         M_t = 𝟙(|x_t − x_{t−1}| > τ)
         Adaptation loss is gated by motion pixels only.

      2. Rhythmic Peak Anchoring
         t* = argmin_{t ∈ W} H(y_t)
         Only the most-confident frame in each window triggers backprop.

    [3D SUPPORT] Changes vs. original:
      • __init__ accepts is_video_model flag.
      • frames_to_tensor() permutes to (C,T,H,W) for 3D models.
      • _rhythmic_adapt() feeds single-frame through the appropriate
        shape: (1,1,C,H,W) for 2D, (1,C,1,H,W) for 3D.
      • predict() passes channel_first clips to the BN warmup and adapt.
    """

    def __init__(
        self,
        model:          nn.Module,
        window_size:    int   = 8,
        adapt_steps:    int   = 1,
        adapt_lr:       float = 1e-4,
        tau:            float = 0.05,
        last_bn_blocks: int   = 3,
        extra_clips:    int   = 2,
        device:         torch.device = torch.device("cpu"),
        use_fp16:       bool  = False,
        is_video_model: bool  = False,
    ):
        self.original_model = model
        self.window_size    = window_size
        self.adapt_steps    = adapt_steps
        self.adapt_lr       = adapt_lr
        self.tau            = tau
        self.last_bn_blocks = last_bn_blocks
        self.extra_clips    = extra_clips
        self.device         = device
        self.use_fp16       = use_fp16 and torch.cuda.is_available()
        # [3D SUPPORT] drives clip shape throughout this class
        self.is_video_model = is_video_model

    # ── Selective BN adaptation ────────────────────────────────────────────────
    @staticmethod
    def _copy_model_to_adapt(model: nn.Module, last_bn_blocks: int = 3):
        """Unfreeze only the last N BN blocks (γ, β)."""
        adapted = copy.deepcopy(model)
        adapted.train()

        bn_params_ordered = []
        for mod_name, module in adapted.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for param_name, _ in module.named_parameters(recurse=False):
                    full_name = (f"{mod_name}.{param_name}"
                                 if mod_name else param_name)
                    bn_params_ordered.append(full_name)

        if bn_params_ordered:
            n_select = min(last_bn_blocks * 2, len(bn_params_ordered))
            selected = set(bn_params_ordered[-n_select:])
        else:
            selected = set()

        if not selected:
            print("[RMGA] ⚠  No BN found — falling back to classifier/FC head.")
            for mod_name, module in adapted.named_modules():
                if mod_name.startswith("classifier") or mod_name.startswith("backbone.fc"):
                    for param_name, _ in module.named_parameters(recurse=False):
                        full_name = (f"{mod_name}.{param_name}"
                                     if mod_name else param_name)
                        selected.add(full_name)

        for name, param in adapted.named_parameters():
            param.requires_grad_(name in selected)

        n_trainable = sum(p.numel() for p in adapted.parameters() if p.requires_grad)
        print(f"[RMGA] Adapting {len(selected)} BN tensors "
              f"({n_trainable:,} scalars) — last {last_bn_blocks} BN blocks.")
        return adapted, selected

    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

    @staticmethod
    def _compute_motion_mask(
        frame_t:    torch.Tensor,   # (C, H, W)
        frame_prev: torch.Tensor,   # (C, H, W)
        tau:        float,
    ) -> torch.Tensor:
        """
        M_t = 𝟙( mean_C |x_t − x_{t−1}| > τ )
        Returns float32 (1, H, W) mask.
        """
        diff = torch.abs(frame_t - frame_prev)
        return (diff.mean(dim=0, keepdim=True) > tau).float()

    def _warmup_bn(self, adapted_model: nn.Module, clip_tensors: list):
        """Forward-only BN statistics alignment from test clips."""
        with torch.no_grad():
            for clip in clip_tensors:
                adapted_model(clip.unsqueeze(0).to(self.device))

    def _rhythmic_adapt(self, adapted_model: nn.Module, clip_tensors: list):
        """
        Core RMGA loop: motion-gated peak-anchored backward pass.

        [3D SUPPORT] For 3D video models, a single-frame input is shaped
        as (1, C, 1, H, W) instead of (1, 1, C, H, W).
        This is because 3D models expect (B, C, T, H, W).

        The motion mask is always computed in (C, H, W) space regardless
        of model type, so _compute_motion_mask is model-agnostic.
        """
        trainable_params = [p for p in adapted_model.parameters()
                            if p.requires_grad]
        if not trainable_params:
            print("[RMGA] ⚠  No trainable params — skipping adaptation.")
            return

        opt    = optim.Adam(trainable_params, lr=self.adapt_lr)
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)

        for _step in range(self.adapt_steps):
            for clip in clip_tensors:
                # clip is either (T,C,H,W) for 2D or (C,T,H,W) for 3D
                # Normalise to (T, C, H, W) internally for motion mask logic
                if self.is_video_model:
                    # (C, T, H, W) → (T, C, H, W) for frame indexing
                    clip_tchw = clip.permute(1, 0, 2, 3)   # (T, C, H, W)
                else:
                    clip_tchw = clip                        # already (T, C, H, W)

                T_len, C, H, W = clip_tchw.shape
                clip_dev       = clip_tchw.to(self.device)

                # ── 1. Build motion masks (T, 1, H, W) ────────────────
                masks = [torch.ones(1, H, W, device=self.device)]
                for t in range(1, T_len):
                    m = self._compute_motion_mask(
                        clip_dev[t], clip_dev[t - 1], self.tau
                    )
                    masks.append(m)
                masks = torch.stack(masks, dim=0)   # (T, 1, H, W)

                # ── 2. Slide windows ───────────────────────────────────
                W_size    = min(self.window_size, T_len)
                n_windows = max(1, (T_len + W_size - 1) // W_size)

                for w in range(n_windows):
                    start   = w * W_size
                    end     = min(start + W_size, T_len)
                    win_f   = clip_dev[start:end]   # (win_len, C, H, W)  in TCHW
                    win_m   = masks[start:end]       # (win_len, 1, H, W)
                    win_len = end - start

                    # ── 3. Per-frame entropy scan (no grad) ───────────
                    with torch.no_grad():
                        # Each frame fed as a single-frame "video"
                        # 2D : (frame, 1, C, H, W)  → (B=frame, T=1, C, H, W)
                        # 3D : (frame, C, 1, H, W)  → (B=frame, C, T=1, H, W)
                        if self.is_video_model:
                            # win_f: (win_len, C, H, W)
                            # need  : (win_len, C, 1, H, W)
                            batch_in = win_f.unsqueeze(2)  # (win_len, C, 1, H, W)
                        else:
                            # win_f: (win_len, C, H, W)
                            # need  : (win_len, 1, C, H, W)
                            batch_in = win_f.unsqueeze(1)  # (win_len, 1, C, H, W)

                        batch_logits    = adapted_model(batch_in)
                        frame_entropies = [
                            self._entropy(batch_logits[t:t+1]).item()
                            for t in range(win_len)
                        ]

                    # ── 4. Peak anchoring: t* = argmin H ──────────────
                    peak_t        = int(np.argmin(frame_entropies))
                    motion_weight = win_m[peak_t].mean().item()

                    if motion_weight < 1e-4:
                        continue   # no motion → skip window

                    # ── 5. Motion-gated backward on peak frame ─────────
                    peak_frame  = win_f[peak_t]             # (C, H, W)
                    peak_mask   = win_m[peak_t]             # (1, H, W)
                    masked_frame = peak_frame * peak_mask   # (C, H, W)

                    # Shape for model forward:
                    # 2D : (1, 1, C, H, W)  i.e. B=1, T=1
                    # 3D : (1, C, 1, H, W)  i.e. B=1, C, T=1
                    if self.is_video_model:
                        # masked_frame (C,H,W) → (1,C,1,H,W)
                        peak_input = masked_frame.unsqueeze(0).unsqueeze(2)
                    else:
                        # masked_frame (C,H,W) → (1,1,C,H,W)
                        peak_input = masked_frame.unsqueeze(0).unsqueeze(0)

                    opt.zero_grad()
                    with torch.cuda.amp.autocast(enabled=self.use_fp16):
                        logits = adapted_model(peak_input)
                        loss   = self._entropy(logits) * motion_weight

                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

    def predict(self, video_path: str, num_frames: int,
                transform, label: int = -1):
        """
        Full RMGA inference pipeline for one video.

        [3D SUPPORT] frames_to_tensor() applies the channel-first permute
        when self.is_video_model is True, so all downstream logic
        (warmup, adapt, aggregate) receives the correct tensor shape.
        """

        def frames_to_tensor(frames):
            """BGR frame list → (T,C,H,W) or (C,T,H,W) depending on model type."""
            tensors = [transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                       for f in frames]
            clip = torch.stack(tensors, dim=0)   # (T, C, H, W)
            # [3D SUPPORT] permute for video models
            if self.is_video_model:
                clip = clip.permute(1, 0, 2, 3).contiguous()   # (C, T, H, W)
            return clip

        # ── Primary sequential clip ────────────────────────────
        primary_frames = sample_frames(video_path, num_frames)
        if not primary_frames:
            return 0, 0.0, 0.0

        primary_clip     = frames_to_tensor(primary_frames)
        all_clip_tensors = [primary_clip]

        # ── Extra diverse clips for BN warm-up ────────────────
        if self.extra_clips > 0:
            for ef in temporal_clips(video_path, num_frames, self.extra_clips):
                if ef:
                    all_clip_tensors.append(frames_to_tensor(ef))

        # ── Build adapted model (selective BN) ────────────────
        adapted, _ = self._copy_model_to_adapt(
            self.original_model, self.last_bn_blocks
        )
        adapted.to(self.device)

        # ── BN warm-up (forward only) ─────────────────────────
        self._warmup_bn(adapted, all_clip_tensors)

        # ── RMGA rhythmic backward ────────────────────────────
        self._rhythmic_adapt(adapted, all_clip_tensors)

        # ── Aggregate predictions ─────────────────────────────
        adapted.eval()
        with torch.no_grad():
            all_probs = []
            for clip in all_clip_tensors:
                logits = adapted(clip.unsqueeze(0).to(self.device))
                all_probs.append(torch.softmax(logits, dim=-1))
            avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
            pred = avg_probs.argmax(dim=-1).item()
            conf = avg_probs.max().item()
            ent  = self._entropy(torch.log(avg_probs + 1e-8)).item()

        return pred, conf, ent


# ─────────────────────────────────────────────────────────────
# 8.  TRAINING LOOP  (UNCHANGED)
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

    avg_loss = total_loss / total
    acc      = correct / total * 100.0
    return avg_loss, acc, class_correct, class_total


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch"    : epoch,
        "model"    : model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"  ✔ Checkpoint saved → {path}")


# ─────────────────────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = build_parser().parse_args()

    # ── Reproducibility ──────────────────────────────────────
    random.seed(args.split_seed)
    np.random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  Action Recognition — UCF50  (device: {device})")
    if torch.cuda.is_available():
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    # [3D SUPPORT] surface the active pipeline in the startup banner
    print(f"  Backbone       : {MODEL_NAME}")
    print(f"  Pipeline       : {'3D Video Model (B,C,T,H,W)' if IS_VIDEO_MODEL else '2D CNN (B,T,C,H,W)'}")
    print(f"  ViTTA          : {'ENABLED' if args.ViTTA else 'DISABLED'}")
    print(f"  RMGA           : {'ENABLED' if args.RMGA  else 'DISABLED'}")
    if args.RMGA:
        print(f"    window={args.rmga_window} | steps={args.rmga_steps} | "
              f"lr={args.rmga_lr} | tau={args.rmga_tau}")
        print(f"    last_bn_blocks={args.rmga_last_blocks} | "
              f"fp16={args.rmga_fp16} | extra_clips={args.rmga_extra_clips}")
    print(f"{'='*65}\n")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Step 1: Load & split data ─────────────────────────────
    print("[DATA] Collecting videos …")
    clean_paths, clean_labels, class_names = collect_videos(args.clean_dir)
    mixed_paths, mixed_labels, _           = collect_videos(args.mixed_dir)

    num_classes = len(class_names)
    print(f"[DATA] Classes       : {num_classes}")
    print(f"[DATA] Clean videos  : {len(clean_paths)}")
    print(f"[DATA] Mixed videos  : {len(mixed_paths)}")

    assert len(clean_paths) == len(mixed_paths), \
        "UCF50 and UCF50_mixed must have the same number of videos!"
    assert clean_labels == mixed_labels, \
        "Label lists must match between UCF50 and UCF50_mixed!"

    print(f"[DATA] Stratified 70-30 split (seed={args.split_seed}) …")
    train_idx, test_idx = stratified_split(
        clean_paths, clean_labels, args.test_ratio, args.split_seed
    )

    train_paths  = [clean_paths[i] for i in train_idx]
    train_labels = [clean_labels[i] for i in train_idx]
    test_paths   = [mixed_paths[i]  for i in test_idx]
    test_labels  = [mixed_labels[i] for i in test_idx]

    print(f"[DATA] Train samples : {len(train_paths)}")
    print(f"[DATA] Test  samples : {len(test_paths)}")

    from collections import Counter
    tr_dist = Counter(train_labels)
    te_dist = Counter(test_labels)
    print("\n[DATA] Per-class sample counts (train | test):")
    for cid, cname in enumerate(class_names):
        print(f"  {cname:<30} train={tr_dist[cid]:>4} | test={te_dist[cid]:>4}")

    # ── Step 2: Datasets & Loaders ────────────────────────────
    train_tf = build_transforms(args.img_size, train=True)
    test_tf  = build_transforms(args.img_size, train=False)

    # [3D SUPPORT] Pass channel_first=IS_VIDEO_MODEL so the dataset
    # returns the correct tensor layout for the active backbone.
    train_ds = VideoDataset(train_paths, train_labels, args.num_frames, train_tf,
                            channel_first=IS_VIDEO_MODEL)
    test_ds  = VideoDataset(test_paths,  test_labels,  args.num_frames, test_tf,
                            channel_first=IS_VIDEO_MODEL)

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

    # ── Step 4: Train ─────────────────────────────────────────
    best_acc = 0.0
    if args.mode == "train_eval":
        print(f"\n{'='*65}")
        print(f"  TRAINING  ({args.epochs} epochs)")
        print(f"{'='*65}")

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
                print(f"    ★ New best model saved → {args.best_model} "
                      f"(Acc={best_acc:.2f}%)")

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

    # ── Step 5: Final Evaluation ──────────────────────────────
    if args.ViTTA and args.RMGA:
        print("\n[WARN] Both --ViTTA and --RMGA are set. "
              "Running RMGA (takes priority). "
              "To run ViTTA, remove --RMGA.")
        args.ViTTA = False

    print(f"\n{'='*65}")
    method_tag = "RMGA" if args.RMGA else ("ViTTA" if args.ViTTA else "Standard")
    print(f"  EVALUATION  (method: {method_tag}  |  pipeline: "
          f"{'3D' if IS_VIDEO_MODEL else '2D'})")
    print(f"{'='*65}\n")

    # ── 5a. Standard evaluation (UNCHANGED) ──────────────────
    if not args.ViTTA and not args.RMGA:
        model.eval()
        test_loss, test_acc, class_correct, class_total = evaluate(
            model, test_loader, criterion, device
        )
        print(f"[RESULT] Test Loss : {test_loss:.4f}")
        print(f"[RESULT] Test Acc  : {test_acc:.2f}%")
        print(f"\n[RESULT] Per-class Accuracy:")
        for cid, cname in enumerate(class_names):
            ct  = class_total.get(cid, 0)
            cc  = class_correct.get(cid, 0)
            pct = (cc / ct * 100) if ct > 0 else 0.0
            print(f"  {cname:<30} {cc:>3}/{ct:>3} = {pct:5.1f}%")
        print(f"\n[RESULT] Overall Accuracy: {test_acc:.2f}%")

    # ── 5b. ViTTA evaluation ──────────────────────────────────
    elif args.ViTTA:
        print(f"[ViTTA] Running per-video adapted inference …")
        print(f"[ViTTA] n_clips={args.vitta_clips} | "
              f"adapt_steps={args.vitta_steps} | adapt_lr={args.vitta_lr}")

        # [3D SUPPORT] is_video_model passed so ViTTA builds correct clip shapes
        vitta_engine = ViTTA(
            model          = model,
            n_clips        = args.vitta_clips,
            adapt_steps    = args.vitta_steps,
            adapt_lr       = args.vitta_lr,
            device         = device,
            is_video_model = IS_VIDEO_MODEL,
        )

        correct, total  = 0, 0
        class_correct   = defaultdict(int)
        class_total     = defaultdict(int)
        entropy_list    = []

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
        print(f"\n[RESULT] ViTTA Test Accuracy     : {test_acc:.2f}%")
        print(f"[RESULT] Mean Prediction Entropy : {np.mean(entropy_list):.4f}")
        print(f"\n[RESULT] Per-class Accuracy (ViTTA):")
        for cid, cname in enumerate(class_names):
            ct  = class_total.get(cid, 0)
            cc  = class_correct.get(cid, 0)
            pct = (cc / ct * 100) if ct > 0 else 0.0
            print(f"  {cname:<30} {cc:>3}/{ct:>3} = {pct:5.1f}%")
        print(f"\n[RESULT] Overall ViTTA Accuracy: {test_acc:.2f}%")

    # ── 5c. RMGA evaluation ───────────────────────────────────
    elif args.RMGA:
        print(f"[RMGA] Running per-video Rhythmic Motion-Gated adaptation …")
        print(f"[RMGA] window={args.rmga_window} | steps={args.rmga_steps} | "
              f"lr={args.rmga_lr} | tau={args.rmga_tau}")
        print(f"[RMGA] last_bn_blocks={args.rmga_last_blocks} | "
              f"fp16={args.rmga_fp16} | extra_clips={args.rmga_extra_clips}")

        # [3D SUPPORT] is_video_model passed so RMGA builds correct clip shapes
        rmga_engine = RMGA(
            model          = model,
            window_size    = args.rmga_window,
            adapt_steps    = args.rmga_steps,
            adapt_lr       = args.rmga_lr,
            tau            = args.rmga_tau,
            last_bn_blocks = args.rmga_last_blocks,
            extra_clips    = args.rmga_extra_clips,
            device         = device,
            use_fp16       = args.rmga_fp16,
            is_video_model = IS_VIDEO_MODEL,
        )

        correct, total  = 0, 0
        class_correct   = defaultdict(int)
        class_total     = defaultdict(int)
        entropy_list    = []
        conf_list       = []

        for idx, (vpath, vlabel) in enumerate(zip(test_paths, test_labels)):
            pred, conf, ent = rmga_engine.predict(
                vpath, args.num_frames, test_tf
            )
            correct += int(pred == vlabel)
            total   += 1
            class_correct[vlabel] += int(pred == vlabel)
            class_total[vlabel]   += 1
            entropy_list.append(ent)
            conf_list.append(conf)

            if (idx + 1) % 20 == 0 or (idx + 1) == len(test_paths):
                running_acc = correct / total * 100.0
                print(f"  [RMGA] {idx+1}/{len(test_paths)} | "
                      f"Running Acc: {running_acc:.2f}% | "
                      f"AvgEnt: {np.mean(entropy_list):.4f} | "
                      f"AvgConf: {np.mean(conf_list):.4f}")

        test_acc      = correct / total * 100.0
        stability_idx = float(np.std(
            [class_correct[c] / max(class_total[c], 1) * 100.0
             for c in class_total]
        ))

        print(f"\n[RESULT] RMGA Test Accuracy           : {test_acc:.2f}%")
        print(f"[RESULT] Mean Prediction Entropy      : {np.mean(entropy_list):.4f}")
        print(f"[RESULT] Mean Confidence              : {np.mean(conf_list):.4f}")
        print(f"[RESULT] Stability Index (std of cls) : {stability_idx:.2f}%")
        print(f"\n[RESULT] Per-class Accuracy (RMGA):")
        for cid, cname in enumerate(class_names):
            ct  = class_total.get(cid, 0)
            cc  = class_correct.get(cid, 0)
            pct = (cc / ct * 100) if ct > 0 else 0.0
            print(f"  {cname:<30} {cc:>3}/{ct:>3} = {pct:5.1f}%")
        print(f"\n[RESULT] Overall RMGA Accuracy: {test_acc:.2f}%")
        print(f"\n[INFO]  Use --mode eval_only with different --mixed_dir "
              f"(e.g. Snow, Blur, Noise) to run per-corruption ablations.")

    print(f"\n{'='*65}")
    print("  DONE.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()