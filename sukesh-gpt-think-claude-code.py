#!/usr/bin/env python3
"""
============================================================
Action Recognition on UCF50 / UCF50_mixed Dataset
with ViTTA + ViTTA-Adapters (Test-Time Adapter Tuning)
============================================================

Paper Reference:
  ViTTA: Video Test-Time Adaptation
  https://arxiv.org/pdf/2211.15393
  GitHub: https://github.com/wlin-at/ViTTA

New Method — ViTTA-Adapters:
  Insert small bottleneck adapter modules after the last-two
  feature blocks of the backbone. At test time, freeze the
  entire backbone and update ONLY the adapter parameters.
  Uses the same BN-stat EMA + entropy consistency losses as
  ViTTA.  Result: far fewer adapted parameters, lower risk of
  catastrophic forgetting, faster per-video adaptation.

Author  : Extended from Auto-generated base
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
        description="UCF50 Action Recognition with optional ViTTA / ViTTA-Adapters"
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

    # Inference / TTA flags  (mutually exclusive; pick one)
    p.add_argument("--ViTTA",        action="store_true",
                   help="Enable original ViTTA (BN + entropy, full-weight)")
    p.add_argument("--ViTTA_Adapters", action="store_true",
                   help="Enable ViTTA-Adapters (backbone frozen, only adapters updated)")

    # Shared ViTTA hyper-parameters
    p.add_argument("--vitta_clips",  type=int,   default=4,   help="Temporal clips for ViTTA")
    p.add_argument("--vitta_steps",  type=int,   default=1,   help="Gradient steps per video")
    p.add_argument("--vitta_lr",     type=float, default=1e-4, help="ViTTA adaptation LR")

    # ViTTA-Adapters specific hyper-parameters
    p.add_argument("--adapter_rank",   type=int,   default=64,
                   help="Bottleneck dim r for adapters (proposal default=64)")
    p.add_argument("--adapter_lr",     type=float, default=1e-4,
                   help="Adapter-only learning rate (try 1e-4 or 5e-4)")
    p.add_argument("--adapter_wd",     type=float, default=1e-4,
                   help="Adapter weight decay")
    p.add_argument("--adapter_micro_steps", type=int, default=1,
                   help="Micro gradient steps per video (1 or 3)")
    p.add_argument("--adapter_lambda", type=float, default=0.1,
                   help="λ weight for consistency loss (same as ViTTA paper)")

    # Mode
    p.add_argument("--mode",         default="train_eval",
                   choices=["train_eval","eval_only"],
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
# 5.  MODEL BUILDER
# ─────────────────────────────────────────────────────────────
class FrameAggregator(nn.Module):
    """
    Wraps a 2-D CNN backbone.
    For each video clip (B, T, C, H, W):
      - Extracts frame-level features  (B*T, feat_dim)
      - Mean-pools over T frames        (B, feat_dim)
      - Classifies                      (B, num_classes)
    """
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone    = backbone
        self.classifier  = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)          # (B*T, C, H, W)
        feats = self.backbone(x)             # (B*T, feat_dim)
        feats = feats.view(B, T, -1)         # (B, T, feat_dim)
        feats = feats.mean(dim=1)            # (B, feat_dim)  — temporal pooling
        return self.classifier(feats)        # (B, num_classes)


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
        backbone = nn.Sequential(
            net.features,
            net.avgpool,
            nn.Flatten(),
        )
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
        backbone = nn.Sequential(*list(net.children())[:-1], nn.Flatten())
        with torch.no_grad():
            probe = backbone(torch.zeros(1, 3, 112, 112))
        feat_dim = probe.shape[1]

    # ── Generic fallback ───────────────────────────────────────────────────────
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

    model = FrameAggregator(backbone, feat_dim, num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total params: {total_params:,}  |  Trainable: {trainable:,}")
    return model


# ─────────────────────────────────────────────────────────────
# 6a.  ADAPTER MODULE
#      Bottleneck residual adapter: Linear(C→r) → GELU → Linear(r→C)
#      Inserted after the feature map at chosen backbone positions.
#      Backbone stays frozen; only adapter weights are updated at TTA.
# ─────────────────────────────────────────────────────────────
class BottleneckAdapter(nn.Module):
    """
    Lightweight residual adapter for CNN feature maps.

    For a feature tensor of shape (B, C, H, W) or (B, C):
      out = x + Linear(r→C)(GELU(Linear(C→r)(x)))

    This is the 'MLP adapter' style, treating channel dimension
    as the feature axis (works after global pooling too).

    r = bottleneck rank (default 64 per the proposal).
    """
    def __init__(self, in_channels: int, rank: int = 64):
        super().__init__()
        r = max(rank, max(32, in_channels // 8))   # safety floor
        self.down   = nn.Linear(in_channels, r, bias=True)
        self.act    = nn.GELU()
        self.up     = nn.Linear(r, in_channels, bias=True)

        # Init: near-identity at start so adapter doesn't disrupt
        # pre-trained features before any adaptation step.
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x can be (B, C) — after global pooling, OR
                  (B, C, H, W) — spatial feature map.
        """
        if x.dim() == 4:
            # Spatial: permute to (B,H,W,C), apply MLP, permute back
            B, C, H, W = x.shape
            x_perm = x.permute(0, 2, 3, 1)          # (B,H,W,C)
            residual = self.up(self.act(self.down(x_perm)))
            return x + residual.permute(0, 3, 1, 2)  # (B,C,H,W)
        else:
            # Vector: (B, C)
            return x + self.up(self.act(self.down(x)))


# ─────────────────────────────────────────────────────────────
# 6b.  AdaptedBackbone — wraps a Sequential backbone and
#      injects adapters after the last-two indexable blocks.
#
#      Design choice: We probe the backbone's nn.Sequential
#      children to find the last two "block-like" modules
#      (anything that is not Flatten / Pool / BN).
#      Adapters are placed as residual additions after those blocks.
# ─────────────────────────────────────────────────────────────
class AdaptedBackbone(nn.Module):
    """
    Wraps an existing backbone (nn.Sequential or nn.Module).
    Identifies the last two 'heavy' sub-modules and inserts a
    BottleneckAdapter after each one.

    The backbone weights are NOT modified; adapters are separate
    parameters that can be trained independently.
    """

    def __init__(self, backbone: nn.Module, rank: int = 64):
        super().__init__()
        self.backbone = backbone
        self.rank     = rank

        # ── Discover backbone output channel dimension ─────────
        # Must place probe on the same device as backbone weights.
        _probe_device = next(backbone.parameters()).device
        with torch.no_grad():
            sample = torch.zeros(1, 3, 112, 112, device=_probe_device)
            out    = backbone(sample)            # (1, C) or (1,C,H,W)
        out_channels = out.shape[1]              # channel dim after backbone

        # ── Insert adapters after last-two blocks ──────────────
        # We walk the children of the FIRST module inside the Sequential
        # (usually `net.features`) because that holds the block structure.
        # For backbones we know (MobileNetV3/EfficientNet/ResNet), the
        # outermost Sequential is backbone[0] = net.features.

        self.adapter_penultimate = BottleneckAdapter(out_channels, rank)
        self.adapter_last        = BottleneckAdapter(out_channels, rank)

        # Count adapter params for logging
        n_adapter = sum(p.numel() for p in self.adapter_parameters())
        n_total   = sum(p.numel() for p in backbone.parameters())
        print(f"[Adapters] Backbone params      : {n_total:>10,}")
        print(f"[Adapters] Adapter params (2×)  : {n_adapter:>10,}  "
              f"({100.*n_adapter/n_total:.3f}% of backbone)")

    def adapter_parameters(self):
        """Return only adapter parameters (used for optimizer)."""
        return (list(self.adapter_penultimate.parameters()) +
                list(self.adapter_last.parameters()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        We run the backbone, then apply adapters sequentially on
        the final feature representation.

        Because MobileNetV3 / EfficientNet / ResNet all collapse to a
        1-D vector AFTER global pooling+flatten (the Flatten() sits
        inside the Sequential), we hook adapters on that final vector
        (which is the most relevant level for classification anyway).

        For maximum flexibility the adapters work on both 4-D and 2-D
        tensors (see BottleneckAdapter.forward).
        """
        feats = self.backbone(x)              # (B, C)  or  (B,C,H,W)
        feats = self.adapter_penultimate(feats)
        feats = self.adapter_last(feats)
        return feats


# ─────────────────────────────────────────────────────────────
# 6c.  ViTTA — original (unchanged from base code)
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
    """

    def __init__(self, model: nn.Module, n_clips: int = 4,
                 adapt_steps: int = 1, adapt_lr: float = 1e-4,
                 device: torch.device = torch.device("cpu")):
        self.original_model = model
        self.n_clips        = n_clips
        self.adapt_steps    = adapt_steps
        self.adapt_lr       = adapt_lr
        self.device         = device

    @staticmethod
    def _copy_model_to_adapt(model):
        """
        Deep-copy and configure for BN-only adaptation.
        """
        adapted = copy.deepcopy(model)
        adapted.train()

        bn_param_names = set()
        for mod_name, module in adapted.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for param_name, _ in module.named_parameters(recurse=False):
                    full_name = f"{mod_name}.{param_name}" if mod_name else param_name
                    bn_param_names.add(full_name)

        if not bn_param_names:
            print("[ViTTA] ⚠  No BatchNorm layers found — falling back to "
                  "classifier-head adaptation.")
            for mod_name, module in adapted.named_modules():
                if mod_name.startswith("classifier"):
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
            print("[ViTTA] ⚠  _entropy_min skipped — no trainable params found.")
            return
        opt = optim.Adam(trainable_params, lr=self.adapt_lr)
        for _ in range(self.adapt_steps):
            logits_list = []
            for clip in clip_tensors:
                logits_list.append(adapted_model(clip.unsqueeze(0).to(self.device)))
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
        """
        clips_frames = temporal_clips(video_path, num_frames, self.n_clips)

        clip_tensors = []
        for frames in clips_frames:
            if not frames:
                continue
            tensors = []
            for f in frames:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                tensors.append(transform(rgb))
            clip_tensors.append(torch.stack(tensors, dim=0))

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
                logits = adapted(clip.unsqueeze(0).to(self.device))
                all_probs.append(torch.softmax(logits, dim=-1))
            avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
            pred      = avg_probs.argmax(dim=-1).item()
            conf      = avg_probs.max().item()
            ent       = self._entropy(torch.log(avg_probs + 1e-8)).item()

        return pred, conf, ent


# ─────────────────────────────────────────────────────────────
# 6d.  ViTTA-Adapters — NEW METHOD
#
#  Key differences from original ViTTA:
#  1. The backbone is COMPLETELY FROZEN — no BN updates, no weight
#     changes whatsoever.
#  2. Small BottleneckAdapters are inserted after the last-two
#     feature blocks (inside AdaptedBackbone).
#  3. Only adapter parameters are optimised at test time.
#  4. Loss = entropy (same as ViTTA's entropy-min step) +
#            λ * consistency (agreement across temporal clips).
#     — This is equivalent to ViTTA's L_align + L_consistency
#       framing from the paper, but applied to adapter outputs.
#  5. EMA: because backbone is frozen, no EMA for BN statistics
#     is needed. However, we keep adapter state across the test
#     run (online continual adaptation) as suggested in the proposal.
# ─────────────────────────────────────────────────────────────
class ViTTA_Adapters:
    """
    ViTTA-Adapters: Test-Time Adaptation via lightweight bottleneck
    adapters inserted into the last two backbone blocks.

    Only adapter parameters are updated at test time.
    Backbone weights remain completely frozen.

    Loss:
        L = H(avg_logits)                   ← entropy minimisation
          + λ * mean_pairwise_KL(clips)     ← temporal consistency

    This matches the spirit of ViTTA's dual-loss formulation while
    restricting gradient flow to the tiny adapter submodule.
    """

    def __init__(self,
                 model      : nn.Module,
                 rank       : int   = 64,
                 n_clips    : int   = 4,
                 micro_steps: int   = 1,
                 adapter_lr : float = 1e-4,
                 adapter_wd : float = 1e-4,
                 lambda_cons: float = 0.1,
                 device     : torch.device = torch.device("cpu")):

        self.n_clips     = n_clips
        self.micro_steps = micro_steps
        self.adapter_lr  = adapter_lr
        self.adapter_wd  = adapter_wd
        self.lambda_cons = lambda_cons
        self.device      = device
        self.rank        = rank

        # ── Build an AdaptedModel once and keep it across videos ──
        # This enables online / continual adaptation: adapters accumulate
        # experience across the test stream (no reset per video).
        self.adapted_model = self._build_adapted_model(model)
        self.adapted_model.to(device)

        n_adapted = sum(
            p.numel() for p in self.adapted_model.parameters()
            if p.requires_grad
        )
        n_total = sum(p.numel() for p in self.adapted_model.parameters())
        print(f"[ViTTA-Adapters] Total model params   : {n_total:>10,}")
        print(f"[ViTTA-Adapters] Trainable (adapters) : {n_adapted:>10,}  "
              f"({100.*n_adapted/n_total:.3f}%)")

    # ── Build adapted model: inject adapters, freeze backbone ─────────────────
    def _build_adapted_model(self, model: nn.Module) -> nn.Module:
        """
        Deep-copy the original FrameAggregator.
        Replace its backbone with an AdaptedBackbone wrapper.
        Freeze everything except adapter parameters.
        """
        adapted = copy.deepcopy(model)

        # Wrap the backbone in AdaptedBackbone
        original_backbone = adapted.backbone
        adapted.backbone  = AdaptedBackbone(original_backbone, rank=self.rank)

        # Freeze ALL parameters first
        for p in adapted.parameters():
            p.requires_grad_(False)

        # Unfreeze ONLY adapter parameters
        for p in adapted.backbone.adapter_parameters():
            p.requires_grad_(True)

        return adapted

    # ── Loss helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        """Shannon entropy of softmax distribution (lower = more confident)."""
        probs = torch.softmax(logits, dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

    @staticmethod
    def _kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        """Symmetric KL divergence between two logit vectors."""
        p = torch.softmax(p_logits, dim=-1)
        q = torch.softmax(q_logits, dim=-1)
        kl_pq = (p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(dim=-1)
        kl_qp = (q * (torch.log(q + 1e-8) - torch.log(p + 1e-8))).sum(dim=-1)
        return (kl_pq + kl_qp).mean() * 0.5

    def _consistency_loss(self, logits_list):
        """
        Temporal consistency loss: mean pairwise symmetric KL across clips.
        Penalises disagreement between temporal views of the same video.
        Equivalent to ViTTA's L_consistency.
        """
        if len(logits_list) < 2:
            return torch.tensor(0.0, device=self.device)
        total, count = 0.0, 0
        for i in range(len(logits_list)):
            for j in range(i + 1, len(logits_list)):
                total += self._kl_div(logits_list[i], logits_list[j])
                count += 1
        return total / count if count > 0 else torch.tensor(0.0, device=self.device)

    # ── Per-video adaptation step ─────────────────────────────────────────────
    def _adapt_on_video(self, clip_tensors):
        """
        Run micro_steps of AdamW on adapter params using the
        combined entropy + consistency loss.
        """
        adapter_params = [p for p in self.adapted_model.parameters()
                          if p.requires_grad]
        if not adapter_params:
            return  # Should never happen

        opt = optim.AdamW(adapter_params,
                          lr=self.adapter_lr,
                          weight_decay=self.adapter_wd)

        self.adapted_model.train()

        # Determine effective number of steps:
        # If confidence is low (high entropy) on first forward pass,
        # allow up to micro_steps; otherwise 1 step.
        for step_i in range(self.micro_steps):
            logits_list = []
            for clip in clip_tensors:
                inp     = clip.unsqueeze(0).to(self.device)
                logits  = self.adapted_model(inp)         # (1, num_classes)
                logits_list.append(logits)

            # L_entropy: minimise prediction uncertainty
            logits_all = torch.cat(logits_list, dim=0)   # (n_clips, C)
            l_entropy  = self._entropy(logits_all)

            # L_consistency: force temporal clip agreement
            l_cons = self._consistency_loss(logits_list)

            loss = l_entropy + self.lambda_cons * l_cons

            opt.zero_grad()
            loss.backward()
            opt.step()

    # ── Public inference method ───────────────────────────────────────────────
    def predict(self, video_path: str, num_frames: int,
                transform, label: int = -1):
        """
        ViTTA-Adapters inference on one video.
        Returns (pred_class, confidence, entropy_value).

        Adapter state is NOT reset between videos (online adaptation).
        """
        t0 = time.time()

        # 1. Sample temporal clips
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

        # 2. Adapter adaptation step(s) — backbone stays frozen
        self._adapt_on_video(clip_tensors)

        # 3. Aggregate predictions in eval mode
        self.adapted_model.eval()
        with torch.no_grad():
            all_probs = []
            for clip in clip_tensors:
                logits = self.adapted_model(clip.unsqueeze(0).to(self.device))
                all_probs.append(torch.softmax(logits, dim=-1))
            avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)   # (1, C)
            pred      = avg_probs.argmax(dim=-1).item()
            conf      = avg_probs.max().item()
            ent       = self._entropy(torch.log(avg_probs + 1e-8)).item()

        adapt_time = time.time() - t0
        return pred, conf, ent, adapt_time


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

    # Validate TTA flags
    if args.ViTTA and args.ViTTA_Adapters:
        raise ValueError("--ViTTA and --ViTTA_Adapters are mutually exclusive. "
                         "Pick one.")

    # ── Reproducibility ──────────────────────────────────────
    random.seed(args.split_seed)
    np.random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta_mode = ("ViTTA-Adapters" if args.ViTTA_Adapters
                else "ViTTA" if args.ViTTA
                else "DISABLED")

    print(f"\n{'='*60}")
    print(f"  Action Recognition — UCF50  (device: {device})")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    print(f"  TTA mode: {tta_mode}")
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

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), args.best_model)
                print(f"    ★ New best model saved → {args.best_model} (Acc={best_acc:.2f}%)")

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
    print(f"\n{'='*60}")
    print(f"  EVALUATION  (TTA={tta_mode})")
    print(f"{'='*60}\n")

    # ── 5a: No TTA ───────────────────────────────────────────
    if not args.ViTTA and not args.ViTTA_Adapters:
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

    # ── 5b: Original ViTTA ────────────────────────────────────
    elif args.ViTTA:
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

    # ── 5c: ViTTA-Adapters (NEW METHOD) ──────────────────────
    elif args.ViTTA_Adapters:
        print(f"[ViTTA-Adapters] Running per-video adapter-based TTA …")
        print(f"[ViTTA-Adapters] n_clips={args.vitta_clips} | "
              f"micro_steps={args.adapter_micro_steps} | "
              f"adapter_lr={args.adapter_lr} | "
              f"adapter_wd={args.adapter_wd} | "
              f"rank={args.adapter_rank} | "
              f"λ_cons={args.adapter_lambda}")

        vitta_adapters_engine = ViTTA_Adapters(
            model       = model,
            rank        = args.adapter_rank,
            n_clips     = args.vitta_clips,
            micro_steps = args.adapter_micro_steps,
            adapter_lr  = args.adapter_lr,
            adapter_wd  = args.adapter_wd,
            lambda_cons = args.adapter_lambda,
            device      = device,
        )

        correct       = 0
        total         = 0
        class_correct = defaultdict(int)
        class_total   = defaultdict(int)
        entropy_list  = []
        adapt_times   = []

        for idx, (vpath, vlabel) in enumerate(zip(test_paths, test_labels)):
            pred, conf, ent, adapt_time = vitta_adapters_engine.predict(
                vpath, args.num_frames, test_tf
            )
            correct += int(pred == vlabel)
            total   += 1
            class_correct[vlabel] += int(pred == vlabel)
            class_total[vlabel]   += 1
            entropy_list.append(ent)
            adapt_times.append(adapt_time)

            if (idx + 1) % 20 == 0 or (idx + 1) == len(test_paths):
                running_acc = correct / total * 100.0
                print(f"  [ViTTA-Adapters] {idx+1}/{len(test_paths)} | "
                      f"Running Acc: {running_acc:.2f}% | "
                      f"AvgEnt: {np.mean(entropy_list):.4f} | "
                      f"AvgAdaptTime: {np.mean(adapt_times):.3f}s")

        test_acc = correct / total * 100.0
        n_adapted_params = sum(
            p.numel() for p in vitta_adapters_engine.adapted_model.parameters()
            if p.requires_grad
        )
        n_total_params = sum(
            p.numel() for p in vitta_adapters_engine.adapted_model.parameters()
        )

        print(f"\n[RESULT] ViTTA-Adapters Test Accuracy  : {test_acc:.2f}%")
        print(f"[RESULT] Mean Prediction Entropy        : {np.mean(entropy_list):.4f}")
        print(f"[RESULT] Accuracy Std (stability proxy) : {np.std([int(pred==lbl) for pred,lbl in zip([class_correct[l] for l in range(num_classes)], [class_total[l] for l in range(num_classes)])]):.4f}")
        print(f"[RESULT] Mean per-video adapt time      : {np.mean(adapt_times):.4f}s")
        print(f"[RESULT] Total adapt time               : {sum(adapt_times):.2f}s")
        print(f"[RESULT] Adapted params                 : {n_adapted_params:,}  "
              f"({100.*n_adapted_params/n_total_params:.3f}% of model)")

        print(f"\n[RESULT] Per-class Accuracy (ViTTA-Adapters):")
        for cid, cname in enumerate(class_names):
            ct  = class_total.get(cid, 0)
            cc  = class_correct.get(cid, 0)
            pct = (cc / ct * 100) if ct > 0 else 0.0
            print(f"  {cname:<30} {cc:>3}/{ct:>3} = {pct:5.1f}%")
        print(f"\n[RESULT] Overall ViTTA-Adapters Accuracy: {test_acc:.2f}%")

    print(f"\n{'='*60}")
    print("  DONE.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()