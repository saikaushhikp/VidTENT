#!/usr/bin/env python3
"""
RMGA evaluation on UCF50/UCF50_mixed from pre-trained weights.

Set MODEL_NAME for the backbone you want to evaluate.
"""

# Model selection
# MODEL_NAME = "mobilenet_v3_small"
FEATURE_DIM  = 576                    # channels out of 2D backbone (ignored for video models)
PRETRAINED   = True                   # use ImageNet / Kinetics pretrained weights
MODEL_NAME = "r3d_18"               # ResNet-3D-18      | input (B,C,T,H,W)
# MODEL_NAME = "mc3_18"
# MODEL_NAME = "r2plus1d_18"
# MODEL_NAME = "efficientnet_b1"      # FEATURE_DIM = 1280
# MODEL_NAME = "resnet18"             # FEATURE_DIM = 512

VIDEO_MODELS = {"r3d_18", "mc3_18", "r2plus1d_18"}
IS_VIDEO_MODEL = MODEL_NAME in VIDEO_MODELS

import os, sys, copy, time, random, argparse, warnings
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tvm
import torchvision.models.video as tvm_video
import torchvision.transforms as T
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")
CLEAN_DIR = Path("./datasets/UCF50")         # Uncorrupted videos for training
MIXED_DIR = Path("./datasets/UCF50_mixed")   # Corrupted videos for testing


# ─────────────────────────────────────────────────────────────
# 1.  ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="UCF50 RMGA evaluation from loaded weights"
    )
    # ── Paths ─────────────────────────────────────────────────
    p.add_argument("--clean_dir",    default=CLEAN_DIR, help="Clean video folder (uncorrupted)")
    p.add_argument("--mixed_dir",    default=MIXED_DIR, help="Corrupted video folder")
    p.add_argument(
        "--load_weights", type=Path, required=True,
        help="Path to model weights (.pth) used for RMGA evaluation",
    )

    # ── Data ──────────────────────────────────────────────────
    p.add_argument("--num_frames",   type=int,   default=16,   help="Frames sampled per video")
    p.add_argument("--img_size",     type=int,   default=112,  help="Spatial resolution (H=W)")
    p.add_argument("--split_seed",   type=int,   default=42,   help="Seed for 70-30 split")
    p.add_argument("--test_ratio",   type=float, default=0.30)

    # ── RMGA ──────────────────────────────────────────────────
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
    return p


# ─────────────────────────────────────────────────────────────
# 2.  DATASET HELPERS
# ─────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def collect_videos(root_dir: str):
    """Collect all videos and class ids from a dataset root."""
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
    """Return stratified train/test indices."""
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    idx = np.arange(len(labels))
    train_idx, test_idx = next(sss.split(idx, labels))
    return train_idx.tolist(), test_idx.tolist()


def sample_frames(video_path: str, num_frames: int):
    """Uniformly sample frames from one video."""
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
    """Sample multiple temporal clips from one video."""
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


def build_model(num_classes: int, model_name=None,
                feat_dim=None, pretrained=None):
    """
    Build and return the appropriate model.

    Defaults fall back to the module-level MODEL_NAME / FEATURE_DIM / PRETRAINED
    constants. Using None here (instead of the globals directly) avoids a
    NameError when Python evaluates default argument values at function-definition
    time, before the constants are guaranteed to be in scope.

    [3D SUPPORT] If model_name is in VIDEO_MODELS, loads from
    torchvision.models.video and wraps in VideoModel3D.
    Otherwise, existing 2D FrameAggregator pipeline is used unchanged.
    """
    # Resolve defaults from module-level constants
    if model_name is None:
        model_name = MODEL_NAME
    if feat_dim is None:
        feat_dim = FEATURE_DIM
    if pretrained is None:
        pretrained = PRETRAINED

    print(f"\n[MODEL] Loading backbone : {model_name} (pretrained={pretrained})")
    print(f"[MODEL] Pipeline         : {'3D Video Model' if model_name in VIDEO_MODELS else '2D CNN + FrameAggregator'}")

    if model_name in VIDEO_MODELS:
        weights_arg = "DEFAULT" if pretrained else None

        _video_constructors = {
            "r3d_18"      : tvm_video.r3d_18,
            "mc3_18"      : tvm_video.mc3_18,
            "r2plus1d_18" : tvm_video.r2plus1d_18,
        }
        constructor = _video_constructors[model_name]
        net         = constructor(weights=weights_arg)

        in_features = net.fc.in_features
        net.fc      = nn.Linear(in_features, num_classes)
        print(f"[MODEL] FC head replaced: {in_features} → {num_classes}")

        total_params = sum(p.numel() for p in net.parameters())
        trainable    = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"[MODEL] Total params: {total_params:,}  |  Trainable: {trainable:,}")
        return net

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
# 5.  RMGA
# ─────────────────────────────────────────────────────────────
class RMGA:
    """Rhythmic Motion-Gated Adaptation for test-time video inference."""

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
        self.is_video_model = is_video_model

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
            print("[RMGA] ⚠  No BN found — falling back to head adaptation.")
            head_prefixes = ("classifier", "fc")
            for mod_name, module in adapted.named_modules():
                if any(mod_name == p or mod_name.startswith(p + ".")
                       for p in head_prefixes):
                    for param_name, _ in module.named_parameters(recurse=False):
                        full_name = (f"{mod_name}.{param_name}"
                                     if mod_name else param_name)
                        selected.add(full_name)

        for name, param in adapted.named_parameters():
            param.requires_grad_(name in selected)

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
        M_t = 𝟙( mean_C |x_t - x_{t-1}| > τ )
        Returns float32 (1, H, W) mask.
        """
        diff = torch.abs(frame_t - frame_prev)
        return (diff.mean(dim=0, keepdim=True) > tau).float()

    def _warmup_bn(self, adapted_model: nn.Module, clip_tensors: list):
        """Update BN running stats using test-time clips."""
        with torch.no_grad():
            for clip in clip_tensors:
                adapted_model(clip.unsqueeze(0).to(self.device))

    @staticmethod
    def _freeze_bn_running_stats(model: nn.Module):
        """Freeze BN running stats while keeping affine params trainable."""
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def _rhythmic_adapt(self, adapted_model: nn.Module, clip_tensors: list):
        """Run motion-gated adaptation over clip windows."""
        trainable_params = [p for p in adapted_model.parameters()
                            if p.requires_grad]
        if not trainable_params:
            print("[RMGA] ⚠  No trainable params — skipping adaptation.")
            return

        opt    = optim.Adam(trainable_params, lr=self.adapt_lr)
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)

        for _step in range(self.adapt_steps):
            for clip in clip_tensors:
                if self.is_video_model:
                    clip_cthw = clip.to(self.device)           # (C, T, H, W)
                    clip_tchw = clip_cthw.permute(1, 0, 2, 3)  # (T, C, H, W)
                else:
                    clip_tchw = clip.to(self.device)           # (T, C, H, W)

                T_len, C, H, W = clip_tchw.shape

                # ── 1. Build motion masks (T, 1, H, W) ────────────────────────
                masks = [torch.ones(1, H, W, device=self.device)]  # t=0: all active
                for t in range(1, T_len):
                    masks.append(self._compute_motion_mask(
                        clip_tchw[t], clip_tchw[t - 1], self.tau
                    ))
                masks = torch.stack(masks, dim=0)   # (T, 1, H, W)

                W_size    = min(self.window_size, T_len)
                n_windows = max(1, (T_len + W_size - 1) // W_size)

                if self.is_video_model:
                    win_entropies     = []
                    win_motion_weights = []

                    with torch.no_grad():
                        for w in range(n_windows):
                            start   = w * W_size
                            end     = min(start + W_size, T_len)
                            win_clip = clip_cthw[:, start:end, :, :]
                            win_in   = win_clip.unsqueeze(0)         # (1, C, win_len, H, W)
                            logits   = adapted_model(win_in)
                            win_entropies.append(self._entropy(logits).item())
                            win_motion_weights.append(
                                masks[start:end].mean().item()
                            )

                    # ── 4a. Peak anchoring: argmin entropy over valid windows ──
                    # A "valid" window has meaningful motion (not pure background)
                    valid = [(i, e) for i, (e, mw) in
                             enumerate(zip(win_entropies, win_motion_weights))
                             if mw >= 1e-4]
                    if not valid:
                        continue

                    peak_w      = min(valid, key=lambda x: x[1])[0]
                    peak_motion = win_motion_weights[peak_w]
                    p_start     = peak_w * W_size
                    p_end       = min(p_start + W_size, T_len)

                    peak_win_cthw = clip_cthw[:, p_start:p_end, :, :]
                    spatial_mask  = masks[p_start:p_end].mean(dim=0)
                    masked_win    = peak_win_cthw * spatial_mask.unsqueeze(0)
                    peak_input    = masked_win.unsqueeze(0)

                    opt.zero_grad()
                    with torch.cuda.amp.autocast(enabled=self.use_fp16):
                        logits = adapted_model(peak_input)
                        loss   = self._entropy(logits) * peak_motion

                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                else:
                    clip_dev = clip_tchw

                    for w in range(n_windows):
                        start   = w * W_size
                        end     = min(start + W_size, T_len)
                        win_f   = clip_dev[start:end]
                        win_m   = masks[start:end]
                        win_len = end - start

                        with torch.no_grad():
                            batch_in     = win_f.unsqueeze(1)
                            batch_logits = adapted_model(batch_in)
                            frame_entropies = [
                                self._entropy(batch_logits[t:t+1]).item()
                                for t in range(win_len)
                            ]

                        peak_t        = int(np.argmin(frame_entropies))
                        motion_weight = win_m[peak_t].mean().item()

                        if motion_weight < 1e-4:
                            continue

                        peak_frame   = win_f[peak_t]
                        masked_frame = peak_frame * win_m[peak_t]
                        peak_input   = masked_frame.unsqueeze(0).unsqueeze(0)

                        opt.zero_grad()
                        with torch.cuda.amp.autocast(enabled=self.use_fp16):
                            logits = adapted_model(peak_input)
                            loss   = self._entropy(logits) * motion_weight

                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()

    def predict(self, video_path: str, num_frames: int,
                transform, label: int = -1):
        """Run RMGA inference for a single video."""

        def frames_to_tensor(frames):
            """Convert BGR frames to model input tensor."""
            tensors = [transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                       for f in frames]
            clip = torch.stack(tensors, dim=0)
            if self.is_video_model:
                clip = clip.permute(1, 0, 2, 3).contiguous()
            return clip

        primary_frames = sample_frames(video_path, num_frames)
        if not primary_frames:
            return 0, 0.0, 0.0

        primary_clip     = frames_to_tensor(primary_frames)
        all_clip_tensors = [primary_clip]

        if self.extra_clips > 0:
            for ef in temporal_clips(video_path, num_frames, self.extra_clips):
                if ef:
                    all_clip_tensors.append(frames_to_tensor(ef))

        adapted, _ = self._copy_model_to_adapt(
            self.original_model, self.last_bn_blocks
        )
        adapted.to(self.device)

        self._warmup_bn(adapted, all_clip_tensors)
        self._freeze_bn_running_stats(adapted)
        self._rhythmic_adapt(adapted, all_clip_tensors)

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
# 6.  CHECKPOINT LOADER
# ─────────────────────────────────────────────────────────────
def smart_load_state_dict(model: nn.Module, path, device):
    """Load checkpoint, with key remapping for older wrapper formats."""
    raw = torch.load(path, map_location=device)

    if isinstance(raw, dict) and "model" in raw and "epoch" in raw:
        print(f"[CKPT] Detected save_checkpoint() format — extracting 'model' key.")
        state_dict = raw["model"]
    else:
        state_dict = raw

    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[CKPT] ✅ Loaded (strict)  ← {path}")
        return
    except RuntimeError:
        pass

    remapped = {"backbone." + k: v for k, v in state_dict.items()}
    try:
        model.load_state_dict(remapped, strict=True)
        print(f"[CKPT] ✅ Loaded (added 'backbone.' prefix)  ← {path}")
        return
    except RuntimeError:
        pass

    stripped = {}
    for k, v in state_dict.items():
        new_key = k[len("backbone."):] if k.startswith("backbone.") else k
        stripped[new_key] = v
    try:
        model.load_state_dict(stripped, strict=True)
        print(f"[CKPT] ✅ Loaded (stripped 'backbone.' prefix)  ← {path}")
        return
    except RuntimeError:
        pass

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[CKPT] ⚠  Loaded with strict=False  ← {path}")
    if missing:
        print(f"[CKPT]    Missing  ({len(missing)}): {missing[:5]}"
              f"{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[CKPT]    Unexpected ({len(unexpected)}): {unexpected[:5]}"
              f"{'…' if len(unexpected) > 5 else ''}")


# ─────────────────────────────────────────────────────────────
# 10.  MAIN
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
    print(f"  Backbone       : {MODEL_NAME}")
    print(f"  Pipeline       : {'3D Video Model (B,C,T,H,W)' if IS_VIDEO_MODEL else '2D CNN (B,T,C,H,W)'}")
    print("  Mode           : RMGA evaluation")
    print(f"  Weights        : {args.load_weights}")
    print(f"  RMGA           : window={args.rmga_window} | steps={args.rmga_steps} | "
          f"lr={args.rmga_lr} | tau={args.rmga_tau}")
    print(f"                   last_bn_blocks={args.rmga_last_blocks} | "
          f"fp16={args.rmga_fp16} | extra_clips={args.rmga_extra_clips}")
    print(f"{'='*65}\n")

    if not args.load_weights.is_file():
        raise FileNotFoundError(f"Weights file not found: {args.load_weights}")

    # ── Step 1: Load data and build test split ───────────────
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
    _, test_idx = stratified_split(
        clean_paths, clean_labels, args.test_ratio, args.split_seed
    )

    test_paths   = [mixed_paths[i]  for i in test_idx]
    test_labels  = [mixed_labels[i] for i in test_idx]

    print(f"[DATA] Test  samples : {len(test_paths)}")

    from collections import Counter
    te_dist = Counter(test_labels)
    print("\n[DATA] Per-class sample counts (test):")
    for cid, cname in enumerate(class_names):
        print(f"  {cname:<30} test={te_dist[cid]:>4}")

    # ── Step 2: Build transform and model ────────────────────
    test_tf  = build_transforms(args.img_size, train=False)
    model = build_model(num_classes).to(device)
    smart_load_state_dict(model, args.load_weights, device)
    print(f"[EVAL] Loaded weights from {args.load_weights}")

    print(f"\n{'='*65}")
    print(f"  EVALUATION  (method: RMGA  |  pipeline: "
          f"{'3D' if IS_VIDEO_MODEL else '2D'})")
    print(f"{'='*65}\n")

    print("[RMGA] Running per-video Rhythmic Motion-Gated adaptation …")

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
        pred, conf, ent = rmga_engine.predict(vpath, args.num_frames, test_tf)
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
    print("\n[INFO]  To run corruption-specific evaluation, point --mixed_dir to that dataset root.")

    print(f"\n{'='*65}")
    print("  DONE.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()