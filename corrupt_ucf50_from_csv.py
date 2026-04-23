#!/usr/bin/env python3
"""
corrupt_ucf50_from_csv.py
=========================
Recreates the UCF50_mixed corrupted dataset using per-video corruption
assignments stored in a CSV file (as produced by corrupt_ucf50.py --mixed).

Expected CSV format (header row required):
    video_path,corruption_type
    BaseballPitch/v_BaseballPitch_g01_c01.avi,gauss
    ...

For each row the corresponding source video is located under --src, the
specified corruption is applied per-frame with probability --prob, and the
result is written to:

    <dst-root>/UCF50_mixed/<video_path>

Usage
-----
  python corrupt_ucf50_from_csv.py --path datasets/UCF50_mixed_labels.csv

  # Custom options
  python corrupt_ucf50_from_csv.py --path datasets/UCF50_mixed_labels.csv --prob 0.70 --severity 4 --workers 4 --seed 42

Dependencies
------------
  pip install opencv-python numpy tqdm
  (ffmpeg with libx265 is optional but recommended for h265_abr)
"""

import argparse
import csv            
import os
import random
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# -- Optional tqdm --------------------------------------------------------------
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    print('[INFO] tqdm not installed; install with  pip install tqdm  for progress bars.')

    class tqdm:  # type: ignore[no-redef]
        """Minimal no-op replacement when tqdm is unavailable."""
        def __init__(self, iterable=None, **_kwargs):
            self._it = iterable
        def __iter__(self):
            return iter(self._it)
        def write(self, s):
            print(s)


# -- Constants ------------------------------------------------------------------

CORRUPTION_TYPES: List[str] = [
    'gauss', 'pepper', 'salt', 'shot',
    'zoom', 'impulse', 'defocus', 'motion',
    'jpeg', 'contrast', 'rain', 'h265_abr',
]

_SCRIPT_DIR = Path(__file__).resolve().parent
_UCF50_DEFAULT = _SCRIPT_DIR / 'datasets' / 'UCF50'


# -- Frame-level corruption functions (BGR uint8 in / BGR uint8 out) ------------

def _gauss(frame: np.ndarray, sev: int) -> np.ndarray:
    """Additive white Gaussian noise."""
    sigma = (8, 12, 18, 26, 38)[sev - 1]
    noise = np.random.randn(*frame.shape).astype(np.float32) * sigma
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _pepper(frame: np.ndarray, sev: int) -> np.ndarray:
    """Random black pixels (pepper noise)."""
    p = (0.02, 0.04, 0.06, 0.08, 0.10)[sev - 1]
    out = frame.copy()
    out[np.random.random(frame.shape[:2]) < p] = 0
    return out


def _salt(frame: np.ndarray, sev: int) -> np.ndarray:
    """Random white pixels (salt noise)."""
    p = (0.02, 0.04, 0.06, 0.08, 0.10)[sev - 1]
    out = frame.copy()
    out[np.random.random(frame.shape[:2]) < p] = 255
    return out


def _shot(frame: np.ndarray, sev: int) -> np.ndarray:
    """Poisson (photon shot) noise."""
    scale = (60, 25, 12, 5, 3)[sev - 1]
    lam = np.maximum(frame.astype(np.float32) / scale, 1e-7)
    return np.clip(np.random.poisson(lam) * scale, 0, 255).astype(np.uint8)


def _impulse(frame: np.ndarray, sev: int) -> np.ndarray:
    """Salt-and-pepper (impulse) noise."""
    p = (0.01, 0.02, 0.05, 0.08, 0.14)[sev - 1]
    out = frame.copy()
    rng = np.random.random(frame.shape[:2])
    out[rng < p / 2] = 0          # pepper
    out[rng > 1.0 - p / 2] = 255  # salt
    return out


def _defocus(frame: np.ndarray, sev: int) -> np.ndarray:
    """Defocus blur (Gaussian disc approximation)."""
    r = (3, 4, 6, 8, 10)[sev - 1]
    return cv2.GaussianBlur(frame, (2 * r + 1, 2 * r + 1), r)


def _motion(frame: np.ndarray, sev: int) -> np.ndarray:
    """Horizontal motion blur."""
    k = (8, 12, 16, 20, 24)[sev - 1]
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(frame, -1, kernel)


def _zoom(frame: np.ndarray, sev: int) -> np.ndarray:
    """Zoom (radial) blur — average of multiple zoom levels."""
    steps   = (3,    4,    5,    6,    7   )[sev - 1]
    max_s   = (0.02, 0.04, 0.06, 0.08, 0.10)[sev - 1]
    h, w    = frame.shape[:2]
    acc     = np.zeros_like(frame, dtype=np.float32)
    for s in np.linspace(1.0, 1.0 + max_s, steps):
        nh, nw = int(h * s), int(w * s)
        zoomed = cv2.resize(frame, (nw, nh))
        y0, x0 = (nh - h) // 2, (nw - w) // 2
        crop   = zoomed[y0: y0 + h, x0: x0 + w]
        if crop.shape[0] != h or crop.shape[1] != w:
            crop = cv2.resize(crop, (w, h))   # safety guard for off-by-one
        acc += crop.astype(np.float32)
    return np.clip(acc / steps, 0, 255).astype(np.uint8)


def _jpeg(frame: np.ndarray, sev: int) -> np.ndarray:
    """JPEG compression artefacts."""
    quality = (25, 18, 15, 10, 7)[sev - 1]
    _, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _contrast(frame: np.ndarray, sev: int) -> np.ndarray:
    """Contrast reduction towards the per-channel mean."""
    factor = (0.75, 0.60, 0.45, 0.30, 0.15)[sev - 1]
    mean   = np.mean(frame.astype(np.float32), axis=(0, 1), keepdims=True)
    out    = mean + factor * (frame.astype(np.float32) - mean)
    return np.clip(out, 0, 255).astype(np.uint8)


def _rain(frame: np.ndarray, sev: int) -> np.ndarray:
    """Synthetic rain streaks drawn with cv2.line onto an overlay."""
    n_drops = (200, 500, 800, 1200, 1800)[sev - 1]
    s_len   = (8,   12,  16,  20,   24  )[sev - 1]
    h, w    = frame.shape[:2]

    overlay  = np.zeros_like(frame)
    xs       = np.random.randint(0, w, n_drops)
    ys       = np.random.randint(0, max(1, h - s_len), n_drops)
    angles   = np.random.uniform(-0.3, 0.3, n_drops)
    intens   = np.random.randint(180, 256, n_drops).astype(int)

    for i in range(n_drops):
        x2  = int(xs[i] + s_len * angles[i])
        y2  = int(ys[i] + s_len)
        val = int(intens[i])
        cv2.line(overlay, (int(xs[i]), int(ys[i])), (x2, y2), (val, val, val), 1)

    return cv2.add(frame, overlay)


# Registry — everything except h265_abr (handled at video level)
_FRAME_FN: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    'gauss':    _gauss,
    'pepper':   _pepper,
    'salt':     _salt,
    'shot':     _shot,
    'impulse':  _impulse,
    'defocus':  _defocus,
    'motion':   _motion,
    'zoom':     _zoom,
    'jpeg':     _jpeg,
    'contrast': _contrast,
    'rain':     _rain,
}


# -- Video I/O helpers ----------------------------------------------------------

def _read_video(path: Path) -> Tuple[List[np.ndarray], float, int, int]:
    """Return (frames, fps, width, height)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: List[np.ndarray] = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps, w, h


def _open_writer(path: Path, fps: float, w: int, h: int) -> cv2.VideoWriter:
    """Try multiple AVI codecs in priority order; return first that opens."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for code in ('XVID', 'mp4v', 'MJPG'):
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, (w, h))
        if wr.isOpened():
            return wr
        wr.release()
    raise RuntimeError(
        f'No usable AVI codec found (tried XVID / mp4v / MJPG) for: {path}'
    )


def _write_frames(frames: List[np.ndarray], path: Path, fps: float, w: int, h: int) -> None:
    wr = _open_writer(path, fps, w, h)
    for f in frames:
        wr.write(f)
    wr.release()


# -- H.265 ABR via ffmpeg ------------------------------------------------------─

def _check_ffmpeg_h265() -> bool:
    """Return True if ffmpeg with libx265 / hevc is available."""
    try:
        r = subprocess.run(
            ['ffmpeg', '-codecs'], capture_output=True, text=True, check=True
        )
        out = r.stdout.lower()
        return 'libx265' in out or ('hevc' in out and 'encoders' not in out.split('hevc')[0][-20:])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _h265_roundtrip(
    frames: List[np.ndarray],
    fps: float, w: int, h: int,
    bitrate_kbps: int,
) -> List[np.ndarray]:
    """
    Encode all frames through libx265 at a low ABR, then decode.
    Returns the decoded frame list (same length as input).
    """
    with tempfile.TemporaryDirectory(prefix='vidtent_h265_') as tmp:
        src_avi = os.path.join(tmp, 'src.avi')
        dst_mp4 = os.path.join(tmp, 'dst.mp4')

        # Write source frames
        wr = cv2.VideoWriter(src_avi, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
        for f in frames:
            wr.write(f)
        wr.release()

        # H.265 encode at low ABR
        subprocess.run(
            ['ffmpeg', '-y', '-i', src_avi,
             '-c:v', 'libx265', '-b:v', f'{bitrate_kbps}k',
             '-preset', 'ultrafast',
             '-x265-params', 'log-level=none',
             dst_mp4],
            capture_output=True, check=True,
        )

        # Decode
        cap = cv2.VideoCapture(dst_mp4)
        decoded: List[np.ndarray] = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            decoded.append(f)
        cap.release()

    # Ensure length matches original (H.265 may drop/duplicate edge frames)
    n = len(frames)
    if len(decoded) < n:
        decoded.extend(frames[len(decoded):])   # pad with originals
    return decoded[:n]


def _jpeg_fallback_batch(frames: List[np.ndarray], quality: int) -> List[np.ndarray]:
    """Per-frame heavy JPEG compression as h265_abr fallback."""
    out = []
    for f in frames:
        _, enc = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, quality])
        out.append(cv2.imdecode(enc, cv2.IMREAD_COLOR))
    return out


# -- Per-video processing ------------------------------------------------------─

def _process_frame_level(
    src: Path, dst: Path,
    corruption: str, sev: int, prob: float, seed: int,
) -> Optional[str]:
    """
    Read src, apply per-frame corruption with probability `prob`, write dst.
    Returns None on success, or an error string on failure.
    """
    try:
        random.seed(seed)
        np.random.seed(seed)

        frames, fps, w, h = _read_video(src)
        if not frames:
            return 'empty video (no readable frames)'

        fn   = _FRAME_FN[corruption]
        out  = [fn(f, sev) if random.random() < prob else f for f in frames]
        _write_frames(out, dst, fps, w, h)
        return None
    except Exception as exc:
        return str(exc)


def _process_h265(
    src: Path, dst: Path,
    sev: int, prob: float, seed: int, has_ffmpeg: bool,
) -> Optional[str]:
    """
    H.265 ABR corruption:
      1. Encode entire video through libx265 at low ABR (or fallback JPEG).
      2. For each frame, choose the compressed version with probability `prob`.
    Returns None on success, or an error string on failure.
    """
    try:
        random.seed(seed)
        np.random.seed(seed)

        frames, fps, w, h = _read_video(src)
        if not frames:
            return 'empty video (no readable frames)'

        if has_ffmpeg:
            bitrate = (400, 200, 100, 50, 25)[sev - 1]
            try:
                compressed = _h265_roundtrip(frames, fps, w, h, bitrate)
            except subprocess.CalledProcessError as e:
                # ffmpeg encode failed for this clip — use JPEG fallback
                quality    = (18, 12, 8, 5, 3)[sev - 1]
                compressed = _jpeg_fallback_batch(frames, quality)
        else:
            quality    = (18, 12, 8, 5, 3)[sev - 1]
            compressed = _jpeg_fallback_batch(frames, quality)

        out = [c if random.random() < prob else f
               for f, c in zip(frames, compressed)]
        _write_frames(out, dst, fps, w, h)
        return None
    except Exception as exc:
        return str(exc)


# -- Multiprocessing worker (must be top-level for pickling) ------------------─

def _worker(task: tuple) -> Tuple[str, Optional[str]]:
    """
    Unpack task tuple and dispatch to the right processing function.
    task = (src_str, dst_str, corruption, sev, prob, seed, has_ffmpeg)
    Returns (src_str, error_or_None).
    """
    src_str, dst_str, corruption, sev, prob, seed, has_ffmpeg = task
    src, dst = Path(src_str), Path(dst_str)

    if corruption == 'h265_abr':
        err = _process_h265(src, dst, sev, prob, seed, has_ffmpeg)
    else:
        err = _process_frame_level(src, dst, corruption, sev, prob, seed)

    return (src_str, err)


# -- CLI ------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        '--path', type=Path, default=Path('./datasets/UCF50_mixed_labels.csv'),
        help='Path to CSV file containing video_path,corruption_type mappings (default: datasets/UCF50_mixed_labels.csv)',
    )
    parser.add_argument(
        '--prob', type=float, default=0.65,
        help='Probability (0-1) of corrupting each frame (default: 0.70)',
    )
    parser.add_argument(
        '--severity', type=int, default=3, choices=range(1, 6), metavar='1-5',
        help='Corruption severity: 1 = mild … 5 = severe (default: 4)',
    )
    parser.add_argument(
        '--src', type=Path, default=_UCF50_DEFAULT,
        help=f'Source UCF50 directory (default: {_UCF50_DEFAULT})',
    )
    parser.add_argument(
        '--dst-root', type=Path, default=None,
        help=(
            'Parent directory for output datasets '
            '(default: same parent as --src, i.e. datasets/)'
        ),
    )
    parser.add_argument(
        '--workers', type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help='Parallel worker processes (default: CPU_count // 2)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Base random seed for reproducibility (default: 42)',
    )
    return parser.parse_args()

def _execute_tasks(tasks: list, workers: int, desc: str, out_dir: Path) -> None:
    """Helper function to run multiprocessing and handle error reporting."""
    errors: List[Tuple[str, str]] = []

    if workers > 1:
        with Pool(processes=workers) as pool:
            bar = tqdm(
                pool.imap_unordered(_worker, tasks),
                total=len(tasks),
                desc=f'{desc:>10s}',
                unit='vid',
            )
            for src_str, err in bar:
                if err:
                    errors.append((src_str, err))
    else:
        for task in tqdm(tasks, desc=f'{desc:>10s}', unit='vid'):
            src_str, err = _worker(task)
            if err:
                errors.append((src_str, err))

    if errors:
        print(f'  ! {len(errors)} video(s) failed:')
        for path_str, msg in errors[:5]:
            print(f'      {Path(path_str).name}: {msg}')
        if len(errors) > 5:
            print(f'      … and {len(errors) - 5} more (increase verbosity to see all).')

    ok_count = len(tasks) - len(errors)
    print(f'  {ok_count}/{len(tasks)} videos written -> {out_dir}\n')


def main() -> None:
    args = _parse_args()

    # -- Read CSV mapping -------------------------------------------------------
    if not args.path.is_file():
        print(f'[ERROR] CSV file not found: {args.path}', file=sys.stderr)
        sys.exit(1)

    csv_mapping: Dict[str, str] = {}
    with open(args.path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_mapping[row['video_path']] = row['corruption_type']

    if not csv_mapping:
        print(f'[ERROR] CSV file is empty: {args.path}', file=sys.stderr)
        sys.exit(1)

    # -- Validate source directory ----------------------------------------------
    if not args.src.is_dir():
        print(f'[ERROR] Source directory not found: {args.src}', file=sys.stderr)
        sys.exit(1)

    dst_root = args.dst_root if args.dst_root else args.src.parent
    out_dir  = dst_root / 'UCF50_mixed'

    # -- Check ffmpeg / libx265 if any video needs h265_abr --------------------
    has_ffmpeg = False
    if 'h265_abr' in csv_mapping.values():
        has_ffmpeg = _check_ffmpeg_h265()
        if not has_ffmpeg:
            print(
                '[WARN] ffmpeg with libx265 not found.\n'
                '       h265_abr will use heavy-JPEG as a block-artefact proxy.\n'
                '       For true H.265 artefacts: sudo apt install ffmpeg\n'
            )

    # -- Collect source videos -------------------------------------------------
    src_videos = sorted(args.src.rglob('*.avi'))
    if not src_videos:
        print(f'[ERROR] No .avi files found under {args.src}', file=sys.stderr)
        sys.exit(1)

    # -- Summary ---------------------------------------------------------------
    print('=' * 60)
    print(f'  CSV         : {args.path}')
    print(f'  Source      : {args.src}')
    print(f'  Videos      : {len(src_videos)}')
    print(f'  Probability : {args.prob}')
    print(f'  Severity    : {args.severity} / 5')
    print(f'  Workers     : {args.workers}')
    print(f'  Seed        : {args.seed}')
    print(f'  Output dir  : {out_dir}')
    print('=' * 60)
    print()

    # -- Build task list -------------------------------------------------------
    tasks = []
    for i, src_p in enumerate(src_videos):
        rel_path = str(src_p.relative_to(args.src))
        if rel_path not in csv_mapping:
            print(f'[ERROR] {rel_path} not found in CSV', file=sys.stderr)
            sys.exit(1)
        chosen_corruption = csv_mapping[rel_path]
        chosen_corruption = "rain" # for testing
        dst_p = out_dir / src_p.relative_to(args.src)
        tasks.append((
            str(src_p),
            str(dst_p),
            chosen_corruption,
            args.severity,
            args.prob,
            args.seed + i,
            has_ffmpeg,
        ))

    _execute_tasks(tasks, args.workers, 'from_csv', out_dir)
    print('All corruptions complete.')

if __name__ == '__main__':
    main()
    