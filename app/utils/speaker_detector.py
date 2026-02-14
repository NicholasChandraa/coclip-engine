"""
Speaker Detector Module - Dynamic Face Tracking Smart Crop

Detects faces per frame and produces smooth crop keyframes that follow
the most prominent face throughout the clip.

Pipeline:
  1. Sample frames at regular intervals
  2. Detect largest face per frame
  3. Compute crop_x keypoints per sample
  4. Smooth keypoints to avoid jitter
  5. FFmpeg renders with animated crop expression
"""

import os
import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from app.utils.logging import logger

# ── Tuning Constants ────────────────────────────────────────────────────
# Ubah nilai-nilai ini untuk adjust behavior smart crop

# Geser crop ke kiri atau kanan
# 0.0 = tetap di tengah (tidak ikuti wajah)
# 0.5 = setengah jalan antara tengah dan wajah
# 0.7 = lebih dekat ke wajah (default)
# 1.0 = pas tepat di wajah
CROP_STRENGTH = 1

# Cek wajah setiap berapa detik
# 0.3 = cek ~3x per detik (responsif)
# 0.5 = cek 2x per detik (balance)
# 1.0 = cek 1x per detik (cepat proses)
SAMPLE_INTERVAL = 0.3

# Smoothing untuk menghindari crop goyang-goyang
# 0.0 = tidak ada smoothing (langsung ikuti wajah, bisa goyang)
# 0.3 = sedikit smooth (responsif tapi stabil)
# 0.5 = smooth sedang
# 0.8 = sangat smooth (lambat bergerak)
SMOOTHING = 0.3

# Durasi smooth transition antar posisi crop (detik)
# Dipakai saat render FFmpeg
# 0.2 = cepat
# 0.3 = default
# 0.5 = pelan
TRANSITION_DURATION = 0.3


@dataclass
class FaceBBox:
    """Bounding box for a detected face."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class CropKeyframe:
    """A single crop position at a point in time (relative to clip start)."""

    time: float  # seconds from clip start
    crop_x: int


@dataclass
class SpeakerPosition:
    """Position info for a detected speaker in a clip."""

    clip_index: int
    crop_x: int  # Primary crop_x (most common position)
    active_speaker_bbox: Optional[FaceBBox] = None
    confidence: float = 0.0
    is_fallback: bool = False
    keyframes: List[CropKeyframe] = field(default_factory=list)


class SpeakerDetector:
    """
    Dynamic face-tracking smart crop detector.

    Samples frames, detects largest face, produces smoothed crop keyframes.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.face_detector = None
        self._loaded = False
        self._stats = {}

    def load(self):
        if self._loaded:
            return
        logger.info("Loading face detector...")
        self._load_face_detector()
        self._loaded = True
        logger.info("Face detector loaded")

    def unload(self):
        if self.face_detector is not None:
            del self.face_detector
            self.face_detector = None
        self._loaded = False
        logger.info("Face detector unloaded")

    def _load_face_detector(self):
        try:
            import sys
            import torch

            logger.info(f"  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(0)
                vram_gb = gpu_props.total_memory / 1024**3
                logger.info(f"  GPU: {gpu_props.name}, VRAM: {vram_gb:.1f}GB")

            loconet_repo = os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "models", "loconet_repo"
                )
            )
            if loconet_repo not in sys.path:
                sys.path.insert(0, loconet_repo)

            original_cwd = os.getcwd()
            os.chdir(loconet_repo)
            try:
                from model.faceDetector.s3fd import S3FD

                self.face_detector = S3FD(device=self.device)
            finally:
                os.chdir(original_cwd)

            logger.info(f"  S3FD face detector loaded on {self.device}")
        except Exception as e:
            logger.warning(f"  S3FD not available, using OpenCV: {e}")
            self.face_detector = None

    # ── Face Detection ──────────────────────────────────────────────────────

    def detect_faces_in_frame(
        self, frame: np.ndarray, conf_threshold: float = 0.7
    ) -> List[FaceBBox]:
        if self.face_detector is not None:
            try:
                bboxes = self.face_detector.detect_faces(frame, conf_th=conf_threshold)
                self._stats["s3fd_ok"] = self._stats.get("s3fd_ok", 0) + 1
                return [
                    FaceBBox(
                        x1=float(b[0]),
                        y1=float(b[1]),
                        x2=float(b[2]),
                        y2=float(b[3]),
                        confidence=float(b[4]),
                    )
                    for b in bboxes
                ]
            except Exception as e:
                self._stats["s3fd_fail"] = self._stats.get("s3fd_fail", 0) + 1
                logger.warning(f"S3FD detection failed (fallback to OpenCV): {e}")
        return self._detect_faces_opencv(frame)

    def _detect_faces_opencv(self, frame: np.ndarray) -> List[FaceBBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        return [
            FaceBBox(
                x1=float(x),
                y1=float(y),
                x2=float(x + w),
                y2=float(y + h),
                confidence=0.9,
            )
            for (x, y, w, h) in faces
        ]

    # ── Crop Position ───────────────────────────────────────────────────────

    def _calc_crop_x(
        self,
        face: FaceBBox,
        frame_width: int,
        target_width: int,
        strength: float = CROP_STRENGTH,
    ) -> int:
        center_x = (frame_width - target_width) // 2
        face_crop_x = int(face.center_x - target_width / 2)
        crop_x = int(center_x + (face_crop_x - center_x) * strength)
        return max(0, min(crop_x, frame_width - target_width))

    # ── Main Detection Pipeline ─────────────────────────────────────────────

    def detect_active_speakers(
        self,
        video_path: str,
        clip_candidates: list,
        frame_width: int,
        frame_height: int,
        target_width: int,
        target_height: int,
        sample_interval: float = SAMPLE_INTERVAL,
    ) -> List[SpeakerPosition]:
        """
        Detect faces across each clip and produce smoothed crop keyframes.

        For each clip:
        1. Sample a frame every `sample_interval` seconds
        2. Detect largest face → compute raw crop_x
        3. Apply exponential smoothing to avoid jitter
        4. Return keyframes for FFmpeg animated crop
        """
        results = []
        center_x = (frame_width - target_width) // 2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return self._fallback_positions(clip_candidates, frame_width, target_width)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        for clip_idx, clip in enumerate(clip_candidates):
            start_time = clip.get("start_time", 0)
            end_time = clip.get("end_time", 0)
            duration = end_time - start_time

            if duration <= 0:
                results.append(
                    self._center_position(clip_idx, frame_width, target_width)
                )
                continue

            # Sample frames
            num_samples = max(2, int(duration / sample_interval) + 1)
            sample_times = [
                start_time + (duration * i / (num_samples - 1))
                for i in range(num_samples)
            ]

            # Detect faces and compute raw crop_x per sample
            raw_keyframes: List[Tuple[float, int, Optional[FaceBBox]]] = []
            prev_crop_x = center_x
            self._stats = {}  # reset per clip
            face_found = 0
            face_miss = 0
            frame_fail = 0

            for sample_time in sample_times:
                frame_idx = int(sample_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    frame_fail += 1
                    raw_keyframes.append((sample_time - start_time, prev_crop_x, None))
                    continue

                faces = self.detect_faces_in_frame(frame)

                if not faces:
                    face_miss += 1
                    raw_keyframes.append((sample_time - start_time, prev_crop_x, None))
                    continue

                face_found += 1
                largest_face = max(faces, key=lambda f: f.area)
                crop_x = self._calc_crop_x(largest_face, frame_width, target_width)
                raw_keyframes.append((sample_time - start_time, crop_x, largest_face))
                prev_crop_x = crop_x

            # Detection stats
            s3fd_fail = self._stats.get("s3fd_fail", 0)
            logger.info(
                f"  Clip {clip_idx + 1} detection stats: "
                f"total={num_samples}, face_found={face_found}, "
                f"face_miss={face_miss}, frame_fail={frame_fail}, "
                f"s3fd_errors={s3fd_fail}"
            )

            # Apply exponential smoothing
            smoothed_keyframes = self._smooth_keyframes(raw_keyframes)

            # Determine primary crop_x (most common / longest held position)
            if smoothed_keyframes:
                crop_values = [kf.crop_x for kf in smoothed_keyframes]
                primary_crop_x = int(np.median(crop_values))
            else:
                primary_crop_x = center_x

            # Find best face for reference
            best_face = None
            best_area = 0
            for _, _, face in raw_keyframes:
                if face and face.area > best_area:
                    best_area = face.area
                    best_face = face

            results.append(
                SpeakerPosition(
                    clip_index=clip_idx,
                    crop_x=primary_crop_x,
                    active_speaker_bbox=best_face,
                    confidence=best_face.confidence if best_face else 0.0,
                    is_fallback=len(smoothed_keyframes) == 0,
                    keyframes=smoothed_keyframes,
                )
            )

            logger.info(
                f"  Clip {clip_idx + 1}: {len(smoothed_keyframes)} keyframes, "
                f"primary crop_x={primary_crop_x}"
            )

        cap.release()
        return results

    def _smooth_keyframes(
        self, raw: List[Tuple[float, int, Optional[FaceBBox]]]
    ) -> List[CropKeyframe]:
        """
        Apply exponential smoothing to raw crop positions.

        This prevents jitter from frame-to-frame face detection noise,
        while still responding to real camera cuts (large position changes).
        """
        if not raw:
            return []

        keyframes = []
        smoothed_x = float(raw[0][1])

        for t, crop_x, _ in raw:
            # If big jump (camera cut), snap immediately instead of smoothing
            if abs(crop_x - smoothed_x) > 50:
                smoothed_x = float(crop_x)
            else:
                # Exponential smoothing: new = old * alpha + target * (1 - alpha)
                smoothed_x = smoothed_x * SMOOTHING + crop_x * (1 - SMOOTHING)

            keyframes.append(CropKeyframe(time=t, crop_x=int(smoothed_x)))

        return keyframes

    def _center_position(
        self, clip_idx: int, frame_width: int, target_width: int
    ) -> SpeakerPosition:
        return SpeakerPosition(
            clip_index=clip_idx,
            crop_x=(frame_width - target_width) // 2,
            is_fallback=True,
        )

    def _fallback_positions(
        self, clips: list, frame_width: int, target_width: int
    ) -> List[SpeakerPosition]:
        return [
            self._center_position(i, frame_width, target_width)
            for i in range(len(clips))
        ]


def smooth_crop_transitions(
    positions: List[SpeakerPosition], smoothing: float = 0.3
) -> List[SpeakerPosition]:
    """Apply smoothing between consecutive clip positions."""
    if len(positions) <= 1:
        return positions

    smoothed = [positions[0]]
    for i in range(1, len(positions)):
        prev_x = smoothed[i - 1].crop_x
        target_x = positions[i].crop_x
        smooth_x = int(prev_x + (target_x - prev_x) * (1 - smoothing))
        smoothed.append(
            SpeakerPosition(
                clip_index=positions[i].clip_index,
                crop_x=smooth_x,
                active_speaker_bbox=positions[i].active_speaker_bbox,
                confidence=positions[i].confidence,
                is_fallback=positions[i].is_fallback,
                keyframes=positions[i].keyframes,
            )
        )
    return smoothed
