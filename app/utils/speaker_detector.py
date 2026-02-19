"""
Speaker Detector Module - Dynamic Face Tracking Smart Crop

Detects faces per frame and produces smooth crop keyframes that follow
the active speaker throughout the clip.

Pipeline:
  1. Sample frames at regular intervals
  2. Detect faces per frame
  3. Use diarization data to identify active speaker → pick correct face
  4. Compute crop_x keypoints per sample
  5. Smooth keypoints to avoid jitter
  6. FFmpeg renders with animated crop expression
"""

import os
import time
import tempfile
import subprocess
import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from app.utils.logging import logger

# Max resolution for face detection (resize frame if larger)
# S3FD ga butuh full resolution — 640px width cukup untuk detect muka
DETECTION_MAX_WIDTH = 640

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

    Samples frames, detects faces, uses diarization to follow the active speaker.
    Falls back to largest face if no diarization data is available.
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
                    os.path.dirname(__file__), "..", "..", "model-asd", "loconet_repo"
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

    def _resize_for_detection(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize frame for faster face detection.
        Returns (resized_frame, scale_factor).
        Scale factor is used to map bbox coordinates back to original size.
        """
        h, w = frame.shape[:2]
        if w <= DETECTION_MAX_WIDTH:
            return frame, 1.0
        scale = DETECTION_MAX_WIDTH / w
        new_w = DETECTION_MAX_WIDTH
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _detect_faces_raw(
        self, frame: np.ndarray, conf_threshold: float = 0.7
    ) -> List[FaceBBox]:
        """Detect faces on an already-resized frame (no resize step)."""
        if self.face_detector is not None:
            try:
                bboxes = self.face_detector.detect_faces(frame, conf_th=conf_threshold)
                self._stats["s3fd_ok"] = self._stats.get("s3fd_ok", 0) + 1
                return [
                    FaceBBox(
                        x1=float(b[0]), y1=float(b[1]),
                        x2=float(b[2]), y2=float(b[3]),
                        confidence=float(b[4]),
                    )
                    for b in bboxes
                ]
            except Exception as e:
                self._stats["s3fd_fail"] = self._stats.get("s3fd_fail", 0) + 1
                logger.warning(f"S3FD detection failed (fallback to OpenCV): {e}")
        return self._detect_faces_opencv(frame)

    def detect_faces_in_frame(
        self, frame: np.ndarray, conf_threshold: float = 0.7
    ) -> List[FaceBBox]:
        # Resize for speed
        small_frame, scale = self._resize_for_detection(frame)

        if self.face_detector is not None:
            try:
                bboxes = self.face_detector.detect_faces(small_frame, conf_th=conf_threshold)
                self._stats["s3fd_ok"] = self._stats.get("s3fd_ok", 0) + 1
                # Scale bbox coordinates back to original resolution
                inv_scale = 1.0 / scale
                return [
                    FaceBBox(
                        x1=float(b[0]) * inv_scale,
                        y1=float(b[1]) * inv_scale,
                        x2=float(b[2]) * inv_scale,
                        y2=float(b[3]) * inv_scale,
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

    # ── Frame Extraction ─────────────────────────────────────────────────────

    def _extract_frames_ffmpeg(
        self, video_path: str, sample_times: List[float], max_width: int = DETECTION_MAX_WIDTH
    ) -> List[Optional[np.ndarray]]:
        """
        Extract frames at regular intervals using a single FFmpeg command.
        Uses fps filter for consistent frame extraction — much faster than per-frame seeking.

        Returns list of frames (or None for failed extractions).
        """
        if not sample_times:
            return []

        tmp_dir = tempfile.mkdtemp(prefix="coclip_frames_")

        try:
            start_time = sample_times[0]
            end_time = sample_times[-1] + 0.5  # small buffer
            duration = end_time - start_time
            num_frames = len(sample_times)

            # Calculate fps to get approximately the right number of frames
            target_fps = num_frames / duration if duration > 0 else 2.0

            out_pattern = os.path.join(tmp_dir, "frame_%04d.jpg")

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.3f}",
                "-t", f"{duration:.3f}",
                "-i", video_path,
                "-vf", f"fps={target_fps:.4f},scale={max_width}:-1",
                "-q:v", "2",
                out_pattern,
            ]

            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=300,
            )

            if result.returncode != 0:
                stderr = result.stderr.decode(errors="ignore")[-200:]
                logger.warning(f"FFmpeg batch extraction failed: {stderr}")
                return self._extract_frames_ffmpeg_fallback(video_path, sample_times, max_width, tmp_dir)

            # Read extracted frames
            frames = []
            i = 1
            while True:
                frame_path = os.path.join(tmp_dir, f"frame_{i:04d}.jpg")
                if not os.path.exists(frame_path):
                    break
                frames.append(cv2.imread(frame_path))
                i += 1

            # Pad or trim to match expected count
            if len(frames) < num_frames:
                # Pad with last frame or None
                last = frames[-1] if frames else None
                while len(frames) < num_frames:
                    frames.append(last)
            elif len(frames) > num_frames:
                frames = frames[:num_frames]

            return frames

        except Exception as e:
            logger.warning(f"FFmpeg frame extraction failed: {e}")
            return [None] * len(sample_times)
        finally:
            # Cleanup temp frames
            for f in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, f))
                except OSError:
                    pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

    def _extract_frames_ffmpeg_fallback(
        self, video_path: str, sample_times: List[float], max_width: int, tmp_dir: str
    ) -> List[Optional[np.ndarray]]:
        """Fallback: extract frames one by one with -ss seeking."""
        frames = []
        for i, t in enumerate(sample_times):
            out_path = os.path.join(tmp_dir, f"fallback_{i:04d}.jpg")
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{t:.3f}",
                "-i", video_path,
                "-vframes", "1",
                "-vf", f"scale={max_width}:-1",
                "-q:v", "2",
                out_path,
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10,
            )
            if result.returncode == 0 and os.path.exists(out_path):
                frames.append(cv2.imread(out_path))
            else:
                frames.append(None)
        return frames

    # ── Speaker-Diarization Helpers ─────────────────────────────────────────

    def _filter_segments_for_clip(self, segments, clip_start: float, clip_end: float) -> list:
        """Pre-filter segments to only those overlapping with clip range."""
        return [
            seg for seg in segments
            if seg.end >= clip_start and seg.start <= clip_end and seg.speaker
        ]

    def _get_active_speaker_at(self, clip_segments: list, timestamp: float) -> Optional[str]:
        """Return speaker label active at given absolute timestamp.

        Args:
            clip_segments: Pre-filtered segments (from _filter_segments_for_clip).
        """
        for seg in clip_segments:
            if seg.start <= timestamp <= seg.end:
                return seg.speaker
        return None

    def _build_speaker_face_map(
        self,
        faces_per_frame: List[List[FaceBBox]],
        sample_times: List[float],
        clip_segments: list,
        frame_width: int,
        target_width: int,
    ) -> dict:
        """
        Map speaker labels to face X positions using diarization data.

        Strategy (dual-mapping + smoothness test for 2 speakers):
          1. From multi-face frames, identify left and right face clusters
          2. Try both possible speaker→face mappings (A=left/B=right, A=right/B=left)
          3. For each mapping, simulate crop_x per frame based on active speaker
          4. Pick the mapping with less jitter (smoother crop = correct mapping)

          Rationale: correct mapping → crop stays on one person during their speech,
          wrong mapping → crop jumps to the wrong person during speech.

        For 1 or 3+ speakers, falls back to single-face direct mapping.

        Returns:
            dict: {"SPEAKER_00": avg_center_x, "SPEAKER_01": avg_center_x, ...}
        """
        unique_speakers = set(seg.speaker for seg in clip_segments if seg.speaker)
        speaker_list = sorted(unique_speakers)

        # For exactly 2 speakers: use dual-mapping smoothness test
        if len(unique_speakers) == 2:
            return self._build_two_speaker_map(
                faces_per_frame, sample_times, clip_segments,
                speaker_list, frame_width, target_width,
            )

        # For 1 or 3+ speakers: direct mapping from single-face frames
        speaker_xs = {}
        for sample_time, faces in zip(sample_times, faces_per_frame):
            if len(faces) != 1:
                continue
            speaker = self._get_active_speaker_at(clip_segments, sample_time)
            if speaker:
                speaker_xs.setdefault(speaker, []).append(faces[0].center_x)

        speaker_map = {
            spk: sum(pos) / len(pos) for spk, pos in speaker_xs.items() if pos
        }
        if speaker_map:
            details = ", ".join(
                f"{spk}=x{int(x)} ({len(speaker_xs[spk])} samples)"
                for spk, x in speaker_map.items()
            )
            logger.info(f"    👥 Speaker→position map (direct): {details}")
        else:
            logger.info(f"    👥 Speaker→position map: empty")
        return speaker_map

    def _build_two_speaker_map(
        self,
        faces_per_frame: List[List[FaceBBox]],
        sample_times: List[float],
        clip_segments: list,
        speaker_list: list,
        frame_width: int,
        target_width: int,
    ) -> dict:
        """
        Build speaker→face mapping for exactly 2 speakers using smoothness test.

        Try both possible mappings and pick the one that produces less crop jitter.
        """
        # Step 1: Collect left/right face positions from multi-face frames
        left_xs = []
        right_xs = []
        for faces in faces_per_frame:
            if len(faces) >= 2:
                sorted_faces = sorted(faces, key=lambda f: f.center_x)
                left_xs.append(sorted_faces[0].center_x)
                right_xs.append(sorted_faces[-1].center_x)

        if not left_xs or not right_xs:
            logger.info(f"    👥 No multi-face frames found, falling back to largest face")
            return {}

        avg_left = sum(left_xs) / len(left_xs)
        avg_right = sum(right_xs) / len(right_xs)

        # If faces are too close together, can't distinguish
        if abs(avg_right - avg_left) < 100:
            logger.info(
                f"    👥 Faces too close (left=x{int(avg_left)}, right=x{int(avg_right)}, "
                f"diff={int(avg_right - avg_left)}px), falling back to largest face"
            )
            return {}

        spk_a, spk_b = speaker_list[0], speaker_list[1]

        # Step 2: Try both mappings and compute jitter for each
        mapping_a = {spk_a: avg_left, spk_b: avg_right}  # A=left, B=right
        mapping_b = {spk_a: avg_right, spk_b: avg_left}  # A=right, B=left

        jitter_a = self._compute_mapping_jitter(
            faces_per_frame, sample_times, clip_segments,
            mapping_a, frame_width, target_width,
        )
        jitter_b = self._compute_mapping_jitter(
            faces_per_frame, sample_times, clip_segments,
            mapping_b, frame_width, target_width,
        )

        # Step 3: Pick mapping with MORE jitter
        # In podcasts, reaction shots (close-up of listener) make the WRONG mapping
        # appear smoother. The CORRECT mapping has more jitter because it actively
        # switches between speakers while reaction shots show the other person.
        if jitter_a >= jitter_b:
            best_map = mapping_a
            logger.info(
                f"    👥 Mapping test: A={spk_a}=left,{spk_b}=right "
                f"(jitter={jitter_a:.0f}) vs B={spk_a}=right,{spk_b}=left "
                f"(jitter={jitter_b:.0f}) → picked A (higher jitter = correct for podcast)"
            )
        else:
            best_map = mapping_b
            logger.info(
                f"    👥 Mapping test: A={spk_a}=left,{spk_b}=right "
                f"(jitter={jitter_a:.0f}) vs B={spk_a}=right,{spk_b}=left "
                f"(jitter={jitter_b:.0f}) → picked B (higher jitter = correct for podcast)"
            )

        details = ", ".join(f"{spk}=x{int(x)}" for spk, x in best_map.items())
        logger.info(f"    👥 Speaker→position map: {details}")
        return best_map

    def _compute_mapping_jitter(
        self,
        faces_per_frame: List[List[FaceBBox]],
        sample_times: List[float],
        clip_segments: list,
        speaker_map: dict,
        frame_width: int,
        target_width: int,
    ) -> float:
        """
        Simulate crop_x trajectory with a given speaker→position mapping.
        Returns total jitter (sum of frame-to-frame crop_x changes).
        Less jitter = smoother = more likely correct mapping.
        """
        prev_crop_x = (frame_width - target_width) // 2
        total_jitter = 0.0

        for sample_time, faces in zip(sample_times, faces_per_frame):
            if not faces:
                continue

            speaker = self._get_active_speaker_at(clip_segments, sample_time)
            if speaker and speaker in speaker_map and len(faces) > 1:
                speaker_x = speaker_map[speaker]
                target_face = min(faces, key=lambda f: abs(f.center_x - speaker_x))
            else:
                target_face = max(faces, key=lambda f: f.area)

            crop_x = self._calc_crop_x(target_face, frame_width, target_width)
            total_jitter += abs(crop_x - prev_crop_x)
            prev_crop_x = crop_x

        return total_jitter

    # ── Main Detection Pipeline ─────────────────────────────────────────────

    def detect_active_speakers(
        self,
        video_path: str,
        clip_candidates: list,
        frame_width: int,
        frame_height: int,
        target_width: int,
        target_height: int,
        segments=None,
        sample_interval: float = SAMPLE_INTERVAL,
    ) -> List[SpeakerPosition]:
        """
        Detect faces across each clip and produce smoothed crop keyframes.

        For each clip:
        1. Extract frames via FFmpeg (fast seeking)
        2. Detect faces per frame
        3. Use diarization (segments) to pick the active speaker's face
        4. Apply exponential smoothing to avoid jitter
        5. Return keyframes for FFmpeg animated crop

        Args:
            segments: TranscriptionSegment list with speaker labels (from WhisperX).
                      If None, falls back to largest-face selection.
        """
        results = []
        center_x = (frame_width - target_width) // 2
        has_diarization = segments is not None and len(segments) > 0

        total_clips = len(clip_candidates)
        logger.info(
            f"Starting face detection: {total_clips} clips, "
            f"video={frame_width}x{frame_height}, "
            f"detection_width={min(frame_width, DETECTION_MAX_WIDTH)}px, "
            f"diarization={'enabled' if has_diarization else 'disabled (largest face)'}"
        )

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

            logger.info(
                f"  Clip {clip_idx + 1}/{total_clips}: "
                f"{duration:.1f}s ({start_time:.1f}-{end_time:.1f}), "
                f"{num_samples} samples"
            )

            # Extract all frames via FFmpeg (fast)
            logger.info(f"    Extracting {num_samples} frames via FFmpeg...")
            t_extract = time.time()
            extracted_frames = self._extract_frames_ffmpeg(video_path, sample_times)
            extract_elapsed = time.time() - t_extract
            extracted_ok = sum(1 for f in extracted_frames if f is not None)
            logger.info(
                f"    Extracted {extracted_ok}/{num_samples} frames in {extract_elapsed:.1f}s"
            )

            # Phase 1: Detect all faces on all frames
            all_faces_per_frame: List[List[FaceBBox]] = []
            self._stats = {}  # reset per clip
            face_found = 0
            face_miss = 0
            frame_fail = 0
            detect_start = time.time()

            for sample_idx, (sample_time, frame) in enumerate(
                zip(sample_times, extracted_frames)
            ):
                if frame is None:
                    frame_fail += 1
                    all_faces_per_frame.append([])
                    continue

                t0 = time.time()
                faces = self._detect_faces_raw(frame)
                detect_ms = (time.time() - t0) * 1000

                if sample_idx % 10 == 0 or sample_idx == num_samples - 1:
                    logger.debug(
                        f"    Detect {sample_idx + 1}/{num_samples} "
                        f"({detect_ms:.0f}ms, faces={len(faces)})"
                    )

                # Scale face coordinates back to original resolution
                scale = frame_width / frame.shape[1] if frame.shape[1] > 0 else 1.0
                scaled_faces = [
                    FaceBBox(
                        x1=f.x1 * scale, y1=f.y1 * scale,
                        x2=f.x2 * scale, y2=f.y2 * scale,
                        confidence=f.confidence,
                    )
                    for f in faces
                ]

                if scaled_faces:
                    face_found += 1
                else:
                    face_miss += 1

                all_faces_per_frame.append(scaled_faces)

            # Phase 2: Build speaker→face position map (if diarization available)
            speaker_face_map = {}
            clip_segments = []
            if has_diarization:
                clip_segments = self._filter_segments_for_clip(segments, start_time, end_time)
                if clip_segments:
                    speaker_face_map = self._build_speaker_face_map(
                        all_faces_per_frame, sample_times, clip_segments,
                        frame_width, target_width,
                    )

            # Phase 3: Select target face per frame and compute crop_x
            raw_keyframes: List[Tuple[float, int, Optional[FaceBBox]]] = []
            prev_crop_x = center_x

            for sample_idx, (sample_time, scaled_faces) in enumerate(
                zip(sample_times, all_faces_per_frame)
            ):
                if not scaled_faces:
                    raw_keyframes.append((sample_time - start_time, prev_crop_x, None))
                    continue

                # Pick target face based on active speaker or largest face
                largest_face = max(scaled_faces, key=lambda f: f.area)
                target_face = largest_face
                pick_reason = "largest"

                if has_diarization and speaker_face_map and len(scaled_faces) > 1:
                    active_speaker = self._get_active_speaker_at(
                        clip_segments, sample_time
                    )
                    if active_speaker and active_speaker in speaker_face_map:
                        speaker_x = speaker_face_map[active_speaker]
                        target_face = min(scaled_faces, key=lambda f: abs(f.center_x - speaker_x))
                        pick_reason = f"speaker:{active_speaker}"

                        # Log when speaker-aware pick differs from largest face
                        if sample_idx % 10 == 0 and target_face is not largest_face:
                            logger.debug(
                                f"    Frame {sample_idx}: {pick_reason} "
                                f"(x={int(target_face.center_x)}) "
                                f"instead of largest (x={int(largest_face.center_x)})"
                            )

                crop_x = self._calc_crop_x(target_face, frame_width, target_width)
                raw_keyframes.append((sample_time - start_time, crop_x, target_face))
                prev_crop_x = crop_x

            clip_elapsed = time.time() - detect_start
            total_elapsed = time.time() - t_extract
            s3fd_fail = self._stats.get("s3fd_fail", 0)
            logger.info(
                f"  Clip {clip_idx + 1}/{total_clips} done in {total_elapsed:.1f}s "
                f"(extract={extract_elapsed:.1f}s, detect={clip_elapsed:.1f}s): "
                f"face_found={face_found}, "
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

            speaker_info = ""
            if has_diarization and clip_segments:
                unique_speakers = set(seg.speaker for seg in clip_segments)
                speaker_info = f", speakers_in_clip={unique_speakers}"
            logger.info(
                f"  Clip {clip_idx + 1}: {len(smoothed_keyframes)} keyframes, "
                f"primary crop_x={primary_crop_x}, "
                f"mode={'speaker-aware' if speaker_face_map else 'largest-face'}"
                f"{speaker_info}"
            )

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
