"""
Computer Vision Module
Analyzes facial expressions, emotions, eye contact, and head movement.
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class FrameFeatures:
    timestamp: float
    emotion: str
    emotion_confidence: float
    eye_contact: bool
    head_pose: dict
    face_detected: bool


@dataclass
class CVAnalysisResult:
    total_frames: int = 0
    frames_with_face: int = 0
    emotion_counts: dict = field(default_factory=dict)
    eye_contact_frames: int = 0
    nervous_frames: int = 0
    head_movement_variance: float = 0.0
    frame_features: list = field(default_factory=list)
    emotion_stability_score: float = 0.0
    eye_contact_percentage: float = 0.0
    nervous_behavior_score: float = 0.0
    confidence_score: float = 0.0
    timestamps_nervous: list = field(default_factory=list)


class CVAnalyzer:
    """Analyzes video frames for facial expressions, eye contact, and head movement."""

    EMOTION_LABELS = ["neutral", "happy", "nervous", "confused", "confident"]

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

        # Landmark indices for key facial features
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.MOUTH_INDICES = [61, 291, 13, 14, 17, 0, 267, 269, 270, 409]
        self.EYEBROW_LEFT = [70, 63, 105, 66, 107]
        self.EYEBROW_RIGHT = [336, 296, 334, 293, 300]

    def _get_landmarks(self, frame_rgb):
        """Extract face mesh landmarks from a frame."""
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

    def _compute_eye_aspect_ratio(self, landmarks, eye_indices, w, h):
        """Compute Eye Aspect Ratio (EAR) to detect blinks and alertness."""
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_indices])
        # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        ear = (A + B) / (2.0 * C + 1e-6)
        return ear

    def _estimate_head_pose(self, landmarks, w, h):
        """Estimate head pose (yaw, pitch) from face landmarks."""
        # Use nose tip and facial extremities
        nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])
        left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
        right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
        chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
        forehead = np.array([landmarks[10].x * w, landmarks[10].y * h])

        eye_center = (left_eye + right_eye) / 2
        face_vertical = chin - forehead
        face_horizontal = right_eye - left_eye

        # Yaw: left-right rotation
        nose_offset = nose_tip[0] - eye_center[0]
        eye_width = np.linalg.norm(face_horizontal) + 1e-6
        yaw = nose_offset / eye_width * 90

        # Pitch: up-down rotation
        nose_vertical = nose_tip[1] - eye_center[1]
        face_height = np.linalg.norm(face_vertical) + 1e-6
        pitch = (nose_vertical / face_height - 0.5) * 90

        return {"yaw": float(yaw), "pitch": float(pitch)}

    def _detect_eye_contact(self, head_pose):
        """Determine if candidate is maintaining eye contact based on head pose."""
        yaw_thresh = 15  # degrees
        pitch_thresh = 15
        return abs(head_pose["yaw"]) < yaw_thresh and abs(head_pose["pitch"]) < pitch_thresh

    def _classify_emotion(self, landmarks, w, h, ear_left, ear_right):
        """
        Rule-based emotion classification using facial geometry.
        Returns (emotion_label, confidence).
        """
        avg_ear = (ear_left + ear_right) / 2

        # Mouth openness
        upper_lip = np.array([landmarks[13].x * w, landmarks[13].y * h])
        lower_lip = np.array([landmarks[14].x * w, landmarks[14].y * h])
        mouth_open = np.linalg.norm(upper_lip - lower_lip)

        # Lip corner raise (smile proxy)
        left_corner = np.array([landmarks[61].x * w, landmarks[61].y * h])
        right_corner = np.array([landmarks[291].x * w, landmarks[291].y * h])
        lip_corners_y = (left_corner[1] + right_corner[1]) / 2
        lip_center_y = (upper_lip[1] + lower_lip[1]) / 2
        smile_metric = lip_center_y - lip_corners_y  # positive = smile

        # Eyebrow raise (stress/surprise proxy)
        left_brow = np.mean([landmarks[i].y * h for i in self.EYEBROW_LEFT])
        left_eye_top = landmarks[159].y * h
        brow_raise = left_eye_top - left_brow  # larger = raised brows

        # Classification logic
        if avg_ear < 0.18:
            return "nervous", 0.75
        elif smile_metric > 5 and avg_ear > 0.22:
            return "happy", 0.80
        elif brow_raise < 15 and mouth_open < 8:
            return "nervous", 0.65
        elif avg_ear > 0.25 and mouth_open < 6:
            return "confident", 0.70
        else:
            return "neutral", 0.60

    def analyze_video(self, video_path: str, sample_rate: int = 5) -> CVAnalysisResult:
        """
        Analyze a video file for facial features.
        
        Args:
            video_path: Path to the video file
            sample_rate: Process every Nth frame (default: every 5th frame)
        
        Returns:
            CVAnalysisResult with aggregated statistics
        """
        result = CVAnalysisResult()
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return result

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        head_yaws = []
        head_pitches = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % sample_rate != 0:
                    continue

                timestamp = frame_idx / fps
                result.total_frames += 1

                h, w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = self._get_landmarks(frame_rgb)

                if landmarks is None:
                    result.frame_features.append(
                        FrameFeatures(
                            timestamp=timestamp,
                            emotion="unknown",
                            emotion_confidence=0.0,
                            eye_contact=False,
                            head_pose={"yaw": 0, "pitch": 0},
                            face_detected=False,
                        )
                    )
                    continue

                result.frames_with_face += 1

                # Eye aspect ratio
                ear_left = self._compute_eye_aspect_ratio(landmarks, self.LEFT_EYE_INDICES, w, h)
                ear_right = self._compute_eye_aspect_ratio(landmarks, self.RIGHT_EYE_INDICES, w, h)

                # Head pose
                head_pose = self._estimate_head_pose(landmarks, w, h)
                head_yaws.append(head_pose["yaw"])
                head_pitches.append(head_pose["pitch"])

                # Eye contact
                eye_contact = self._detect_eye_contact(head_pose)
                if eye_contact:
                    result.eye_contact_frames += 1

                # Emotion
                emotion, confidence = self._classify_emotion(landmarks, w, h, ear_left, ear_right)
                result.emotion_counts[emotion] = result.emotion_counts.get(emotion, 0) + 1

                if emotion == "nervous":
                    result.nervous_frames += 1
                    result.timestamps_nervous.append(round(timestamp, 2))

                result.frame_features.append(
                    FrameFeatures(
                        timestamp=timestamp,
                        emotion=emotion,
                        emotion_confidence=confidence,
                        eye_contact=eye_contact,
                        head_pose=head_pose,
                        face_detected=True,
                    )
                )

        finally:
            cap.release()

        # Aggregate scores
        if result.frames_with_face > 0:
            result.eye_contact_percentage = round(
                result.eye_contact_frames / result.frames_with_face * 100, 2
            )
            result.nervous_behavior_score = round(
                result.nervous_frames / result.frames_with_face * 100, 2
            )

            # Emotion stability: how consistent are emotions (lower variance = more stable)
            dominant_emotion_pct = max(result.emotion_counts.values()) / result.frames_with_face
            result.emotion_stability_score = round(dominant_emotion_pct * 100, 2)

        if head_yaws:
            yaw_var = np.var(head_yaws)
            pitch_var = np.var(head_pitches)
            result.head_movement_variance = round(float(yaw_var + pitch_var), 4)

        # Composite confidence score from CV signals
        eye_score = result.eye_contact_percentage
        stability_score = result.emotion_stability_score
        calm_score = max(0, 100 - result.nervous_behavior_score)
        movement_penalty = min(30, result.head_movement_variance / 10)
        result.confidence_score = round(
            (eye_score * 0.4 + stability_score * 0.3 + calm_score * 0.3) - movement_penalty, 2
        )
        result.confidence_score = max(0, min(100, result.confidence_score))

        logger.info(
            f"CV Analysis: {result.frames_with_face}/{result.total_frames} frames with face. "
            f"Eye contact: {result.eye_contact_percentage}%, Nervous: {result.nervous_behavior_score}%"
        )
        return result
