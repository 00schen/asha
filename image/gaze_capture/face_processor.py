import dlib
import cv2
import math
import numpy as np
from .ITrackerData import loadMetadata
import os
from pathlib import Path
main_dir = str(Path(__file__).resolve().parents[1])

class FaceProcessor:
    def __init__(self, predictor_path):
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.img_dim = 224
        self.face_grid_dim = 25
        self.left_eye_points = [42, 43, 44, 45, 46, 47]
        self.right_eye_points = [36, 37, 38, 39, 40, 41]

        self.face_mean = loadMetadata(os.path.join(main_dir,'gaze_capture','model_files','mean_face_224.mat'),
                                      silent=True)['image_mean']
        self.left_eye_mean = loadMetadata(os.path.join(main_dir,'gaze_capture','model_files','mean_left_224.mat'),
                                          silent=True)['image_mean']
        self.right_eye_mean = loadMetadata(os.path.join(main_dir,'gaze_capture','model_files','mean_right_224.mat'),
                                           silent=True)['image_mean']

    def get_gaze_features(self, frame):
        height, width = frame.shape[:2]
        diff = height - width

        # crop image to square
        if diff > 0:
            frame = frame[math.floor(diff / 2): -math.ceil(diff / 2)]
        elif diff < 0:
            frame = frame[:, -math.floor(diff / 2): math.ceil(diff / 2)]

        gs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_box = self._get_facial_detections(gs_frame)

        if face_box is None:
            return None

        face = self._get_face(frame, face_box)
        if face is None:
            return None

        face = (face - self.face_mean) / 255

        face_grid = self._get_face_grid(frame, face_box)

        landmarks = self.predictor(gs_frame, face_box)

        og_left_eye = self._get_eye(frame, landmarks, self.left_eye_points)
        og_right_eye = self._get_eye(frame, landmarks, self.right_eye_points)

        left_eye = (og_left_eye - self.left_eye_mean) / 255
        right_eye = (og_right_eye - self.right_eye_mean) / 255

        face = np.moveaxis(face, -1, 0)
        left_eye = np.moveaxis(left_eye, -1, 0)
        right_eye = np.moveaxis(right_eye, -1, 0)
        return face, left_eye, right_eye, face_grid,  # og_left_eye, og_right_eye

    def _get_face(self, frame, face_box):
        try:
            face = frame[face_box.top(): face_box.bottom(), face_box.left(): face_box.right()]
            face = cv2.resize(face, (self.img_dim, self.img_dim))
            face = np.flip(face, axis=2)
        except:
            return None
        return face

    def _get_face_grid(self, frame, face_box):
        frame_dim = len(frame)
        top = math.floor(face_box.top() * self.face_grid_dim / frame_dim)
        bottom = math.ceil(face_box.bottom() * self.face_grid_dim / frame_dim)
        left = math.floor(face_box.left() * self.face_grid_dim / frame_dim)
        right = math.ceil(face_box.right() * self.face_grid_dim / frame_dim)
        face_grid = np.zeros((self.face_grid_dim, self.face_grid_dim))
        face_grid[top: bottom, left: right] = 1
        return face_grid

    def _get_eye(self, frame, landmarks, points):
        eye_landmarks = self._get_landmarks(landmarks, points)
        left, top, width, height = cv2.boundingRect(eye_landmarks)

        w_margin = int(width / 3)
        h_margin = (width + 2 * w_margin - height) / 2
        top_margin = math.ceil(h_margin)
        bot_margin = math.floor(h_margin)

        eye = frame[top - top_margin: top + height + bot_margin, left - w_margin: left + width + w_margin]
        eye = cv2.resize(eye, (self.img_dim, self.img_dim))
        eye = np.flip(eye, axis=2)

        return eye

    def get_eye_aspect_ratio(self, frame):
        gs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_box = self._get_facial_detections(gs_frame)

        if face_box is None:
            return None

        landmarks = self.predictor(gs_frame, face_box)
        left_eye_landmarks = self._get_landmarks(landmarks, self.left_eye_points)
        right_eye_landmarks = self._get_landmarks(landmarks, self.right_eye_points)
        left_eye_aspect_ratio = self._eye_aspect_ratio(left_eye_landmarks)
        right_eye_aspect_ratio = self._eye_aspect_ratio(right_eye_landmarks)
        return (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2

    def _get_facial_detections(self, gs_frame):
        detections = self.face_detector(gs_frame)
        if len(detections) == 0:
            return None
        return detections[0]

    @staticmethod
    def _get_landmarks(landmarks, points):
        return np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])

    @staticmethod
    def _eye_aspect_ratio(eye_landmarks):
        v_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        return (v_1 + v_2) / (2 * h)
