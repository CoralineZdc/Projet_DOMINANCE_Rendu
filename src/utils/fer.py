"""FER2013 VAD Dataset class."""

from __future__ import print_function

import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None


class FER2013(data.Dataset):
    """FER2013 Dataset.

    Args:
        split: One of "Training", "PublicTest", "PrivateTest".
        transform: Transform pipeline applied at sample fetch time.
        align_faces: If True and OpenCV is available, apply Haar-cascade face crop.
    """

    label_mean = None
    label_std = None
    image_mean = None
    image_std = None
    face_detector = None
    data_protocol = "small_split"
    split_file_overrides = {}

    @classmethod
    def _repo_root(cls):
        return os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def set_data_protocol(cls, protocol):
        if protocol not in {"small_split"}:
            raise ValueError("Unknown data protocol: {}".format(protocol))
        cls.data_protocol = protocol

    @classmethod
    def _get_split_candidates(cls, split):
        if split in cls.split_file_overrides and cls.split_file_overrides[split]:
            return [cls.split_file_overrides[split]]

        small_split_to_file = {
            "Training": ["./data/train-fer2013-vad.csv"],
            "PublicTest": ["./data/test-fer2013-vad.csv"],
            "PrivateTest": ["./data/val-fer2013-vad.csv"],
        }

        return small_split_to_file[split]

    @classmethod
    def _resolve_data_file(cls, candidates):
        for candidate in candidates:
            search_paths = [candidate]
            if not os.path.isabs(candidate):
                search_paths.append(os.path.join(cls._repo_root(), candidate))
                search_paths.append(os.path.join(cls._repo_root(), "data", candidate))

            for path in search_paths:
                if os.path.exists(path):
                    return path
        raise FileNotFoundError("No dataset file found. Tried: {}".format(", ".join(candidates)))

    @classmethod
    def set_split_files(cls, train_file=None, public_file=None, private_file=None):
        split_map = {
            "Training": train_file,
            "PublicTest": public_file,
            "PrivateTest": private_file,
        }
        for split_name, file_path in split_map.items():
            if file_path:
                cls.split_file_overrides[split_name] = file_path
            elif split_name in cls.split_file_overrides:
                del cls.split_file_overrides[split_name]

        cls.label_mean = None
        cls.label_std = None
        cls.image_mean = None
        cls.image_std = None

    @classmethod
    def _ensure_label_stats(cls):
        if cls.label_mean is not None and cls.label_std is not None:
            return

        train_path = cls._resolve_data_file(cls._get_split_candidates("Training"))
        train_df = pd.read_csv(train_path)
        label_array = train_df[["Valence", "Arousal", "Dominance"]].to_numpy(dtype=np.float32)
        cls.label_mean = torch.tensor(label_array.mean(axis=0), dtype=torch.float32)
        cls.label_std = torch.tensor(label_array.std(axis=0), dtype=torch.float32)
        cls.label_std[cls.label_std == 0] = 1.0

    @classmethod
    def _ensure_image_stats(cls):
        if cls.image_mean is not None and cls.image_std is not None:
            return

        train_path = cls._resolve_data_file(cls._get_split_candidates("Training"))
        train_df = pd.read_csv(train_path)
        pixels = []
        for pixel_entry in train_df["pixels"]:
            pixel_str = str(pixel_entry).strip()
            if pixel_str.lower() == "nan" or not pixel_str:
                continue
            try:
                values = list(map(int, pixel_str.split()))
            except ValueError:
                continue
            if len(values) != 48 * 48:
                continue
            pixels.append(np.array(values, dtype=np.float32) / 255.0)

        if not pixels:
            cls.image_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            cls.image_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            return

        pixel_array = np.stack(pixels, axis=0)
        mean = float(pixel_array.mean())
        std = float(pixel_array.std())
        if std == 0.0:
            std = 1.0

        cls.image_mean = torch.tensor([mean, mean, mean], dtype=torch.float32)
        cls.image_std = torch.tensor([std, std, std], dtype=torch.float32)

    @classmethod
    def _ensure_face_detector(cls):
        if cls.face_detector is not None:
            return
        if cv2 is None:
            cls.face_detector = False
            return
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        cls.face_detector = detector if not detector.empty() else False

    @classmethod
    def _align_face(cls, image):
        cls._ensure_face_detector()
        if not cls.face_detector:
            return image

        rgb_image = np.array(image)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        faces = cls.face_detector.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(18, 18),
        )

        if len(faces) == 0:
            return image

        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        margin_x = int(0.15 * w)
        margin_y = int(0.15 * h)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(rgb_image.shape[1], x + w + margin_x)
        y2 = min(rgb_image.shape[0], y + h + margin_y)

        cropped = rgb_image[y1:y2, x1:x2]
        if cropped.size == 0:
            return image

        return Image.fromarray(cropped).resize((48, 48), Image.BILINEAR)

    def __init__(self, split="Training", transform=None, align_faces=False):
        self.transform = transform
        self.split = split
        self.align_faces = align_faces

        if self.split not in {"Training", "PublicTest", "PrivateTest"}:
            raise ValueError("Unknown split: {}".format(self.split))

        split_candidates = self._get_split_candidates(self.split)
        split_file = self._resolve_data_file(split_candidates)
        self.data = pd.read_csv(split_file)
        pixels_series = self.data["pixels"]
        valence_series = self.data["Valence"]
        arousal_series = self.data["Arousal"]
        dominance_series = self.data["Dominance"]

        processed_images = []
        processed_labels = []
        dropped_outliers = 0

        for idx, pixel_entry in enumerate(pixels_series):
            pixel_str = str(pixel_entry).strip()
            if pixel_str.lower() == "nan" or not pixel_str:
                continue

            try:
                pixels = list(map(int, pixel_str.split()))
            except ValueError:
                print("Warning: Could not parse {} pixel string '{}' at index {}. Skipping this entry.".format(self.split, pixel_str, idx))
                continue

            if len(pixels) != 48 * 48:
                print("Warning: {} pixel string has incorrect length ({}) for 48x48 image at index {}. Skipping this entry.".format(self.split, len(pixels), idx))
                continue

            label_values = [
                valence_series.iloc[idx],
                arousal_series.iloc[idx],
                dominance_series.iloc[idx],
            ]

            label_array = np.asarray(label_values, dtype=np.float32)
            if (label_array < -2.0).any() or (label_array > 2.0).any():
                dropped_outliers += 1
                continue

            arr_2d = np.array(pixels, dtype=np.uint8).reshape(48, 48)
            arr_3d = np.stack([arr_2d, arr_2d, arr_2d], axis=2)
            processed_images.append(arr_3d)
            processed_labels.append(label_values)

        if dropped_outliers > 0:
            print(
                "[Label Warning] '{}' dropped {} rows with labels outside [-2, 2].".format(
                    split_file,
                    dropped_outliers,
                )
            )

        self.images = processed_images
        self._ensure_label_stats()
        labels = torch.tensor(processed_labels, dtype=torch.float32)
        self.labels = (labels - self.label_mean) / self.label_std

        if self.split == "Training":
            self.train_data = self.images
            self.train_labels = self.labels
        elif self.split == "PublicTest":
            self.PublicTest_data = self.images
            self.PublicTest_labels = self.labels
        else:
            self.PrivateTest_data = self.images
            self.PrivateTest_labels = self.labels

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index])
        if self.align_faces:
            img = self._align_face(img)
        if self.transform is not None:
            img = self.transform(img)
        target = self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)
