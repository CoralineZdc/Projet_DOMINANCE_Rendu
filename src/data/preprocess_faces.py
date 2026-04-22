from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_face_detector():
    if cv2 is None:
        return None
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return None
    return detector


def detect_largest_face_bbox(image_rgb: np.ndarray, detector) -> Optional[tuple[int, int, int, int]]:
    if detector is None:
        return None

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(18, 18))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    return int(x), int(y), int(w), int(h)


def square_crop(image_rgb: np.ndarray, bbox: Optional[tuple[int, int, int, int]], margin: float, fallback: str) -> np.ndarray:
    h, w = image_rgb.shape[:2]

    if bbox is None:
        if fallback == "full":
            return image_rgb
        side = min(h, w)
        x1 = max(0, (w - side) // 2)
        y1 = max(0, (h - side) // 2)
        return image_rgb[y1 : y1 + side, x1 : x1 + side]

    x, y, bw, bh = bbox
    side = int(max(bw, bh) * (1.0 + 2.0 * margin))
    side = max(1, side)
    cx = x + bw / 2.0
    cy = y + bh / 2.0

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        image_rgb = np.pad(
            image_rgb,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        x1 += pad_left
        y1 += pad_top

    return image_rgb[y1 : y1 + side, x1 : x1 + side]


def to_gray_resized(image_rgb: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(image_rgb)
    image = image.convert("L").resize((size, size), Image.BILINEAR)
    return np.array(image, dtype=np.uint8)


def to_pixels_string(gray: np.ndarray) -> str:
    return " ".join(str(int(v)) for v in gray.reshape(-1))


def load_rows(input_csv: Optional[Path], input_dir: Optional[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    if input_csv is not None:
        with input_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Input CSV has no header: {input_csv}")
            for row in reader:
                rows.append(dict(row))
        return rows

    if input_dir is None:
        raise ValueError("Provide either --input_csv or --input_dir")

    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append({"path": str(path)})
    return rows


def resolve_image_path(row: dict[str, str], input_dir: Optional[Path]) -> Path:
    for key in ("path", "image_path", "file_path", "filename"):
        value = row.get(key)
        if value:
            p = Path(value)
            if p.is_absolute() or input_dir is None:
                return p
            return (input_dir / p).resolve()
    raise ValueError(f"No path-like column found in row: {row}")


def infer_label(row: dict[str, str], image_path: Path, input_dir: Optional[Path]) -> Optional[str]:
    for key in ("label", "emotion", "class", "split"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)

    if input_dir is not None:
        try:
            rel_parent = image_path.resolve().relative_to(input_dir.resolve()).parent
            if rel_parent.parts:
                return rel_parent.parts[0]
        except Exception:
            pass

    return None


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def write_manifest_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess face datasets into centered grayscale square crops and optional FER-style CSV rows."
    )
    parser.add_argument("--input_dir", type=str, default="", help="Root directory containing input images")
    parser.add_argument("--input_csv", type=str, default="", help="Optional CSV with image paths and metadata")
    parser.add_argument("--output_dir", type=str, default="", help="Optional output directory for processed PNG files")
    parser.add_argument("--output_csv", type=str, default="", help="Optional output CSV path")
    parser.add_argument("--size", type=int, default=48, help="Output image size (square)")
    parser.add_argument("--margin", type=float, default=0.15, help="Margin around detected face as fraction of face size")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of images to process (0 means all)")
    parser.add_argument("--face_fallback", choices=["center", "full"], default="center", help="Fallback crop if no face is found")
    parser.add_argument("--preserve_class_dirs", action="store_true", help="Preserve first-level class folder under --output_dir")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else None
    input_csv = Path(args.input_csv) if args.input_csv else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    output_csv = Path(args.output_csv) if args.output_csv else None

    if input_dir is None and input_csv is None:
        raise ValueError("Provide either --input_dir or --input_csv")
    if output_dir is None and output_csv is None:
        raise ValueError("Provide at least one output: --output_dir and/or --output_csv")

    detector = get_face_detector()
    if detector is None:
        print("[Info] OpenCV face detector unavailable; using fallback crop")

    rows = load_rows(input_csv=input_csv, input_dir=input_dir)
    if args.limit > 0:
        rows = rows[: args.limit]

    manifest_rows: list[dict[str, str]] = []
    processed = 0
    faces_found = 0

    for row in rows:
        image_path = resolve_image_path(row, input_dir)
        if not image_path.exists():
            print(f"[Skip] Missing image: {image_path}")
            continue

        try:
            rgb = read_rgb(image_path)
            bbox = detect_largest_face_bbox(rgb, detector)
            if bbox is not None:
                faces_found += 1
            crop = square_crop(rgb, bbox, margin=args.margin, fallback=args.face_fallback)
            gray = to_gray_resized(crop, size=args.size)
        except Exception as exc:
            print(f"[Skip] Failed to process {image_path}: {exc}")
            continue

        label = infer_label(row, image_path, input_dir)
        rel_out_path: Optional[Path] = None

        if output_dir is not None:
            if args.preserve_class_dirs and label:
                rel_out_path = Path(label) / f"{image_path.stem}.png"
            else:
                rel_out_path = Path(f"{image_path.stem}.png")
            save_path = output_dir / rel_out_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(gray, mode="L").save(save_path)

        out_row: dict[str, str] = {
            "source_path": str(image_path),
            "face_found": "1" if bbox is not None else "0",
            "pixels": to_pixels_string(gray),
        }

        if rel_out_path is not None:
            out_row["processed_path"] = rel_out_path.as_posix()
        if label is not None:
            out_row["label"] = label

        for key in ("Valence", "Arousal", "Dominance"):
            if key in row and row[key] not in (None, ""):
                out_row[key] = str(row[key])

        manifest_rows.append(out_row)
        processed += 1

    if output_csv is not None:
        write_manifest_csv(output_csv, manifest_rows)

    print(f"Processed {processed} images")
    print(f"Faces detected in {faces_found} images")
    if output_dir is not None:
        print(f"Saved images to {output_dir}")
    if output_csv is not None:
        print(f"Saved CSV to {output_csv}")


if __name__ == "__main__":
    main()
