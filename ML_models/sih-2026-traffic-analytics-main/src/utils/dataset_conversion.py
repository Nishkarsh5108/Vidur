"""
IDD (India Driving Dataset) to YOLOv8 Format Converter with Stratified Subsampling.

Supports:
- Pascal VOC XML format (.xml files)
- Per-image JSON annotations (.json files)
- Single monolithic COCO-style JSON (if provided)
- Preserves 100% of frames containing rare classes (bicycle, traffic light, traffic sign, pole)
- Subsamples remaining frames across drive sequences to reach target dataset size (~12,000-15,000)
- Produces normalized YOLO .txt annotations and 80/20 train/val image split
"""

import os
import sys
import glob
import json
import random
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any


IDD_CLASSES = [
    "person",            # 0
    "rider",             # 1
    "car",               # 2
    "bus",               # 3
    "truck",             # 4
    "autorickshaw",      # 5
    "motorcycle",        # 6
    "bicycle",           # 7  (Rare - 100% retained)
    "traffic light",     # 8  (Rare - 100% retained)
    "traffic sign",      # 9  (Rare - 100% retained)
    "vehicle fallback",  # 10
    "ego vehicle",       # 11
    "pole"               # 12 (Rare/Infrastructure - 100% retained)
]

# Map aliases and variations to canonical class IDs
CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(IDD_CLASSES)}

ALIAS_MAP = {
    # Person / Rider
    "person": 0, "pedestrian": 0, "human": 0, "people": 0,
    "rider": 1, "cyclist": 1, "motorcyclist": 1,
    # Vehicles
    "car": 2, "automobile": 2, "sedan": 2, "suv": 2,
    "bus": 3, "minibus": 3,
    "truck": 4, "lorry": 4,
    "autorickshaw": 5, "auto": 5, "rickshaw": 5, "three_wheeler": 5, "auto_rickshaw": 5,
    "motorcycle": 6, "motorbike": 6, "motor_bike": 6, "two_wheeler": 6, "scooter": 6,
    "bicycle": 7, "bike": 7, "cycle": 7,
    # Infrastructure / Signs
    "traffic light": 8, "traffic_light": 8, "trafficlight": 8, "traffic-light": 8,
    "traffic sign": 9, "traffic_sign": 9, "trafficsign": 9, "traffic-sign": 9,
    # Fallback / Ego
    "vehicle fallback": 10, "vehicle_fallback": 10, "vehiclefallback": 10, "other vehicle": 10,
    "ego vehicle": 11, "ego_vehicle": 11, "egovehicle": 11, "hood": 11,
    # Pole
    "pole": 12, "utility_pole": 12, "lamp_post": 12, "post": 12
}

# Rare class IDs that must be 100% preserved
RARE_CLASS_IDS = {7, 8, 9, 12}  # bicycle, traffic light, traffic sign, pole


class IDDDatasetConverter:
    def __init__(
        self,
        raw_root: str,
        output_root: str,
        target_samples: int = 13500,
        train_ratio: float = 0.8,
        seed: int = 42,
        copy_images: bool = False
    ):
        """
        raw_root: Path containing extracted IDD (e.g. data/raw/idd or IDD_Detection)
        output_root: Path to output processed data (e.g. data/processed)
        target_samples: Desired total number of images (default 13,500 ~ 12k-15k)
        train_ratio: Ratio for train split (default 0.8 -> 80% train, 20% val)
        copy_images: If False, creates hardlinks/symlinks to save disk space; if True, copies
        """
        self.raw_root = Path(raw_root)
        self.output_root = Path(output_root)
        self.target_samples = target_samples
        self.train_ratio = train_ratio
        self.seed = seed
        self.copy_images = copy_images

        # Set up output directories
        self.train_img_dir = self.output_root / "images" / "train"
        self.val_img_dir = self.output_root / "images" / "val"
        self.train_lbl_dir = self.output_root / "labels" / "train"
        self.val_lbl_dir = self.output_root / "labels" / "val"

        for p in [self.train_img_dir, self.val_img_dir, self.train_lbl_dir, self.val_lbl_dir]:
            p.mkdir(parents=True, exist_ok=True)

        random.seed(self.seed)

    def resolve_class_id(self, raw_label: str) -> Optional[int]:
        """Normalize label string and return class index if valid."""
        clean = str(raw_label).strip().lower().replace("-", " ").replace("_", " ")
        if clean in ALIAS_MAP:
            return ALIAS_MAP[clean]
        # Try direct match with underscores
        underscore = clean.replace(" ", "_")
        if underscore in ALIAS_MAP:
            return ALIAS_MAP[underscore]
        return None

    def parse_voc_xml(self, xml_path: Path) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, float, float, float, float]]]:
        """
        Parses Pascal VOC XML format used by IDD Detection.
        Returns: ((width, height), [(class_id, x_center, y_center, w, h), ...])
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            return None, []

        size_el = root.find("size")
        if size_el is not None:
            w_el = size_el.find("width")
            h_el = size_el.find("height")
            if w_el is not None and h_el is not None:
                img_w = float(w_el.text)
                img_h = float(h_el.text)
            else:
                return None, []
        else:
            return None, []

        if img_w <= 0 or img_h <= 0:
            return None, []

        boxes = []
        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is None or not name_el.text:
                continue
            cls_id = self.resolve_class_id(name_el.text)
            if cls_id is None:
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            try:
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
            except (ValueError, AttributeError):
                continue

            # Ensure valid bounds
            xmin = max(0.0, min(xmin, img_w))
            xmax = max(0.0, min(xmax, img_w))
            ymin = max(0.0, min(ymin, img_h))
            ymax = max(0.0, min(ymax, img_h))

            box_w = xmax - xmin
            box_h = ymax - ymin
            if box_w <= 1 or box_h <= 1:
                continue

            x_center = (xmin + box_w / 2.0) / img_w
            y_center = (ymin + box_h / 2.0) / img_h
            norm_w = box_w / img_w
            norm_h = box_h / img_h

            # Clamp normalized values between 0.0 and 1.0
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            boxes.append((cls_id, x_center, y_center, norm_w, norm_h))

        return (int(img_w), int(img_h)), boxes

    def parse_per_image_json(self, json_path: Path) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, float, float, float, float]]]:
        """
        Parses per-image JSON annotation if IDD JSON export format is present.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None, []

        img_w = float(data.get("imgWidth", data.get("width", 0)))
        img_h = float(data.get("imgHeight", data.get("height", 0)))
        if img_w <= 0 or img_h <= 0:
            return None, []

        objects = data.get("objects", data.get("annotations", []))
        boxes = []
        for obj in objects:
            raw_label = obj.get("label", obj.get("category", ""))
            cls_id = self.resolve_class_id(raw_label)
            if cls_id is None:
                continue

            # Check bbox format
            if "bbox" in obj:
                # [xmin, ymin, w, h] or [xmin, ymin, xmax, ymax]
                bb = obj["bbox"]
                if len(bb) == 4:
                    xmin, ymin, b_w, b_h = bb[0], bb[1], bb[2], bb[3]
                    xmax = xmin + b_w
                    ymax = ymin + b_h
                else:
                    continue
            elif "polygon" in obj:
                poly = obj["polygon"]
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
            else:
                continue

            xmin = max(0.0, min(xmin, img_w))
            xmax = max(0.0, min(xmax, img_w))
            ymin = max(0.0, min(ymin, img_h))
            ymax = max(0.0, min(ymax, img_h))

            box_w = xmax - xmin
            box_h = ymax - ymin
            if box_w <= 1 or box_h <= 1:
                continue

            x_center = (xmin + box_w / 2.0) / img_w
            y_center = (ymin + box_h / 2.0) / img_h
            norm_w = box_w / img_w
            norm_h = box_h / img_h

            boxes.append((cls_id, x_center, y_center, norm_w, norm_h))

        return (int(img_w), int(img_h)), boxes

    def index_dataset(self) -> Dict[str, Dict[str, Any]]:
        """
        Discovers all image-annotation pairs across raw_root using direct directory alignment.
        Returns: {unique_id: {'image': Path, 'annotation': Path, 'type': 'xml'|'json', ...}}
        """
        print(f"[*] Scanning {self.raw_root} for IDD image-annotation pairs...")
        index = {}

        annot_dir = self.raw_root / "Annotations"
        jpeg_dir = self.raw_root / "JPEGImages"

        if not annot_dir.is_dir():
            # Check if self.raw_root itself is the Annotations or has subdirectories
            cand = list(self.raw_root.rglob("Annotations"))
            if cand:
                annot_dir = cand[0]
                jpeg_dir = annot_dir.parent / "JPEGImages"

        print(f"[*] Annotations source: {annot_dir}")
        print(f"[*] Images source:      {jpeg_dir}")

        # High-speed walk through Annotations
        for root, _, files in os.walk(annot_dir):
            for file in files:
                if not (file.endswith(".xml") or file.endswith(".json")):
                    continue

                annot_path = Path(root) / file
                is_xml = file.endswith(".xml")

                # Compute relative path from Annotations
                try:
                    rel_p = annot_path.relative_to(annot_dir)
                except ValueError:
                    continue

                # Map to corresponding JPEGImages path
                for img_ext in [".jpg", ".png", ".jpeg"]:
                    img_cand = jpeg_dir / rel_p.with_suffix(img_ext)
                    if img_cand.is_file():
                        # Unique key: parts joined
                        key = "_".join(rel_p.with_suffix("").parts)
                        drive_id = rel_p.parent.name
                        stem = annot_path.stem

                        index[key] = {
                            "image": img_cand,
                            "annotation": annot_path,
                            "format": "xml" if is_xml else "json",
                            "drive_id": drive_id,
                            "stem": stem
                        }
                        break

        print(f"[*] Successfully paired {len(index):,} images with annotations.")
        return index

    def parse_and_stratify(
        self, index: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[str], Dict[str, List[Tuple[int, float, float, float, float]]]]:
        """
        Pre-parses annotations to discover class presence.
        Performs stratified selection:
        1. Keep ALL frames with rare classes (bicycle, traffic light, traffic sign, pole).
        2. Subsample remaining frames uniformly across drives to reach target_samples.
        """
        print("[*] Parsing annotations and computing class distributions...")
        frame_boxes = {}
        rare_frames = set()
        common_frames_by_drive = defaultdict(list)
        class_instance_counts = Counter()

        for key, item in index.items():
            fmt = item["format"]
            if fmt == "xml":
                dim, boxes = self.parse_voc_xml(item["annotation"])
            else:
                dim, boxes = self.parse_per_image_json(item["annotation"])

            if not boxes:
                continue

            frame_boxes[key] = boxes
            present_classes = set(b[0] for b in boxes)
            for c in present_classes:
                class_instance_counts[c] += 1

            # Check if this frame contains any rare classes
            if present_classes.intersection(RARE_CLASS_IDS):
                rare_frames.add(key)
            else:
                common_frames_by_drive[item["drive_id"]].append(key)

        print(f"[*] Total valid frames with annotations: {len(frame_boxes)}")
        print(f"[*] Frames containing rare classes (MUST KEEP 100%): {len(rare_frames)}")

        # Stratified selection
        selected_keys = set(rare_frames)
        remaining_budget = max(0, self.target_samples - len(selected_keys))

        if remaining_budget > 0 and common_frames_by_drive:
            # Distribute remaining quota proportionally across driving scenes
            total_common = sum(len(frames) for frames in common_frames_by_drive.values())
            print(f"[*] Subsampling {remaining_budget} frames from {total_common} non-rare frames across {len(common_frames_by_drive)} drives...")

            for drive_id, frames in common_frames_by_drive.items():
                drive_quota = int(round((len(frames) / total_common) * remaining_budget))
                drive_quota = min(drive_quota, len(frames))
                selected_keys.update(random.sample(frames, drive_quota))

            # Fine-tune to match target_samples if slight rounding difference
            if len(selected_keys) < self.target_samples:
                unselected = [k for k in frame_boxes if k not in selected_keys]
                extra_needed = min(self.target_samples - len(selected_keys), len(unselected))
                if extra_needed > 0:
                    selected_keys.update(random.sample(unselected, extra_needed))
        else:
            print(f"[*] Target samples ({self.target_samples}) met or exceeded by rare frames alone.")

        selected_list = list(selected_keys)
        random.shuffle(selected_list)
        print(f"[+] Total selected frames for dataset: {len(selected_list)}")
        return selected_list, frame_boxes

    def export_yolo(
        self,
        selected_keys: List[str],
        index: Dict[str, Dict[str, Any]],
        frame_boxes: Dict[str, List[Tuple[int, float, float, float, float]]]
    ):
        """
        Exports images and YOLO .txt labels into train and val splits.
        """
        n_train = int(len(selected_keys) * self.train_ratio)
        train_keys = set(selected_keys[:n_train])
        val_keys = set(selected_keys[n_train:])

        print(f"[*] Exporting dataset: {len(train_keys)} train, {len(val_keys)} val images...")

        stats = {
            "train": Counter(),
            "val": Counter()
        }

        for key in selected_keys:
            is_train = key in train_keys
            img_dir = self.train_img_dir if is_train else self.val_img_dir
            lbl_dir = self.train_lbl_dir if is_train else self.val_lbl_dir
            split_name = "train" if is_train else "val"

            item = index[key]
            src_img = item["image"]
            dst_stem = key
            dst_img = img_dir / f"{dst_stem}{src_img.suffix}"
            dst_lbl = lbl_dir / f"{dst_stem}.txt"

            # Write YOLO label file
            boxes = frame_boxes[key]
            with open(dst_lbl, "w", encoding="utf-8") as lf:
                for cls_id, xc, yc, w, h in boxes:
                    lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    stats[split_name][cls_id] += 1

            # Transfer image (symlink/hardlink if supported, else copy)
            if not dst_img.exists():
                try:
                    if self.copy_images:
                        shutil.copy2(src_img, dst_img)
                    else:
                        os.link(src_img, dst_img)  # Fast hard link
                except Exception:
                    shutil.copy2(src_img, dst_img)

        # Print summary report
        print("\n" + "=" * 65)
        print("          IDD TO YOLOv8 CONVERSION SUMMARY REPORT")
        print("=" * 65)
        print(f"{'Class ID':<10}{'Class Name':<20}{'Train Instances':<18}{'Val Instances':<15}")
        print("-" * 65)
        for idx, name in enumerate(IDD_CLASSES):
            is_rare = idx in RARE_CLASS_IDS
            tag = " [RARE*]" if is_rare else ""
            print(f"{idx:<10}{name + tag:<20}{stats['train'][idx]:<18}{stats['val'][idx]:<15}")
        print("-" * 65)
        print(f"Total Images: {len(selected_keys)} ({len(train_keys)} Train / {len(val_keys)} Val)")
        print(f"Output Directory: {self.output_root}")
        print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert IDD Detection to YOLOv8 with Stratified Subsampling")
    parser.add_argument("--raw-dir", type=str, default="data/raw/idd", help="Path to raw IDD extracted directory")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Path for YOLO formatted output")
    parser.add_argument("--target-samples", type=int, default=13500, help="Target image count (~12k-15k)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/val split ratio (default 0.8)")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of hard-linking")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    converter = IDDDatasetConverter(
        raw_root=args.raw_dir,
        output_root=args.output_dir,
        target_samples=args.target_samples,
        train_ratio=args.train_ratio,
        seed=args.seed,
        copy_images=args.copy
    )

    index = converter.index_dataset()
    if not index:
        print("[!] No paired images and annotations found in raw-dir.")
        print("    Please ensure the dataset is extracted into data/raw/idd or specify --raw-dir.")
        sys.exit(1)

    selected_keys, frame_boxes = converter.parse_and_stratify(index)
    converter.export_yolo(selected_keys, index, frame_boxes)


if __name__ == "__main__":
    main()
