"""
YOLO セグメンテーション形式のアノテーションを元画像に重ね合わせて可視化するスクリプト

使用方法:
  python visualize_annotations.py
  python visualize_annotations.py --dataset dataset --output visualized
  python visualize_annotations.py --alpha 0.4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


# クラスごとの色 (BGR)
COLORS = [
    (0, 0, 255),    # crack: 赤
    (0, 255, 255),  # flaking: 黄
    (0, 255, 0),    # 予備: 緑
    (255, 0, 0),    # 予備: 青
    (255, 0, 255),  # 予備: マゼンタ
]


def load_class_names(dataset_yaml: Path) -> list[str]:
    with open(dataset_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("names", [])


def parse_label_file(label_path: Path):
    """YOLOラベルファイルを読み込み、(class_id, points) のリストを返す"""
    annotations = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = list(map(float, line.split()))
            class_id = int(values[0])
            coords = values[1:]
            # x1 y1 x2 y2 ... -> [(x1,y1), (x2,y2), ...]
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            annotations.append((class_id, points))
    return annotations


def draw_annotations(image: np.ndarray, annotations: list, class_names: list[str], alpha: float) -> np.ndarray:
    h, w = image.shape[:2]
    overlay = image.copy()

    for class_id, points in annotations:
        color = COLORS[class_id % len(COLORS)]
        # 正規化座標 -> ピクセル座標
        pts = np.array([(int(x * w), int(y * h)) for x, y in points], dtype=np.int32)

        # 塗りつぶし（半透明）
        cv2.fillPoly(overlay, [pts], color)

        # 輪郭線
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

        # クラス名ラベル
        if class_names and class_id < len(class_names):
            label = class_names[class_id]
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(image, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # 半透明合成
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


def visualize(dataset_dir: Path, output_dir: Path, split: str, alpha: float):
    yaml_path = dataset_dir / "dataset.yaml"
    class_names = load_class_names(yaml_path) if yaml_path.exists() else []

    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    out_dir = output_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not image_files:
        print(f"画像が見つかりません: {images_dir}")
        return

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"  スキップ（ラベルなし）: {img_path.name}")
            continue

        # 日本語パス対応: バイト列で読んでからデコード
        image = cv2.imdecode(np.frombuffer(img_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  読み込み失敗: {img_path.name}")
            continue

        annotations = parse_label_file(label_path)
        result = draw_annotations(image, annotations, class_names, alpha)

        out_path = out_dir / img_path.name
        ext = img_path.suffix.lower()
        success, buf = cv2.imencode(ext, result)
        if success:
            out_path.write_bytes(buf.tobytes())
        print(f"  保存: {out_path}")

    print(f"\n完了: {len(image_files)} 枚処理 -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="YOLOアノテーション可視化")
    parser.add_argument("--dataset", default="dataset", help="datasetディレクトリのパス")
    parser.add_argument("--output", default="visualized", help="出力ディレクトリのパス")
    parser.add_argument("--split", default="train", help="処理するsplit (train/val)")
    parser.add_argument("--alpha", type=float, default=0.35, help="塗りつぶしの透明度 (0.0〜1.0)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)

    print(f"データセット: {dataset_dir}")
    print(f"出力先: {output_dir}")
    print(f"Split: {args.split}")
    print()

    visualize(dataset_dir, output_dir, args.split, args.alpha)


if __name__ == "__main__":
    main()
