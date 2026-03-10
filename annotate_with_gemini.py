"""
Gemini API を使用して画像をアノテーションし、
YOLO セグメンテーション形式（ポリゴン）でデータセットを保存するプログラム

YOLO セグメンテーション形式:
  <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
  座標は画像サイズで正規化（0.0〜1.0）
"""

import os
import re
import json
import base64
import argparse
import shutil
from pathlib import Path

import google.generativeai as genai
from PIL import Image


# -------------------------
# 設定っ！
# -------------------------
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def encode_image_base64(image_path: Path) -> str:
    """画像をbase64エンコードして返す"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_size(image_path: Path) -> tuple[int, int]:
    """画像の (width, height) を返す"""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def build_prompt(class_names: list[str], user_instruction: str) -> str:
    """Gemini に送るプロンプトを構築する"""
    classes_str = "\n".join(f"  - {i}: {name}" for i, name in enumerate(class_names))
    return f"""あなたは画像アノテーションの専門家です。
以下の指示に従って、画像内のオブジェクトをポリゴンでアノテーションしてください。

## アノテーション指示
{user_instruction}

## クラス定義
{classes_str}

## 出力形式（必ず以下のJSON形式のみで返答してください）
```json
{{
  "annotations": [
    {{
      "class_id": 0,
      "class_name": "クラス名",
      "polygon": [
        {{"x": 0.12, "y": 0.34}},
        {{"x": 0.56, "y": 0.78}},
        ...
      ]
    }}
  ]
}}
```

## 注意事項
- polygon の座標は画像の幅・高さで正規化した値（0.0〜1.0）で指定してください
- 1つのオブジェクトに対して8〜20点程度の頂点でポリゴンを描いてください
- 複数のオブジェクトが存在する場合はすべてアノテーションしてください
- 対象オブジェクトが存在しない場合は "annotations": [] としてください
- JSON以外のテキストは一切含めないでください
"""


def parse_gemini_response(response_text: str) -> list[dict]:
    """Gemini のレスポンスからアノテーションを抽出する"""
    # コードブロック内のJSONを抽出
    json_match = re.search(r"```(?:json)?\s*(.*?)```", response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # コードブロックなしでJSONを探す
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"JSONが見つかりません:\n{response_text}")
        json_str = json_match.group(0)

    data = json.loads(json_str)
    return data.get("annotations", [])


def annotations_to_yolo(annotations: list[dict], img_w: int, img_h: int) -> list[str]:
    """
    アノテーションを YOLO セグメンテーション形式の文字列リストに変換する

    形式: <class_id> <x1> <y1> <x2> <y2> ...
    """
    lines = []
    for ann in annotations:
        class_id = ann["class_id"]
        polygon = ann["polygon"]
        if len(polygon) < 3:
            print(f"  警告: ポリゴンの頂点が3点未満のためスキップ (class_id={class_id})")
            continue

        coords = []
        for pt in polygon:
            x = float(pt["x"])
            y = float(pt["y"])
            # 0〜1の範囲にクリップ
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            coords.extend([f"{x:.6f}", f"{y:.6f}"])

        line = f"{class_id} " + " ".join(coords)
        lines.append(line)
    return lines


def annotate_image(
    model: genai.GenerativeModel,
    image_path: Path,
    class_names: list[str],
    user_instruction: str,
) -> list[dict]:
    """1枚の画像をGeminiでアノテーションする"""
    prompt = build_prompt(class_names, user_instruction)

    # 画像をアップロード
    uploaded = genai.upload_file(str(image_path))
    response = model.generate_content([prompt, uploaded])

    annotations = parse_gemini_response(response.text)
    return annotations


def save_yaml(output_dir: Path, class_names: list[str]) -> None:
    """YOLO用のdataset.yamlを保存する"""
    yaml_content = f"""# YOLO Dataset Configuration
path: {output_dir.resolve()}
train: images/train
val: images/val

nc: {len(class_names)}
names: {json.dumps(class_names, ensure_ascii=False)}
"""
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"  dataset.yaml を保存: {yaml_path}")


def process_images(
    input_dir: Path,
    output_dir: Path,
    class_names: list[str],
    user_instruction: str,
    api_key: str,
    model_name: str = "gemini-2.0-flash",
    split: str = "train",
    dry_run: bool = False,
) -> None:
    """フォルダ内の全画像を処理してデータセットを作成する"""

    # Gemini API 初期化
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # 出力ディレクトリ作成
    images_out = output_dir / "images" / split
    labels_out = output_dir / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # 画像ファイル一覧
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        print(f"画像ファイルが見つかりません: {input_dir}")
        return

    print(f"\n{len(image_files)} 枚の画像を処理します")
    print(f"出力先: {output_dir}")
    print(f"クラス: {class_names}\n")

    success_count = 0
    error_count = 0

    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {image_path.name} を処理中...")

        try:
            img_w, img_h = get_image_size(image_path)

            if dry_run:
                print(f"  DRY RUN: スキップ (size={img_w}x{img_h})")
                continue

            # Gemini でアノテーション
            annotations = annotate_image(model, image_path, class_names, user_instruction)
            print(f"  検出オブジェクト数: {len(annotations)}")

            # YOLO形式に変換
            yolo_lines = annotations_to_yolo(annotations, img_w, img_h)

            # アノテーションファイル保存
            label_path = labels_out / (image_path.stem + ".txt")
            label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

            # 画像をコピー
            dest_image = images_out / image_path.name
            shutil.copy2(image_path, dest_image)

            # アノテーション詳細を表示
            for ann in annotations:
                print(f"  - {ann['class_name']} (class_id={ann['class_id']}, "
                      f"頂点数={len(ann['polygon'])})")

            success_count += 1

        except Exception as e:
            print(f"  エラー: {e}")
            error_count += 1

    # dataset.yaml を保存
    save_yaml(output_dir, class_names)

    print(f"\n完了: 成功={success_count}, エラー={error_count}")
    print(f"データセット: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Gemini API で画像をアノテーションして YOLO セグメンテーション形式で保存"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="入力画像フォルダのパス"
    )
    parser.add_argument(
        "--output", "-o", default="dataset",
        help="出力データセットフォルダのパス (デフォルト: dataset)"
    )
    parser.add_argument(
        "--classes", "-c", required=True, nargs="+",
        help="クラス名のリスト (例: --classes crack spall rust)"
    )
    parser.add_argument(
        "--instruction", required=True,
        help="アノテーション指示（プロンプト）"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API キー (環境変数 GEMINI_API_KEY でも指定可)"
    )
    parser.add_argument(
        "--model", default="gemini-2.0-flash",
        help="使用する Gemini モデル (デフォルト: gemini-2.0-flash)"
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "val", "test"],
        help="データセットの分割 (デフォルト: train)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="実際には処理せず動作確認のみ行う"
    )

    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            "Gemini API キーを指定してください。\n"
            "  --api-key オプション、または\n"
            "  環境変数 GEMINI_API_KEY を設定してください。"
        )

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        parser.error(f"入力フォルダが存在しません: {input_dir}")

    process_images(
        input_dir=input_dir,
        output_dir=Path(args.output),
        class_names=args.classes,
        user_instruction=args.instruction,
        api_key=args.api_key,
        model_name=args.model,
        split=args.split,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
