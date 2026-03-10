"""
Gemini API を使用して画像をアノテーションし、
YOLO セグメンテーション形式（ポリゴン）でデータセットを保存するプログラム

YOLO セグメンテーション形式:
  <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
  座標は画像サイズで正規化（0.0〜1.0）
"""

import os
import json
import time
import argparse
import shutil
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel


# -------------------------
# 設定
# -------------------------
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# レート制限時のリトライ設定
MAX_RETRIES = 5
RETRY_WAIT_BUFFER = 5  # APIが指定する待機時間に加算する余裕秒数

# レスポンスの最大トークン数（モデルの上限に合わせる）
MAX_OUTPUT_TOKENS = 65536


# -------------------------
# Pydantic スキーマ定義
# -------------------------
class Point(BaseModel):
    x: float
    y: float

class Annotation(BaseModel):
    class_id: int
    class_name: str
    polygon: list[Point]

class AnnotationResult(BaseModel):
    annotations: list[Annotation]


def get_image_size(image_path: Path) -> tuple[int, int]:
    """画像の (width, height) を返す"""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def build_prompt(class_names: list[str], user_instruction: str, max_points: int = 20) -> str:
    """Gemini に送るプロンプトを構築する"""
    classes_str = "\n".join(f"  - {i}: {name}" for i, name in enumerate(class_names))
    return f"""あなたは画像アノテーションの専門家です。
以下の指示に従って、画像内のオブジェクトをポリゴンでアノテーションしてください。

## アノテーション指示
{user_instruction}

## クラス定義
{classes_str}

## 注意事項
- polygon の座標は画像の幅・高さで正規化した値（0.0〜1.0）で指定してください
- 1つのオブジェクトに対して最大{max_points}点の頂点でポリゴンを描いてください
- 複数のオブジェクトが存在する場合はすべてアノテーションしてください
- 対象オブジェクトが存在しない場合は annotations を空配列にしてください
"""


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
            x = max(0.0, min(1.0, float(pt["x"])))
            y = max(0.0, min(1.0, float(pt["y"])))
            coords.extend([f"{x:.6f}", f"{y:.6f}"])

        lines.append(f"{class_id} " + " ".join(coords))
    return lines


def extract_retry_delay(error_message: str) -> float | None:
    """エラーメッセージからリトライ待機秒数を抽出する"""
    import re
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(error_message))
    if match:
        return float(match.group(1))
    match = re.search(r"seconds:\s*(\d+)", str(error_message))
    if match:
        return float(match.group(1))
    return None


def annotate_image_with_retry(
    client: genai.Client,
    model_name: str,
    image_path: Path,
    class_names: list[str],
    user_instruction: str,
) -> list[dict]:
    """1枚の画像をGeminiでアノテーションする（レート制限・JSONエラー時はリトライ）"""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".bmp": "image/bmp", ".webp": "image/webp",
    }
    mime_type = mime_map.get(image_path.suffix.lower(), "image/jpeg")

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=AnnotationResult,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    # リトライごとにポリゴン点数の上限を段階的に減らす
    max_points_schedule = [20, 12, 8, 6, 4]

    for attempt in range(1, MAX_RETRIES + 1):
        max_points = max_points_schedule[min(attempt - 1, len(max_points_schedule) - 1)]
        prompt = build_prompt(class_names, user_instruction, max_points=max_points)

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
                config=config,
            )
            result = AnnotationResult.model_validate_json(response.text)
            return [ann.model_dump() for ann in result.annotations]

        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            is_json_error = "json_invalid" in error_str or "EOF" in error_str or "JSONDecodeError" in error_str

            # 1日の上限超過は待機しても解決しないため即停止
            if is_rate_limit and "GenerateRequestsPerDay" in error_str and "limit: 0" in error_str:
                raise RuntimeError(
                    "1日あたりの無料枠リクエスト上限に達しました。\n"
                    "明日以降に再試行するか、Google AI Studio で課金を有効にしてください。\n"
                    "  https://aistudio.google.com/apikey"
                ) from e

            if is_rate_limit and attempt < MAX_RETRIES:
                wait_sec = extract_retry_delay(error_str)
                if wait_sec is None:
                    wait_sec = 60 * attempt
                wait_sec += RETRY_WAIT_BUFFER
                print(f"  レート制限 (試行 {attempt}/{MAX_RETRIES}): {wait_sec:.0f}秒待機します...")
                time.sleep(wait_sec)
                continue

            # JSONが途中で切れた場合はポリゴン点数を減らしてリトライ
            if is_json_error and attempt < MAX_RETRIES:
                next_points = max_points_schedule[min(attempt, len(max_points_schedule) - 1)]
                print(f"  JSONエラー (試行 {attempt}/{MAX_RETRIES}): ポリゴン点数を{max_points}→{next_points}に削減してリトライ...")
                continue

            raise


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
    model_name: str = "gemini-2.5-flash",
    split: str = "train",
    dry_run: bool = False,
) -> None:
    """フォルダ内の全画像を処理してデータセットを作成する"""

    client = genai.Client(api_key=api_key)

    images_out = output_dir / "images" / split
    labels_out = output_dir / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

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

            annotations = annotate_image_with_retry(
                client, model_name, image_path, class_names, user_instruction
            )
            print(f"  検出オブジェクト数: {len(annotations)}")

            yolo_lines = annotations_to_yolo(annotations, img_w, img_h)

            label_path = labels_out / (image_path.stem + ".txt")
            label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

            shutil.copy2(image_path, images_out / image_path.name)

            for ann in annotations:
                print(f"  - {ann['class_name']} (class_id={ann['class_id']}, "
                      f"頂点数={len(ann['polygon'])})")

            success_count += 1

        except RuntimeError as e:
            print(f"\n致命的エラー: {e}")
            break

        except Exception as e:
            print(f"  エラー: {e}")
            error_count += 1

    save_yaml(output_dir, class_names)

    print(f"\n完了: 成功={success_count}, エラー={error_count}")
    print(f"データセット: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Gemini API で画像をアノテーションして YOLO セグメンテーション形式で保存"
    )
    parser.add_argument(
        "--input", "-i", default="images",
        help="入力画像フォルダのパス (デフォルト: images)"
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
        "--model", default="gemini-2.5-flash",
        help="使用する Gemini モデル (デフォルト: gemini-2.5-flash)"
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
