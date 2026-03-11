import os
import io
import math
import time
import shutil
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

# -------------------------
# Pydantic スキーマ定義
# -------------------------
class Point(BaseModel):
    x: float  # 0.0〜1.0 (正規化座標)
    y: float

class CrackAnnotation(BaseModel):
    """ひび割れ：中心線上の点列（ポリライン）"""
    class_name: str  # "crack"
    points: List[Point]  # ひびの中心線を順番に追った点列

class AnnotationResult(BaseModel):
    cracks: List[CrackAnnotation]

# -------------------------
# 設定・定数
# -------------------------
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TILE_OVERLAP = 0.1  # タイル間の重なり（10%）

CLASSES = {0: "crack"}

def build_prompt(user_instruction: str) -> str:
    return f"""あなたは材料科学およびタイル外壁診断，建物外壁診断，コンクリート診断の専門家です。
画像内のひび割れ（crack）を検出し、以下の形式で出力してください。

## 座標系の定義
- 画像の左上を (x=0.0, y=0.0)、右下を (x=1.0, y=1.0) とします。
- 小数点以下3桁まで正確に出力してください。

## crack（ひび割れ）の出力形式
- ひびの【中心線上の点列】を `points` として出力してください。
- ひびを始点から終点まで順番に追いながら、曲がり角や分岐ごとに点を打ってください（1本のひびにつき5〜50点程度）。
- ポリゴンや面積ではなく、線（ポリライン）として表現します。
- 【重要】目地（タイル・石材の継ぎ目、規則的なグリッド直線）は絶対に crack としないこと。
- 汚れ・影・光沢・色むら・表面テクスチャは除外。物理的に材料が割れた不規則な亀裂のみを対象とする。

## アノテーション指示
{user_instruction}
"""


def polyline_to_polygon(
    points: List[Tuple[float, float]],
    half_w: float
) -> Optional[List[Tuple[float, float]]]:
    """
    ポリライン（点列）を太らせて閉じたポリゴンに変換する。

    各線分の左側オフセット点列と右側オフセット点列を結合して
    閉じたポリゴンを生成する（始点・終点は半円ではなく矩形で処理）。
    """
    if len(points) < 2:
        return None

    left_pts: List[Tuple[float, float]] = []
    right_pts: List[Tuple[float, float]] = []

    for i in range(len(points)):
        x, y = points[i]

        # 接線方向を前後の点から計算
        if i == 0:
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
        elif i == len(points) - 1:
            dx = points[-1][0] - points[-2][0]
            dy = points[-1][1] - points[-2][1]
        else:
            dx = points[i+1][0] - points[i-1][0]
            dy = points[i+1][1] - points[i-1][1]

        length = math.sqrt(dx*dx + dy*dy)
        if length < 1e-9:
            continue

        # 法線方向（左: +, 右: -）
        nx = -dy / length * half_w
        ny =  dx / length * half_w

        left_pts.append((x + nx, y + ny))
        right_pts.append((x - nx, y - ny))

    if len(left_pts) < 2:
        return None

    # 左側 → 右側（逆順）で閉じたポリゴンを作る
    polygon = left_pts + list(reversed(right_pts))
    return polygon


def make_tiles(image: Image.Image, grid_cols: int, grid_rows: int) -> List[Tuple[bytes, float, float, float, float]]:
    """画像をタイルに分割し、元画像に対する位置情報を返す"""
    W, H = image.size
    step_x = W / grid_cols
    step_y = H / grid_rows
    ovlp_x = step_x * TILE_OVERLAP
    ovlp_y = step_y * TILE_OVERLAP

    tiles = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x0 = max(0, int(col * step_x - ovlp_x))
            y0 = max(0, int(row * step_y - ovlp_y))
            x1 = min(W, int((col + 1) * step_x + ovlp_x))
            y1 = min(H, int((row + 1) * step_y + ovlp_y))

            tile = image.crop((x0, y0, x1, y1))
            buf = io.BytesIO()
            tile.save(buf, format="JPEG", quality=95)
            tiles.append((buf.getvalue(), x0/W, y0/H, (x1-x0)/W, (y1-y0)/H))
    return tiles

def process_images(args):
    client = genai.Client(api_key=args.api_key)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    half_w = args.crack_width / 2.0

    img_out = output_dir / "images" / args.split
    lbl_out = output_dir / "labels" / args.split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    print(f"{len(image_files)} 枚の画像を {args.model} で処理します (Tile: {args.tile_grid}, crack_width: {args.crack_width})")

    cols, rows = map(int, args.tile_grid.split('x'))

    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        with Image.open(img_path) as img:
            tiles = make_tiles(img, cols, rows)

        yolo_lines = []
        for idx, (tile_bytes, ox, oy, sx, sy) in enumerate(tiles):
            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=[
                        types.Part.from_bytes(data=tile_bytes, mime_type="image/jpeg"),
                        build_prompt(args.instruction)
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=AnnotationResult,
                    )
                )

                result = AnnotationResult.model_validate_json(response.text)

                # --- crack: ポリライン → 太らせてポリゴンに変換 ---
                for crack in result.cracks:
                    # タイル内座標 → 元画像座標に変換
                    global_pts = [
                        (ox + pt.x * sx, oy + pt.y * sy)
                        for pt in crack.points
                    ]
                    polygon = polyline_to_polygon(global_pts, half_w)
                    if polygon and len(polygon) >= 3:
                        coords = " ".join(
                            f"{max(0.0, min(1.0, x)):.6f} {max(0.0, min(1.0, y)):.6f}"
                            for x, y in polygon
                        )
                        yolo_lines.append(f"0 {coords}")

                print(f"  Tile {idx+1}/{len(tiles)}: {len(result.cracks)} cracks")
                time.sleep(0.5)  # Rate limit 対策

            except Exception as e:
                print(f"  Error in tile {idx}: {e}")

        (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(yolo_lines))
        shutil.copy2(img_path, img_out / img_path.name)

    yaml_content = f"path: {output_dir.resolve()}\ntrain: images/train\nval: images/val\nnc: {len(CLASSES)}\nnames: {list(CLASSES.values())}"
    (output_dir / "dataset.yaml").write_text(yaml_content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="images")
    parser.add_argument("--output", "-o", default="dataset")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", ""))
    parser.add_argument("--split", default="train")
    parser.add_argument("--tile-grid", default="2x2")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--crack-width", type=float, default=0.005,
                        help="ひび割れポリゴンの太さ（正規化座標、デフォルト: 0.005 = 画像幅の0.5%%）")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key または環境変数 GEMINI_API_KEY を設定してください。")

    process_images(args)

if __name__ == "__main__":
    main()
