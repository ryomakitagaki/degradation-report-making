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

class FlakingAnnotation(BaseModel):
    """剥落：輪郭ポリゴン"""
    class_name: str  # "flaking"
    polygon: List[Point]  # 剥落領域の輪郭点列

class AnnotationResult(BaseModel):
    cracks: List[CrackAnnotation]
    flakings: List[FlakingAnnotation]

# -------------------------
# 設定・定数
# -------------------------
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TILE_OVERLAP = 0.1  # タイル間の重なり（10%）

CLASSES = {0: "crack", 1: "flaking"}

def build_prompt(user_instruction: str) -> str:
    return f"""あなたは材料科学およびタイル外壁診断，建物外壁診断，コンクリート診断の専門家です。
画像内の劣化箇所（crack・flaking）を検出し、以下の形式で出力してください。

## 座標系の定義
- 画像の左上を (x=0.0, y=0.0)、右下を (x=1.0, y=1.0) とします。
- 小数点以下3桁まで正確に出力してください。

## crack（ひび割れ）の出力形式
- ひびの【中心線上の点列】を `points` として出力してください。
- ひびを始点から終点まで順番に追いながら、曲がり角や分岐ごとに点を打ってください（1本のひびにつき5〜30点程度）。
- ポリゴンや面積ではなく、線（ポリライン）として表現します。
- 【重要】目地（タイル・石材の継ぎ目、規則的なグリッド直線）は絶対に crack としないこと。
- 汚れ・影・光沢・色むら・表面テクスチャは除外。物理的に材料が割れた不規則な亀裂のみを対象とする。

## flaking（剥落）の出力形式
- 剥落領域の輪郭を `polygon` として出力してください（6〜20点程度）。
- 【重要】光沢・影・色むら・汚れは flaking としないこと。実際に材料が物理的に欠損・剥落している箇所のみを対象とする。
- 確信が持てない場合は出力しないでください（過検出より見逃しのほうが望ましい）。

## アノテーション指示
{user_instruction}
"""

def segment_to_rect(
    x1: float, y1: float,
    x2: float, y2: float,
    half_w: float
) -> Optional[List[Tuple[float, float]]]:
    """2点を中心線とした細長い四角形の4頂点を返す"""
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-9:
        return None
    nx = -dy / length * half_w
    ny =  dx / length * half_w
    return [
        (x1 + nx, y1 + ny),
        (x1 - nx, y1 - ny),
        (x2 - nx, y2 - ny),
        (x2 + nx, y2 + ny),
    ]

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

    img_out = output_dir / "images" / args.split
    lbl_out = output_dir / "labels" / args.split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    print(f"{len(image_files)} 枚の画像を Gemini 2.5 Pro で処理します (Tile: {args.tile_grid})")

    cols, rows = map(int, args.tile_grid.split('x'))
    half_w = args.crack_width / 2.0

    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        with Image.open(img_path) as img:
            tiles = make_tiles(img, cols, rows)

        yolo_lines = []
        for idx, (tile_bytes, ox, oy, sx, sy) in enumerate(tiles):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
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

                # --- crack: ポリライン → 線分ごとに細長い四角形へ変換 ---
                for crack in result.cracks:
                    pts = crack.points
                    for i in range(len(pts) - 1):
                        gx1 = ox + pts[i].x * sx
                        gy1 = oy + pts[i].y * sy
                        gx2 = ox + pts[i+1].x * sx
                        gy2 = oy + pts[i+1].y * sy
                        rect = segment_to_rect(gx1, gy1, gx2, gy2, half_w)
                        if rect:
                            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in rect)
                            yolo_lines.append(f"0 {coords}")

                # --- flaking: 通常のポリゴン ---
                for flk in result.flakings:
                    coords = []
                    for pt in flk.polygon:
                        gx = ox + pt.x * sx
                        gy = oy + pt.y * sy
                        coords.extend([f"{gx:.6f}", f"{gy:.6f}"])
                    if len(coords) >= 6:
                        yolo_lines.append("1 " + " ".join(coords))

                print(f"  Tile {idx+1}/{len(tiles)}: {len(result.cracks)} cracks, {len(result.flakings)} flakings")
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
    parser.add_argument("--crack-width", type=float, default=0.005,
                        help="crack ポリゴンの太さ（正規化座標、デフォルト: 0.005 = 画像幅の0.5%%）")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key または環境変数 GEMINI_API_KEY を設定してください。")

    process_images(args)

if __name__ == "__main__":
    main()
