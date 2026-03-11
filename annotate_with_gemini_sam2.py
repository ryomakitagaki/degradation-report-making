import os
import cv2
import numpy as np
import csv
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image

# --- 1. 設定 ---
API_KEY  = os.environ.get("GEMINI_API_KEY", "")  # 環境変数 or 直接入力
MODEL_ID = "gemini-2.5-flash-image"

INPUT_DIR        = Path("images")    # 入力画像フォルダ
OUTPUT_DIR       = Path("dataset")   # 出力ルートフォルダ
SPLIT            = "train"           # train / val / test

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MIME_MAP = {
    ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".bmp": "image/bmp", ".webp": "image/webp",
}

# --- プロンプトの設定 ---
PROMPT_FOR_NANOBANANA = """
このコンクリート表面の画像を解析してください。
表面のすべてのひび割れ、特に微細なヘアクラックを網羅的に特定してください。
特定したすべてのひび割れの上に、鮮明な赤色の線（太さ2-3ピクセル）を上書きした画像を生成してください。
タイルの目地とひび割れを区別し、目地には赤線を引かないでください。
元の床の色と赤線のみで構成された画像を返してください。
"""


# --- 2. Gemini API 初期化 ---
client = genai.Client(api_key=API_KEY)


def get_image_bytes(image_path: str):
    """画像をバイト列に変換する"""
    with open(image_path, "rb") as f:
        return f.read()


def generate_traced_image(model_id: str, image_path: str, prompt: str):
    """Gemini API を呼び出してひび割れ強調画像を生成する"""
    print("Gemini API を呼び出しています...")

    image_bytes = get_image_bytes(image_path)
    mime_type = MIME_MAP.get(Path(image_path).suffix.lower(), "image/jpeg")

    response = client.models.generate_content(
        model=model_id,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    # レスポンスから画像パートを取得
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            print("強調画像が生成されました。")
            return part.inline_data.data

    print("画像の生成に失敗しました。（テキスト応答のみ）")
    return None


def extract_point_cloud(image_bytes: bytes, csv_path: Path, vis_path: Path):
    """生成された画像から赤色領域を検出し、点群を CSV・強調画像を保存する"""
    print("  点群を抽出しています...")

    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # 赤色領域の検出（BGR形式）
    lower_red = np.array([0,   0,   150])
    upper_red = np.array([50,  50,  255])
    mask = cv2.inRange(img, lower_red, upper_red)

    # マスクされたピクセル座標を取得 (y, x)
    points = np.column_stack(np.where(mask > 0)) if mask.any() else []

    # CSV に保存
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y", "x"])
        if len(points):
            writer.writerows(points)

    # 強調画像を保存
    cv2.imwrite(str(vis_path), img)
    print(f"  → 点群 {len(points)} 点を保存: {csv_path.name}")
    print(f"  → 強調画像を保存: {vis_path.name}")


def csv_to_yolo(
    csv_path: str,
    image_path: str,
    output_label_path: str,
    class_id: int = 0,
    min_area_px: int = 10,
    approx_epsilon_ratio: float = 0.002,
):
    """
    点群CSVを読み込み、輪郭抽出してYOLOセグメンテーション形式で保存する。

    CSVフォーマット: ヘッダー行(y,x) + ピクセル座標行
    YOLOフォーマット: <class_id> <x1> <y1> <x2> <y2> ... (正規化座標)
    """
    print(f"CSV → YOLO 変換: {csv_path}")

    # 元画像サイズを取得
    with Image.open(image_path) as img:
        W, H = img.size

    # CSV から点群を読み込む
    points = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append((int(row["y"]), int(row["x"])))

    if not points:
        print("  点群が空です。YOLOラベルは生成されません。")
        Path(output_label_path).write_text("")
        return

    # 点群をマスク画像に変換
    mask = np.zeros((H, W), dtype=np.uint8)
    for y, x in points:
        if 0 <= y < H and 0 <= x < W:
            mask[y, x] = 255

    # モルフォロジー処理（点を繋いでひび割れ領域を閉じる）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # 輪郭抽出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_px:
            continue
        # ダグラス-ポイカー近似
        epsilon = approx_epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue

        coords = " ".join(
            f"{max(0.0, min(1.0, pt[0][0] / W)):.6f} {max(0.0, min(1.0, pt[0][1] / H)):.6f}"
            for pt in approx
        )
        yolo_lines.append(f"{class_id} {coords}")

    Path(output_label_path).write_text("\n".join(yolo_lines))
    print(f"  {len(yolo_lines)} 個のポリゴンを検出")
    print(f"  YOLOラベルを保存: {output_label_path}")


# --- メイン処理 ---
if __name__ == "__main__":
    if not API_KEY:
        raise ValueError("API_KEY が設定されていません。環境変数 GEMINI_API_KEY を設定してください。")

    # 出力フォルダを作成
    img_out = OUTPUT_DIR / "images" / SPLIT
    lbl_out = OUTPUT_DIR / "labels" / SPLIT
    vis_out = OUTPUT_DIR / "visualized"
    csv_out = OUTPUT_DIR / "pointclouds"
    for d in [img_out, lbl_out, vis_out, csv_out]:
        d.mkdir(parents=True, exist_ok=True)

    # 入力画像一覧
    image_files = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_files:
        raise FileNotFoundError(f"画像が見つかりません: {INPUT_DIR}")

    print(f"\n{len(image_files)} 枚の画像を処理します (model: {MODEL_ID})")
    print(f"出力先: {OUTPUT_DIR}\n")

    success, errors = 0, 0

    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {img_path.name}")
        try:
            # Step 1: Gemini で強調画像を生成
            traced_bytes = generate_traced_image(MODEL_ID, str(img_path), PROMPT_FOR_NANOBANANA)

            if traced_bytes is None:
                print("  強調画像の生成に失敗しました。スキップします。")
                errors += 1
                continue

            # Step 2: 赤色検出 → CSV・強調画像を保存
            csv_path = csv_out / (img_path.stem + ".csv")
            vis_path = vis_out / (img_path.stem + "_visualized.jpg")
            extract_point_cloud(traced_bytes, csv_path, vis_path)

            # Step 3: CSV → YOLO ラベル変換
            label_path = lbl_out / (img_path.stem + ".txt")
            csv_to_yolo(
                csv_path=str(csv_path),
                image_path=str(img_path),
                output_label_path=str(label_path),
            )

            # 元画像をコピー
            import shutil
            shutil.copy2(img_path, img_out / img_path.name)

            success += 1

        except Exception as e:
            print(f"  エラー: {e}")
            errors += 1

    # dataset.yaml を保存
    yaml_content = (
        f"path: {OUTPUT_DIR.resolve()}\n"
        f"train: images/train\nval: images/val\n"
        f"nc: 1\nnames: ['crack']"
    )
    (OUTPUT_DIR / "dataset.yaml").write_text(yaml_content)

    print(f"\n完了: 成功={success}, エラー={errors}")
    print(f"データセット: {OUTPUT_DIR}")
