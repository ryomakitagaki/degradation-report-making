
# 建築物劣化アノテーション自動生成ツール

Gemini の画像生成 API を使って建築物のひび割れを検出し、YOLO セグメンテーション形式のアノテーションを自動生成するツール。

## 処理フロー

```
入力画像
  │
  ▼
[1] Gemini API（gemini-2.5-flash-image）
    ひび割れを赤線で上書きした画像を生成
  │
  ▼
[2] 赤色領域検出（OpenCV）
    赤色ピクセル座標を点群 CSV として保存
  │
  ▼
[3] 点群 → YOLO ラベル変換
    モルフォロジー処理 → 輪郭抽出 → 正規化座標
  │
  ▼
dataset/ （YOLOデータセット）
```

## ファイル構成

```
.
├── annotate2_with_gemini.py    # メインスクリプト（アノテーション生成）
├── visualize_annotations.py    # アノテーション可視化スクリプト
├── requirements_annotate.txt   # 依存パッケージ
├── images/                     # 入力画像フォルダ
├── _original_images/           # 元画像のバックアップ
└── dataset/                    # 出力データセット
    ├── images/train/           # 画像（YOLOデータセット用）
    ├── labels/train/           # YOLOラベル (.txt)
    ├── pointclouds/            # 点群 CSV（中間ファイル）
    ├── visualized/             # Gemini 生成の強調画像
    └── dataset.yaml            # YOLO 用データセット設定
```

## 事前準備

### 1. パッケージのインストール

```bash
pip install -r requirements_annotate.txt
```

### 2. API キーの設定

```bash
export GEMINI_API_KEY="自分のAPIキー"
```

## 使い方

### アノテーション生成

スクリプト冒頭の設定を確認・編集してから実行：

```bash
python annotate2_with_gemini.py
```

**主な設定項目（スクリプト内）：**

| 変数 | デフォルト | 説明 |
|---|---|---|
| `MODEL_ID` | `gemini-2.5-flash-image` | 使用する Gemini モデル |
| `INPUT_DIR` | `images` | 入力画像フォルダ |
| `OUTPUT_DIR` | `dataset` | 出力データセットフォルダ |
| `SPLIT` | `train` | データセット分割 |
| `PROMPT_FOR_NANOBANANA` | `V1`（強め） | 使用するプロンプト |

**プロンプトの切り替え：**
- `PROMPT_FOR_NANOBANANA_V1` — ヘアクラックも網羅的に検出（強め）
- `PROMPT_FOR_NANOBANANA_V2` — 明確なひび割れのみ検出（弱め）

### アノテーション可視化

生成されたアノテーションを元画像に重ねて確認：

```bash
python visualize_annotations.py
```

**オプション：**

| オプション | デフォルト | 説明 |
|---|---|---|
| `--dataset` | `dataset` | データセットフォルダ |
| `--output` | `visualized` | 出力フォルダ |
| `--split` | `train` | 処理する split |
| `--alpha` | `0.35` | 塗りつぶしの透明度（0.0〜1.0） |

```bash
python visualize_annotations.py --dataset dataset --output visualized --alpha 0.4
```

## 出力フォーマット

**YOLO セグメンテーション形式：**
```
<class_id> <x1> <y1> <x2> <y2> ...
```
座標は画像サイズで正規化（0.0〜1.0）。

**dataset.yaml：**
```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 1
names: ['crack']
```
