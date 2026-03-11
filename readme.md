
# annotate_with_gemini_sam2.py の使い方

Gemini API で劣化箇所の点座標を検出し、SAM 2 で精密なセグメンテーションを行い、YOLO セグメンテーション形式で保存するスクリプト。

## 処理フロー

1. Gemini API → 劣化箇所のプロンプト点（中心付近の座標）を出力
2. SAM 2（ローカル）→ 点をプロンプトとして精密マスクを生成
3. マスク → ポリゴン変換（OpenCV findContours）
4. YOLO セグメンテーション形式で保存

## 事前準備

### 1. パッケージのインストール

```bash
pip install -r requirements_annotate.txt
```

### 2. API キーの設定

ターミナルで以下を実行しておく（セッションごとに必要）：

```bash
export GEMINI_API_KEY="自分のAPIキー"
```

## 基本的な使い方

```bash
python annotate_with_gemini_sam2.py \
  --classes <クラス名> \
  --prompt-file <プロンプトファイル> \
  --input <入力画像フォルダ> \
  --output <出力フォルダ>
```

## オプション一覧

| オプション | 短縮形 | デフォルト | 説明 |
|---|---|---|---|
| `--input` | `-i` | `images` | 入力画像フォルダ |
| `--output` | `-o` | `dataset` | 出力データセットフォルダ |
| `--classes` | `-c` | （必須） | クラス名（複数指定可） |
| `--prompt-file` | `-p` | — | アノテーション指示テキストファイル |
| `--instruction` | — | — | アノテーション指示テキスト（直接入力） |
| `--api-key` | — | 環境変数から取得 | Gemini API キー |
| `--gemini-model` | — | `gemini-2.5-flash` | Gemini モデル名 |
| `--sam2-model` | — | `tiny` | SAM 2 モデルサイズ（tiny / small / base_plus / large） |
| `--split` | — | `train` | データセット分割（train / val / test） |
| `--dry-run` | — | — | API 呼び出しなしで動作確認のみ |

## 使用例

### ひび割れのアノテーション

```bash
python annotate_with_gemini_sam2.py \
  --classes crack \
  --prompt-file prompts/crack.txt \
  --input images \
  --output dataset
```

### 複数クラスのアノテーション

```bash
python annotate_with_gemini_sam2.py \
  --classes crack spalling stain \
  --prompt-file prompts/multi_defect.txt \
  --input images \
  --output dataset
```

### SAM 2 モデルを large に変更して精度を上げる

```bash
python annotate_with_gemini_sam2.py \
  --classes crack \
  --prompt-file prompts/crack.txt \
  --sam2-model large
```

### 動作確認（API 呼び出しなし）

```bash
python annotate_with_gemini_sam2.py \
  --classes crack \
  --prompt-file prompts/crack.txt \
  --dry-run
```

## 出力

```
dataset/
├── images/
│   └── train/       # 入力画像のコピー
├── labels/
│   └── train/       # YOLO セグメンテーション形式のアノテーション (.txt)
└── dataset.yaml     # YOLO 用データセット設定ファイル
```

## プロンプトファイル

`prompts/` フォルダにアノテーション指示テキストを置く。

- `prompts/crack.txt` — ひび割れ用
- `prompts/multi_defect.txt` — 複数劣化クラス用
