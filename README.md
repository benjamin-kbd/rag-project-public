# RAG チャットシステム

BGE-M3 埋め込み + Qdrant ベクトルDB + BGE Reranker + Llama 3.1 ベースの RAG システム

---

## 技術スタック

| 項目 | 技術 |
|------|------|
| 言語 | Python 3.11 |
| API サーバー | FastAPI |
| 埋め込みモデル | HuggingFace BGE-M3 |
| ベクトルDB | Qdrant Cloud |
| 再ランキング | HuggingFace BGE Reranker |
| LLM | Llama 3.1 8B (HuggingFace + cerebras) |
| チャンキング | Semantic Chunking (Colab) / RecursiveCharacterTextSplitter (サーバー) |
| デプロイ | Render |
| 開発環境 | Google Colab |

---

## プロジェクト構成

```
rag-project/
├── app/
│   ├── config.py        # 環境変数設定 (pydantic-settings)
│   ├── main.py          # FastAPI サーバーエントリーポイント
│   ├── embedder.py      # BGE-M3 埋め込み (HuggingFace API)
│   ├── vectorstore.py   # Qdrant ベクトルDB 連携
│   ├── chunker.py       # テキストチャンキング (日本語・英語対応)
│   ├── reranker.py      # BGE Reranker 再ランキング
│   ├── llm.py           # LLM 回答生成 (Llama 3.1)
│   └── rag.py           # RAG パイプライン統合
├── ingest/
│   └── ingest.py        # Semantic Chunking + ドキュメントアップロード (Colab用)
├── static/
│   └── index.html       # チャット UI (HTML/JS)
├── .env.example         # 環境変数テンプレート
├── .gitignore
├── requirements.txt
├── render.yaml          # Render デプロイ設定
└── runtime.txt          # Python バージョン指定
```

---

## RAG パイプラインの流れ

```
ユーザーの質問
    ↓
[embedder.py]    BGE-M3 で質問を埋め込み (テキスト → 1024次元ベクトル)
    ↓
[vectorstore.py] Qdrant で類似文書の上位10件を検索 (コサイン類似度)
    ↓
[reranker.py]    BGE Reranker で精密な再ランキング → 上位3件を選択
    ↓
[llm.py]         Llama 3.1 8B でコンテキストに基づいた回答を生成
    ↓
[rag.py]         パイプライン全体を統合 → レスポンスを返す
```

---

## API エンドポイント

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/` | サーバー状態確認 |
| GET | `/health` | ヘルスチェック |
| GET | `/chat` | チャット UI |
| GET | `/docs` | API ドキュメント (Swagger) |
| POST | `/query` | 質問 → RAG 回答 |
| POST | `/ingest` | テキストチャンクをベクトルDBに保存 |
| POST | `/upload` | PDF/TXT ファイルアップロード |

### `/query` リクエスト例

```json
POST /query
{
  "question": "RAGとは何ですか？",
  "top_k": 5
}
```

```json
レスポンス
{
  "answer": "RAGとは検索と生成を組み合わせたAI技術です...",
  "question": "RAGとは何ですか？",
  "sources": [
    {
      "text": "RAG（Retrieval-Augmented Generation）は...",
      "score": 0.9231,
      "reranked": true
    }
  ]
}
```

### `/upload` リクエスト例

```bash
curl -X POST https://your-app.onrender.com/upload \
  -F "file=@document.pdf"
```

```json
レスポンス
{
  "message": "document.pdf アップロード完了",
  "chunks": 42,
  "text_length": 15823
}
```

---

## 環境変数の設定

`.env.example` をコピーして `.env` ファイルを作成します。

```bash
cp .env.example .env
```

```env
# HuggingFace（埋め込み・再ランキング）
HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxx
HF_EMBED_MODEL=BAAI/bge-m3

# Qdrant Cloud
QDRANT_URL=https://xxxx.us-east-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=rag_collection

# Groq（LLM・オプション）
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama-3.1-8b-instant
```

> `.env` ファイルは `.gitignore` に含まれているため、GitHub にはアップロードされません。

---

## API キーの取得方法

### 1. HuggingFace API キー

1. [huggingface.co](https://huggingface.co) にアカウント登録
2. Settings → Access Tokens → New token（Read 権限）
3. [Llama 3.1 モデルページ](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)でライセンスに同意

### 2. Qdrant Cloud

1. [cloud.qdrant.io](https://cloud.qdrant.io) にアカウント登録
2. Create Cluster（AWS、US East、無料プラン）
3. API Keys タブでキーを生成

### 3. Groq API（オプション）

1. [console.groq.com](https://console.groq.com) にアカウント登録
2. API Keys → Create API Key

---

## ローカル実行（Google Colab）

```python
# パッケージのインストール
!pip install -r requirements.txt

# サーバー起動
import subprocess
proc = subprocess.Popen(
    ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd="/content/drive/MyDrive/rag-project"
)
```

---

## ドキュメントのアップロード方法

### 方法 1. ブラウザ UI から

```
https://your-app.onrender.com/chat にアクセス
→ 右上の「文書アップロード」をクリック
→ PDF または TXT ファイルを選択
```

### 方法 2. Colab で Semantic Chunking を適用してアップロード

```python
!pip install langchain-huggingface langchain-experimental

import os
os.chdir("/content/drive/MyDrive/rag-project")

# ファイルパスを指定
!python ingest/ingest.py /content/document.pdf
```

### 方法 3. API を直接呼び出す

```python
import httpx

res = httpx.post(
    "https://your-app.onrender.com/upload",
    files={"file": open("document.pdf", "rb")},
    timeout=120,
)
print(res.json())
```

---

## 対応ファイル形式

| 形式 | エンコーディング |
|------|----------------|
| PDF | 自動抽出 |
| TXT | UTF-8、Shift-JIS、EUC-JP、CP932 を自動判定 |

---

## Render へのデプロイ

### 自動デプロイの設定

GitHub リポジトリを Render に接続すると、`main` ブランチへの push 時に自動デプロイされます。

### 環境変数（Render ダッシュボード → Environment）

| Key | 説明 |
|-----|------|
| `HF_API_KEY` | HuggingFace API キー |
| `QDRANT_URL` | Qdrant クラスター URL |
| `QDRANT_API_KEY` | Qdrant API キー |
| `GROQ_API_KEY` | Groq API キー（オプション） |
| `PYTHON_VERSION` | `3.11.9` |

### デプロイ確認

```bash
curl https://your-app.onrender.com/health
# {"status": "healthy"}
```

---

## 主な制限事項

| 項目 | 内容 |
|------|------|
| HuggingFace 無料枠 | 埋め込み・再ランキング専用。LLM は cerebras プロバイダー経由 |
| Qdrant 無料プラン | 1GB のストレージ制限 |
| Render 無料プラン | 月750時間、15分間無操作でスリープ |
| Groq 無料プラン | 1分あたり30リクエスト制限 |

---

## 開発環境

- Python 3.11.9
- Google Colab（開発・ドキュメントアップロード）
- GitHub（コード管理）
- Render（サーバーデプロイ）
