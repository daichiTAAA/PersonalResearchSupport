# PostgreSQL pgvector RAG MCP Server

このリポジトリは、PostgreSQL と pgvector を利用して Retrieval-Augmented Generation (RAG) を実装するMCP (Model Context Protocol) サーバーを提供します。Claude や他のMCPクライアントと連携して、PDFやテキストドキュメントを利用したRAG機能を実現します。

## 機能

- PostgreSQL + pgvector を使用したベクトルデータベースでのドキュメント保存と検索
- PDF、TXT、MD、CSVなど様々な形式のドキュメントの処理
- ドキュメントのチャンキングと埋め込み生成
- 意味検索と質問応答
- MCP (Model Context Protocol) 準拠のAPIインターフェース

## セットアップ

### 前提条件

- Docker と Docker Compose
- Python 3.9+
- OpenAI API キー

### インストール手順

1. リポジトリをクローン
```bash
git clone <this-repository-url>
cd path/to/your/repository
cd src/backend/postgre
```

2. 環境変数の設定
```bash
cp .env.example .env
# .envファイルを編集して、OpenAI APIキーなどを設定
```

3. Docker Composeでデータベースを起動
```bash
docker compose up -d
```

4. 必要なPythonパッケージをインストール
```bash
uv sync
```

### MCP サーバーの起動

```bash
uvx run server.py
```

## 使用方法

### Claude Desktop での設定

1. Claude Desktop の設定ファイル (`claude_desktop_config.json`) を編集して、このMCPサーバーを追加します：

```json
{
  "mcpServers": {
    "pgvector_rag": {
      "command": "uvx",
      "args": ["run", "/path/to/server.py"]
    }
  }
}
```

2. Claude Desktop を再起動します。

### MCP サーバーの機能

このMCPサーバーは、以下の主要な機能を提供します：

#### ドキュメントの処理

```
process_document(filepath, chunk_size=1000, chunk_overlap=200, metadata=None)
```

指定されたファイルを読み取り、テキストを抽出し、チャンク分割、埋め込み生成を行い、データベースに保存します。

#### ドキュメントの検索

```
search_similar_documents(query, limit=5)
```

クエリテキストに意味的に似ているドキュメントを検索します。

#### RAG質問応答

```
ask_rag(question, limit=5)
```

質問に対してRAGを使用して回答します。関連するコンテキストが提供されます。

#### その他の機能

- `list_documents()`: データベース内のドキュメント一覧を取得
- `delete_document(document_name)`: 名前でドキュメントを削除

#### Web UIの実行方法
Web UIを使用して、ドキュメントのアップロードや質問応答を行うことができます。
```bash
uvx run app.py
```
Webブラウザで `http://localhost:8080` にアクセスします。

#### 補足

このコードセットには以下の要素が含まれています：

1. **Docker Compose 設定** (`docker-compose.yml`)
   - pgvector拡張機能が有効化されたPostgreSQLコンテナを設定
   - データの永続化のためのボリュームを設定

2. **MCP Server 実装** (`server.py`)
   - FastMCPを使用したMCPプロトコル対応サーバー
   - PDF、テキストファイルなどのドキュメント処理
   - テキストの分割と埋め込み生成
   - PostgreSQLとpgvectorを使用した意味検索
   - RAGベースの質問応答機能

3. **Web UI** (`app.py` と `templates/index.html`)
   - ドキュメントのアップロードと管理
   - 質問応答インターフェース
   - 検索結果の視覚化