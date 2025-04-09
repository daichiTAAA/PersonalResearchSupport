# PostgreSQLをMCP Server経由でRAGに活用する方法

PostgreSQLをMCP Server経由でRAG(Retrieval-Augmented Generation)に活用する方法をまとめました。この方法を使用することで、ClaudeなどのLLMに専門知識や特定の情報を提供し、情報検索能力を向上させることができます。

## 1. 概要

MCP(Model Context Protocol)はAnthropicが開発したオープンプロトコルで、LLMと外部ツールやデータソースを標準化された方法で接続することができます。PostgreSQLとPGVectorの組み合わせにより、ベクトル検索機能を持つデータベースを構築し、RAGシステムのバックエンドとして使用できます。

## 2. 必要な技術スタック

- **PostgreSQL**: ベースとなるリレーショナルデータベース
- **pgvector**: PostgreSQLの拡張機能で、ベクトル検索機能を提供
- **Python**: MCP Serverの実装に使用
- **SQLAlchemy**: Pythonでのデータベース操作(ORM)
- **OpenAI埋め込みモデル**: テキストをベクトル化するための埋め込みモデル
- **Claude Desktop**: MCP Serverと接続して使用するクライアント

## 3. インストール手順

### 3.1 PostgreSQLとpgvectorのセットアップ

PostgreSQLに`pgvector`拡張機能をインストールする方法はいくつかあります：

#### Dockerを使用する方法

```bash
# pgvector拡張機能を含むPostgreSQLコンテナを実行
docker run --name postgres-vector -e POSTGRES_USER=admin -e POSTGRES_PASSWORD=admin_password -e POSTGRES_DB=vectordb -p 5432:5432 -d pgvector/pgvector:pg16
```

#### 手動インストール（Linuxの場合）

```bash
# pgvector拡張機能をインストール
cd /tmp
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# PostgreSQLに接続して拡張機能を有効化
psql -U postgres
CREATE EXTENSION vector;
```

### 3.2 必要なPythonパッケージのインストール

```bash
pip install fastmcp sqlalchemy pgvector openai "psycopg[binary]" pydantic python-dotenv
```

## 4. MCPサーバーの実装

以下は基本的なMCP Serverの実装例です。

```python
# server.py
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from openai import OpenAI

# 環境変数のロード
load_dotenv()

# PostgreSQLの接続設定
DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

# SQLAlchemyの設定
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# OpenAIクライアントの初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ベクトル埋め込みの次元数
EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-large の場合

# ドキュメントモデルの定義
class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255))
    content = Column(Text)
    embedding = Column(Vector(EMBEDDING_DIM))

# テーブル作成
Base.metadata.create_all(engine)

# MCP Serverの初期化
app = FastMCP()

# テキストを埋め込みベクトルに変換する関数
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# ドキュメント追加ツール
@app.tool
def add_document(title: str, content: str) -> dict:
    """
    Add a document to the RAG database.
    
    Args:
        title: The title of the document
        content: The text content of the document
    
    Returns:
        A confirmation message with the document ID
    """
    embedding = get_embedding(content)
    
    with Session() as session:
        document = Document(
            title=title,
            content=content,
            embedding=embedding
        )
        session.add(document)
        session.commit()
        return {"status": "success", "document_id": document.id}

# ドキュメント検索ツール
@app.tool
def search_documents(query: str, limit: int = 5) -> list:
    """
    Search for documents similar to the query text.
    
    Args:
        query: The search query text
        limit: Maximum number of results to return (default: 5)
    
    Returns:
        A list of matching documents with their content and similarity score
    """
    query_embedding = get_embedding(query)
    
    with Session() as session:
        # コサイン類似度を使用してドキュメントを検索
        results = session.query(
            Document.id,
            Document.title,
            Document.content,
            Document.embedding.cosine_distance(query_embedding).label('distance')
        ).order_by('distance').limit(limit).all()
        
        return [
            {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
                "similarity": 1 - doc.distance
            }
            for doc in results
        ]

# サーバー起動コード
if __name__ == "__main__":
    # サーバーの起動
    app.run()
```

## 5. Claude Desktopとの連携設定

Claude Desktopと連携するには、設定ファイル`claude_desktop_config.json`を編集します。

```json
{
  "mcpServers": {
    "rag_postgres": {
      "command": "python",
      "args": ["path/to/your/server.py"]
    }
  }
}
```

## 6. RAGの利用例

Claude Desktopを使って以下のような機能を実行できます：

1. **ドキュメントの追加**:
    
    ```
    ドキュメント「PostgreSQLの基本操作」を追加してください。内容は「PostgreSQLはオープンソースのリレーショナルデータベース管理システムです。主な特徴として...」
    ```
    
2. **知識ベースへの質問**:
    
    ```
    PostgreSQLの主な特徴について教えてください。
    ```
    
3. **複雑な検索と要約**:
    
    ```
    「ベクトル検索」に関する保存済みドキュメントを検索し、要約してください。
    ```
    

## 7. 拡張と最適化のヒント

1. **インデックス作成**: 大量のベクトルデータを扱う場合、パフォーマンス向上のためにIVFFlat、HNSW、HNSWを使ったインデックスを作成します。
    
    ```sql
    CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
    ```
    
2. **チャンキング**: 長いテキストを適切なサイズに分割して保存することで、より関連性の高い結果を得られます。
    
3. **並列処理**: pgvector 0.5.1以降では、HNSW/IVFFlatインデックスのパラレルビルドがサポートされており、インデックス構築が最大30倍高速化されています。
    
4. **メタデータの活用**: PostgreSQLのリレーショナル機能を活かして、ベクトル検索結果とメタデータを組み合わせたより高度な検索クエリを実装できます。
    

## 8. 課題と解決策

1. **埋め込みの品質**: 高品質な埋め込みを生成するために、適切なモデル（text-embedding-3-large等）を選択することが重要です。
    
2. **セキュリティ**: MCP Server経由でのデータベースアクセスには適切な認証と権限設定を行いましょう。
    
3. **パフォーマンス**: 大規模データセットでは、インデックスの最適化と適切なハードウェアリソースの割り当てが必要です。
    

## 9. まとめ

PostgreSQLとpgvectorをMCP Server経由でRAGシステムに活用することで、ClaudeなどのLLMに特定の知識ベースへのアクセスを提供できます。この仕組みは個人研究のサポートにも非常に有効で、文献や研究ノートの管理と活用に役立ちます。実装は比較的簡単で、基本的なPythonとSQLの知識があれば構築可能です。

---

この情報が個人研究サポートアプリケーションの開発に役立つことを願っています。PostgreSQLとpgvectorは強力なベクトル検索機能を持ちながら、既存のリレーショナルデータベースの利点も活かせる優れた選択肢です。