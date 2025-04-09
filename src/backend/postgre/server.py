#!/usr/bin/env python

import os
import asyncio
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
from fastmcp import FastMCP
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from pypdf import PdfReader
import chardet

# 環境変数のロード
load_dotenv()

# データベース接続情報
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "vectordb")
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# OpenAI設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# OpenAIクライアントの初期化
client = OpenAI(api_key=OPENAI_API_KEY)

# MCPサーバーの初期化
app = FastMCP()


# データベース接続関数
def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME
    )
    return conn


# データベーステーブルの初期化
def init_database():
    conn = get_db_connection()
    cur = conn.cursor()

    # pgvector拡張機能が存在しない場合は作成
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # ドキュメントテーブルの作成
    cur.execute(
        f"""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        name TEXT,
        content TEXT,
        metadata JSONB,
        embedding vector({EMBEDDING_DIMENSIONS})
    );
    """
    )

    # インデックス作成（ANN検索用）
    try:
        cur.execute(
            f"""
        CREATE INDEX IF NOT EXISTS documents_embedding_idx 
        ON documents 
        USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100);
        """
        )
    except:
        # PostgreSQLの低バージョンでエラーが発生するため、
        # 通常のインデックスを作成
        conn.rollback()
        cur.execute(
            f"""
        CREATE INDEX IF NOT EXISTS documents_embedding_idx 
        ON documents 
        USING vector_cosine_ops (embedding);
        """
        )

    conn.commit()
    cur.close()
    conn.close()


# データベーステーブルを削除して再作成する関数
def reset_database():
    """
    Drop and recreate the documents table and its associated index.
    This will delete all stored documents and embeddings.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # ドキュメントテーブルの削除
        cur.execute("DROP TABLE IF EXISTS documents;")

        # pgvector拡張機能が存在しない場合は作成
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # テーブルの再作成
        cur.execute(
            f"""
        CREATE TABLE documents (
            id SERIAL PRIMARY KEY,
            name TEXT,
            content TEXT,
            metadata JSONB,
            embedding vector({EMBEDDING_DIMENSIONS})
        );
        """
        )

        # インデックス作成（ANN検索用）
        try:
            # 新しいバージョン用の構文を試行
            cur.execute(
                f"""
            CREATE INDEX documents_embedding_idx 
            ON documents 
            USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
            """
            )
        except Exception as index_error:
            # インデックス作成のエラーをロールバック
            conn.rollback()
            try:
                # 代替方法1: 単純なINDEX作成
                cur.execute(
                    f"""
                CREATE INDEX documents_embedding_idx 
                ON documents 
                USING vector_ip (embedding);
                """
                )
            except Exception:
                conn.rollback()
                # 代替方法2: オペレータクラスを明示的に指定せずインデックス作成
                try:
                    cur.execute(
                        f"""
                    CREATE INDEX documents_embedding_idx 
                    ON documents (embedding);
                    """
                    )
                except Exception:
                    conn.rollback()
                    # それでもダメな場合はインデックスなしで続行
                    pass

        conn.commit()
        return {"status": "success", "message": "データベースを初期化しました"}
    except Exception as e:
        conn.rollback()
        return {
            "status": "error",
            "message": f"データベースの初期化中にエラーが発生しました: {str(e)}",
        }
    finally:
        cur.close()
        conn.close()


# テーブルデータの全件取得
def get_all_documents(limit=1000, offset=0):
    """
    Get all documents from the database with pagination.

    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip

    Returns:
        List of documents
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # ドキュメント数の取得
        cur.execute("SELECT COUNT(*) FROM documents")
        total_count = cur.fetchone()[0]

        # ドキュメントの取得（ページネーション付き）
        cur.execute(
            """
            SELECT id, name, content, metadata
            FROM documents
            ORDER BY id
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        )

        results = cur.fetchall()
        documents = []

        for doc_id, name, content, metadata in results:
            documents.append(
                {"id": doc_id, "name": name, "content": content, "metadata": metadata}
            )

        cur.close()
        conn.close()

        return {"status": "success", "total": total_count, "documents": documents}
    except Exception as e:
        return {
            "status": "error",
            "message": f"ドキュメントの取得中にエラーが発生しました: {str(e)}",
        }


# ファイルを読み込んでテキストを抽出する関数
def extract_text_from_file(file_path):
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".pdf":
        # PDFからのテキスト抽出
        pdf = PdfReader(file_path)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        return text

    elif file_path.suffix.lower() in [".txt", ".md", ".csv", ".json"]:
        # テキストファイルからの抽出
        # エンコーディングを自動検出
        with open(file_path, "rb") as f:
            raw_data = f.read()
            detected = chardet.detect(raw_data)
            encoding = detected["encoding"] if detected["encoding"] else "utf-8"

        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # fallback to UTF-8
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    else:
        # サポート外のファイル形式
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


# テキストを埋め込みベクトルに変換する関数
def get_embedding(text):
    text = text.replace("\n", " ")  # 改行を空白に置換
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


# テキストを分割する関数
def split_text(text, chunk_size=1000, chunk_overlap=200):
    encoding = tiktoken.get_encoding("cl100k_base")

    # テキスト分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(encoding.encode(x)),
    )

    return text_splitter.split_text(text)


# ドキュメント処理：ファイルからテキスト抽出、分割、埋め込み
@app.tool()
def process_document(
    filepath: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: dict = None,
) -> dict:
    """
    Process a document file (PDF, TXT, MD, etc.), split it into chunks, and store in the vector database.

    Args:
        filepath: Path to the document file
        chunk_size: Size of text chunks (tokens)
        chunk_overlap: Overlap between chunks (tokens)
        metadata: Additional metadata for the document

    Returns:
        Information about the processed document
    """
    try:
        # ファイルの存在確認
        if not os.path.exists(filepath):
            return {"status": "error", "message": f"File not found: {filepath}"}

        # テキスト抽出
        text = extract_text_from_file(filepath)
        if not text or text.strip() == "":
            return {"status": "error", "message": "No text content extracted from file"}

        # テキストの分割
        chunks = split_text(text, chunk_size, chunk_overlap)
        if not chunks:
            return {"status": "error", "message": "Failed to split text into chunks"}

        # メタデータ準備
        filename = os.path.basename(filepath)
        doc_metadata = {
            "source": filename,
            "filepath": filepath,
            "type": os.path.splitext(filename)[1].lower()[1:],
        }

        # ユーザー定義のメタデータがあれば追加
        if metadata:
            doc_metadata.update(metadata)

        # ドキュメントチャンクの処理と保存
        conn = get_db_connection()
        cur = conn.cursor()

        inserted_count = 0
        for i, chunk in enumerate(chunks):
            # チャンクごとのメタデータ
            chunk_metadata = doc_metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunk_id"] = str(uuid.uuid4())

            # 埋め込みの取得
            embedding = get_embedding(chunk)

            # データベースに保存
            cur.execute(
                """
                INSERT INTO documents (name, content, metadata, embedding) 
                VALUES (%s, %s, %s, %s)
                """,
                (filename, chunk, json.dumps(chunk_metadata), embedding),
            )
            inserted_count += 1

        conn.commit()
        cur.close()
        conn.close()

        return {
            "status": "success",
            "message": f"Document processed and stored successfully",
            "chunks_created": inserted_count,
            "document_name": filename,
        }

    except Exception as e:
        return {"status": "error", "message": f"Error processing document: {str(e)}"}


# 類似ドキュメントの検索
@app.tool()
def search_similar_documents(query: str, limit: int = 5) -> list:
    """
    Search for documents similar to the query text.

    Args:
        query: The search query text
        limit: Maximum number of results to return (default: 5)

    Returns:
        A list of matching documents with their content and similarity score
    """
    try:
        # クエリの埋め込みを取得
        query_embedding = get_embedding(query)

        # 類似度検索の実行
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, name, content, metadata, 
                   1 - (embedding <=> %s) as similarity
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, query_embedding, limit),
        )

        results = cur.fetchall()
        cur.close()
        conn.close()

        # 検索結果の整形
        documents = []
        for doc_id, name, content, metadata, similarity in results:
            documents.append(
                {
                    "id": doc_id,
                    "name": name,
                    "content": content,
                    "metadata": metadata,
                    "similarity": similarity,
                }
            )

        return documents

    except Exception as e:
        return {"status": "error", "message": f"Error searching documents: {str(e)}"}


# ドキュメントの削除
@app.tool()
def delete_document(document_name: str) -> dict:
    """
    Delete documents by name.

    Args:
        document_name: Name of the document to delete

    Returns:
        Confirmation of deletion
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            "DELETE FROM documents WHERE name = %s RETURNING id", (document_name,)
        )

        deleted_ids = cur.fetchall()
        deleted_count = len(deleted_ids)

        conn.commit()
        cur.close()
        conn.close()

        if deleted_count == 0:
            return {
                "status": "warning",
                "message": f"No documents found with name: {document_name}",
            }

        return {
            "status": "success",
            "message": f"Successfully deleted {deleted_count} document chunks with name: {document_name}",
        }

    except Exception as e:
        return {"status": "error", "message": f"Error deleting document: {str(e)}"}


# ドキュメントの一覧取得
@app.tool()
def list_documents() -> list:
    """
    List all documents in the database grouped by name.

    Returns:
        A list of document names and their chunk counts
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT name, COUNT(*) as chunk_count, 
                   MIN(metadata::json->>'source') as source,
                   MIN(metadata::json->>'type') as type
            FROM documents 
            GROUP BY name 
            ORDER BY name
            """
        )

        results = cur.fetchall()
        cur.close()
        conn.close()

        documents = []
        for name, chunk_count, source, doc_type in results:
            documents.append(
                {
                    "name": name,
                    "chunk_count": chunk_count,
                    "source": source,
                    "type": doc_type,
                }
            )

        return documents

    except Exception as e:
        return {"status": "error", "message": f"Error listing documents: {str(e)}"}


# RAG質問応答
@app.tool()
def ask_rag(question: str, limit: int = 5) -> dict:
    """
    Ask a question and get an answer using RAG (Retrieval-Augmented Generation).

    Args:
        question: The question to ask
        limit: Maximum number of documents to retrieve

    Returns:
        Answer with source references
    """
    try:
        # 類似ドキュメントの検索
        documents = search_similar_documents(question, limit)

        if not documents or len(documents) == 0:
            return {
                "answer": "I don't have enough context to answer this question.",
                "sources": [],
            }

        # コンテキストの準備
        context = "\n\n".join(
            [f"Document: {doc['name']}\nContent: {doc['content']}" for doc in documents]
        )

        # ソース情報の準備
        sources = []
        for doc in documents:
            metadata = (
                json.loads(doc["metadata"])
                if isinstance(doc["metadata"], str)
                else doc["metadata"]
            )
            sources.append(
                {
                    "name": doc["name"],
                    "similarity": doc["similarity"],
                    "metadata": metadata,
                }
            )

        # OpenAIへのプロンプト
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided context to answer the user's question. If the answer is not in the context, say you don't know.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.3, max_tokens=500
        )

        return {"answer": response.choices[0].message.content, "sources": sources}

    except Exception as e:
        return {"status": "error", "message": f"Error in RAG process: {str(e)}"}


# データベースリセット用のツール
@app.tool()
def reset_database_tool() -> dict:
    """
    Reset the database by dropping and recreating the documents table.
    WARNING: This will delete all stored documents and embeddings.

    Returns:
        Status of the operation
    """
    return reset_database()


# ドキュメント全件取得用のツール
@app.tool()
def get_all_documents_tool(limit: int = 1000, offset: int = 0) -> dict:
    """
    Get all documents from the database with pagination.

    Args:
        limit: Maximum number of records to return (default: 1000)
        offset: Number of records to skip (default: 0)

    Returns:
        List of documents with pagination metadata
    """
    return get_all_documents(limit, offset)


# データベースの初期化
init_database()

# サーバー起動コード
if __name__ == "__main__":
    app.run()
