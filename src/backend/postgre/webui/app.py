#!/usr/bin/env python

import os
import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import tempfile

# server.py をインポートできるようにパスを追加
sys.path.append(str(Path(__file__).resolve().parent.parent))
from server import process_document, semantic_search_tool, cluster_documents_tool, literature_qa_tool, compare_papers_tool, check_research_coverage_tool, recommend_papers_tool, identify_unexplored_topics_tool
from literature_navigator_ui import render_literature_navigator

# 環境変数のロード
load_dotenv()

# データベース接続情報
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "vectordb")

# アプリケーションタイトルとスタイル設定
st.set_page_config(
    page_title="PostgreSQL pgvector RAG 管理インターフェース",
    layout="wide",
    initial_sidebar_state="expanded",
)

# カスタムCSS
st.markdown(
    """
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
        color: #4f4f4f;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4a76c9 !important;
        color: white !important;
    }
    .stTabs [aria-selected="false"] {
        background-color: #e0e0e0 !important;
        color: #333333 !important;
    }
    div.row-widget.stRadio > div {
        flex-direction: row;
        align-items: center;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        margin-right: 15px;
    }
    .highlight {
        background-color: #f1f3f6;
        border-radius: 4px;
        padding: 10px;
        margin: 10px 0;
        border-left: 4px solid #4a76c9;
    }
</style>
""",
    unsafe_allow_html=True,
)


# データベース接続関数
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=DB_NAME,
            cursor_factory=RealDictCursor,
        )
        return conn
    except Exception as e:
        st.error(f"データベース接続エラー: {str(e)}")
        return None


# ドキュメント一覧を取得する関数
def get_documents():
    conn = get_db_connection()
    if not conn:
        return []

    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT name, 
                   COUNT(*) as chunk_count, 
                   MIN(metadata::json->>'source') as source,
                   MIN(metadata::json->>'type') as type
            FROM documents 
            GROUP BY name 
            ORDER BY name
        """
        )
        documents = cursor.fetchall()
        return documents
    except Exception as e:
        st.error(f"ドキュメント一覧取得エラー: {str(e)}")
        return []
    finally:
        cursor.close()
        conn.close()


# ドキュメントのチャンク一覧を取得する関数
def get_document_chunks(document_name):
    conn = get_db_connection()
    if not conn:
        return []

    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, content, metadata
            FROM documents
            WHERE name = %s
            ORDER BY (metadata::json->>'chunk_index')::int
        """,
            (document_name,),
        )
        chunks = cursor.fetchall()

        # JSONの文字列をパースしてオブジェクトに変換
        for chunk in chunks:
            if isinstance(chunk["metadata"], str):
                chunk["metadata"] = json.loads(chunk["metadata"])

        return chunks
    except Exception as e:
        st.error(f"チャンク一覧取得エラー: {str(e)}")
        return []
    finally:
        cursor.close()
        conn.close()


# ドキュメントを削除する関数
def delete_document(document_name):
    conn = get_db_connection()
    if not conn:
        return False, "データベース接続エラー"

    cursor = conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM documents WHERE name = %s RETURNING id", (document_name,)
        )
        deleted_ids = cursor.fetchall()
        conn.commit()
        return (
            True,
            f"ドキュメント '{document_name}' を削除しました（{len(deleted_ids)}チャンク）",
        )
    except Exception as e:
        conn.rollback()
        return False, f"削除エラー: {str(e)}"
    finally:
        cursor.close()
        conn.close()


# データベース情報を取得する関数
def get_database_info():
    conn = get_db_connection()
    if not conn:
        return None

    cursor = conn.cursor()
    try:
        # テーブル一覧を取得（システムテーブルを除外）
        cursor.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
              AND table_name NOT LIKE 'pg_%'
            ORDER BY table_name
        """
        )
        tables = [row["table_name"] for row in cursor.fetchall()]

        # データを格納するための変数を初期化
        documents_structure = []
        document_count = 0
        unique_document_count = 0

        # documentsテーブルが存在する場合のみ、その情報を取得
        if "documents" in tables:
            # documentsテーブルの構造を取得
            cursor.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'documents'
                ORDER BY ordinal_position
            """
            )
            documents_structure = cursor.fetchall()

            # ドキュメント数を取得
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            document_count = cursor.fetchone()["count"]

            # ユニークなドキュメント名の数を取得
            cursor.execute("SELECT COUNT(DISTINCT name) as count FROM documents")
            unique_document_count = cursor.fetchone()["count"]

        return {
            "tables": tables,
            "documents_structure": documents_structure,
            "total_chunks": document_count,
            "unique_documents": unique_document_count,
        }
    except Exception as e:
        st.error(f"データベース情報取得エラー: {str(e)}")
        return None
    finally:
        cursor.close()
        conn.close()


# サイドバーの作成
def render_sidebar():
    with st.sidebar:
        st.title("PostgreSQL pgvector RAG")
        st.markdown(
            "PostgreSQL + pgvectorを使用したRAGシステムの管理インターフェースです。"
        )

        # ナビゲーション
        st.sidebar.header("メニュー")
        menu = st.sidebar.radio(
            "メニュー選択",
            [
                "ダッシュボード",
                "ドキュメント管理",
                "文献ナビゲーター",
                "データベース情報",
            ],
        )

        # 情報表示
        st.sidebar.header("情報")
        db_info = get_database_info()
        if db_info:
            st.sidebar.metric("保存ドキュメント数", db_info["unique_documents"])
            st.sidebar.metric("チャンク数", db_info["total_chunks"])

        # フッター
        st.sidebar.markdown("---")
        st.sidebar.info("RAG = Retrieval-Augmented Generation")

    return menu


# タブコンテンツのレンダリング
def render_dashboard():
    st.header("ダッシュボード")

    # データベース接続チェック
    conn = get_db_connection()
    if not conn:
        st.error("PostgreSQLデータベースに接続できませんでした。")
        st.info(
            "接続設定を確認し、PostgreSQLサーバーが実行中であることを確認してください。"
        )
        return
    else:
        conn.close()
        st.success("PostgreSQLデータベースに正常に接続しました。")

    # 基本統計情報
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("システム概要")
        st.markdown(
            """
        このシステムはPostgreSQL + pgvectorを使用したドキュメント管理システムです。
        
        **主な機能:**
        - ドキュメントのアップロード・管理
        - データベース情報の閲覧
        """
        )

    with col2:
        st.subheader("使い方")
        st.markdown(
            """
        1. **ドキュメント管理** - ドキュメントのアップロード・削除
        2. **データベース情報** - テーブル情報の確認・管理
        """
        )

    # 最近のドキュメント
    st.subheader("保存済みドキュメント")
    documents = get_documents()
    if documents:
        # データフレーム作成
        df = pd.DataFrame(documents)
        # カラム名を日本語に変更
        df.columns = ["ドキュメント名", "チャンク数", "ソース", "タイプ"]
        # データフレーム表示
        st.dataframe(df, use_container_width=True)
    else:
        st.info(
            "保存されているドキュメントはありません。ドキュメント管理タブからアップロードしてください。"
        )


def render_document_management():
    st.header("ドキュメント管理")

    # タブの作成
    tab1, tab2, tab3 = st.tabs(
        ["ドキュメント一覧", "ドキュメントのアップロード", "ドキュメントの削除"]
    )

    # タブ1: ドキュメント一覧
    with tab1:
        st.subheader("保存済みドキュメント一覧")
        documents = get_documents()
        if documents:
            # テーブル表示
            doc_df = pd.DataFrame(documents)
            doc_df.columns = ["ドキュメント名", "チャンク数", "ソース", "タイプ"]
            st.dataframe(doc_df, use_container_width=True)

            # ドキュメント詳細表示
            st.subheader("ドキュメント詳細")
            doc_names = [doc["name"] for doc in documents]
            selected_doc = st.selectbox("ドキュメントを選択", doc_names)

            if selected_doc:
                chunks = get_document_chunks(selected_doc)
                if chunks:
                    st.write(f"チャンク数: {len(chunks)}")

                    # メタデータ表示
                    if chunks[0]["metadata"]:
                        st.subheader("メタデータ")
                        st.json(chunks[0]["metadata"])

                    # チャンク内容表示
                    st.subheader("チャンク内容")
                    for i, chunk in enumerate(chunks):
                        with st.expander(f"チャンク {i+1}"):
                            st.write(chunk["content"])
        else:
            st.info("保存されているドキュメントはありません。")

    # タブ2: ドキュメントのアップロード
    with tab2:
        st.subheader("新規ドキュメントのアップロード")

        # ファイルアップローダー
        uploaded_file = st.file_uploader(
            "PDFまたはテキストファイルをアップロード",
            type=["pdf", "txt", "md", "csv", "json"],
        )

        # チャンク設定
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider(
                "チャンクサイズ", min_value=100, max_value=2000, value=1000, step=100
            )
        with col2:
            chunk_overlap = st.slider(
                "チャンクオーバーラップ", min_value=0, max_value=500, value=200, step=50
            )

        # アップロードボタン
        if st.button("アップロード") and uploaded_file is not None:
            with st.spinner("ドキュメントを処理中..."):
                # ファイルを一時ファイルに保存
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
                ) as temp:
                    temp.write(uploaded_file.getbuffer())
                    temp_path = temp.name

                # ドキュメント処理
                try:
                    result = process_document(
                        temp_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        metadata={"source": uploaded_file.name},
                    )

                    # 一時ファイルを削除
                    os.unlink(temp_path)

                    # 結果表示
                    if result.get("status") == "success":
                        st.success(
                            f"ドキュメントのアップロードに成功しました。作成されたチャンク数: {result.get('chunks_created')}"
                        )
                    else:
                        st.error(f"エラー: {result.get('message')}")
                except Exception as e:
                    st.error(f"ドキュメント処理エラー: {str(e)}")
                    # 一時ファイルを削除
                    os.unlink(temp_path)

    # タブ3: ドキュメントの削除
    with tab3:
        st.subheader("ドキュメントの削除")

        # ドキュメント選択
        documents = get_documents()
        if documents:
            doc_names = [doc["name"] for doc in documents]
            doc_to_delete = st.selectbox("削除するドキュメントを選択", doc_names)

            # 確認と削除ボタン
            st.warning(f"'{doc_to_delete}' を削除しますか？この操作は元に戻せません。")
            if st.button("削除する"):
                success, message = delete_document(doc_to_delete)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        else:
            st.info("削除するドキュメントはありません。")


def render_database_info():
    st.header("データベース情報")

    # タブの作成
    tab1, tab2, tab3 = st.tabs(["基本情報", "テーブル内容一覧", "データベース管理"])

    # タブ1: 基本情報
    with tab1:
        # データベース情報取得
        db_info = get_database_info()
        if db_info:
            col1, col2 = st.columns(2)

            with col1:
                # 基本情報
                st.subheader("基本情報")
                st.metric("テーブル数", len(db_info.get("tables", [])))
                st.metric("ドキュメント数", db_info.get("unique_documents", 0))
                st.metric("チャンク数", db_info.get("total_chunks", 0))

            with col2:
                # テーブル一覧
                st.subheader("テーブル一覧")
                for table in db_info.get("tables", []):
                    st.write(f"- {table}")

            # テーブル構造
            st.subheader("documentsテーブルの構造")
            if db_info.get("documents_structure"):
                # データフレーム作成
                df = pd.DataFrame(db_info["documents_structure"])
                df.columns = ["カラム名", "データ型"]
                st.dataframe(df, use_container_width=True)

            # サンプルデータ表示
            st.subheader("サンプルデータ")
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT * FROM documents LIMIT 5")
                    sample_data = cursor.fetchall()
                    if sample_data:
                        # JSONをきれいに表示
                        for row in sample_data:
                            if isinstance(row.get("metadata"), str):
                                row["metadata"] = json.loads(row["metadata"])
                            # 埋め込みベクトルは長すぎるので省略
                            if "embedding" in row:
                                row["embedding"] = "[ ... 埋め込みベクトル ... ]"

                        # データフレームに変換して表示
                        df = pd.DataFrame(sample_data)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("テーブルにデータがありません。")
                except Exception as e:
                    st.error(f"サンプルデータ取得エラー: {str(e)}")
                finally:
                    cursor.close()
                    conn.close()
        else:
            st.error("データベース情報を取得できませんでした。")

    # タブ2: テーブル内容一覧表示
    with tab2:
        st.subheader("テーブル内容一覧")

        # ページネーション設定
        col1, col2 = st.columns([3, 1])
        with col1:
            limit = st.slider(
                "表示件数", min_value=10, max_value=1000, value=100, step=10
            )
        with col2:
            page = st.number_input("ページ", min_value=1, value=1, step=1)

        offset = (page - 1) * limit

        # server.pyのget_all_documents関数をインポート
        try:
            from server import get_all_documents

            # データ取得
            result = get_all_documents(limit=limit, offset=offset)

            if result.get("status") == "success":
                total = result.get("total", 0)
                documents = result.get("documents", [])

                # 総件数とページ情報を表示
                st.write(
                    f"総件数: {total} 件 (表示中: {offset+1}～{min(offset+limit, total)}件)"
                )

                if documents:
                    # メタデータをJSON文字列から辞書に変換
                    for doc in documents:
                        if isinstance(doc.get("metadata"), str):
                            doc["metadata"] = json.loads(doc["metadata"])

                    # データフレーム作成と表示
                    df = pd.DataFrame(documents)
                    st.dataframe(df, use_container_width=True)

                    # 詳細表示
                    if "id" in df.columns:
                        selected_id = st.selectbox(
                            "詳細を表示する行を選択", df["id"].tolist()
                        )
                        if selected_id:
                            selected_doc = next(
                                (doc for doc in documents if doc["id"] == selected_id),
                                None,
                            )
                            if selected_doc:
                                with st.expander(f"ID: {selected_id} の詳細"):
                                    st.subheader("コンテンツ")
                                    st.write(selected_doc.get("content", ""))

                                    st.subheader("メタデータ")
                                    st.json(selected_doc.get("metadata", {}))
                else:
                    st.info("データがありません。")
            else:
                st.error(f"データ取得エラー: {result.get('message', '不明なエラー')}")
        except Exception as e:
            st.error(f"データ取得中にエラーが発生しました: {str(e)}")

    # タブ3: データベース管理
    with tab3:
        st.subheader("データベース管理")

        # データベースリセット
        st.warning(
            "⚠️ 危険な操作: データベースリセットは全てのデータを削除します。この操作は取り消せません。"
        )

        # リセット確認用の文字列入力
        confirm_text = st.text_input("リセットするには「RESET」と入力してください")

        if st.button("データベースをリセット", disabled=(confirm_text != "RESET")):
            if confirm_text == "RESET":
                try:
                    # server.pyのreset_database関数をインポート
                    from server import reset_database

                    # データベースリセット
                    result = reset_database()

                    if result.get("status") == "success":
                        st.success(
                            result.get("message", "データベースをリセットしました。")
                        )
                        # アプリを再起動してデータを更新
                        st.rerun()
                    else:
                        st.error(
                            f"リセットエラー: {result.get('message', '不明なエラー')}"
                        )
                except Exception as e:
                    st.error(f"データベースリセット中にエラーが発生しました: {str(e)}")
            else:
                st.error("確認文字列が一致しません。リセットは実行されませんでした。")


# メイン関数
def main():
    # サイドバーのレンダリング
    menu = render_sidebar()

    # 選択されたメニューに応じたコンテンツのレンダリング
    if menu == "ダッシュボード":
        render_dashboard()
    elif menu == "ドキュメント管理":
        render_document_management()
    elif menu == "文献ナビゲーター":
        render_literature_navigator(get_db_connection, get_documents)
    elif menu == "データベース情報":
        render_database_info()


if __name__ == "__main__":
    main()
