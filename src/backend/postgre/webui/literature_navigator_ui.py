
import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from server import semantic_search_tool, cluster_documents_tool, literature_qa_tool, compare_papers_tool, check_research_coverage_tool, recommend_papers_tool, identify_unexplored_topics_tool

def get_document_chunks(document_name, get_db_connection):
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

def render_literature_navigator(get_db_connection, get_documents):
    st.header("インテリジェント文献ナビゲーター")
    
    tab1, tab2, tab3, tab4 = st.tabs(
        ["意味検索＆クラスタリング", "文献ベースQ&A", "論文比較", "網羅性チェック＆推薦"]
    )
    
    with tab1:
        st.subheader("意味検索＆クラスタリング")
        
        search_query = st.text_input("研究トピックや質問を入力してください", placeholder="例: 深層学習を用いた自然言語処理の最新手法")
        
        col1, col2 = st.columns(2)
        with col1:
            search_limit = st.slider("検索結果数", min_value=5, max_value=50, value=20, step=5)
        with col2:
            cluster_count = st.slider("クラスター数", min_value=2, max_value=10, value=5, step=1)
        
        if st.button("検索") and search_query:
            with st.spinner("検索中..."):
                search_results = semantic_search_tool(search_query, search_limit)
                
                if search_results.get("status") == "success":
                    results = search_results.get("results", [])
                    
                    if results:
                        st.success(f"{len(results)}件の関連文献が見つかりました")
                        
                        st.subheader("検索結果")
                        for i, doc in enumerate(results):
                            with st.expander(f"{i+1}. {doc.get('name', '無題')} (類似度: {doc.get('similarity', 0):.2f})"):
                                st.write(doc.get('content', '')[:500] + "...")
                                
                                if doc.get('metadata'):
                                    with st.expander("メタデータ"):
                                        st.json(doc.get('metadata'))
                        
                        with st.spinner("クラスタリング中..."):
                            cluster_results = cluster_documents_tool(search_query, cluster_count, search_limit)
                            
                            if cluster_results.get("status") == "success":
                                clusters = cluster_results.get("clusters", {})
                                
                                if clusters:
                                    st.subheader("トピック別クラスタリング結果")
                                    
                                    for topic, docs in clusters.items():
                                        with st.expander(f"トピック: {topic} ({len(docs)}件)"):
                                            for i, doc in enumerate(docs):
                                                st.markdown(f"**{i+1}. {doc.get('name', '無題')}**")
                                                st.write(doc.get('content', '')[:300] + "...")
                                                st.markdown("---")
                                else:
                                    st.info("クラスタリングできる十分な文献がありませんでした。")
                            else:
                                st.error(f"クラスタリングエラー: {cluster_results.get('message', '不明なエラー')}")
                    else:
                        st.info("検索条件に一致する文献が見つかりませんでした。")
                else:
                    st.error(f"検索エラー: {search_results.get('message', '不明なエラー')}")
    
    with tab2:
        st.subheader("文献ベースQ&A")
        
        question = st.text_area("質問を入力してください", placeholder="例: 深層学習モデルのファインチューニングの最適な方法は何ですか？")
        
        st.subheader("特定の文献を選択（オプション）")
        st.info("特定の文献に基づいて回答を得たい場合は、以下から選択してください。選択しない場合は、質問に関連する文献が自動的に使用されます。")
        
        documents = get_documents()
        if documents:
            doc_names = [doc["name"] for doc in documents]
            selected_docs = st.multiselect("参照する文献を選択", doc_names)
            
            selected_doc_ids = []
            if selected_docs:
                for doc_name in selected_docs:
                    chunks = get_document_chunks(doc_name, get_db_connection)
                    if chunks:
                        selected_doc_ids.extend([chunk["id"] for chunk in chunks])
        else:
            st.info("保存されている文献がありません。")
            selected_doc_ids = None
        
        if st.button("質問する") and question:
            with st.spinner("回答を生成中..."):
                qa_result = literature_qa_tool(question, selected_doc_ids if selected_doc_ids else [])
                
                if qa_result.get("status") == "success":
                    result = qa_result.get("result", {})
                    
                    if result:
                        st.subheader("回答")
                        st.markdown(f"<div class='highlight'>{result.get('answer', '')}</div>", unsafe_allow_html=True)
                        
                        sources = result.get("sources", [])
                        if sources:
                            st.subheader("出典")
                            for i, source in enumerate(sources):
                                st.markdown(f"**{i+1}. {source.get('name', '無題')}**")
                                if 'similarity' in source:
                                    st.write(f"関連度: {source.get('similarity', 0):.2f}")
                                st.markdown("---")
                    else:
                        st.info("回答を生成できませんでした。")
                else:
                    st.error(f"Q&Aエラー: {qa_result.get('message', '不明なエラー')}")
    
    with tab3:
        st.subheader("論文比較")
        
        documents = get_documents()
        if documents:
            doc_names = [doc["name"] for doc in documents]
            
            col1, col2 = st.columns(2)
            with col1:
                paper1 = st.selectbox("1つ目の論文を選択", doc_names, key="paper1")
            with col2:
                paper2 = st.selectbox("2つ目の論文を選択", doc_names, key="paper2", index=min(1, len(doc_names)-1))
            
            aspect = st.selectbox(
                "比較観点（オプション）",
                [None, "研究手法", "実験結果", "理論的背景", "応用分野", "限界と課題"]
            )
            
            paper1_id = None
            paper2_id = None
            
            if paper1 and paper2:
                chunks1 = get_document_chunks(paper1, get_db_connection)
                chunks2 = get_document_chunks(paper2, get_db_connection)
                
                if chunks1 and chunks2:
                    paper1_id = chunks1[0]["id"]
                    paper2_id = chunks2[0]["id"]
            
            if st.button("論文を比較") and paper1_id and paper2_id:
                with st.spinner("比較分析中..."):
                    compare_result = compare_papers_tool(paper1_id, paper2_id, aspect if aspect else "")
                    
                    if compare_result.get("status") == "success":
                        explanation = compare_result.get("explanation", "")
                        
                        if explanation:
                            st.subheader("比較結果")
                            st.markdown(f"<div class='highlight'>{explanation}</div>", unsafe_allow_html=True)
                        else:
                            st.info("比較結果を生成できませんでした。")
                    else:
                        st.error(f"比較エラー: {compare_result.get('message', '不明なエラー')}")
        else:
            st.info("比較する論文がありません。ドキュメント管理タブから論文をアップロードしてください。")
    
    with tab4:
        st.subheader("網羅性チェック＆推薦")
        
        research_topic = st.text_input("研究トピックを入力", placeholder="例: 量子コンピューティングの機械学習応用")
        
        st.subheader("レビュー済みの論文（オプション）")
        st.info("既にレビュー済みの論文を選択してください。選択しない場合は、全ての論文が未レビューと見なされます。")
        
        documents = get_documents()
        if documents:
            doc_names = [doc["name"] for doc in documents]
            reviewed_docs = st.multiselect("レビュー済みの論文", doc_names)
            
            reviewed_doc_ids = []
            if reviewed_docs:
                for doc_name in reviewed_docs:
                    chunks = get_document_chunks(doc_name, get_db_connection)
                    if chunks:
                        reviewed_doc_ids.extend([chunk["id"] for chunk in chunks])
        else:
            st.info("保存されている論文がありません。")
            reviewed_doc_ids = None
        
        if st.button("分析する") and research_topic:
            with st.spinner("網羅性を分析中..."):
                coverage_result = check_research_coverage_tool(research_topic, reviewed_doc_ids if reviewed_doc_ids else [])
                
                if coverage_result.get("status") == "success":
                    report = coverage_result.get("report", {})
                    
                    if report:
                        st.subheader("網羅性分析")
                        st.markdown(f"<div class='highlight'>{report.get('analysis', '')}</div>", unsafe_allow_html=True)
                    else:
                        st.info("網羅性分析を生成できませんでした。")
                else:
                    st.error(f"網羅性分析エラー: {coverage_result.get('message', '不明なエラー')}")
            
            with st.spinner("論文を推薦中..."):
                recommend_result = recommend_papers_tool(research_topic, reviewed_doc_ids if reviewed_doc_ids else [])
                
                if recommend_result.get("status") == "success":
                    recommendations = recommend_result.get("recommendations", [])
                    
                    if recommendations:
                        st.subheader("推薦論文")
                        for i, doc in enumerate(recommendations):
                            with st.expander(f"{i+1}. {doc.get('name', '無題')} (関連度: {doc.get('similarity', 0):.2f})"):
                                st.write(doc.get('content', '')[:500] + "...")
                                
                                if doc.get('metadata'):
                                    with st.expander("メタデータ"):
                                        st.json(doc.get('metadata'))
                    else:
                        st.info("推薦できる論文がありませんでした。")
                else:
                    st.error(f"論文推薦エラー: {recommend_result.get('message', '不明なエラー')}")
            
            with st.spinner("未調査トピックを特定中..."):
                topics_result = identify_unexplored_topics_tool(research_topic, reviewed_doc_ids if reviewed_doc_ids else [])
                
                if topics_result.get("status") == "success":
                    topics = topics_result.get("topics", [])
                    
                    if topics:
                        st.subheader("未調査トピック")
                        st.markdown(f"<div class='highlight'>{topics}</div>", unsafe_allow_html=True)
                    else:
                        st.info("未調査トピックを特定できませんでした。")
                else:
                    st.error(f"未調査トピック特定エラー: {topics_result.get('message', '不明なエラー')}")
