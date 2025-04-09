
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.cluster import KMeans
import networkx as nx

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "vectordb")

client = OpenAI(api_key=OPENAI_API_KEY)

class LiteratureNavigator:
    """インテリジェント文献ナビゲーターの実装"""
    
    def __init__(self):
        """初期化"""
        self.cache = {}  # 検索結果やクラスタリング結果のキャッシュ
    
    def get_db_connection(self):
        """データベース接続を取得する"""
        conn = psycopg2.connect(
            host=DB_HOST, 
            port=DB_PORT, 
            user=DB_USER, 
            password=DB_PASSWORD, 
            dbname=DB_NAME,
            cursor_factory=RealDictCursor
        )
        return conn
    
    def get_embedding(self, text: str) -> List[float]:
        """テキストの埋め込みベクトルを取得する"""
        text = text.replace("\n", " ")  # 改行を空白に置換
        response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return response.data[0].embedding
    
    def semantic_search(self, query: str, filters: Dict = None, limit: int = 20) -> List[Dict]:
        """
        自然言語クエリに基づいて意味検索を実行する
        
        Args:
            query: 検索クエリ
            filters: 検索フィルター
            limit: 取得する最大件数
            
        Returns:
            検索結果のリスト
        """
        try:
            query_embedding = self.get_embedding(query)
            
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            cur.execute(
                """
                SELECT id, name, content, metadata, 
                       1 - (embedding <=> %s) as similarity
                FROM documents
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding, query_embedding, limit)
            )
            
            results = cur.fetchall()
            documents = []
            
            for doc in results:
                metadata = doc['metadata']
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                    
                document = {
                    "id": doc['id'],
                    "name": doc['name'],
                    "content": doc['content'],
                    "metadata": metadata,
                    "similarity": doc['similarity']
                }
                documents.append(document)
            
            cur.close()
            conn.close()
            
            return documents
            
        except Exception as e:
            print(f"意味検索中にエラーが発生しました: {str(e)}")
            return []
    def cluster_results(self, documents: List[Dict], method: str = "kmeans", n_clusters: int = 5) -> Dict[str, List[Dict]]:
        """
        検索結果をクラスタリングする
        
        Args:
            documents: 検索結果のドキュメントリスト
            method: クラスタリング手法 (現在はkmeans)
            n_clusters: クラスター数
            
        Returns:
            クラスターをキーとするドキュメントのリスト
        """
        try:
            if not documents:
                return {}
                
            doc_embeddings = []
            for doc in documents:
                if 'embedding' in doc:
                    embedding = doc['embedding']
                else:
                    embedding = self.get_embedding(doc['content'])
                doc_embeddings.append(embedding)
            
            embeddings_array = np.array(doc_embeddings)
            
            if method == "kmeans":
                actual_n_clusters = min(n_clusters, len(documents))
                
                kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42)
                clusters = kmeans.fit_predict(embeddings_array)
                
                clustered_docs = {}
                for i, cluster_id in enumerate(clusters):
                    cluster_id_str = str(cluster_id)
                    if cluster_id_str not in clustered_docs:
                        clustered_docs[cluster_id_str] = []
                    clustered_docs[cluster_id_str].append(documents[i])
                
                labeled_clusters = {}
                for cluster_id, docs in clustered_docs.items():
                    topic = self._extract_cluster_topic(docs)
                    labeled_clusters[topic] = docs
                
                return labeled_clusters
            
            else:
                raise ValueError(f"未サポートのクラスタリング手法: {method}")
                
        except Exception as e:
            print(f"クラスタリング中にエラーが発生しました: {str(e)}")
            return {}
    
    def _extract_cluster_topic(self, documents: List[Dict]) -> str:
        """
        クラスター内のドキュメントからトピックを抽出する
        
        Args:
            documents: クラスター内のドキュメントリスト
            
        Returns:
            トピックラベル
        """
        try:
            contents = [doc['content'][:500] for doc in documents]  # 各ドキュメントの先頭500文字を使用
            combined_content = "\n\n".join(contents)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "あなたは研究論文の分析を行うアシスタントです。"},
                    {"role": "user", "content": f"以下の複数の論文/ドキュメント断片は同じクラスターに属しています。これらに共通するトピックや研究テーマを1つの短いフレーズ（5単語以内）で表現してください。余計な説明は不要です。\n\n{combined_content}"}
                ],
                max_tokens=50,
                temperature=0.3,
            )
            
            topic = response.choices[0].message.content.strip()
            return topic
            
        except Exception as e:
            print(f"トピック抽出中にエラーが発生しました: {str(e)}")
            return f"クラスター {hash(str(documents)) % 1000}"  # エラー時はハッシュ値でラベル付け
    
    def expand_query(self, query: str) -> List[str]:
        """
        検索クエリを拡張する
        
        Args:
            query: 元のクエリ
            
        Returns:
            拡張クエリのリスト
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "あなたは研究論文の検索を支援するアシスタントです。"},
                    {"role": "user", "content": f"以下の研究クエリに関連する検索キーワードやフレーズを5つ生成してください。関連する概念、同義語、上位/下位概念なども考慮してください。JSON形式のリストで返してください。\n\nクエリ: {query}"}
                ],
                max_tokens=200,
                temperature=0.5,
            )
            
            content = response.choices[0].message.content
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            if json_start >= 0 and json_end > 0:
                json_str = content[json_start:json_end]
                expanded_queries = json.loads(json_str)
            else:
                expanded_queries = [q.strip() for q in content.split('\n') if q.strip()]
            
            if query not in expanded_queries:
                expanded_queries.insert(0, query)
                
            return expanded_queries
            
        except Exception as e:
            print(f"クエリ拡張中にエラーが発生しました: {str(e)}")
            return [query]  # エラー時は元のクエリのみを返す
    
    def ask_literature(self, question: str, context_docs: List[str] = None, limit: int = 5) -> Dict:
        """
        文献ベースで質問に回答する
        
        Args:
            question: 質問
            context_docs: コンテキストとして使用するドキュメントのID（指定がない場合は検索）
            limit: 検索する最大ドキュメント数
            
        Returns:
            回答と出典情報
        """
        try:
            if not context_docs:
                docs = self.semantic_search(question, limit=limit)
            else:
                conn = self.get_db_connection()
                cur = conn.cursor()
                
                placeholders = ','.join(['%s'] * len(context_docs))
                cur.execute(
                    f"""
                    SELECT id, name, content, metadata
                    FROM documents
                    WHERE id IN ({placeholders})
                    """,
                    context_docs
                )
                
                results = cur.fetchall()
                docs = []
                
                for doc in results:
                    metadata = doc['metadata']
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                        
                    document = {
                        "id": doc['id'],
                        "name": doc['name'],
                        "content": doc['content'],
                        "metadata": metadata
                    }
                    docs.append(document)
                
                cur.close()
                conn.close()
            
            if not docs:
                return {
                    "answer": "質問に回答するための関連文献が見つかりませんでした。",
                    "sources": []
                }
            
            context = "\n\n".join([
                f"ドキュメント: {doc['name']}\n内容: {doc['content']}"
                for doc in docs
            ])
            
            sources = []
            for doc in docs:
                source = {
                    "id": doc['id'],
                    "name": doc['name'],
                    "metadata": doc['metadata']
                }
                if 'similarity' in doc:
                    source['similarity'] = doc['similarity']
                sources.append(source)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "あなたは研究論文の情報を元に質問に回答するアシスタントです。提供されたコンテキストのみを使用して回答し、コンテキストに情報がない場合は「コンテキストに十分な情報がありません」と答えてください。"},
                    {"role": "user", "content": f"コンテキスト:\n{context}\n\n質問: {question}"}
                ],
                max_tokens=500,
                temperature=0.3,
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            print(f"文献Q&A中にエラーが発生しました: {str(e)}")
            return {
                "answer": f"エラーが発生しました: {str(e)}",
                "sources": []
            }
    
    def explain_difference(self, paper_id1: str, paper_id2: str, aspect: str = None) -> str:
        """
        2つの論文の違いを説明する
        
        Args:
            paper_id1: 1つ目の論文ID
            paper_id2: 2つ目の論文ID
            aspect: 比較する観点（手法、結果など）
            
        Returns:
            違いの説明
        """
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            cur.execute(
                """
                SELECT id, name, content, metadata
                FROM documents
                WHERE id IN (%s, %s)
                """,
                (paper_id1, paper_id2)
            )
            
            results = cur.fetchall()
            if len(results) < 2:
                return "指定された論文の一方または両方が見つかりませんでした。"
            
            papers = {}
            for doc in results:
                papers[str(doc['id'])] = {
                    "name": doc['name'],
                    "content": doc['content'],
                    "metadata": doc['metadata'] if isinstance(doc['metadata'], dict) else json.loads(doc['metadata'])
                }
            
            cur.close()
            conn.close()
            
            paper1 = papers.get(str(paper_id1))
            paper2 = papers.get(str(paper_id2))
            
            if not paper1 or not paper2:
                return "論文の取得中にエラーが発生しました。"
            
            aspect_prompt = ""
            if aspect:
                aspect_prompt = f"特に{aspect}の観点から"
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "あなたは研究論文を比較分析するアシスタントです。"},
                    {"role": "user", "content": f"以下の2つの論文・文献を{aspect_prompt}比較し、主な違いを詳細に説明してください。\n\n論文1: {paper1['name']}\n{paper1['content']}\n\n論文2: {paper2['name']}\n{paper2['content']}"}
                ],
                max_tokens=800,
                temperature=0.3,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"論文比較中にエラーが発生しました: {str(e)}")
            return f"エラーが発生しました: {str(e)}"
    
    def check_coverage(self, research_topic: str, reviewed_papers: List[str] = None) -> Dict:
        """
        研究トピックに対する文献調査の網羅性をチェックする
        
        Args:
            research_topic: 研究トピック
            reviewed_papers: レビュー済みの論文ID（指定がない場合は全文献から判断）
            
        Returns:
            網羅性レポート
        """
        try:
            reviewed_docs = []
            if reviewed_papers:
                conn = self.get_db_connection()
                cur = conn.cursor()
                
                placeholders = ','.join(['%s'] * len(reviewed_papers))
                cur.execute(
                    f"""
                    SELECT id, name, content, metadata
                    FROM documents
                    WHERE id IN ({placeholders})
                    """,
                    reviewed_papers
                )
                
                results = cur.fetchall()
                for doc in results:
                    metadata = doc['metadata']
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                        
                    document = {
                        "id": doc['id'],
                        "name": doc['name'],
                        "content": doc['content'],
                        "metadata": metadata
                    }
                    reviewed_docs.append(document)
                
                cur.close()
                conn.close()
            
            if reviewed_docs:
                reviewed_content = "\n\n".join([
                    f"論文: {doc['name']}\n概要: {doc['content'][:500]}..."  # 各論文の先頭500文字を使用
                    for doc in reviewed_docs
                ])
            else:
                reviewed_content = "レビュー済み論文はありません。"
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "あなたは研究調査の網羅性を分析するアシスタントです。"},
                    {"role": "user", "content": f"以下の研究トピックと、ユーザーがレビュー済みの論文リストを分析し、調査の網羅性を評価してください。\n\n研究トピック: {research_topic}\n\nレビュー済み論文:\n{reviewed_content}\n\n以下の観点からレポートを作成してください：\n1. 調査の強み (カバーされている重要分野)\n2. 調査の弱み (不足している可能性のある分野)\n3. 網羅性スコア (0-100)\n4. 調査すべき追加分野や観点の提案"}
                ],
                max_tokens=1000,
                temperature=0.3,
            )
            
            coverage_report = {
                "research_topic": research_topic,
                "reviewed_papers_count": len(reviewed_docs),
                "analysis": response.choices[0].message.content
            }
            
            return coverage_report
            
        except Exception as e:
            print(f"網羅性チェック中にエラーが発生しました: {str(e)}")
            return {
                "research_topic": research_topic,
                "error": f"エラーが発生しました: {str(e)}"
            }
    
    def recommend_papers(self, research_topic: str, reviewed_papers: List[str] = None, limit: int = 10) -> List[Dict]:
        """
        研究トピックに基づいて重要論文を推薦する
        
        Args:
            research_topic: 研究トピック
            reviewed_papers: レビュー済みの論文ID（除外リスト）
            limit: 推薦する最大論文数
            
        Returns:
            推薦論文リスト
        """
        try:
            expanded_queries = self.expand_query(research_topic)
            
            all_results = []
            for query in expanded_queries[:3]:  # 最初の3つのクエリのみ使用
                results = self.semantic_search(query, limit=limit)
                all_results.extend(results)
            
            unique_docs = {}
            for doc in all_results:
                if doc['id'] not in unique_docs:
                    unique_docs[doc['id']] = doc
            
            if reviewed_papers:
                for paper_id in reviewed_papers:
                    if paper_id in unique_docs:
                        del unique_docs[paper_id]
            
            recommended_papers = sorted(
                list(unique_docs.values()),
                key=lambda x: x.get('similarity', 0),
                reverse=True
            )
            
            return recommended_papers[:limit]
            
        except Exception as e:
            print(f"論文推薦中にエラーが発生しました: {str(e)}")
            return []
    
    def identify_unexplored_topics(self, research_topic: str, reviewed_papers: List[str] = None) -> List[Dict]:
        """
        研究トピックに関連する未調査トピックを特定する
        
        Args:
            research_topic: 研究トピック
            reviewed_papers: レビュー済みの論文ID
            
        Returns:
            未調査トピックのリスト
        """
        try:
            reviewed_docs = []
            if reviewed_papers:
                conn = self.get_db_connection()
                cur = conn.cursor()
                
                placeholders = ','.join(['%s'] * len(reviewed_papers))
                cur.execute(
                    f"""
                    SELECT id, name, content, metadata
                    FROM documents
                    WHERE id IN ({placeholders})
                    """,
                    reviewed_papers
                )
                
                results = cur.fetchall()
                for doc in results:
                    metadata = doc['metadata']
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                        
                    document = {
                        "id": doc['id'],
                        "name": doc['name'],
                        "content": doc['content'][:1000],  # 長すぎるコンテンツを制限
                        "metadata": metadata
                    }
                    reviewed_docs.append(document)
                
                cur.close()
                conn.close()
            
            if reviewed_docs:
                reviewed_content = "\n\n".join([
                    f"論文: {doc['name']}\n概要: {doc['content']}"
                    for doc in reviewed_docs[:5]  # 最初の5件のみ使用
                ])
            else:
                reviewed_content = "レビュー済み論文はありません。"
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "あなたは研究トピックの分析を行うアシスタントです。"},
                    {"role": "user", "content": f"以下の研究トピックと、ユーザーがレビュー済みの論文リストを分析し、まだ調査されていない可能性のある関連トピックを特定してください。\n\n研究トピック: {research_topic}\n\nレビュー済み論文:\n{reviewed_content}\n\n未調査の可能性がある関連トピックを5つ、JSON形式の配列で返してください。各トピックは「name」（トピック名）と「description」（簡単な説明）を含む辞書形式にしてください。"}
                ],
                max_tokens=800,
                temperature=0.5,
            )
            
            content = response.choices[0].message.content
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start >= 0 and json_end > 0:
                json_str = content[json_start:json_end]
                try:
                    unexplored_topics = json.loads(json_str)
                except:
                    unexplored_topics = [{"name": "解析エラー", "description": content}]
            else:
                unexplored_topics = [{"name": "解析エラー", "description": content}]
            
            return unexplored_topics
            
        except Exception as e:
            print(f"未調査トピック特定中にエラーが発生しました: {str(e)}")
            return [{"name": "エラー", "description": str(e)}]
