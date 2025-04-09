# Milvus MCP使用ガイド: AIによるベクトルデータベース活用法

このドキュメントは、AIシステムがMilvus MCPを効果的に活用するための包括的なガイドです。Milvusはベクトルデータベースとして広く使われており、大規模なベクトル検索や類似性検索に適しています。このガイドを通じて、AIシステムはMilvus MCPの基本から応用までをマスターできます。

## 目次

1. [Milvusとは](https://claude.ai/chat/ecf22d1f-5403-484a-be2f-dd90fa3ba490#1-milvus%E3%81%A8%E3%81%AF)
2. [Milvus MCPの基本](https://claude.ai/chat/ecf22d1f-5403-484a-be2f-dd90fa3ba490#2-milvus-mcp%E3%81%AE%E5%9F%BA%E6%9C%AC)
3. [利用可能なツール一覧](https://claude.ai/chat/ecf22d1f-5403-484a-be2f-dd90fa3ba490#3-%E5%88%A9%E7%94%A8%E5%8F%AF%E8%83%BD%E3%81%AA%E3%83%84%E3%83%BC%E3%83%AB%E4%B8%80%E8%A6%A7)
4. [基本的な使用方法](https://claude.ai/chat/ecf22d1f-5403-484a-be2f-dd90fa3ba490#4-%E5%9F%BA%E6%9C%AC%E7%9A%84%E3%81%AA%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95)
    - [コレクションの作成](https://claude.ai/chat/ecf22d1f-5403-484a-be2f-dd90fa3ba490#%E3%82%B3%E3%83%AC%E3%82%AF%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%AE%E4%BD%9C%E6%88%90)
    - [データの挿入](https://claude.ai/chat/ecf22d1f-5403-484a-be2f-dd90fa3ba490#%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E6%8C%BF%E5%85%A5)
    - [データのクエリと検索](https://claude.ai/chat/ecf22d1f-5403-484a-be2f-dd90fa3ba490#%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E3%82%AF%E3%82%A8%E3%83%AA%E3%81%A8%E6%A4%9C%E7%B4%A2)
    - [コレクションの管理](https://claude.ai/chat/ecf22d1f-5403-484a-be2f-dd90fa3ba490#%E3%82%B3%E3%83%AC%E3%82%AF%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%AE%E7%AE%A1%E7%90%86)
5. [高度な使用方法](https://claude.ai/chat/ecf22d1f-5403-484a-be2f-dd90fa3ba490#5-%E9%AB%98%E5%BA%A6%E3%81%AA%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95)
    - [ベクトル検索の最適化](https://claude.ai/chat/ecf22d1f-5403-484a-be2f-dd90fa3ba490#%E3%83%99%E3%82%AF%E3%83%88%E3%83%AB%E6%A4%9C%E7%B4%A2%E3%81%AE%E6%9C%80%E9%81%A9%E5%8C%96)
    - [複合クエリの作成](https://claude.ai/chat/ecf22d1f-5403-484a-be2f-dd90fa3ba490#%E8%A4%87%E5%90%88%E3%82%AF%E3%82%A8%E3%83%AA%E3%81%AE%E4%BD%9C%E6%88%90)
6. [一般的なエラーと対処法](https://claude.ai/chat/ecf22d1f-5403-484a-be2f-dd90fa3ba490#6-%E4%B8%80%E8%88%AC%E7%9A%84%E3%81%AA%E3%82%A8%E3%83%A9%E3%83%BC%E3%81%A8%E5%AF%BE%E5%87%A6%E6%B3%95)
7. [ベストプラクティス](https://claude.ai/chat/ecf22d1f-5403-484a-be2f-dd90fa3ba490#7-%E3%83%99%E3%82%B9%E3%83%88%E3%83%97%E3%83%A9%E3%82%AF%E3%83%86%E3%82%A3%E3%82%B9)
8. [ユースケース例](https://claude.ai/chat/ecf22d1f-5403-484a-be2f-dd90fa3ba490#8-%E3%83%A6%E3%83%BC%E3%82%B9%E3%82%B1%E3%83%BC%E3%82%B9%E4%BE%8B)

## 1. Milvusとは

Milvusは、大規模なベクトル検索エンジンとして設計されたオープンソースのベクトルデータベースです。機械学習モデルから生成された特徴ベクトルを保存し、高速にベクトル類似性検索を行うことができます。以下のような特徴があります：

- **高性能**: 数十億規模のベクトルに対する効率的な類似性検索
- **柔軟性**: スカラーデータとベクトルデータの両方を保存・検索可能
- **スケーラビリティ**: 分散環境でのスケールアウトをサポート
- **使いやすさ**: 様々な言語のクライアントSDKを提供

Milvusは主に以下のような用途に使われます：

- 画像検索・類似画像検索
- レコメンデーションシステム
- チャットボットや質問応答システム
- 異常検知

## 2. Milvus MCPの基本

MCP (Microservice Component Protocol) は、AIシステムがMilvusの機能にアクセスするためのインターフェースを提供します。MCPを通じて、AIはMilvusの高度なベクトル検索機能を活用できます。

Milvus MCPは以下のコンポーネントで構成されています：

- **MilvusConnector**: Milvusデータベースとの接続を管理するクラス
- **ツール関数群**: コレクション管理、データ操作、検索機能などを提供する関数セット

## 3. 利用可能なツール一覧

Milvus MCPでは以下のツールが利用可能です：

### コレクション管理

- `milvus_list_collections`: データベース内のすべてのコレクションを一覧表示
- `milvus_create_collection`: 新しいコレクションを作成
- `milvus_drop_collection`: コレクションを削除
- `milvus_load_collection`: コレクションをメモリにロード
- `milvus_release_collection`: コレクションをメモリから解放

### データ操作

- `milvus_insert_data`: コレクションにデータを挿入
- `milvus_delete_entities`: フィルター式に基づいてエンティティを削除
- `milvus_upsert_data`: データを挿入または更新（存在する場合）

### 検索・クエリ

- `milvus_query`: フィルター式を使用してコレクションをクエリ
- `milvus_vector_search`: ベクトル類似性検索を実行
- `milvus_text_search`: テキスト検索を実行
- `milvus_hybrid_search`: ベクトル類似性検索と属性フィルタリングを組み合わせた検索

### データベース管理

- `milvus_list_databases`: Milvusインスタンス内のすべてのデータベースを一覧表示
- `milvus_use_database`: 異なるデータベースに切り替え

## 4. 基本的な使用方法

### コレクションの作成

コレクションはデータを保存する論理的なグループです。スキーマを定義して新しいコレクションを作成できます。

```
コレクション作成時の重要なパラメータ:
- collection_name: コレクションの名前
- collection_schema: コレクションのスキーマ定義
  - dimension: ベクトルの次元数
  - other_fields: スカラーフィールドの定義
```

例：

```python
# 4次元ベクトルと追加フィールドを持つコレクションを作成
collection_schema = {
    "dimension": 4, 
    "other_fields": [
        {"name": "title", "dtype": "VarChar", "max_length": 100},
        {"name": "category", "dtype": "VarChar", "max_length": 50},
        {"name": "rating", "dtype": "Int64"}
    ]
}

await milvus_create_collection(
    collection_name="example_collection",
    collection_schema=collection_schema
)
```

### データの挿入

データはコレクションスキーマに従って挿入する必要があります。データはフィールド名とそれに対応する値のリストのマッピングとして提供します。

重要な注意点: データ挿入には各フィールドの値リストが必要で、すべてのリストは同じ長さである必要があります。

例：

```python
# 5つのレコードを挿入
data = {
    "id": [1, 2, 3, 4, 5],
    "title": ["機械学習入門", "データサイエンスの基礎", "ベクトルデータベースガイド", 
              "AIアプリケーション開発", "コンピュータビジョン実践"],
    "rating": [92, 88, 95, 90, 93],
    "vector": [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.8, 0.7, 0.6],
        [0.4, 0.3, 0.2, 0.1],
        [0.5, 0.5, 0.5, 0.5]
    ],
    "category": ["機械学習", "データサイエンス", "データベース", 
                "AI開発", "コンピュータビジョン"]
}

await milvus_insert_data(
    collection_name="example_collection",
    data=data
)
```

### データのクエリと検索

データを検索するには、まずコレクションをメモリにロードし、その後クエリまたは検索リクエストを実行します。

1. コレクションをロード:

```python
await milvus_load_collection(collection_name="example_collection")
```

2. データのクエリ:

```python
# すべてのデータをクエリ
results = await milvus_query(
    collection_name="example_collection",
    filter_expr="id >= 0"
)

# 特定の条件でフィルタリング
results = await milvus_query(
    collection_name="example_collection",
    filter_expr="rating > 90"
)

# 複合条件でフィルタリング
results = await milvus_query(
    collection_name="example_collection",
    filter_expr="rating >= 90 and rating < 95"
)

# 特定のカテゴリでフィルタリング
results = await milvus_query(
    collection_name="example_collection",
    filter_expr="category == 'データベース'"
)
```

3. ベクトル検索（類似性検索）:

```python
# 類似のベクトルを検索
results = await milvus_vector_search(
    collection_name="example_collection",
    vector=[0.9, 0.8, 0.7, 0.6],
    vector_field="vector",
    limit=3
)
```

### コレクションの管理

コレクションのライフサイクルを管理するためのユーティリティ関数:

```python
# コレクションのリスト表示
collections = await milvus_list_collections()

# コレクションのロード（メモリにロード）
await milvus_load_collection(collection_name="example_collection")

# コレクションのリリース（メモリから解放）
await milvus_release_collection(collection_name="example_collection")

# コレクションの削除
await milvus_drop_collection(collection_name="example_collection")
```

## 5. 高度な使用方法

### ベクトル検索の最適化

ベクトル検索のパフォーマンスを最適化するには:

1. 適切なインデックスタイプの選択:
    
    - IVF_FLAT: 速度と精度のバランスが取れている
    - HNSW: 高速だが、メモリ使用量が多い
2. 検索パラメータの調整:
    
    - metric_type: コサイン類似度(COSINE)、ユークリッド距離(L2)、内積(IP)
    - nprobe: 検索する分割数。大きいほど精度が向上するが、パフォーマンスは低下

### 複合クエリの作成

複合クエリは、ベクトル類似性検索とスカラーフィルタリングを組み合わせることができます:

```python
# レーティングが90以上で、特定のベクトルに類似したエンティティを検索
results = await milvus_hybrid_search(
    collection_name="example_collection",
    vector=[0.9, 0.8, 0.7, 0.6],
    vector_field="vector",
    filter_expr="rating >= 90",
    limit=5
)
```

## 6. 一般的なエラーと対処法

### データ挿入エラー

**エラー**: `The Input data type is inconsistent with defined schema, {id} field should be a int64, but got a {<class 'list'>} instead.`

**解決策**: データ形式の修正。Milvus MCPの修正版では、データは変換されますが、データの型がスキーマと一致していることを確認してください。

### クエリエラー

**エラー**: `Collection not loaded`

**解決策**: クエリ前にコレクションをロードする:

```python
await milvus_load_collection(collection_name="your_collection")
```

**エラー**: `Cannot parse expression`

**解決策**: 正しいクエリ構文を使用する:

- 等価比較には `==` を使用（`=` ではなく）
- 文字列リテラルには単一引用符 `'` を使用
- 複合条件には `and`, `or` を使用

例: `category == 'データベース' and rating > 90`

### ベクトル検索エラー

**エラー**: `The length of float data should divide the dim`

**解決策**: ベクトルの次元数がコレクション作成時に指定した次元数と一致していることを確認してください。

## 7. ベストプラクティス

1. **効率的なデータモデル設計**:
    
    - 必要なフィールドだけを含める
    - ベクトルの次元数を適切に選択する
2. **バッチ処理**:
    
    - 大量のデータを挿入する場合は、`bulk_insert`メソッドを使用する
    - 適切なバッチサイズ（例: 1000件）を選択する
3. **リソース管理**:
    
    - 使用しないコレクションはメモリからリリースする
    - 大規模なデータセットでは、検索前にインデックスを作成する
4. **エラーハンドリング**:
    
    - すべての操作で適切なエラーハンドリングを実装する
    - 問題発生時は詳細なエラーメッセージを確認する
5. **パフォーマンス最適化**:
    
    - 頻繁に使用するコレクションはメモリにロードしたままにする
    - 検索パラメータを用途に合わせて調整する

## 8. ユースケース例

### セマンティック検索システム

テキストデータをベクトル化してMilvusに保存し、意味的に類似したテキストを検索できるシステムを構築できます:

1. テキストをベクトル化（埋め込み）
2. ベクトルとメタデータをMilvusに保存
3. ユーザークエリをベクトル化し、類似性検索を実行

### レコメンデーションエンジン

ユーザープロファイルと商品をベクトル化し、類似性に基づいてレコメンデーションを生成:

1. ユーザー行動データから特徴ベクトルを生成
2. 商品の特徴ベクトルを生成
3. ベクトル類似性検索を使用して、ユーザーにマッチする商品を推奨

### 画像検索システム

画像の特徴ベクトルを抽出してMilvusに保存し、視覚的に類似した画像を検索:

1. 画像から特徴ベクトルを抽出（例: CNNモデルの出力）
2. ベクトルとメタデータをMilvusに保存
3. クエリ画像から特徴ベクトルを抽出し、類似性検索を実行

---

このガイドがAIシステムのMilvus MCP活用の一助となることを願っています。Milvusの高度なベクトル検索機能を活用することで、多様な検索、推薦、分析システムを構築できます。