* Exa MCP Serverを使用可能にする
  * [exa-mcp-server](https://github.com/exa-labs/exa-mcp-server)
* neo4j MCP Serverを使用可能にする
  * neo4jデスクトップをインストールする
  * neo4jデスクトップを起動する
  * neo4jデスクトップの左上の「+」をクリックして新しいDBを作成する
  * [mcp-neo4j-cypher](https://github.com/neo4j-contrib/mcp-neo4j/tree/main/servers/mcp-neo4j-cypher)
    * `uv add mcp-neo4j-cypher`
* Milvus MCP Serverを使用可能にする
  * [Milvus](https://milvus.io/)をインストールする
    * [Milvusのインストール](https://milvus.io/docs/install_standalone-docker.md)
    * `~/milvus`ディレクトリを作成して移動し下記のコマンドを実行する
```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```
このコンテナは、以下の手順で停止および削除することができます。
```bash
bash standalone_embed.sh stop
bash standalone_embed.sh delete
```
Milvusの最新バージョンへのアップグレードは以下の手順で行うことができます。
```bash
bash standalone_embed.sh upgrade
```
  * MilvusのWebUI
    * http://localhost:9091/webui/
  * [mcp-server-milvus](https://github.com/zilliztech/mcp-server-milvus)

Claude Desktopでは使用可能だがGitHub Copilotではエラーとなり使用できないため下記は使用しない。
* Qdrant MCP Serverを使用可能にする
  * [mcp-server-qdrant](https://github.com/qdrant/mcp-server-qdrant)
    * Qdrantの実行
      * [Claude Desktopとmcp-server-qdrantで超お手軽ナレッジベースの構築](https://zenn.dev/inurun/articles/fc0ec63cad574b)
      * `docker run -d --name qdrant_local -p 6333:6333 -p 6334:6334 -v "$HOME/qdrant_storage:/qdrant/storage:z" -e QDRANT_API_KEY="your-api-key" -e COLLECTION_NAME="your-collection" qdrant/qdrant`
      * アクセスポイント
        * REST API: http://localhost:6333/
        * Web UI: http://localhost:6333/dashboard
    * MCP Serverの実行
      * `uv python install 3.12`
      * `uv venv --python=3.12`
      * `uv add mcp-server-qdrant`