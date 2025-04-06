* Exa MCP Serverを使用可能にする
  * [exa-mcp-server](https://github.com/exa-labs/exa-mcp-server)
* neo4j MCP Serverを使用可能にする
  * [mcp-neo4j-cypher](https://github.com/neo4j-contrib/mcp-neo4j/tree/main/servers/mcp-neo4j-cypher)
    * `uv add mcp-neo4j-cypher`
* Qdrant MCP Serverを使用可能にする
  * [mcp-server-qdrant](https://github.com/qdrant/mcp-server-qdrant)
    * Qdrantの実行
      * [Claude Desktopとmcp-server-qdrantで超お手軽ナレッジベースの構築](https://zenn.dev/inurun/articles/fc0ec63cad574b)
      * `docker run -d --name qdrant_local -p 6333:6333 -p 6334:6334 -v "$HOME/qdrant_storage:/qdrant/storage:z" -e QDRANT_API_KEY="your-api-key" -e COLLECTION_NAME="your-collection" qdrant/qdrant`
      * アクセスポイント
        * REST API: http://localhost:6333/
        * Web UI: http://localhost:6333/dashboard
    * MCP Serverの実行
      * `uv add mcp-server-qdrant`