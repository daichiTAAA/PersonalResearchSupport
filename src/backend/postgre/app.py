#!/usr/bin/env python

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import json
from pathlib import Path

# MCPサーバーのツールをインポート
from server import (
    process_document,
    list_documents,
    search_similar_documents,
    ask_rag,
    delete_document,
)

app = FastAPI(title="PgVector RAG Web Interface")

# テンプレートとスタティックファイルの設定
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 必要なディレクトリの作成
os.makedirs("uploads", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    docs = list_documents()
    return templates.TemplateResponse(
        "index.html", {"request": request, "documents": docs}
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    # アップロードされたファイルを保存
    file_path = Path("uploads") / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # ドキュメント処理
    result = process_document(str(file_path))

    # ドキュメント一覧を取得して表示
    docs = list_documents()
    return templates.TemplateResponse(
        "index.html", {"request": request, "documents": docs, "upload_result": result}
    )


@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    result = ask_rag(question)
    docs = list_documents()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "documents": docs,
            "question": question,
            "rag_result": result,
        },
    )


@app.post("/delete/{doc_name}")
async def delete_doc(request: Request, doc_name: str):
    result = delete_document(doc_name)
    return RedirectResponse(url="/", status_code=303)


@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, query: str, limit: int = 5):
    result = search_similar_documents(query, limit)
    docs = list_documents()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "documents": docs,
            "search_query": query,
            "search_results": result,
        },
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
