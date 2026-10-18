"""Rebuild the video embedding index with a local Ollama embedding model.

embedding_index_3m.json ships with OpenAI ada-002 vectors, which can't be compared
with vectors from a local model. This script embeds each row's title + summary with
OLLAMA_EMBEDDING_MODEL and writes ../embedding_index_ollama.json with an "embedding"
column in place of "ada_v2".

Usage (from this folder): python ollama_embed_index.py
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama",  # required by the client library, ignored by Ollama
)
model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

LESSON_DIR = Path(__file__).resolve().parent.parent
SOURCE = LESSON_DIR / "embedding_index_3m.json"
TARGET = LESSON_DIR / "embedding_index_ollama.json"
BATCH_SIZE = 32

rows = json.loads(SOURCE.read_text(encoding="utf-8"))
texts = [f"{row['title']}\n{row['summary']}" for row in rows]

for start in range(0, len(rows), BATCH_SIZE):
    batch = client.embeddings.create(input=texts[start:start + BATCH_SIZE], model=model)
    for row, item in zip(rows[start:start + BATCH_SIZE], batch.data):
        row.pop("ada_v2", None)
        row["embedding"] = item.embedding
    print(f"embedded {min(start + BATCH_SIZE, len(rows))}/{len(rows)}")

TARGET.write_text(json.dumps(rows), encoding="utf-8")
print(f"wrote {TARGET}")
