"""Chunking, Gemini embeddings, retrieval and Groq answering."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
EMBED_DIM = 1536

SYSTEM_PROMPT = (
    "You answer strictly from the provided document excerpts. "
    "Do not include citations, chunk tags, or bracketed numbers like [Chunk X] in your response. "
    "Only provide the direct, clear answer to the user's question. "
    "If the excerpts do not contain the answer, say so plainly. "
    "Keep answers concise and concrete."
)


@dataclass
class Chunk:
    idx: int
    content: str


@dataclass
class Citation:
    idx: int
    content: str
    similarity: float


class ConfigError(Exception):
    pass


def chunk_text(text: str, size: int = 1100, overlap: int = 150) -> list[str]:
    clean = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()
    chunks: list[str] = []
    i = 0
    while i < len(clean):
        end = min(i + size, len(clean))
        if end < len(clean):
            window = clean[i:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > size * 0.5:
                end = i + cut + 1
        piece = clean[i:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(clean):
            break
        i = max(end - overlap, i + 1)
    return chunks[:600]


def _gemini():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ConfigError("GOOGLE_API_KEY is missing from your .env file.")
    from google import genai

    return genai.Client(api_key=key)


def embed_texts(texts: list[str], task_type: str) -> np.ndarray:
    """Embed a list of texts with Google Gemini. Returns an (n, dim) float32 array."""
    from google.genai import types

    client = _gemini()
    vectors: list[list[float]] = []
    for text in texts:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type.upper(),
                output_dimensionality=EMBED_DIM,
            ),
        )
        if not result.embeddings:
            raise ConfigError("Gemini returned no embeddings. Check your Google API key and model access.")
        values = result.embeddings[0].values
        if not values:
            raise ConfigError("Gemini returned an empty embedding for a document chunk.")
        vectors.append(values)

    if len(vectors) != len(texts):
        raise ConfigError("Gemini did not return an embedding for every document chunk.")

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_document(chunks: list[str]) -> np.ndarray:
    return embed_texts(chunks, task_type="retrieval_document")


def embed_query(question: str) -> np.ndarray:
    return embed_texts([question], task_type="retrieval_query")[0]


def top_matches(
    query_vector: np.ndarray, matrix: np.ndarray, chunks: list[str], k: int = 6
) -> list[Citation]:
    scores = matrix @ query_vector
    order = np.argsort(-scores)[:k]
    return [Citation(idx=int(i), content=chunks[int(i)], similarity=float(scores[int(i)])) for i in order]


def answer_question(question: str, citations: list[Citation], history: list[dict]) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ConfigError("GROQ_API_KEY is missing from your .env file.")
    from groq import Groq

    context = "\n\n---\n\n".join(f"[Chunk {c.idx + 1}]\n{c.content}" for c in citations)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])
    messages.append(
        {
            "role": "user",
            "content": f"Document excerpts:\n\n{context}\n\nQuestion: {question}",
        }
    )

    client = Groq(api_key=key)
    completion = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.2
    )
    ans = (completion.choices[0].message.content or "").strip() or "No answer was returned."
    ans = re.sub(r"\[Chunk\s*\d+\]", "", ans)
    ans = re.sub(r"\(Chunk\s*\d+\)", "", ans)
    ans = re.sub(r"[ \t]+", " ", ans).strip()
    return ans


def used_citations(answer: str, citations: list[Citation]) -> list[Citation]:
    used = {int(m) - 1 for m in re.findall(r"\[Chunk (\d+)\]", answer)}
    chosen = [c for c in citations if c.idx in used]
    return chosen or citations[:2]
