"""Chunkers. Each returns spans with exact character offsets into a processed document."""

from rag_arena.chunking.base import Chunk, Chunker, ChunkerSpec, make_chunker

__all__ = ["Chunk", "Chunker", "ChunkerSpec", "make_chunker"]
