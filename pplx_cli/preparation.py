"""Explicitly prepare local retrieval models for offline queries."""

from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from .local import local_only_enabled
from .notes import NotesDB
from .rag.embeddings import EmbeddingModel, _load_rag_config


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def prepare_local_models(include_reranker: bool = False) -> list[str]:
    """Download the configured retrieval models as an explicit user action.

    Local-only operations must never invoke this function, since preparation may
    access the model registry. No source text or query is passed to the models.
    """
    if local_only_enabled():
        raise RuntimeError("Model preparation cannot run in local-only mode. Run `perplexity prepare-models` separately.")

    config = _load_rag_config()
    if config.get("embedding_provider", "local") != "local":
        raise RuntimeError(
            "Local model preparation requires the local embedding provider. "
            "Set embedding_provider to 'local' in the RAG configuration and retry."
        )
    selected = config.get("model") or EmbeddingModel.DEFAULT_MODEL
    bge_model = EmbeddingModel.MODELS.get(selected, selected)
    cache_dir = Path.home() / ".cache" / "pplx-cli" / "models"
    prepared = []
    models = [
        (bge_model, SentenceTransformer, {"cache_folder": str(cache_dir)}),
        (NotesDB.DEFAULT_MODEL, SentenceTransformer, {}),
    ]
    if include_reranker:
        models.append((RERANKER_MODEL, CrossEncoder, {}))
    for name, loader, kwargs in models:
        try:
            loader(name, **kwargs)
        except Exception as error:
            raise RuntimeError(
                f"Could not prepare model {name}. Check model availability and network access, "
                "then retry `perplexity prepare-models`."
            ) from error
        prepared.append(name)
    return prepared
