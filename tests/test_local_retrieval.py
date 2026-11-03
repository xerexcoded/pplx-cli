from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pplx_cli.local import local_only_mode
from pplx_cli.notes import NotesDB
from pplx_cli.rag.embeddings import EmbeddingModel, PerplexityEmbeddingModel, get_embedding_model, reset_embedding_model


def test_notes_missing_model_does_not_fall_back(tmp_path):
    db = NotesDB(tmp_path)
    with patch('pplx_cli.notes.SentenceTransformer', side_effect=OSError('missing')) as load:
        with local_only_mode(), pytest.raises(RuntimeError, match='local cache'):
            db.search_similar_notes('question')
        load.assert_called_once_with(db.DEFAULT_MODEL, local_files_only=True)
        with local_only_mode(), pytest.raises(RuntimeError, match='local cache'):
            db.add_note('title', 'content')
    assert db.list_notes() == []


def test_notes_reloads_nonlocal_model(tmp_path):
    db = NotesDB(tmp_path)
    with patch('pplx_cli.notes.SentenceTransformer') as load:
        old = db.model
        with local_only_mode():
            assert db.model is not old or load.call_count == 2
        assert load.call_args.kwargs == {'local_files_only': True}


def test_bge_missing_model_does_not_change_model_or_fallback(tmp_path):
    model = EmbeddingModel('base', cache_dir=tmp_path, quantize=False)
    with patch('pplx_cli.rag.embeddings.SentenceTransformer', side_effect=OSError('missing')) as load:
        with local_only_mode(), pytest.raises(RuntimeError, match='local cache'):
            model.encode('question')
    assert load.call_count == 1
    assert load.call_args.kwargs['local_files_only'] is True
    assert model.model_name == model.MODELS['base']


def test_bge_rechecks_model_and_discards_old_cached_vector(tmp_path):
    model = EmbeddingModel('base', cache_dir=tmp_path, quantize=False)
    first, second = MagicMock(), MagicMock()
    first.encode.return_value = np.array([1.0])
    second.encode.return_value = np.array([2.0])
    with patch('pplx_cli.rag.embeddings.SentenceTransformer', side_effect=[first, second]) as load:
        assert model.encode('question')[0] == 1
        with local_only_mode():
            assert model.encode('question')[0] == 2
        assert load.call_args.kwargs['local_files_only'] is True


def test_remote_singleton_and_existing_instance_rejected():
    reset_embedding_model()
    try:
        remote = get_embedding_model(provider='perplexity')
        with local_only_mode():
            with pytest.raises(RuntimeError, match='local-only'):
                get_embedding_model()
            with patch('pplx_cli.rag.embeddings.requests.post') as post:
                with pytest.raises(RuntimeError, match='local-only'):
                    remote.encode('private')
                post.assert_not_called()
    finally:
        reset_embedding_model()
