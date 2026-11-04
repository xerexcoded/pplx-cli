from unittest.mock import patch

import pytest

from pplx_cli.local import local_only_mode
from pplx_cli.preparation import RERANKER_MODEL, prepare_local_models
from pplx_cli.rag.embeddings import EmbeddingModel
from pplx_cli.notes import NotesDB


def test_prepares_selected_models_with_correct_caches(tmp_path, monkeypatch):
    monkeypatch.setattr('pathlib.Path.home', lambda: tmp_path)
    with patch('pplx_cli.preparation._load_rag_config', return_value={'model': 'large'}), \
         patch('pplx_cli.preparation.SentenceTransformer') as load, \
         patch('pplx_cli.preparation.CrossEncoder') as cross:
        assert prepare_local_models() == [EmbeddingModel.MODELS['large'], NotesDB.DEFAULT_MODEL]
        assert load.call_args_list[0].args == (EmbeddingModel.MODELS['large'],)
        assert load.call_args_list[0].kwargs == {'cache_folder': str(tmp_path / '.cache/pplx-cli/models')}
        assert load.call_args_list[1].args == (NotesDB.DEFAULT_MODEL,)
        assert load.call_args_list[1].kwargs == {}
        cross.assert_not_called()


def test_optional_reranker():
    with patch('pplx_cli.preparation._load_rag_config', return_value={}), \
         patch('pplx_cli.preparation.SentenceTransformer'), \
         patch('pplx_cli.preparation.CrossEncoder') as cross:
        assert prepare_local_models(True)[-1] == RERANKER_MODEL
        cross.assert_called_once_with(RERANKER_MODEL)


def test_remote_provider_rejected_before_loading():
    with patch('pplx_cli.preparation._load_rag_config', return_value={'embedding_provider': 'perplexity'}), \
         patch('pplx_cli.preparation.SentenceTransformer') as load:
        with pytest.raises(RuntimeError, match='embedding_provider'):
            prepare_local_models()
        load.assert_not_called()


def test_local_scope_rejected_before_loading():
    with patch('pplx_cli.preparation.SentenceTransformer') as load:
        with local_only_mode(), pytest.raises(RuntimeError, match='local-only'):
            prepare_local_models()
        load.assert_not_called()


def test_failure_does_not_fall_back():
    with patch('pplx_cli.preparation._load_rag_config', return_value={'model': 'base'}), \
         patch('pplx_cli.preparation.SentenceTransformer', side_effect=OSError('missing')) as load:
        with pytest.raises(RuntimeError, match='Could not prepare model'):
            prepare_local_models()
        assert load.call_count == 1
        assert load.call_args.args == (EmbeddingModel.MODELS['base'],)
