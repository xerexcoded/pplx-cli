import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pplx_cli.rag.embeddings import EmbeddingModel, get_embedding_model, reset_embedding_model


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_embedding_model()
    yield
    reset_embedding_model()


@patch("pplx_cli.rag.embeddings.SentenceTransformer")
def test_embedding_model_initialization(mock_st):
    mock_instance = MagicMock()
    mock_instance.get_sentence_embedding_dimension.return_value = 768
    mock_st.return_value = mock_instance

    model = EmbeddingModel(model_name="small", device="cpu", quantize=False)
    assert model.model_name == "BAAI/bge-small-en-v1.5"
    assert model.device == "cpu"
    assert model.quantize is False


@patch("pplx_cli.rag.embeddings.SentenceTransformer")
def test_lazy_loading(mock_st):
    mock_instance = MagicMock()
    mock_instance.get_sentence_embedding_dimension.return_value = 768
    mock_st.return_value = mock_instance

    model = EmbeddingModel(model_name="small", device="cpu", quantize=False)
    assert model._model is None
    _ = model.model
    assert model._model is not None
    mock_st.assert_called_once()


@patch("pplx_cli.rag.embeddings.SentenceTransformer")
def test_encode_single(mock_st):
    mock_instance = MagicMock()
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    mock_instance.encode.return_value = np.array([0.1, 0.2, 0.3] * 128).reshape(384)
    mock_st.return_value = mock_instance

    model = EmbeddingModel(model_name="small", device="cpu", quantize=False)
    model._model = mock_instance
    result = model.encode("test text", use_cache=False)
    assert result.shape == (384,)
    assert result.dtype == np.float32


@patch("pplx_cli.rag.embeddings.SentenceTransformer")
def test_encode_batch(mock_st):
    mock_instance = MagicMock()
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    mock_instance.encode.return_value = np.zeros((3, 384), dtype=np.float32)
    mock_st.return_value = mock_instance

    model = EmbeddingModel(model_name="small", device="cpu", quantize=False)
    model._model = mock_instance
    texts = ["text1", "text2", "text3"]
    result = model.encode(texts, use_cache=False)
    assert result.shape == (3, 384)


@patch("pplx_cli.rag.embeddings.SentenceTransformer")
def test_encode_with_cache(mock_st):
    mock_instance = MagicMock()
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    mock_instance.encode.return_value = np.ones(384, dtype=np.float32)
    mock_st.return_value = mock_instance

    model = EmbeddingModel(model_name="small", device="cpu", quantize=False)
    model._model = mock_instance
    model.encode("test query", use_cache=True)
    model.encode("test query", use_cache=True)
    assert mock_instance.encode.call_count == 1


@patch("pplx_cli.rag.embeddings.SentenceTransformer")
def test_clear_cache(mock_st):
    mock_instance = MagicMock()
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    mock_instance.encode.return_value = np.ones(384, dtype=np.float32)
    mock_st.return_value = mock_instance

    model = EmbeddingModel(model_name="small", device="cpu", quantize=False)
    model._model = mock_instance
    model.encode("cached text", use_cache=True)
    model.clear_cache()
    model.encode("cached text", use_cache=True)
    assert mock_instance.encode.call_count == 2


@patch("pplx_cli.rag.embeddings.SentenceTransformer")
def test_similarity(mock_st):
    mock_instance = MagicMock()
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    mock_st.return_value = mock_instance

    model = EmbeddingModel(model_name="small", device="cpu", quantize=False)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert model.similarity(a, b) == pytest.approx(1.0)


@patch("pplx_cli.rag.embeddings.SentenceTransformer")
def test_get_model_info(mock_st):
    mock_instance = MagicMock()
    mock_instance.get_sentence_embedding_dimension.return_value = 768
    mock_st.return_value = mock_instance

    model = EmbeddingModel(model_name="base", device="cpu", quantize=True)
    model._model = mock_instance
    info = model.get_model_info()
    assert info["model_name"] == "BAAI/bge-base-en-v1.5"
    assert info["device"] == "cpu"
    assert info["embedding_dim"] == 768
    assert info["quantized"] is True


@patch("pplx_cli.rag.embeddings.SentenceTransformer")
def test_get_embedding_model_singleton(mock_st):
    mock_instance = MagicMock()
    mock_instance.get_sentence_embedding_dimension.return_value = 768
    mock_st.return_value = mock_instance

    model1 = get_embedding_model("small")
    model2 = get_embedding_model("base")
    assert model1 is model2


@patch("pplx_cli.rag.embeddings.SentenceTransformer")
def test_reset_embedding_model(mock_st):
    mock_instance = MagicMock()
    mock_instance.get_sentence_embedding_dimension.return_value = 768
    mock_st.return_value = mock_instance

    model1 = get_embedding_model("small")
    reset_embedding_model()
    model2 = get_embedding_model("base")
    assert model1 is not model2


@patch("pplx_cli.rag.embeddings.SentenceTransformer")
def test_get_optimal_device_cuda(mock_st):
    mock_st.return_value = MagicMock()
    with patch("torch.cuda.is_available", return_value=True):
        model = EmbeddingModel(model_name="small", quantize=False)
        assert model.device == "cuda"


@patch("pplx_cli.rag.embeddings.SentenceTransformer")
def test_get_optimal_device_cpu(mock_st):
    mock_st.return_value = MagicMock()
    with patch("torch.cuda.is_available", return_value=False):
        with patch("torch.backends.mps", create=True) as mock_mps:
            mock_mps.is_available.return_value = False
            model = EmbeddingModel(model_name="small", quantize=False)
            assert model.device == "cpu"


@patch("pplx_cli.rag.embeddings.SentenceTransformer")
def test_warm_up(mock_st):
    mock_instance = MagicMock()
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    mock_st.return_value = mock_instance

    model = EmbeddingModel(model_name="small", device="cpu", quantize=False)
    model._model = mock_instance
    model.warm_up()
    mock_instance.encode.assert_called()


@patch("pplx_cli.rag.embeddings.SentenceTransformer")
def test_load_fallback_on_error(mock_st):
    mock_small = MagicMock()
    mock_small.get_sentence_embedding_dimension.return_value = 384
    mock_st.side_effect = [Exception("load failed"), mock_small]

    with patch.object(EmbeddingModel, '_apply_quantization', return_value=None):
        model = EmbeddingModel(model_name="base", device="cpu", quantize=False)
        try:
            _ = model.model
        except Exception:
            pass
        assert model.model_name == "BAAI/bge-small-en-v1.5"


def test_models_dict():
    assert EmbeddingModel.MODELS["small"] == "BAAI/bge-small-en-v1.5"
    assert EmbeddingModel.MODELS["base"] == "BAAI/bge-base-en-v1.5"
    assert EmbeddingModel.MODELS["large"] == "BAAI/bge-large-en-v1.5"
