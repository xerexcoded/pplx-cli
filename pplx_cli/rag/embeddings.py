"""
Advanced embedding models for fast RAG implementation.

This module provides optimized embedding models using BGE (BAAI General Embedding)
with quantization and caching for improved performance.
"""

import os
import hashlib
from typing import List, Optional, Union
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    High-performance embedding model using BGE with optimization features.
    
    Features:
    - BGE model for better quality (63.55 MTEB score)
    - Quantization for faster inference
    - LRU caching for frequent queries
    - Batch processing support
    - Lazy loading for faster startup
    """
    
    # BGE models ranked by performance/size tradeoff
    MODELS = {
        "small": "BAAI/bge-small-en-v1.5",      # 33M params, 62.17 MTEB
        "base": "BAAI/bge-base-en-v1.5",        # 109M params, 63.55 MTEB  
        "large": "BAAI/bge-large-en-v1.5",      # 335M params, 64.23 MTEB
    }
    
    DEFAULT_MODEL = "base"  # Best balance of quality/speed
    CACHE_SIZE = 1000       # LRU cache size for embeddings
    MAX_SEQUENCE_LENGTH = 512  # Optimal for BGE models
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        quantize: bool = True,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Model size ('small', 'base', 'large') or full model path
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            quantize: Enable quantization for faster inference
            cache_dir: Directory for model cache
        """
        self.model_name = self.MODELS.get(model_name, model_name)
        self.device = device or self._get_optimal_device()
        self.quantize = quantize
        self.cache_dir = cache_dir or Path.home() / ".cache" / "pplx-cli" / "models"
        
        self._model = None
        self._embedding_cache = {}
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized EmbeddingModel with {self.model_name} on {self.device}")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model only when needed."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _get_optimal_device(self) -> str:
        """Determine the best device for the current system."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load and optimize the sentence transformer model."""
        try:
            logger.info(f"Loading model {self.model_name}...")
            
            # Load model with optimizations
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=str(self.cache_dir)
            )
            
            # Set max sequence length for optimal performance
            self._model.max_seq_length = self.MAX_SEQUENCE_LENGTH
            
            # Apply quantization if enabled and on CPU
            if self.quantize and self.device == "cpu":
                self._apply_quantization()
                
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to smaller model
            if self.model_name != self.MODELS["small"]:
                logger.info("Falling back to small model...")
                self.model_name = self.MODELS["small"]
                self._load_model()
            else:
                raise
    
    def _apply_quantization(self):
        """Apply quantization to the model for faster inference."""
        try:
            # Apply dynamic quantization to linear layers
            self._model = torch.quantization.quantize_dynamic(
                self._model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            logger.info("Applied quantization for faster inference")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension()
    
    @lru_cache(maxsize=CACHE_SIZE)
    def _cached_encode(self, text_hash: str, text: str) -> np.ndarray:
        """Cache-enabled encoding for frequent queries."""
        return self._encode_single(text)
    
    def _encode_single(self, text: str) -> np.ndarray:
        """Encode a single text without caching."""
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=False
        )
        return embedding.astype(np.float32)
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        use_cache: bool = True,
        show_progress: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode texts into embeddings with optimization.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing multiple texts
            use_cache: Use LRU cache for frequent queries
            show_progress: Show progress bar for large batches
            
        Returns:
            Embedding(s) as numpy array(s)
        """
        if isinstance(texts, str):
            # Single text
            if use_cache:
                text_hash = hashlib.md5(texts.encode()).hexdigest()
                return self._cached_encode(text_hash, texts)
            else:
                return self._encode_single(texts)
        
        # Multiple texts - batch processing
        if len(texts) <= batch_size:
            # Single batch
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress
            )
            return embeddings.astype(np.float32)
        
        # Multiple batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            all_embeddings.append(batch_embeddings.astype(np.float32))
        
        return np.vstack(all_embeddings)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Embeddings are already normalized, so dot product = cosine similarity
        return float(np.dot(embedding1, embedding2))
    
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dim": self.embedding_dim,
            "quantized": self.quantize,
            "cache_size": self.CACHE_SIZE,
            "max_sequence_length": self.MAX_SEQUENCE_LENGTH
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cached_encode.cache_clear()
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def warm_up(self):
        """Warm up the model with a test encoding."""
        try:
            self.encode("This is a test sentence for warming up the model.")
            logger.info("Model warmed up successfully")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")


# Singleton instance for global access
_default_model: Optional[EmbeddingModel] = None


def get_embedding_model(
    model_name: str = EmbeddingModel.DEFAULT_MODEL,
    **kwargs
) -> EmbeddingModel:
    """
    Get the default embedding model instance (singleton pattern).
    
    Args:
        model_name: Model to use if creating new instance
        **kwargs: Additional arguments for EmbeddingModel
        
    Returns:
        EmbeddingModel instance
    """
    global _default_model
    
    if _default_model is None:
        _default_model = EmbeddingModel(model_name, **kwargs)
    
    return _default_model


def reset_embedding_model():
    """Reset the default embedding model (for testing/configuration changes)."""
    global _default_model
    _default_model = None