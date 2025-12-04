"""Configuration settings for the Recall MCP server.

This module provides Pydantic Settings for configuration management with:
- Environment variable support (RECALL_ prefix)
- CLI argument override support
- Type validation and defaults
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RecallSettings(BaseSettings):
    """Configuration settings for the Recall MCP server.

    Settings are loaded from environment variables with the RECALL_ prefix.
    CLI arguments can override these settings when provided.

    Attributes:
        sqlite_path: Path to SQLite database (default: ~/.recall/recall.db)
        chroma_path: Path to ChromaDB storage (default: ~/.recall/chroma_db)
        collection_name: ChromaDB collection name (default: memories)
        ollama_host: Ollama server host URL (default: http://localhost:11434)
        ollama_model: Embedding model name (default: mxbai-embed-large)
        log_level: Logging level (default: INFO)

    Example:
        >>> settings = RecallSettings()
        >>> print(settings.ollama_host)
        http://localhost:11434

        >>> # Override via environment
        >>> # RECALL_OLLAMA_HOST=http://custom:11434
        >>> settings = RecallSettings()
        >>> print(settings.ollama_host)
        http://custom:11434
    """

    model_config = SettingsConfigDict(
        env_prefix="RECALL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Storage paths
    sqlite_path: Optional[Path] = Field(
        default=None,
        description="Path to SQLite database (default: ~/.recall/recall.db)",
    )
    chroma_path: Optional[Path] = Field(
        default=None,
        description="Path to ChromaDB storage (default: ~/.recall/chroma_db)",
    )
    collection_name: str = Field(
        default="memories",
        description="ChromaDB collection name",
    )

    # Ollama configuration
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server host URL",
    )
    ollama_model: str = Field(
        default="mxbai-embed-large",
        description="Embedding model name",
    )
    ollama_timeout: int = Field(
        default=30,
        description="Ollama request timeout in seconds",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # Memory defaults
    default_namespace: str = Field(
        default="global",
        description="Default namespace for memories",
    )
    default_importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default importance score for memories",
    )

    # Token budget
    default_token_budget: int = Field(
        default=4000,
        description="Default token budget for context generation",
    )

    def get_sqlite_path(self) -> Optional[Path]:
        """Get the SQLite path, resolving to default if not set."""
        if self.sqlite_path:
            return self.sqlite_path.expanduser().resolve()
        return None

    def get_chroma_path(self) -> Optional[Path]:
        """Get the ChromaDB path, resolving to default if not set."""
        if self.chroma_path:
            return self.chroma_path.expanduser().resolve()
        return None
