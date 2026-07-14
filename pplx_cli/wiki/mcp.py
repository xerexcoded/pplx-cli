"""Read-only MCP interface for a wiki workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .workspace import WikiWorkspace


def run_mcp_server(root_dir: Path | str) -> None:
    """Run a stdio MCP server; write operations intentionally remain CLI-only."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as error:
        raise RuntimeError("wiki mcp requires the mcp dependency") from error

    workspace = WikiWorkspace(root_dir)
    server = FastMCP("pplx-cli wiki")

    @server.tool()
    def wiki_search(query: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Search authoritative source chunks in the configured local wiki."""
        return [
            {
                "source_id": result.source_id,
                "title": result.title,
                "uri": result.uri,
                "locator": result.locator,
                "content": result.content,
                "score": result.score,
                "citation": result.citation,
            }
            for result in workspace.search(query, limit=limit)
        ]

    @server.tool()
    def wiki_read_page(path: str) -> Dict[str, str]:
        """Read one generated Markdown wiki page relative to the wiki directory."""
        candidate = (workspace.wiki_dir / path).resolve()
        if not candidate.is_relative_to(workspace.wiki_dir.resolve()) or not candidate.is_file():
            raise ValueError("Wiki page does not exist")
        return {"path": str(candidate.relative_to(workspace.wiki_dir)), "content": candidate.read_text(encoding="utf-8")}

    @server.tool()
    def wiki_read_source(source_id: int) -> Dict[str, Any]:
        """Read authoritative source chunks and their page/heading locators."""
        return workspace.get_source(source_id)

    @server.tool()
    def wiki_status() -> Dict[str, Any]:
        """Return source, chunk, page, embedding, and vector-backend diagnostics."""
        return workspace.status()

    server.run()
