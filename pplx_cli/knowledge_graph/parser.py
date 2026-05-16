import re
import os
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    id: str
    name: str
    path: str
    type: str = "internal"
    weight: int = 1
    tags: List[str] = field(default_factory=list)


@dataclass
class GraphEdge:
    source: str
    target: str
    label: str = ""


class MarkdownKnowledgeGraph:
    WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]+)?\]\]")
    MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir).resolve()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.file_map: Dict[str, str] = {}
        self.parse_errors: List[Tuple[str, str]] = []

    def discover_and_parse(self) -> None:
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            logger.warning("Directory not found: %s", self.root_dir)
            return

        md_files = sorted(self.root_dir.rglob("*.md"))

        for md_path in md_files:
            self._register_file(md_path)

        for md_path in md_files:
            self._parse_file(md_path)

        self._compute_weights()

    def _register_file(self, file_path: Path) -> None:
        relative_path = str(file_path.relative_to(self.root_dir))
        slug = self._slugify(relative_path)
        name = file_path.stem

        self.nodes[slug] = GraphNode(
            id=slug, name=name, path=relative_path, type="internal"
        )
        self.file_map[slug] = relative_path
        if name.lower() not in self.file_map:
            self.file_map[name.lower()] = relative_path
        if relative_path.lower() not in self.file_map:
            self.file_map[relative_path.lower()] = relative_path

    def _parse_file(self, file_path: Path) -> None:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            self.parse_errors.append((str(file_path), str(e)))
            return

        relative_path = str(file_path.relative_to(self.root_dir))
        source_slug = self._slugify(relative_path)

        for match in self.WIKILINK_RE.finditer(content):
            link_target = match.group(1).strip()
            target_slug = self._resolve_target(link_target, file_path)
            self.edges.append(GraphEdge(source=source_slug, target=target_slug, label=""))

        for match in self.MARKDOWN_LINK_RE.finditer(content):
            link_url = match.group(2).strip()

            if link_url.startswith("#"):
                continue

            parsed = urlparse(link_url)
            if parsed.scheme and parsed.scheme not in ("file", ""):
                name = match.group(1).strip() or link_url
                ext_slug = f"ext:{link_url}"
                if ext_slug not in self.nodes:
                    self.nodes[ext_slug] = GraphNode(
                        id=ext_slug, name=name[:60], path=link_url, type="external"
                    )
                self.edges.append(GraphEdge(source=source_slug, target=ext_slug, label=""))
                continue

            if parsed.scheme == "file" or not parsed.scheme:
                link_path = parsed.path or link_url
                resolved = self._resolve_markdown_path(link_path, file_path)
                self.edges.append(GraphEdge(source=source_slug, target=resolved, label=""))

    def _resolve_target(self, link_text: str, source_file: Path) -> str:
        clean = link_text.strip()

        if "." in clean:
            resolved = self._resolve_markdown_path(clean, source_file)
            if resolved != clean:
                return resolved

        lower = clean.lower()
        for candidate in [clean, f"{clean}.md", lower, f"{lower}.md"]:
            if candidate in self.file_map:
                return self._slugify(self.file_map[candidate])

        parts = clean.replace("\\", "/").split("/")
        for i in range(len(parts)):
            suffix = "/".join(parts[i:])
            for candidate in [suffix, f"{suffix}.md", suffix.lower(), f"{suffix.lower()}.md"]:
                if candidate in self.file_map:
                    return self._slugify(self.file_map[candidate])

        slug = self._slugify(clean)
        if slug not in self.nodes:
            self.nodes[slug] = GraphNode(
                id=slug, name=clean, path=clean, type="external"
            )
        return slug

    def _resolve_markdown_path(self, link_path: str, source_file: Path) -> str:
        source_dir = source_file.parent

        try:
            candidate = (source_dir / link_path).resolve()
            if candidate.exists() and candidate.is_relative_to(self.root_dir):
                return self._slugify(str(candidate.relative_to(self.root_dir)))
        except (ValueError, OSError):
            pass

        try:
            candidate = (self.root_dir / link_path).resolve()
            if candidate.exists() and candidate.is_relative_to(self.root_dir):
                return self._slugify(str(candidate.relative_to(self.root_dir)))
        except (ValueError, OSError):
            pass

        clean = link_path.replace("\\", "/")
        lower = clean.lower()
        for candidate in [clean, f"{clean}.md", lower, f"{lower}.md"]:
            if candidate in self.file_map:
                return self._slugify(self.file_map[candidate])

        slug = self._slugify(clean)
        if slug not in self.nodes:
            self.nodes[slug] = GraphNode(
                id=slug, name=os.path.basename(clean).rsplit(".", 1)[0] or clean,
                path=clean, type="external"
            )
        return slug

    def _compute_weights(self) -> None:
        in_degree: Dict[str, int] = {}
        out_degree: Dict[str, int] = {}

        for edge in self.edges:
            out_degree[edge.source] = out_degree.get(edge.source, 0) + 1
            in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

        for slug, node in self.nodes.items():
            node.weight = max(
                in_degree.get(slug, 0) + out_degree.get(slug, 0),
                1
            )

    @staticmethod
    def _slugify(path: str) -> str:
        return path.lower().replace("\\", "/").replace(" ", "-")

    def to_json(self) -> dict:
        node_list = []
        edge_list = []

        for node in self.nodes.values():
            node_list.append({
                "id": node.id,
                "name": node.name,
                "path": node.path,
                "type": node.type,
                "weight": node.weight,
                "tags": node.tags,
            })

        for edge in self.edges:
            edge_list.append({
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
            })

        return {"nodes": node_list, "edges": edge_list}

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)
