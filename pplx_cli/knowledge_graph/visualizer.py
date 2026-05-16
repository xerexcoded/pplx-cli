import json
import logging
from pathlib import Path
from typing import Optional

from .parser import MarkdownKnowledgeGraph
from .template import D3_JS_TEMPLATE

logger = logging.getLogger(__name__)


def generate_knowledge_graph_html(
    root_dir: Path,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> Path:
    graph = MarkdownKnowledgeGraph(root_dir)
    graph.discover_and_parse()

    graph_json = json.dumps(graph.to_json(), ensure_ascii=False)
    html_title = title or root_dir.name

    html_content = D3_JS_TEMPLATE.format(
        title=html_title,
        graph_json=graph_json,
        node_count=graph.node_count,
        edge_count=graph.edge_count,
    )

    if output_path:
        out = output_path
    else:
        out = Path.home() / ".local" / "share" / "perplexity" / "knowledge_graph.html"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_content, encoding="utf-8")

    if graph.parse_errors:
        logger.warning("Parse errors in %d files:", len(graph.parse_errors))
        for fpath, err in graph.parse_errors[:5]:
            logger.warning("  %s: %s", fpath, err)

    return out
