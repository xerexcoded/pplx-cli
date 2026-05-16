import pytest
import tempfile
from pathlib import Path
from pplx_cli.knowledge_graph.parser import MarkdownKnowledgeGraph, GraphNode, GraphEdge


@pytest.fixture
def sample_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    (vault / "index.md").write_text("""# Home
See [[notes/ideas]] for thoughts.
[External link](https://example.com)
""")

    notes_dir = vault / "notes"
    notes_dir.mkdir()
    (notes_dir / "ideas.md").write_text("""# Ideas
Refer back to [[index]].
Related: [[projects/work]]
""")

    projects_dir = vault / "projects"
    projects_dir.mkdir()
    (projects_dir / "work.md").write_text("""# Work
Connects to [[../notes/ideas|Ideas]] and [[index]].
""")

    (vault / "orphan.md").write_text("# Orphan\nNo links here.\n")

    return vault


@pytest.fixture
def empty_vault(tmp_path):
    vault = tmp_path / "empty"
    vault.mkdir()
    return vault


def test_graph_node_creation():
    node = GraphNode(id="test", name="Test", path="test.md", type="internal", weight=3, tags=["tag1"])
    assert node.id == "test"
    assert node.name == "Test"
    assert node.type == "internal"
    assert node.weight == 3
    assert "tag1" in node.tags


def test_graph_edge_creation():
    edge = GraphEdge(source="a", target="b", label="related")
    assert edge.source == "a"
    assert edge.target == "b"
    assert edge.label == "related"


def test_parse_empty_vault(empty_vault):
    graph = MarkdownKnowledgeGraph(empty_vault)
    graph.discover_and_parse()
    assert graph.node_count == 0
    assert graph.edge_count == 0


def test_parse_vault_with_files(sample_vault):
    graph = MarkdownKnowledgeGraph(sample_vault)
    graph.discover_and_parse()
    assert graph.node_count >= 4


def test_wikilinks_parsed(sample_vault):
    graph = MarkdownKnowledgeGraph(sample_vault)
    graph.discover_and_parse()
    assert graph.edge_count > 0

    edge_targets = {e.target for e in graph.edges}
    has_notes_ideas = any("notes/ideas" in t for t in edge_targets)
    has_index = any("index" in t for t in edge_targets)
    assert has_notes_ideas or has_index


def test_markdown_link_parsing(sample_vault):
    graph = MarkdownKnowledgeGraph(sample_vault)
    graph.discover_and_parse()

    external_nodes = [n for n in graph.nodes.values() if n.type == "external"]
    external_urls = {n.path for n in external_nodes}
    assert "https://example.com" in external_urls


def test_node_types(sample_vault):
    graph = MarkdownKnowledgeGraph(sample_vault)
    graph.discover_and_parse()

    internal_count = sum(1 for n in graph.nodes.values() if n.type == "internal")
    external_count = sum(1 for n in graph.nodes.values() if n.type == "external")
    assert internal_count >= 4
    assert external_count >= 1


def test_compute_weights(sample_vault):
    graph = MarkdownKnowledgeGraph(sample_vault)
    graph.discover_and_parse()

    all_weights = [n.weight for n in graph.nodes.values()]
    assert all(w >= 1 for w in all_weights)


def test_to_json(sample_vault):
    graph = MarkdownKnowledgeGraph(sample_vault)
    graph.discover_and_parse()
    data = graph.to_json()

    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == graph.node_count
    assert len(data["edges"]) == graph.edge_count


def test_orphan_file(sample_vault):
    graph = MarkdownKnowledgeGraph(sample_vault)
    graph.discover_and_parse()

    orphan_node = next((n for n in graph.nodes.values() if n.name == "orphan"), None)
    assert orphan_node is not None
    assert orphan_node.weight == 1


def test_wikilink_with_alias(sample_vault):
    vault = sample_vault
    (vault / "aliases.md").write_text("# Aliases\nSee [[notes/ideas|Ideas Page]] for info.\n")
    graph = MarkdownKnowledgeGraph(vault)
    graph.discover_and_parse()

    targets = {e.target for e in graph.edges if e.source.startswith("aliases")}
    assert any("notes/ideas" in t for t in targets)


def test_wikilink_with_heading(sample_vault):
    vault = sample_vault
    (vault / "headings.md").write_text("# Headings\nSee [[notes/ideas#section]] and [[index#Introduction]].\n")
    graph = MarkdownKnowledgeGraph(vault)
    graph.discover_and_parse()

    targets = {e.target for e in graph.edges if e.source.startswith("headings")}
    assert any("notes/ideas" in t for t in targets)
    assert any("index" in t for t in targets)


def test_slugify():
    graph = MarkdownKnowledgeGraph(Path("/tmp"))
    assert graph._slugify("Notes/Ideas.md") == "notes/ideas.md"
    assert graph._slugify("My File Name.md") == "my-file-name.md"
    assert graph._slugify("path\\to\\file.md") == "path/to/file.md"


def test_relative_path_link(sample_vault):
    vault = sample_vault
    (vault / "projects" / "deep.md").write_text("# Deep\nLink to [Ideas](../notes/ideas.md)\n")
    graph = MarkdownKnowledgeGraph(vault)
    graph.discover_and_parse()

    targets = {e.target for e in graph.edges if "deep" in e.source}
    assert any("ideas" in t for t in targets)


def test_markdown_link_bare(sample_vault):
    vault = sample_vault
    (vault / "barelink.md").write_text("# Bare\nSee [](notes/ideas.md) for more.\n")
    graph = MarkdownKnowledgeGraph(vault)
    graph.discover_and_parse()

    targets = {e.target for e in graph.edges if "barelink" in e.source}
    assert any("ideas" in t for t in targets)


def test_non_existent_vault(tmp_path):
    graph = MarkdownKnowledgeGraph(tmp_path / "nonexistent")
    graph.discover_and_parse()
    assert graph.node_count == 0
    assert graph.edge_count == 0